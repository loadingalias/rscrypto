# TLS connection benchmarks

This unpublished tool compares authenticated TLS 1.3 connections through rscrypto-rustls,
rustls/AWS-LC, rustls/*ring*, and OpenSSL 4.0.3.
It calls each library's production implementation.
The workspace is separate so its C comparison libraries do not enter the provider's shipped graph.

`just bench tls-provider --list` lists the catalog workloads without building their fixtures.
A case builds and checks its fixture only when it runs,
so a filtered campaign needs no unselected peer.
Criterion's `--test` mode (`cargo bench --manifest-path tools/provider-bench/Cargo.toml --features ml-dsa --bench tls_connections -- --test`) checks every fixture once.
Measure one catalog algorithm per invocation.
`tls-provider` is an umbrella for discovery, not a promise that every family together fits the ten-minute limit.
Each selector keeps all implementations of a comparison together.
For example:

```bash
just bench tls-handshake-p256
just bench tls-stateful
just bench tls-traffic-bidirectional
just bench-provider-scorecard rscrypto mldsa65 X25519MLKEM768 AES256-GCM-SHA384 full --iterations 256 --workers 4
```

This connection workload has no internal-kernel diagnostic mode; `--diag` is rejected before compilation.

The linked OpenSSL version must be exactly 4.0.3.
Use its development headers and libraries; `OPENSSL_DIR` can select their installation.
The benchmark catalog records this as the target's external prerequisite.
CI does not provision it, so `just bench all` omits this target and the Bench workflow rejects it.
The optional `vendored-openssl` feature lets cross-target lint jobs build the bindings without a target installation.
The bindings currently vendor OpenSSL 3.6.3,
so that feature alone does not supply the required measurement baseline.
The runtime version check rejects it.
With that feature enabled, `OPENSSL_NO_VENDOR=1` selects the explicit 4.0.3 installation.
C builds need a compiler, Perl and Make.

The original handshake matrix covers seven credentials, four groups, three suites,
full and statelessly resumed connections, and five backends.
Its case IDs are unchanged.
*ring* has no ML-KEM or ML-DSA, so profiles that need either have no `ring` row.
The expanded families vary one workload dimension:

| Catalog selector | Work and comparison |
| --- | --- |
| `tls-handshake-p256`, `tls-handshake-p384`, `tls-handshake-ed25519`, `tls-handshake-rsa3072`, `tls-handshake-mldsa44`, `tls-handshake-mldsa65`, `tls-handshake-mldsa87` | Original mutual-authentication matrix, split by credential. |
| `tls-server-auth` | Each original credential, X25519/AES-128-GCM, full handshake with no client certificate or CertificateRequest. Compare with the matching original full row. |
| `tls-intermediate` | Each original credential, X25519/AES-128-GCM, full mutual authentication; both peers send leaf plus one intermediate and omit the root. Compare with the matching direct-root row. The chain fixtures use fresh keys. |
| `tls-stateful` | P-256 credentials, every original KX group and suite, stateful resumption. Compare with the matching original stateless row. |
| `tls-rsa` | RSA-3072 with PSS SHA-384/SHA-512; RSA-4096 with PSS SHA-256/SHA-384/SHA-512. X25519 and every suite, full mutual authentication. Original RSA-3072 uses PSS SHA-256. The digest labels select CertificateVerify, not the certificate issuer's signature or record cipher hash. |
| `tls-hrr` | P-256 credentials, AES-128-GCM, full mutual authentication. Client sends X25519 initially; the server requires P-256 and forces one HelloRetryRequest. Compare with the original P-256 KX row. |
| `tls-compact-key` | ML-DSA-65, X25519/AES-128-GCM, full mutual authentication through the two rscrypto verifier configurations. Uses the production `sign::Provider::compact()` loader. Compare with the original prepared-signing-key row; this is a key-storage tradeoff, with no external compact-mode substitute. |
| `tls-traffic` | Client-to-server writes of 64 B, 256 B, 1 KiB, 16 KiB, 64 KiB and 1 MiB, under each suite. |
| `tls-traffic-bidirectional` | The same sizes per direction; send both directions before reading each record pair. Throughput counts both plaintext directions once. |

Peers use the same credentials, roots, validation time, ALPN, selected group,
suite and CertificateVerify algorithm.
OpenSSL's automatic chain building is off.
Early data is disabled.
Full workloads issue no tickets.
Resumed workloads issue exactly one ticket per connection;
every sample consumes a real ticket and processes its replacement.
Stateful TLS 1.3 resumption sends an opaque cache reference in NewSessionTicket,
rather than using TLS 1.2's session-ID exchange.
Rustls uses its 256-entry server session store with no ticketer.
OpenSSL uses its 256-entry server cache and `SSL_OP_NO_TICKET`, which selects stateful tickets in TLS 1.3.
See [OpenSSL's option contract](https://docs.openssl.org/4.0/man3/SSL_CTX_set_options/).

Every connection, timed or not, must complete the selected handshake kind in three flights,
or four when a ticket follows.
The retry case must take exactly five flights and rustls must report `FullWithHelloRetryRequest`.
Before timing, a check connection verifies accepted ticket count, equal exporters,
the selected suite, group and ALPN, and each expected peer identity.
The server-only case requires absent client identity;
the chain case checks exact leaf/intermediate bytes on both endpoints.
A resumed OpenSSL session restores the peer leaf but may omit the transmitted chain.

Application rows verify two consecutive authenticated transfers before timing,
using distinct bytes in each direction to detect swapped or stale outputs.
Their untimed handshake uses P-256/X25519 on every backend.
Multi-record writes are driven in 16 KiB chunks through both real TLS endpoints;
the harness does not encrypt or decrypt records.
This bounds transport and plaintext storage and keeps the same record boundaries across stacks.
Bidirectional rows model in-process duplex traffic, without threads or simulated network overlap.

Stateless ticket protection uses each stack's production implementation: this provider's AES-256-GCM ticketer,
the rustls/AWS-LC and rustls/*ring* ticketers, and OpenSSL's default stateless tickets.
Ticket formats and protection work therefore differ.
Resumption comparisons measure the complete configured stacks;
they do not isolate the negotiated record cipher's cost.

Configuration and private-key import occur outside connection timing.
The backends verify certificates differently:

- `rscrypto` is the provider's recommended configuration.
  Its `PreparedServerVerifier` and `PreparedClientVerifier` prepare the roots' P-256, P-384 and ML-DSA keys with the configuration.
  A direct-root leaf is verified with a prepared anchor.
  With an intermediate, only the signature linking
  that intermediate to the root uses the prepared anchor;
  the leaf signature also requires the intermediate key.
  Both CertificateVerify verification keys are imported per connection.
- `rscrypto-webpki` uses the same provider with rustls's WebPKI verifiers, as `aws-lc-rs` and `ring` do.
  Compare it with those backends to isolate the provider's cryptography from anchor preparation.
- `openssl` uses OpenSSL's X.509 verification.

The two rscrypto backends do the same work for Ed25519 and RSA anchors, and for resumed connections,
which exchange no certificates.
The provider prepares ML-DSA signing keys once, when it loads them,
except in the explicit compact-key case, which prepares temporary signing storage per signature.
Each rustls backend advertises only the credential's signature scheme.
OpenSSL's RSA list also includes `rsa_pkcs1_sha256`, which TLS 1.3 allows only for certificate signatures;
CertificateVerify uses the row's selected RSA-PSS digest on every stack.

Connection allocation, handshake cryptography, certificate validation, byte transport copies,
ticket processing, close-notify creation and destruction are timed.
A bounded in-process byte queue connects the peers.
It adds no network RTT or loss.
Each direction preallocates 32 KiB and copies with `memcpy` for every stack.
Close notifications are created on destruction; their delivery is not part of this workload.
Every stack decrypts into a retained plaintext buffer and then copies to the caller,
so each makes two plaintext copies.
The rustls clients keep tickets in rustls's default 256-entry store.
The OpenSSL client receives them through its new-session callback,
without its internal session cache.
Record rows include both endpoints and transport copies on a reused connection.
Size labels are plaintext bytes per direction, not wire bytes.

At start, the benchmark prints one JSON line, `provider_bench_provenance`, to standard error.
It records:

- OpenSSL's version, configure flags and platform;
- on Linux, the paths and SHA-256 of the loaded libssl and libcrypto images,
  or the executable's when OpenSSL is linked statically; elsewhere, those of the
  libraries under the build's `OPENSSL_DIR`, which identify the configured
  installation rather than the image the loader chose;
- `RUSTFLAGS` and `CARGO_ENCODED_RUSTFLAGS` at build and run time;
- every `OPENSSL_*cap` capability override, which OpenSSL and AWS-LC both read.

The scorecard JSON includes the same object.
Discovery (`--list`) and case matching omit the line.
Retain it with every result.

Build choices affect the backends unequally.
The bench profile keeps `overflow-checks = true`, as the repository's profiles do.
That adds checks to Rust code only,
which includes rscrypto and rustls but not AWS-LC's C and assembly or OpenSSL.
On local macOS, `just bench` also builds Rust with `-C target-cpu=native`; a direct `cargo bench` does not.
Homebrew's OpenSSL 4.0.3 on Apple silicon includes assembly for AES-GCM, ChaCha20, P-256 and Keccak.
It has no specialized P-384 code, which needs `enable-ec_nistp_64_gcc_128`,
so OpenSSL's P-384 rows use its generic elliptic-curve arithmetic.

Criterion owns elapsed-time estimates and uncertainty.
The supplementary scorecard retains individual samples, empirical percentiles, process CPU,
wire flights and whole-process high-water RSS.
RSS includes the libraries, fixture, key owners and worker stacks.
It is not an allocation count, heap bound, or per-connection memory bound.
Up to 16 workers and 10,000 connections per worker are accepted;
each resource run has a ten-minute watchdog.

Run comparisons on the same quiet native machine with the same compiler, features,
build profile and workload.
Record contention and unsupported environments.
These measurements do not establish constant-time behavior, network latency,
denial-of-service limits, or performance leadership by themselves.
The normal benchmark artifact workflow retains source, commands, build identity and raw results.

## Competitor coverage and claims

Use the fastest measured external backend for each exact workload,
not a single preferred competitor averaged over the suite.
The catalog declares TLS handshake and traffic comparisons for `just bench-compare`;
`rscrypto-webpki` remains a provider variant and cannot count as an external competitor.
Inspect individual medians and intervals as well as aggregate scores,
and retain losses and missing rows.
Compact-key rows compare configurations within rscrypto and have no external comparison score.

The present peers are strong baselines, but this suite does not establish a worldwide ranking.
AWS-LC under rustls is not the s2n-tls stack, and ring is not the BoringSSL stack.
These additional comparisons remain necessary before a broader leadership claim:

- [BoringSSL](https://github.com/google/boringssl): use its production TLS stack with pinned source and matched verification,
  cipher and ticket work.
  The [Cloudflare Rust bindings](https://github.com/cloudflare/boring) inspected on 2026-10-18 lack TLS 1.3 cipher-list
  and ticket-count setters.
  Adding a row with unmatched defaults would not close this gap.
- [Graviola](https://github.com/ctz/graviola): its production `rustls-graviola` adapter targets rustls 0.23,
  while this harness pins 0.24-dev.
  A separate versioned comparison must expose that protocol-library difference;
  a locally rewritten crypto adapter would not measure the shipped provider.
- [s2n-tls](https://github.com/aws/s2n-tls): its complete TLS stack needs an
  independently checked adapter and policy equivalence; rustls/AWS-LC results
  cannot stand in for it.

Run prospective, repeated campaigns on quiet native x86-64 and AArch64 hosts
before reporting performance leadership.
A fixture check, one cloud-host run, a quick estimate,
or a budget-validation run supplies no such conclusion.
The current families measure complete two-endpoint operations.
They localize cost by controlled workload changes;
attributing a loss to a production function requires a focused profile of that exact case.
