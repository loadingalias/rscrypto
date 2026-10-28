# TLS and QUIC connection benchmarks

This unpublished tool compares authenticated TLS 1.3 connections through rscrypto-rustls,
rustls/AWS-LC, rustls/*ring*, and OpenSSL 4.0.3.
It calls each library's production implementation.
The workspace is separate so its C comparison libraries do not enter the provider's shipped graph.

The [QUIC and Quinn rows](#quic-and-quinn) use the same pinned rustls providers and credential policy.

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
just bench-provider-resources mldsa87 X25519MLKEM768 AES256-GCM-SHA384 full --workers 64
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
| `tls-record-gather` | Encrypt one 16 KiB TLS 1.3 record from 1, 4 or 64 plaintext chunks through the production record adapter. Includes framing, content type, encryption and tag append; keys and reserved output capacity stay outside timing. Compare matching pre-change and candidate builds to measure copy-then-seal against direct gather output. AWS-LC and ring rows are controls. |
| `tls-record` | Seal, or open, one TLS 1.3 application-data record of 64 B, 256 B, 1 KiB or 16 KiB through each production record adapter, without connection or transport work. Sealing includes framing, the content type and the tag; opening first copies the sealed body back into the record buffer, the same for every provider. Separates provider cryptography from the `tls-traffic` connection costs. |

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
using distinct bytes in each direction and changing both inputs between checks
to detect swapped or stale outputs. The timed payloads remain fixed.
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

`bench-provider-resources` observes this provider's live Rust heap on macOS and
Linux. It counts System allocator usable payload bytes, including TLS endpoints,
preparation, fixture transport and worker owners. Configuration construction also
includes its correctness-checking warmup. OS thread stacks, allocator arena
metadata and C allocations are outside this counter. The report separates
construction, repeated individual pairs and simultaneous pairs. A pair includes
both endpoints, so its peak conservatively bounds either endpoint; it is never
divided by two. Every individual pair also transfers a full TLS record.
The observer uses `GlobalAlloc`'s default moving reallocation, so old and new
payloads are counted while both are live. It conservatively forgoes production
System allocation's possible in-place reuse. It does not hide a moving
reallocation's temporary allocation inside `System::realloc`.

The selected qualification caps are 8 MiB of requested stack per worker, 1 MiB
of incremental heap per endpoint, 64 admitted handshakes and 8 MiB of shared
prepared-key storage. The observer uses the stricter 1 MiB cap for an entire
pair and the entire configuration-construction peak for the shared-key cap.
Each concurrent worker constructs its connections before the handshake barrier;
all completed pairs remain live until every worker completes. Resumed workers
have separately seeded session stores and share the Arc-owned signing keys and
trust-anchor verifiers. `--workload compact`, `intermediate` and `stateful` select
the existing catalog workloads. A cap failure retains its report and exits
nonzero. These observations do not establish whole-call stack bounds, hostile
input bounds, or admission control in an application.

Run comparisons on the same quiet native machine with the same compiler, features,
build profile and workload.
Record contention and unsupported environments.
These measurements do not establish constant-time behavior, network latency,
denial-of-service limits, or performance leadership by themselves.
The normal benchmark artifact workflow retains source, commands, build identity and raw results.

## Server-verifier rows

`tls-server-verifier` times server-certificate verification without a connection.
It builds rustls's WebPKI server verifier over the Debian or Fedora root bundle,
with this provider's algorithms (`rscrypto`) or AWS-LC's (`aws-lc-rs`), or builds `PreparedServerVerifier` (`rscrypto-prepared`).
The WebPKI `rscrypto` row borrows `DEFAULT_PROVIDER`.
It then verifies each of four captured origin chains and names at the chains' capture time.
Lifetime rows time one build plus 1, 3, or 10 verifications of one origin,
for the two rscrypto verifiers only.
Bundle parsing is untimed, and TLS 1.3 CertificateVerify is excluded.
Each row first verifies every origin outside timing.
[The fixture record](fixtures/server-verifier/README.md) identifies the bundles and chains.

## OpenSSL primitive comparisons

`openssl_primitives` also compares production SHA-2, SHA-3 and AES-GCM against OpenSSL 4.0.3.
It provides the missing SHA3-224 streaming comparison and a native IBM Z measurement path
for OpenSSL's CPACF dispatch.
Run `just bench openssl_primitives --list` for discovery, then measure one `openssl-<algorithm>` selector, such as `just bench openssl-sha3-224` or `just bench openssl-aes128-gcm`.
This optional target has the same external OpenSSL prerequisite and provenance output
as the TLS target.
It is excluded from default and hosted Bench runs.

Hash rows time context construction, every update, finalization and destruction.
Streaming absorbs 1 MiB in 64-byte or 4 KiB chunks.
AES-GCM `new-key/seal` and `new-key/open` construct a fresh key/context for each message;
they include the production API's input copies, authentication and destruction.
Both sides receive immutable input and write into preallocated output.
OpenSSL's EVP context allocation remains timed.
The rows use a 96-bit nonce, the same AAD and a 128-bit tag.
Repeated public fixture nonces protect no application data.

Each hash output and sealed ciphertext/tag must equal the other implementation's output
before timing.
Open rows must recover the fixture plaintext and reject a changed tag.
They call the same production operation that is measured.
The suite adds no direct FFI or unsafe code.
Its costs include the public API and cannot be interpreted as isolated compression
or cipher-kernel timings.

On IBM Z, build against target OpenSSL headers and libraries with assembly enabled,
then run on native hardware and retain the loaded library identity and `OPENSSL_s390xcap` overrides.
The workspace also generates its AWS-LC bindings on IBM Z and POWER64LE GNU,
so these builds need libclang.
Cross-compilation alone supplies no CPACF execution or performance evidence.
The shipped provider's C-dependency prohibition remains unchanged;
this independent measurement workspace owns the OpenSSL dependency.

## Competitor coverage and claims

Use the fastest measured external backend for each exact workload,
not a single preferred competitor averaged over the suite.
The catalog declares TLS handshake, traffic, and server-verifier comparisons for `just bench-compare`;
`rscrypto-webpki` remains a provider variant and cannot count as an external competitor.
Inspect individual medians and intervals as well as aggregate scores,
and retain losses and missing rows.
Compact-key and server-verifier lifetime rows compare configurations within rscrypto
and have no external comparison score.

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

## QUIC and Quinn

`quic_connections` adds 103 cases, all using QUIC v1.
Discover them with `just bench quic-provider --list`; measure one selector at a time:

```bash
just bench quic-handshake
just bench quic-packet
just bench quic-header
just bench quinn-connect
just bench quinn-retry
just bench quinn-address-token
just bench quinn-reset
```

The target is optional and has the same OpenSSL 4.0.3 prerequisite as the TLS binary.
OpenSSL supplies shared fixture/provenance support, not a QUIC comparison row.
CI does not provision this target, and `just bench all` omits it.
The `quinn` feature enables the approved pinned Quinn dependency and its production Bloom token replay log;
it does not enable Quinn's rustls 0.23 integration.
The rscrypto-rustls adapter connects Quinn to the pinned rustls 0.24-dev providers.

| Selector              | Cases | Timed work |
| --------------------- | ----- | ---------- |
| `quic-handshake`      | 12    | Both rustls QUIC TLS handshakes through raw TLS events, including Handshake and 1-RTT key derivation, connection allocation, input copies and destruction. No packet protection or Quinn transport. |
| `quic-packet`         | 36    | 1-RTT payload seal or open, 1200 or 1350 plaintext bytes, under each suite and provider. Retained keys and buffers; buffer restoration and packet-number handling are timed. |
| `quic-header`         | 18    | Apply or remove header protection under each suite and provider: short header, four-byte packet number and a fixed 16-byte sample. Header restoration is timed; key derivation is not. |
| `quinn-connect`       | 12    | Fresh Quinn endpoints and a full connection on both sides, datagram copies, immediate post-handshake control traffic and destruction. |
| `quinn-retry`         | 12    | The same connection with exactly one server Retry and successful address validation before acceptance. |
| `quinn-address-token` | 12    | A full TLS handshake consuming one real NEW_TOKEN token and processing two newly issued tokens. TLS resumption stays disabled. |
| `quinn-reset`         | 1     | Production stateless reset emission for a previously established connection that the server has lost; input copy, rate-limit check, random padding, reset authentication token and output are timed. |

Connection rows use P-256 server authentication, X25519 and each of the three TLS suites.
They compare `rscrypto`, `rscrypto-webpki`, `aws-lc-rs` and `ring` with the same certificate, validation time and ALPN as the TLS fixtures.
There is no client certificate, early data or TLS ticket issuance.
Configuration, private-key import, endpoint reset/token keys and token-cache seeding are untimed.
Shared credential setup also checks an untimed TCP connection.
The crypto handshake checks distinct authenticated transport-parameter bytes in both directions.

Packet and header rows compare three production providers;
certificate-verifier variants would do identical work.
They derive keys outside timing from the same public synthetic traffic secret through each
provider's production QUIC v1 key derivation.
Packet sizes describe plaintext, not UDP size: a packet adds a 13-byte header and 16-byte tag.
Header protection is a separate operation.
Seal uses increasing packet numbers;
open repeatedly authenticates a fixed independent-provider ciphertext.
The packet component has no transport replay policy.
Both operations restore their input with a timed copy into retained storage.
Throughput counts plaintext once.
These are hot-buffer component costs; they do not represent a cold-cache network stack.

The Quinn transport calls `Endpoint::connect`, `handle`, `accept` and `retry`, and drives the returned connections' events,
transmits and timers.
It copies one UDP datagram per send without sockets, GSO, loss or actual delay.
One-millisecond simulated scheduler ticks advance pacing and timers without sleeping.
The pump stops after both Connected events and delivery of all immediate datagrams;
it does not wait out every future delayed ACK or idle timer.
Every timed connection must take the selected Retry/token path.
The pinned Quinn queues one token when it accepts an already validated Initial and another
when it receives the Handshake packet.
The untimed seed connection issues one; each timed connection issues exactly two.
Token rows retain Quinn's one-entry `TokenMemoryCache` and its production `BloomTokenLog` across connections;
the cache keeps the newest token and discards the other.
Both issuance and eviction are timed.
Quinn's default replay log starts as a hash set and converts to Bloom filters at its memory limit.
Its growth during a row is timed; these short runs do not model a saturated replay log.
Without Quinn's `bloom` feature, its default token log would reject every address token;
the fixture checks accepted address validation so that fallback cannot masquerade
as a token benchmark.

The shared adapter always uses **rscrypto for Initial packets, Retry integrity, address-token
protection, connection-ID filtering and stateless reset**.
The negotiated TLS handshake, Handshake/1-RTT packet protection
and header protection use the selected provider.
Quinn comparisons therefore measure these configured stacks,
not wholly independent QUIC implementations.
Reset has one rscrypto row and no external comparison score.
It retains the original endpoint keys after losing the connection,
and advances simulated time by Quinn's default 20 ms minimum reset interval before each emission.
The rate limit remains active; measured CPU cost does not imply that emission frequency is allowed.

Each row checks its actual operation before timing.
Connection checks cover full handshake kind, suite, group, ALPN, server identity,
equal exporters and cross-provider interoperability in both directions.
Crypto handshake checks also authenticate application bytes with the emitted 1-RTT keys.
Packet/header checks require exact agreement with an independent provider;
packet checks reject altered tags, headers and packet numbers.
Retry checks require exactly one Retry; token checks require consumption,
validation and replacement.
A real client rejects a reset with an altered token and accepts the exact production reset.
Criterion `--test` runs these checks without a measurement campaign:

```bash
cargo bench --locked --manifest-path tools/provider-bench/Cargo.toml --features quinn --bench quic_connections -- --test
```

The catalog declares complete matched comparisons for all multi-provider families.
The provenance and interpretation rules above apply unchanged.
Keep raw estimates, uncertainty, source/build identities and all slower rows.
A bounded validation run proves neither a hardware limit nor worldwide performance leadership.
Network RTT, throughput under congestion, loss recovery,
multithreaded scaling and HTTP/3 are outside these CPU workloads.
Packet and header boundaries follow [RFC 9001](https://www.rfc-editor.org/rfc/rfc9001.html);
address validation and reset semantics follow [RFC 9000](https://www.rfc-editor.org/rfc/rfc9000.html).
