# TLS connection benchmarks

This unpublished tool compares complete mutually authenticated TLS 1.3
connections through rscrypto-rustls, rustls/AWS-LC, rustls/*ring*, and OpenSSL 4.0.3.
It calls each library's production implementation. The workspace is separate
so its C comparison libraries do not enter the provider's shipped graph.

`just bench tls-provider --list` lists the catalog workloads without building
their fixtures. A case builds and checks its fixture only when it runs, so a
filtered campaign needs no unselected peer. Criterion's `--test` mode
(`cargo bench --manifest-path tools/provider-bench/Cargo.toml --features ml-dsa
--bench tls_connections -- --test`) checks every fixture once.
Use `--filter` for a bounded campaign. For example:

```sh
just bench tls-provider --filter '^tls13/handshake/p256/X25519/AES128-GCM-SHA256/'
just bench tls-provider --filter '^tls13/traffic/'
just bench-provider-scorecard rscrypto mldsa65 X25519MLKEM768 AES256-GCM-SHA384 full --iterations 256 --workers 4
```

This connection workload has no internal-kernel diagnostic mode; `--diag`
is rejected before compilation.

The linked OpenSSL version must be exactly 4.0.3. Use its development headers
and libraries; `OPENSSL_DIR` can select their installation. The benchmark catalog
records this as the target's external prerequisite. CI does not provision it, so
`just bench all` omits this target and the Bench workflow rejects it. The optional
`vendored-openssl` feature lets cross-target lint jobs build the bindings without
a target installation. The bindings currently vendor OpenSSL 3.6.3, so that
feature alone does not supply the required measurement baseline. The runtime
version check rejects it. With that feature enabled, `OPENSSL_NO_VENDOR=1`
selects the explicit 4.0.3 installation. C builds need a compiler, Perl and Make.

The catalog covers all seven credentials, four groups, three suites, full and
statelessly resumed connections, and five backends. *ring* has no ML-KEM or
ML-DSA, so profiles that need either have no `ring` row. Both peers use the same
credentials, roots, validation time, ALPN, group, suite and CertificateVerify
algorithm. Each credential chain is a root and a leaf, and every stack sends the
leaf alone; OpenSSL's automatic chain building is off. Early data is disabled.
In resumed workloads, every stack issues exactly one ticket per connection,
because OpenSSL issues at most one after a resumption. Every resumed sample
consumes a real ticket and processes its one replacement. Full workloads issue
no tickets.

Every connection, timed or not, must complete the selected handshake kind in
three flights, or four when a ticket follows. A HelloRetryRequest fails the
fixture. Before timing, a check connection must also show one accepted ticket
per resumed-workload connection and none otherwise, equal exporters, the
selected suite, group and ALPN, and the peer's leaf alone on both endpoints.
Application record rows check decrypted bytes before timing. Their untimed
setup handshake uses X25519 on every backend, so `ring` runs them too; record
keys do not depend on the key exchange group.

Ticket protection uses each stack's production implementation: this provider's
AES-256-GCM ticketer, the rustls/AWS-LC and rustls/*ring* ticketers, and
OpenSSL's default stateless tickets. Ticket formats and protection work therefore
differ. Resumption comparisons measure the complete configured stacks; they do
not isolate the negotiated record cipher's cost.

Configuration and private-key import occur outside connection timing. The
backends verify certificates differently:

- `rscrypto` is the provider's recommended configuration. Its
  `PreparedServerVerifier` and `PreparedClientVerifier` prepare the roots'
  P-256, P-384 and ML-DSA keys with the configuration. Each catalog leaf is
  signed by its root, so a full handshake with those credentials verifies both
  certificate chains with prepared keys. Both CertificateVerify keys are
  imported per connection.
- `rscrypto-webpki` uses the same provider with rustls's WebPKI verifiers, as
  `aws-lc-rs` and `ring` do. Compare it with those backends to isolate the
  provider's cryptography from anchor preparation.
- `openssl` uses OpenSSL's X.509 verification.

The two rscrypto backends do the same work for Ed25519 and RSA anchors, and for
resumed connections, which exchange no certificates. The provider also prepares
ML-DSA signing keys once, when it loads them. Each rustls backend advertises only
the credential's signature scheme. OpenSSL's RSA list also includes
`rsa_pkcs1_sha256`, which TLS 1.3 allows only for certificate signatures;
CertificateVerify uses RSA-PSS-SHA256 on every stack.

Connection allocation, handshake cryptography, certificate validation, byte
transport copies, ticket processing, close-notify creation and destruction are
timed. A bounded in-process byte queue connects the peers. It adds no network
RTT or loss. Each direction preallocates 32 KiB and copies with `memcpy` for
every stack. Close notifications are created on destruction; their delivery is
not part of this workload. Every stack decrypts into a retained plaintext
buffer and then copies to the caller, so each makes two plaintext copies.
The rustls clients keep tickets in rustls's default 256-entry store. The OpenSSL
client receives them through its new-session callback, without its internal
session cache. Record rows include both endpoints and transport copies on a
reused connection, at 1 KiB and 16 KiB.

At start, the benchmark prints one JSON line, `provider_bench_provenance`, to
standard error. It records:

- OpenSSL's version, configure flags and platform;
- on Linux, the paths and SHA-256 of the loaded libssl and libcrypto images,
  or the executable's when OpenSSL is linked statically; elsewhere, those of the
  libraries under the build's `OPENSSL_DIR`, which identify the configured
  installation rather than the image the loader chose;
- `RUSTFLAGS` and `CARGO_ENCODED_RUSTFLAGS` at build and run time;
- every `OPENSSL_*cap` capability override, which OpenSSL and AWS-LC both read.

The scorecard JSON includes the same object. Discovery (`--list`) and case
matching omit the line. Retain it with every result.

Build choices affect the backends unequally. The bench profile keeps
`overflow-checks = true`, as the repository's profiles do. That adds checks to
Rust code only, which includes rscrypto and rustls but not AWS-LC's C and
assembly or OpenSSL. On local macOS, `just bench` also builds Rust with
`-C target-cpu=native`; a direct `cargo bench` does not. Homebrew's OpenSSL 4.0.3
on Apple silicon includes assembly for AES-GCM, ChaCha20, P-256 and Keccak. It has
no specialized P-384 code, which needs `enable-ec_nistp_64_gcc_128`, so OpenSSL's
P-384 rows use its generic elliptic-curve arithmetic.

Criterion owns elapsed-time estimates and uncertainty. The supplementary
scorecard retains individual samples, empirical percentiles, process CPU,
wire flights and whole-process high-water RSS. RSS includes the libraries,
fixture, key owners and worker stacks. It is not an allocation count, heap
bound, or per-connection memory bound. Up to 16 workers and 10,000 connections
per worker are accepted; each resource run has a ten-minute watchdog.

Run comparisons on the same quiet native machine with the same compiler,
features, build profile and workload. Record contention and unsupported
environments. These measurements do not establish constant-time behavior,
network latency, denial-of-service limits, or performance leadership by
themselves. The normal benchmark artifact workflow retains source, commands,
build identity and raw results.
