# TLS connection benchmarks

This unpublished tool compares complete mutually authenticated TLS 1.3
connections through rscrypto-rustls, rustls/AWS-LC, and OpenSSL 4.0.3.
It calls each library's production implementation. The workspace is separate
so its C comparison libraries do not enter the provider's shipped graph.

`just bench tls-provider --list` validates and lists the catalog workloads.
Use `--filter` for a bounded campaign. For example:

```sh
just bench tls-provider --filter '^tls13/handshake/p256/X25519/AES128-GCM-SHA256/'
just bench tls-provider --filter '^tls13/traffic/'
just bench-provider-scorecard rscrypto mldsa65 X25519MLKEM768 AES256-GCM-SHA384 full --iterations 256 --workers 4
```

This connection workload has no internal-kernel diagnostic mode; `--diag`
is rejected before compilation.

The linked OpenSSL version must be exactly 4.0.3. Use its development headers
and libraries; `OPENSSL_DIR` can select their installation. The optional
`vendored-openssl` feature lets cross-target lint jobs build the bindings without
a target installation. The bindings currently vendor OpenSSL 3.6.3, so that
feature alone does not supply the required measurement baseline. The runtime
version check rejects it. With that feature enabled, `OPENSSL_NO_VENDOR=1`
selects the explicit 4.0.3 installation. C builds need a compiler, Perl and Make.

The catalog covers all seven credentials, four groups, three suites, full and
statelessly resumed connections, and all three backends. Both peers use the
same credentials, roots, validation time, ALPN, group, suite and
CertificateVerify algorithm. Early data is disabled. Every resumed sample
consumes a real ticket and processes its replacement. The fixture checks both
endpoints' resumption state and exporter equality before timing. Application
record rows check decrypted bytes before timing.

Ticket protection uses each stack's production implementation: this provider's
AES-256-GCM ticketer, rustls/AWS-LC's AES-256-CBC plus HMAC-SHA256 ticketer, and
OpenSSL's default stateless tickets. Ticket formats and protection work therefore
differ. Resumption comparisons measure the complete configured stacks; they do
not isolate the negotiated record cipher's cost.

Configuration and private-key import occur outside connection timing.
Connection allocation, handshake cryptography, certificate validation, byte
transport copies, ticket processing, close-notify creation and destruction are
timed. A bounded in-process byte queue connects the peers. It adds no network
RTT or loss. Close notifications are created on destruction; their delivery is
not part of this workload. Record rows include both endpoints and transport
copies on a reused connection, at 1 KiB and 16 KiB.

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
