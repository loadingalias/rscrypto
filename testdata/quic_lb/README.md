# QUIC-LB connection-ID vectors from independent implementations

The two vector files are independent evidence for the QUIC-LB connection-ID encoding of
[draft-ietf-quic-load-balancers-22](https://datatracker.ietf.org/doc/draft-ietf-quic-load-balancers-22/).
rscrypto code computed none of these bytes. `tests/quic_lb.rs` reads them, and
`rscrypto-rustls/tests/vectors/quic_lb/` holds byte-equal copies for the provider's tests.

Each line is `rotation key server_id nonce connection_id`: the rotation codepoint in decimal,
then hexadecimal octets. The first octet of each connection ID is the rotation in its three high
bits and the server-ID and nonce total in its five low bits (draft section 3.3). The encrypted
octets follow it.

- `quiche-vectors.txt` (351 lines): google/quiche made every line. It covers each server-ID length
  from 1 to 15 with each nonce length from 4 that keeps the total at 19 or less, except a nonce of
  17 or 18 octets: quiche refuses nonces longer than 16 octets (`kLoadBalancerMaxNonceLen`). That
  is 117 of the draft's 120 length pairs. The harness issues each ID with quiche's
  `LoadBalancerConfig::FourPassEncrypt` or `BlockEncrypt`, as `LoadBalancerEncoder` does, and
  checks it with `LoadBalancerDecoder::GetServerId`.
- `hopf-vectors.txt` (9 lines): hopf-quic 0.7.0 made every line, for the 3 pairs that quiche
  refuses: server-ID length 1 with nonce 17 and 18, and length 2 with nonce 17. The emitter
  checks each ID with `QuicLbConfig::decode_server_id`.
- Each pair has three lines, one for each key: the Appendix B.2 key and two generated keys. The
  rotation steps through 0 to 6 from line to line. The server IDs, nonces, and generated keys come
  from a splitmix64 stream with a fixed seed in each generator.

## Provenance

Made on 2026-11-18 on an ephemeral AWS c8i.4xlarge host, Ubuntu 26.04.1 LTS (Linux 7.0.0-1016-aws).

| Input | Identity |
|---|---|
| google/quiche | commit `00f72ede07a4e6c58116a6411a9f21662c4e292e` (2026-11-16) |
| Bazel | 8.2.1 through Bazelisk v1.30.0 (`a248ae2b2b62b5baebb748f6fba7203bff1a27e5d01b55399a8406744cb539e8`) |
| C compiler | clang 23.1.6 (the rscrypto tooling LLVM) |
| quiche harness binary | SHA-256 `b346bf6667876aea66f381371023f2896ae118d09b7164880d6d8cee704d3153` |
| hopf-quic | 0.7.0 from crates.io (`eb04ba46f3ae5b6c52e017def4800b6c1e119821beec1421d73e0b5e457503f0`) |

`SHA256SUMS` lists both vector files. Verify them from this directory with
`shasum -a 256 -c SHA256SUMS`.

## Make the files again

The harness sources in [`harness/`](harness/) are provenance only: no Cargo package builds or
includes them. On an x86-64 Ubuntu host:

```bash
sudo apt install libicu-dev clang lld
curl -sSfL -o bazelisk https://github.com/bazelbuild/bazelisk/releases/download/v1.30.0/bazelisk-linux-amd64
echo "a248ae2b2b62b5baebb748f6fba7203bff1a27e5d01b55399a8406744cb539e8  bazelisk" | sha256sum -c -
chmod +x bazelisk
git clone https://github.com/google/quiche && git -C quiche checkout 00f72ede07a4e6c58116a6411a9f21662c4e292e
mkdir quiche/quiche/quic_lb_harness
cp harness/quic_lb_vectors.cc quiche/quiche/quic_lb_harness/
cat harness/quiche-rule.bazel >> quiche/quiche/BUILD.bazel
(cd quiche && ../bazelisk build -c opt //quiche:quic_lb_vectors)
quiche/bazel-bin/quiche/quic_lb_vectors vectors > quiche-vectors.txt
```

The rule goes into quiche's own `BUILD.bazel`, because `//quiche:quiche_core` declares no visibility.

For `hopf-vectors.txt`, unpack the hopf-quic 0.7.0 crate and make `QuicLbConfig::encode` public
(`src/transport/quic_lb.rs`), so that the caller chooses the nonce. That is the only change. Then
build `harness/hopf_vectors.rs` as the `main.rs` of a package with this manifest, and run it:

```toml
[package]
name = "hopf-vectors"
version = "0.0.0"
edition = "2024"
publish = false

[workspace]

[dependencies]
hopf-quic = { path = "/path/to/hopf-quic-0.7.0" }
```
