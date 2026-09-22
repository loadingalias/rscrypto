# RSA verification fixtures

These nine project-generated fixtures were introduced on 2026-05-22 in rscrypto
commit `e2996405`. They are covered by rscrypto's `MIT OR Apache-2.0` license and
have no external retrieval source. Each SPKI contains an RSA public key with
public exponent 65537. The corresponding signatures cover
`b"rscrypto RSA-PSS verification fixture"` and
`b"rscrypto RSA-PKCS1-v1_5 verification fixture"` with SHA-256.

`SHA256SUMS` is the complete three-key, six-signature inventory. Verify it from
this directory with `shasum -a 256 -c SHA256SUMS`. RSA integration tests, the
RSA example, RSA fuzz targets, source-level internal tests, and `benches/rsa.rs`
consume these fixtures. They live under `testdata/` because benchmarks are only
one of several consumers.
