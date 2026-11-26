# Code cleanup tasks

## Intentional copies

- Keep the provider's QUIC vectors in its crate.
  Its tests consume those files from the independently packaged archive.
  [Vector provenance](../../rscrypto-rustls/tests/vectors/README.md) · [Package check](../../scripts/check/package.py#L85)
- Keep the generated paired RSA kernel source.
  Its generator checks that the committed output matches.
  [Generated kernel](../../src/auth/rsa_aarch64_pair.rs) · [Generator](../../scripts/asm/rsa_pair.py#L8)
- Keep the generated Ed25519 AVX2 and IFMA radix-16 tables.
  Compile-time derivation from the portable table reproduced every word and kept the binary size.
  It added 450 ms to each x86_64 `cargo check` with `ed25519,x25519` (772 ms to 1,222 ms).
  Each AVX2 word is an IFMA limb split into 26 and 25 bits.
  Tests compare every entry with portable field arithmetic and the runtime vector conversion.
  [AVX2 table](../../src/auth/ed25519/basepoint_radix16_avx2.rs) · [IFMA table](../../src/auth/ed25519/basepoint_radix16_ifma.rs) · [Representation checks](../../src/auth/ed25519/point_avx2.rs#L1118)
