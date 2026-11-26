# Code cleanup tasks

## Intentional copies

- Keep the provider's QUIC vectors in its crate.
  Its tests consume those files from the independently packaged archive.
  [Vector provenance](../../rscrypto-rustls/tests/vectors/README.md) · [Package check](../../scripts/check/package.py#L85)
- Keep the generated paired RSA kernel source.
  Its generator checks that the committed output matches.
  [Generated kernel](../../src/auth/rsa_aarch64_pair.rs) · [Generator](../../scripts/asm/rsa_pair.py#L8)
- Keep the generated Ed25519 IFMA radix-16 table.
  Compile-time derivation from the portable table reproduced every word and kept the binary size.
  It added 450 ms to each x86_64 `cargo check` with `ed25519,x25519` (772 ms to 1,222 ms).
  The AVX2 path splits each selected IFMA entry into 26- and 25-bit limbs, so it needs no table.
  On Windows Granite Rapids with IFMA masked, the split moved no fixed-base row beyond the 0.5% run-to-run
  drift (Ed25519 public key +0.3%, X25519 public key -0.2%).
  Tests compare every entry with portable field arithmetic and each AVX2 selection with the runtime conversion.
  [IFMA table](../../src/auth/ed25519/basepoint_radix16_ifma.rs) · [Representation checks](../../src/auth/ed25519/point_avx2.rs#L1105)
