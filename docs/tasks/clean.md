# Code cleanup tasks

- [ ] **T009 Derive the Ed25519 SIMD tables at compile time** (20,295 existing source lines).
  Derive the AVX2 and IFMA radix-16 representations from the portable table.
  Keep conversion outside signing.
  For all 512 entries, splitting each IFMA limb into 26 and 25 bits reproduces the stored AVX2 words.
  Existing tests compare every entry with portable field arithmetic
  and the exact vector representation.
  Compare emitted bytes, including partially reduced limbs.
  Check compile time and binary size under the affected features before adopting the change.
  [AVX2 table](../../src/auth/ed25519/basepoint_radix16_avx2.rs) · [IFMA table](../../src/auth/ed25519/basepoint_radix16_ifma.rs) · [Portable table](../../src/auth/ed25519/basepoint_tables.rs) · [Representation checks](../../src/auth/ed25519/point_avx2.rs#L1102)

## Intentional copies

- Keep the provider's QUIC vectors in its crate.
  Its tests consume those files from the independently packaged archive.
  [Vector provenance](../../rscrypto-rustls/tests/vectors/README.md) · [Package check](../../scripts/check/package.py#L85)
- Keep the generated paired RSA kernel source.
  Its generator checks that the committed output matches.
  [Generated kernel](../../src/auth/rsa_aarch64_pair.rs) · [Generator](../../scripts/asm/rsa_pair.py#L8)
