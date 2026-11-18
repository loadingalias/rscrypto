# SLH-DSA

rscrypto implements SLH-DSA (FIPS 205) for all 12 parameter sets, in the pure form and as
HashSLH-DSA, with the RFC 9909 key and signature encodings. SLH-DSA rests only on the security of
its hash function. Its signatures are large and slow to make, so it fits trust roots and update
signing, which sign rarely and verify often.

## Choose a profile

Use **SLH-DSA-SHA2-128s** for a trust root or an update key. It has the smallest standard
signature (7,856 bytes), and its verification uses the hardware SHA-256 of x86-64, AArch64, and
Apple silicon. Use **SLH-DSA-SHA2-192s** (16,224 bytes) when a policy requires security category 3.
The `f` sets sign faster but double the signature size, so they suit a signer that signs often.

| Set | Public key | Secret key | Signature |
| --- | ---: | ---: | ---: |
| SHA2-128s, SHAKE-128s | 32 | 64 | 7,856 |
| SHA2-128f, SHAKE-128f | 32 | 64 | 17,088 |
| SHA2-192s, SHAKE-192s | 48 | 96 | 16,224 |
| SHA2-192f, SHAKE-192f | 48 | 96 | 35,664 |
| SHA2-256s, SHAKE-256s | 64 | 128 | 29,792 |
| SHA2-256f, SHAKE-256f | 64 | 128 | 49,856 |

Sizes are in bytes. Each set has a pure type, such as `SlhDsaSha2_128s`, and a HashSLH-DSA type,
such as `HashSlhDsaSha2_128sWithSha256`. The feature is `slh-dsa`.

## Use it

[`examples/slhdsa_update_verify.rs`](../examples/slhdsa_update_verify.rs) shows the whole flow:

1. Generate the root key pair with `try_generate_keypair` (operating-system entropy) or with
   `generate_keypair` and a caller-supplied generator.
2. Pin the root as its RFC 9909 SubjectPublicKeyInfo: `to_spki_der` on the public key.
3. Sign each manifest with `try_sign` or `sign_with` (hedged), or with `sign_deterministic`, under
   a protocol context of at most 255 bytes.
4. On the device, import the root with `from_spki_der` and call `verify_with_context`.

`from_spki_der` and `from_pkcs8_der` accept only the unique DER of the exact profile. A key of
another set, or the pure form where HashSLH-DSA is expected, returns
`SlhDsaKeyError::UnsupportedAlgorithm`. Every invalid signature, a signature of the wrong length,
and a context over 255 bytes return the same opaque `VerificationError`.

Secret keys are not `Clone`. Their SK.seed and SK.prf are cleared on drop, and `expose_secret` is
the only way to read them. Importing a secret key recomputes its public root, which costs one key
generation.

## Evidence

These checks cover every set unless the row names one.

| Evidence | Scope |
| --- | --- |
| NIST ACVP (`testdata/slhdsa/acvp/`) | All 1,248 pinned cases: key generation, signing, and verification, pure and pre-hash, for every FIPS 205 hash function |
| Differential tests | RustCrypto `slh-dsa` 0.2.0-rc.5 (key generation, deterministic and hedged signing) and `fips205` 0.5.0 (pure and HashSLH-DSA signing and verification) |
| RFC 9909 Appendix C (`testdata/slhdsa/rfc9909/`) | SHA2-128s only: the published key pair and the certificate signature |
| RFC 9909 encodings | SPKI and PKCS #8 for all 24 profiles against DER built from the RFC's OIDs, with one rejection test for each rule, and an exchange with RustCrypto and the OpenSSL CLI |
| Fuzzing | The `auth_slhdsa` target checks verification and the canonical key encodings for all 24 profiles |
| WebAssembly | Scalar and SIMD128 builds generate each set's ACVP key, sign one deterministic case again, and verify pure and pre-hash signatures |

## Limits

- **Timing.** `ct.toml` registers `signature.slhdsa` with a best-effort claim and no timing
  harness yet. Signing and key generation get DudeCT cases (fixed against random secret key) in a
  later change.
- **Stack and residue.** No linked stack bound or secret-residue evidence exists yet for SLH-DSA.
- **Targets.** POWER, IBM Z, and RISC-V run no small-set differentials or fuzzing. Windows AArch64
  runs nothing.
- **No Wycheproof suite** exists for SLH-DSA. OpenSSL and RustCrypto have no HashSLH-DSA codecs, so
  those encodings rest on the RFC-derived DER.
- NIST is working on smaller SLH-DSA sets for a limited number of signatures. Check that work
  before you freeze a root profile.
