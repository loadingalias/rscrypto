# Compliance

`rscrypto` provides FIPS-oriented building blocks, not a FIPS 140-3 validated module.

Validation requires a defined module boundary, operational environment, self-tests, documentation, and lab review. This crate classifies primitives for deployment planning; it does not claim a certificate.

## NIST-Aligned Building Blocks

Inside a typical boundary, these are the NIST-aligned items:

| Category | Examples |
|---------|----------|
| Symmetric AEAD | `Aes128Gcm`, `Aes256Gcm` (SP 800-38D) |
| Hash / XOF | `Sha*`, `Shake*` (`FIPS 180-4`, `FIPS 202`) |
| MAC / KDF | `HmacSha*`, `Kmac256`, `HkdfSha256`, `HkdfSha384` |
| Password-based KDF | `Pbkdf2Sha256`, `Pbkdf2Sha512` (SP 800-132) |
| RSA key generation | `RsaPrivateKey::generate` follows FIPS 186-5 Appendix A.1.3 probable-prime generation in code and seeds an internal HMAC_DRBG from `getrandom` |

This table is a standards-alignment inventory only; it is not a validation
claim. In particular, the RSA key-generation DRBG is internal implementation
machinery, not a crate-wide validated random bit generator service.

## Non-Approved In This Release

| Category | Examples |
|----------|----------|
| Cipher variants / AEAD | `Aes128GcmSiv`, `Aes256GcmSiv`, `Aegis256`, `ChaCha20Poly1305`, `XChaCha20Poly1305` |
| Hashes | `Blake*`, `Ascon*` (`SHA`/FIPS boundary not yet established) |
| Public-key primitives | `Ed25519*`, `X25519*` |
| Password hashing | `Argon2*`, `Scrypt` |
| Non-crypto | `Crc*`, `Xxh3`, `RapidHash` |

This table classifies APIs only; it is not a validation claim.

## Roadmap

FIPS validation would require a defined module boundary, selected operational
environments, algorithm validation, self-tests, documentation, and review by an
accredited lab. That work is tracked as a pre-1.0 roadmap item; until then,
describe `rscrypto` as FIPS-oriented building blocks, not a validated module.

Migration guides for replacing existing crypto crates live in
[`docs/migration/`](migration/). They are migration aids, not compliance
attestations.
