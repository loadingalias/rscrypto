# Compliance

`rscrypto` provides FIPS-oriented building blocks, not a FIPS 140-3 validated module.

Validation requires a defined module boundary, operational environment, self-tests, documentation, and lab review. This crate classifies primitives for deployment planning; it does not claim a certificate.

## NIST-Aligned Building Blocks

Inside a typical boundary, these are the NIST-aligned items:

| Category | Examples |
|---------|----------|
| Symmetric AEAD | `Aes256Gcm` (SP 800-38D) |
| Hash / XOF | `Sha*`, `Shake*` (`FIPS 180-4`, `FIPS 202`) |
| MAC / KDF | `HmacSha*`, `Kmac256`, `HkdfSha256`, `HkdfSha384` |
| Password-based KDF | `Pbkdf2Sha256`, `Pbkdf2Sha512` (SP 800-132) |

## Non-Approved In This Release

| Category | Examples |
|----------|----------|
| Cipher variants / AEAD | `Aes256GcmSiv`, `Aegis256`, `ChaCha20Poly1305`, `XChaCha20Poly1305` |
| Hashes | `Blake*`, `Ascon*` (`SHA`/FIPS boundary not yet established) |
| Public-key primitives | `Ed25519*`, `X25519*` |
| Password hashing | `Argon2*`, `Scrypt` |
| Non-crypto | `Crc*`, `Xxh3`, `RapidHash` |

This table classifies APIs only; it is not a validation claim.

**NOTE**: In a perfect world I would define the ideal FIPS module boundary within `rscrypto`; pick the operational environments covering x86_64 Linux, aarch64 Linux, and aarch64 MacOS; then have NIST's CAVP validate the algorithms. This would allow me to contact an accredited CSTL lab and get the module validated for FIPS. This costs about $25k to get done, end to end. I am still working on a 2021 MBP M1 and can barely afford CICD costs. This will have to wait.
