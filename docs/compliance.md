# Compliance Posture

`rscrypto` can support a compliance program, but it is not compliance by
itself. It is a pure Rust cryptographic primitives crate with public source,
explicit feature flags, documented security boundaries, vector coverage, and a
scoped constant-time evidence model.

Use it when your policy allows non-validated Rust primitives, or when you are
building your own module boundary and evidence package. Do not use it as a
FIPS 140-3 answer by itself.

External standards references in this page were checked against NIST CSRC on
2026-06-23. Regulated deployments should check the current standard text and
their own assessor requirements before release.

## Quick Answer

| User question | Answer |
|---|---|
| Do I get a FIPS 140-3 validated module by depending on `rscrypto`? | No. There is no CMVP certificate for this crate today. |
| Can I use `rscrypto` inside a FIPS-oriented project? | Possibly, if your project owns the module boundary, operational environment, allowed algorithms, self-tests, documentation, and validation path. |
| Which APIs are standards-aligned? | See the inventory below. It is an algorithm and API map, not a validation claim. |
| Which evidence can I cite during review? | Start with [`constant-time.md`](constant-time.md), [`test-vector-coverage.md`](test-vector-coverage.md), [`features.md`](features.md), and [`platforms.md`](platforms.md). |
| What if procurement requires a validated cryptographic module? | Use a validated module, or validate a larger module that includes `rscrypto` under a defined boundary. |

## Standards-Aligned Primitives

These surfaces are useful for NIST-oriented design reviews because the public
API maps to named standards or profiles. They are not FIPS 140-3 validation
claims.

| Area | `rscrypto` surface | Boundary users must still own |
|---|---|---|
| AES-GCM AEAD | `Aes128Gcm`, `Aes256Gcm` (`aes-gcm`) follow the SP 800-38D GCM shape. | Key lifecycle, nonce lifecycle, invocation limits, protocol binding, and allowed-use policy. |
| SHA-2 / SHA-3 / SHAKE | `Sha224`, `Sha256`, `Sha384`, `Sha512`, `Sha512_256`, `Sha3_*`, `Shake128`, `Shake256`. | Algorithm selection, message/domain separation, and any protocol-specific hash profile. |
| KMAC / cSHAKE | `Kmac256`, `Cshake256`. | Customization strings, key management, and protocol profile. |
| HMAC / HKDF / PBKDF2 | `HmacSha*`, `HkdfSha256`, `HkdfSha384`, `Pbkdf2Sha256`, `Pbkdf2Sha512`. | Key separation, salt/IKM policy, iteration counts, output lengths, and password policy. |
| ECDSA | P-256/SHA-256 and P-384/SHA-384 signing and verification (`ecdsa-p256`, `ecdsa-p384`, `ecdsa`). | Key generation policy, signature format, protocol profile, and acceptance criteria. |
| RSA | RSA-PSS, RSASSA-PKCS1-v1_5, OAEP, RSAES-PKCS1-v1_5, DER import/export, and `RsaPrivateKey::generate` under the crate's FIPS 186-5 Appendix A.1.3 probable-prime contract. | Entropy source, key policy, protocol profile, padding choice, blinding policy, and module-level validation. |
| ML-KEM | `MlKem512`, `MlKem768`, `MlKem1024` expose the FIPS 203 parameter sets with typed keys, ciphertexts, shared secrets, prepared-key paths, ACVP vectors, and `fips203` differential tests. | Entropy source, key establishment protocol, hybrid/PQ migration policy, and validation boundary. |

For exact public types and features, use [`types.md`](types.md) and
[`features.md`](features.md).

## Not a FIPS-Oriented Claim

The following APIs may be correct and useful, but they should not be presented
as FIPS 140-3 validated, CMVP certified, or part of the FIPS-oriented inventory
above unless your own compliance target explicitly allows them:

| Area | Examples |
|---|---|
| Misuse-resistant or non-NIST AEADs | `Aes128GcmSiv`, `Aes256GcmSiv`, `ChaCha20Poly1305`, `XChaCha20Poly1305`, `Aegis256` |
| Other hashes / XOFs | `Blake*`, `Blake3`, `Ascon*`, `Xxh3`, `RapidHash` |
| Other public-key primitives | `Ed25519*`, `X25519*` |
| Password hashing outside SP 800-132 | `Argon2*`, `Scrypt` |
| Checksums and fast hashes | `Crc*`, `Xxh3`, `RapidHash` |

This table is about compliance positioning, not engineering quality. For
example, Argon2 and scrypt are appropriate password-hashing choices in many
systems, but they are not a FIPS 140-3 validation claim.

## Evidence Users Can Review

`rscrypto` publishes evidence that can help a security review or vendor-risk
review, but none of it replaces an audit or validation certificate.

| Evidence | Where |
|---|---|
| Security posture and non-claims | [`../README.md#security`](../README.md#security) and this page |
| Constant-time threat model, target scope, and evidence rules | [`constant-time.md`](constant-time.md) and [`../ct.toml`](../ct.toml) |
| Official vectors, Wycheproof coverage, ACVP ML-KEM vectors, differential tests, fuzz corpus replay, and Miri coverage | [`test-vector-coverage.md`](test-vector-coverage.md) |
| Feature and dependency control | [`features.md`](features.md) |
| Platform and dispatch model | [`platforms.md`](platforms.md) |
| Vulnerability reporting process | [`../SECURITY.md`](../SECURITY.md) |

## What Users Still Own

If `rscrypto` is part of a regulated system, the integrator owns:

- The cryptographic module boundary and operational environment.
- Algorithm allowlists for the target framework and deployment.
- Key generation, storage, rotation, destruction, and access control.
- Entropy-source selection and health policy.
- Nonce, salt, and counter lifecycle rules.
- Power-up self-tests, known-answer tests, and error-state behavior required by
  the target validation.
- Protocol composition, certificate/path validation, and key-establishment
  policy.
- Build provenance, compiler flags, target features, and binary distribution.
- Assessor, lab, auditor, or customer evidence requests.

`portable-only` can help audit-constrained builds by forcing runtime dispatch
toward portable backends. It does not remove SIMD code from the binary, create
a constant-time proof, or create a FIPS validation boundary.

## Suggested Wording

Use wording like this in downstream docs:

```text
This project uses rscrypto as a pure Rust cryptographic primitives crate.
rscrypto is not a FIPS 140-3 validated module and does not claim third-party
audit or formal verification. Selected primitive APIs are standards-aligned and
are backed by public test-vector, differential-test, platform, and scoped
constant-time evidence documented by the project.
```

Avoid:

```text
FIPS validated
FIPS certified
audited
approved module
drop-in compliance replacement
```

## Primary References

- [NIST CMVP FIPS 140-3 Standards](https://csrc.nist.gov/projects/cryptographic-module-validation-program/fips-140-3-standards)
- [NIST FIPS 180-4: Secure Hash Standard](https://csrc.nist.gov/pubs/fips/180-4/upd1/final)
- [NIST FIPS 202: SHA-3 Standard](https://csrc.nist.gov/pubs/fips/202/final)
- [NIST FIPS 203: Module-Lattice-Based Key-Encapsulation Mechanism](https://csrc.nist.gov/pubs/fips/203/final)
- [NIST FIPS 186-5: Digital Signature Standard](https://csrc.nist.gov/pubs/fips/186-5/final)
- [NIST SP 800-38D: GCM and GMAC](https://csrc.nist.gov/pubs/sp/800/38/d/final)
- [NIST SP 800-132: Password-Based Key Derivation](https://csrc.nist.gov/pubs/sp/800/132/final)
- [NIST SP 800-185: SHA-3 Derived Functions](https://csrc.nist.gov/pubs/sp/800/185/final)
