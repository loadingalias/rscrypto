# Migration: `openssl` -> `rscrypto`

`openssl` is a binding to a full system cryptography, TLS, X.509, and provider
toolkit. rscrypto can replace selected primitive operations. It does not replace
OpenSSL as a platform.

This repo does not depend on the Rust `openssl` crate for migration examples.
RSA interoperability is checked against the OpenSSL CLI when available in the
RSA test lane.

## Cargo.toml

```toml
# Before
openssl = "0.10"

# After: choose only the primitive features you use
rscrypto = { version = "0.3.1", default-features = false, features = ["sha2", "hmac", "hkdf", "pbkdf2", "aes-gcm", "chacha20poly1305", "ed25519", "x25519", "rsa"] }
```

## Map

| OpenSSL use | rscrypto | Status |
|---|---|---|
| SHA-2 / SHA-3 / BLAKE2 hashing | hash types | Supported primitive replacement |
| HMAC / HKDF / PBKDF2 | auth types | Supported primitive replacement |
| AES-GCM, ChaCha20-Poly1305 | AEAD types | Supported primitive replacement |
| Ed25519, X25519 | auth key types | Supported primitive replacement |
| RSA-PSS, RSASSA-PKCS1-v1_5, OAEP, RSAES-PKCS1-v1_5 | RSA key/profile APIs | Supported for implemented profiles |
| TLS, X.509 path validation, engines, providers, CMS, PKCS#12, OCSP | none | Keep OpenSSL or a protocol crate |

## Practical Path

1. Separate primitive calls from platform calls.
2. Move primitive calls to the matching focused guide:
   - hashes: `RustCrypto/sha2.md`, `RustCrypto/sha3.md`, `RustCrypto/blake2.md`
   - MAC/KDF: `RustCrypto/hmac.md`, `RustCrypto/hkdf.md`, `RustCrypto/pbkdf2.md`
   - AEAD: `RustCrypto/aes-gcm.md`, `RustCrypto/chacha20poly1305.md`
   - signatures/RSA: `RustCrypto/ed25519-dalek.md`, `RustCrypto/rsa.md`
3. Keep OpenSSL for TLS, PKI, provider, engine, and FIPS-provider behavior.

## RSA Boundary

rscrypto supports DER key import/export, RSA-PSS, RSASSA-PKCS1-v1_5, OAEP, and
RSAES-PKCS1-v1_5 for the implemented SHA-2 profiles. The RSA test lane checks
interoperability against RustCrypto `rsa`, `ring`, AWS-LC, CAVP/Wycheproof
vectors, and OpenSSL CLI when it is installed.

Use `RustCrypto/rsa.md` for concrete rscrypto RSA call-site examples. Keep
OpenSSL if you need certificate validation, provider configuration, engine
integration, or OpenSSL FIPS provider semantics.
