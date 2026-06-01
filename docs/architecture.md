# Architecture

Internal module map and advanced entry points for contributors and users who need dispatch introspection.

## Module Hierarchy

```text
src/
+-- lib.rs              # Public API, re-exports
+-- aead/               # AES-GCM, AES-GCM-SIV, ChaCha20, XChaCha20, AEGIS, Ascon
+-- auth/               # HMAC, HKDF, PBKDF2, KMAC, Argon2, scrypt, PHC, Ed25519, RSA, X25519
+-- checksum/           # CRC families, config, buffered, introspection
+-- hashes/
|   +-- crypto/         # SHA-2, SHA-3, SHAKE, cSHAKE, Blake2, Blake3, Ascon, Keccak
|   +-- fast/           # XXH3, RapidHash
+-- hex.rs              # Hex encoding, DisplaySecret
+-- platform/           # CPU detection, SIMD dispatch
+-- backend/            # Internal dispatch infrastructure (curve25519, etc.)
+-- traits/             # Checksum, Digest, Mac, Xof, FastHash, Aead, ct, io
```

## Advanced Modules

| Module | Gate | Purpose |
|--------|------|---------|
| `checksum::config` | -- | Force-dispatch controls |
| `checksum::buffered` | `alloc` | Buffered CRC wrappers |
| `checksum::introspect` | `diag` | Kernel selection reporting |
| `hashes::introspect` | `diag` | Hash kernel reporting |
| `aead::introspect` | `diag` | AEAD backend reporting |
| `platform` | -- | CPU detection, override control |
| `traits::io` | `std` | `ChecksumReader/Writer`, `DigestReader/Writer` |
