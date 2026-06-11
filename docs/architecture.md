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
+-- platform/           # CPU feature detection and dispatch metadata
+-- backend/            # Internal caches and shared kernels (curve25519, Ascon)
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

## Keccak Direct-Call Permuter

The direct-call Keccak permuter in `src/hashes/crypto/keccak.rs` should remain
portable for x86_64 and generic targets unless new measurements overturn this
decision.

Rejected SIMD paths from the closed KECCAK-4 evaluation:

- AVX-512 chi-only was 9-38% slower on Zen 4, Zen 5, Ice Lake, and Sapphire
  Rapids because GPR/SIMD crossing cost outweighed `VPTERNLOG` savings.
- AVX2 was worse than AVX-512 because it lacks `VPTERNLOG` and needs three ops
  for chi.
- BMI2 did not justify a separate path because LLVM already emits `RORX`, and
  `ANDN` saved less than five ops per round after lane-complementing chi.
- Full SIMD needs at least 13 YMM registers for 25 lanes, while theta, rho, and
  pi do not map cleanly enough to recover the register pressure.

Do not reopen this without benchmark rows that include the current portable
direct-call permuter. Keep the aarch64 two-state SHA3 CE path separate from this
single-state portable decision.
