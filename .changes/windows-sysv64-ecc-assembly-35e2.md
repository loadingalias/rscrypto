---
"rscrypto" = "patch"
---

Windows x86-64 now runs the same System V assembly as Linux x86-64 for ECDSA P-256 and P-384 signing and public-key derivation, P-256 ECDH, and X25519. Windows calls these routines through `extern "sysv64"`; before, ECDSA and X25519 there ran portable arithmetic, and P-256 ECDH ran separate Microsoft x64 wrapper copies of the same assembly.
