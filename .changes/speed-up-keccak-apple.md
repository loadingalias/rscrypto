---
"rscrypto" = "patch"
---

Speed up SHA-3, SHAKE, cSHAKE, KMAC, and HMAC-SHA-3 on Apple Silicon. The single-state Keccak permutation
runs as inline assembly with a fixed register allocation, about 10% faster than the previous kernel, whose
compiled rounds spilled a lane on the critical path. On M1 Pro, one-shot SHA-3 of up to one block takes
151–157 ns instead of 170–211 ns.
