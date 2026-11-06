---
"rscrypto" = "patch"
---

Speed up one-shot KMAC and cSHAKE for short messages on every target: 9–14% for KMAC and 10–15% for cSHAKE
on Apple M1 Pro, and 9–10% for KMAC and 4–6% for cSHAKE on Intel Granite Rapids. One-shot calls now build a
single sponge instead of cloning reset states, `bytepad` skips its zero padding instead of absorbing it, and
partial input lanes are assembled in a register. A KMAC state now holds two Keccak sponges instead of four.
