---
"rscrypto" = "patch"
---

Mask the decryption inverse NTT in prepared ML-KEM decapsulation keys. Preparation
derives a secret dense polynomial from the implicit-rejection secret; decryption adds
it before the inverse NTT, and the transform's fused final pass removes it, so a
ciphertext cannot make the transform process sparse data. Outputs and per-call work
are unchanged. One-shot decapsulation is unchanged.
