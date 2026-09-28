---
"rscrypto" = "patch"
---

Mask the decryption inverse NTT in prepared ML-KEM decapsulation keys. Preparation
derives a secret dense polynomial from the implicit-rejection secret; decryption adds
it before the inverse NTT and removes its transform afterwards, so a ciphertext cannot
make the transform process sparse data. Outputs are unchanged and per-call cost is
256 modular additions. One-shot decapsulation is unchanged.
