---
"rscrypto" = "patch"
---

Mask the decryption inverse NTT in prepared ML-KEM decapsulation keys.
Preparation derives a secret dense polynomial from the implicit-rejection secret.
Decryption adds it before the inverse NTT, and the fused final pass of the transform removes it.
A ciphertext therefore cannot make the transform process sparse data.
Outputs and the work for each call do not change.
One-shot decapsulation does not change.
