---
"rscrypto" = "patch"
---

Fix a timing difference in the RSA blinding-factor inverse on AArch64. Native timing tests on Graviton4 and Apple Silicon separated a fixed blinding factor from random ones in the 62-bit inverse. Blinded private operations now invert the factor multiplied by a fresh random mask, then multiply the result by the mask, so the inverse never reads the factor itself on any target. Results are unchanged. Each blinding attempt makes one more modulus-width entropy request, so the `*_with_random_fill*` signing methods call their entropy callback once more per attempt.
