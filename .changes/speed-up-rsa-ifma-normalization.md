---
"rscrypto" = "patch"
---

Speed up RSA private operations on AVX-512 IFMA CPUs by exposing the fixed vector count in
Montgomery carry normalization. Preserve the arithmetic, table access, dispatch and scratch sizes.
