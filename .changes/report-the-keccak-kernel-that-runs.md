---
"rscrypto" = "patch"
---

With the `diag` feature, `hashes::introspect::kernel_for` reports the Keccak kernel that SHA-3, SHAKE, and cSHAKE run: the AVX-512 or BMI rounds on x86-64, the SHA3 instructions on Apple AArch64, the lazy-rotation rounds on other AArch64, and KIMD on IBM Z. Before, it read a separate table and reported `portable` on every CPU.
