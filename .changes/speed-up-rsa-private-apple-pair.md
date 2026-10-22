---
"rscrypto" = "patch"
---

Speed up RSA private operations on Apple AArch64. Keys whose two CRT primes share a width of 1024, 1536, or 2048 bits run both CRT exponentiations together, one in each lane of the vector unit. The constant-time schedule is unchanged and the results are identical. Other AArch64 targets keep the scalar Montgomery kernels.
