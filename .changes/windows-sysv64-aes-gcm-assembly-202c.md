---
"rscrypto" = "patch"
---

Windows x86-64 now runs the same System V VAES-512 assembly as Linux x86-64 for AES-128-GCM and AES-256-GCM sealing and opening, and for AES-GCM-SIV counter mode, on processors with VAES, VPCLMULQDQ, and AVX-512. Windows calls the kernels through `extern "sysv64"`; before, it ran intrinsics only. Each AES-GCM context on Windows now keeps 128 GHASH powers, as on Linux, instead of 16.
