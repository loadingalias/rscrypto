---
"rscrypto" = "patch"
---

Reduce Linux AArch64 RSA private-operation overhead with paired Montgomery strip loads and fixed
five-bit exponent windows that reuse existing scratch capacity. Give 16-limb CRT reduction a
register window, normalize within the kernel, and omit the cancelled lowest product. Expose
the 16-limb portable kernels and existing five-bit window schedule on POWER, IBM Z and RISC-V.
Keep the original scalar kernels and the four-bit fallback when the existing scratch does not
hold the wider table.
