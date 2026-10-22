---
"rscrypto" = "patch"
---

Speed up RSA private-key operations on 64-bit RISC-V and IBM Z. The 31-bit blinding-factor inverse multiplies words with the hardware multiplier, as the RSA exponentiation already does, instead of a 64-step software multiply; on RISC-V it sets both operands' top bits. 32-bit RISC-V keeps the software multiply. Native constant-time evidence qualifies each target; results are unchanged.
