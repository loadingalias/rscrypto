---
"rscrypto" = "patch"
---

Speed up RSA private-key operations on 64-bit RISC-V and IBM Z. The blinding-factor inverse multiplies words with the hardware multiplier, with both operands' top bits set on RISC-V, as the RSA exponentiation already does, instead of a 64-step software multiply. 32-bit RISC-V keeps the software multiply. Native constant-time evidence qualifies each target; results are unchanged.
