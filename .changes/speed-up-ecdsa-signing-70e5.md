---
"rscrypto" = "patch"
---

Speed up ECDSA signing and key generation on 64-bit RISC-V and IBM Z, and P-256 and P-384 ECDH on IBM Z. Secret limb products now use the hardware multiplier, with both operands' top bits set on RISC-V, as RSA private operations and RISC-V ECDH already do, instead of a fixed-work software multiply. 32-bit RISC-V keeps the software multiply. Native constant-time evidence qualifies each target; results are unchanged.
