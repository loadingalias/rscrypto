---
"rscrypto" = "patch"
---

Speed up ECDSA signing and key generation on 64-bit RISC-V. Secret limb products now use the hardware multiplier with both operands' top bits set, as RISC-V ECDH already does, instead of a fixed-work software multiply. 32-bit RISC-V and IBM Z keep the software multiply. Native constant-time evidence qualifies the change; results are unchanged.
