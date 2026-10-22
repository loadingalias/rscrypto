---
"rscrypto" = "patch"
---

Speed up ECDSA P-384 verification on targets without P-384 field assembly, including RISC-V, POWER, IBM Z and portable builds: the portable Montgomery reduction uses the field prime's special limbs instead of multiplications, and squaring has its own kernel. P-256 and P-384 public-key import now checks the curve equation in the verification field, which removes the fixed-work software multiply from key import on RISC-V and IBM Z. Verification results and accepted keys are unchanged.
