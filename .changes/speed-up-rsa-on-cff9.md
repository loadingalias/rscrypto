---
"rscrypto" = "patch"
---

Speed up RSA on targets that use the portable Montgomery arithmetic, including POWER, RISC-V, IBM Z, WebAssembly, and x86-64 outside Linux. Squaring has its own kernel with half the cross products, multiplication folds the per-row shift into its reduction pass, and limb products no longer carry overflow checks. Results are unchanged.
