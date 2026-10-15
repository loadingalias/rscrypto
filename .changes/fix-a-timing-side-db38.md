---
"rscrypto" = "patch"
---

Fix a timing side channel in portable P-256 ECDH agreement on RISC-V: LLVM compiled one word of the window-table selection into a branch on the secret scalar digit. The selection mask is now opaque on RISC-V, IBM Z, and x86-64, as for the field limbs; results are unchanged.
