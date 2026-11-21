---
"rscrypto" = "patch"
---

On RISC-V, P-384 ECDSA field arithmetic no longer branches on the secret reduction borrow. LLVM compiled the masked final subtraction of each field multiply and square to a conditional branch. Native DudeCT found the resulting timing difference in P-384 public-key derivation.
