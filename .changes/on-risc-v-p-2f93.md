---
"rscrypto" = "patch"
---

On RISC-V, ECDSA no longer branches on secrets in two places. LLVM compiled the masked final subtraction of each P-384 field multiply and square to a conditional branch on the reduction borrow, and native DudeCT found the timing difference in P-384 public-key derivation. LLVM also compiled the final sign selection of the P-256 and P-384 safegcd order inversion, which computes the signing nonce inverse, to a branch on the sign of a secret-derived value.
