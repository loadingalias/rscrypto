---
"rscrypto" = "patch"
---

Consolidated CI validation into explicit quality, feature-contract, native,
cross-target, and supply-chain owners. The complete compile and executable
feature matrices remain blocking, MUSL targets are now passed explicitly, and
IBM Z, POWER10, and RISC-V runners retain native all-features correctness and
backend-equivalence coverage without repeating architecture-independent work.

Updated repository automation to cargo-rail 0.17.0 and cargo-rail-action v5.
Configuration sync now records the public open-world consumer boundary, CI
checks the compiler-backed unified Cargo graph, and pull requests require
change-file coverage from the planner's resolved base reference.
