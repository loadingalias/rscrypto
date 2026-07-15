---
"rscrypto" = "patch"
---

Hardened release finalization so signed tags require successful exact-commit CI,
Cargo graph assurance, constant-time evidence, and RSA evidence. Adopted
cargo-rail 0.17.3 for recoverable release pushes and isolated fast-hash
allocation accounting from parallel test-harness threads.
