---
"rscrypto" = "patch"
---

With the `portable-only` feature, or under Miri, ECDSA no longer calls its AArch64 and x86-64 assembly.
It uses the portable Rust implementation, as other targets do.
