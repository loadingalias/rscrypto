---
"rscrypto" = "minor"
---

Raise the minimum supported Rust version to 1.100.0, the release that
stabilizes the allocator API subset. Until Rust 1.100 is stable, build with a
1.100 nightly. POWER, IBM Z, and RISC-V builds require a nightly compiler even
after that; `docs/platforms.md` lists their unstable features.

Exclude x86 SIMD backends on soft-float x86-64 targets such as
`x86_64-unknown-none`. Those targets never selected them, because their
compile-time capabilities lack SSE2; the portable backends remain. Rust is
phasing out `#[target_feature]` SSE functions on soft-float targets.
