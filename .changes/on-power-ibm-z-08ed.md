---
"rscrypto" = "patch"
---

On POWER, IBM Z, and RV64, every build without `portable-only` now enables the nightly features that the accelerated backends use, whatever features it selects. These builds need nightly Rust, as `docs/platforms.md` states. A feature selection can no longer fail because the crate missed a nightly feature for it, as `chacha20-rng` alone did.
