---
"rscrypto" = "patch"
---

On POWER, IBM Z, RV64, and RV32, a build that selects no feature with a nightly backend now compiles on stable Rust 1.100 without portable-only. For example, websocket-sha1 alone builds on stable RV64 and IBM Z. docs/platforms.md lists the features that need nightly Rust on each target.
