---
"rscrypto" = "minor"
---

Add `hashes::expert::BLAKE3_PORTABLE_SOURCE`, the source text of the portable BLAKE3 compression function. rscrypto compiles the same file as its Portable kernel. The text uses only `core`, has no `unsafe` code and no conditional compilation, and builds with Rust 1.85 or later, edition 2024. Code that cannot link rscrypto can embed it inline, as a `mod` file, or with `include!`. The names, values, and signatures of its items are public API.
