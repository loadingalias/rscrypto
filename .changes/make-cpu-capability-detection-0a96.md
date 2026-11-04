---
"rscrypto" = "patch"
---

Make CPU capability detection lock-free and owned. std builds no longer initialize detection
through a std `OnceLock`: every build caches it the same way, each first caller detects on its own
stack, and one compare-and-swap publishes the result. A panicking probe now leaves the cache
uninitialized instead of blocking other threads forever in `no_std` builds. Linux AArch64, POWER,
and IBM Z read their hardware capabilities with `getauxval` instead of opening `/proc/self/auxv`,
and Apple AArch64 reads its `sysctl` keys directly instead of the standard library's detector.
First-call detection on those targets performs no file I/O or heap allocation. The AArch64
data-independent-timing guard caches its own FEAT_DIT probe, which capability overrides and
`portable-only` builds do not disable.
Cached dispatch choices, including the CRC configurations, use the same lock-free cache. AES-GCM,
AES-GCM-SIV, ChaCha20-Poly1305, XChaCha20-Poly1305, and `ChaCha20Rng` select their GHASH and
POLYVAL multiply, Poly1305 block, and ChaCha20 keystream kernels from the cached capabilities at
each call, then call the kernel directly instead of through a cached function pointer.
