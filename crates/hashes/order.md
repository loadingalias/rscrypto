# Hash Implementation Order

Same rules for every algorithm: pure Rust, no deps, no C libs, SIMD on every
arch that supports it (x86_64, aarch64, wasm32), portable scalar fallback
always present, runtime feature detection on std targets. Bounds of physics.

---

## Status

### Done

| Algorithm | Category | Notes |
|-----------|----------|-------|
| **Blake3** | Crypto | Flagship. Full SIMD dispatch (SSE4.1, AVX2, AVX-512, NEON, ASM). XOF, keyed, derive modes. |
| **SHA-256** | Crypto | SHA-NI, ARMv8 SHA2, portable scalar. |
| **SHA-512** | Crypto | Portable scalar. Shares core with SHA-384, SHA-512/256. |
| **SHA-384** | Crypto | Truncated SHA-512. |
| **SHA-512/256** | Crypto | Truncated SHA-512. |
| **SHA-224** | Crypto | Truncated SHA-256. |
| **SHA3-256** | Crypto | Keccak sponge. |
| **SHA3-512** | Crypto | Keccak sponge. |
| **SHA3-384** | Crypto | Keccak sponge. |
| **SHA3-224** | Crypto | Keccak sponge. |
| **SHAKE128** | Crypto | Keccak XOF. |
| **SHAKE256** | Crypto | Keccak XOF. |
| **Ascon-Hash256** | Crypto | Lightweight. NIST LWC winner. |
| **Ascon-XOF128** | Crypto | Lightweight XOF. |
| **XXHash3-64** | Fast | SIMD-native accumulator design. |
| **XXHash3-128** | Fast | 128-bit variant of XXH3. |
| **RapidHash** | Fast | 64-bit and 128-bit. Fast short-key hashing. |

### Remaining

| Priority | Algorithm | Category | Why |
|----------|-----------|----------|-----|
| 1 | **XXHash64** | Fast | Predecessor to XXH3. Still everywhere — file checksums, data formats, legacy compatibility. Simpler than XXH3. |
| 2 | **wyhash** | Fast | Extremely fast on short keys. Excellent for hash maps. Simple construction (multiply-xor). Go's `maphash` is wyhash-derived. |

---

## SIMD notes

**SHA-2**: SHA-NI on x86_64 (available since Zen 1 / Ice Lake) gives ~3-4x over
scalar. ARMv8.0 has SHA256 instructions. Multi-buffer (2-way / 4-way) AVX2
for parallel independent messages. No wasm SHA intrinsics yet — scalar only.

**SHA-3 (Keccak)**: No dedicated hardware instructions on any mainstream arch.
SIMD wins come from vectorized Keccak-f[1600] — lane-parallel AVX2/NEON
implementations. Smaller gains than SHA-2 hardware acceleration but still
meaningful. Turbo-SHAKE / parallel tree modes can exploit wider vectors.

**XXHash3**: Designed for SIMD from the ground up — accumulator is 8×64-bit
stripes, maps directly to AVX2 (256-bit) or pairs of NEON 128-bit ops. SSE2
fallback for older x86. Scalar path uses 64-bit multiply-accumulate. The
algorithm *is* the SIMD — portable scalar is the slow path by design.

**wyhash**: Minimal SIMD opportunity — the algorithm is inherently serial
(multiply-xor chain). Speed comes from the small number of operations, not
parallelism. Focus on inlining and avoiding unnecessary work.

---

## Removed

| Algorithm | Reason |
|-----------|--------|
| Blake2b | Superseded by Blake3. Niche adoption. |
| Blake2s | 32-bit Blake2 variant. Even less used than Blake2b. |
| SHA-512/224 | Extremely rare. Almost nothing uses it. |
| SipHash | Rust std already provides it. No reason to reimplement. |

## Not planned

| Algorithm | Reason |
|-----------|--------|
| MD5 | Broken. No legitimate new use. |
| SHA-1 | Broken (SHAttered, 2017). Chrome, Firefox, Git all moved off. |
| FNV | Too weak distribution for modern use. wyhash instead. |
| CityHash / FarmHash | Google-internal, superseded by XXHash3 / wyhash. |
| Highwayhash | Interesting but low adoption. |
| KangarooTwelve | Keccak variant — consider after SHA-3 if there's demand. |

---

## Implementation order summary

```
Done:   Blake3, SHA-256, SHA-512, SHA-384, SHA-512/256, SHA-224
        SHA3-256, SHA3-512, SHA3-384, SHA3-224, SHAKE128, SHAKE256
        Ascon-Hash256, Ascon-XOF128
        XXHash3-64, XXHash3-128, RapidHash
Next:   XXHash64 → wyhash
```

XXHash64 next for legacy compatibility coverage. wyhash last because it's
small and simple.
