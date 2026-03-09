# Hash Implementation Order

Blake3 is done. This is what comes next.

Same rules for every algorithm: pure Rust, no deps, no C libs, SIMD on every
arch that supports it (x86_64, aarch64, wasm32), portable scalar fallback
always present, runtime feature detection on std targets. Bounds of physics.

---

## Crypto

These are non-negotiable for any serious crypto crate. Ordered by real-world
deployment footprint — what the most systems actually depend on today.

| Priority | Algorithm | Why |
|----------|-----------|-----|
| 1 | **SHA-256** | Most deployed hash on earth. TLS, X.509, HMAC, HKDF, Bitcoin, FIPS 140-3, PCI-DSS. Nothing ships without it. |
| 2 | **SHA-512** | Required by TLS 1.3 cipher suites, Ed25519, HMAC-SHA-512. Faster than SHA-256 on 64-bit due to native 64-bit arithmetic. |
| 3 | **SHA-384** | Truncated SHA-512. Required by TLS_AES_256_GCM_SHA384 (the default TLS 1.3 suite). Trivial once SHA-512 exists. |
| 4 | **SHA-512/256** | SHA-512 internals, 256-bit output. Faster than SHA-256 on 64-bit, immune to length-extension. Underused but correct choice when you want 256 bits on modern hardware. |
| 5 | **SHA-224** | Truncated SHA-256. Rare but FIPS-listed. Trivial once SHA-256 exists. |
| 6 | **SHA3-256** | NIST's SHA-2 alternative. Keccak sponge. Required for post-quantum transition plans, some government specs. Different construction — if SHA-2 breaks, SHA-3 survives. |
| 7 | **SHA3-512** | Same Keccak core, 512-bit security. Trivial once SHA3-256 exists. |
| 8 | **SHA3-384 / SHA3-224** | Completeness. Same core, different capacity. |
| 9 | **SHAKE128 / SHAKE256** | Keccak XOF (extendable output). Variable-length output for KDFs, mask generation, domain separation. Same core as SHA-3. |

### SIMD notes

**SHA-2**: SHA-NI on x86_64 (available since Zen 1 / Ice Lake) gives ~3-4x over
scalar. ARMv8.0 has SHA256 instructions. Multi-buffer (2-way / 4-way) AVX2
for parallel independent messages. No wasm SHA intrinsics yet — scalar only.

**SHA-3 (Keccak)**: No dedicated hardware instructions on any mainstream arch.
SIMD wins come from vectorized Keccak-f[1600] — lane-parallel AVX2/NEON
implementations. Smaller gains than SHA-2 hardware acceleration but still
meaningful. Turbo-SHAKE / parallel tree modes can exploit wider vectors.

---

## Fast

Non-cryptographic. For hash tables, checksums, fingerprinting, deduplication.
Ordered by adoption and the gap left if we don't have them.

| Priority | Algorithm | Why |
|----------|-----------|-----|
| 1 | **XXHash3-64** | Current speed king for general-purpose hashing. Fastest non-crypto hash with excellent distribution. Used by Linux kernel, lz4, Zstandard. 64-bit output. |
| 2 | **XXHash3-128** | Same core as XXH3-64 with 128-bit output. Better collision resistance for fingerprinting and dedup. Trivial once XXH3-64 exists. |
| 3 | **XXHash64** | Predecessor. Still everywhere — file checksums, data formats, legacy compatibility. Simpler than XXH3, good baseline. |
| 4 | **wyhash** | Extremely fast on short keys. Excellent for hash maps. Simple construction (multiply-xor). Go's `maphash` is wyhash-derived. |

### SIMD notes

**XXHash3**: Designed for SIMD from the ground up — accumulator is 8×64-bit
stripes, maps directly to AVX2 (256-bit) or pairs of NEON 128-bit ops. SSE2
fallback for older x86. Scalar path uses 64-bit multiply-accumulate. The
algorithm *is* the SIMD — portable scalar is the slow path by design.

**wyhash**: Minimal SIMD opportunity — the algorithm is inherently serial
(multiply-xor chain). Speed comes from the small number of operations, not
parallelism. Focus on inlining and avoiding unnecessary work.

---

## Not planned

| Algorithm | Reason |
|-----------|--------|
| MD5 | Broken. No legitimate new use. |
| SHA-1 | Broken (SHAttered, 2017). Chrome, Firefox, Git all moved off. |
| SipHash | Rust std already provides it. No reason to reimplement. |
| FNV | Too weak distribution for modern use. SipHash or wyhash instead. |
| CityHash / FarmHash | Google-internal, superseded by XXHash3 / wyhash. |
| Highwayhash | Interesting but low adoption. |
| KangarooTwelve | Keccak variant — consider after SHA-3 if there's demand. |

---

## Implementation order summary

```
Done:   Blake3
Next:   SHA-256 → SHA-512 → SHA-384 → SHA-512/256 → SHA-224
Then:   XXHash3-64 → XXHash3-128 → XXHash64
Then:   SHA3-256 → SHA3-512 → SHA3-384 → SHA3-224 → SHAKE128 → SHAKE256
Last:   wyhash
```

SHA-2 first because it unblocks HMAC, HKDF, and every AEAD that needs a hash.
XXHash3 next because it's the fast-hash flagship and the SIMD story is the
strongest. SHA-3 after because it's important but less urgent — SHA-2 covers
most real-world needs today. wyhash last because it's small and simple, a
palate cleanser.
