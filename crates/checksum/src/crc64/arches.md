# CRC-64 Architecture Optimization Guide

This document defines the fastest possible CRC-64 (XZ and NVME) implementations
for each target architecture, ordered by market importance and deployment priority.

Last updated: 2025-12-22

---

## Status Summary

| Architecture | Status | Peak Throughput | Priority |
|--------------|--------|-----------------|----------|
| x86-64 | ✅ Complete | ~52 GB/s (VPCLMUL 7-way) | — |
| aarch64 | ✅ Complete | ~18 GB/s (EOR3 3-way) | — |
| powerpc64 | ✅ Complete | ~52 GB/s (VPMSUM 8-way) | — |
| s390x | ✅ Complete | ~30 GB/s (VGFM 4-way) | — |
| riscv64 | ✅ Implemented | Variable (ZVBC vector, Zbc scalar) | P2 |
| wasm32 | ❌ N/A | ~2 GB/s (table) | — |

---

## Tier 0: x86-64 (Intel/AMD) — COMPLETE

### Implemented Kernels
- **PCLMULQDQ**: 1/2/4/7-way folding (SSE4.2 + CLMUL)
- **VPCLMULQDQ**: 1/2/4/7-way folding (AVX-512F/VL/BW + VPCLMUL)

### Throughput (measured)
| Tier | Streams | Throughput | Notes |
|------|---------|------------|-------|
| PCLMUL | 7-way | ~26 GB/s | AMD Zen4 optimal |
| VPCLMUL | 7-way | ~52 GB/s | Sapphire Rapids |
| VPCLMUL | 4-way | ~45 GB/s | Most Intel AVX-512 |

### Theoretical Maximum
- VPCLMUL: 1 instruction/cycle × 64 bytes/instruction = **64 bytes/cycle**
- At 4 GHz: **256 GB/s** (memory-bound before this)

### References
- [Intel Fast CRC Using PCLMULQDQ](https://www.intel.com/content/dam/www/public/us/en/documents/white-papers/fast-crc-computation-generic-polynomials-pclmulqdq-paper.pdf)
- [komrad36/CRC](https://github.com/komrad36/CRC) - Fastest x86 CRC32 reference

---

## Tier 0: aarch64 (ARM) — COMPLETE

### Implemented Kernels
- **PMULL**: 1/2/3-way folding (NEON + Crypto)
- **PMULL+EOR3**: 1/2/3-way folding (NEON + SHA3)
- **SVE2-PMULL**: 1/2/3-way (for Neoverse V2/V3, Graviton4+)

### Throughput (measured, Apple M1/M2/M3)
| Tier | Streams | Throughput | Notes |
|------|---------|------------|-------|
| PMULL | 3-way | ~15 GB/s | Baseline |
| EOR3 | 3-way | ~18 GB/s | SHA3 required |
| SVE2 | 3-way | ~20 GB/s | Graviton4/Neoverse V2 |

### ARMv9 Opportunities (FUTURE)

#### SVE2 PMULL128 (FEAT_SVE_PMULL128)
- **Instructions**: `pmullb`, `pmullt` (polynomial multiply bottom/top)
- **Feature flag**: `__ARM_FEATURE_SVE2_AES`
- **Benefit**: Scalable vector width (128-2048 bits)
- **Status**: Rust intrinsics stabilizing; current SVE2 cores are 128-bit only

#### SME2 (Apple M4+, Neoverse V3)
- **Not applicable for CRC**: SME2 targets matrix operations (AI/ML)
- Apple M4 has SME2 but **no SVE** (only streaming SVE subset)
- PMULL via NEON remains optimal path on Apple Silicon

#### Recommendation
Current EOR3 3-way is optimal for Apple M1-M5. For Graviton4/5 and Neoverse V2/V3,
consider SVE2-PMULL with wider multi-way folding when VL > 128 bits becomes common.

### References
- [ARM PMULL Documentation](https://developer.arm.com/documentation/ddi0596/2021-12/SIMD-FP-Instructions/PMULL--PMULL2--Polynomial-Multiply-Long-)
- [ARM SVE2 Crypto Extensions](https://arm-software.github.io/acle/main/acle.html)
- [Apple M4 SME2 Exploration](https://github.com/tzakharko/m4-sme-exploration)

---

## Tier 1: powerpc64 (IBM POWER) — COMPLETE

### Hardware Capability
- **VPMSUM** (Vector Polynomial Multiply Sum): POWER8+ (2014)
- **Intrinsics**: `vec_pmsum_be()` in `altivec.h`
- **HWCAP2**: `PPC_FEATURE2_VEC_CRYPTO`

### Implemented Kernels
- **VPMSUMD**: 1/2/4/8-way folding (128B blocks)
- Small-buffer tier (16B lanes) + large-buffer tier (8×16B blocks)

Big-endian `powerpc64` is supported by normalizing loads to match the
little-endian lane interpretation used by the folding algorithm.

### Throughput (expected)

```
Algorithm: 8-way VPMSUM folding (128B blocks)
Latency: 6 cycles per VPMSUM
Throughput: 1 VPMSUM/cycle
Theoretical: 16 bytes/cycle = 64 GB/s @ 4 GHz
Measured: ~52 GB/s (13.6 bytes/cycle) on POWER8 4.1 GHz
```

### Implementation Notes
- Code: `crates/checksum/src/crc64/powerpc64.rs`
- Dispatch: `crates/checksum/src/crc64/mod.rs`

### References
- [antonblanchard/crc32-vpmsum](https://github.com/antonblanchard/crc32-vpmsum) - Reference CRC32 implementation
- [Linux Kernel POWER CRC](https://www.mail-archive.com/linuxppc-dev@lists.ozlabs.org/msg242286.html)

---

## Tier 2: s390x (IBM Z) — COMPLETE

### Hardware Capability
- **VGFM** (Vector Galois Field Multiply): z13+ (2015)
- **VGFMA** (Vector Galois Field Multiply and Add): z13+
- **Intrinsics**: `vec_gfmsum_128()`, `vec_gfmsum_accum_128()`

### Implemented Kernels

```
Algorithm: VGFM-based 128-bit folding
Vector width: 128-bit fixed
Element sizes: byte, halfword, word, doubleword
Theoretical: ~30 GB/s on z15/z16
```

Kernels:
- `vgfm`: 1/2/4-way folding (128B blocks)

Implementation notes:
- Code: `crates/checksum/src/crc64/s390x.rs` (inline asm `vgfm …, 3`)
- Dispatch/caps: `platform::caps::s390x::VECTOR`

### References
- [linux-on-ibm-z/crc32-s390x](https://github.com/linux-on-ibm-z/crc32-s390x) - Reference CRC32 implementation
- [IBM VGFM Documentation](https://www.ibm.com/docs/en/zos/2.5.0?topic=arithmetic-vec-gfmsum-accum-vector-galois-field-multiply-sum-accumulate)
- [MongoDB s390x CRC](https://fossies.org/linux/mongo/src/third_party/wiredtiger/src/checksum/zseries/README.md)

---

## Tier 3: riscv64 (RISC-V) — IMPLEMENTED (Zbc + ZVBC; needs HW validation)

### Hardware Capability
- **Zbc** (Scalar carryless multiply): `clmul`, `clmulh`, `clmulr`
- **Zvbc** (Vector carryless multiply): `vclmul.vv`, `vclmul.vx`, `vclmulh.vv`, `vclmulh.vx`
- **Vector width**: Implementation-defined (128-65536 bits)

### Implemented
- Scalar Zbc folding (1/2/4-way) using `clmul`/`clmulh`
- Vector Zvbc folding (1/2/4-way) using RVV `vclmul{,h}.vx` with VL-agnostic chunking
- Code: `crates/checksum/src/crc64/riscv64.rs`
- Dispatch/caps: `platform::caps::riscv::{ZBC,ZVBC}`

### Implementation Notes
- ZVBC requires nightly Rust target-feature gating (`riscv_target_feature`).
- The hot loop folds 8×16B lanes per 128B block using RVV with `e64,m1` and `vsetvli` to adapt to any VLEN.
- Tail reduction uses a VL=1 ZVBC microkernel to keep the hot path simple.
- Benchmarking on real ZVBC hardware is still outstanding; tune defaults are heuristic.

### Kernel Structure
```
crc64_{xz,nvme}_zbc*      - Scalar clmul baseline + ILP variants (1/2/4-way)
crc64_{xz,nvme}_zvbc*     - Vector clmul baseline + ILP variants (1/2/4-way)
fold_block_128_zvbc()     - VL-agnostic RVV folding core
```

### Zvbc32e (DRAFT Extension)
- Extends ZVBC to 32-bit element width (ELEN=32 cores)
- Enables 4× 32-bit clmul units vs 2× 64-bit
- Useful for embedded RISC-V with shorter vectors

### References
- [Accelerating CRC with RISC-V Vector](https://fprox.substack.com/p/accelerating-crc-with-risc-v-vector)
- [riscv/riscv-crypto ZVBC spec](https://github.com/riscv/riscv-crypto/blob/main/doc/vector/riscv-crypto-vector-zvbc.adoc)
- [GCC RISC-V vclmul patches](https://www.mail-archive.com/gcc-patches@gcc.gnu.org/msg371024.html)
- [Hadoop RISC-V CRC32](https://www.mail-archive.com/common-issues@hadoop.apache.org/msg321911.html)

---

## Tier 5: wasm32 (WebAssembly) — NOT APPLICABLE

### Hardware Capability
- **SIMD128**: 128-bit fixed-width SIMD
- **Relaxed SIMD**: Platform-specific optimizations
- **No CLMUL**: No polynomial multiply instruction

### Current Limitation
WebAssembly SIMD128 explicitly **does not support**:
- CRC32 hardware instructions
- Carryless/polynomial multiplication

From [Emscripten docs](https://emscripten.org/docs/porting/simd.html):
> The CRC32 intrinsics `_mm_crc32_u16`, `_mm_crc32_u32`, `_mm_crc32_u64`, and
> `_mm_crc32_u8` are not supported - any code referencing these intrinsics
> will not compile.

### Best Available
Slice-by-8 table-based: ~2 GB/s (same as portable)

### Future: WASM Crypto Extensions?
No current proposal for polynomial multiply in WebAssembly.

### Recommendation
**Do not implement**. No performance gain possible over portable.

---

## Emerging Research

### Chorba Algorithm (December 2024)
A novel CRC computation method without lookup tables or hardware CLMUL:
- Claims 100% throughput improvement over state-of-art table methods
- Potentially matches hardware-accelerated on some platforms
- Paper: [arxiv.org/html/2412.16398v1](https://arxiv.org/html/2412.16398v1)

**Evaluation needed**: Could benefit wasm32/wasm64 and other targets where CLMUL is absent.

### Multi-Stream Folding Limits
| Architecture | Max Practical Streams | Reason |
|--------------|----------------------|--------|
| x86-64 VPCLMUL | 7-way | Register pressure (ZMM0-ZMM15) |
| x86-64 PCLMUL | 7-way | XMM register pressure |
| aarch64 PMULL | 3-way | 32 NEON regs, but memory BW limited |
| POWER VPMSUM | 8-way | 32 VSX regs, 6-cycle latency |
| s390x VGFM | 4-way | 32 VRs, memory BW limited |
| RISC-V ZVBC | LMUL-dependent | Variable, VL-agnostic design needed |

---

## Implementation Priority

### P0: powerpc64le (VPMSUM) — DONE
- **Impact**: IBM POWER servers (cloud, enterprise, HPC)
- **Speedup**: 40× over portable
 - **Note**: Both endiannesses are supported.

### P1: s390x (VGFM) — DONE
- **Impact**: IBM Z mainframes (banking, enterprise)
- **Speedup**: 15-20× over portable

### P2: riscv64 (ZVBC) — IN PROGRESS
- **Impact**: Emerging (SiFive, Alibaba T-Head, future Apple?)
- **Speedup**: 10-30× over portable (depends on VLEN)
- **Current**: scalar Zbc folding implemented; ZVBC still pending.

### Skip: wasm32
- No performance benefit possible

---

## Appendix: Capability Detection

### Current Caps (platform crate)
```rust
// x86-64
x86::PCLMUL_READY    // SSE4.2 + PCLMULQDQ
x86::VPCLMUL_READY   // AVX-512F/VL/BW + VPCLMULQDQ

// aarch64
aarch64::PMULL_READY      // AES (includes PMULL)
aarch64::PMULL_EOR3_READY // AES + SHA3
aarch64::SVE2_PMULL       // SVE2 + crypto

// powerpc64le
powerpc64::VPMSUM_READY   // VEC_CRYPTO

// s390x
s390x::VECTOR             // Vector facility (z13+)

// riscv64
riscv::ZBC                // Scalar carryless multiply
riscv::ZVBC               // Vector carryless multiply (future CRC64)
```

---

## Appendix: Benchmark Methodology

All throughput numbers should be measured with:
1. **Large buffers** (≥1 MB) to amortize setup costs
2. **Pre-warmed caches** (run 10+ iterations before timing)
3. **Single-threaded** (measure instruction throughput, not memory BW)
4. **Multiple buffer sizes** to identify crossover points

Recommended test sizes: 64, 128, 256, 512, 1K, 4K, 16K, 64K, 256K, 1M, 4M bytes

---

## References

### Papers
- [Intel Fast CRC Computation](https://www.intel.com/content/dam/www/public/us/en/documents/white-papers/fast-crc-computation-generic-polynomials-pclmulqdq-paper.pdf)
- [Chorba: A novel CRC32 implementation](https://arxiv.org/html/2412.16398v1)
- [Accelerating CRC with RISC-V Vector](https://fprox.substack.com/p/accelerating-crc-with-risc-v-vector)

### Implementations
- [antonblanchard/crc32-vpmsum](https://github.com/antonblanchard/crc32-vpmsum) (POWER)
- [linux-on-ibm-z/crc32-s390x](https://github.com/linux-on-ibm-z/crc32-s390x) (Z)
- [riscv/riscv-crypto](https://github.com/riscv/riscv-crypto) (RISC-V spec)
- [komrad36/CRC](https://github.com/komrad36/CRC) (x86 reference)

### Architecture Manuals
- [ARM ACLE](https://arm-software.github.io/acle/main/acle.html)
- [IBM z/Architecture Reference Summary](https://www.ibm.com/support/pages/sites/default/files/2021-05/SA22-7871-10.pdf)
