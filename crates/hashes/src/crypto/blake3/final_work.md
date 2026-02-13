# Blake3 Final Work (Refreshed Against Current Code + Baselines)

Last refreshed: 2026-02-13

This file replaces stale assumptions with findings verified against:
- `crates/hashes/src/crypto/blake3/*.rs`
- `crates/hashes/benches/blake3.rs`
- `crates/hashes/src/crypto/blake3/bench_baseline/*.txt`

## 1) What Changed Since Earlier Drafts

The previous version of this file had multiple stale items. Current status:

1. Dispatch fallback is no longer flat-only.
`dispatch.rs::resolve()` already performs tiered capability fallback, e.g. AVX-512 -> AVX2 -> SSE4.1 -> portable.

2. Kernel cross-check coverage is broader than previously documented.
`kernel_test.rs` now validates:
- all available kernels vs official crate
- streaming chunk splits
- keyed + derive modes
- xof prefix correctness
- hash_many_contiguous agreement across kernels
- large input correctness for std/parallel paths

3. “No per-kernel bench at all” is partially stale.
`benches/blake3.rs` includes parent-folding microbench by kernel name via `hashes::bench` hooks. We still do *not* have full per-kernel `compress/hash_many` Criterion isolation across all kernel IDs.

4. Known blocker text around “Graviton tiny oneshot +56-65% slower” is stale.
Current baseline gap on Graviton tiny oneshot is roughly ~4-5%, not 56-65%.

## 2) Current Code-Truth Gaps (Still Real)

### A. No external forced-kernel mode for field debugging
We have internal helpers (`digest_with_kernel_id`, `stream_chunks_with_kernel_pair_id`) but they are `pub(crate)` and primarily used by tests/bench internals. There is no public env override like CRC64’s `RSCRYPTO_CRC64_FORCE`.

### B. x86 SSSE3 kernel is still present but not used by tuned dispatch tables
`Blake3KernelId::X86Ssse3` remains in registry and test paths. It is mostly maintenance surface unless we intentionally keep it as a reference/debug tier.

### C. Baseline coverage still missing for several tuned profiles
Missing baseline files for tuned kinds such as Zen5c, Intel GNR, Apple M4/M5, Graviton2.

## 3) Completed In This Final Pass

### A. Arithmetic policy debt fixed in hot update path
File: `crates/hashes/src/crypto/blake3/mod.rs`
- Replaced block length update with `strict_add`.
- Replaced guarded block reservation decrement with `strict_sub` and explicit nonzero check.

### B. Tiny first-update fast path added (<=64B)
File: `crates/hashes/src/crypto/blake3/mod.rs`
- Added an ultra-tiny fast path in `Digest::update` for a fresh hasher state.
- This bypasses generic `update_with` dispatch/loop overhead for the most latency-sensitive keyed/derive/xof init cases.

### C. Derive-key context cache simplified for lower overhead
File: `crates/hashes/src/crypto/blake3/mod.rs`
- Removed global `Mutex<HashMap<...>>` derive context cache.
- Kept thread-local single-entry cache, which is the true hot path for repeated contexts.
- This reduces complexity and significantly shrinks `derive_context_key_words_cached` code size.

### D. Parallel cold-path isolation kept
File: `crates/hashes/src/crypto/blake3/mod.rs`
- `commit_parallel_batch` and `try_parallel_update_batch` marked `#[cold] #[inline(never)]`.
- `update_with` now gates parallel batch attempts behind cheap checks (`chunk_state.len()==0 && input.len()>CHUNK_LEN`).

## 4) Benchmark Reality (From Current Baseline Files)

Sources:
- `linux_arm_graviton3.txt`
- `linux_arm_graviton4.txt`
- `mac_arm_m1.txt`
- `linux_x86_intelicl.txt`
- `linux_x86_intelspr.txt`
- `linux_x86_zen4.txt`
- `linux_x86_zen5.txt`

### A. One-shot small-input gap is mainly an x86 Zen problem now

Approx rscrypto/official throughput ratio for oneshot 1B:
- Graviton3: 0.960
- Graviton4: 0.953
- M1: 0.992
- Intel ICL: 0.999
- Intel SPR: 0.994
- Zen4: 0.841
- Zen5: 0.857

Interpretation:
- ARM tiny-input overhead is no longer catastrophic.
- Zen4/Zen5 small-input latency remains a clear gap.

### B. Biggest remaining losses are keyed/derive and XOF init+read on small inputs

Worst recurring gaps across platforms:
- `blake3/keyed/*`: often 20-50% behind official on small/medium sizes.
- `blake3/derive-key/*`: often 14-42% behind on small/medium sizes.
- `blake3/xof/init+read/{1B,64B}-in/32B-out`: often 30-50% behind.

Examples from current baselines:
- Intel ICL keyed 31/32/63/64: ~0.48-0.50 ratio.
- Intel SPR xof init+read 1B-in/32B-out: ~0.50 ratio.
- Zen4/Zen5 xof init+read 64B-in/32B-out: ~0.71 ratio.
- Graviton3/4 keyed small: roughly 0.74-0.81 ratio.

### C. Large-input one-shot throughput is generally strong
For 1 MiB one-shot, rscrypto is generally competitive or significantly ahead in these baselines, so the largest competitive upside is not in large-bulk one-shot.

## 5) Final Priority Order (Actionable)

## P0: Highest remaining competitiveness work
1. Add a public forced-kernel override for production-like A/B debugging.
2. Fill missing baseline artifacts for tuned profiles (Zen5c/GNR/M4/M5/Graviton2).

## P1: Attack the actual performance losses
1. XOF init+read small-output path (32B focus):
- isolate init vs squeeze overhead (bench already supports read-only mode)
- collapse overhead in finalize_xof setup for tiny inputs

2. Zen4/Zen5 tiny oneshot latency:
- re-check size-class boundaries and stream/bulk kernel choice for <=128B
- verify whether current SSE4.1 choice at tiny sizes is still optimal on latest microcode/toolchain

## P2: Tooling and diagnosability
1. Add true per-kernel Criterion isolation benches for `compress`, `hash_many_contiguous`, and parent CV.
2. Keep using `cargo asm`, `cargo llvm-lines`, and `samply` for each P1 item (one bottleneck at a time).

## P3: Scope-expansion items (not blocking current competitiveness)
1. wasm SIMD kernel.
2. SVE2 path.
3. Additional architecture SIMD breadth (s390x/POWER/RISC-V).

## 6) Implementation Notes

- Keep changes minimal and measurable; avoid ornamental abstractions.
- Every performance change should be tied to one benchmark family and one target bottleneck.
- Validate with:
`just check-all && just test`
