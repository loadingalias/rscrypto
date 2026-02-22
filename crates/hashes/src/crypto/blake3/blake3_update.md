# BLAKE3 Performance Plan (Locked State + Next Move)

## Objective
Beat upstream BLAKE3 on enforced `kernel-ab` and `oneshot` gates with simpler code, no API complexity, and no policy noise.

## Current Locked State
- Keep commit `1434fa2` (`hashes: split blake3 oneshot fallback into cold helper`).
- Reject commit `d21f59a` (`#[inline(never)]` on `root_output_oneshot`) because it regressed too many lanes.
- Validation on kept state:
  - `just check-all`: pass
  - `just test`: pass
  - CI `commit.yaml` run `22280869114`: success
  - CI `bench.yaml` run `22280874270` (`filter=kernel-ab`, `quick=false`, kernel gate enforced): fail

Why we keep `1434fa2` anyway:
- It is cleaner than the rejected variant and did not add policy complexity.
- It recovered part of the damage from `d21f59a` (notably restored `amd-zen4` gate pass).
- It is a better stable base for kernel work than continuing boundary-policy churn.

## What Is Proven and Frozen
1. Main deficit is short-input kernel performance (`256`, `1024`), not large-input throughput.
2. Boundary-policy alphabet search has diminishing returns and unstable cross-lane behavior.
3. Global x86 boundary retunes are frozen until kernel quality improves.
4. No new runtime special-case policy without repeatable cross-lane wins.
5. Kernel-first rule: kernel-only evidence before API-path/policy changes.

## Latest Gate Snapshot (Run `22280874270`)
- Passed lanes:
  - `amd-zen4`
  - `ibm-power10`
- Failed lanes (kernel-ab):
  - `intel-spr`: `256 +18.60%`, `1024 +16.23%`, `4096 +6.94%` (limit +6%)
  - `intel-icl`: `256 +18.96%`, `1024 +17.11%`, `4096 +8.02%`
  - `amd-zen5`: `256 +15.01%`, `1024 +9.38%`
  - `graviton3`: `256 +30.77%`, `1024 +13.51%`
  - `graviton4`: `256 +33.16%`, `1024 +13.90%`
  - `ibm-s390x`: `256 +12.25%` (limit +12%, near miss)

Interpretation:
- Short-size kernel gap is still the blocker.
- ARM short-size gap is now the largest and most consistent loss across enforced ARM lanes.

## Next Logical Improvement (Immediate)
Focus first on **aarch64 NEON short-size kernel quality** before any more policy edits.

Why this is next:
- Biggest absolute gap is on `graviton3/graviton4` at `256/1024`.
- Same loss pattern on both ARM lanes means high confidence, not runner noise.
- We can iterate locally on macOS M1 (aarch64) with the same ISA family.

Scope:
1. Audit hot NEON symbols used by `kernel-ab` `256/1024`.
2. Remove avoidable front-end waste: extra shuffles, spills, and setup overhead.
3. Keep large-input path behavior unchanged.
4. Avoid dispatch or API changes in this phase.

First candidate to try:
- In `digest_one_chunk_root_hash_words_aarch64` (`crates/hashes/src/crypto/blake3/mod.rs`), remove the generic
  `kernels::chunk_compress_blocks_inline(kernel.id, ...)` call and invoke the aarch64 NEON path directly.
- Rationale: this helper is already aarch64-only and guarded to `Aarch64Neon`; keeping an extra kernel-id dispatch
  in the hot short-input path is likely pure overhead.

## Tooling (Approved Baseline, macOS M1 Safe)
- `cargo bench`: canonical perf harness used by CI.
- `cargo-asm`: inspect emitted assembly for hot symbols.
- `cargo-show-asm`: alternate asm/LLVM/MIR views.
- `cargo-llvm-lines`: detect code-size and inlining bloat.
- `cargo-samply`: local sampling profiler for short paths.

Optional only with explicit need:
- `perf`/PMU tools on Linux runners for cycle/uop confirmation.
- `llvm-mca` when instruction-level throughput modeling is required.

## Candidate Protocol (Non-Negotiable)
1. One focused kernel change per commit.
2. Run `just check-all && just test`.
3. Run CI kernel-only (`kernel-ab`, `quick=false`, enforced lanes).
4. If kernel result is promising, run API-path benches.
5. Keep/revert decision based on repeatable cross-lane net effect.

## Definition of Done
- `blake3/kernel-ab` and `blake3/oneshot` pass on enforced lanes.
- Short-size wins are repeatable.
- Complexity is equal or lower than current baseline.
