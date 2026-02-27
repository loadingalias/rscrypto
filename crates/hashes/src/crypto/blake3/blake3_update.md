# BLAKE3 Performance Plan (Locked Loop)

## Mission
Beat upstream on enforced `kernel-ab` and `oneshot` lanes with simpler or equal complexity, no API churn, and repeatable results. If we see a win that makes our code more complex, it's fine... but it must be a real win over the 'official Blake3' crate/implementation/benches.

## Locked Baseline
- Baseline is the last accepted pre-N/O line (Candidate I family).
- Candidate N and Candidate O were rejected and rolled back.
- Baseline validation already passed: `just check-all`, `just test`.

## What Is Settled
- Biggest deficit is short sizes (`256`, `1024`), not large-size throughput.
- Boundary-policy churn has low ROI without kernel root-cause data. We've addressed it extensively before settling on this baseline.
- Some control-path cleanups helped, but did not close the short-size gap.

## Frozen Policy Rules
1. Keep current dispatch boundaries from the accepted baseline.
2. No global x86 boundary retune (no blanket `avx2 -> sse4.1` shifts).
3. No per-lane runtime exceptions unless they are repeatably net-positive across reruns.
4. Keep large-input behavior stable while improving short-input compute quality.
5. Any policy change requires kernel-only evidence first, API-path evidence second.
6. We're allowed to update the dispatch_tables.rs for Blake3 but this can't be our only path. Once we're at the point where we're pretty sure our dispatch is clean and ideal... we should stop and focus on more impactful code updates.

## Locked Optimization Loop
This loop is mandatory for every candidate and is now the default process.

1. Measure kernel gap first (`kernel-ab`, same runner class as upstream comparison).
2. Attribute hot paths (`cargo-samply`) and inspect generated code (`cargo-asm`, `cargo-show-asm`).
3. Quantify size/inlining pressure (`cargo-llvm-lines`).
4. Make one minimal kernel/codegen change.
5. Validate in order:
   - `just check-all && just test`
   - CI kernel-only benches
   - CI API-path benches (`oneshot`, short-input attribution)
6. Decide immediately: keep or revert.

If a candidate is not a clear cross-lane win, revert. No alphabet exploration without hard evidence.

## Tooling Contract (macOS M1 + CI)
Primary tools:
- `cargo bench` (Criterion benches already wired to repo/CI)
- `cargo-asm`
- `cargo-show-asm`
- `cargo-llvm-lines`
- `cargo-samply`

Why this set:
- Covers measurement, assembly inspection, code-size/inlining analysis, and hotspot attribution end-to-end.
- Works on macOS aarch64 for fast local iteration.
- Maps cleanly to CI validation on Linux/Windows/IBM runners.

Optional tools only with a specific question they uniquely answer:
- Linux `perf`/PMU tools for cycle-level confirmation on CI x86 hosts.
- `llvm-mca` or VTune for unresolved microarchitectural questions.

## Host vs CI Roles
- macOS M1 host: correctness, quick bench trends, asm/codegen/profiling loop.
- CI runners: source of truth for pass/fail and cross-arch competitiveness.

## Definition of Done
- Enforced `blake3/kernel-ab` and `blake3/oneshot` lanes are green with repeatable wins.
- Short-size gap materially closes at `256` and `1024` without meaningful `4096+` regressions.
- No new public API complexity and no ornamental hot-path abstractions.

## Commit Discipline
- One candidate per commit.
- Subject format: `hashes: <short action summary>`.
- Every candidate carries CI run IDs and explicit keep/reject outcome in notes.

## Progress
### 2026-02-22 - Candidate P (`3a57016`)
- Change:
  - Enabled lane-native vector bulk kernels in BLAKE3 dispatch tables for IBM Z (`s390x`) and POWER (`powerpc64`) families.
- Validation:
  - Local: `just check-all && just test` passed (`166/166` tests).
  - CI Bench run: `22284378402` (targeted lanes only: `ibm-s390x`, `ibm-power10`).
- CI outcomes:
  - `ibm-power10`
    - `blake3/oneshot` gate failed at `256`: need `+9.68%` vs `+4.80%` limit.
    - `blake3/kernel-ab` gate passed for `powerpc64/vsx`.
  - `ibm-s390x`
    - `blake3/oneshot` gate failed at `1024`: need `+10.08%` vs `+6.80%` limit.
    - `blake3/kernel-ab` gate failed at `256`: need `+13.40%` vs `+12.00%` limit.
- Decision:
  - Keep as partial progress (POWER kernel-ab is now green and medium/large are ahead), but not a win.
  - Next candidate must target shared short-input oneshot overhead across architectures (not IBM-only).

### 2026-02-22 - Candidate Q (planned)
- Hypothesis:
  - The shared one-chunk generic path pays avoidable overhead on block-aligned inputs (`64/128/256/512/1024`) by always materializing/copying the final 64-byte block.
  - This overhead is hit on non-x86/aarch64 fast-path kernels and on portable short-input lanes; reducing it should help `oneshot` and `kernel-ab` short sizes, especially `256`/`1024`.
- Planned change:
  - In `digest_one_chunk_root_hash_words_generic`:
    - switch pre-tail compression to `kernels::chunk_compress_blocks_inline(kernel.id, ...)` (avoid function-pointer indirection on this hot path),
    - use zero-copy final-block handling when `last_len == 64` (read directly from input instead of staging into a stack buffer),
    - keep partial-tail behavior unchanged (still pad and compress with the existing root flags contract).
- Validation plan:
  - `just check-all && just test`
  - CI benches (granular first):
    - `blake3/oneshot` gate
    - `blake3/kernel-ab` gate
  - Keep only if cross-lane trend is positive; otherwise revert immediately.

### 2026-02-23 - Candidate S (`9f4459a`)
- Hypothesis:
  - Short-input losses on `intel-spr` (`256/1024`) and `graviton4` kernel-ab are dominated by per-call asm/trampoline overhead in chunk block compression, not by large-input SIMD throughput.
  - For one-chunk-or-less work (<=16 blocks), direct inlined CV compression loops should reduce fixed overhead and improve short-size competitiveness.
- Change:
  - `x86_64` AVX-512: in `chunk_compress_blocks_avx512`, use `compress_cv_avx512_bytes` for short block batches (`<=16` blocks), keep asm path for larger batches.
  - `aarch64` NEON: in `chunk_compress_blocks_neon`, skip chunk-loop asm path for short block batches (`<=16` blocks), keep asm path for larger aligned batches.
- Validation:
  - Local: `just check-all && just test` passed.
  - CI bench run: `22292690228` (targeted lanes only: `intel-spr`, `graviton4`; `blake3/oneshot` + kernel gate diagnostics).
- CI outcomes:
  - `intel-spr`
    - `blake3/oneshot`: `256` improved (`+23.95%` vs prior `+28.38%`), `1024` regressed (`+33.44%` vs prior `+32.28%`).
    - `blake3/kernel-ab`: `256` improved (`+12.87%` vs prior `+16.23%`), `1024` regressed (`+16.29%` vs prior `+15.21%`).
  - `graviton4`
    - `blake3/oneshot`: roughly flat (`256 +1.55%`, `1024 +2.60%`).
    - `blake3/kernel-ab`: severe regression at `256` (`+88.00%` vs prior `+33.44%`), `1024` effectively unchanged (`+13.91%`).
- Decision:
  - Reject and revert.
  - Net result is not a cross-lane win, and the Graviton kernel-ab `256` regression is unacceptable.

### 2026-02-23 - Candidate T (`77997e4`)
- Hypothesis:
  - Candidate S improved Intel `256` but hurt Intel `1024`, and broke Graviton due to a broad short-path switch.
  - Restricting the AVX-512 non-asm short path to only very small batches (`<=4` blocks, i.e. 256B) should preserve the Intel `256` gain while avoiding the `1024` regression and eliminating cross-arch risk.
- Change:
  - `x86_64` only: in `chunk_compress_blocks_avx512`, use `compress_cv_avx512_bytes` only when `num_blocks <= 4`.
  - Keep asm path for `num_blocks >= 5` (including 1024B = 16 blocks).
  - No `aarch64`/IBM code changes.
- Validation:
  - Local: `just check-all && just test` passed.
  - CI bench run: `22311267431` (targeted lane only: `intel-spr`; `blake3/oneshot` + kernel gate diagnostics).
- CI outcomes:
  - `intel-spr`
    - `blake3/oneshot`: regressed at short sizes (`256 +29.75%`, `1024 +34.85%`).
    - `blake3/kernel-ab`: `256` slightly improved (`+12.10%` vs prior `+12.87%`), but `1024` regressed (`+16.59%` vs prior `+16.29%`) and `4096` regressed past gate (`+7.14%` vs `+6.00%` limit).
- Decision:
  - Reject and revert.
  - Not a net win on Intel, and it introduces a new kernel gate failure at `4096`.

### 2026-02-24 - Candidate U (`f2c61ad`..`main`)
- Hypothesis:
  - Current Intel SPR short-input losses are partly self-inflicted by oneshot size-class routing (`65..1024 -> AVX2`) and by per-block AVX-512 one-chunk compression overhead.
  - Reusing the existing AVX-512 asm `hash_many` backend for exact-block one-chunk inputs (`64/128/256/512/1024`) should reduce fixed overhead for kernel-ab short sizes.
  - Switching Intel SPR `dispatch.s` from `x86_64/avx2` to `x86_64/avx512` should convert that kernel win into a direct oneshot win at `256`/`1024`.
- Change:
  - `x86_64` one-chunk path:
    - in `digest_one_chunk_root_hash_words_x86`, add an AVX-512 exact-block fast path that calls `asm::hash_many_avx512` with `num_inputs=1`, `blocks=len/64`, and `flags_end` including `ROOT`.
    - keep existing logic for partial-tail inputs and non-AVX-512 kernels.
  - `dispatch_tables`:
    - Intel SPR profile only: set oneshot size-class `s` kernel to `KernelId::X86Avx512` (was `KernelId::X86Avx2`).
- Validation:
  - Local: `just check-all && just test` passed.
  - CI targeted bench lane: `22331493553` (`intel-spr`, `blake3/oneshot` + kernel diagnostics/gates).
- CI outcomes:
  - `intel-spr`
    - `blake3/oneshot`: gate passed (`256 -3.41%`, `1024 -2.95%`, `4096 +6.35%`).
    - `blake3/kernel-ab`: gate passed (`256 -3.94%`, `1024 -2.85%`, `4096 +5.93%`).
- Decision:
  - Keep.
  - This is the first clear short-size win on Intel SPR in the current loop.

### 2026-02-24 - Candidate V (`b7c7984`)
- Hypothesis:
  - Candidate U's AVX-512 one-chunk fast path is SPR-specific in practice because Zen4/Zen5/ICL still route `65..1024` through AVX2.
  - Adding the same exact-block one-chunk fast path for AVX2 (`hash_many_avx2` with `num_inputs=1`) should reduce fixed overhead at `256/1024` for non-SPR x86 lanes without touching ARM64 behavior.
- Change:
  - `x86_64` one-chunk path:
    - in `digest_one_chunk_root_hash_words_x86`, add AVX2 exact-block fast path (`64..1024`, block-aligned) using asm `hash_many_avx2`.
    - keep existing partial-tail and non-AVX2 behavior unchanged.
- Validation:
  - Local: `just check-all && just test` passed.
  - CI plan (targeted): `intel-icl`, `amd-zen4`, `amd-zen5` oneshot gap-gate lanes.
- CI outcomes:
  - CI run: `22334612615`
  - `intel-icl`
    - `blake3/oneshot`: gate failed (`256 +23.55%`, `1024 +21.36%`).
  - `amd-zen5`
    - `blake3/oneshot`: gate failed (`256 +18.64%`, `1024 +17.25%`).
  - `amd-zen4`
    - `blake3/oneshot`: gate passed (`256 -4.82%`, `1024 -0.82%`).
- Decision:
  - Reject and revert.
  - Keep Candidate U intact (Intel SPR AVX-512 short-input win remains the current baseline).
- Narrow reintroduction direction:
  - Do not use a global AVX2 `hash_many` one-chunk fast path.
  - Reintroduce only behind microarchitecture gating (start with `TuneKind::Zen4` allowlist), then validate `intel-icl` + `amd-zen5` remain neutral while checking whether Zen4 still benefits.

### 2026-02-24 - Candidate W (in progress)
- Hypothesis:
  - Revert-only baseline is materially worse than Candidate V on `zen4/zen5/icl` at `256/1024`.
  - Candidate V behavior should be reintroduced narrowly, not globally:
    - enable AVX2 one-chunk `hash_many` only for known x86 tune kinds where it helped (`Zen4`, `Zen5`, `Zen5c`, `IntelIcl`),
    - keep other architectures/kinds unchanged.
- Change:
  - `x86_64` one-chunk path:
    - restore AVX2 exact-block one-chunk `hash_many_avx2` fast path.
    - gate the path with a tune-kind allowlist (`Zen4`, `Zen5`, `Zen5c`, `IntelIcl`).
  - dispatch plumbing:
    - expose resolved BLAKE3 tune kind via `dispatch::tune_kind()` for hot-path gating.
- Validation:
  - Local: `just check-all && just test` passed.
  - CI plan (targeted): `intel-icl`, `amd-zen4`, `amd-zen5`, `intel-spr` with `blake3/oneshot` gap gate.
- CI outcomes:
  - CI run: `22353969789`
  - `amd-zen4`: gate passed (`256 -4.78%`, `1024 -0.75%`).
  - `intel-spr`: gate passed (`256 -3.50%`, `1024 -3.65%`, `4096 +6.75%`).
  - `amd-zen5`: gate failed, but improved vs prior (`256 +19.34%`, `1024 +17.08%`).
  - `intel-icl`: gate failed, but improved vs prior (`256 +25.52%`, `1024 +21.81%`).
- Decision:
  - Keep as new baseline.
  - This restores wins on `zen4` and preserves/extends the Intel SPR win while reducing the `zen5`/`icl` short-size deficit.

### 2026-02-24 - Candidate X (`3cbfa0c`)
- Hypothesis:
  - Remaining misses are concentrated at `256/1024` on `zen5` and `intel-icl`, where size-class `s` still routes through AVX2.
  - Promoting `s` to AVX-512 for these two families should force short exact-block one-chunk inputs onto the AVX-512 asm fast path and close the residual gap.
- Change:
  - `dispatch_tables`:
    - `PROFILE_X86_ZEN5.dispatch.s`: `X86Avx2 -> X86Avx512`.
    - `PROFILE_X86_INTEL_ICL.dispatch.s`: `X86Avx2 -> X86Avx512`.
- Validation:
  - Local: `just check-all && just test` passed.
  - CI plan (targeted): `intel-icl`, `amd-zen5`, plus `amd-zen4`/`intel-spr` guard lanes.
- CI outcomes:
  - CI run: `22468964422` (2026-02-27; targeted `oneshot` + enforced `kernel-ab` gate).
  - `amd-zen5`
    - `blake3/oneshot`: gate passed (`256 +0.27%`, `1024 -1.47%`, `4096 +5.64%`, `65536 -0.32%`).
    - `blake3/kernel-ab`: gate passed (`256 -2.97%`, `1024 -1.41%`, `4096 +4.08%`, `65536 +2.16%`).
  - `amd-zen4`
    - `blake3/oneshot`: gate passed (`256 -4.74%`, `1024 -0.77%`, `4096 +6.00%`, `65536 +0.77%`).
    - `blake3/kernel-ab`: gate passed (`256 -10.49%`, `1024 -5.63%`, `4096 +5.71%`, `65536 +0.65%`).
  - `intel-icl`
    - `blake3/oneshot`: gate passed (`256 -0.55%`, `1024 -2.14%`, `4096 +7.34%`, `65536 +1.66%`).
    - `blake3/kernel-ab`: gate passed (`256 -0.79%`, `1024 -2.40%`, `4096 +6.89%`, `65536 +1.54%`).
  - `intel-spr` (guard lane)
    - `blake3/oneshot`: gate passed (`256 -3.81%`, `1024 -2.54%`, `4096 +6.35%`, `65536 +2.74%`).
    - `blake3/kernel-ab`: gate failed at `4096` (`+7.96%` vs `+6.00%` limit); other enforced sizes passed (`256 -4.28%`, `1024 -1.76%`, `65536 +2.29%`).
- Decision:
  - Hold/keep Candidate X for now; do not revert from this single run.
  - Immediate next step: rerun a targeted `intel-spr` `blake3/kernel-ab` gate check to confirm whether the `4096` miss is repeatable or run noise before final keep/reject.
  - SPR confirmation rerun: CI run `22470059519` (2026-02-27, `intel-spr` only, `filter=kernel-ab`, enforced kernel gate) also failed at `4096`, but narrowly (`+6.08%` vs `+6.00%` limit; `256 -2.46%`, `1024 -4.63%`, `65536 -0.40%` passed).
  - Interpretation: the SPR `4096` kernel-ab miss is repeatable but near-threshold; treat as a real guard-lane regression risk until we either claw back ~0.1-0.3% at `4096` or retune gate policy with explicit justification.

### 2026-02-27 - Candidate Y (`5c52746`)
- Hypothesis:
  - The remaining Intel SPR `kernel-ab` miss at `4096` is a fixed-overhead issue in exact power-of-two oneshot tree reduction.
  - Reducing stack zero-init pressure for small exact trees (`<=16` chunks) should recover margin without touching dispatch policy.
- Change:
  - In `root_output_oneshot` exact-tree fast path:
    - split into two tiers:
      - `full_chunks <= MAX_SIMD_DEGREE` uses small stack buffers (`16/8` CV slots),
      - `full_chunks <= FAST_TREE_MAX_CHUNKS` keeps the existing larger (`128/64`) path.
  - No dispatch-table changes.
- Validation:
  - Local: `just check-all && just test` passed.
  - CI bench run: `22470926772` (2026-02-27, `intel-spr` only, `filter=kernel-ab`, enforced kernel gate).
- CI outcomes:
  - `intel-spr` `blake3/kernel-ab` gate passed:
    - `256 -6.75%`
    - `1024 +0.19%`
    - `4096 +1.83%`
    - `16384 +1.90%`
    - `65536 +1.78%`
- Decision:
  - Keep.
  - This closed the repeatable `4096` guard-lane miss with margin.

### 2026-02-27 - Candidate Z (`3e9bac9`)
- Hypothesis:
  - Exact-tree reduction still pays avoidable copy-back overhead each level (`next -> cur`), especially at `16KiB`/`64KiB` trees.
  - Ping-ponging source/destination buffers per level should reduce fixed reduction cost while preserving semantics.
- Change:
  - In `root_output_oneshot` exact-tree fast path:
    - replace per-level `copy_from_slice` back into `cur` with alternating `cur`/`next` ping-pong parent folding,
    - apply to both `<=16` and `<=128` chunk paths, for little- and non-little-endian branches.
  - No dispatch or API changes.
- Validation:
  - Local: `just check-all && just test` passed.
  - CI bench run: `22494693671` (2026-02-27, `intel-spr` only, `filter=kernel-ab`, enforced kernel gate).
- CI outcomes:
  - `intel-spr` `blake3/kernel-ab` gate passed:
    - `256 -3.17%`
    - `1024 -4.67%`
    - `4096 -1.34%`
    - `16384 -2.61%`
    - `65536 -1.42%`
- Decision:
  - Keep as current baseline.
  - SPR guard-lane risk from Candidate X is resolved; all enforced SPR kernel-ab sizes are now wins in this targeted run.
