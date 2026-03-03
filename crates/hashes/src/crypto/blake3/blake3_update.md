# BLAKE3 Performance Plan (Locked Loop)

## 2026-03-01 Competitive Snapshot Update

- Data source: [benchmark-results/ci-22530359798-blake3/blake3_tally.json](/Users/mr.wolf/loadingalias/rscrypto/benchmark-results/ci-22530359798-blake3/blake3_tally.json)
- Run: `22530359798` (`quick=false`, `crates=hashes`, `benches=comp`, `only=blake3`, all 8 arches), commit `b160f5ac...`.
- Overall: `228W / 211L / 1T` (`440` cases, `51.8%` wins).
- Delta vs `22527347170`: `-2W / +5L / -3T`.
- Baseline status: `22527347170` remains the best full-matrix Blake3 snapshot in-repo.

Per arch (`wins/total`):
- `amd_zen4`: `30/55` (`54.5%`)
- `amd_zen5`: `18/55` (`32.7%`)
- `aws_graviton3`: `26/55` (`47.3%`)
- `aws_graviton4`: `28/55` (`50.9%`)
- `ibm_power10_ppc64le`: `34/55` (`61.8%`)
- `ibm_z_s390x`: `35/55` (`63.6%`)
- `intel_ice_lake`: `27/55` (`49.1%`)
- `intel_sapphire_rapids`: `30/55` (`54.5%`)

Per group:
- `derive-key`: `98W / 6L` (`94.2%`) - strong and stable.
- `hash`: `55W / 49L` (`52.9%`) - mixed, with persistent `65B`/`65536` weak points.
- `keyed-hash`: `52W / 51L / 1T` (`50.0%`) - mixed, similar edge-case weakness.
- `streaming`: `12W / 52L` (`18.8%`) - primary deficit.
- `xof`: `11W / 53L` (`17.2%`) - primary deficit and slightly worse than prior baseline.

Clear loss clusters:
- Streaming short chunks (`64B`, `128B`, `256B`, `512B`, `1024B`) are `0/8` each.
- XOF short-output (`32B-out`) at `1B-in`, `64B-in`, `1024B-in` are `0/8` each.
- `65B` edge remains weak (`hash/65` and `keyed-hash/65` lose on `7/8` arches).
- Large rayon compare points (`65536`, `1048576`) remain red on select arches (notably Graviton and Intel SPR surfaces).

## 2026-02-28 Current Competitive Snapshot

- Data source: [benchmark-results/ci-22527347170-blake3/blake3_tally.json](/Users/mr.wolf/loadingalias/rscrypto/benchmark-results/ci-22527347170-blake3/blake3_tally.json)
- Run: `22527347170` (`quick=false`, `crates=hashes`, `benches=comp`, `only=blake3`, all 8 arches), commit `171d60de...`.
- Overall: `230W / 206L / 4T` (`440` cases, `52.3%` wins).
- Baseline status: this is the best clean full-matrix Blake3 snapshot so far in-repo (`+2W`, `-6L`, `+3T` vs `22523614273`).

Per arch (`wins/total`):
- `amd_zen4`: `31/55` (`56.4%`)
- `amd_zen5`: `18/55` (`32.7%`)
- `aws_graviton3`: `27/55` (`49.1%`)
- `aws_graviton4`: `28/55` (`50.9%`)
- `ibm_power10_ppc64le`: `34/55` (`61.8%`)
- `ibm_z_s390x`: `36/55` (`65.5%`)
- `intel_ice_lake`: `27/55` (`49.1%`)
- `intel_sapphire_rapids`: `29/55` (`52.7%`)

Per group:
- `derive-key`: `97W / 6L / 1T` (`93.3%`) - strong.
- `hash`: `55W / 47L / 2T` (`52.9%`) - mixed.
- `keyed-hash`: `53W / 50L / 1T` (`51.0%`) - mixed.
- `streaming`: `12W / 52L` (`18.8%`) - primary deficit.
- `xof`: `13W / 51L` (`20.3%`) - improved, still primary deficit.

Clear loss clusters:
- Streaming: `64B`, `128B`, `256B`, `512B`, `1024B` chunk modes are `0/8` each.
- XOF short-output (`32B`) remains red across all arches (`1B-in`, `64B-in`, `1024B-in` are `0/8`).
- Keyed/hash edge case around `65B` is still weak (`keyed-hash/65` is `0/8`, `hash/65` is `1/8`).

## Next Steps (Blake3)

1. Hit streaming short-chunk overhead first (`64..1024`): remove avoidable per-call finalize/copy work on one-chunk and few-chunk paths.
2. Add an explicit XOF short-output fast path (`32B-out`) that avoids generic squeeze setup overhead and first-block overwork.
3. Fix the persistent `65B` edge in `hash`/`keyed-hash` by reducing small-tail branching and one-block setup overhead.
4. Keep dispatch-table churn frozen unless a change is proven by kernel/codegen evidence.

## First Step (Blake3)

- Implement one minimal kernel/codegen candidate only for short paths:
  - target streaming/xof paths at `64..1024` and `32B-out`,
  - no dispatch table edits in this first pass.
- Validate with targeted non-quick lanes first (`graviton3`, `graviton4`, `intel-icl`, `intel-spr`, `amd-zen5`), then rerun full 8-arch `blake3/comp` if trend is clean.
- Keep/revert immediately based on cross-lane net wins.

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
### 2026-03-03 - XOF Helper Signature Refactor (No Suppression)
- Change:
  - Removed `#[allow(clippy::too_many_arguments)]` from `xof_many_via_compress`.
  - Introduced `XofCompressInput` to carry invariant XOF parameters and shrink helper arity structurally.
  - Added the missing `// SAFETY:` note on the NEON XOF tail path.
  - Cleaned cfg-dependent AVX-512 wrapper lint fallout (`unused mut`, `needless return`) without behavior changes.
- Validation:
  - Local: `just check-all` passed (host + cross-target groups).
  - Local: `just test` passed (`167/167`).
- Notes:
  - This is a hygiene/code-shape change only; no intended algorithmic or dispatch-policy shift.

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

### 2026-02-27 - Candidate AA (`99874ad`)
- Hypothesis:
  - Generic one-chunk root hashing still pays avoidable dispatch overhead on non-x86/aarch64 lanes.
  - Replacing the final-step `kernel.compress` function-pointer call with an inline kernel-id dispatch helper should improve IBM short-input `kernel-ab` gaps (`256`/`1024`).
- Change:
  - Added `kernels::compress_inline(...)` and switched `digest_one_chunk_root_hash_words_generic` final-block compression to use it.
  - No dispatch-table changes.
- Validation:
  - Local: `just check-all && just test` passed.
  - CI targeted bench run: `22495902076` (`ibm-s390x` + `ibm-power10`, `filter=kernel-ab`, enforced kernel gate).
- CI outcomes:
  - `ibm-power10`: gate passed; all enforced sizes remained wins.
  - `ibm-s390x`: gate failed at short sizes:
    - `256 +14.54%` vs `+12.00%` limit (prior `+13.40%` in `22284378402`)
    - `1024 +10.10%` vs `+10.00%` limit (prior `+9.20%` in `22284378402`)
    - medium/large sizes remained strong wins.
- Decision:
  - Reject and revert.
  - This change did not close the intended IBM short-input gap and slightly worsened `s390x` `256/1024`.

### 2026-03-01 - Candidate AB (`bd61c53`)
- Hypothesis:
  - Single-chunk XOF short-output (`32B-out`) pays avoidable fixed overhead from eager zero-init of `Blake3Xof` scratch buffers (`buf` + `root_hash_cache`).
  - Making these buffers lazy-initialized should improve `xof init+read` at `1B`/`64B`/`1024B` inputs.
- Change:
  - In `Blake3Xof`:
    - switched `buf` and `root_hash_cache` to `MaybeUninit`,
    - initialized `buf` only in `refill`,
    - initialized `root_hash_cache` only when short-root fast path is used.
- Validation:
  - Local: `just check-all && just test` passed.
  - CI targeted bench run: `22546811144` (`crates=hashes`, `benches=blake3`, `filter=xof/`,
    lanes: `intel-icl`, `intel-spr`, `amd-zen5`, `graviton3`, `graviton4`).
- CI outcomes (`xof/init+read/*-in/32B-out`):
  - Net: `2W / 13L` vs official; median gap remained strongly negative.
  - Wins only on `amd-zen5` (`1B`, `64B`), but multiple severe x86 losses remained (`intel-icl`, `intel-spr`).
- Decision:
  - Reject and revert.
  - Not a cross-lane win on the targeted 32B-out gap cluster.

### 2026-03-01 - Candidate AC (`aea12be`)
- Hypothesis:
  - Single-chunk `finalize_xof()` can inherit a deferred streaming kernel (`portable`) from `update()`, hurting short XOF competitiveness.
  - Rebuilding single-chunk XOF output with one-shot size-class dispatch kernel should reduce this pinning.
- Change:
  - In `finalize_xof` single-chunk/no-tree path:
    - switched from `self.kernel` to `dispatch_plan.size_class_kernel(self.chunk_state.len())`,
    - rebuilt output via `chunk_output_with_kernel(kernel)`.
- Validation:
  - Local: `just check-all && just test` passed.
  - CI targeted bench run: `22548433906` (same scoped `xof/` lane set as Candidate AB).
- CI outcomes (`xof/init+read/*-in/32B-out`):
  - Net: `0W / 15L` vs official.
  - Average gap: `-17.75%` (median: `-18.38%`).
  - Notable losses persisted across all five lanes.
- Decision:
  - Reject and revert.
  - This change improved selected Intel points versus AB but remained a full loss set against official on the target surface.

## Immediate Next Candidate (after AC revert)

1. Narrow single-chunk XOF kernel override:
   - keep current `finalize_xof()` behavior by default,
   - only override kernel when `self.kernel == Portable` **and** tune-kind is an x86 Intel family (`IntelIcl`/`IntelSpr`), using size-class selection.
2. Keep non-Intel lanes unchanged in this candidate (avoid another cross-lane cliff).
3. Re-run the same targeted `bench.yaml` scope (`xof/`, 5 lanes above), then keep/revert immediately based on net `32B-out` results.

### 2026-03-01 - Candidate AD (`853fd53`)
- Hypothesis:
  - AC was too broad; rebuilding single-chunk XOF output with size-class selection on all lanes caused cross-lane losses.
  - Restricting the override to Intel x86_64 when update-path dispatch remained `Portable` should improve Intel short XOF cases while keeping non-Intel behavior stable.
- Change:
  - In `finalize_xof` single-chunk/no-tree path:
    - default path stayed unchanged (`self.chunk_state.output()`, `self.kernel`),
    - override to `dispatch_plan.size_class_kernel(self.chunk_state.len())` only when:
      - target is `x86_64`,
      - `self.kernel == Portable`,
      - tune-kind is `IntelIcl` or `IntelSpr`.
- Validation:
  - Local: `just check-all && just test` passed.
  - CI targeted bench run: `22549116807` (`crates=hashes`, `benches=blake3`, `filter=xof/`,
    lanes: `intel-icl`, `intel-spr`, `amd-zen5`, `graviton3`, `graviton4`).
  - Note: this run used pre-fix bench planning and emitted both `filter=blake3` and `filter=xof/`; decisions below are based on the dedicated `xof/` pass only.
- CI outcomes (`xof/init+read/*-in/32B-out`):
  - Net: `4W / 16L` vs official.
  - Average gap: `-10.60%` (median: `-15.03%`).
  - Intel remained the blocker (`intel-icl`: `0W / 4L`, `intel-spr`: `0W / 4L`).
- Decision:
  - Reject and revert.
  - The narrow Intel override did not improve the target `32B-out` gap cluster enough to keep.

### 2026-03-01 - Candidate AE (`dd8be8e`)
- Hypothesis:
  - XOF `init+read/*-in/32B-out` remains weak because single-chunk finalize can stay pinned to `Portable` after deferred-SIMD updates.
  - Streaming remains weak on x86 because profile tables still select `SSE4.1` for the per-update stream kernel.
- Change:
  - `finalize_xof` single-chunk path:
    - when current kernel is `Portable`, rebuild chunk output with `dispatch_plan.stream_kernel()` and use that for XOF state.
    - keep existing behavior for non-Portable and multi-chunk states.
  - x86 dispatch tables:
    - set default streaming kernel to `X86Avx2`,
    - set `Zen4`, `Zen5`, `IntelIcl`, and `IntelSpr` streaming kernels to `X86Avx2` (bulk remains `X86Avx512`).
- Validation:
  - Local: `just check-all && just test` passed.
  - CI targeted bench run: `22552154576` (`crates=hashes`, `benches=blake3`,
    `filter=xof/init+read/,streaming/`, lanes: `intel-icl`, `intel-spr`, `amd-zen5`).
  - Note: `xof/init+read/` filter matched no benchmarks in this run (`xof_cases=0` on all three lanes); run evidence is valid for streaming only.
- CI outcomes:
  - `streaming/*`: `0W / 24L` vs official (3 lanes x 8 chunk sizes).
  - Aggregate streaming gap: avg `-19.98%` (median `-20.61%`).
  - Lane-average streaming delta vs prior baseline run (`22549116807`):
    - `intel-icl`: `-18.72%` -> `-20.76%` (`-2.04 pp`),
    - `intel-spr`: `-16.09%` -> `-19.70%` (`-3.61 pp`),
    - `amd-zen5`: `-18.03%` -> `-19.48%` (`-1.45 pp`).
- Decision:
  - Reject and revert.
  - This candidate regressed streaming and did not produce actionable XOF data from the scoped run.

### 2026-03-01 - Candidate AF (`c3eaa72`)
- Hypothesis:
  - XOF short-output losses persist when single-chunk finalize inherits a deferred `Portable` kernel.
  - Streaming losses at exact `1024B` chunk updates are partly due to conservative stream-kernel selection at chunk boundaries.
- Change:
  - `finalize_xof` single-chunk path:
    - when current kernel is `Portable`, rebuild chunk output with `dispatch_plan.stream_kernel()` and seed XOF with that kernel.
  - `update` full-chunk policy:
    - for exact `CHUNK_LEN` updates at chunk boundary, force active chunk kernel to `bulk_kernel_for_update(input.len())`.
- Validation:
  - Local: `just check-all && just test` passed.
  - CI targeted bench run: `22554313539` (`crates=hashes`, `benches=blake3`,
    `filter=blake3/xof/,blake3/streaming/`, lanes: `intel-icl`, `intel-spr`, `amd-zen5`).
- CI outcomes:
  - Aggregate (`xof` + `streaming`): `0W / 48L` vs official.
  - Surface aggregates:
    - `streaming/*`: `0W / 24L`, avg gap `-21.56%`.
    - `xof/init+read/*`: `0W / 24L`, avg gap `-21.83%`.
  - Lane aggregates:
    - `intel-icl`: streaming `-23.08%`, xof `-31.14%`.
    - `intel-spr`: streaming `-20.13%`, xof `-16.74%`.
    - `amd-zen5`: streaming `-21.48%`, xof `-17.59%`.
  - Notable regressions remained severe on tiny-XOF cases (e.g., `intel-icl 1B-in/32B-out: -66.12%`, `intel-spr 1B-in/32B-out: -52.24%`).
- Decision:
  - Reject and revert.
  - Reverted commit `c3eaa72`; this approach does not close either target gap.

### 2026-03-01 - Candidate AG (`b1f25ea`)
- Hypothesis:
  - Tiny XOF `init+read` still pays fixed overhead from `OutputState::root_hash_bytes()` in first squeeze.
  - Precomputing/caching single-chunk root hash at `finalize_xof()` should improve `*-in/32B-out` cases.
- Change:
  - Added direct single-chunk root-hash helper and reused it in `finalize()`.
  - `finalize_xof()` single-chunk path precomputed root hash and seeded `Blake3Xof` cache.
  - `Blake3Xof::squeeze()` tiny first-read path used cache when available.
- Validation:
  - Local: `just check-all && just test` passed.
  - CI targeted bench run: `22555765324` (`crates=hashes`, `benches=blake3`,
    `filter=blake3/xof/`, lanes `intel-icl`, `intel-spr`, `amd-zen5`).
- CI outcomes:
  - Aggregate (`xof/*`): `2W / 22L` vs official.
  - Aggregate xof gap: avg `-42.03%`.
  - Lane-average gaps:
    - `intel-icl`: `-53.32%`.
    - `intel-spr`: `-45.39%`.
    - `amd-zen5`: `-27.37%`.
  - Delta vs prior xof baseline run (`22554313539`) by lane:
    - `intel-icl`: `-31.14%` -> `-53.32%` (`-22.18 pp`).
    - `intel-spr`: `-16.74%` -> `-45.39%` (`-28.65 pp`).
    - `amd-zen5`: `-17.59%` -> `-27.37%` (`-9.78 pp`).
- Decision:
  - Reject and revert.
  - Reverted commit `b1f25ea`; eager root-hash precompute in `finalize_xof()` regressed XOF broadly.

### 2026-03-02 - Candidate AH (`03d9943`)
- Hypothesis:
  - AG regressed because eager root-hash precompute at `finalize_xof()` adds fixed cost to all XOF calls.
  - A lazy tiny-first-squeeze optimization should improve `*-in/32B-out` without harming large-output XOF.
- Change:
  - Added single-chunk tail hint plumbing from `finalize_xof()` into `Blake3Xof` for single-chunk/no-tree states.
  - Tiny first-squeeze path (`out.len() <= 32`) used direct chunk-tail root hashing from the hint.
  - Kept large-output XOF path unchanged (no eager finalize-time precompute).
- Validation:
  - Local: `just check-all && just test` passed.
  - CI targeted bench run: `22556963869` (`crates=hashes`, `benches=blake3`,
    `filter=blake3/xof/`, lanes: `intel-icl`, `intel-spr`, `amd-zen5`).
- CI outcomes:
  - Aggregate (`xof/*`): `1W / 23L` vs official.
  - Aggregate xof gap: avg `-29.00%`.
  - Lane-average gaps:
    - `intel-icl`: `-40.78%`.
    - `intel-spr`: `-29.37%`.
    - `amd-zen5`: `-16.84%`.
  - Delta vs prior non-AG baseline run (`22554313539`) by lane:
    - `intel-icl`: `-31.14%` -> `-40.78%` (`-9.64 pp`).
    - `intel-spr`: `-16.74%` -> `-29.37%` (`-12.63 pp`).
    - `amd-zen5`: `-17.59%` -> `-16.84%` (`+0.75 pp`).
- Decision:
  - Reject and revert.
  - Reverted commit `03d9943`; this path improved over AG but remained a clear regression vs baseline on Intel lanes.

### 2026-03-02 - Candidate AI (`39855a8`)
- Hypothesis:
  - `OutputState` still pays function-pointer dispatch (`kernel.compress`) on finalize/XOF scalar block generation.
  - Replacing those calls with inline `kernel.id` dispatch might reduce fixed overhead on XOF and streaming surfaces.
- Change:
  - Added `kernels::compress_inline(id, ...)`.
  - Routed `OutputState::chaining_value`, `OutputState::root_hash_words`, and scalar fallback in
    `root_output_blocks_into` through `compress_inline`.
  - No dispatch-table or algorithm changes.
- Validation:
  - Local pre-push validation passed: `just check-all && just test`.
  - CI targeted bench run: `22559324946` (`crates=hashes`, `benches=blake3`,
    `filter=blake3/xof/,blake3/streaming/`, lanes: `intel-icl`, `intel-spr`, `amd-zen5`).
  - Scope check: execution plan contained exactly two rows (`blake3/xof/`, `blake3/streaming/`), no oneshot scope.
- CI outcomes (time-based gap vs official; positive = slower):
  - Aggregate (`xof` + `streaming`): `2W / 46L`, avg gap `+22.94%`.
  - `xof/*`: `2W / 22L`, avg gap `+24.39%` (median `+21.50%`).
  - `streaming/*`: `0W / 24L`, avg gap `+21.48%` (median `+22.15%`).
  - Lane aggregates:
    - `intel-icl`: `xof 0W/8L` avg `+41.09%`; `streaming 0W/8L` avg `+23.51%`.
    - `intel-spr`: `xof 0W/8L` avg `+20.62%`; `streaming 0W/8L` avg `+19.02%`.
    - `amd-zen5`: `xof 2W/6L` avg `+11.48%`; `streaming 0W/8L` avg `+21.91%`.
  - Notable regressions:
    - `intel-icl xof init+read/1B-in/32B-out`: `+93.93%`.
    - `intel-icl xof init+read/64B-in/32B-out`: `+90.49%`.
- Decision:
  - Reject and revert.
  - Reverted `39855a8` via `a787d48`.

### 2026-03-02 - Candidate AJ (`57b90a8`)
- Hypothesis:
  - XOF `init+read` tiny-output is still paying avoidable per-call cost by using full `[u32; 16]` compression to derive the root hash words.
  - Streaming remains pinned to conservative x86 stream kernels despite having AVX-512 asm `compress_in_place` path available.
- Change:
  - Added kernel-aware `OutputState` root-hash path:
    - `root_hash_words_with_kernel` / `root_hash_bytes_with_kernel`,
    - CV-only x86 (`x86_compress_cv_bytes`) and aarch64 NEON (`compress_cv_neon_bytes`) paths when available.
  - Updated XOF tiny first-read fast path to:
    - opportunistically upgrade from `Portable` to tuned stream kernel on x86,
    - compute first root hash via kernel-aware CV-only path.
  - Retuned x86 streaming tables:
    - `IntelIcl`, `IntelSpr`, `Zen5` stream kernel set to `X86Avx512` (bulk unchanged `X86Avx512`).
- Validation:
  - Local: `just check-all && just test` passed.
  - CI targeted bench run: `22560608081` (`crates=hashes`, `benches=blake3`,
    `filter=blake3/xof/,blake3/streaming/`, lanes: `intel-icl`, `intel-spr`, `amd-zen5`).
  - Scope check: execution plan contained exactly two rows (`blake3/xof/`, `blake3/streaming/`), no oneshot scope.
- CI outcomes (time-based gap vs official; positive = slower):
  - Aggregate (`xof` + `streaming`): `0W / 48L`, avg gap `+16.46%`.
  - `xof/*`: `0W / 24L`, avg gap `+19.37%` (median `+15.92%`).
  - `streaming/*`: `0W / 24L`, avg gap `+13.56%` (median `+13.60%`).
  - Lane aggregates:
    - `intel-icl`: `xof 0W/8L` avg `+26.25%`; `streaming 0W/8L` avg `+13.38%`.
    - `intel-spr`: `xof 0W/8L` avg `+17.02%`; `streaming 0W/8L` avg `+13.54%`.
    - `amd-zen5`: `xof 0W/8L` avg `+14.83%`; `streaming 0W/8L` avg `+13.76%`.
  - Directional delta vs prior baseline run (`22559324946`):
    - aggregate: `+22.94%` -> `+16.46%` (`-6.47 pp`),
    - streaming: `+21.48%` -> `+13.56%` (`-7.92 pp`),
    - xof: `+24.39%` -> `+19.37%` (`-5.03 pp`).
- Decision:
  - Reject and revert.
  - Reverted `57b90a8` via `70e7519`.

### 2026-03-02 - Candidate AK (`bd2f124`)
- Hypothesis:
  - Streaming short-chunk loss still includes avoidable per-block control-path overhead from repeated `kernel.id` inline dispatch inside `ChunkState` block compression.
  - XOF tiny first-read root-hash cache path may be fighting the kernel fast path and diverging from official reader behavior.
- Change:
  - In `ChunkState` hot paths:
    - replaced repeated `kernels::chunk_compress_blocks_inline(self.kernel.id, ...)` calls with direct
      `(self.kernel.chunk_compress_blocks)(...)` calls.
  - In `Blake3Xof`:
    - removed `root_hash_cache` / `root_hash_pos` state and the `out.len() <= 32` first-read root-hash fast path,
    - kept first output generation on normal block path (`root_output_blocks_into`) only.
  - No dispatch-table edits.
- Validation:
  - Local: `just check-all && just test` passed.
  - CI targeted bench run: `22588630907` (`crates=hashes`, `benches=blake3`,
    `filter=blake3/xof/,blake3/streaming/`, lanes: `intel-icl`, `intel-spr`, `amd-zen5`).
  - Scope check: execution plan contained exactly two rows (`blake3/xof/`, `blake3/streaming/`) on all lanes.
- CI outcomes (time-based gap vs official; positive = slower):
  - Aggregate (`xof` + `streaming`): `2W / 46L`, avg gap `+23.51%`.
  - `xof/*`: `2W / 22L`, avg gap `+25.77%`.
  - `streaming/*`: `0W / 24L`, avg gap `+21.26%`.
  - Lane aggregates:
    - `intel-icl`: `xof 0W/8L` avg `+44.77%`; `streaming 0W/8L` avg `+22.91%`.
    - `intel-spr`: `xof 0W/8L` avg `+21.74%`; `streaming 0W/8L` avg `+18.94%`.
    - `amd-zen5`: `xof 2W/6L` avg `+10.80%`; `streaming 0W/8L` avg `+21.93%`.
  - Directional delta vs prior targeted baseline (`22560608081`, Candidate AJ):
    - aggregate: `+16.46%` -> `+23.51%` (`+7.05 pp`),
    - streaming: `+13.56%` -> `+21.26%` (`+7.70 pp`),
    - xof: `+19.37%` -> `+25.77%` (`+6.40 pp`).
  - Notable worst regressions:
    - `intel-icl xof init+read/64B-in/32B-out`: `+101.63%`.
    - `intel-icl xof init+read/1B-in/32B-out`: `+94.53%`.
    - `intel-spr xof init+read/1B-in/32B-out`: `+48.78%`.
- Decision:
  - Reject and revert.
  - Reverted `bd2f124` via `b0c088b`.

### 2026-03-02 - Candidate AL (`e7fbaf9`)
- Hypothesis:
  - XOF gaps are still dominated by tiny-input kernel pinning and extra runtime upgrade logic in `Blake3Xof::squeeze`.
  - Streaming `64..1024` update gaps still include avoidable first-chunk control overhead on x86 and over-deferral of SIMD for repeated block-aligned updates.
- Change:
  - Added x86 one-full-chunk `ChunkState` fast path:
    - `try_chunk_state_one_chunk_x86_out(...)` uses AVX2/AVX-512 asm `hash_many` with `blocks=15` to materialize pre-final-block CV and copy block 15 bytes.
  - Simplified XOF kernel model:
    - removed runtime XOF kernel/dispatch-upgrade state from `Blake3Xof`,
    - moved kernel selection to finalize time (`finalize_xof` => stream kernel; `finalize_xof_sized` => stream/bulk by expected output),
    - retagged one-shot XOF outputs in `dispatch::xof` and `Blake3::keyed_xof` to stream kernel.
  - Streaming defer tweak:
    - in `Digest::update` first-update path, skip deferred-SIMD for likely chunked workloads (`existing buffered chunk data + block-aligned incoming update`).
- Validation:
  - Local pre-push: `just check-all && just test` passed.
  - CI targeted bench run: `22593524598` (`crates=hashes`, `benches=blake3`,
    `filter=blake3/xof/,blake3/streaming/`, lanes: `intel-icl`, `intel-spr`, `amd-zen5`).
  - Scope check: execution plan contained exactly two rows (`blake3/xof/`, `blake3/streaming/`) on all lanes.
- CI outcomes (time-based gap vs official; positive = slower):
  - Aggregate (`xof` + `streaming`): `0W / 48L`, avg gap `+62.89%`.
  - `xof/*`: `0W / 24L`, avg gap `+104.06%`.
  - `streaming/*`: `0W / 24L`, avg gap `+21.72%`.
  - Lane aggregates:
    - `intel-icl`: `xof 0W/8L` avg `+96.56%`; `streaming 0W/8L` avg `+23.47%`.
    - `intel-spr`: `xof 0W/8L` avg `+99.93%`; `streaming 0W/8L` avg `+19.43%`.
    - `amd-zen5`: `xof 0W/8L` avg `+115.69%`; `streaming 0W/8L` avg `+22.25%`.
  - Notable worst regressions:
    - `amd-zen5 xof init+read/1B-in/1024B-out`: `+364.61%`.
    - `amd-zen5 xof init+read/64B-in/1024B-out`: `+359.63%`.
    - `intel-spr xof init+read/64B-in/1024B-out`: `+295.21%`.
- Decision:
  - Reject and revert.
  - Reverted `e7fbaf9` via `71d45a1`.

### 2026-03-02 - Candidate AM (`5a7df00` + `8885536`)
- Hypothesis:
  - Streaming and XOF short-path gaps on Linux x86_64 are still dominated by call overhead in the Rust SSE4.1 CV path.
  - Wiring the generated Linux SSE4.1 asm backend into streaming/XOF hot paths should reduce fixed overhead for `64..1024` and `32B-out` cases.
- Change:
  - Added Linux SSE4.1 asm entry points for chunk CV compression and `compress_in_place`.
  - Hooked x86_64 streaming/XOF hot paths to use those asm hooks on Linux when SSE4.1 is selected.
  - Fixed assembler section directive for Linux buildability:
    - `.static_data` -> `.section .rodata` in `rscrypto_blake3_sse41_x86-64_unix_linux.s`.
- Validation:
  - Local pre-push: `just check-all && just test` passed.
  - CI targeted bench run: `22600518259` (`crates=hashes`, `benches=blake3`,
    `filter=blake3/xof/,blake3/streaming/`, lanes: `intel-icl`, `intel-spr`, `amd-zen5`).
  - Scope check: execution plan contained exactly two rows (`blake3/xof/`, `blake3/streaming/`) on all lanes (no oneshot).
- CI outcomes (time-based gap vs official; positive = slower):
  - Aggregate (`xof` + `streaming`): `2W / 46L`, avg gap `+24.80%`.
  - `xof/*`: `2W / 22L`, avg gap `+25.56%`.
  - `streaming/*`: `0W / 24L`, avg gap `+24.04%`.
  - Lane aggregates:
    - `intel-icl`: `xof 0W/8L` avg `+40.90%`; `streaming 0W/8L` avg `+24.48%`.
    - `intel-spr`: `xof 0W/8L` avg `+23.42%`; `streaming 0W/8L` avg `+22.54%`.
    - `amd-zen5`: `xof 2W/6L` avg `+12.35%`; `streaming 0W/8L` avg `+25.11%`.
  - Notable points:
    - worst: `intel-icl xof init+read/1B-in/32B-out` at `+91.41%`.
    - worst: `intel-icl xof init+read/64B-in/32B-out` at `+83.21%`.
    - only wins: `amd-zen5 xof init+read/1B-in/32B-out` (`-12.46%`) and `64B-in/32B-out` (`-20.54%`).
- Decision:
  - Reject and revert.
  - Reverted `8885536` and `5a7df00` via `097edf6`.

### 2026-03-03 - Candidate AN (`56485ac`)
- Hypothesis:
  - Remaining streaming/XOF losses were dominated by control-path complexity (defer/upgrade/cache logic) rather than raw SIMD throughput.
  - Simplifying to direct stream-kernel dispatch and reducing XOF state should lower fixed overhead on `blake3/streaming/*` and `blake3/xof/*`.
- Change:
  - `Blake3::update`:
    - removed deferred-SIMD gating (`should_defer_simd`) from the hot path,
    - now directly selects `stream_kernel()` + `bulk_kernel_for_update(input.len())` on update calls.
  - `Blake3::finalize_xof`:
    - routes through `finalize_xof_sized(OUT_LEN)` to use one output-construction path.
  - `Blake3Xof`:
    - removed `root_hash_cache` / `root_hash_pos`,
    - removed runtime squeeze kernel-upgrade state (`kernel`, `dispatch_plan`),
    - simplified `squeeze()` to buffered block generation only.
  - x86 profile policy:
    - set `PROFILE_X86_ZEN5`, `PROFILE_X86_INTEL_ICL`, and `PROFILE_X86_INTEL_SPR` streaming kernel to `X86Avx512`.
  - Cleanup:
    - removed dead `HasherDispatch::simd_threshold` field and `should_defer_simd` method.
- Validation:
  - Local: `just check-all && just test` passed (`167/167`).
  - CI targeted bench run: `22601614792` (`crates=hashes`, `benches=blake3`,
    `filter=blake3/xof/,blake3/streaming/`, lanes: `intel-icl`, `intel-spr`, `amd-zen5`).
  - Scope check: execution plan contained exactly two rows (`blake3/xof/`, `blake3/streaming/`) on all lanes.
- CI outcomes (time-based gap vs official; positive = slower):
  - Aggregate (`xof` + `streaming`): `0W / 48L`, avg gap `+29.60%`.
  - `streaming/*`: `0W / 24L`, avg gap `+16.18%`.
  - `xof/*`: `0W / 24L`, avg gap `+43.02%`.
  - Lane aggregates:
    - `intel-icl`: `0W/16L`, avg `+60.64%`.
    - `intel-spr`: `0W/16L`, avg `+13.01%`.
    - `amd-zen5`: `0W/16L`, avg `+15.15%`.
  - Directional delta vs prior targeted run (`22600518259`, Candidate AM):
    - aggregate: `+24.80%` -> `+29.60%` (`+4.80 pp`, worse),
    - streaming: `+24.04%` -> `+16.18%` (`-7.86 pp`, better),
    - xof: `+25.56%` -> `+43.02%` (`+17.46 pp`, much worse).
  - Notable regressions:
    - `intel-icl xof init+read/64B-in/1024B-out`: `+270.52%` (from `+20.95%`).
    - `intel-icl xof init+read/1B-in/1024B-out`: `+263.39%` (from `+31.92%`).
    - `amd-zen5 xof init+read/64B-in/32B-out`: `+18.07%` (from `-20.54%`).
- Decision:
  - Reject and revert.
  - Reverted `56485ac` via `661da62`.

### 2026-03-03 - Candidate AO (`ec296b8`)
- Hypothesis:
  - XOF short-output overhead was still inflated by reader state complexity (`buf`/`root_hash_cache`/runtime dispatch state) and extra control branches.
  - Shrinking `Blake3Xof` state and making squeeze block-generation/direct-copy first should reduce fixed overhead while preserving large-read kernel throughput.
- Change:
  - `dispatch::xof` / `Blake3::finalize_xof` / `Blake3::keyed_xof`:
    - precompute `large_squeeze_kernel = bulk_kernel_for_update(512)`,
    - construct XOF with that kernel directly.
  - `Blake3Xof` refactor:
    - removed `buf`, `buf_pos`, `root_hash_cache`, `root_hash_pos`, and stored dispatch plan state,
    - added `position_within_block` + `large_squeeze_kernel`,
    - added `fill_one_block()` and rewired `squeeze()` to direct block writes plus a single large-read kernel upgrade.
  - Cleanup:
    - removed `new_with_kernel` constructor and obsolete refill/cache machinery.
- Validation:
  - Local pre-push for the candidate commit was green before CI (`just check-all && just test`).
  - CI targeted bench run: `22605740146` (`crates=hashes`, `benches=comp`, `only=blake3`, `quick=false`,
    lanes: `amd-zen5`, `intel-icl`, `intel-spr`, `graviton3`, `graviton4`).
  - Scope check: lane artifacts contained the `blake3/streaming/*` and `blake3/xof/*` matrices used for delta extraction.
- CI outcomes (time-based gap vs official; positive = slower):
  - Aggregate (`xof` + `streaming`, 5 lanes): `18W / 62L`, avg gap `+10.50%`.
  - `streaming/*`: `6W / 34L`, avg gap `+12.82%`.
  - `xof/*`: `12W / 28L`, avg gap `+8.19%`.
  - Lane aggregates:
    - `intel-icl`: `0W/16L`, avg `+29.12%` (`streaming +23.07%`, `xof +35.17%`).
    - `intel-spr`: `0W/16L`, avg `+21.44%` (`streaming +19.34%`, `xof +23.54%`).
    - `amd-zen5`: `2W/14L`, avg `+16.02%` (`streaming +22.30%`, `xof +9.73%`).
    - `graviton3`: `8W/8L`, avg `-7.51%` (`streaming -0.67%`, `xof -14.36%`).
    - `graviton4`: `8W/8L`, avg `-6.55%` (`streaming +0.06%`, `xof -13.16%`).
  - Directional delta vs prior x86 run (`22601614792`, Candidate AN; x86 lanes only):
    - aggregate: `+29.60%` -> `+22.19%` (`-7.41 pp`, better),
    - streaming: `+16.18%` -> `+21.57%` (`+5.39 pp`, worse),
    - xof: `+43.02%` -> `+22.82%` (`-20.20 pp`, better).
  - Notable regressions vs prior x86 run:
    - `intel-spr xof init+read/1B-in/32B-out`: `+22.06%` -> `+67.80%` (`+45.74 pp`).
    - `intel-spr xof init+read/64B-in/32B-out`: `+1.67%` -> `+24.65%` (`+22.98 pp`).
    - `intel-icl xof init+read/64B-in/32B-out`: `+54.84%` -> `+76.95%` (`+22.11 pp`).
    - `intel-icl xof init+read/1B-in/32B-out`: `+59.62%` -> `+78.48%` (`+18.86 pp`).
    - `amd-zen5 streaming/64B-chunks`: `+6.08%` -> `+21.18%` (`+15.10 pp`).
- Decision:
  - Reject and revert.
  - Reverted `ec296b8` via `d4ffe54`.

### 2026-03-03 - Candidate AP (`working tree`)
- Hypothesis:
  - The remaining x86 gap is now mostly control-path overhead in XOF output production, not raw kernel throughput.
  - We still diverge from upstream architecture by doing runtime kernel-ID laddering in `root_output_blocks_into` and by carrying extra state/branches in `Blake3Xof`.
- Change:
  - Kernel dispatch model:
    - added `xof_block` function pointer to `kernels::Kernel` for single-block XOF output,
    - added `xof_many` function pointer to `kernels::Kernel`,
    - wired all kernel variants (`portable`, `x86 ssse3/sse4.1/avx2/avx512`, `aarch64 neon`, and other SIMD targets) to explicit XOF block generators.
  - Output path:
    - added `OutputState::root_output_block_into()` that dispatches through `kernel.xof_block`,
    - replaced `OutputState::root_output_blocks_into` kernel-ID `match` ladder with one indirect call through `kernel.xof_many`.
  - Reader state simplification:
    - rewrote `Blake3Xof` to upstream-style state (`output`, `block_counter`, `position_within_block`),
    - removed `buf`, `buf_pos`, `root_hash_cache`, `root_hash_pos`, runtime kernel-upgrade state, and `dispatch_plan` from `Blake3Xof`,
    - removed `new_with_kernel`/`refill`; added `fill_one_block`,
    - `fill_one_block()` now uses `xof_block` (single-block path) while full-block regions use `xof_many`.
  - Constructor cleanup:
    - updated `dispatch::xof`, `Blake3::keyed_xof`, `Blake3::finalize_xof`, and `Blake3::finalize_xof_sized` to use the simplified `Blake3Xof::new(output)`.
  - Cleanup:
    - removed dead helper `words16_to_le_bytes`.
- Validation:
  - Local: `cargo test -p hashes blake3 --lib` passed (`20/20`).
  - Local spot-check bench (`cargo bench -p hashes --bench blake3`, Apple M1, short streaming/XOF cases):
    - `streaming`: `64B +2.35%`, `128B -4.93%`, `256B +3.01%`, `512B +4.48%`, `1024B +2.52%`.
    - `xof init+read/*-in/32B-out`: `1B +23.84%`, `64B +18.98%`, `1024B +4.42%`.
  - CI run `22631079643` completed, but executed commit `d4ffe54` (remote `main`), not Candidate AP commit `686773a`; this run is invalid for AP evaluation.
    - Invalid-run outcomes (`d4ffe54`, same 5 lanes): aggregate `18W / 62L`, avg `+11.34%` (`streaming +12.79%`, `xof +9.89%`), consistent with previously rejected behavior.
  - CI targeted bench run: pending.
- Decision:
  - Pending targeted CI benches (`blake3/streaming/*` + `blake3/xof/*`).
