# BLAKE3 Performance Plan (Locked Loop)

## Read This First

### Current Truth

- `oneshot` is not the problem. We are already strong there.
- The real deficits are still:
  - repeated short streaming updates (`64..1024B` chunk modes),
  - `finalize_xof()` setup cost,
  - short-output XOF (`32B-out`).
- If a candidate improves long-read XOF or one-call `update-only` but does not flip those three surfaces, it is not a solution.

### Most Informative Failed Directions

| Theme | Candidate(s) | What happened | Verdict |
|---|---|---|---|
| Remove or weaken `pending_chunk_cv` behavior | `AV` | Helped a narrow streaming slice, broke larger streaming behavior | Dead end |
| First-read / root-hash shortcut only | `AW` | Small directional change, did not close short XOF | Not enough |
| Direct one-block XOF helper only | `AZ`, `BA` | Did not materially improve `squeeze-32-only` or short streaming | Dead end |
| Dispatch / policy churn on x86 | `AX`, `AY` | Noise at best, regression at worst | Stop doing this |
| Lazy single-chunk XOF reader | `BC` | Helped long squeezes, did not fix short streaming or `finalize_xof` | Partial signal only |
| Shrink hot state / localize bulk dispatch only | `BD` | Best setup-only movement so far, still failed target clusters | Partial signal only |
| Finalize-time compact XOF/output materialization | `BB` | Catastrophic `finalize-xof-only` regression | Hard no |
| Raw-byte generic `OutputState` rewrite | `BE` | Regressed setup, XOF, and short streaming again | Hard no |

### What Not To Do Again

- Do not retune dispatch tables as the primary strategy.
- Do not add more one-off XOF reader helpers on top of the current architecture.
- Do not move more work into `finalize_xof()`.
- Do not treat a better `update-only` microbench as proof that repeated short streaming is fixed.
- Do not touch long-read squeeze throughput unless the short surfaces are already green.

### What The Next Real Candidate Must Prove

- `blake3/streaming/64B..1024B-chunks` must stop being all-loss on x86.
- `blake3/xof-phase/finalize-xof-only` must materially improve, not just move sideways.
- `blake3/xof/init+read/*/32B-out` must improve with cross-lane wins, not just a smaller average loss.

### Archive Rule

- The chronology below is the raw experiment log.
- Use the summary above to decide what to try next.
- If a new idea matches one of the failed themes above, assume it is wrong until new evidence proves otherwise.

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
  - CI targeted bench run: `22634499864` (`crates=hashes`, `benches=blake3`,
    `filter=blake3/xof/,blake3/streaming/`, `quick=false`,
    lanes: `amd-zen5`, `intel-icl`, `intel-spr`, `graviton3`, `graviton4`).
  - Scope/commit check:
    - workflow completed `success`,
    - executed commit `a780ad164e636ef81769ba685b80943f79e98c12` (expected SHA; valid run).
  - CI outcomes (time-based gap vs official; positive = slower):
    - Aggregate (`xof` + `streaming`, 5 lanes): `13W / 67L`, avg gap `+67.90%`.
    - `streaming/*`: `6W / 34L`, avg gap `+12.83%`.
    - `xof/*`: `7W / 33L`, avg gap `+122.98%`.
    - Lane aggregates:
      - `intel-icl`: `0W/16L`, avg `+120.59%` (`streaming +23.49%`, `xof +217.70%`).
      - `intel-spr`: `0W/16L`, avg `+109.21%` (`streaming +19.36%`, `xof +199.07%`).
      - `amd-zen5`: `2W/14L`, avg `+107.73%` (`streaming +22.61%`, `xof +192.85%`).
      - `graviton3`: `5W/11L`, avg `+0.85%` (`streaming -0.76%`, `xof +2.45%`).
      - `graviton4`: `6W/10L`, avg `+1.12%` (`streaming -0.56%`, `xof +2.81%`).
  - Notable regressions:
    - `intel-icl xof init+read/64B-in/1024B-out`: `+734.81%`.
    - `intel-icl xof init+read/1B-in/1024B-out`: `+732.21%`.
    - `amd-zen5 xof init+read/64B-in/1024B-out`: `+725.26%`.
    - `amd-zen5 xof init+read/1B-in/1024B-out`: `+721.32%`.
    - `intel-spr xof init+read/64B-in/1024B-out`: `+705.18%`.
    - `intel-spr xof init+read/1B-in/1024B-out`: `+673.96%`.
- Decision:
  - Reject and revert.

### 2026-03-03 - Candidate AQ (`working tree`)
- Hypothesis:
  - The remaining x86 XOF short-output gap is concentrated in first-block root-hash generation (`init+read/*-in/32B-out`), where we still pay generic compression path overhead.
  - A narrow, kernel-aware root-hash fast path should reduce first-block latency without policy churn or broad XOF state rewrites.
- Change:
  - `OutputState` root-hash path:
    - added `root_hash_words_with_kernel(kernel)` and `root_hash_bytes_with_kernel(kernel)`,
    - on `x86_64`, route SIMD kernels through `kernel.x86_compress_cv_bytes`,
    - on `aarch64` NEON, route through `compress_cv_neon_bytes`,
    - keep portable fallback unchanged.
  - `Blake3Xof::squeeze` first-block fast path (`out.len() <= 32`):
    - compute root hash via `root_hash_bytes_with_kernel(...)`,
    - on `x86_64`, if current root kernel is `Portable`, promote only this path to `dispatch_plan.stream_kernel()` when available.
  - No dispatch-table edits, no API changes, no streaming-path rewrites.
- Validation:
  - Local: `just check-all` passed.
  - Local: `just test` passed (`167/167`).
  - CI targeted bench run: `22636554814` (`crates=hashes`, `benches=blake3`,
    `filter=blake3/xof/,blake3/streaming/`, `quick=false`,
    lanes: `amd-zen5`, `intel-icl`, `intel-spr`).
  - Scope/commit check:
    - workflow completed `success`,
    - executed commit `bf58df99569450e3088650298e4947dc2140a4bf` (expected SHA; valid run).
  - CI outcomes (time-based gap vs official; positive = slower):
    - Aggregate (`xof` + `streaming`, 3 lanes): `0W / 48L`, avg gap `+23.19%`.
    - `streaming/*`: `0W / 24L`, avg gap `+21.63%`.
    - `xof/*`: `0W / 24L`, avg gap `+24.74%`.
    - Lane aggregates:
      - `intel-icl`: `0W/16L`, avg `+28.39%` (`streaming +23.61%`, `xof +33.17%`).
      - `intel-spr`: `0W/16L`, avg `+19.43%` (`streaming +18.59%`, `xof +20.28%`).
      - `amd-zen5`: `0W/16L`, avg `+21.74%` (`streaming +22.69%`, `xof +20.78%`).
  - Target-cluster check:
    - `xof init+read/*-in/32B-out`: `0W / 12L`, avg gap `+31.24%`.
    - `streaming 64..1024B chunks`: `0W / 15L`, avg gap `+23.91%`.
  - Directional delta vs strong prior targeted baseline (`22560608081`, same 3 lanes/scope):
    - aggregate: `+16.30%` -> `+23.19%` (`+6.89 pp`, worse),
    - streaming: `+13.44%` -> `+21.63%` (`+8.20 pp`, worse),
    - xof: `+19.15%` -> `+24.74%` (`+5.59 pp`, worse).
- Decision:
  - Reject and revert.

### 2026-03-03 - Candidate AR (`1107ec6`)
- Hypothesis:
  - Remaining `streaming` and `xof init+read` losses were dominated by control-path overhead in per-update chunk handling, not SIMD kernel throughput.
  - Removing short-update specialization (`try_update_block_aligned_short`) and simplifying `ChunkState::update` control flow should reduce fixed overhead in the hot `64..1024B` update surface.
- Change:
  - Removed `ChunkState::try_update_block_aligned_short`.
  - Simplified `ChunkState::update` generic path:
    - retained the existing aarch64 one-chunk fast path,
    - replaced the inner loop/continue structure with one bulk-compress step plus tail buffering,
    - removed the strict-subtract branch in block-count handling.
  - `Digest::update`:
    - removed `try_update_block_aligned_short` call from the single-chunk-fit fast path.
- Validation:
  - Local: `just check-all` passed.
  - Local: `just test` passed (`167/167`).
  - CI targeted bench run: `22637943751` (`crates=hashes`, `benches=blake3`,
    `filter=blake3/xof/,blake3/streaming/`, `quick=false`,
    lanes: `amd-zen5`, `intel-icl`, `intel-spr`).
  - Scope/commit check:
    - workflow completed `success`,
    - executed commit `1107ec6f009a9d329b2aa5c51ccbf19d6f9ea77e` (expected SHA; valid run).
  - CI outcomes (time-based gap vs official; positive = slower):
    - Aggregate (`xof` + `streaming`, 3 lanes): `2W / 46L`, avg gap `+24.85%`.
    - `streaming/*`: `0W / 24L`, avg gap `+22.86%`.
    - `xof/*`: `2W / 22L`, avg gap `+26.84%`.
    - Lane aggregates:
      - `intel-icl`: `0W/16L`, avg `+34.40%` (`streaming +23.31%`, `xof +45.49%`).
      - `intel-spr`: `0W/16L`, avg `+23.47%` (`streaming +23.01%`, `xof +23.93%`).
      - `amd-zen5`: `2W/14L`, avg `+16.69%` (`streaming +22.27%`, `xof +11.11%`).
  - Target-cluster check:
    - `xof init+read/*-in/32B-out`: `2W / 10L`, avg gap `+31.60%`.
    - `streaming 64..1024B chunks`: `0W / 15L`, avg gap `+25.75%`.
  - Directional delta vs strong prior targeted baseline (`22560608081`, same 3 lanes/scope):
    - aggregate: `+16.30%` -> `+24.85%` (`+8.55 pp`, worse),
    - streaming: `+13.44%` -> `+22.86%` (`+9.42 pp`, worse),
    - xof: `+19.15%` -> `+26.84%` (`+7.69 pp`, worse).
  - Notable regressions:
    - `intel-icl xof init+read/1B-in/32B-out`: `+101.26%`.
    - `intel-icl xof init+read/64B-in/32B-out`: `+97.09%`.
    - `intel-spr xof init+read/1B-in/32B-out`: `+47.59%`.
    - `amd-zen5 streaming/16384B-chunks`: `+32.83%`.
- Decision:
  - Reject and revert.

### 2026-03-03 - Candidate AS (`6de31a0`)
- Hypothesis:
  - The most reliable prior directional improvement came from x86 stream-kernel policy (`AVX-512` stream selection) rather than control-path rewrites.
  - Current first-update defer policy still pins x86 short streaming/XOF init paths to conservative kernels and inflates fixed cost.
- Change:
  - x86 dispatch-table policy:
    - set `PROFILE_X86_ZEN5`, `PROFILE_X86_INTEL_SPR`, and `PROFILE_X86_INTEL_ICL` streaming kernel to `X86Avx512`.
  - `Digest::update` policy (x86_64 only):
    - bypass first-update SIMD defer gating in the `input.len() <= CHUNK_LEN` path,
    - bypass short-chunk re-defer check when deciding whether to promote from `Portable` to tuned stream kernel.
  - No `Blake3Xof` state/model changes and no kernel implementation changes.
- Validation:
  - Local: `just check-all` passed.
  - Local: `just test` passed (`167/167`).
  - CI targeted bench run: `22638507919` (`crates=hashes`, `benches=blake3`,
    `filter=blake3/xof/,blake3/streaming/`, `quick=false`,
    lanes: `amd-zen5`, `intel-icl`, `intel-spr`).
  - Scope/commit check:
    - workflow completed `success`,
    - executed commit `6de31a0999444bd66cc481fc5cc38f91fb1c968c` (expected SHA; valid run).
  - CI outcomes (time-based gap vs official; positive = slower):
    - Aggregate (`xof` + `streaming`, 3 lanes): `2W / 46L`, avg gap `+18.98%`.
    - `streaming/*`: `0W / 24L`, avg gap `+16.08%`.
    - `xof/*`: `2W / 22L`, avg gap `+21.88%`.
    - Lane aggregates:
      - `intel-icl`: `0W/16L`, avg `+30.88%` (`streaming +23.29%`, `xof +38.48%`).
      - `intel-spr`: `0W/16L`, avg `+15.99%` (`streaming +11.63%`, `xof +20.36%`).
      - `amd-zen5`: `2W/14L`, avg `+10.06%` (`streaming +13.32%`, `xof +6.80%`).
  - Target-cluster check:
    - `xof init+read/*-in/32B-out`: `2W / 10L`, avg gap `+27.12%`.
    - `streaming 64..1024B chunks`: `0W / 15L`, avg gap `+16.50%`.
  - Directional delta vs strong prior targeted baseline (`22560608081`, same 3 lanes/scope):
    - aggregate: `+16.30%` -> `+18.98%` (`+2.68 pp`, worse),
    - streaming: `+13.44%` -> `+16.08%` (`+2.64 pp`, worse),
    - xof: `+19.15%` -> `+21.88%` (`+2.73 pp`, worse).
  - Directional delta vs prior candidate run (`22637943751`, Candidate AR):
    - aggregate: `+24.85%` -> `+18.98%` (`-5.87 pp`, better),
    - streaming: `+22.86%` -> `+16.08%` (`-6.78 pp`, better),
    - xof: `+26.84%` -> `+21.88%` (`-4.96 pp`, better).
  - Notable regressions:
    - `intel-icl xof init+read/1B-in/32B-out`: `+87.66%`.
    - `intel-icl xof init+read/64B-in/32B-out`: `+84.80%`.
    - `intel-spr xof init+read/1B-in/32B-out`: `+49.31%`.
  - Notable wins:
    - `amd-zen5 xof init+read/1B-in/32B-out`: `-14.44%`.
    - `amd-zen5 xof init+read/64B-in/32B-out`: `-21.33%`.
- Decision:
  - Reject and revert.
  - This is an improvement over AR, but it is still slower than the strong baseline on aggregate, streaming, and xof surfaces.

### 2026-03-03 - Candidate AT (`eb61536`)
- Hypothesis:
  - Tiny x86 first-update fast path (`input.len() <= 64`) was returning before stream-kernel lock-in, leaving single-update XOF/streaming calls pinned to `Portable`.
  - Forcing x86 stream-kernel lock-in in that ultra-tiny first-update branch should reduce fixed overhead on `xof init+read` (`1B/64B`) and short streaming chunks.
- Change:
  - `Digest::update` tiny first-update branch (`<= BLOCK_LEN`):
    - on `x86_64`, when `self.kernel == Portable`, set:
      - `self.kernel = dispatch_plan.stream_kernel()`,
      - `self.bulk_kernel = dispatch_plan.bulk_kernel_for_update(input.len())`,
      - `self.chunk_state.kernel = stream`.
    - keep existing block copy + early return behavior unchanged.
  - No dispatch-table edits, no XOF state/model rewrites, no kernel implementation changes.
- Validation:
  - Local: `just check-all` passed.
  - Local: `just test` passed (`167/167`).
  - CI targeted bench run: `22645562535` (`crates=hashes`, `benches=blake3`,
    `filter=blake3/xof/,blake3/streaming/`, `quick=false`,
    lanes: `amd-zen5`, `intel-icl`, `intel-spr`).
  - Scope/commit check:
    - workflow completed `success`,
    - executed commit `eb61536401be7d19433c0da87bcbd99578f790a8` (expected SHA; valid run).
- CI outcomes (time-based gap vs official; positive = slower):
  - Aggregate (`xof` + `streaming`, 3 lanes): `0W / 48L`, avg gap `+23.75%`.
  - `streaming/*`: `0W / 24L`, avg gap `+21.22%`.
  - `xof/*`: `0W / 24L`, avg gap `+26.29%`.
  - Lane aggregates:
    - `intel-icl`: `0W/16L`, avg `+28.19%` (`streaming +23.31%`, `xof +33.06%`).
    - `intel-spr`: `0W/16L`, avg `+20.26%` (`streaming +18.88%`, `xof +21.64%`).
    - `amd-zen5`: `0W/16L`, avg `+22.81%` (`streaming +21.45%`, `xof +24.17%`).
  - Target-cluster check:
    - `xof init+read/*-in/32B-out`: `0W / 12L`, avg gap `+36.50%`.
    - `streaming 64..1024B chunks`: `0W / 15L`, avg gap `+23.84%`.
  - Notable regressions:
    - `intel-icl xof init+read/1B-in/32B-out`: `+69.94%`.
    - `intel-icl xof init+read/64B-in/32B-out`: `+60.98%`.
    - `intel-spr xof init+read/1B-in/32B-out`: `+43.52%`.
    - `amd-zen5 xof init+read/1B-in/32B-out`: `+35.17%`.
- Decision:
  - Reject and revert.
  - The tiny first-update kernel lock-in did not close the target surfaces and produced an all-loss run.

### 2026-03-03 - Candidate AU (`6d508e6`)
- Hypothesis:
  - The remaining deficit is control-path architecture, not kernel throughput.
  - Matching upstream runtime shape for streaming/XOF hot paths (no portable-first defer gate, minimal XOF reader state) should reduce fixed overhead.
- Change:
  - Runtime-path simplification in `mod.rs`:
    - removed `Digest::update()` portable-first defer/tiny special branches; always dispatch through resolved `(stream, bulk)` pair.
    - removed `ChunkState::try_update_block_aligned_short`.
    - simplified XOF to upstream-style state (`output`, `block_counter`, `position_within_block`) and removed cache/upgrade state machine.
    - simplified `finalize_xof()` to direct `root_output()` and made `finalize_xof_sized()` a compatibility alias.
    - default `Blake3::new_internal()` now starts on `dispatch_plan.stream_kernel()` instead of forced portable.
  - Dispatch policy alignment:
    - `dispatch_tables.rs`: x86 default stream kernel `SSE4.1 -> AVX2`.
    - x86 lane profiles (`Zen4`, `Zen5`, `IntelIcl`, `IntelSpr`) stream kernel `SSE4.1 -> AVX512` (bulk unchanged).
  - Dispatch plumbing cleanup:
    - `dispatch::xof()` now constructs `Blake3Xof::new(output)` directly.
    - removed stale lazy-defer reporting path in `streaming_dispatch_info`.
- Validation:
  - Local: `just check` passed.
  - Local: `cargo test -p hashes blake3 --lib` passed (`20/20`).
  - CI targeted bench run: `22650809810` (`crates=hashes`, `benches=blake3`,
    `filter=blake3/xof/,blake3/streaming/`, `quick=false`,
    lanes: `amd-zen5`, `intel-icl`, `intel-spr`).
  - Scope/commit check:
    - workflow completed `success`,
    - executed commit `6d508e6a030582bb441bc1e7dbef78f8f660ae57` (expected SHA; valid run).
  - CI outcomes (time-based gap vs official; positive = slower):
    - Aggregate (`xof` + `streaming`, 3 lanes): `0W / 48L`, avg gap `+21.32%`.
    - `streaming/*`: `0W / 24L`, avg gap `+16.07%`.
    - `xof/*`: `0W / 24L`, avg gap `+26.56%`.
    - Lane aggregates:
      - `intel-icl`: `0W/16L`, avg `+23.88%` (`streaming +20.39%`, `xof +27.38%`).
      - `intel-spr`: `0W/16L`, avg `+22.66%` (`streaming +14.28%`, `xof +31.04%`).
      - `amd-zen5`: `0W/16L`, avg `+17.41%` (`streaming +13.55%`, `xof +21.27%`).
  - Target-cluster check:
    - `xof init+read/*-in/32B-out`: `0W / 12L`, avg gap `+29.88%`.
    - `streaming 64..1024B chunks`: `0W / 15L`, avg gap `+16.58%`.
  - Directional delta vs prior candidate (`22645562535`, Candidate AT):
    - aggregate: `+23.75%` -> `+21.32%` (`-2.43 pp`, better),
    - streaming: `+21.22%` -> `+16.07%` (`-5.15 pp`, better),
    - xof: `+26.29%` -> `+26.56%` (`+0.27 pp`, worse).
  - Directional delta vs strong prior targeted baseline (`22560608081`, same 3 lanes/scope):
    - aggregate: `+16.30%` -> `+21.32%` (`+5.02 pp`, worse),
    - streaming: `+13.44%` -> `+16.07%` (`+2.63 pp`, worse),
    - xof: `+19.15%` -> `+26.56%` (`+7.41 pp`, worse).
  - Notable regressions:
    - `intel-spr xof init+read/1B-in/32B-out`: `+62.21%`.
    - `intel-spr xof init+read/64B-in/32B-out`: `+55.76%`.
    - `intel-icl xof init+read/1B-in/32B-out`: `+45.68%`.
    - `intel-icl xof init+read/64B-in/32B-out`: `+39.86%`.
- Decision:
  - Reject and revert.
  - This pass reduced streaming gap materially but still produced an all-loss run and did not crack xof short-output overhead vs official.

### 2026-03-04 - Candidate AV (`abc1a2f`)
- Hypothesis:
  - Removing deferred last-full-chunk CV state (`pending_chunk_cv`) and simplifying terminal merge flow would reduce control-path overhead on streaming/XOF hot paths.
- Change:
  - Removed `pending_chunk_cv` and related helper/state handling in `mod.rs`.
  - Updated SIMD/parallel batch commit paths to consume only `commit` chunks and leave one full terminal chunk in `ChunkState` on exact-boundary inputs.
  - Simplified root merge path to official-style stack fold without `pending_chunk_cv` branch.
  - Kept minimal XOF reader shape (`output`, `block_counter`, `position_within_block`) and compatibility `finalize_xof_sized` alias.
- Validation:
  - Local: `cargo fmt --all` passed.
  - Local: `just check` passed.
  - Local: `cargo test -p hashes blake3 --lib` passed (`20/20`).
  - CI targeted bench run: `22652495044` (`crates=hashes`, `benches=blake3`,
    `filter=blake3/xof/,blake3/streaming/`, `quick=false`,
    lanes: `amd-zen5`, `intel-icl`, `intel-spr`).
  - Scope/commit check:
    - workflow completed `success`,
    - executed commit `abc1a2f07c21b249e469e1002c8f4c9e75f98891` (expected SHA; valid run).
  - CI outcomes (time-based gap vs official; positive = slower):
    - Aggregate (`xof` + `streaming`, 3 lanes): `0W / 48L`, avg gap `+48.17%`.
    - `streaming/*`: `0W / 24L`, avg gap `+60.81%`.
    - `xof/*`: `0W / 24L`, avg gap `+35.54%`.
    - Lane aggregates:
      - `intel-icl`: `0W/16L`, avg `+41.32%` (`streaming +50.49%`, `xof +32.15%`).
      - `intel-spr`: `0W/16L`, avg `+41.99%` (`streaming +52.31%`, `xof +31.68%`).
      - `amd-zen5`: `0W/16L`, avg `+61.20%` (`streaming +79.63%`, `xof +42.77%`).
  - Target-cluster check:
    - `xof init+read/*-in/32B-out`: `0W / 12L`, avg gap `+37.91%`.
    - `streaming 64..1024B chunks`: `0W / 15L`, avg gap `+12.87%`.
  - Directional delta vs prior candidate (`22650809810`, Candidate AU):
    - aggregate: `+21.32%` -> `+48.17%` (`+26.85 pp`, worse),
    - streaming: `+16.07%` -> `+60.81%` (`+44.74 pp`, much worse),
    - xof: `+26.56%` -> `+35.54%` (`+8.98 pp`, worse),
    - `xof 32B-out` cluster: `+29.88%` -> `+37.91%` (`+8.03 pp`, worse),
    - `streaming 64..1024B` cluster: `+16.58%` -> `+12.87%` (`-3.71 pp`, better).
  - Notable regressions:
    - `amd-zen5 streaming/16384B-chunks`: `+302.95%`.
    - `amd-zen5 streaming/4096B-chunks`: `+184.57%`.
    - `intel-spr streaming/4096B-chunks`: `+178.47%`.
    - `intel-icl streaming/4096B-chunks`: `+164.97%`.
    - `intel-icl streaming/16384B-chunks`: `+128.23%`.
- Decision:
  - Reject and revert.
  - Key inference: fully removing `pending_chunk_cv` is not viable; it improves the small streaming cluster but catastrophically regresses large-chunk streaming throughput and worsens XOF short-output competitiveness.

### 2026-03-04 - Candidate AW (`4097712`)
- Hypothesis:
  - XOF short first reads were paying unnecessary fixed cost by materializing a full 64-byte output block even when the caller requests at most 32 bytes.
  - A narrow `squeeze()` fast path for the first `<=32` bytes should improve `xof init+read/*-in/32B-out` without perturbing large streaming throughput.
- Change:
  - In `Blake3Xof::squeeze`:
    - added a first-read fast path when `block_counter == 0`, `position_within_block == 0`, and `out.len() <= OUT_LEN`,
    - returns bytes directly from `output.root_hash_bytes()` and advances `position_within_block`,
    - bypasses first-block materialization for this narrow case.
  - No dispatch-table changes and no update/streaming pipeline changes.
- Validation:
  - Local: `cargo fmt --all` passed.
  - Local: `cargo test -p hashes blake3 --lib` passed (`20/20`).
  - CI targeted bench run: `22654769160` (`crates=hashes`, `benches=blake3`,
    `filter=blake3/xof/,blake3/streaming/`, `quick=false`,
    lanes: `amd-zen5`, `intel-icl`, `intel-spr`).
  - Scope/commit check:
    - workflow completed `success`,
    - executed commit `409771272c712c8799fd4dea64475f0474e58052` (expected SHA; valid run).
  - CI outcomes (time-based gap vs official; positive = slower):
    - Aggregate (`xof` + `streaming`, 3 lanes): `0W / 48L`, avg gap `+19.79%`.
    - `streaming/*`: `0W / 24L`, avg gap `+13.46%`.
    - `xof/*`: `0W / 24L`, avg gap `+26.11%`.
    - Lane aggregates:
      - `intel-icl`: `0W/16L`, avg `+20.05%` (`streaming +13.78%`, `xof +26.31%`).
      - `intel-spr`: `0W/16L`, avg `+21.82%` (`streaming +13.53%`, `xof +30.11%`).
      - `amd-zen5`: `0W/16L`, avg `+17.49%` (`streaming +13.07%`, `xof +21.92%`).
  - Target-cluster check:
    - `xof init+read/*-in/32B-out`: `0W / 12L`, avg gap `+30.51%`.
    - `streaming 64..1024B chunks`: `0W / 15L`, avg gap `+12.37%`.
  - Directional delta vs prior candidate (`22652495044`, Candidate AV):
    - aggregate: `+48.17%` -> `+19.79%` (`-28.38 pp`, better),
    - streaming: `+60.81%` -> `+13.46%` (`-47.35 pp`, better),
    - xof: `+35.54%` -> `+26.11%` (`-9.43 pp`, better).
  - Directional delta vs prior stable structural candidate (`22650809810`, Candidate AU):
    - aggregate: `+21.32%` -> `+19.79%` (`-1.53 pp`, better),
    - streaming: `+16.07%` -> `+13.46%` (`-2.61 pp`, better),
    - xof: `+26.56%` -> `+26.11%` (`-0.45 pp`, better),
    - `xof 32B-out` cluster: `+29.88%` -> `+30.51%` (`+0.63 pp`, worse),
    - `streaming 64..1024B` cluster: `+16.58%` -> `+12.37%` (`-4.21 pp`, better).
  - Notable regressions:
    - `intel-spr xof init+read/1B-in/32B-out`: `+65.50%`.
    - `intel-spr xof init+read/64B-in/32B-out`: `+53.01%`.
    - `intel-icl xof init+read/1B-in/32B-out`: `+49.60%`.
    - `intel-icl xof init+read/64B-in/32B-out`: `+41.55%`.
- Decision:
  - Reject and revert.
  - This pass materially improves aggregate/streaming vs recent candidates, but remains all-loss and does not solve the primary xof short-output deficit on Intel lanes.

### 2026-03-04 - Candidate AX (`6ba2b95`)
- Hypothesis:
  - Intel short-output XOF losses were potentially AVX-512 stream warmup/fixed-cost driven.
  - For `intel-icl` and `intel-spr`, setting streaming `stream` kernel to `AVX2` (while keeping `bulk=AVX512`) could reduce fixed cost on `xof init+read`.
- Change:
  - Dispatch policy only (`dispatch_tables.rs`):
    - `PROFILE_X86_INTEL_SPR.streaming.stream`: `X86Avx512 -> X86Avx2`.
    - `PROFILE_X86_INTEL_ICL.streaming.stream`: `X86Avx512 -> X86Avx2`.
    - `bulk` remained `X86Avx512` for both.
  - No update/XOF state-machine changes.
- Validation:
  - Local: `cargo test -p hashes blake3 --lib` passed (`20/20`).
  - CI targeted bench run: `22677938256` (`crates=hashes`, `benches=blake3`,
    `filter=blake3/xof/,blake3/streaming/`, `quick=false`,
    lanes: `amd-zen5`, `intel-icl`, `intel-spr`).
  - Scope/commit check:
    - workflow completed `success`,
    - executed commit `6ba2b95ce63a065906b6657c6979f75674c6fe28` (expected SHA; valid run).
  - CI outcomes (time-based gap vs official; positive = slower):
    - Aggregate (`xof` + `streaming`, 3 lanes): `0W / 48L`, avg gap `+37.00%`.
    - `streaming/*`: `0W / 24L`, avg gap `+22.33%`.
    - `xof/*`: `0W / 24L`, avg gap `+51.67%`.
    - Lane aggregates:
      - `intel-icl`: `0W/16L`, avg `+45.68%` (`streaming +26.69%`, `xof +64.67%`).
      - `intel-spr`: `0W/16L`, avg `+48.48%` (`streaming +27.15%`, `xof +69.82%`).
      - `amd-zen5`: `0W/16L`, avg `+16.83%` (`streaming +13.15%`, `xof +20.51%`).
  - Target-cluster check:
    - `xof init+read/*-in/32B-out`: `0W / 12L`, avg gap `+44.91%`.
    - `streaming 64..1024B chunks`: `0W / 15L`, avg gap `+26.37%`.
  - Directional delta vs prior candidate (`22654769160`, Candidate AW):
    - aggregate: `+19.79%` -> `+37.00%` (`+17.21 pp`, worse),
    - streaming: `+13.46%` -> `+22.33%` (`+8.87 pp`, worse),
    - xof: `+26.11%` -> `+51.67%` (`+25.56 pp`, much worse),
    - `xof 32B-out` cluster: `+30.51%` -> `+44.91%` (`+14.40 pp`, worse).
  - Notable regressions:
    - `intel-spr xof init+read/64B-in/1024B-out`: `+144.02%`.
    - `intel-spr xof init+read/1B-in/1024B-out`: `+129.24%`.
    - `intel-icl xof init+read/64B-in/1024B-out`: `+111.57%`.
    - `intel-icl xof init+read/1B-in/1024B-out`: `+103.37%`.
    - `intel-spr xof init+read/1B-in/32B-out`: `+93.77%`.
    - `intel-icl xof init+read/1B-in/32B-out`: `+92.92%`.
- Decision:
  - Reject and revert.
  - Intel stream-kernel downtier (`AVX512 -> AVX2`) is decisively the wrong direction for this benchmark surface.

### 2026-03-05 - Candidate AY (`f4fc8e9`)
- Hypothesis:
  - Constructor-time dispatch lookup/ref handling was still adding fixed cost to hot `xof`/`streaming` paths.
  - Caching a shared hasher dispatch-plan reference in `Blake3` construction paths could reduce `new -> update -> finalize/fill` fixed overhead.
- Change:
  - `Blake3` stored a shared `&'static dispatch::HasherDispatch` and reused it in constructor paths.
  - Added static dispatch-plan helper in `dispatch.rs` and wired call sites in `mod.rs`.
- Validation:
  - CI targeted bench run: `22734116057` (`crates=hashes`, `benches=blake3`,
    `filter=blake3/xof-phase/,blake3/xof/,blake3/streaming/`, `quick=false`,
    lanes: `amd-zen5`, `intel-icl`, `intel-spr`).
  - Scope/commit check:
    - executed commit `f4fc8e96cd2f78f7c8a52e3416b64f3d0e1bc4ab` (expected SHA; valid run).
  - Infra status:
    - `intel-icl` runner was stuck in setup (GitHub Actions infra; no usable artifact).
    - `amd-zen5` and `intel-spr` completed successfully.
- CI outcomes on completed lanes (time-based gap vs official; positive = slower):
  - Target surfaces (`xof` + `streaming`):
    - `amd-zen5`: `0W / 16L`, avg gap `+16.09%`.
    - `intel-spr`: `0W / 16L`, avg gap `+18.97%`.
    - combined completed lanes: `0W / 32L`, avg gap `+17.53%`.
  - Target clusters:
    - `xof init+read/*-in/32B-out`:
      - `amd-zen5`: avg `+17.59%`,
      - `intel-spr`: avg `+25.98%`.
    - `streaming 64..1024B chunks`:
      - `amd-zen5`: avg `+9.55%`,
      - `intel-spr`: avg `+15.55%`.
  - `xof-phase` attribution signals:
    - `finalize-xof-only - drop-after-update-only` remained dramatically slower than official (large fixed finalize/XOF object cost signal).
    - `squeeze-32-only` remained materially slower, while `squeeze-1024-only` was near parity/slightly better on SPR (short-read control-path overhead signal).
- Decision:
  - Reject and revert.
  - Reverted by `3b5f51d`.
  - Constructor dispatch-reference optimization does not solve the streaming/xof deficit.

## Anti-Repeat Guardrails (2026-03-05)

- Do not repeat candidate AV-style full `pending_chunk_cv` removal as-is (`abc1a2f`): it catastrophically regressed large-chunk streaming.
- Do not repeat candidate AX-style Intel stream-kernel downtier (`6ba2b95`): it decisively worsened Intel xof/streaming.
- Do not run constructor-only tuning again as a standalone strategy (`f4fc8e9`): no wins on target surfaces.
- Do not retry first-read root-hash shortcut alone (`4097712`): it improved aggregate trend but did not solve short-output xof losses.

## Active Next 1/2/3 Plan (Post-AY)

1. Stop kernel micro-tuning for this pass.
2. Align runtime model to official where current evidence points to fixed-cost overhead:
   - reduce exact-boundary finalize tree-walk overhead,
   - reduce short XOF first-block/small-read control overhead.
3. Re-run the same targeted CI scope immediately after this structural simplification.

### 2026-03-05 - Candidate AZ (`2408197`)
- Hypothesis:
  - Exact-boundary finalize overhead was inflated by retaining only a single pending leaf CV, forcing extra parent-chain reconstruction.
  - Short XOF reads were paying avoidable control overhead by routing one-block extraction through the generic multi-block path.
- Change:
  - `mod.rs` runtime-structure update (no dispatch-table/kernel tuning):
    - added aligned right-edge pending subtree retention (`pending_subtree_chunks`) instead of always retaining one chunk CV.
    - added subtree-aware pending commit logic on next `update()`.
    - added single-block XOF emission helper (`root_output_block`) and switched `Blake3Xof::fill_one_block` to use it.
  - No kernel implementation changes, no dispatch policy edits.
- Validation:
  - Local: `cargo fmt --all` passed.
  - Local: `cargo test -p hashes blake3 --lib` passed (`20/20`).
  - Local: `cargo bench -p hashes --bench blake3 --no-run` passed.
  - CI targeted bench run: `22741769225` (`crates=hashes`, `benches=blake3`,
    `filter=blake3/xof-phase/,blake3/xof/,blake3/streaming/`, `quick=false`,
    lanes: `amd-zen5`, `intel-icl`, `intel-spr`).
  - Scope/commit check:
    - workflow completed `success`,
    - executed commit `2408197c8ad135d3ef9480add26307686e9393bf` (expected SHA; valid run).
- CI outcomes (time-based gap vs official; positive = slower):
  - Aggregate (`xof` + `streaming`, 3 lanes): `0W / 48L`, avg gap `+18.56%`.
  - `streaming/*`: `0W / 24L`, avg gap `+10.90%`.
  - `xof/*`: `0W / 24L`, avg gap `+26.21%`.
  - Lane aggregates:
    - `amd-zen5`: `0W/16L`, avg `+14.61%` (`streaming +8.97%`, `xof +20.25%`).
    - `intel-icl`: `0W/16L`, avg `+19.18%` (`streaming +11.09%`, `xof +27.28%`).
    - `intel-spr`: `0W/16L`, avg `+21.87%` (`streaming +12.63%`, `xof +31.11%`).
  - Target-cluster check:
    - `xof init+read/*-in/32B-out`: `0W / 12L`, avg gap `+30.68%`.
    - `streaming 64..1024B chunks`: `0W / 15L`, avg gap `+12.94%`.
  - Attribution signal:
    - `xof-phase/finalize-xof-only/65536B-in` improved materially vs AY-era shape (finalize-minus-drop gaps dropped from multi-hundred ns to double-digit ns), but this did not translate into wins on `xof init+read`.
    - `xof` short-output paths (`1B/64B -> 32B-out`) remained heavily behind on Intel lanes.
- Decision:
  - Reject and revert.
  - The candidate improves one internal hotspot (exact-boundary finalize) but remains all-loss on targeted surfaces and does not close short-output XOF competitiveness.

### 2026-03-06 - Candidate BA (`539d0de`, fixup `ab1bf11`)
- Hypothesis:
  - Short XOF reads still spend too much time materializing a one-block output through the generic multi-block path.
  - Small streaming updates that remain within the current chunk still pay avoidable `update_with` admission overhead.
- Change:
  - `mod.rs` only:
    - added `OutputState::root_output_block(output_block_counter)` for direct single-block XOF emission,
    - switched `Blake3Xof::fill_one_block` to use that helper instead of going through `root_output_blocks_into`,
    - restored a narrow `Digest::update` fast path for inputs that fit in the current chunk and have no pending subtree/CV state.
  - Follow-up fixup commit `ab1bf11` moved `// SAFETY:` comments to the exact locations Clippy requires.
  - No dispatch-table changes, no kernel selection retuning, no API changes.
- Validation:
  - Local: `cargo fmt --all` passed.
  - Local: `cargo test -p hashes blake3 --lib` passed (`20/20`).
  - Local: `cargo bench -p hashes --bench blake3 --no-run` passed.
  - Local quick signal (Apple M1): slight movement in the right direction for `streaming/64B-chunks`, but short XOF phase remained mixed and not decision-grade.
  - CI targeted bench run: `22780800112` (`crates=hashes`, `benches=blake3`,
    `filter=blake3/xof-phase/,blake3/xof/,blake3/streaming/`, `quick=false`,
    lanes: `amd-zen5`, `intel-icl`, `intel-spr`, `graviton3`, `graviton4`).
  - Scope/commit check:
    - workflow completed `success`,
    - executed commit `539d0dec80ed6fd2ffd7627d652d07be9cc79e07` (valid run).
- CI outcomes (time-based gap vs official; positive = slower):
  - Aggregate target API surfaces (`xof` + `streaming`, 5 lanes): `10W / 70L`, avg gap `+10.51%`.
  - `streaming/*`: `6W / 34L`, avg gap `+8.53%`.
  - `xof/*`: `4W / 36L`, avg gap `+12.49%`.
  - Lane aggregates (`xof` + `streaming`):
    - `amd-zen5`: `0W / 16L`, avg `+15.37%`.
    - `intel-icl`: `0W / 16L`, avg `+14.25%`.
    - `intel-spr`: `0W / 16L`, avg `+17.09%`.
    - `graviton3`: `5W / 11L`, avg `+3.31%`.
    - `graviton4`: `5W / 11L`, avg `+2.52%`.
  - Target-cluster check:
    - `xof init+read/*-in/32B-out`: `2W / 18L`, avg gap `+15.42%`.
    - `streaming 64..1024B chunks`: `0W / 25L`, avg gap `+9.72%`.
  - Attribution signal:
    - `xof-phase` remained dominated by fixed-cost reader/setup overhead on x86.
    - The direct one-block helper did not materially close `squeeze-32-only`, and the restored in-chunk update fast path did not convert the short streaming cluster.
- Decision:
  - Reject and revert.
  - This candidate is directionally less bad than several earlier failures, but it is still not competitive where we need it to be.
  - The decisive blocker is unchanged: x86 lanes remain all-loss, short streaming is still `0W / 25L`, and short-output XOF remains overwhelmingly red.

### 2026-03-06 - Candidate BB (`ea0734c`, run `22789997965`)
- Hypothesis:
  - Our XOF reader/runtime shape was still too heavy compared with upstream.
  - Moving XOF ownership onto a compact output object with its own counter, and materializing raw block bytes once, would reduce short-read overhead and improve `xof/streaming` without touching oneshot paths.
- Change:
  - `mod.rs` structural rewrite:
    - introduced `XofOutputState` as a compact XOF/output holder,
    - converted `Blake3Xof` to own that compact state directly,
    - moved XOF counter ownership into the reader,
    - routed short reads through `root_output_block()` and full-block reads through the compact bulk emitter,
    - removed the old generic `OutputState` XOF block emission path from the hot reader flow.
  - No dispatch-table changes and no kernel retuning.
- Validation:
  - Local: `just check-all` passed after fixing unrelated CRC64 aarch64 target-feature contracts and Blake3 clippy cleanup.
  - Local: `just test` passed (`517/517`).
  - CI targeted bench run: `22789997965` (`crates=hashes`, `benches=blake3`,
    `filter=blake3/xof-phase/,blake3/xof/,blake3/streaming/`, `quick=false`,
    lanes: `amd-zen5`, `intel-icl`, `intel-spr`, `graviton3`, `graviton4`).
  - Scope/commit check:
    - workflow completed `success`,
    - executed the intended `main` candidate state (valid run).
- CI outcomes (time-based gap vs official; positive = slower):
  - Aggregate scoped target surfaces (`xof-phase` + `xof` + `streaming`, 5 lanes): `12W / 188L`, avg gap `+62.96%`.
  - `streaming/*`: `6W / 34L`, avg gap `+8.86%`.
  - `xof/*`: `4W / 36L`, avg gap `+18.71%`.
  - `xof-phase/*`: `2W / 118L`, avg gap `+95.74%`.
  - Target clusters:
    - `streaming 64..1024B chunks`: `0W / 25L`, avg gap `+10.22%`.
    - `xof init+read/*-in/32B-out`: `2W / 18L`, avg gap `+25.95%`.
    - `xof init+read/*-in/1024B-out`: `2W / 18L`, avg gap `+11.46%`.
  - XOF phase attribution:
    - `finalize-xof-only`: `0W / 20L`, avg gap `+333.10%`.
    - `squeeze-32-only`: `0W / 20L`, avg gap `+36.42%`.
    - `squeeze-1024-only`: `0W / 20L`, avg gap `+8.04%`.
    - `new-only`: `0W / 20L`, avg gap `+141.52%`.
    - `drop-after-update-only`: `0W / 20L`, avg gap `+20.67%`.
  - Notable regressions:
    - `finalize-xof-only/65536B-in` blew up on every lane:
      - `amd-zen5`: `+1622.84%`
      - `graviton3`: `+1190.37%`
      - `intel-icl`: `+1064.24%`
      - `graviton4`: `+988.53%`
      - `intel-spr`: `+797.11%`
- Decision:
  - Reject and revert.
  - This is the clearest evidence yet that finalize-time XOF state materialization is the wrong direction.
  - We did not remove fixed cost; we relocated and amplified it, especially in `finalize_xof()`.
  - The target failure remains unchanged where it matters: x86 short streaming is still all-loss, short-output XOF is still overwhelmingly red, and `xof-phase` got materially worse.

## Anti-Repeat Guardrails (2026-03-06)

- Do not repeat finalize-time compact XOF/output materialization as a strategy (`Candidate BB`): it massively regresses `finalize-xof-only` and does not convert short-read XOF wins.

### 2026-03-06 - Candidate BC (`1e5778d`, run `22791127078`)
- Hypothesis:
  - We were still paying too much fixed cost in the hot hasher object and on the single-chunk `finalize_xof()` path.
  - Shrinking `Blake3` state and giving `finalize_xof()` a lazy single-chunk reader path would cut `new-only`, short streaming, and short-output XOF overhead without repeating finalize-time XOF materialization.
- Change:
  - `mod.rs`:
    - removed persistent `bulk_kernel` from `Blake3` to shrink hot-state footprint,
    - added a `SingleChunkXofState` / `Blake3XofInner` split,
    - made `finalize_xof()` return a direct single-chunk reader when there is no tree state,
    - deferred promotion to generic `OutputState` until the reader actually needed the generic path,
    - avoided bulk-kernel selection on obviously in-chunk `update()` calls.
  - Support cleanup:
    - updated `kernel_test.rs` and `kernels.rs` for the removed field and cfg-clean builds.
- Validation:
  - Local: `cargo fmt --all` passed.
  - Local: `cargo test -p hashes blake3 --lib` passed.
  - Local: `cargo bench -p hashes --bench blake3 --no-run` passed.
  - Local: `cargo clippy -p hashes --lib --tests -- -D warnings` passed.
  - Local: `just check-all` passed.
  - Local: `just test` passed.
  - CI targeted bench run: `22791127078` (`crates=hashes`, `benches=blake3`,
    `filter=blake3/xof-phase/,blake3/xof/,blake3/streaming/`, `quick=false`,
    lanes: `amd-zen5`, `intel-icl`, `intel-spr`, `graviton3`, `graviton4`).
  - Scope/commit check:
    - workflow completed `success`,
    - executed the intended candidate commit (valid run).
- CI outcomes (time-based gap vs official; positive = slower):
  - Aggregate scoped target surfaces (`xof-phase` + `xof` + `streaming`, 5 lanes): `28W / 172L`, avg gap `+49.75%`.
  - `streaming/*`: `6W / 34L`, avg gap `+7.98%`.
  - `xof/*`: `10W / 30L`, avg gap `+5.43%`.
  - `xof-phase/*`: `12W / 108L`, avg gap `+78.45%`.
  - Target clusters:
    - `streaming 64..1024B chunks`: `0W / 25L`, avg gap `+8.93%`.
    - `xof init+read/*-in/32B-out`: `5W / 15L`, avg gap `+6.43%`.
    - `xof init+read/*-in/1024B-out`: `5W / 15L`, avg gap `+4.44%`.
  - XOF phase attribution:
    - `finalize-xof-only`: `0W / 20L`, avg gap `+308.91%`.
    - `squeeze-32-only`: `0W / 20L`, avg gap `+26.17%`.
    - `squeeze-1024-only`: `10W / 10L`, avg gap `-1.30%` (win).
    - `new-only`: `0W / 20L`, avg gap `+112.83%`.
    - `update-only`: `2W / 18L`, avg gap `+10.18%`.
    - `drop-after-update-only`: `0W / 20L`, avg gap `+11.30%`.
  - Notable behavior:
    - `xof` improved meaningfully versus `Candidate BB`, and `squeeze-1024-only` flipped fully green.
    - But the decisive blockers remained:
      - short streaming stayed all-loss,
      - short-output XOF stayed net-loss,
      - `finalize-xof-only` remained catastrophically slower on every lane,
      - `new-only` stayed heavily red.
- Decision:
  - Reject and revert.
  - This candidate proved the lazy single-chunk reader idea has some value on longer XOF reads, but it does not solve the actual problem.
  - The architecture is still paying too much fixed setup cost before any useful streaming/XOF work happens.

## Anti-Repeat Guardrails (2026-03-06, updated)

- Do not repeat finalize-time compact XOF/output materialization as a strategy (`Candidate BB`): it massively regresses `finalize-xof-only` and does not convert short-read XOF wins.
- Do not treat lazy single-chunk XOF reader promotion alone as the fix (`Candidate BC`): it helps long XOF squeezes, but short streaming, `new-only`, and `finalize-xof-only` remain structurally too expensive.

### 2026-03-07 - Candidate BD (`04ab0e8`, run `22800786046`)
- Hypothesis:
  - We were still carrying too much hot setup state per hasher, and paying bulk-dispatch overhead even on small streaming updates.
  - Replacing full `Kernel` bundles in hot state with compact kernel IDs, removing persistent `bulk_kernel`, and localizing bulk dispatch to large updates would reduce `new-only`, `update-only`, and short XOF setup cost without touching the reader model.
- Change:
  - `mod.rs`:
    - changed `Blake3`, `ChunkState`, and `OutputState` hot state to store `Blake3KernelId` where only selection was needed,
    - removed persistent `bulk_kernel` from `Blake3`,
    - threaded `bulk_kernel` only through the large update batching paths that actually use it,
    - avoided bulk-kernel selection entirely for obviously in-chunk `update()` calls.
  - `kernel_test.rs`:
    - updated forced-kernel helpers to match the leaner state model.
- Validation:
  - Local: `cargo fmt --all` passed.
  - Local: `cargo test -p hashes blake3 --lib` passed.
  - Local: `cargo clippy -p hashes --lib --tests -- -D warnings` passed.
  - Local: `cargo bench -p hashes --bench blake3 --no-run` passed.
  - Local: `just check-all` passed.
  - Local: `just test` passed.
  - CI targeted bench run: `22800786046` (`crates=hashes`, `benches=blake3`,
    `filter=blake3/xof-phase/,blake3/xof/,blake3/streaming/`, `quick=false`,
    lanes: `amd-zen5`, `intel-icl`, `intel-spr`, `graviton3`, `graviton4`).
  - Scope/commit check:
    - workflow completed `success`,
    - executed the intended candidate commit (valid run).
- CI outcomes (time-based gap vs official; positive = slower):
  - Aggregate scoped target surfaces (`xof-phase` + `xof` + `streaming`, 5 lanes): `37W / 163L`, avg gap `+44.98%`.
  - `streaming/*`: `6W / 34L`, avg gap `+8.19%`.
  - `xof/*`: `10W / 30L`, avg gap `+4.75%`.
  - `xof-phase/*`: `21W / 99L`, avg gap `+70.65%`.
  - Target clusters:
    - `streaming 64..1024B chunks`: `0W / 25L`, avg gap `+9.24%`.
    - `xof init+read/*-in/32B-out`: `5W / 15L`, avg gap `+4.71%`.
    - `xof init+read/*-in/1024B-out`: `5W / 15L`, avg gap `+4.79%`.
  - XOF phase attribution:
    - `new-only`: `0W / 20L`, avg gap `+95.38%`.
    - `update-only`: `5W / 15L`, avg gap `+4.27%`.
    - `finalize-xof-only`: `0W / 20L`, avg gap `+293.43%`.
    - `squeeze-32-only`: `4W / 16L`, avg gap `+19.45%`.
    - `squeeze-1024-only`: `12W / 8L`, avg gap `+1.53%`.
    - `drop-after-update-only`: `0W / 20L`, avg gap `+9.89%`.
  - Notable behavior:
    - This was the best setup-focused candidate so far on `new-only` and `update-only`; it materially reduced those gaps versus `Candidate BC`.
    - It also improved short-output XOF somewhat versus `Candidate BC`.
    - But the decisive blockers remained:
      - short streaming stayed all-loss,
      - short-output XOF stayed net-loss,
      - `finalize-xof-only` remained catastrophically slower on every lane,
      - x86 still did not convert the target cluster.
- Decision:
  - Reject and revert.
  - Shrinking hot state and localizing bulk dispatch is directionally right, but by itself it is not enough.
  - The dominant remaining architectural failure is still `finalize_xof()` setup cost, with short streaming also paying more fixed control tax than upstream.

## Anti-Repeat Guardrails (2026-03-07, updated)

- Do not repeat finalize-time compact XOF/output materialization as a strategy (`Candidate BB`): it massively regresses `finalize-xof-only` and does not convert short-read XOF wins.
- Do not treat lazy single-chunk XOF reader promotion alone as the fix (`Candidate BC`): it helps long XOF squeezes, but short streaming, `new-only`, and `finalize-xof-only` remain structurally too expensive.
- Do not expect hot-state shrinking and localized bulk dispatch alone to break through (`Candidate BD`): it helps `new-only`, `update-only`, and some XOF setup cost, but it does not fix `finalize_xof()` or short streaming.

## 2026-03-07 Root-Cause Reset

- We have been over-reading `xof-phase/update-only`.
  - That benchmark measures one `update()` call.
  - The losing `blake3/streaming/64..1024B-chunks` surfaces measure thousands of repeated `update()` calls across 1 MiB.
  - Better one-call `update-only` results do not prove repeated short streaming is fixed.
- The kernels are not the core problem.
  - `oneshot` is already strong.
  - The persistent deficits are fixed-cost serial-path overhead:
    - repeated short streaming updates,
    - `finalize_xof()` setup,
    - short-output XOF (`32B-out`).
- The most recent failures say exactly what not to do:
  - do not move more work into `finalize_xof()`,
  - do not rely on XOF reader helper tweaks,
  - do not use dispatch-table churn as the primary strategy,
  - do not assume a more upstream-looking generic `OutputState` is automatically a win.

### 2026-03-07 - Candidate BE (`d5374db`, run `22802604951`)
- Hypothesis:
  - The remaining wall was serial-path setup cost.
  - Making generic `OutputState` raw-byte based, adding a direct single-block XOF path, and adding a dedicated short serial `update()` path would finally close short streaming/XOF.
- Change:
  - `mod.rs`:
    - rewrote generic `OutputState` around raw block bytes,
    - routed `fill_one_block()` through a direct single-block root-output path,
    - added a dedicated short serial `update()` fast path.
  - `benches/blake3.rs`:
    - added repeated-update / repeated-finalize-XOF diagnostics.
  - `blake3_update.md`:
    - added this root-cause reset section.
- Validation:
  - Local: `cargo fmt --all` passed.
  - Local: `cargo test -p hashes blake3 --lib` passed.
  - Local: `cargo clippy -p hashes --lib --tests -- -D warnings` passed.
  - Local: `cargo bench -p hashes --bench blake3 --no-run` passed.
  - Local: `just check-all` passed.
  - Local: `just test` passed.
  - CI targeted bench run: `22802604951` (`crates=hashes`, `benches=blake3`,
    `filter=blake3/xof-phase/,blake3/xof/,blake3/streaming/`, `quick=false`,
    lanes: `amd-zen5`, `intel-icl`, `intel-spr`, `graviton3`, `graviton4`).
- CI outcomes (time-based gap vs official; positive = slower):
  - Aggregate scoped target surfaces (`xof-phase` + `xof` + `streaming`, 5 lanes): `12W / 188L`, avg gap `+57.15%`.
  - `streaming/*`: `6W / 34L`, avg gap `+7.96%`.
  - `xof/*`: `4W / 36L`, avg gap `+11.80%`.
  - `xof-phase/*`: `2W / 118L`, avg gap `+88.66%`.
  - Target clusters:
    - `streaming 64..1024B chunks`: `0W / 25L`, avg gap `+8.98%`.
    - `xof init+read/*-in/32B-out`: `2W / 18L`, avg gap `+14.85%`.
    - `xof init+read/*-in/1024B-out`: `2W / 18L`, avg gap `+8.75%`.
  - XOF phase attribution:
    - `new-only`: `0W / 20L`, avg gap `+122.80%`.
    - `update-only`: `2W / 18L`, avg gap `+26.29%`.
    - `finalize-xof-only`: `0W / 20L`, avg gap `+326.57%`.
    - `squeeze-32-only`: `0W / 20L`, avg gap `+28.94%`.
    - `squeeze-1024-only`: `0W / 20L`, avg gap `+6.02%`.
    - `drop-after-update-only`: `0W / 20L`, avg gap `+21.33%`.
- Decision:
  - Reject and revert.
  - The repeated-update diagnostics were the right investigative addition.
  - The raw-byte generic `OutputState` rewrite was the wrong production change.
  - This candidate made the hot setup path worse and did not convert any of the real target clusters.

## Anti-Repeat Guardrails (2026-03-07, final)

- Do not repeat finalize-time compact XOF/output materialization as a strategy (`Candidate BB`).
- Do not treat lazy single-chunk XOF reader promotion alone as the fix (`Candidate BC`).
- Do not expect hot-state shrinking and localized bulk dispatch alone to break through (`Candidate BD`).
- Do not rewrite the generic `OutputState` around raw bytes as a standalone fix (`Candidate BE`).
- Do not assume an upstream-literal serial streaming/XOF rewrite alone is sufficient (`Candidate BF`).

### 2026-03-07 - Candidate BF (`675f96a`, run `22805525301`)
- Hypothesis:
  - The current hybrid serial engine was the real blocker.
  - Replacing only the serial `new -> update -> finalize -> finalize_xof -> reader` path with an upstream-literal state machine would finally convert:
    - repeated short streaming updates,
    - `finalize_xof()` setup,
    - short-output XOF (`32B-out`).
- Change:
  - `mod.rs`:
    - introduced a separate internal `SerialHasherState` with:
      - `key_words`,
      - `chunk_state`,
      - lazy `cv_stack`,
      - `flags`,
      - no `pending_chunk_cv`,
    - rewrote serial `update_with()` around upstream-style flow:
      - partial chunk first,
      - whole-chunk handling next,
      - final chunk state last,
    - rewrote final fold to upstream-style `final_output()` logic,
    - kept the existing oneshot path, kernels, public API, and parallel/oneshot machinery intact,
    - kept XOF reader minimal on top of `OutputState`.
  - `kernel_test.rs`:
    - updated forced-kernel helpers to point at the new serial state.
  - `mod.rs` tests:
    - added repeated-short-update digest/XOF-32 regression coverage.
- Validation:
  - Local: `just check-all` passed.
  - Local: `just test` passed.
  - CI targeted bench run: `22805525301` (`crates=hashes`, `benches=blake3`,
    `filter=blake3/streaming/,blake3/xof/,blake3/xof-phase/`, `quick=false`,
    lanes: `amd-zen5`, `intel-icl`, `intel-spr`, `graviton3`, `graviton4`).
- CI outcomes (time-based gap vs official; positive = slower):
  - Aggregate scoped target surfaces (`streaming` + `xof` + `xof-phase`, 5 lanes): `20W / 180L`, avg gap `+62.97%`.
  - `streaming/*`: `2W / 38L`, avg gap `+25.92%`.
  - `xof/*`: `4W / 36L`, avg gap `+21.90%`.
  - `xof-phase/*`: `14W / 106L`, avg gap `+89.01%`.
  - Target clusters:
    - `streaming 64..1024B chunks`: `0W / 25L`
      - `64B`: `+5.30%` to `+21.32%`
      - `128B`: `+6.15%` to `+18.06%`
      - `256B`: `+4.60%` to `+16.30%`
      - `512B`: `+4.44%` to `+15.11%`
      - `1024B`: `+3.63%` to `+14.69%`
    - `xof init+read/*-in/32B-out`: `0W / 15L`
      - `1B-in`: `+22.97%` to `+44.71%`
      - `64B-in`: `+10.01%` to `+39.80%`
      - `1024B-in`: `+6.36%` to `+17.19%`
    - `xof-phase/finalize-xof-only/{1B,64B,1024B}-in`: `0W / 15L`
      - `1B-in`: `+18.35%` to `+31.22%`
      - `64B-in`: `+2.11%` to `+14.26%`
      - `1024B-in`: `+2.04%` to `+14.50%`
  - Notable behavior:
    - The rewrite preserved correctness and kept the implementation boundary clean.
    - It did not convert a single target-cluster lane.
    - x86 remained materially behind on the short streaming and short-XOF surfaces.
    - The architectural simplification was real; the performance win was not.
- Decision:
  - Reject and revert.
  - This was the right cleanup to try, but it is not the missing performance lever.
  - The evidence now says the remaining problem is not just “our serial engine shape is less upstream-like”.
  - There is still a fixed-cost/codegen problem in the repeated short-update path and in short XOF setup/read that this rewrite did not remove.

### 2026-03-08 - Candidate BG (`4ae6bc3`, run `22809940913`)
- Hypothesis:
  - The accepted baseline was still carrying too much cold control flow in the short serial hot path.
  - Splitting repeated short `update()` calls and single-chunk `finalize_xof()` into tiny leaves, while outlining the bulk/parallel/SIMD logic, would convert:
    - `streaming 64..1024B chunks`,
    - `finalize-xof-only`,
    - `xof init+read/*-in/32B-out`.
- Change:
  - `mod.rs` only:
    - added `ChunkState::try_update_fast()` and outlined the old generic block/chunk flow into cold `update_slow()`,
    - added `Blake3::try_update_chunk_local()` so repeated short updates could stay on the current chunk even when tree state already existed,
    - outlined the old serial control path into cold `update_with_slow()`,
    - made public `Digest::update()` take the chunk-local leaf before bulk-kernel selection,
    - added a single-chunk `finalize_xof()` leaf that returns `self.chunk_state.output()` directly when no tree fold is needed.
  - No dispatch-table changes, no kernel changes, no oneshot changes, no reader redesign.
- Validation:
  - Local: `cargo test -p hashes blake3 --lib` passed.
  - Local: `just check-all` passed.
  - Local: `just test` passed (`167/167`).
  - Local codegen signal:
    - `Digest::update`: `26` LLVM lines.
    - `ChunkState::update`: `56` LLVM lines.
    - slow paths moved into `update_with_slow` / `update_slow`.
  - Local targeted benches moved in the right direction on Apple M1, especially for short streaming and `finalize-xof-only`, but those numbers were not accepted as decision-grade.
  - CI targeted bench run: `22809940913` (`crates=hashes`, `benches=blake3`,
    `filter=blake3/streaming/,blake3/xof/,blake3/xof-phase/`, `quick=false`,
    lanes: `amd-zen5`, `intel-icl`, `intel-spr`, `graviton3`, `graviton4`).
- CI outcomes (time-based gap vs official; positive = slower):
  - Target-cluster check:
    - `streaming 64..1024B chunks`: `0W / 15L`
      - `64B`: `+1.47%` to `+13.09%`
      - `256B`: `+3.98%` to `+15.01%`
      - `1024B`: `+2.41%` to `+16.66%`
    - `xof init+read/*-in/32B-out`: `4W / 16L`
      - `1B-in`: `+3.05%` to `+21.73%`
      - `64B-in`: `-3.77%` to `+15.68%`
      - `1024B-in`: `+3.78%` to `+16.99%`
      - `65536B-in`: `-7.89%` to `+26.11%`
    - `xof-phase/finalize-xof-only/{1B,64B,1024B}-in`: `0W / 15L`
      - `1B-in`: `+19.94%` to `+30.30%`
      - `64B-in`: `+1.74%` to `+16.68%`
      - `1024B-in`: `+1.53%` to `+13.12%`
  - Per-lane summary on the exact target surfaces:
    - `amd-zen5`: `1W / 9L`
    - `intel-icl`: `1W / 9L`
    - `intel-spr`: `0W / 10L`
    - `graviton3`: `1W / 9L`
    - `graviton4`: `1W / 9L`
  - Notable behavior:
    - The local code-size cleanup was real.
    - The CI target clusters did not convert.
    - `finalize-xof-only` improved dramatically versus the catastrophic BF-style failures, but still remained net-loss on every lane.
    - Short streaming stayed decisively behind on every lane.
- Decision:
  - Reject and revert.
  - Hot/cold splitting and broader chunk-local fast-path admission are not enough.
  - This candidate confirms that simply making the accepted serial path smaller in source and LLVM IR does not remove the remaining cross-lane cost.

### 2026-03-08 - Candidate BH (local-only, not pushed)
- Hypothesis:
  - The remaining short streaming/XOF gap was still paying repeated kernel-ID dispatch inside serial inner primitives.
  - Resolving those operations once in `Kernel` and using direct entry points for:
    - chunk block compression,
    - parent CV folding,
    - root/XOF block emission,
    would lower fixed cost without another state-machine rewrite.
- Change:
  - `kernels.rs`:
    - extended `Kernel` with direct `parent_cv`, `root_output_block`, and `root_output_blocks` entry points,
    - added per-arch wrappers so serial code stopped re-matching on `kernel.id`.
  - `mod.rs`:
    - switched `ChunkState::update`, parent folding, and `OutputState::root_output_blocks_into()` to those direct entry points,
    - added a tiny first-read XOF shortcut for `<=32B` at `block_counter == 0`.
  - No dispatch-table retune, no oneshot rewrite, no public API change.
- Validation:
  - Local: `cargo test -p hashes blake3 --lib --tests --no-run` passed.
  - Local: `cargo test -p hashes blake3 --lib --tests` passed.
  - Full `just check-all` / `just test` were not run, because the local perf gate failed first.
  - Local targeted results on Apple M1 (median vs official):
    - `blake3/streaming`:
      - `64B-chunks`: `1.3398 ms` vs `1.3223 ms` (`+1.33%`)
      - `256B-chunks`: `1.3268 ms` vs `1.2726 ms` (`+4.26%`)
      - `1024B-chunks`: `1.3109 ms` vs `1.3215 ms` (`-0.80%`)
    - `blake3/xof/init+read/*/32B-out` exact rerun:
      - `1B-in`: `104.19 ns` vs `79.19 ns` (`+31.57%`)
      - `64B-in`: `101.47 ns` vs `80.49 ns` (`+26.06%`)
      - `1024B-in`: `1.2371 us` vs `1.1868 us` (`+4.24%`)
    - `blake3/xof-phase/finalize-xof-only/1B-in` exact rerun:
      - `21.63 ns` vs `19.53 ns` (`+6.02%`)
  - Attribution from the exact `xof-phase` rerun exposed the more important fact:
    - `new-only/1B-in`: `63.97 ns` vs `28.85 ns` (`+121.7%`)
    - `update-only/1B-in`: `49.99 ns` vs `32.08 ns` (`+55.8%`)
    - `new-only/64B-in`: `64.30 ns` vs `28.99 ns` (`+121.8%`)
    - `update-only/64B-in`: `53.18 ns`, still far behind official and regressed vs our prior local baseline
- Decision:
  - Reject locally. Do not push.
  - The direct serial-op snapshot is not the missing lever.
  - The remaining short-XOF gap is dominated by constructor/setup and first-update fixed cost, not by the old finalize/squeeze dispatch ladders.
  - The tiny `<=32B` first-read shortcut did not change that conclusion.

### 2026-03-08 - Candidate BI (local-only, not pushed)
- Hypothesis:
  - The base `Blake3` object was still too large for the short-path setup surfaces.
  - Moving tree-only state (`pending_chunk_cv` + `cv_stack`) out of the base object on `std` builds and allocating it lazily only when the stream actually becomes multi-chunk would reduce:
    - `new-only`,
    - first tiny `update()`,
    - `xof init+read/*-in/32B-out`,
    without touching the accepted algorithm shape.
- Change:
  - `mod.rs` only:
    - introduced a `TreeState` sidecar,
    - stored tree-only state behind `Option<Box<TreeState>>` on `std`,
    - kept the inline tree state on non-`std`,
    - routed stack/pending helpers through the sidecar,
    - left kernels, oneshot, XOF reader shape, and public API unchanged.
- Validation:
  - Local: `cargo test -p hashes blake3 --lib --tests --no-run` passed.
  - Local: `cargo test -p hashes blake3 --lib --tests` passed.
  - Full `just check-all` / `just test` were not run, because the local perf gate failed first.
  - Local `xof-phase` attribution moved dramatically:
    - `new-only/1B-in`: `12.48 ns` vs official `29.08 ns`
    - `update-only/1B-in`: `21.12 ns` vs official `32.32 ns`
    - `new-only/64B-in`: `12.45 ns` vs official `29.03 ns`
    - `update-only/64B-in`: `21.91 ns` vs official `33.10 ns`
    - `finalize-xof-only/64B-in`: `20.42 ns` vs official `19.67 ns`
    - `finalize-xof-only/1024B-in`: `19.87 ns` vs official `19.48 ns`
  - But the actual target surfaces did not convert:
    - `blake3/streaming/64B-chunks`: `1.3875 ms` vs official `1.3026 ms` (`+6.52%`)
    - `blake3/streaming/256B-chunks`: `1.3455 ms` vs official `1.2758 ms` (`+5.46%`)
    - `blake3/streaming/1024B-chunks`: `1.3187 ms` vs official `1.2861 ms` (`+2.54%`)
    - `blake3/xof/init+read/1B-in/32B-out`: `102.73 ns` vs official `78.51 ns` (`+30.85%`)
    - `blake3/xof/init+read/64B-in/32B-out`: `99.29 ns` vs official ~`80 ns` (still badly red)
- Decision:
  - Reject locally. Do not push.
  - The lazy tree sidecar fixed setup attribution benches and still failed the real acceptance surfaces.
  - Inference:
    - `xof-phase/new-only` and `xof-phase/update-only` are over-weighting whole-hasher move/black-box cost and are not reliable acceptance proxies for `xof init+read`.
    - The sidecar also introduced an expensive multi-chunk transition in streaming, which showed up exactly where short chunked streaming is already weak.

## Anti-Repeat Guardrails (2026-03-08)

- Do not repeat finalize-time compact XOF/output materialization as a strategy (`Candidate BB`).
- Do not treat lazy single-chunk XOF reader promotion alone as the fix (`Candidate BC`).
- Do not expect hot-state shrinking and localized bulk dispatch alone to break through (`Candidate BD`).
- Do not rewrite the generic `OutputState` around raw bytes as a standalone fix (`Candidate BE`).
- Do not assume an upstream-literal serial streaming/XOF rewrite alone is sufficient (`Candidate BF`).
- Do not assume hot/cold outlining plus broader chunk-local update admission is sufficient (`Candidate BG`).
- Do not assume direct serial primitive snapshots plus another tiny first-read XOF shortcut are sufficient (`Candidate BH`).
- Do not assume shrinking the base hasher with a lazy tree sidecar is sufficient (`Candidate BI`).

### 2026-03-08 - Truthful short-path attribution
- Goal:
  - stop trusting `xof-phase/new-only` and `xof-phase/update-only` as previously measured,
  - add attribution that mirrors the real acceptance surfaces without black-boxing whole hashers by value,
  - determine whether the remaining short-XOF gap lives in:
    - constructor / first update setup,
    - `finalize_xof()` setup,
    - or the first 32-byte squeeze itself.
- Change:
  - `crates/hashes/benches/blake3.rs`
    - changed `blake3/xof-phase/{new-only,update-only,finalize-xof-only}` to black-box references instead of moving whole hashers/readers by value,
    - added `finalize-xof+read32-only/*` so the main BLAKE3 bench can attribute first-read cost without constructor/update.
  - `crates/hashes/benches/blake3_short_input_attribution.rs`
    - added `blake3/short-input/xof-target-attribution`,
    - added `blake3/short-input/stream-target-attribution`,
    - both mirror the actual target shapes instead of synthetic by-value ownership traffic.
- Validation:
  - `cargo test -p hashes --benches --no-run` passed.
  - `cargo test -p hashes blake3 --lib --tests --no-run` passed.
  - `cargo bench --profile bench -p hashes --bench blake3_short_input_attribution -- 'blake3/short-input/(xof-target-attribution|stream-target-attribution)'`
    completed locally.
- Local findings on Apple M1:
  - XOF target split:
    - `init-only(ref)/1B-in`: `8.31 ns` vs official `3.76 ns`
    - `init+update(ref)/1B-in`: `17.63 ns` vs `7.91 ns`
    - `init+update+finalize-xof(ref)/1B-in`: `28.20 ns` vs `11.35 ns`
    - `finalize-xof+read32(clone)/1B-in`: `119.82 ns` vs `110.33 ns`
    - `init+read32(target)/1B-in`: `104.83 ns` vs `82.33 ns`
    - `read32-only/1B-in`: `74.92 ns` vs `74.26 ns`
    - `init+update(ref)/64B-in`: `18.80 ns` vs `8.86 ns`
    - `init+update+finalize-xof(ref)/64B-in`: `28.27 ns` vs `12.03 ns`
    - `finalize-xof+read32(clone)/64B-in`: `118.26 ns` vs `99.76 ns`
    - `init+read32(target)/64B-in`: `103.82 ns` vs `82.48 ns`
    - `read32-only/64B-in`: `74.16 ns` vs `73.38 ns`
  - Streaming target split:
    - `update-loop-only(ref)/64B-chunks`: `1.4648 ms` vs official ~`1.54 ms` (noisy, not useful as primary evidence)
    - `full(target)/64B-chunks`: `1.4494 ms` vs `1.3642 ms`
    - `update-loop-only(ref)/256B-chunks`: `1.3996 ms` vs `1.3313 ms`
    - `finalize-only(clone)/256B-chunks`: `851.66 ns` vs `843.98 ns`
    - `full(target)/256B-chunks`: `1.3986 ms` vs `1.3227 ms`
    - `update-loop-only(ref)/1024B-chunks`: `1.3616 ms` vs `1.3322 ms`
    - `finalize-only(clone)/1024B-chunks`: `846.90 ns` vs `844.90 ns`
    - `full(target)/1024B-chunks`: `1.3698 ms` vs `1.3410 ms`
- Decision / inference:
  - The short `read32-only` path is basically at parity. Do not treat `Blake3Xof::squeeze()` / first-block copy as the primary culprit anymore.
  - The short `init+read32` gap is mostly before or at `finalize_xof()`:
    - constructor + first update are still materially behind,
    - `finalize_xof()` setup adds another fixed-cost gap on `1B` / `64B`,
    - the pure 32-byte squeeze is not where the remaining loss lives.
  - The short streaming loss is overwhelmingly the repeated `update()` loop.
    - `finalize()` after a fully-populated 1 MiB stream is near parity,
    - so another `finalize()`-centric rewrite would be ornamental complexity.

## Next Step After Truthful Attribution

1. Keep the accepted baseline architecture and keep the new attribution benches.
2. Do not target `Blake3Xof::squeeze()` / reader-only paths as the primary fix.
3. Next candidate should attack only two fixed-cost surfaces:
   - `Blake3::new_internal()` / first tiny `update()` setup,
   - short `finalize_xof()` / `root_output()` setup before any bytes are squeezed.
4. Keep the candidate narrow:
   - no serial-state rewrite,
   - no oneshot changes,
   - no dispatch-table retune,
   - no XOF reader redesign unless the new attribution moves.
5. Validate in this order:
   - local attribution:
     - `blake3/short-input/xof-target-attribution`
     - `blake3/short-input/stream-target-attribution`
   - local acceptance surfaces:
     - `blake3/streaming/64B..1024B-chunks`
     - `blake3/xof/init+read/*/32B-out`
   - only if both move:
     - `just check-all && just test`
     - then the same 5-lane CI bench scope.
6. Keep/reject rule:
   - if the candidate improves only constructor-only attribution or only pure `read32-only`, reject it before CI.

### 2026-03-08 - Candidate BJ (local-only, not pushed)
- Hypothesis:
  - We were still paying redundant public-path dispatch/setup work on every short streaming update:
    - `new_internal()` fetched the hasher dispatch twice,
    - `update()` reloaded the stream kernel, recomputed the default table-bulk choice, and rewrote the same kernel fields even when the short-input path would not change dispatch.
  - Eliminating that fixed work without changing the state machine could cut:
    - `new_internal()` cost,
    - first tiny `update()`,
    - and some short `finalize_xof()` setup indirectly.
- Change:
  - `mod.rs` / `dispatch.rs` only:
    - fetched the constructor dispatch plan once,
    - cached the default table-bulk kernel at construction,
    - split the `update_with` body so short inputs below `bulk_sizeclass_threshold` reused the existing stream/table-bulk pair instead of re-running dispatch setup each call.
  - No kernel changes, no state-machine rewrite, no reader redesign, no API change.
- Validation:
  - `cargo test -p hashes blake3 --lib --tests --no-run` passed.
  - `cargo test -p hashes --benches --no-run` passed.
  - Local truthful attribution:
    - `rscrypto init-only(ref)/1B-in`: `7.65 ns` vs prior `8.31 ns`
    - `rscrypto init+update(ref)/1B-in`: `12.61 ns` vs prior `17.63 ns`
    - `rscrypto init+update+finalize-xof(ref)/1B-in`: `23.76 ns` vs prior `28.20 ns`
    - `rscrypto init+update(ref)/64B-in`: `13.82 ns` vs prior `18.80 ns`
    - `rscrypto init+update+finalize-xof(ref)/64B-in`: `23.18 ns` vs prior `28.27 ns`
    - `rscrypto init+read32(target)/1B-in`: `97.23 ns` vs prior `104.83 ns`
    - `rscrypto init+read32(target)/64B-in`: `97.80 ns` vs prior `103.82 ns`
  - Local target surfaces still failed:
    - `blake3/streaming/64B-chunks`: `1.3996 ms` vs official `1.3860 ms`
    - `blake3/streaming/256B-chunks`: `1.3995 ms` vs official `1.3318 ms`
    - `blake3/streaming/1024B-chunks`: `1.3664 ms` vs official `1.3356 ms`
    - `blake3/xof/init+read/1B-in/32B-out`: `101.89 ns` vs official `82.75 ns`
    - `blake3/xof/init+read/64B-in/32B-out`: `96.52 ns` vs official `82.57 ns`
    - `blake3/xof/init+read/1024B-in/32B-out`: `1.2670 us` vs official `1.2442 us`
- Decision:
  - Reject locally. Do not push.
  - This candidate proved that redundant dispatch/setup on the public short path is real and worth removing in principle.
  - But it still does not convert the actual repeated short streaming surfaces, which remain red exactly where we need wins.
  - The remaining bottleneck is not just dispatch-plan churn at the API boundary.

### 2026-03-08 - Candidate BK (local-only, not pushed)
- Hypothesis:
  - The remaining single-chunk `finalize_xof()` setup gap might still include unnecessary tail handling in `ChunkState::output()`.
  - Keeping the chunk buffer tail zeroed after buffered-block compression would let `output()` stop zero-filling the last block before building `OutputState`, potentially reducing short XOF setup cost without changing reader or tree state.
- Change:
  - `mod.rs` only:
    - after compressing a buffered full block, reset `self.block` back to `[0; 64]`,
    - removed the explicit zero-fill from `ChunkState::output()` and relied on the zeroed-tail invariant instead.
  - No reader changes, no kernel changes, no dispatch changes.
- Validation:
  - `cargo test -p hashes blake3 --lib --tests --no-run` passed.
  - `cargo test -p hashes --benches --no-run` passed.
  - Local truthful XOF attribution regressed on the measured setup phases:
    - `init+update(ref)/1B-in`: `17.30 ns` vs prior baseline `17.63 ns` but far worse than `Candidate BJ` and still not competitive
    - `init+update+finalize-xof(ref)/1B-in`: `24.84 ns` vs baseline `28.20 ns`, but not enough to explain the target gap
    - `init+update(ref)/64B-in`: `18.94 ns` vs baseline `18.80 ns`
    - `init+update+finalize-xof(ref)/64B-in`: `26.25 ns` vs baseline `28.27 ns`
    - `init+read32(target)/64B-in`: `97.97 ns`, essentially flat versus the restored baseline signal
  - Local target surfaces:
    - `blake3/xof/init+read/1B-in/32B-out`: `98.06 ns` vs official `80.87 ns`
    - `blake3/xof/init+read/64B-in/32B-out`: `99.43 ns` vs official `82.07 ns`
    - `blake3/xof/init+read/1024B-in/32B-out`: `1.2607 us` vs official `1.2284 us`
- Decision:
  - Reject locally. Do not push.
  - The zero-tail/output cleanup is too small and too noisy to be a real lever.
  - More importantly, it does not improve the truthful setup phases in a way that matches the acceptance targets.

### 2026-03-08 - Candidate BL (local-only, not pushed)
- Hypothesis:
  - The remaining short-XOF gap might come from converting the single-chunk tail block into words before first-block root/XOF emission.
  - A very narrow x86-only raw-byte first-block path beneath `finalize_xof()` could bypass that conversion without repeating the failed generic `OutputState` raw-byte rewrite.
- Change:
  - `mod.rs` / `x86_64/{sse41,avx2,avx512}.rs` only:
    - added x86 raw-byte single-block root-output helpers,
    - taught `finalize_xof()` to preserve the padded raw tail block for single-chunk/no-tree states,
    - taught `Blake3Xof` to use that raw-byte path for block counter `0` only, then fall back to the existing generic `OutputState` path.
  - No oneshot changes, no generic `OutputState` rewrite, no dispatch-table changes, no public API change.
- Validation:
  - Correctness smoke before revert:
    - `cargo test -p hashes blake3 --lib --tests --no-run` passed.
    - `cargo test -p hashes --benches --no-run` passed.
    - `cargo test -p hashes single_chunk_xof_prefix_matches_official_crate_for_forced_kernels -- --nocapture` passed.
    - `cargo test -p hashes xof_repeated_small_squeezes_match_single_read -- --nocapture` passed.
  - Exact local target surfaces moved the wrong way immediately:
    - truthful attribution:
      - `rscrypto init+update+finalize-xof(ref)/1B-in`: `27.71 ns` vs baseline `24.84 ns`
      - `rscrypto finalize-xof+read32(clone)/1B-in`: `118.30 ns` vs baseline `110.70 ns`
    - target benches:
      - `blake3/xof/init+read/1B-in/32B-out`: `105.50 ns`, `+7.59%` vs prior local baseline
      - `blake3/xof-phase/finalize-xof-only/1B-in`: `22.66 ns`, `+4.85%` vs prior local baseline
      - `blake3/xof-phase/finalize-xof-only/64B-in`: `21.05 ns`, `+3.06%` vs prior local baseline
  - I stopped the remaining local bench runs once the exact XOF target surfaces were decisively red, then reverted the code immediately.
- Decision:
  - Reject locally. Do not push.
  - This kills the narrow “x86-only raw-byte first-block XOF leaf” variant.
  - The byte-to-word conversion we were targeting is not the dominant lever in practice once the reader/setup plumbing needed to exploit it is included.
  - Combined with `Candidate BE`, this is enough evidence to stop spending time on partial raw-byte root/XOF rewrites.

### 2026-03-08 - External attribution snapshot before any further code changes
- Environment / tools:
  - Host: macOS 26.3.1, arm64 (Apple Silicon).
  - Available attribution tools here are `cargo asm`, `cargo llvm-lines`, `sample`, `samply`, `xctrace`, and `dtrace`.
  - Linux-style hardware counter tools (`perf stat`, `perf record`) are not available on this host, so the nearest equivalent here is assembly + sampled time-profiler evidence.
- Method:
  - Built a disposable crate outside the repo that exposes exact `init -> update -> finalize_xof -> read 32B` wrappers for:
    - `hashes::crypto::Blake3`
    - official `blake3::Hasher`
  - Compared:
    - wrapper assembly with `cargo asm`,
    - wrapper and internal code size with `cargo llvm-lines`,
    - runtime stack samples with macOS `sample` on tight wrapper loops.
- Wrapper code-size result:
  - Exact 1B / 64B wrapper size (`cargo llvm-lines`, local disposable crate):
    - official `init+read32`: `84-86` LLVM lines
    - rscrypto `init+read32`: `237` LLVM lines
  - In our workspace build:
    - `OutputState::root_output_blocks_into`: `3297` LLVM lines
    - `Blake3::update_with`: `1120`
    - `Blake3::root_output`: `332`
    - `Blake3Xof::squeeze`: `250`
    - `Blake3Xof::fill_one_block`: `65`
- Assembly result:
  - Official `init+read32` wrapper stays simple:
    - initialize hasher,
    - call `Hasher::update`,
    - call `Hasher::finalize_xof`,
    - call `OutputReader::fill`.
  - Our `init+read32` wrapper pulls in materially more fixed work before the first 32 output bytes:
    - repeated `HASHER_DISPATCH_REF` / once-lock load checks,
    - repeated `stream_kernel` reloads,
    - `bulk_kernel_for_update` / size-class selection,
    - a much larger stack frame,
    - call chain through `Blake3::update_with`, `Blake3::root_output`, and `Blake3Xof::squeeze`.
  - Stack frame size in the wrapper diff was also not subtle:
    - official: `2080` bytes
    - rscrypto: `2352` bytes
- Sampled runtime result on the exact `1B -> read32` wrapper loop:
  - rscrypto top-of-stack buckets:
    - `compress`: `2615`
    - `Blake3::root_output`: `280`
    - `Blake3::update_with`: `190`
    - `Blake3Xof::squeeze`: `178`
    - `memset`: `177`
    - `memmove`: `102`
    - `try_simd_update_batch`: `45`
  - official top-of-stack buckets:
    - `portable::compress_xof`: `3185`
    - `Hasher::final_output`: `180`
    - `Hasher::update`: `179`
    - `OutputReader::fill`: `131`
    - `memmove`: `153`
    - `Hasher::finalize_xof`: `37`
  - Interpretation:
    - both implementations spend most samples inside compression, as expected;
    - the gap is the extra fixed-cost control work around compression on our side, not a single missing reader helper.
- What this rules in:
  - The short-XOF deficit is structurally closer to:
    - `update_with` control flow,
    - `root_output` construction/folding,
    - `squeeze` / `root_output_blocks_into` setup and copy behavior,
    than to any one raw-byte conversion or first-block micro-helper.
- What this rules out:
  - Stop chasing isolated:
    - first-read XOF helpers,
    - tail-zeroing tweaks,
    - partial raw-byte root/XOF leaves.
  - Those are too small relative to the whole-path fixed cost the asm and samples show.
- Working conclusion:
  - If we continue at all, the next candidate has to be justified against this whole-path evidence, not against a single helper diff.
  - If we do not have a candidate that can remove meaningful control-flow/code-size from `short update + root_output + squeeze` together, we should stop rather than keep churning BLAKE3 internals.

### 2026-03-08 - Candidate BM (pushed, CI run `22813005832`)
- Hypothesis:
  - A no-tree frontier mode that removes `update_with -> root_output -> generic squeeze` from the exact short serial/XOF path might finally convert the short clusters without touching oneshot or kernels.
- Change:
  - Added a frontier-mode short path for:
    - small serial `update()` while still in the first chunk,
    - single-chunk `finalize_xof()`,
    - a dedicated `FrontierXof` reader for the no-tree case.
  - Kept the existing tree path for everything else.
- CI scope:
  - Workflow run: `22813005832`
  - Lanes: `amd-zen4`, `amd-zen5`, `intel-icl`, `intel-spr`, `graviton3`, `graviton4`
  - Filter: `blake3/streaming/`, `blake3/xof/`, `blake3/xof-phase/`
- Result:
  - Rejected.
  - Short streaming and `finalize-xof-only` stayed red across the completed lanes.
  - The only real improvement cluster was `xof/init+read/64B-in/32B-out`, which is too narrow to justify keeping the rewrite.
  - Graviton4 was not needed to make the call.
- What it proved:
  - Removing the generic no-tree `update_with/root_output/squeeze` stack frame is a real lever.
  - But a frontier path that only owns the first chunk is too small a slice of the real hot surfaces.

### 2026-03-08 - Candidate BN (pushed, CI run `22813859207`)
- Hypothesis:
  - If the frontier idea is carried through chunk rollover and short final-output setup, not just the first chunk, it might solve short streaming without sacrificing short XOF.
- Change:
  - Extended the short serial frontier path so `update()` can carry across chunk rollover instead of bailing out after the first chunk.
  - Replaced the split tree/frontier XOF reader with a compact root-state reader that preserves the minimum state needed for root emission.
  - Preserved public API, oneshot path, and kernels.
- Validation before CI:
  - `cargo clean`
  - `just check-all`
  - `just test`
  - all passed after fixing an internal root-state counter regression during development.
- CI scope:
  - Workflow run: `22813859207`
  - Lanes: `amd-zen4`, `amd-zen5`, `intel-icl`, `intel-spr`, `graviton3`, `graviton4`
  - Filter: `blake3/streaming/`, `blake3/xof/`, `blake3/xof-phase/`
- Result:
  - Rejected.
  - Exact target summary vs official:
    - `blake3/streaming/{64B,256B,1024B}-chunks`: `0W / 18L`
    - `blake3/xof/init+read/{1B,64B,1024B}-in/32B-out`: `0W / 18L`
    - `blake3/xof-phase/finalize-xof-only/{1B,64B,1024B}`: `0W / 18L`
- Useful new evidence:
  - Short streaming improved materially versus `BM`:
    - `64B-chunks`: improved on all 5 comparable prior lanes
    - `256B-chunks`: improved on all 5 comparable prior lanes
    - `1024B-chunks`: improved on 2 comparable lanes and did not materially regress elsewhere
  - Representative remaining streaming gaps vs official:
    - `graviton4 64B-chunks`: `+3.14%`
    - `graviton3 64B-chunks`: `+3.58%`
    - `graviton4 256B-chunks`: `+3.71%`
    - `graviton3 1024B-chunks`: `+2.28%`
  - XOF did not convert:
    - `init+read/1B-in/32B-out`: still `0W / 6L`
    - `init+read/64B-in/32B-out`: still `0W / 6L`
    - `finalize-xof-only/*`: still `0W / 6L` on every input size
  - Worse, parts of short XOF regressed relative to `BM`, especially on x86 for:
    - `init+read/64B-in/32B-out`
    - `finalize-xof-only/*`
- Decision:
  - Revert.
  - `BN` proves the frontier idea is a valid streaming lever, but not a combined streaming-plus-XOF solution.
  - The remaining problem is no longer “generic short streaming control flow” by itself.
  - The XOF setup/final-output path is still the blocker, and the current frontier/root-state direction did not solve it cleanly.

### Current conclusion after `BM` + `BN`
- The frontier idea should not be treated as an all-up answer.
- What is now defensible to say:
  - short serial streaming can be improved by owning more of the rollover path locally;
  - short XOF setup/final-output still does not have a winning design;
  - a unified rewrite that tries to solve both surfaces at once is still failing the bar.
- If work continues, the next candidate should be explicitly narrower:
  - either a streaming-only candidate built on the lessons from `BN`,
  - or a separate XOF/final-output candidate justified by new attribution first.

### 2026-03-08 - Next execution plan after `BM` + `BN`
- Non-negotiable framing:
  - Stop treating short streaming and short XOF as one optimization problem.
  - Keep the public API, oneshot path, and existing kernels.
  - Do not spend another cycle on generic `OutputState` churn, reader-only helpers, or dispatch-table retuning as the primary move.
  - Every candidate below must be independently shippable and independently revertable.

#### Track 1 - Streaming-only candidate (`BO`)
- Goal:
  - Convert `blake3/streaming/{64B,256B,1024B}-chunks` without touching XOF behavior.
- Why this track exists:
  - `BN` materially improved short streaming on every comparable lane.
  - `BN` failed because of XOF, not because the streaming direction was wrong.
- Scope:
  - Restrict code changes to the serial streaming state machine in:
    - `Blake3::update`
    - `Blake3::update_with`
    - chunk-rollover handling
    - pending-chunk/tree admission only as needed for streaming correctness
  - No changes to:
    - `Blake3Xof`
    - `finalize_xof`
    - `OutputState`
    - oneshot APIs
    - dispatch tables
    - kernel implementations
- Intended design:
  - Keep a frontier-style local serial path for repeated short updates.
  - Carry that path cleanly across chunk rollover, not just inside the first chunk.
  - Fall back to the current tree/bulk path only when:
    - parallel admission actually matters,
    - SIMD full-chunk batching actually matters,
    - or the update is no longer a short serial case.
  - Do not try to share this candidate with XOF setup or root-output redesign.
- Local gate before CI:
  - `cargo clean`
  - `just check-all`
  - `just test`
  - `cargo bench --profile bench -p hashes --bench blake3 -- 'blake3/streaming/(rscrypto|official)/(64B-chunks|256B-chunks|1024B-chunks)'`
  - Treat local benches as directional only.
- CI acceptance bar:
  - Dispatch targeted `bench.yaml` with:
    - `filter=blake3/streaming/`
    - lanes: `amd-zen4`, `amd-zen5`, `intel-icl`, `intel-spr`, `graviton3`, `graviton4`
  - Keep only if:
    - the streaming cluster shows clear cross-lane wins or near-parity conversion,
    - and no correctness regressions appear locally.
  - Revert immediately if:
    - streaming is still net red across all lanes,
    - or wins are too narrow to justify carrying the extra control path.

#### Track 2 - Dedicated backend-level root/XOF path (`BP`)
- Goal:
  - Convert:
    - `blake3/xof/init+read/{1B,64B,1024B}-in/32B-out`
    - `blake3/xof-phase/finalize-xof-only/{1B,64B,1024B}`
- Why this track exists:
  - External attribution showed the remaining deficit is the whole short final-output path:
    - `update_with`
    - `root_output`
    - `squeeze`
    - `root_output_blocks_into`
  - The reader-only and partial raw-byte attempts are already ruled out.
  - `BN` also showed that carrying frontier ideas into XOF setup is not enough under the current root/output model.
- Scope:
  - Build a real internal root-output/XOF object below the public API.
  - Keep the public `Blake3Xof` type and external behavior unchanged.
  - Keep tree semantics, oneshot behavior, and kernel selection rules unchanged.
- Intended design:
  - Introduce a compact internal root-output carrier for XOF/final-output setup.
  - This carrier should store exactly the state needed for root emission:
    - input CV
    - final block bytes
    - block length
    - chunk/root flags
    - chunk counter / output block counter as required
    - resolved emission kernel or direct backend entry points
  - Add backend-level direct root/XOF emission entry points so the short path does not go through:
    - generic `OutputState`
    - generic `root_output_blocks_into`
    - generic reader promotion logic
  - Keep the generic tree/output path for the cases that actually need it.
  - The short serial `finalize_xof -> read32` path should be able to stay entirely on this compact backend path.
- Explicit non-goals:
  - no generic raw-byte `OutputState` rewrite,
  - no x86-only special case as the main design,
  - no eager finalize-time precompute that pushes work into every XOF call,
  - no dispatch-table tuning as a substitute for design work.
- Required attribution before implementation:
  - Re-run code-size and asm checks on the exact wrapper path:
    - `cargo llvm-lines`
    - `cargo asm`
    - `sample` / `samply`
  - The new design is only worth landing if it materially removes the current short-path frames from the wrapper:
    - `Blake3::root_output`
    - `Blake3Xof::squeeze`
    - `OutputState::root_output_blocks_into`
- Local gate before CI:
  - `cargo clean`
  - `just check-all`
  - `just test`
  - `cargo bench --profile bench -p hashes --bench blake3 -- 'blake3/xof/(rscrypto|official)/init\\+read/(1B-in|64B-in|1024B-in)/32B-out'`
  - `cargo bench --profile bench -p hashes --bench blake3 -- 'blake3/xof-phase/(rscrypto|official)/finalize-xof-only/(1B-in|64B-in|1024B-in)'`
  - Treat local benches as directional only.
- CI acceptance bar:
  - Dispatch targeted `bench.yaml` with:
    - `filter=blake3/xof/,blake3/xof-phase/`
    - lanes: `amd-zen4`, `amd-zen5`, `intel-icl`, `intel-spr`, `graviton3`, `graviton4`
  - Keep only if:
    - short-XOF surfaces show real cross-lane wins,
    - and `finalize-xof-only` is no longer universally red.
  - Revert immediately if:
    - the result is still `0W / N` on `finalize-xof-only`,
    - or improvements are confined to one input size or one architecture family.

#### Execution order
- Do not combine these tracks again.
- Run them in this order:
  1. `BO` streaming-only candidate.
  2. If `BO` wins, keep it isolated and move on to `BP`.
  3. If `BO` fails, revert it and still proceed to `BP` from baseline.
- Reason:
  - the evidence is already strong enough that streaming and XOF need different designs and different acceptance bars.

#### Keep / stop rule
- If `BO` fails and `BP` also fails, stop BLAKE3 short-path churn.
- At that point the remaining gap is either:
  - inherent to the chosen Rust/control-flow architecture around the existing kernels,
  - or only recoverable with a larger backend/kernel model change than is currently justified.

### 2026-03-08 - Candidate `BO` (pushed, CI run `22823651733`)
- Hypothesis:
  - Extending only the short-update frontier across chunk rollover, while leaving XOF and final-output code untouched, would convert the short streaming cluster by itself.
- Change:
  - Broadened the short serial `update()` path so any `<= CHUNK_LEN` update could stay on the local serial path across chunk rollover and pending-chunk commit.
  - Left `finalize_xof`, `Blake3Xof`, `OutputState`, oneshot code, and kernels unchanged.
- Validation before CI:
  - `just check-all`
  - `just test`
  - all passed locally.
- CI scope:
  - Workflow run: `22823651733`
  - Lanes: `amd-zen4`, `amd-zen5`, `intel-icl`, `intel-spr`, `graviton3`, `graviton4`
  - Filter: `blake3/streaming/`
- Result:
  - Rejected.
  - Exact target summary vs official:
    - `blake3/streaming/{64B,256B,1024B}-chunks`: `0W / 18L`
- Representative gaps vs official:
  - `amd-zen4`: `64B +12.07%`, `256B +14.15%`, `1024B +14.56%`
  - `amd-zen5`: `64B +10.96%`, `256B +12.73%`, `1024B +12.61%`
  - `intel-icl`: `64B +10.08%`, `256B +13.82%`, `1024B +14.18%`
  - `intel-spr`: `64B +14.73%`, `256B +16.32%`, `1024B +15.90%`
  - `graviton3`: `64B +4.18%`, `256B +5.07%`, `1024B +3.23%`
  - `graviton4`: `64B +3.01%`, `256B +4.29%`, `1024B +2.76%`
- What it proved:
  - Another streaming-only frontier follow-up does not break through on x86 and does not improve enough on Arm to justify keeping it.
  - `BO` is not clearly better than the stronger streaming signal already observed inside `BN`.
  - The frontier idea has yielded the signal it is going to yield.
- Decision:
  - Revert.
  - Stop spending time on more streaming-only frontier churn.
  - The next candidate is `BP`, not `BO2`.

### Current conclusion after `BO`
- The combined streaming+XOF rewrites failed.
- The streaming-only frontier follow-up also failed.
- What is still left standing:
  - the remaining high-value unsolved problem is short XOF/final-output setup,
  - and the only technically credible next move is the dedicated backend-level root/XOF path (`BP`).
- Practical rule from here:
  - do not spend another candidate on `update()` control-flow reshaping alone,
  - do not revisit reader-only helpers,
  - do not revisit generic `OutputState` rewrites,
  - move directly to `BP` or stop.
