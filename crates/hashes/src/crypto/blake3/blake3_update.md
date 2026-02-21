# BLAKE3 Update Plan

## Goal
Close the remaining BLAKE3 gaps so `rscrypto` is consistently at or ahead of official in the enforced CI lanes, with special focus on 256/1024-byte workloads.

## Current State
- Biggest deficits remain short-input (256/1024), especially on x86 and arm64 kernel-ab lanes.
- Canonical full-lane baseline capture below was run on `main` at `4d35d53` via workflow run `22259222437`.

## #1/#2/#3 Status
- #1 Bench infra (s390x/power kernel-ab data + prefix alignment): complete.
  - Evidence: `ibm-s390x` now emits `rscrypto/s390x/vector`; `ibm-power10` emits `rscrypto/powerpc64/vsx` in `kernel-ab`.
- #2 Gate sequencing/observability (both gates visible in one run): complete.
  - Evidence: run `22259222437` produced both `oneshot` and `kernel-ab` outputs for all enforced lanes.
- #3 Baseline lock for optimization phase: complete.
  - Canonical baseline data tables below are now pinned to `main` run `22259222437`.

## Full Baseline Capture (2026-02-21)

### Source Runs
- CI (canonical): `Bench` workflow run `22259222437` on `main` at `4d35d53` (all 8 lanes enabled, `enforce_blake3_gap_gate=true`, `enforce_blake3_kernel_gate=true`).
- Local sanity: forced-kernel override validated with `RSCRYPTO_BLAKE3_FORCE_KERNEL=aarch64/neon` (`parent-folding` run prints forced-kernel diagnostics).

### Forced-Kernel Sanity Sweep (2026-02-21, `main`)
- x86 (`x86_64/avx512` forced): run `22259741524` (lanes: `amd-zen4`, `intel-spr`, `intel-icl`, `amd-zen5`) completed `success`.
- arm64 (`aarch64/neon` forced): run `22259791226` (lanes: `graviton3`, `graviton4`) completed `success`.
- power10 (`powerpc64/vsx` forced): run `22259836546` (lane: `ibm-power10`) completed `success`.
- s390x (`s390x/vector` forced): run `22259901223` (lane: `ibm-s390x`) completed `success`.
- Evidence captured in each run log:
  - workflow input `blake3_force_kernel` echoed in run config,
  - `RSCRYPTO_BLAKE3_FORCE_KERNEL=<kernel>` exported into bench execution environment,
  - filter override used (`parent-folding,streaming-dispatch,kernel-ab`).

### Oneshot Gate Need (%; positive means rscrypto is behind official)

| Lane | 256 | 1024 | 4096 | 16384 | 65536 |
|---|---:|---:|---:|---:|---:|
| amd-zen4 | +7.29 | +14.60 | +7.57 | +5.17 | +0.72 |
| intel-spr | +30.99 | +34.32 | +7.15 | +2.30 | +2.79 |
| intel-icl | +34.64 | +37.51 | +9.24 | +5.09 | +2.11 |
| amd-zen5 | +25.43 | +25.63 | +5.68 | +3.42 | +1.04 |
| graviton3 | +3.74 | +3.75 | -5.28 | -10.03 | -11.30 |
| graviton4 | +4.47 | +3.53 | -5.86 | -10.51 | -11.74 |
| ibm-power10 | +9.78 | -9.11 | -64.55 | -67.57 | -68.38 |
| ibm-s390x | +0.22 | +4.47 | +2.93 | +1.05 | +0.91 |

### Kernel-ab Gate Need (%; best lane-native rscrypto kernel vs official)

| Lane | 256 | 1024 | 4096 | 16384 | 65536 | Notes |
|---|---:|---:|---:|---:|---:|---|
| amd-zen4 | -0.14 | +3.41 | +5.87 | +3.61 | +0.42 | `rscrypto/x86_64/sse4.1` best at 256 on this lane |
| intel-spr | +17.37 | +16.01 | +4.53 | +3.31 | -5.16 | `rscrypto/x86_64/avx512` |
| intel-icl | +19.73 | +17.25 | +6.76 | +3.53 | +1.61 | `rscrypto/x86_64/avx512` |
| amd-zen5 | +15.09 | +9.28 | +5.32 | +2.88 | +0.97 | `rscrypto/x86_64/avx512` |
| graviton3 | +31.91 | +13.44 | -4.98 | -9.90 | -11.27 | `rscrypto/aarch64/neon` |
| graviton4 | +33.61 | +13.94 | -5.61 | -10.42 | -11.71 | `rscrypto/aarch64/neon` |
| ibm-power10 | -7.54 | -9.82 | -64.46 | -67.13 | -67.84 | `rscrypto/powerpc64/vsx` (ahead) |
| ibm-s390x | +14.15 | +9.87 | -55.93 | -58.61 | -59.28 | `rscrypto/s390x/vector` now emitted |

### Streaming Dispatch Snapshot (from same CI run, lengths 256/1024)

| Lane | plain(256) | plain(1024) | keyed(256) | derive(256) |
|---|---|---|---|---|
| amd-zen4 | `sse4.1 -> avx512` | `sse4.1 -> avx512` | `sse4.1 -> avx512` | `sse4.1 -> avx512` |
| intel-spr | `sse4.1 -> avx512` | `sse4.1 -> avx512` | `sse4.1 -> avx512` | `sse4.1 -> avx512` |
| intel-icl | `sse4.1 -> avx512` | `sse4.1 -> avx512` | `sse4.1 -> avx512` | `sse4.1 -> avx512` |
| amd-zen5 | `sse4.1 -> avx512` | `sse4.1 -> avx512` | `sse4.1 -> avx512` | `sse4.1 -> avx512` |
| graviton3 | `neon -> neon` | `neon -> neon` | `neon -> neon` | `neon -> neon` |
| graviton4 | `neon -> neon` | `neon -> neon` | `neon -> neon` | `neon -> neon` |
| ibm-power10 | `portable -> portable` | `portable -> portable` | `portable -> portable` | `portable -> portable` |
| ibm-s390x | `portable -> portable` | `portable -> portable` | `portable -> portable` | `portable -> portable` |

### Root-Cause Signal (Phase 2 framing)
- x86 lanes show large positive oneshot-vs-kernel deltas at 256/1024, which isolates short-path overhead beyond pure kernel throughput.
- arm64 lanes show oneshot far better than kernel-ab at 256/1024, indicating current oneshot path is avoiding part of the kernel-ab short-input penalty.
- power10 is already ahead in kernel and oneshot at 1024+; only 256 remains a oneshot issue.
- s390x oneshot is close to parity while kernel-ab is far ahead/behind by size, reinforcing that dispatch/path selection needs targeted investigation before kernel rewrites.

### Phase 2 Local Kickoff (Apple arm64, 2026-02-21)
- Command: `cargo bench --profile bench -p hashes --bench blake3_short_input_attribution -- --noplot`
- Focus sizes: `64, 128, 256, 512, 1024`
- Local dispatch snapshot (stream64, plain): `portable+aarch64/neon` for all tested short sizes (`64..1024`), no parallelization expected/observed.

#### Short-Path Split (selected medians)

| Metric | 256 | 1024 |
|---|---:|---:|
| rscrypto plain full | 342.70 ns | 1.2769 us |
| official plain full | 304.15 ns | 1.2189 us |
| rscrypto gap vs official | +12.68% | +4.76% |
| rscrypto keyed vs plain | -0.89% | -0.25% |
| rscrypto derive vs plain | -0.62% | -0.09% |

#### Dispatch Overhead (selected medians)

| Metric | 256 | 1024 |
|---|---:|---:|
| oneshot auto | 342.90 ns | 1.2360 us |
| oneshot direct/portable | 311.71 ns | 1.2079 us |
| oneshot auto overhead | +10.00% | +2.33% |
| stream64 auto (plain) | 362.16 ns | 1.3890 us |
| stream64 direct/portable (plain) | 345.51 ns | 1.2995 us |
| stream64 auto overhead | +4.82% | +6.89% |

#### Local Conclusions
- Short-input loss is measurably front-loaded at 256B; gap shrinks by 1024B.
- Keyed/derive overhead inside rscrypto is not the dominant short-input penalty on this host.
- Auto-dispatch/control-path overhead is significant at short sizes and must be reduced before kernel-level changes.

### Phase 2 CI Capture (all enforced lanes, 2026-02-21)
- Canonical CI run: `22260772858` on `main` at `664401e` (`crates=hashes`, `benches=blake3_short_input_attribution`, `quick=false`, gates disabled).
- Earlier run `22260482068` failed only because the new bench target was not yet present on `main` at that SHA.

#### CI Summary (kernel-aware dispatch overhead)

| Lane | rs gap 256 | rs gap 1024 | oneshot kernel (256/1024) | oneshot dispatch overhead (256/1024) | stream kernel (256/1024) | stream dispatch overhead (256/1024) |
|---|---:|---:|---|---:|---|---:|
| amd-zen4 | +23.10% | +9.75% | `x86_64/avx2` / `x86_64/avx2` | +14.49% / -4.21% | `x86_64/sse4.1` / `x86_64/sse4.1` | +2.88% / +6.52% |
| intel-spr | +51.47% | +31.68% | `x86_64/avx2` / `x86_64/avx2` | +17.76% / -0.81% | `x86_64/sse4.1` / `x86_64/sse4.1` | +7.23% / +9.09% |
| intel-icl | +58.16% | +35.87% | `x86_64/avx2` / `x86_64/avx2` | +21.18% / +0.33% | `x86_64/sse4.1` / `x86_64/sse4.1` | +12.56% / +8.73% |
| amd-zen5 | +37.07% | +23.19% | `x86_64/avx2` / `x86_64/avx2` | +11.25% / -0.62% | `x86_64/sse4.1` / `x86_64/sse4.1` | -0.37% / -0.36% |
| graviton3 | +45.53% | +22.15% | `portable` / `portable` | +43.66% / +18.66% | `aarch64/neon` / `aarch64/neon` | +1.15% / +4.79% |
| graviton4 | +46.12% | +22.56% | `portable` / `portable` | +45.12% / +19.47% | `aarch64/neon` / `aarch64/neon` | +3.09% / +5.85% |
| ibm-power10 | +25.81% | +14.77% | `portable` / `powerpc64/vsx` | +22.03% / +27.50% | `portable` / `portable` | +2.35% / +7.87% |
| ibm-s390x | +17.66% | +6.03% | `portable` / `portable` | +18.82% / +4.31% | `portable` / `portable` | +1.62% / +6.00% |

#### CI Conclusions
- x86: short-input deficit remains severe; oneshot dispatch overhead at 256 is material (+11% to +21%), but cannot explain all gap alone.
- arm64: oneshot path is selecting `portable` at 256/1024 while stream path is already `neon`; this is a primary dispatch-policy target.
- power/s390x: both oneshot and stream frequently remain portable at short sizes; dispatch/path policy dominates before kernel micro-optimizations.
- 1024 overhead is often lower than 256, confirming Phase 3 should prioritize tiny/short path first.

### Phase 3 Progress (2026-02-21)
- Implemented (local): `PROFILE_AARCH64_SERVER_NEON` now maps oneshot size classes `xs/s` to `Aarch64Neon` (previously `Portable`), aligning Graviton-family oneshot short-input dispatch with lane-native SIMD capability.
- Rationale: CI Phase 2 showed Graviton3/4 oneshot selecting `portable` at `256/1024` while streaming already selected `neon`.
- Validation completed on `main`:
  - run `22261409598` (`blake3_short_input_attribution`, lanes: `graviton3`, `graviton4`, `quick=false`)
  - commit under test: `main` at `66457ee`

#### Arm64 Post-Change Snapshot (run `22261409598`)

| Lane | rs gap 256 | rs gap 1024 | oneshot kernel (256/1024) | oneshot dispatch overhead (256/1024) | stream kernel (256/1024) | stream dispatch overhead (256/1024) |
|---|---:|---:|---|---:|---|---:|
| graviton3 | +45.65% | +23.75% | `aarch64/neon` / `aarch64/neon` | +12.24% / +8.17% | `aarch64/neon` / `aarch64/neon` | +1.00% / +4.67% |
| graviton4 | +46.24% | +22.80% | `aarch64/neon` / `aarch64/neon` | +10.84% / +7.92% | `aarch64/neon` / `aarch64/neon` | +2.77% / +5.65% |

#### Arm64 Delta vs Prior Phase-2 CI (`22260772858`)
- Fixed: oneshot kernel selection at `256/1024` is now `neon` (was `portable` on both G3/G4).
- Fixed: oneshot auto-vs-direct overhead dropped from ~`+44-45% / +19%` to ~`+11-12% / +8%`.
- Remaining issue: absolute rs-vs-official gap at `256/1024` is still large (~`+46% / +23%`), so next gains must come from short-input compute path and/or oneshot setup work beyond kernel selection.

#### X86 Forced-SSE4.1 Sweep (run `22262543115`, 2026-02-21)
- Validation run completed on `main` at `66457ee` with `blake3_force_kernel=x86_64/sse4.1` and lanes: `amd-zen4`, `amd-zen5`, `intel-icl`, `intel-spr`.
- Purpose: isolate whether replacing current x86 oneshot short-input `avx2` choice with `sse4.1` should be a global dispatch-table retune.

| Lane | rs gap 256 (old → forced-sse4.1) | rs gap 1024 (old → forced-sse4.1) | oneshot dispatch overhead 256 (old → forced-sse4.1) | oneshot dispatch overhead 1024 (old → forced-sse4.1) |
|---|---:|---:|---:|---:|
| amd-zen4 | `+23.10% -> +21.06%` | `+9.75% -> +9.08%` | `+14.49% -> +0.46%` | `-4.21% -> -0.02%` |
| intel-spr | `+51.47% -> +56.77%` | `+31.68% -> +30.06%` | `+17.76% -> +2.52%` | `-0.81% -> -0.85%` |
| intel-icl | `+58.16% -> +58.16%` | `+35.87% -> +35.85%` | `+21.18% -> +3.36%` | `+0.33% -> +0.86%` |
| amd-zen5 | `+37.07% -> +38.92%` | `+23.19% -> +23.65%` | `+11.25% -> -0.04%` | `-0.62% -> +0.45%` |

#### X86 Sweep Conclusions
- Forcing `sse4.1` removes most measured auto-dispatch overhead at `256` on all x86 lanes.
- But absolute rs-vs-official short-input gap does **not** improve consistently: `zen4` improves, `icl` is flat, `spr` and `zen5` regress at `256`.
- Decision: do **not** apply a global x86 `avx2 -> sse4.1` short-size dispatch retune.
- Next action: optimize x86 short-input compute/setup path (`digest_one_chunk_root_hash_words_x86`) and then consider lane-specific threshold/kernel splits only where the data supports them.

### Phase 3 Candidate A (local-only, pending CI validation)
- Implemented on local workspace:
  - simplified one-chunk setup math in `digest_one_chunk_root_hash_words_x86` and `digest_one_chunk_root_hash_words_aarch64` (remove `div_ceil/max` path and derive `CHUNK_START` directly from `full_blocks == 0`),
  - removed tiny-input conversion roundtrip (`words -> bytes -> words`) in `digest_oneshot_words` by introducing `hash_tiny_to_root_words`.
- Guardrails:
  - no dispatch-table or threshold changes,
  - no large-input tree/reduction path changes.
- Local verification:
  - `cargo fmt --all --check`: pass
  - `cargo test -p hashes blake3:: --quiet`: pass
- Next required measurement:
  - CI `Bench` workflow on pushed SHA with `crates=hashes`, `benches=blake3_short_input_attribution`, `quick=false`, x86 lanes first (`amd-zen4`, `intel-spr`, `intel-icl`, `amd-zen5`), then arm64 to check for collateral deltas.

### Phase 3 Candidate A CI Validation (2026-02-21)
- Run: `22263080239` on `main` at `857485f` (`crates=hashes`, `benches=blake3_short_input_attribution`, all 8 enforced lanes, gates disabled).

| Lane | rs gap 256 | rs gap 1024 | oneshot kernel (256/1024) | oneshot dispatch overhead (256/1024) | stream kernel (256/1024) | stream dispatch overhead (256/1024) |
|---|---:|---:|---|---:|---|---:|
| amd-zen4 | +21.40% | +9.01% | `x86_64/avx2` / `x86_64/avx2` | +13.65% / -4.54% | `x86_64/sse4.1` / `x86_64/sse4.1` | +10.03% / +13.71% |
| intel-spr | +55.40% | +31.74% | `x86_64/avx2` / `x86_64/avx2` | +17.69% / -2.45% | `x86_64/sse4.1` / `x86_64/sse4.1` | +7.25% / +8.87% |
| intel-icl | +59.35% | +35.91% | `x86_64/avx2` / `x86_64/avx2` | +20.22% / -0.38% | `x86_64/sse4.1` / `x86_64/sse4.1` | +14.00% / +12.33% |
| amd-zen5 | +37.60% | +23.26% | `x86_64/avx2` / `x86_64/avx2` | +10.47% / -1.15% | `x86_64/sse4.1` / `x86_64/sse4.1` | -1.06% / -0.09% |
| graviton3 | +44.77% | +22.05% | `aarch64/neon` / `aarch64/neon` | +10.16% / +8.07% | `aarch64/neon` / `aarch64/neon` | +1.24% / +5.09% |
| graviton4 | +45.93% | +22.49% | `aarch64/neon` / `aarch64/neon` | +10.24% / +7.94% | `aarch64/neon` / `aarch64/neon` | +2.21% / +5.30% |
| ibm-power10 | +26.70% | +13.98% | `portable` / `powerpc64/vsx` | +20.95% / +27.77% | `portable` / `portable` | +4.82% / +9.23% |
| ibm-s390x | +16.51% | +6.84% | `portable` / `portable` | +18.07% / +4.17% | `portable` / `portable` | +1.38% / +4.14% |

#### Candidate A vs Prior CI (`22260772858` / arm compare note)
- Candidate A did **not** produce a cross-lane, repeatable short-gap improvement.
- Improvements were limited/partial:
  - `amd-zen4`: rs gap improved at `256/1024` (`+23.10 -> +21.40`, `+9.75 -> +9.01`).
  - `graviton3/4`: modest rs gap improvement at both sizes (~`0.3` to `1.7` points).
  - `ibm-s390x`: improved at `256` (`+17.66 -> +16.51`).
- Regressions/flat lanes remain:
  - `intel-spr` and `intel-icl` worsened at `256`.
  - `amd-zen5` worsened slightly at `256/1024`.
  - `ibm-power10` worsened at `256`.
- Conclusion: keep Candidate A as a correctness-preserving cleanup, but it is insufficient for gate closure; next gains must come from deeper short-input compute/setup improvements (especially x86 `avx2` one-chunk path) and lane-specific policy tuning backed by lane data.

### Phase 3 Candidate B (local-only, pending CI validation)
- Implemented on local workspace:
  - rewired x86 one-chunk root path (`digest_one_chunk_root_hash_words_x86`) to process prefix full blocks via direct `x86_compress_cv_bytes` loop instead of the generic `chunk_compress_blocks` function-pointer wrapper.
- Rationale:
  - this is a targeted short-input compute/setup reduction in the exact hotspot identified by Phase 2/3 x86 data (`avx2` one-shot path at `256/1024`).
  - no dispatch-table retune and no large-input path edits.
- Local verification:
  - `cargo fmt --all --check`: pass
  - `cargo test -p hashes blake3:: --quiet`: pass
- Next required measurement:
  - rerun CI `blake3_short_input_attribution` on pushed SHA, x86 lanes first (`amd-zen4`, `intel-spr`, `intel-icl`, `amd-zen5`), then full enforced matrix for parity/regression check.

### Phase 3 Candidate B x86 CI Validation (2026-02-21)
- Run: `22263827674` on `main` at `a39e33a` (`crates=hashes`, `benches=blake3_short_input_attribution`, x86 lanes only: `amd-zen4`, `intel-spr`, `intel-icl`, `amd-zen5`).

| Lane | rs gap 256 (A → B) | rs gap 1024 (A → B) | oneshot dispatch overhead 256 (A → B) | oneshot dispatch overhead 1024 (A → B) |
|---|---:|---:|---:|---:|
| amd-zen4 | `+21.40% -> +22.26%` | `+9.01% -> +9.27%` | `+13.65% -> +17.58%` | `-4.54% -> -0.18%` |
| intel-spr | `+55.40% -> +51.60%` | `+31.74% -> +30.50%` | `+17.69% -> +24.87%` | `-2.45% -> +3.98%` |
| intel-icl | `+59.35% -> +58.64%` | `+35.91% -> +35.82%` | `+20.22% -> +27.13%` | `-0.38% -> +6.28%` |
| amd-zen5 | `+37.60% -> +37.60%` | `+23.26% -> +23.44%` | `+10.47% -> +12.62%` | `-1.15% -> +0.74%` |

#### Candidate B x86 Conclusion
- Not accepted as-is.
- Although `spr/icl` showed small absolute rs-gap improvement, one-shot auto-vs-direct overhead worsened materially on **all** x86 lanes at `256/1024`.
- Net signal is inconsistent and does not advance reliable gate closure.
- Next action: revert to Candidate A baseline for x86 short path and pursue a lower-risk improvement that reduces one-shot control/setup overhead without increasing auto-vs-direct deltas.

### Phase 3 Candidate C (local-only, pending CI validation)
- Status:
  - Candidate B loop rewrite has been reverted in code.
  - New Candidate C is now implemented on top of Candidate A baseline.
- Implemented on local workspace:
  - x86 one-chunk root path keeps existing prefix processing (`chunk_compress_blocks`) but optimizes final-block handling:
    - remove unconditional zero-init of a 64-byte padded block on aligned-final-block path,
    - replace indirect final compress function-pointer call with direct kernel-specific calls (`sse4.1` / `avx2` / `avx512`, including AVX-512 asm path where enabled).
- Rationale:
  - target fixed overhead in the exact final-block step of short one-shot hashing without changing dispatch policy or widening per-block loop control paths.
  - lower risk than Candidate B: fewer moving parts and no multi-call rewrite of prefix blocks.
- Local verification:
  - `cargo fmt --all --check`: pass
  - `cargo test -p hashes blake3:: --quiet`: pass
- Next required measurement:
  - CI `blake3_short_input_attribution` on pushed SHA, x86 lanes first (`amd-zen4`, `intel-spr`, `intel-icl`, `amd-zen5`), compare against Candidate A run `22263080239`.

## Hard Targets
- Pass `blake3/oneshot` gap gate on all enforced platforms.
- Pass `blake3/kernel-ab` gate on all enforced platforms.
- No regressions at 4096+ sizes while fixing 256/1024.

## Work Plan

### Phase 1: Reproduce + Baseline
- [x] Capture fresh per-lane baselines for `oneshot` and `kernel-ab` across all enforced CI lanes.
- [x] Store baseline summaries (size, ours, official, required delta) in a single tracking table.
- [x] Re-run CI baseline on `main` with gate sequencing fixed so `kernel-ab` is emitted for all failing oneshot lanes.
- [x] Capture `streaming-dispatch` diagnostics per lane and attach to this file.
- [x] Confirm forced-kernel behavior per lane using `RSCRYPTO_BLAKE3_FORCE_KERNEL` sanity runs.

### Phase 2: Short-Input Root Cause
- [x] Instrument short path overhead split:
  - object/init/setup cost
  - tiny update path
  - finalize cost
  - keyed/derive branch overhead
- [x] Collect and summarize initial local Phase 2 measurements from `blake3_short_input_attribution` bench target.
- [x] Quantify dispatch overhead vs compute for lengths: `64, 128, 256, 512, 1024` on CI lanes.
- [x] Validate threshold behavior from `streaming_dispatch_info` for small inputs on CI lanes.

### Phase 3: Code Optimizations (Short Path First)
- [ ] Reduce oneshot setup overhead for tiny inputs.
- [ ] Minimize small-input branching and flag plumbing on hot path.
- [ ] Re-tune architecture-specific dispatch thresholds around 256/1024.
- [ ] Keep large-input path unchanged unless required by measured regressions.

### Phase 4: Lane-Specific Tightening
- [ ] x86_64 (ICL/SPR/Zen4/Zen5): tune short-length handoff between scalar/SSE/AVX families.
- [ ] aarch64 (G3/G4): tune NEON selection and short-update boundaries.
- [ ] s390x/powerpc64: verify kernel parity and ensure no accidental portable fallback.

### Phase 5: Lock-In
- [ ] Add/adjust deterministic perf guardrails where signal is stable.
- [ ] Update gate thresholds only when improvement is proven and repeatable.
- [ ] Document final threshold rationale and per-lane behavior.

## Acceptance Criteria
- [ ] CI gap gates pass on all enforced lanes.
- [ ] At least 5 independent reruns show no flaky gate failures.
- [ ] No functional regressions in differential tests/fuzz/proptests.

## Execution Commands
- `just bench-blake3-gap-gate`
- `BENCH_ENFORCE_BLAKE3_KERNEL_GATE=true BENCH_PLATFORM=<lane> scripts/bench/bench.sh crates=hashes benches=blake3 filter=kernel-ab quick=false`
- `RSCRYPTO_BLAKE3_BENCH_DIAGNOSTICS=1 cargo bench --profile bench -p hashes --bench blake3 -- oneshot,streaming-dispatch,kernel-ab`
- `cargo bench --profile bench -p hashes --bench blake3_short_input_attribution -- --noplot`

## Notes
- Do not accept ornamental abstractions in hot paths.
- Every optimization change must include before/after numbers for `256` and `1024` first, then full size sweep.
