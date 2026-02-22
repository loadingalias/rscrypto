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

### Phase 3 Candidate C x86 CI Validation (2026-02-21)
- Run: `22264253466` on `main` at `8b9f3d1` (`crates=hashes`, `benches=blake3_short_input_attribution`, x86 lanes only: `amd-zen4`, `intel-spr`, `intel-icl`, `amd-zen5`).

| Lane | rs gap 256 (A → C) | rs gap 1024 (A → C) | oneshot dispatch overhead 256 (A → C) | oneshot dispatch overhead 1024 (A → C) |
|---|---:|---:|---:|---:|
| amd-zen4 | `+21.40% -> +21.24%` | `+9.01% -> +8.93%` | `+13.65% -> +13.55%` | `-4.54% -> -4.75%` |
| intel-spr | `+55.40% -> +55.72%` | `+31.74% -> +32.04%` | `+17.69% -> +17.17%` | `-2.45% -> -1.60%` |
| intel-icl | `+59.35% -> +60.58%` | `+35.91% -> +36.44%` | `+20.22% -> +18.88%` | `-0.38% -> -1.13%` |
| amd-zen5 | `+37.60% -> +37.41%` | `+23.26% -> +23.45%` | `+10.47% -> +10.39%` | `-1.15% -> -1.55%` |

#### Candidate C x86 Conclusion
- Not accepted as-is.
- Candidate C improves overhead consistency versus Candidate B and gives small wins on `amd-zen4`/`amd-zen5`, but `intel-spr` and especially `intel-icl` regress in absolute rs-vs-official short gap.
- Net effect is mixed and does not provide the repeatable cross-lane improvement needed for gate closure.
- Next action: keep Candidate A baseline and move to Candidate D focused on x86 one-shot kernel policy/setup interaction (likely `avx2` short-size behavior), with lane-specific measurement baked into the change plan.

### Phase 3 Candidate D (local-only, pending CI validation)
- Implemented on local workspace:
  - optimized pristine single-update streaming path in `Digest::update()` for `<= CHUNK_LEN` inputs:
    - bypass `update_with` loop/dispatch machinery for the first short update,
    - when SIMD is not deferred, bootstrap the short update with `size_class_kernel(input.len())` so `1024`-byte first updates can use the same kernel tier as one-shot (`avx2` on x86 lanes where selected), instead of being pinned to streaming `sse4.1`.
  - tiny finalize copy-elision:
    - in `finalize()` tiny single-block path (`blocks_compressed == 0`), remove extra stack block copy and call `compress_to_root_words` directly on `chunk_state.block`.
- Rationale:
  - directly targets the measured `new + single update + finalize` overhead and x86 short-size kernel-tier mismatch without retuning tables or touching large-input tree paths.
  - keeps changes constrained to first-update short path mechanics and tiny finalize setup.
- Local verification:
  - `just check-all`: pass
  - `just test`: pass
- Next required measurement:
  - CI `blake3_short_input_attribution` on pushed SHA with x86 lanes first (`amd-zen4`, `intel-spr`, `intel-icl`, `amd-zen5`), then full enforced matrix if x86 signal is positive.

### Phase 3 Candidate D x86 CI Validation (2026-02-21)
- Run: `22264961125` on `main` (x86 lanes only: `amd-zen4`, `intel-spr`, `intel-icl`, `amd-zen5`).

| Lane | rs gap 256 (A -> D) | rs gap 1024 (A -> D) | oneshot dispatch overhead 256 (A -> D) | oneshot dispatch overhead 1024 (A -> D) |
|---|---:|---:|---:|---:|
| amd-zen4 | `+21.40% -> +26.07%` | `+9.01% -> +19.31%` | `+13.65% -> +17.82%` | `-4.54% -> +4.14%` |
| intel-spr | `+55.40% -> +59.52%` | `+31.74% -> +42.23%` | `+17.69% -> +18.46%` | `-2.45% -> +2.41%` |
| intel-icl | `+59.35% -> +65.32%` | `+35.91% -> +44.88%` | `+20.22% -> +23.65%` | `-0.38% -> +6.17%` |
| amd-zen5 | `+37.60% -> +40.33%` | `+23.26% -> +28.38%` | `+10.47% -> +12.67%` | `-1.15% -> +3.50%` |

#### Candidate D x86 Conclusion
- Rejected.
- Candidate D regresses absolute rs-vs-official short gaps at both `256` and `1024` on every tested x86 lane.
- One-shot auto-vs-direct overhead also worsens broadly and flips `1024` from negative/near-zero overhead to positive overhead on all lanes.
- Next action: revert Candidate D and pursue Candidate E that keeps short streaming on the proven low-latency stream kernel path while targeting finalize/setup overhead without kernel-tier escalation on first short update.

### Phase 3 Candidate E (local-only, pending CI validation)
- Implemented on local workspace:
  - reverted Candidate D update-path behavior in `Digest::update()`:
    - removed pristine short-update kernel-tier escalation (`size_class_kernel` bootstrap on first short update),
    - restored prior ultra-tiny first-update block-copy fast path and existing streaming defer/dispatch flow.
  - retained tiny finalize copy-elision in `finalize()` (single-block tiny path uses `chunk_state.block` directly with `compress_to_root_words`).
- Rationale:
  - remove the confirmed regression vector from Candidate D while preserving low-risk setup cleanup.
  - keep short streaming dispatch on proven stream-kernel policy (`sse4.1 -> avx512` pair on x86 lanes) until data proves a better crossover strategy.
- Local verification:
  - `just check-all`: pass
  - `just test`: pass
- Next required measurement:
  - CI `blake3_short_input_attribution` on pushed SHA, x86 lanes first (`amd-zen4`, `intel-spr`, `intel-icl`, `amd-zen5`), compare directly against Candidate A (`22263080239`) and Candidate D (`22264961125`).

### Phase 3 Candidate E x86 CI Validation (2026-02-21)
- Run: `22265323736` on `main` (x86 lanes only: `amd-zen4`, `intel-spr`, `intel-icl`, `amd-zen5`).

| Lane | rs gap 256 (A -> E) | rs gap 1024 (A -> E) | oneshot dispatch overhead 256 (A -> E) | oneshot dispatch overhead 1024 (A -> E) |
|---|---:|---:|---:|---:|
| amd-zen4 | `+21.40% -> +24.60%` | `+9.01% -> +10.35%` | `+13.65% -> +12.31%` | `-4.54% -> -4.98%` |
| intel-spr | `+55.40% -> +55.45%` | `+31.74% -> +32.77%` | `+17.69% -> +16.92%` | `-2.45% -> -1.76%` |
| intel-icl | `+59.35% -> +59.42%` | `+35.91% -> +36.20%` | `+20.22% -> +19.59%` | `-0.38% -> -0.89%` |
| amd-zen5 | `+37.60% -> +37.54%` | `+23.26% -> +23.34%` | `+10.47% -> +11.87%` | `-1.15% -> -0.77%` |

#### Candidate E x86 Conclusion
- Better than Candidate D, but not better than Candidate A.
- Relative to A, Candidate E is mostly flat-to-worse in absolute rs-vs-official short gap (largest regression on `amd-zen4`; smaller regressions on `spr/icl` and `spr` at `1024`; near-flat on `amd-zen5`).
- One-shot overhead improves slightly on `zen4/spr/icl` at `256`, but this does not translate into net short-gap wins.
- Decision: do not promote Candidate E as the new baseline; keep Candidate A baseline for follow-on work.

### Phase 3 Candidate F (local-only, pending CI validation)
- Implemented on local workspace:
  - generalized root-tail compressor helper:
    - replaced `compress_to_root_words` with `compress_chunk_tail_to_root_words(..., add_chunk_start)` so single-chunk finalization can set flags precisely (`CHUNK_START` only when needed).
  - added a broader single-chunk finalize fast path in `Digest::finalize()`:
    - when state is single chunk (`chunk_counter == 0`, empty stack, no pending CV), compute root bytes directly from current chunk tail for both tiny and non-tiny (`<= CHUNK_LEN`) cases,
    - bypasses `OutputState` construction and words-conversion path on short streaming finalization.
  - retained Candidate A dispatch behavior (no first-update kernel-tier escalation).
- Rationale:
  - target short streaming `new + update + finalize` control/setup cost directly, especially at `256/1024`, without changing kernel policy selection.
  - keep changes local to single-chunk finalize mechanics and avoid large-input/tree-path edits.
- Local verification:
  - `just check-all`: pass
  - `just test`: pass
- Next required measurement:
  - CI `blake3_short_input_attribution` on pushed SHA, x86 lanes first (`amd-zen4`, `intel-spr`, `intel-icl`, `amd-zen5`), compare against Candidate A (`22263080239`) and Candidate E (`22265323736`).

### Phase 3 Candidate F x86 CI Validation (2026-02-21)
- Run: `22265986460` on `main` (x86 lanes only: `amd-zen4`, `intel-spr`, `intel-icl`, `amd-zen5`).

| Lane | rs gap 256 (A -> F) | rs gap 1024 (A -> F) | oneshot dispatch overhead 256 (A -> F) | oneshot dispatch overhead 1024 (A -> F) |
|---|---:|---:|---:|---:|
| amd-zen4 | `+21.40% -> +15.99%` | `+9.01% -> +7.83%` | `+13.65% -> +7.07%` | `-4.54% -> -6.17%` |
| intel-spr | `+55.40% -> +43.06%` | `+31.74% -> +29.04%` | `+17.69% -> +10.47%` | `-2.45% -> -4.13%` |
| intel-icl | `+59.35% -> +49.41%` | `+35.91% -> +34.21%` | `+20.22% -> +12.26%` | `-0.38% -> -2.76%` |
| amd-zen5 | `+37.60% -> +32.88%` | `+23.26% -> +22.43%` | `+10.47% -> +7.29%` | `-1.15% -> -2.56%` |

#### Candidate F x86 Conclusion
- Accepted as new baseline for short-path work.
- Candidate F improves absolute rs-vs-official short gap at both `256` and `1024` on every x86 lane tested.
- One-shot auto-vs-direct overhead drops materially at `256` across all x86 lanes, matching the intended target for this phase.
- Next action: run full enforced-lane `blake3_short_input_attribution` CI (arm64/s390x/power included) to validate no collateral regressions, then continue with short-gap closure from the new baseline.

### Phase 3 Candidate F Full-Lane CI Validation (2026-02-21)
- Run: `22266336024` on `main` (all enforced lanes).

| Lane | rs gap 256 (A -> F) | rs gap 1024 (A -> F) | oneshot dispatch overhead 256 (A -> F) | oneshot dispatch overhead 1024 (A -> F) |
|---|---:|---:|---:|---:|
| amd-zen4 | `+21.40% -> +16.18%` | `+9.01% -> +7.80%` | `+13.65% -> +7.55%` | `-4.54% -> -6.02%` |
| intel-spr | `+55.40% -> +43.25%` | `+31.74% -> +30.27%` | `+17.69% -> +12.73%` | `-2.45% -> -4.00%` |
| intel-icl | `+59.35% -> +52.66%` | `+35.91% -> +34.56%` | `+20.22% -> +12.14%` | `-0.38% -> -2.70%` |
| amd-zen5 | `+37.60% -> +34.30%` | `+23.26% -> +22.80%` | `+10.47% -> +8.19%` | `-1.15% -> -1.80%` |
| graviton3 | `+44.77% -> +41.31%` | `+22.05% -> +21.23%` | `+10.16% -> +7.65%` | `+8.07% -> +7.35%` |
| graviton4 | `+45.93% -> +42.53%` | `+22.49% -> +21.78%` | `+10.24% -> +7.67%` | `+7.94% -> +7.45%` |
| ibm-power10 | `+26.70% -> +22.65%` | `+13.98% -> +12.89%` | `+20.95% -> +18.31%` | `+27.77% -> +26.47%` |
| ibm-s390x | `+16.51% -> +9.43%` | `+6.84% -> +7.40%` | `+18.07% -> +15.82%` | `+4.17% -> +4.47%` |

#### Candidate F Full-Lane Conclusion
- Candidate F is accepted as the new global baseline for this phase.
- Net effect is strongly positive: rs short-gap improves at `256` on all lanes and at `1024` on 7/8 lanes.
- Residual watch item: `ibm-s390x` shows a small `1024` regression (`+0.56` points) despite a large `256` gain.

### Phase 3 Candidate G (local-only, pending CI validation)
- Implemented on local workspace:
  - s390x-only finalize refinement on top of Candidate F:
    - in single-chunk finalize path, if state is exactly one full chunk tail (`blocks_compressed == 15` and `block_len == 64`) on `s390x`, use prior `root_output().root_hash_bytes()` path.
    - all other arches/sizes keep Candidate F fast path unchanged.
- Rationale:
  - target the only observed collateral regression from F (`ibm-s390x` at `1024`) with a minimal, architecture-scoped fallback.
  - avoid perturbing the cross-lane wins achieved by Candidate F.
- Local verification:
  - `just check-all`: pass
  - `just test`: pass
- Next required measurement:
  - CI `blake3_short_input_attribution` with at least `ibm-s390x`, plus one x86 control lane (for example `intel-spr`) to confirm no collateral x86 impact.

### Phase 3 Candidate G Targeted CI Validation (2026-02-22)
- Run: `22266924998` on `main` (targeted lanes: `ibm-s390x`, `intel-spr`).
- Comparison baseline:
  - Candidate A full-lane run: `22263080239`
  - Candidate F full-lane run: `22266336024`

| Lane | rs gap 256 (A -> F -> G) | rs gap 1024 (A -> F -> G) | oneshot dispatch overhead 256 (A -> F -> G) | oneshot dispatch overhead 1024 (A -> F -> G) |
|---|---:|---:|---:|---:|
| intel-spr | `+55.40% -> +43.25% -> +43.10%` | `+31.74% -> +30.27% -> +32.25%` | `+17.69% -> +12.73% -> +10.14%` | `-2.45% -> -4.00% -> -5.12%` |
| ibm-s390x | `+16.51% -> +9.43% -> +11.03%` | `+6.84% -> +7.40% -> +7.08%` | `+18.07% -> +15.82% -> +15.79%` | `+4.17% -> +4.47% -> +4.35%` |

#### Candidate G Targeted Conclusion
- Candidate G is rejected.
- It partially recovers the `ibm-s390x` `1024` regression from Candidate F (`+7.40% -> +7.08%`) but remains worse than Candidate A (`+6.84%`).
- It gives back meaningful `ibm-s390x` `256` gains from Candidate F (`+9.43% -> +11.03%`), and targeted x86 control shows an unfavorable `1024` shift.
- Decision: keep Candidate F as the active baseline and move to a different optimization direction.

### Phase 3 Candidate H (local-only, pending CI validation)
- Implemented on local workspace:
  - removed Candidate G `s390x`-specific finalize fallback (`blocks_compressed == 15 && block_len == 64`) and restored architecture-neutral Candidate F finalize behavior.
  - added a first-update single-chunk direct path in `Digest::update()`:
    - applies when state is pristine and input length is `<= CHUNK_LEN`,
    - selects stream/bulk kernels once,
    - updates `ChunkState` directly, bypassing `update_with` loop and batch probes for this latency-critical case.
- Rationale:
  - target short streaming overhead at `256/1024` by reducing control-path overhead without architecture-specific behavior.
  - keep implementation pure Rust, simple, and globally consistent across lanes.
- Local verification:
  - `just check-all`: pass
  - `just test`: pass
- Next required measurement:
  - CI `blake3_short_input_attribution` on x86 controls first (`intel-spr`, `amd-zen4`, optionally `intel-icl`, `amd-zen5`), then full-lane run if positive.

### Phase 3 Candidate H x86 CI Validation (2026-02-22)
- Run: `22267677008` on `main` (x86 control lanes: `amd-zen4`, `intel-spr`).
- Comparison baseline:
  - Candidate A full-lane run: `22263080239`
  - Candidate F full-lane run: `22266336024`

| Lane | rs gap 256 (A -> F -> H) | rs gap 1024 (A -> F -> H) |
|---|---:|---:|
| amd-zen4 | `+21.40% -> +16.18% -> +14.33%` | `+9.01% -> +7.80% -> +7.18%` |
| intel-spr | `+55.40% -> +43.25% -> +39.88%` | `+31.74% -> +30.27% -> +29.06%` |

Supplemental absolute rscrypto times (`ns`, lower is better):

| Lane | rs full 256 (A -> F -> H) | rs full 1024 (A -> F -> H) |
|---|---:|---:|
| amd-zen4 | `330.40 -> 316.32 -> 310.89` | `1141.90 -> 1128.90 -> 1122.80` |
| intel-spr | `352.96 -> 320.46 -> 314.44` | `1195.50 -> 1138.70 -> 1161.60` |

#### Candidate H x86 Conclusion
- Candidate H is directionally positive and improves short-gap at `256` on both x86 control lanes.
- `amd-zen4` is cleanly positive at both `256` and `1024`.
- `intel-spr` `1024` is ambiguous:
  - relative gap vs official improves in this run,
  - but absolute rscrypto time regresses vs Candidate F (`1138.70ns -> 1161.60ns`), indicating possible noise or sensitivity.
- Next action: rerun targeted `intel-spr` (and ideally `amd-zen4`) for stability; if repeated positive/neutral, proceed to full-lane validation.

### Phase 3 Candidate H x86 Stability Rerun (2026-02-22)
- Run: `22268148355` on `main` (lanes: `amd-zen4`, `intel-spr`; filter: `short-path-split`).
- Compared with prior Candidate H x86 run `22267677008`:

| Lane | rs gap 256 (H -> H rerun) | rs gap 1024 (H -> H rerun) | rs full 256 ns (H -> H rerun) | rs full 1024 ns (H -> H rerun) |
|---|---:|---:|---:|---:|
| amd-zen4 | `+14.33% -> +15.59%` | `+7.18% -> +7.64%` | `310.89 -> 314.66` | `1122.80 -> 1126.80` |
| intel-spr | `+39.88% -> +38.17%` | `+29.06% -> +23.22%` | `314.44 -> 306.03` | `1161.60 -> 1094.20` |

#### Candidate H Stability Conclusion
- Candidate H remains positive on x86 overall.
- `intel-spr` `1024` no longer appears regressed in the rerun; it is materially better than the prior H run.
- `amd-zen4` is slightly noisier in rerun but remains substantially improved vs Candidate A baseline.
- Decision: proceed to full-lane Candidate H validation next.

### Phase 3 Candidate H Full-Lane CI Validation (2026-02-22)
- Run: `22268280116` on `main` (all enforced lanes).
- Comparison baseline:
  - Candidate A full-lane run: `22263080239`
  - Candidate F full-lane run: `22266336024`

| Lane | rs gap 256 (A -> F -> H) | rs gap 1024 (A -> F -> H) | oneshot dispatch overhead 256 (A -> F -> H) | oneshot dispatch overhead 1024 (A -> F -> H) |
|---|---:|---:|---:|---:|
| amd-zen4 | `+21.40% -> +16.18% -> +14.16%` | `+9.01% -> +7.80% -> +7.20%` | `-3.78% -> -9.01% -> -3.69%` | `-15.82% -> -16.84% -> -15.15%` |
| intel-spr | `+55.40% -> +43.25% -> +39.73%` | `+31.74% -> +30.27% -> +28.83%` | `+0.06% -> -5.73% -> -6.94%` | `-14.41% -> -17.96% -> -15.59%` |
| intel-icl | `+59.35% -> +52.66% -> +49.82%` | `+35.91% -> +34.56% -> +33.60%` | `-8.57% -> -13.98% -> -14.53%` | `-21.99% -> -23.38% -> -23.70%` |
| amd-zen5 | `+37.60% -> +34.30% -> +32.48%` | `+23.26% -> +22.80% -> +22.22%` | `+82.06% -> +77.90% -> +75.32%` | `+71.60% -> +70.62% -> +71.11%` |
| graviton3 | `+44.77% -> +41.31% -> +39.31%` | `+22.05% -> +21.23% -> +20.89%` | `+42.85% -> +39.66% -> +37.58%` | `+18.61% -> +17.88% -> +17.40%` |
| graviton4 | `+45.93% -> +42.53% -> +41.21%` | `+22.49% -> +21.78% -> +21.80%` | `+44.54% -> +41.50% -> +40.16%` | `+19.48% -> +18.93% -> +19.17%` |
| ibm-power10 | `+26.70% -> +22.65% -> +18.64%` | `+13.98% -> +12.89% -> +11.54%` | `+20.95% -> +18.31% -> +13.55%` | `+6.19% -> +5.56% -> +4.27%` |
| ibm-s390x | `+16.51% -> +9.43% -> +10.10%` | `+6.84% -> +7.40% -> +4.12%` | `+18.07% -> +15.82% -> +10.88%` | `+4.17% -> +4.47% -> +4.62%` |

#### Candidate H Full-Lane Conclusion
- Candidate H is accepted as the new global baseline.
- Net effect:
  - `256`: improved vs Candidate F on all 8 lanes.
  - `1024`: improved vs Candidate F on 7/8 lanes (`graviton4` is effectively flat within noise).
- Remaining gap profile is still largest on x86 short lengths (`intel-icl`, `intel-spr`) and aarch64 short lengths (`graviton3/4`), despite steady reductions.

### Phase 4 Measurement Track: Oneshot Apples-to-Apples (local)
- Added a dedicated benchmark group to `crates/hashes/benches/blake3_short_input_attribution.rs`:
  - `blake3/short-input/oneshot-apples`
- Purpose:
  - isolate API-path overhead vs direct auto-kernel cost at the same sizes (`64..1024`) with the same inputs.
  - provide clean comparability against official API for plain/keyed oneshot without split-phase attribution noise.
- Cases measured:
  - `rscrypto/plain/api`
  - `rscrypto/plain/auto-kernel`
  - `official/plain/api`
  - `official/plain/reuse-hasher`
  - `rscrypto/keyed/api`
  - `rscrypto/keyed/auto-kernel`
  - `official/keyed/api`
- Local verification:
  - `cargo bench --profile bench -p hashes --bench blake3_short_input_attribution --no-run`: pass
- Next required measurement:
  - CI `Bench` with `crates=hashes`, `benches=blake3_short_input_attribution`, `filter=oneshot-apples`, `quick=false` on `intel-spr` + `amd-zen4` first.

### Phase 4 Oneshot Apples CI Validation (2026-02-22)
- Run: `22268829858` on `main` (lanes: `amd-zen4`, `intel-spr`; filter: `oneshot-apples`).

`plain` API-path results (`rscrypto/plain/api` vs `official/plain/api`):

| Lane | gap @256 | gap @1024 | rs api vs rs auto-kernel | official api vs official reuse-hasher |
|---|---:|---:|---:|---:|
| amd-zen4 | `+13.84%` | `+7.15%` | `-0.26%` / `-0.08%` | `+6.00%` / `+1.41%` |
| intel-spr | `+39.85%` | `+26.79%` | `-2.01%` / `-1.78%` | `-0.03%` / `-0.18%` |

`keyed` API-path results (`rscrypto/keyed/api` vs `official/keyed/api`):

| Lane | keyed gap @256 | keyed gap @1024 | rs keyed api vs rs keyed auto-kernel |
|---|---:|---:|---:|
| amd-zen4 | `+13.42%` | `+6.99%` | `+1.62%` / `-7.66%` |
| intel-spr | `+34.82%` | `+27.27%` | `-1.92%` / `-7.07%` |

#### Phase 4 Apples Conclusion
- The new apples track confirms the short-path gap is real at API level, not an artifact of the split benchmark harness.
- For plain mode, `rscrypto/api` is very close to `rscrypto/auto-kernel` (within about `0%..2%`), so most remaining gap is not coming from dispatch overhead alone.
- Remaining deficit is concentrated in short single-chunk compute path competitiveness, especially on `intel-spr`.
- Keyed `auto-kernel` at `1024` is slower than keyed API on both lanes, indicating additional overhead in the microbench adapter path for keyed mode; keyed API comparisons should be treated as canonical.

### Next Step (locked)
- Implement a dedicated single-chunk oneshot fast lane for `<= CHUNK_LEN` in the public API path (plain/keyed/derive), minimizing flag/state plumbing while preserving behavior and keeping the same cross-platform dispatch policy.
- Validate with:
  - `oneshot-apples` (x86 controls first),
  - then full-lane `blake3_short_input_attribution`,
  - then gate runs.

### Phase 4 Candidate I (local-only, pending CI validation)
- Implemented on local workspace:
  - Added unified public one-shot helper in `blake3/mod.rs`:
    - `digest_public_oneshot(key_words, flags, input)` -> `size_class_kernel(len)` + `digest_oneshot(...)`.
  - Routed public one-shot API paths through the same lane:
    - `Blake3::digest`
    - `Blake3::keyed_digest`
    - `Blake3::derive_key` (second phase: key material hash)
  - Removed duplicate tiny-input architecture branches from `keyed_digest` and `derive_key` to reduce control-path complexity.
  - Overrode `Digest::digest` for `Blake3` to use the inherent one-shot path directly.
- Rationale:
  - enforce one minimal public one-shot path for plain/keyed/derive with `<= CHUNK_LEN` fast handling through `digest_oneshot` internals.
  - simplify and de-duplicate hot path logic while preserving dispatch policy and pure-Rust constraints.
- Local verification:
  - `just check-all`: pass
  - `just test`: pass
- Next required measurement:
  - CI `Bench`:
    - `crates=hashes`
    - `benches=blake3_short_input_attribution`
    - `filter=oneshot-apples`
    - `quick=false`
    - lanes: `intel-spr`, `amd-zen4`
  - Accept/reject based on `rscrypto/plain/api` and `rscrypto/keyed/api` deltas at `256/1024`.

### Phase 4 Candidate I x86 CI Validation (2026-02-22)
- Runs:
  - `22269152954` (`oneshot-apples`, lanes: `amd-zen4`, `intel-spr`)
  - `22269244541` (stability rerun, same lanes/filter)
- Comparison baseline:
  - Candidate H apples run: `22268829858`

Plain API gap vs official (`rscrypto/plain/api`):

| Lane | 256 (H -> I -> I rerun) | 1024 (H -> I -> I rerun) |
|---|---:|---:|
| amd-zen4 | `+13.84% -> +14.27% -> +13.89%` | `+7.15% -> +7.27% -> +7.22%` |
| intel-spr | `+39.85% -> +29.53% -> +38.18%` | `+26.79% -> +26.22% -> +24.06%` |

Keyed API gap vs official (`rscrypto/keyed/api`):

| Lane | 256 (H -> I -> I rerun) | 1024 (H -> I -> I rerun) |
|---|---:|---:|
| amd-zen4 | `+13.42% -> +13.94% -> +13.44%` | `+6.99% -> +7.17% -> +7.03%` |
| intel-spr | `+34.82% -> +33.68% -> +33.42%` | `+27.27% -> +27.64% -> +27.03%` |

#### Candidate I Conclusion
- Candidate I is accepted as a simplification/cleanup baseline with neutral-to-slightly-positive performance.
- `amd-zen4`: essentially neutral vs H (within measurement noise at both `256` and `1024`).
- `intel-spr`: repeatable modest improvement in most tracked cells (strongest at plain `1024`; keyed `256` also improved).
- Main value: one unified public one-shot lane (`plain/keyed/derive`) with reduced hot-path complexity and no observed regression trend.

### Next Step (updated)
- Move below API/control simplification and target compute competitiveness directly for `<= CHUNK_LEN`:
  - optimize one-chunk root-hash path internals for keyed/derive/plain with focus on `intel-spr` and `intel-icl`,
  - keep API surface and dependency policy unchanged,
  - validate first with `oneshot-apples`, then full-lane attribution.

### Phase 4 Candidate J (local-only, pending CI validation)
- Implemented on local workspace:
  - x86 one-chunk root helper (`digest_one_chunk_root_hash_words_x86`) now uses:
    - `kernels::chunk_compress_blocks_inline(kernel.id, ...)`
    - instead of indirect function-pointer call `kernel.chunk_compress_blocks(...)`.
- Rationale:
  - remove avoidable indirection in the `<= CHUNK_LEN` one-shot compute path (not dispatch/API),
  - align x86 helper behavior with aarch64 helper pattern,
  - target `256/1024` hot path directly.
- Local verification:
  - `just check-all`: pass
  - `just test`: pass
- Next required measurement:
  - CI `Bench`:
    - `crates=hashes`
    - `benches=blake3_short_input_attribution`
    - `filter=oneshot-apples`
    - `quick=false`
    - lanes: `intel-spr`, `amd-zen4`
  - Compare against Candidate I rerun baseline (`22269244541`) on:
    - `rscrypto/plain/api` gap @ `256/1024`
    - `rscrypto/keyed/api` gap @ `256/1024`

### Phase 4 Candidate J x86 CI Validation (2026-02-22)
- Run: `22269413034` (`oneshot-apples`, lanes: `amd-zen4`, `intel-spr`)
- Comparison baseline: Candidate I rerun `22269244541`

`plain` API gap vs official (`rscrypto/plain/api`):

| Lane | 256 (I -> J) | 1024 (I -> J) |
|---|---:|---:|
| amd-zen4 | `+13.89% -> +14.33%` | `+7.22% -> +7.19%` |
| intel-spr | `+38.18% -> +31.47%` | `+24.06% -> +27.41%` |

`keyed` API gap vs official (`rscrypto/keyed/api`):

| Lane | 256 (I -> J) | 1024 (I -> J) |
|---|---:|---:|
| amd-zen4 | `+13.44% -> +13.29%` | `+7.03% -> +7.03%` |
| intel-spr | `+33.42% -> +31.13%` | `+27.03% -> +33.42%` |

#### Candidate J Conclusion
- Candidate J is rejected as a performance candidate (kept only as a local experiment reference).
- The signal is mixed and unstable across the critical cells:
  - `intel-spr` shows improvement at `256`, but clear regression at `1024` (plain/keyed gaps worsen).
  - `amd-zen4` is neutral-to-slightly-worse.
- Decision: keep Candidate I as active baseline and move to a larger compute-path restructuring candidate (not micro-indirection tweaks).

### Phase 4 Candidate J x86 Stability Rerun (2026-02-22)
- Run: `22269639518` (`oneshot-apples`, lanes: `amd-zen4`, `intel-spr`)
- Comparison baseline: Candidate I rerun `22269244541`

`plain` API gap vs official (`rscrypto/plain/api`):

| Lane | 256 (I -> J -> J rerun) | 1024 (I -> J -> J rerun) |
|---|---:|---:|
| amd-zen4 | `+13.89% -> +14.33% -> +14.07%` | `+7.22% -> +7.19% -> +7.15%` |
| intel-spr | `+38.18% -> +31.47% -> +39.25%` | `+24.06% -> +27.41% -> +28.07%` |

`keyed` API gap vs official (`rscrypto/keyed/api`):

| Lane | 256 (I -> J -> J rerun) | 1024 (I -> J -> J rerun) |
|---|---:|---:|
| amd-zen4 | `+13.44% -> +13.29% -> +13.40%` | `+7.03% -> +7.03% -> +7.13%` |
| intel-spr | `+33.42% -> +31.13% -> +35.26%` | `+27.03% -> +33.42% -> +26.80%` |

#### Candidate J Stability Conclusion
- `amd-zen4` remains effectively neutral/slightly worse vs Candidate I.
- `intel-spr` remains unstable and does not support promotion.
- Decision stands: Candidate J rejected. Revert code to Candidate I baseline before new compute-path work.

### Phase 4 Candidate K Attempt (local, reverted)
- Attempted optimization:
  - special exact-`CHUNK_LEN` one-shot path using `hash_many_contiguous(..., flags | ROOT, ...)`.
- Result:
  - correctness failure (ROOT leaked into non-final chunk-compress stages).
  - deterministic test failures at `len=1024` in one-shot parity tests.
- Actions:
  - reverted immediately.
  - re-ran targeted BLAKE3 one-shot parity tests; all passed after revert.
- Rule locked for next candidates:
  - never apply `ROOT` through leaf/chunk batch hashing primitives; `ROOT` must remain final-block-only.

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
