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
- [ ] Quantify dispatch overhead vs compute for lengths: `64, 128, 256, 512, 1024` on CI lanes.
- [ ] Validate threshold behavior from `streaming_dispatch_info` for small inputs on CI lanes.

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
