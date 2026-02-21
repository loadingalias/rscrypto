# BLAKE3 Update Plan

## Goal
Close the remaining BLAKE3 gaps so `rscrypto` is consistently at or ahead of official in the enforced CI lanes, with special focus on 256/1024-byte workloads.

## Current State
- Biggest deficits remain short-input (256/1024), especially on x86 and arm64 kernel-ab lanes.
- Full-lane baseline capture below was run on fix branch `codex/blake3-full-baseline-capture` at `b0af07ccbfc37b66a7b9b3c0eeb53403b2e93902` via workflow run `22258241064`.

## Full Baseline Capture (2026-02-21)

### Source Runs
- CI: `Bench` workflow run `22258241064` (all 8 lanes enabled, `enforce_blake3_gap_gate=true`, `enforce_blake3_kernel_gate=true`).
- Local (Apple arm64 host): `just bench-blake3-gate` and `just bench-blake3-kernel-gate graviton4`.

### Oneshot Gate Need (%; positive means rscrypto is behind official)

| Lane | 256 | 1024 | 4096 | 16384 | 65536 |
|---|---:|---:|---:|---:|---:|
| amd-zen4 | +7.27 | +14.61 | +7.63 | +4.99 | +0.91 |
| intel-spr | +30.28 | +34.57 | +9.73 | +3.49 | +1.41 |
| intel-icl | +34.23 | +36.87 | +8.81 | +4.47 | +2.13 |
| amd-zen5 | +24.83 | +24.59 | +6.41 | +3.42 | +1.14 |
| graviton3 | +3.74 | +3.69 | -5.30 | -10.06 | -11.37 |
| graviton4 | +3.24 | +3.13 | -6.11 | -10.61 | -11.79 |
| ibm-power10 | +9.89 | -9.21 | -64.30 | -67.35 | -68.14 |
| ibm-s390x | +2.42 | +1.45 | +2.52 | +2.34 | +0.32 |
| local-apple-arm64 | +2.65 | +2.09 | +7.43 | +2.13 | +0.48 |

### Kernel-ab Gate Need (%; best lane-native rscrypto kernel vs official)

| Lane | 256 | 1024 | 4096 | 16384 | 65536 | Notes |
|---|---:|---:|---:|---:|---:|---|
| amd-zen4 | -0.24 | +3.70 | +6.52 | +3.73 | +0.20 | `rscrypto/x86_64/sse4.1` best at 256/1024 on this lane |
| intel-spr | +18.01 | +17.82 | +4.63 | +1.97 | +0.01 | `rscrypto/x86_64/avx512` |
| intel-icl | +19.02 | +16.94 | +6.51 | +3.63 | +1.57 | `rscrypto/x86_64/avx512` |
| amd-zen5 | +15.13 | +9.38 | +5.27 | +3.07 | +1.14 | `rscrypto/x86_64/avx512` |
| graviton3 | +31.91 | +13.52 | -5.15 | -10.02 | -11.30 | `rscrypto/aarch64/neon` |
| graviton4 | +33.32 | +13.84 | -5.72 | -10.48 | -11.75 | `rscrypto/aarch64/neon` |
| ibm-power10 | -9.89 | -10.43 | -64.92 | -67.60 | -68.26 | `rscrypto/powerpc64/vsx` (ahead) |
| ibm-s390x | +12.91 | +8.22 | -56.05 | -59.35 | -59.17 | `rscrypto/s390x/vector` now emitted |
| local-apple-arm64 (graviton4 thresholds) | +28.16 | +8.26 | +7.17 | +2.02 | +0.63 | `rscrypto/aarch64/neon` |

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

## Hard Targets
- Pass `blake3/oneshot` gap gate on all enforced platforms.
- Pass `blake3/kernel-ab` gate on all enforced platforms.
- No regressions at 4096+ sizes while fixing 256/1024.

## Work Plan

### Phase 1: Reproduce + Baseline
- [x] Capture fresh per-lane baselines for `oneshot` and `kernel-ab` across all enforced CI lanes.
- [x] Store baseline summaries (size, ours, official, required delta) in a single tracking table.
- [x] Re-run CI baseline on a branch containing gate-sequencing fixes so `kernel-ab` is emitted for all failing oneshot lanes.
- [x] Capture `streaming-dispatch` diagnostics per lane and attach to this file.
- [ ] Confirm forced-kernel behavior per lane using `RSCRYPTO_BLAKE3_FORCE_KERNEL` sanity runs.

### Phase 2: Short-Input Root Cause
- [ ] Instrument short path overhead split:
  - object/init/setup cost
  - tiny update path
  - finalize cost
  - keyed/derive branch overhead
- [ ] Quantify dispatch overhead vs compute for lengths: `64, 128, 256, 512, 1024`.
- [ ] Validate threshold behavior from `streaming_dispatch_info` for small inputs.

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

## Notes
- Do not accept ornamental abstractions in hot paths.
- Every optimization change must include before/after numbers for `256` and `1024` first, then full size sweep.
