# BLAKE3 Performance Plan (Locked Baseline + Next Work)

## Objective
Ship a clean, maintainable, world-class BLAKE3 implementation that wins `oneshot` and `kernel-ab` against upstream across enforced CI lanes, without adding usage complexity or ornamental code.

## Current Baseline (Locked)
- Active baseline: **Candidate I line** (latest accepted simplification/perf baseline before rejected N/O policy experiments).
- Status: Candidate N and Candidate O were tested, rejected, and rolled back because they did not provide stable cross-lane wins.
- Validation on this rollback state:
  - `just check-all`: pass
  - `just test`: pass

## What We Have Proven
- Main remaining deficit is still short input (`256`, `1024`) competitiveness, not broad large-input throughput.
- We have already captured enough evidence that random boundary-policy churn is low ROI.
- Some short-path/control-path cleanups helped, but they did not close the gap alone.
- Kernel-policy changes that improve one lane often regress others; global x86 retunes were unstable.

## Locked Boundary/Policy Decisions
These are frozen until new data proves otherwise.

1. Keep current dispatch boundary policy from the accepted baseline.
2. Do **not** globally retune x86 short-size boundaries (no blanket `avx2 -> sse4.1` shift).
3. Do **not** merge per-lane special cases into runtime policy unless they are repeatably net-positive across reruns.
4. Keep large-input behavior stable while we optimize short-input compute paths.
5. Any future boundary change must be justified by kernel-only data first, then full API-path data.

## What We Explicitly Ruled Out
- More alphabet-style policy candidates without a hard kernel root-cause target.
- Changes that only move dispatch overhead around without improving API-level short-size gaps.
- Architecture-specific one-off exceptions that complicate the code but do not hold up in reruns.

## Working Hypothesis (Forward)
The remaining gap is primarily **kernel compute quality and generated code quality** on hot short-size paths, not missing dispatch knobs.

## Tooling Stack (Rust Ecosystem, macOS M1 Compatible)
We will use this exact stack unless a tool is demonstrably better.

1. `cargo bench` (Criterion harness in-repo)
   - Why: canonical, reproducible benchmark surface already wired into CI/gates.
   - Host support: native on macOS aarch64.
2. `cargo-asm`
   - Why: fastest way to inspect emitted assembly for specific Rust symbols on host target.
   - Host support: works on macOS M1 for aarch64 output.
3. `cargo-show-asm`
   - Why: complementary assembly/LLVM/MIR views and better ergonomics for some symbol flows.
   - Host support: works on macOS M1.
4. `cargo-llvm-lines`
   - Why: quantify code-size/inlining bloat in hot paths; prevents accidental complexity inflation.
   - Host support: works on macOS M1.
5. `cargo-samply` (sample profiler)
   - Why: low-friction CPU sampling to identify real time sinks in short-path code.
   - Host support: supported on macOS; good first-pass profiler.

Optional additions only if needed and justified:
- Linux CI perf tools (`perf`, `pmu-tools`) for cycle/uop-level confirmation on x86 runners.
- VTune or `llvm-mca` only when specific microarchitectural questions remain unanswered by the above.

## Execution Model (Host vs CI)
- Host machine (macOS M1):
  - Fast inner loop for correctness, short-path microbench trends, codegen inspection, and profiling.
- CI runners (Linux/Windows/IBM):
  - Source of truth for cross-arch competitiveness and final pass/fail decisions.
  - Kernel-only and full API-path comparisons must run on the same runner classes as gates.

## Immediate Work Plan

### Phase 1: Reconfirm Kernel Truth (No Policy Edits)
1. Run kernel-only A/B (`filter=kernel-ab`, `quick=false`) on enforced x86 and arm lanes.
2. Compare rscrypto kernels vs upstream on identical runners.
3. Produce per-lane table: throughput + cycles (where available), by key sizes (`256`, `1024`, `4096+`).

Exit criteria:
- We can state exactly which kernels lose, by how much, on which lanes.

### Phase 2: Kernel Quality Audit
1. For losing kernels, inspect hot symbols with `cargo-asm` and `cargo-show-asm`.
2. Audit instruction mix, register pressure, spills, branch shape, and unnecessary flag/setup overhead.
3. Use `cargo-llvm-lines` to detect inlining/code-size issues hurting I-cache/front-end.
4. Profile short-path benchmarks with `cargo-samply` to confirm top cycle consumers.

Exit criteria:
- Ranked list of concrete kernel/codegen defects with expected impact.

### Phase 3: Targeted Kernel Improvements
1. Implement minimal, high-impact changes per kernel family (x86 AVX2/AVX512, SSE4.1 where relevant; aarch64 NEON).
2. Keep API surface unchanged and avoid new dispatch complexity.
3. Validate each change in this order:
   - correctness (`just check-all`, `just test`)
   - kernel-only bench
   - oneshot/API bench
   - full-lane CI confirmation

Exit criteria:
- Cross-lane net win at `256/1024` with no meaningful regression at `4096+`.

### Phase 4: Only Then Revisit Policy (If Needed)
- Reopen boundary/policy tuning only if kernel improvements plateau and data shows a clear, repeatable opportunity.

## Benchmark Protocol (Required)
For every candidate:
1. `just check-all && just test`
2. CI kernel-only bench on same runners as upstream comparison
3. CI oneshot/API-path bench (`blake3_short_input_attribution`, `oneshot-apples` where relevant)
4. At least one stability rerun for ambiguous deltas
5. Decision: keep, revert, or iterate

## Definition of Done
- `blake3/kernel-ab` and `blake3/oneshot` gates pass on enforced lanes.
- Wins are repeatable (no fragile one-run artifacts).
- Code remains simpler or equal in complexity vs baseline.
- No new public API complexity.

## Commit Discipline
- One candidate per commit.
- Commit message format: `hashes: <short action summary>`.
- Every candidate commit must include linked CI run IDs and a keep/reject decision in PR notes.

## Non-Negotiables
- Measure first, optimize second.
- No ornamental abstractions in hot paths.
- If a change is not clearly faster and cleaner, it does not ship.
