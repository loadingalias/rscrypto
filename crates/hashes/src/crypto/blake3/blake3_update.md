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

### 2026-02-22 - Candidate R (planned)
- Goal:
  - Keep the `s390x` kernel-ab gain from Candidate Q, improve `power10` oneshot `256`, and shave short-input helper overhead on `intel-spr`/`graviton4`.
- Planned changes:
  - `dispatch_tables.rs`:
    - `PROFILE_POWER10.dispatch.s`: switch from `Portable` to `POWER_VSX_KERNEL` so `<=256` auto-dispatch no longer forces scalar on POWER10.
  - `mod.rs` one-chunk helpers:
    - x86: replace `(kernel.chunk_compress_blocks)(...)` with `kernels::chunk_compress_blocks_inline(kernel.id, ...)` to avoid function-pointer indirection on short hot path.
    - aarch64: call `aarch64::chunk_compress_blocks_neon(...)` directly in the NEON-only helper to remove generic kernel-id branch cost in this path.
- Validation plan:
  - `just check-all && just test`
  - CI granular bench run (same 4 lanes): `intel-spr`, `graviton4`, `ibm-s390x`, `ibm-power10`
  - Gates: `blake3/oneshot` + `blake3/kernel-ab`
