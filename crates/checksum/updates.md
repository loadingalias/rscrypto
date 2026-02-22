# Checksum Audit Update Plan (validated 2026-02-22)

## Scope and bar

Target: push `crates/checksum` toward physics-bound throughput while preserving strict correctness, simplifying architecture, and improving UX/DX for tuning and diagnostics.

Principles:
- No ornamental abstraction on hot paths.
- Fix truthfulness and test coverage before tuning changes.
- Generate or centralize repeated policy logic; do not hand-maintain large duplicated tables/loops.

## What was validated on this host

Executed on 2026-02-22:
- `cargo check -p checksum --all-features` passed.
- `cargo test -p checksum --tests --lib` passed.

## Findings from the prior audit list

### Confirmed (real issues)

1. Introspection correctness bug in one dispatch table.
- `crates/checksum/src/dispatch.rs:1298`
- `crates/checksum/src/dispatch.rs:1299`
- `crates/checksum/src/dispatch.rs:1300`
- `crates/checksum/src/dispatch.rs:1301`

`crc32_bytewise_ieee`/`crc32c_bytewise` are labeled `"reference/bitwise"`; these are bytewise portable kernels. Name/function mismatch poisons diagnostics and any kernel-name-based tuning analysis.

2. Provenance docs in dispatch are stale/misaligned with current baseline files.
- `crates/checksum/src/dispatch.rs:21`
- `crates/checksum/src/dispatch.rs:1888`
- `crates/checksum/src/dispatch.rs:1889`
- `crates/checksum/bench_baseline`

Top-level comments reference files not present today and include an explicit "historical Windows baseline" note with no in-tree artifact.

3. Vectored dispatch logic is heavily duplicated.
- `crates/checksum/src/dispatch.rs` (slice + `IoSlice` loops repeated per algorithm)
- `crates/checksum/src/crc16/mod.rs:116`
- `crates/checksum/src/crc32/mod.rs:350`
- `crates/checksum/src/crc64/mod.rs:273`

`last_set` + per-buffer table switch loops are repeated many times, increasing drift/regression risk.

4. Per-call forced-mode branch remains in `update()` path.
- `crates/checksum/src/crc16/mod.rs:97`
- `crates/checksum/src/crc24/mod.rs:95`
- `crates/checksum/src/crc32/mod.rs:331`
- `crates/checksum/src/crc64/mod.rs:254`

`config::get()` is cached via `OnceLock`, but every call still branches on `effective_force`; for tiny-chunk streaming this can be measurable.

5. Config modules are partially near-clones.
- `crates/checksum/src/crc16/config.rs`
- `crates/checksum/src/crc24/config.rs`

CRC16/CRC24 are structurally very similar and should share parser/clamp scaffolding.

6. Benchmark harness duplication is real.
- `crates/checksum/src/bench.rs`
- `crates/checksum/benches/kernels.rs`
- `crates/checksum/benches/comp.rs`

Large repeated benchmark group setup and per-algorithm loops invite drift.

### Partially correct / needs reframing

1. Non-host architecture confidence gap is a CI execution gap, not a total missing-test gap.
- `crates/checksum/src/crc16/kernel_test.rs`
- `crates/checksum/src/crc24/kernel_test.rs`
- `crates/checksum/src/crc64/kernel_test.rs`

Power/s390x/riscv64 parity harnesses exist in source under `cfg(target_arch=...)`; what is missing is regular multi-arch execution in CI and explicit gating policy.

2. "Global fixed boundaries [64,256,4096]" is no longer fully true.
- `crates/checksum/src/dispatch.rs`

Many tables use `[64,256,4096]`, but there are existing variants like `[64,128,4096]` and `[64,64,4096]`. The real issue is boundary policy transparency/provenance, not complete uniformity.

3. `tune.rs` is not dead today, but conceptually overlaps dispatch-era model.
- `crates/checksum/src/lib.rs:147`
- `crates/checksum/src/tune.rs`

It is public API and tested. Decision needed: keep as compatibility API (documented as advisory), or deprecate with migration path.

## Revised priority plan (world-class, minimal complexity)

## P0: Correctness truthfulness and evidence hygiene

1. Fix kernel labels in `GENERIC_ARM_PMULL_NO_CRC_TABLE`.
- Use names matching actual functions (`reference/bytewise` or `portable/bytewise`, but be consistent project-wide).

2. Add dispatch introspection invariants.
- Add tests asserting table function pointer family aligns with reported kernel name for all variants in critical tables.
- Include CRC32/CRC32C checks for the PMULL-no-CRC table.

3. Normalize provenance comments to existing artifacts only.
- Update `dispatch.rs` "Data Sources" to match current `bench_baseline` filenames.
- Remove or explicitly quarantine historical/untracked baselines.

Acceptance:
- No name/function mismatch in dispatch tables.
- Tests fail if any future mismatch is introduced.
- Every provenance reference resolves to an existing in-tree artifact.

## P1: Cross-arch confidence you can trust

1. Define CI matrix policy for kernel parity suites.
- Run kernel parity tests on x86_64, aarch64, powerpc64le, s390x, riscv64 (native/runner/emulated as practical).

2. Promote kernel parity tests to required checks for supported arch targets.
- `kernel_test.rs` modules already provide most logic; wire execution policy and pass/fail criteria.

Acceptance:
- CI reports per-arch parity pass/fail.
- No architecture can merge kernel changes without parity run (or explicit temporary waiver).

## P2: Remove dispatch duplication without slowing hot paths

1. Extract a generic internal vectored core.
- One helper for `&[&[u8]]` and one for `&[IoSlice]`, generic over state type + kernel selector.
- Keep inlinable, monomorphized helpers in `dispatch.rs` internal module.

2. Reuse the same core in algorithm modules for stateful update-vectored paths.

Acceptance:
- No duplicated `last_set` loops across CRC16/24/32/64 modules.
- Criterion microbench before/after shows no regression (or improvement) on hot paths.

## P3: Config simplification and steady-state call-through

1. Share force parsing/clamping infrastructure.
- Consolidate CRC16/24 first (low risk).
- Evaluate extending shared infra to CRC32/64 where semantics match.

2. Move forced-mode branching out of update fast path.
- Resolve function pointers at type init (or OnceLock by algorithm + force mode).
- `update()` should be direct call-through in common auto mode.

Acceptance:
- Reduced branch footprint in `update()` path.
- No behavior drift for env-force modes.

## P4: Dispatch table generation + provenance enforcement

1. Generate `KernelTable` literals from a single source-of-truth input.
- Keep generated code checked in, but do not hand-edit table blocks.

2. Add CI guardrails.
- Verify referenced baseline files exist.
- Verify generated output matches committed dispatch tables.

Acceptance:
- Manual table drift becomes structurally hard.
- Provenance mismatch fails CI.

## P5: Benchmark UX/DX cleanup

1. De-duplicate benchmark harness shape with reusable helper/macro.
- Keep clarity; avoid over-generic API soup.

2. Clarify `tune.rs` role in docs/API.
- If retained: mark as advisory thresholds, not dispatch selectors.
- If deprecated: soft deprecate with replacement guidance.

Acceptance:
- Smaller benchmark boilerplate with equal readability.
- No ambiguity about tune-vs-dispatch responsibilities.

## Concrete next implementation batch (recommended)

1. P0 only in first PR:
- fix label mismatch,
- add introspection invariant tests,
- update provenance comments,
- add small CI check script for provenance file existence.

2. P2 in second PR:
- extract vectored helper,
- replace duplicated loops in `dispatch.rs` + one algorithm module first,
- benchmark and validate,
- then roll through remaining modules.

3. P3 in third PR:
- shared config infrastructure for CRC16/24,
- cached dispatch function pointers for update fast path,
- re-run tests/benchmarks.

## Benchmark and validation gates for each PR

Mandatory:
- `just check-all`
- `just test --all`

For performance-touching PRs:
- `just bench crate=checksum bench=kernels`
- `just bench crate=checksum bench=comp`
- capture before/after for xs/s/m/l at minimum.

## Strong opinion

The single highest leverage move is to stop hand-maintaining dispatch policy artifacts. If we keep a giant manually curated `dispatch.rs`, we will keep paying the same regression tax. Truthful introspection + generated tables + CI provenance checks is the shortest path to both speed and reliability.
