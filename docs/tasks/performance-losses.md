# Cross-target performance losses

## Outcome

Explain and close the catastrophic P-256 and ECDSA losses measured on RISC-V
and IBM Z in benchmark run [#34874736834](https://github.com/loadingalias/rscrypto/actions/runs/34874736834).
Build a reusable CI-only profiling path first
because these machines are not available through an interactive shell.

Do not treat the ten rows as ten optimizations.
They strongly suggest shared field, scalar, inversion, table-selection, or generated-code costs.
One demonstrated cause should drive one bounded production change.

This task is complete when:

- the exact workloads have native CI profiles with attributable symbols and retained raw evidence;
- each shared cause is demonstrated rather than inferred from portable source;
- corrected cases reach the repository's `0.95x` tie boundary,
  or a reviewed equivalent-work analysis records why a remaining comparison is not a valid target;
  and
- the complete benchmark matrix and applicable correctness, CT, cleanup, dispatch,
  and target evidence pass on the final revision.

## Retained baseline

- Source: `ae6f54afedaa652858fd2bcbd8f56f339e663a4f` on `main`.
- Measurement date: 2026-09-14.
- Toolchain: `rustc 1.99.0-nightly (3d6c19bb9 2026-08-11)`.
- Profile: repository `bench` profile.
- Criterion: 20 samples, 100 ms warm-up, 400 ms measurement, 10,000 resamples, 95% confidence.
- Artifacts: `bench-s390x-linux-34874736834-1` and `bench-riscv64-linux-34874736834-1`.
- Ratio: fastest equivalent external time divided by rscrypto time; lower is worse.

| Rank | Target | Exact rscrypto case | Rscrypto | Fastest external | Ratio |
| ---: | --- | --- | ---: | --- | ---: |
| 1 | IBM Z | `p256-ecdh/public-key/rscrypto-selected` | 9.627 ms | `crrl-pure-rust` | 0.00750x |
| 2 | IBM Z | `p256-ecdh/agreement/rscrypto-selected` | 19.075 ms | `crrl-pure-rust` | 0.00924x |
| 3 | RISC-V | `p256-ecdh/public-key/rscrypto-selected` | 13.584 ms | `crrl-pure-rust` | 0.00992x |
| 4 | RISC-V | `p256-ecdh/agreement/rscrypto-selected` | 30.860 ms | `crrl-pure-rust` | 0.01076x |
| 5 | RISC-V | `ecdsa-p384/public-key/rscrypto-blinded` | 159.707 ms | `rustcrypto-p384` | 0.01540x |
| 6 | IBM Z | `ecdsa-p384/public-key/rscrypto-blinded` | 96.934 ms | `rustcrypto-p384` | 0.01560x |
| 7 | IBM Z | `p256-ecdh/parse/rscrypto` | 22.726 us | `crrl-pure-rust` | 0.01577x |
| 8 | RISC-V | `ecdsa-p256/public-key/rscrypto-blinded` | 36.110 ms | `rustcrypto-p256` | 0.01638x |
| 9 | RISC-V | `ecdsa-p384/sign/rscrypto-deterministic/0` | 50.215 ms | `aws-lc-rs` | 0.01702x |
| 10 | RISC-V | `ecdsa-p384/sign/rscrypto-deterministic/32` | 50.220 ms | `aws-lc-rs` | 0.01710x |

The retained confidence intervals are narrow compared with these 58x-133x gaps.
The measurements establish a severe cost, not its cause.

## Phase 1 — Add a CI-only profile workflow

Add a separate manual workflow rather than overloading benchmark collection.
Profiling has different privileges, artifacts, failure modes,
and acceptance rules from elapsed-time measurement.

### Request and build contract

- [ ] Accept exactly one architecture, catalog benchmark target, exact case,
      and bounded capture duration.
      Initially support `riscv64-linux`, `s390x-linux`, and `powerpc64le-linux`.
- [ ] Validate requests against `.config/benchmark-matrix.json`; discovery must prove the case matches exactly once.
- [ ] Reuse the existing cross-build and sealed artifact-transfer machinery.
      Build the real benchmark with the same features, target, optimized `bench` profile,
      debug information, and CPU flags used by measurement.
- [ ] Transfer the executable, debug information, build ID, source identity, toolchain identity,
      and hashes.
      Do not rebuild production code on the native runner before capture.
- [ ] Bound preparation and native capture separately, use fail-fast cancellation, disable caches,
      and upload evidence even when collection fails.

### Native capability probe

- [ ] Record `uname`, `/proc/cpuinfo`, `lscpu`, kernel version, perf version, available PMUs/events,
      CPU governor and frequency data when exposed, `perf_event_paranoid`, `kptr_restrict`, relevant capabilities, and resource limits.
- [ ] Probe `perf stat` and `perf record` with a trivial command before executing the benchmark.
      Report permission, event, unwind, and symbol failures as distinct machine-readable outcomes.
- [ ] Do not silently weaken host security settings on donated runners.
      If `perf_event_open` is blocked, follow the [kernel perf security model](https://docs.kernel.org/admin-guide/perf-security.html):
      ask the runner owner for `CAP_PERFMON` or an agreed `perf_event_paranoid` setting and retain the failed probe as evidence.

### Capture ladder

- [ ] Start with `perf stat` for elapsed time, task clock, cycles, instructions, branches, branch misses,
      cache references, and cache misses.
      Record each unsupported event instead of failing the whole capture.
- [ ] Record a bounded on-CPU sample of the exact Criterion profile case.
      Prefer DWARF call chains from the unchanged optimized artifact;
      record a flat profile if the target's unwinder cannot produce trustworthy stacks.
- [ ] Produce `perf report --stdio`, `perf script`, build-ID output, symbol tables, function sizes,
      and annotated disassembly on the native runner.
      Retain raw `perf.data` and the exact binary/debug files as well.
- [ ] Generate static code evidence on the cross-build host for the same artifact:
      LLVM IR attribution, target assembly, calls to compiler runtime helpers, branches, spills,
      symbol sizes, and relevant loop bodies.
- [ ] Seal all output with a manifest containing source, target, CPU, toolchain, features,
      backend diagnostics, command, collector settings, hashes, status, and limitations.
      Upload one artifact with at least 30-day retention.

Samply is not the first collector for these targets.
Its pinned Linux release uses perf events and its [published Linux binaries](https://github.com/mstange/samply/releases/tag/samply-v0.13.1) cover only x86-64
and AArch64.
Its [stack unwinder](https://github.com/mstange/framehop) also currently covers only those architectures.
Native `perf` gives the smallest credible path.
The workflow must capability-test each donated runner rather than assume its kernel exposes a usable
PMU.

If sampling is unavailable, static codegen plus exact elapsed measurements remain useful
but cannot establish the hot path.
That is an explicit blocker, not permission to guess.
A profiling-only frame-pointer build may be used as a secondary experiment,
but it must be labeled as a different artifact and confirmed against the unchanged production build
before driving an optimization.

### Workflow acceptance

- [ ] Unit tests cover request validation, exact-case selection, budgets, partial event support,
      collector failure, evidence sealing, and failed-run artifact retention.
- [ ] One successful capture from each architecture has attributable rscrypto and dependency frames,
      complete machine identity, and locally readable text reports.
- [ ] Re-running the same request does not depend on an interactive shell
      or unretained runner state.

## Phase 2 — Profile the shared cause

Use this order because it maximizes information per CI run:

1. IBM Z `p256-ecdh/public-key/rscrypto-selected`.
1. RISC-V `p256-ecdh/public-key/rscrypto-selected`.
1. IBM Z and RISC-V `p256-ecdh/agreement/rscrypto-selected`.
1. IBM Z `p256-ecdh/parse/rscrypto`.
1. RISC-V P-256/P-384 public derivation and P-384 signing.
1. POWER control captures for any hot symbol changed by the proposed fix.

- [ ] Attribute fixed-base multiplication, arbitrary-point multiplication,
      field multiplication/reduction, scalar reduction, inversion, coordinate conversion,
      masked table selection, encoding, entropy, and cleanup.
- [ ] Compare native instruction and branch counts with the exact external winner
      when equivalent symbols and work can be identified.
      Do not compare totals across different operation contracts.
- [ ] Determine whether compiler runtime division/multiplication helpers, missed inlining,
      limb width, excessive masked table scans, spills,
      or an algorithmic representation explains the gap.
- [ ] Confirm the leading cause with a repeat capture
      or one controlled perturbation of the real production path.
- [ ] Record a cause once, then link every affected row.
      Do not open separate implementations until the shared-cause hypothesis is falsified.

## Phase 3 — Fix and prove

- [ ] Prefer target-shaped safe Rust, arithmetic representation,
      and data layout before intrinsics or assembly.
- [ ] Change only production-reachable code.
      Preserve portable authority, deterministic output, blinding, failure opacity, secret cleanup,
      constant-time selection, feature independence, and fallback behavior.
- [ ] Route any unsafe, intrinsic, SIMD, assembly, ABI,
      or target-feature change through the required specialist proof.
- [ ] Rerun the exact profile and a longer Criterion baseline/candidate comparison on the same
      machine identity.
- [ ] Run independent vectors and portable-versus-optimized differentials, target-native tests,
      CT evidence, optimized cleanup checks, dispatch evidence,
      and codegen review for the changed boundary.
- [ ] Run the full benchmark matrix and update `benchmark_results/OVERVIEW.md` only from complete retained artifacts.
- [ ] Verify POWER and other targets sharing the changed code do not regress materially.

## Deferred performance queue

After the catastrophic cross-target rows close:

1. Linux x86-64 P-384 signing.
1. `RapidStreamHasher` large one-write throughput on x86-64.
1. ML-KEM decapsulation, especially where the current aggregate loses.
1. Short-message AES-GCM/AES-GCM-SIV fixed cost and RISC-V XXH3 only
   if a fresh focused run confirms material impact.

Do not restore the stale ML-KEM key-generation priority:
the September campaign measured key generation as a win.

## Non-goals

- Hiding target losses in cross-platform averages.
- Weakening blinding, validation, cleanup, failure opacity, or constant-time work.
- Treating cross-compilation, source inspection, static counters,
  or a changed profiling build as native wall-clock proof.
- Adding a copied algorithm or an external implementation as rscrypto's benchmark/profile path.
