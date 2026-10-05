# Benchmark Overview

This file keeps the retained benchmark campaigns.
Each section names its revision, toolchain, hosts, method, and limits.
The dated campaign records come first.
The 2026-08-18 Linux snapshot near the end is historical:
its aggregate ratios are withdrawn as performance claims (see [Corrections](#corrections)).

## 2026-10-05: Graviton5 HMAC timing characterization

**Decision:** retain the HMAC threshold and existing release qualification. CPU
pinning and a one-second confirmation delay did not reliably prevent threshold
crossings. The archived release binary also crossed the threshold when both
classes had valid tags, and when both classes had invalid tags. A validity
difference is therefore unnecessary for this class-associated timing effect.
This completes the requested host/control characterization; it does not identify
the underlying processor, kernel, or harness mechanism, or prove HMAC constant
time. A failed release measurement still blocks qualification. No
retry-until-pass rule was introduced.

### Archived failure and prior observations

[Release run 37098163639, attempt 1](https://github.com/loadingalias/rscrypto/actions/runs/37098163639/attempts/1)
used commit `1dd2a51a470dda129283f124b70306de590cea84` on `c9g.2xlarge`. The
HMAC-SHA256 valid/invalid case screened at |t| **17.28731** with 20,000 samples,
then confirmed at **69.95775** with 80,000 samples, against limit 10.
Publication stopped. Independent replay of the retained raw durations reproduces
both statistics. The largest cropped statistics use durations below 229 ns: the
valid class was faster by **0.45049 ns** in screening and **0.88539 ns** in
confirmation. These are measured class differences, not a diagnosis of their
cause.

Earlier diagnostics on `87a8f220` observed 32.7 → 12.3, 14.9 → 1.30, and a
screening result of 1.50. A dedicated same-host, 20,000-sample comparison
observed 1.6, 5.7, 1.7, 2.4, 1.4 on `87a8f220`, and 6.6, 12.2, 2.5 on
`a06861e4`, whose CI result had been 2.62. The measured code was unchanged
between those revisions; this excludes that code change as the cause, not a
pre-existing defect. Before Arm DIT at `b4dfdd78`, this case and the ML-DSA
dense Montgomery probe had shown intermittent offsets of about 1 ns. HMAC
remains outside per-call DIT. [Release run
37135507119](https://github.com/loadingalias/rscrypto/actions/runs/37135507119)
passed all CT platforms on `87a8f220`, and v0.10.0 shipped. Those historical
passes do not erase the failures.

### Native campaign

A disposable `c9g.2xlarge` provided eight Neoverse-V3 cores, Linux
`7.0.0-1014-aws`, and native `aarch64-unknown-linux-gnu` execution. The archived
artifact used kernel `7.0.0-1011-aws` originally; replay does not recreate every
detail of the original OS environment. The current harness and separate control
binary used `nightly-2026-09-30`, rustc
`5c543b0b8c73c7b72bc8284ced4fb22ead15734d`, LLVM 23.1.1, release optimization,
fat LTO, one codegen unit, and the harness's `std/full/parallel/diag/getrandom`
features with `rscrypto_internal`. The current source was the dirty tree over
`d4045559`, bound by `source-identity.json`:
`bde51461faa05898bc16f7899f00353ef59e70ab734879c7207de62bedcb5391`. The HMAC
implementation and release harness were not edited for this investigation.

Every campaign planned three repetitions of CPU 0 pinning versus affinity to
CPUs 0–7, and immediate versus one-second-separated confirmation. Each pair used
new processes at 20,000 and 80,000 samples. All pairs ran regardless of
screening results; these are diagnostic measurements, not simulated release
decisions. Policy order reversed on alternate repetitions, and variant order
rotated. All class sequences matched for a given sample count.

| Campaign / case | Runs | Max abs(t) | 20k >10 | 80k >10 |
| --- | ---: | ---: | ---: | ---: |
| Initial / archived valid-invalid | 24 | 21.51848 | 2 | 1 |
| Initial / current valid-invalid | 24 | 28.43055 | 2 | 2 |
| Initial / control valid-invalid | 24 | 7.64625 | 0 | 0 |
| Initial / control invalid-valid | 24 | 5.77738 | 0 | 0 |
| Initial / control valid-valid | 24 | 7.93745 | 0 | 0 |
| Initial / control invalid-invalid | 24 | 6.42172 | 0 | 0 |
| No transfer / archived valid-invalid | 24 | 37.79537 | 3 | 4 |
| No transfer / archived valid-valid | 24 | 17.55567 | 3 | 1 |
| No transfer / archived invalid-invalid | 24 | 35.92848 | 1 | 2 |

The separate control's different code layout did not reproduce the release case.
The stronger archived controls change only one four-byte instruction at virtual
address `0x96a1c` (file offset 616988), in expected-tag preparation before
measurement. `eor w8, w19, w8` becomes `eor w8, w19, wzr` for both valid, or
`eor w8, w19, #1` for both invalid. Here `w19` contains the correct first tag
byte and `w8` the class bit. Both substitutions preserve instruction width,
registers written, and flags; all other bytes, including every timed instruction
and its address, remain identical. The patcher rejects any input except the
exact archived SHA-256 and checks the original opcode. LLVM disassembly
independently confirmed both substitutions. These are diagnostic fixtures only.

An intervening 72-measurement archived-control campaign overlapped artifact
collection at 50 measurement boundaries. It is retained separately, including
its failures, but the table uses the fixed follow-up with no transfers at any
measurement boundary. Across all three campaigns, **288 measurements and
14,400,000 raw observations** are retained, including all failures. No build ran
during measurement. Process, load, CPU, affinity, and timestamp snapshots bound
each run; this is not a claim that the OS or hypervisor was noise-free.

The initial pinned archived pair was -21.51848 → -20.37705. The no-transfer
delayed archived pair was -15.12602 → -37.79537. Thus neither pinning nor delay
is a reliable remedy. The same-layout A/A crossings also mean that raising the
threshold or accepting a later pass would hide an unresolved measurement effect.
The full workflow already launches a fresh process for each measurement; no
process-isolation fix was needed.

### Retained identity, validation, and limits

Local evidence lives in `benchmark_results/2026-10-05/graviton5-hmac/`.
`original-artifact.json` identifies failed job `111133295986` and artifact
`11265082865`; the default jobs API shows the later, successful second attempt
instead. The downloaded ZIP matched its published SHA-256:
`bc9df1a7a13ed93eb500e3f4370e10c3d6790531ca262021ca902345f695c919`.
`ct-aarch64-linux-full.tar.gz` retains the original full evidence.
`hmac-native-all.tar.gz` retains all native binaries, disassembly, prepared
metadata, campaign plans, raw CSVs, output, and host snapshots. Its SHA-256 is:
`7c029fc93d6c5e98bfc1034b15b8b2efb6e022a1ef74e3fe7ad888c73f1fa232`.
`campaign-summary.json` verifies binary/raw hashes, observation counts,
ordering, and class sequences. The campaign, patch, and analysis scripts, build
logs, source manifest, original raw analysis, and host lifecycle logs are
retained beside the archives.

Binary SHA-256 identities:

- Archived: `ef9815afcd675746ec2bc5adcb05299a73610472c184119e4ec1f0759ae75356`.
- Current: `2e754e0df79678b811aed6981e5fd1f1512e0909d83210a1697176bea2a39de1`.
- Separate control:
  `3d92a127ae0abf50dfd8b0979f05061c33ac69e180c0eb6a0139eeaa9c418446`.
- Archived valid-valid:
  `34a3deb7bc3ce1f1cedbe6cad88f6124931d434b1afc41688482d7621d471071`.
- Archived invalid-invalid:
  `7fd81c655c550298b018375ec80330246c3a530bde52dff13fd1c1de9a6a2739`.

`just ct-test` passed the harness/exporter and orchestration checks plus 1,265
native and 1,236 portable internal tests on the local Mac.
`just ct-validate --manifest-only`, focused control-driver Clippy with warnings
denied, and Rustfmt passed. These local checks are separate from Graviton5 timing.
The host was terminated and its attached volume removed after evidence collection.

Control preparation also exposed a local toolchain-wrapper defect: an inner
Cargo `--` delimiter was rejected before the child command ran. The wrapper now
forwards every token after `--exec` unchanged. The regression failed before the
fix and passed for all eight host configurations afterward, including nested
flags, spaces, empty arguments, and invalid wrapper modes. The original Clippy
command also passed through the repaired wrapper. `just test-scripts` passed
with four existing platform skips. No dependency was added.

This is bounded characterization, not a new release qualification or a proof
that every historical HMAC failure had the same cause. The exact low-level
mechanism remains unknown. Any future harness or qualification-policy change
needs causal evidence and fresh target qualification; the present threshold,
sample budgets, confirmation decision, and fail-closed behavior remain
unchanged.

## 2026-10-04: Ed25519 and X25519 vector fixed-base tables

On native x86-64 Windows, precomputing the 512 public conversions used by each fixed-base multiply
reduces short-message Ed25519 signing time by 54% / 59% with IFMA and 43% / 42% with AVX2 (Intel /
AMD).
The public APIs, scalar recoding, point addition, and dispatch stay the same.
Normal Linux Ed25519 and X25519 public-key operations use separate assembly
and do not gain from this change.
These are primitive results, not authenticated-channel measurements.

Baseline: effective `main` at `d8db85642e1ff2a176c7925eb873fa2f23021c30`, with the same benchmark capability switch in both trees.
Baseline `point_avx2.rs` SHA-256: `35b435987711a077737a118c5e1acb9659221a1605694eb8ab9ae483c099686e`; measured candidate: `964278728d945bdb5ff7d13a95590870c32c987fe7cee052fcae0965dd5da3ab`.
The later selector-source changes add safety documentation only.
Both hosts used `nightly-2026-09-30`, rustc `1.101.0-nightly (5c543b0b8 2026-09-29)`, LLVM 23.1.1, `x86_64-pc-windows-msvc`, the repository `bench` profile, and only `--cfg rscrypto_internal` in Rust flags.
The `auth` benchmark used its catalog features.

- Intel: AWS `c8i.4xlarge`, Xeon 6975P-C, 8 cores / 16 logical processors.
- AMD: Azure `Standard_F8as_v7`, EPYC 9V45, 8 cores / 8 logical processors.
- Ten same-host baseline/candidate rounds, alternating order, with IFMA and forced AVX2.
  Each case used 300 ms warmup, 700 ms measurement, and 30 samples.
  No observations or rounds were removed.
- Values below are medians of the ten Criterion slope estimates, in microseconds.
  Percentage changes use the median of the ten paired candidate/baseline ratios.
  Full rows, paired ranges, and deterministic 95% bootstrap intervals are in the retained summary.

| Public operation                 |    Intel IFMA |    Intel AVX2 |      AMD IFMA | AMD AVX2 |
| -------------------------------- | ------------: | ------------: | ------------: | -------: |
| Ed25519 public key               |  19.76 → 8.85 | 20.42 → 11.52 |  16.83 → 6.71 | 14.53 → 8.28 |
| Ed25519 keypair                  |  19.76 → 8.85 | 20.42 → 11.62 |  16.76 → 6.69 | 14.55 → 8.22 |
| Ed25519 keypair sign, 32 B       |  20.16 → 9.25 | 20.85 → 11.89 |  17.12 → 7.03 | 14.81 → 8.55 |
| Ed25519 direct-secret sign, 32 B | 39.91 → 18.10 | 41.27 → 23.32 | 33.89 → 13.73 | 29.30 → 16.77 |
| Ed25519 keypair sign, 16 KiB     | 67.34 → 56.07 | 67.96 → 58.96 | 50.63 → 40.69 | 48.34 → 42.06 |
| X25519 public key                |  19.49 → 8.59 | 20.35 → 11.54 |  16.55 → 6.55 | 14.25 → 8.08 |

Verification (0, 32, 1,024, and 16,384 B) and X25519 agreement were controls.
Their median changes range from −0.41% to +0.25%, below Criterion's configured 1% noise threshold.
Some Intel IFMA verification intervals extend to +1.64%;
this campaign does not rule out small layout or host effects in every control.
The first candidate build occurred between the first baseline and candidate measurements;
that limitation and all ten rounds remain in the record.

The matched Intel benchmark EXE grows by 162,304 bytes
(158.5 KiB): raw `.rdata` grows by 163,840 bytes and `.text` shrinks by 1,536 bytes.
Both linked selectors have no EVEX instructions or nested calls.
Fixed-base worker frames get smaller, but the X25519 wrapper gets larger;
these observations do not establish lower whole-operation peak stack use.
The [secret-lifecycle boundary](../docs/secret-lifecycle.md#ed25519-and-x25519) records unwiped arithmetic temporaries without claiming complete
stack or register cleanup.

Correctness evidence includes exhaustive table-entry and signed-digit comparisons,
portable-field oracles, and native vector differential tests:
102 focused Linux tests and 99 on each Windows host.
The two Linux BINSEC selector proofs report `secure`; the IFMA leaf is now a required kernel.
The final proof archives match the working tree's selector, table, harness, and manifest hashes.
AVX2 completes one path in 534 instructions and IFMA in 452, with no unknown instructions or cuts.
Both reports, executables, disassemblies, and source hashes are in `final-linux-proofs.tgz` below.
Both Windows hosts pass the five selected Ed25519/X25519 timing cases at their manifest budgets
(20,000 samples, or 200,000 for signing commitment), with maximum |t| of 2.35 and 2.94.
These selected runs are diagnostic evidence, not the full release CT matrix.

Reproduce each tree with `just bench --bench auth` and the filter `^(x25519/|ed25519/(sign|verify|public-key-from-secret|keypair-from-secret)/).*rscrypto`, plus `--warmup-ms 300 --measure-ms 700 --sample-size 30 --output-dir PATH`.
Set `RSCRYPTO_BENCH_DISABLE_IFMA=1` for the AVX2 run before process initialization.
Intel key construction was measured in a separate ten-round pass
after correcting the initial filter.
The early metadata collector did not list that new environment key; the preserved script,
per-backend directories, and captured capability output identify it explicitly.
The collector now records it for subsequent runs.

Local raw archives, exact plans and hashes, timing reports, linked-code review, scripts,
and `summary.json` are retained under `benchmark_results/2026-10-04/ed25519-tables/`.
That directory is ignored; this overview preserves the measurements and limits in Git.

## 2026-10-04: P-384 reduction overlap rejected

Advancing the next Montgomery quotient with flag-preserving SHLX/LEA instructions made P-384
agreement 2.42% slower on Intel Granite Rapids.
The candidate passed correctness checks but was removed.
No P-384 production or generator change remains from this experiment.

The candidate changed only the square reduction schedule inside the fused x86-64 doubling kernel,
retaining the 21-MULX square product.
Its premise was to overlap the next quotient calculation with the current borrow chain.
This measurement rejects that schedule;
it does not establish the microarchitectural cause of the loss.

Baseline: effective `main` at `d8db85642e1ff2a176c7925eb873fa2f23021c30`.
Both trees included the same pending Ed25519 changes.
Baseline `p384_x86_64.rs` SHA-256: `a8a1314398d09935ec54af0c7d56b1095c9dd527023f25d1c04c4e3059e6fb80`; candidate: `4feee455e0e90458402ee82071d6b5ab406acaaf331df29c77aaf958d68314d4`.
The host was AWS `c8i.4xlarge`, Xeon 6975P-C, 8 cores / 16 logical processors, running `x86_64-unknown-linux-gnu`, `nightly-2026-09-30`, rustc `1.101.0-nightly (5c543b0b8 2026-09-29)`,
and LLVM 23.1.1.
Both artifacts were built before measurement, using the repository `bench` profile, the `auth` catalog features,
and `--cfg rscrypto_internal`.

Ten same-host rounds alternated baseline/candidate order.
Each case used 300 ms warmup, 1,000 ms measurement, and 40 samples.
All ten rounds are retained.
Values below are medians of Criterion slope estimates;
changes and intervals use the paired candidate/baseline ratios.
The deterministic bootstrap uses 10,000 resamples and seed `20261004`.

| Agreement implementation | Baseline | Candidate | Paired median change | Paired bootstrap 95% interval |
| --- | ---: | ---: | ---: | ---: |
| rscrypto | 122.23 µs | 125.14 µs | +2.42% | +1.98% to +2.67% |
| AWS-LC control | 120.06 µs | 119.99 µs | −0.02% | −0.14% to +0.04% |

Correctness evidence: 3,000 cases per generated kernel against Python integer arithmetic,
simulator negative controls including a carry-flag mutation, and 26 native P-384 tests.
The performance loss stopped qualification before new CT evidence or an AMD run.
The restored generator passes `python3 scripts/asm/p384.py check`.

Reproduce the comparison with `just bench --bench auth`, filter `^p384-ecdh/agreement/(rscrypto-selected|aws-lc-rs-native)$`, and `--warmup-ms 300 --measure-ms 1000 --sample-size 40 --output-dir PATH`.
The rejected source, exact diff, run script, plans, hashes, raw measurements,
and `summary.json` are retained locally under `benchmark_results/2026-10-04/p384-overlap/` (ignored).
The x86 agreement gap and the remaining architecture backends are still open.

## Corrections

**Comparison validity.**
The historical ML-KEM comparisons mixed caller-supplied and internal entropy, key preparation,
and output representations.
The Argon2 comparisons gave `rscrypto` and RustCrypto a longer salt than dryoc.
The affected ratios, rankings,
and aggregates that contain those rows are withdrawn as performance claims.
The 2026-10-03 run below has a like-for-like summary for hashes, checksums, MACs, XOFs, and scrypt.
ML-KEM and Argon2 still have no replacement aggregate.
The numerical effect on the historical scorecard has not been measured.
The tables and raw artifacts stay as historical records, not corrected results.
See the [current comparison contracts](../docs/benchmarking.md#ml-kem-and-argon2-comparison-contracts).

**Workload identity.**
Historical AEAD encrypt and decrypt rows, and ChaCha XOR rows, include timed buffer restoration.
The former BLAKE2 host-overhead rows measure complete hashes and duplicate the main groups.
Its plain parameter rows also duplicate the main one-shot cases.
The former Ascon `ascon-hash256/scalar-loop` and `ascon-xof128/scalar-loop` labels both call `rscrypto`.
Treat those rows as internal comparisons, not external comparisons.
The current [timed-boundary policy](../docs/benchmarking.md#timed-workload-boundaries) names the actual work and removes duplicate cases.
No historical ratio was recomputed after these changes.
The 2026-09-14 campaign uses the corrected workload identities.

## Sources

- Full benchmark workflow run
  [#37092266645](https://github.com/loadingalias/rscrypto/actions/runs/37092266645),
  commit `18fb791fef2b5d92e300b757408dac19d327c30c`, 2026-10-03, eight platforms.
- Full benchmark workflow run
  [#34874736834](https://github.com/loadingalias/rscrypto/actions/runs/34874736834),
  created 2026-09-14 17:26:15 UTC and completed 2026-09-14 18:33:05 UTC.
- Full-run commit: `ae6f54afedaa652858fd2bcbd8f56f339e663a4f` on `main`.
- Linux benchmark snapshot created 2026-08-18 21:03:07 UTC.
- Linux commit: `7eb44e9a38ef7a031d9181dc8c4c0fad38f46504`.
- Linux artifacts: eight successful `benchmark-*` artifacts extracted into `benchmark_results/2026-08-18/linux/*/results.txt`.
- Local macOS run: `benchmark_results/2026-07-04/macos/aarch64/results.txt` at commit `596498f0e07e869eac71fd31c157aa1b22186239`, carried forward unchanged.
- Local Ed25519 direct-secret before/after diagnostic, recorded below.
- Local P-256 ECDH development run on Apple M1, 2026-09-03, based on
  `fdd4eec6` with uncommitted Phase 4 changes; curated below and not treated as
  release or cross-target evidence.
- Physical AWS Graviton4 P-256 ECDH development run, 2026-09-03,
  from an intermediate Phase 4 worktree.
  The sealed Criterion and native-evidence bundles are under `benchmark_results/2026-09-03/linux/aarch64/graviton4/`.
- Physical AWS Graviton3 P-256 ECDH development run, 2026-09-03,
  from an intermediate Phase 4 worktree.
  The sealed Criterion and native-evidence bundles are under `benchmark_results/2026-09-03/linux/aarch64/graviton3/`.
- Physical AWS Intel Granite Rapids P-256 ECDH development run, 2026-09-03,
  from an intermediate Phase 4 worktree.
  The sealed Criterion and native-evidence bundles are under `benchmark_results/2026-09-03/linux/x86_64/intel-gnr/`.
- Physical AWS Windows x86-64 Intel Granite Rapids P-256 ECDH development runs, 2026-09-03.
  The full native-backend run used an intermediate Phase 4 worktree;
  the final batch-parser comparison matches the current P-256 source.
  Both are under `benchmark_results/2026-09-03/windows/x86_64/intel-gnr/`.

## 2026-10-03 full benchmark run (v0.10.0)

Bench run [#37092266645](https://github.com/loadingalias/rscrypto/actions/runs/37092266645) measured commit `18fb791f`
on eight platforms with `architectures=all`, `selection=all`, and diagnostics off.
The source and benchmark code are the same as the v0.10.0 tag (`87a8f220`). Only the package version differs.
All jobs used `rustc 1.101.0-nightly (5c543b0b8 2026-09-29)` and the catalog Criterion defaults:
20 samples, 100 ms warm-up, 400 ms measurement, 10,000 resamples, 95% confidence, 1% noise threshold.

| Platform | Host | Completed cases | Artifact |
| --- | --- | ---: | --- |
| Intel Linux | `c8i.2xlarge` | 2,698 | `bench-x86_64-linux-intel-37092266645-1` |
| AMD Linux | `c8a.2xlarge` | 2,698 | `bench-x86_64-linux-amd-37092266645-1` |
| Intel Windows | `c8i.2xlarge` | 2,698 | `bench-x86_64-win-intel-37092266645-1` |
| AMD Windows | `c8a.2xlarge` | 2,698 | `bench-x86_64-win-amd-37092266645-1` |
| Graviton5 Linux | `c9g.2xlarge` | 2,702 | `bench-aarch64-linux-37092266645-1` |
| POWER10 Linux | native GitHub runner | 2,432 | `bench-powerpc64le-linux-37092266645-1` |
| IBM Z Linux | native GitHub runner | 2,414 | `bench-s390x-linux-37092266645-1` |
| RISC-V Linux | native GitHub runner | 2,694 | `bench-riscv64-linux-37092266645-1` |

POWER10, IBM Z, and RISC-V ran binaries that x86-64 compiled for them.

### Method

Each comparison uses the case names `group/implementation/input`.
For each group and input, the comparison divides the median time of the fastest external crate
by the median time of `rscrypto`.
A ratio above 1.00x means `rscrypto` is faster.
"Within 3%" means a ratio from 0.97x to 1.03x.
Each platform has the same 346 comparisons: hashes, checksums, MACs, XOFs, and scrypt.

The comparisons do not include AEAD, signature, key-exchange, ML-KEM, ML-DSA, Argon2, or RSA cases.
Those cases use other names, and each one needs a review under the
[comparison contracts](../docs/benchmarking.md#ml-kem-and-argon2-comparison-contracts) before it gives a ratio.

### Results by platform

| Platform | Faster | Within 3% | Slower | Median ratio |
| --- | ---: | ---: | ---: | ---: |
| Intel Linux | 211 | 88 | 47 | 1.09x |
| AMD Linux | 256 | 81 | 9 | 1.13x |
| Intel Windows | 186 | 105 | 55 | 1.05x |
| AMD Windows | 262 | 68 | 16 | 1.09x |
| Graviton5 Linux | 147 | 121 | 78 | 1.01x |
| POWER10 Linux | 206 | 122 | 18 | 1.07x |
| IBM Z Linux | 301 | 19 | 26 | 2.77x |
| RISC-V Linux | 197 | 79 | 70 | 1.05x |

### Results by family

Each cell is the geometric mean ratio for that family on that platform.

| Family | Rows | Intel Linux | AMD Linux | Intel Windows | AMD Windows | Graviton5 | POWER10 | IBM Z | RISC-V |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CRC | 77 | 6.62x | 5.96x | 6.93x | 6.39x | 5.01x | 11.16x | 7.20x | 1.32x |
| XXH3 | 33 | 1.24x | 1.28x | 1.20x | 1.26x | 1.08x | 1.32x | 1.65x | 0.79x |
| RapidHash | 22 | 1.32x | 1.13x | 1.28x | 1.19x | 1.16x | 1.19x | 1.06x | 1.09x |
| BLAKE3 | 11 | 1.11x | 1.23x | 0.92x | 1.19x | 1.67x | 1.83x | 1.77x | 1.11x |
| Ascon | 22 | 1.09x | 1.07x | 1.09x | 1.10x | 1.07x | 1.06x | 1.05x | 1.06x |
| SHA-2 | 55 | 1.03x | 1.08x | 1.06x | 1.08x | 1.01x | 1.03x | 5.32x | 1.03x |
| HMAC-SHA-2 | 33 | 1.01x | 1.06x | 0.95x | 1.07x | 0.96x | 1.02x | 4.95x | 1.02x |
| SHA-3 | 44 | 1.17x | 1.14x | 0.99x | 1.09x | 0.95x | 2.37x | 8.24x | 2.64x |
| SHAKE | 22 | 0.98x | 1.16x | 1.14x | 1.10x | 0.99x | 1.01x | 4.27x | 1.16x |
| cSHAKE/KMAC | 22 | 0.98x | 1.13x | 0.97x | 1.09x | 1.01x | 0.94x | 3.96x | 1.09x |
| scrypt | 5 | 0.96x | 0.98x | 0.91x | 1.00x | 1.33x | 1.60x | 1.41x | 1.60x |
| **All** | **346** | **1.63x** | **1.63x** | **1.61x** | **1.65x** | **1.48x** | **2.05x** | **3.99x** | **1.23x** |

How to read the table:

- CRC ratios are large because some comparators use tables, not carry-less multiplication.
  The largest CRC ratios are CRC-16 and CRC-24, where the only comparator is the `crc` crate.
  The CRC rows also raise the "All" row, so the median ratio above is the better summary.
- On IBM Z, `rscrypto` uses the CPACF hash instructions (KIMD) for SHA-2 and SHA-3.
  The fastest comparators there are portable crates (`sha2`, `sha3`, `tiny-keccak`, and RustCrypto `hmac`),
  and AWS-LC is not in the IBM Z comparison. This explains most of the IBM Z lead.
- `rscrypto` is slower than the fastest comparator (family mean below 0.97x) in eight results:
  XXH3 on RISC-V (0.79x), BLAKE3 on Intel Windows (0.92x), HMAC-SHA-2 on Intel Windows (0.95x)
  and Graviton5 (0.96x), SHA-3 on Graviton5 (0.95x), cSHAKE and KMAC on POWER10 (0.94x),
  and scrypt on Intel Linux (0.96x) and Intel Windows (0.91x).

### Limits

This is one run on shared or cloud hosts, with short measurement windows.
It is not a regression gate.
Do not compare absolute times between platforms.
It does not include macOS: the last macOS numbers are from 2026-07-04 (see the macOS local snapshot below).

## 2026-09-14 full benchmark run

Run [#34874736834](https://github.com/loadingalias/rscrypto/actions/runs/34874736834) completed successfully on its first attempt.
It selected `all` architectures and `all` benchmark groups with diagnostic features disabled.
The plan, three cross-build preparation jobs, and all eight measurement jobs passed.
The run measured commit `ae6f54af` from `main`.
Native jobs used Rust 1.98.1; the cross-built POWER, s390x,
and RISC-V binaries used Rust 1.99.0-nightly (`3d6c19bb9`, 2026-08-11).

The campaign used the catalog defaults: 20 samples, 100 ms warm-up, 400 ms measurement time,
10,000 resamples, 95% confidence, and a 1% noise threshold.
It executed 14 benchmark binaries per platform.
In total, the retained measurement artifacts contain 19,614 completed Criterion cases.

| Measurement job    | System  | Runner shape | Rust toolchain | Completed cases | Artifact |
| ------------------ | ------- | ------------ | -------------- | --------------: | -------- |
| x86_64-linux-amd   | Linux   | c8a.2xlarge  | 1.98.1         |           2,517 | `bench-x86_64-linux-amd-34874736834-1` |
| x86_64-linux-intel | Linux   | c8i.2xlarge  | 1.98.1         |           2,517 | `bench-x86_64-linux-intel-34874736834-1` |
| aarch64-linux      | Linux   | c9g.2xlarge  | 1.98.1         |           2,521 | `bench-aarch64-linux-34874736834-1` |
| powerpc64le-linux  | Linux   | native       | 1.99.0-nightly |           2,255 | `bench-powerpc64le-linux-34874736834-1` |
| s390x-linux        | Linux   | native       | 1.99.0-nightly |           2,255 | `bench-s390x-linux-34874736834-1` |
| riscv64-linux      | Linux   | native       | 1.99.0-nightly |           2,515 | `bench-riscv64-linux-34874736834-1` |
| x86_64-win-amd     | Windows | c8a.2xlarge  | 1.98.1         |           2,517 | `bench-x86_64-win-amd-34874736834-1` |
| x86_64-win-intel   | Windows | c8i.2xlarge  | 1.98.1         |           2,517 | `bench-x86_64-win-intel-34874736834-1` |

Case counts differ where the target-specific catalog omits unavailable implementations
or adds target-specific coverage.
Each artifact retains the exact source state, build environment, host identity, plan,
case inventory, Criterion estimates, and samples.
The short per-case measurement window
and non-uniform hosts make this a broad cross-platform snapshot,
not a regression gate or a license to compare absolute times between machines.

No ratios from this campaign are folded into the historical scorecard below.
That scorecard uses a different eight-host Linux matrix and predates the current ML-KEM, Argon2,
and timed-workload comparison contracts.
Replacing it requires a fresh fastest-equivalent-case curation rather than combining the two
campaigns.

## 2026-09-30 BLAKE3 batch and portable one-chunk runs

Native AWS hosts, `nightly-2026-09-25`, catalog Criterion defaults, `blake3,parallel,std`. x86-64 is `c8i.4xlarge`
(Intel Xeon 6975P-C, AVX-512 lanes); AArch64 is `c8g.4xlarge` (Graviton4, Neoverse-V2, NEON lanes).
The source was the uncommitted working tree on top of `5ef7858a`.
The machines were destroyed after the runs; per-run summaries are local only.

`Blake3::digest_batch` over 64 equal-length messages, versus one `Blake3::digest` call each (`blake3/batch` and `blake3/batch-serial`, medians):

| Message | x86-64 batch | x86-64 serial | Speedup | Graviton4 batch | Graviton4 serial | Speedup |
| ------: | -----------: | ------------: | ------: | --------------: | ---------------: | ------: |
|    21 B |      0.82 µs |       4.01 µs |    4.9× |         3.41 µs |          6.28 µs |    1.8× |
|    64 B |      0.75 µs |       2.79 µs |    3.7× |         2.72 µs |          5.99 µs |    2.2× |
|   256 B |      2.47 µs |      14.80 µs |    6.0× |         9.79 µs |         22.43 µs |    2.3× |
| 1,024 B |      9.45 µs |      47.12 µs |    5.0× |        37.99 µs |         88.76 µs |    2.3× |

The `blake3` crate, called once per message,
was within 15% of rscrypto's serial row at every size on both hosts.
The Graviton4 batch run preceded the x86 partial-block kernel change,
which does not touch the NEON path.

Portable one-chunk digest (`blake3/rscrypto-portable/*`, `--diag`), three interleaved rounds of `2cbc2cb2` (before `Blake3::digest_const`), `12cd0bfd`
(current `main`),
and the working tree, which inlines the shared one-chunk helper
and reads a full final block in place.
Median change versus `2cbc2cb2`:

| Input | x86-64 `12cd0bfd` | x86-64 working tree | Graviton4 `12cd0bfd` | Graviton4 working tree |
| ---: | ---: | ---: | ---: | ---: |
| 0 B | +7.4% | +6.0% | +6.0% | +3.0% |
| 32 B | +8.5% | +6.0% | +4.2% | +2.4% |
| 64 B | +14.6% | +5.0% | +6.6% | +1.0% |
| 256 B | −1.5% | −2.0% | +1.6% | +0.5% |
| 1,024 B | −0.8% | −0.9% | +1.0% | +0.6% |
| keyed 0–64 B | +8.1 to +16.1% | +6.9 to +7.2% | +1.7 to +6.5% | +0.5 to +3.9% |

The remaining 2–5 ns at 0–64 bytes comes from sharing one const helper between `Blake3::digest_const`
and the runtime portable path:
keyed mode needs every intermediate in caller-owned scratch so it can clear it,
and those escaping references keep the scratch in memory in every mode.
Only the portable backend runs this path.
Dispatched SIMD rows (`blake3/rscrypto/*`, 0 B to 1 MiB, plain and keyed) stayed within ±1% of `2cbc2cb2` on x86-64,
including after the x86 owned hash-many kernels gained a final-block length.

`b18697fd` sends only keyed and derive-key portable inputs through the shared helper;
unkeyed inputs return to the tiny-input and generic one-chunk paths.
Rerun on 2026-10-01 on fresh hosts of the same types, three interleaved rounds of `2cbc2cb2` and `b18697fd`,
same filter and features.
Median change versus `2cbc2cb2`, with the per-round range:

|   Input |       x86-64 unkeyed |          x86-64 keyed |    Graviton4 unkeyed | Graviton4 keyed |
| ------: | -------------------: | --------------------: | -------------------: | --------------: |
|     0 B |  −0.1% (−0.1 to 0.0) | +7.3% (+7.1 to +11.7) |  +0.1% (0.0 to +0.3) | +1.1% (+1.0 to +1.2) |
|     1 B |  −0.1% (−0.2 to 0.0) |  +6.2% (+6.2 to +8.5) | +0.1% (−0.2 to +0.4) | +2.7% (+2.7 to +3.2) |
|    32 B | −0.3% (−0.3 to −0.1) |  +6.5% (+5.9 to +7.3) | +1.1% (+1.0 to +1.1) | +0.8% (+0.7 to +0.9) |
|    64 B | +0.5% (+0.3 to +0.6) |  +7.0% (−0.9 to +7.0) | −1.8% (−1.9 to −1.6) | +2.3% (+2.2 to +2.3) |
|   256 B | −0.9% (−1.0 to −0.8) |  −1.6% (−2.0 to −1.6) | +0.4% (+0.3 to +0.4) | −0.8% (−0.8 to −0.7) |
| 1,024 B | −0.1% (−0.2 to −0.1) |  −0.8% (−0.8 to −0.8) |  +0.1% (0.0 to +0.1) | −0.4% (−0.4 to −0.4) |

Unkeyed portable digests are back at the `2cbc2cb2` cost.
The keyed 0–64 byte cost remains, because keyed mode still clears every secret-derived intermediate.
Round 2 on x86-64 had noisy keyed rows in both trees, so those ranges are wide.
The machines were destroyed after the runs; per-run summaries are local only.

## 2026-10-01 ML-KEM Keccak stack scrub

GitHub Bench `mlkem768,mlkem1024`, `nightly-2026-09-25`, on `c8i.2xlarge` (Intel), `c8a.2xlarge` (AMD), and `c9g.2xlarge` (Graviton5).
Run [#36376376070](https://github.com/loadingalias/rscrypto/actions/runs/36376376070) measured `931b738f` before the scrub; run [#36904620694](https://github.com/loadingalias/rscrypto/actions/runs/36904620694) measured `c2af568c`,
which runs G, J, and the PRF in the scrubbed worker (`b20a18de`).
The runs used different physical hosts,
so the comparison uses rscrypto's median divided by the competitor median from the same run.

Change in that ratio for ML-KEM-768 and ML-KEM-1024 decapsulation, one-shot and reused encoded key:

| Host         | Versus libcrux | Versus AWS-LC |
| ------------ | -------------- | ------------- |
| x86-64 Intel | −1.3% to +0.3% | +2.5% to +3.1% |
| x86-64 AMD   | −2.4% to +0.5% | −2.1% to +1.6% |
| Graviton5    | +1.4% to +2.6% | +1.5% to +2.4% |

The scrub costs about 2% of decapsulation where the signal is clear.
Median confidence half-widths were ≤0.5% except the earlier AMD run (≤3.3%);
the AMD ML-KEM-1024 rows are now ≤0.07%.
The earlier run measured decapsulation only,
so key generation and encapsulation have no pre-scrub comparison on these hosts.

Standing after the scrub, ML-KEM-768 (Intel, AMD, Graviton5):

- Key generation from a seed: rscrypto 10.71, 8.27, and 8.82 µs; libcrux 16.03, 11.72, and 17.23 µs.
  AWS-LC's row, which also times its internal entropy, is 12.48, 12.99, and 11.98 µs.
- One-shot decapsulation (`import-encoded`, identical work in every library): rscrypto 1.79x,
  1.33x, and 1.78x AWS-LC; 1.39x, 1.44x, and 0.97x libcrux.
- One-shot encapsulation (`import-encoded`): rscrypto 1.13x, 1.15x, and 0.77x libcrux.
  AWS-LC's internal-entropy row is faster on every host (rscrypto 1.62x, 1.17x, and 1.43x).

## 2026-10-01 P-384 ECDH agreement

GitHub Bench `p384-ecdh` on `c8i.2xlarge` (Intel Xeon 6975P-C), `c8a.2xlarge` (AMD EPYC 9R45), and `c9g.2xlarge` (Graviton5, Neoverse V3).
Each row is the median of `p384-ecdh/agreement/rscrypto-selected`; the ratio divides it by the AWS-LC (`aws-lc-rs-native`) median from the same run,
so values above 1.00x mean rscrypto is slower.
Median confidence half-widths are ≤0.11% except the AMD row of run #36932716916 (≤0.40%).

| Run | Source | Toolchain | Intel | AMD | Graviton5 |
| --- | --- | --- | --- | --- | --- |
| [#36908596212](https://github.com/loadingalias/rscrypto/actions/runs/36908596212) | `c2af568c` | `nightly-2026-09-25` | 127.22 µs, 1.067x | 97.70 µs, 1.077x | 131.38 µs, 1.007x |
| [#36930494688](https://github.com/loadingalias/rscrypto/actions/runs/36930494688) | `fce695f6` | `nightly-2026-09-30` | 121.34 µs, 1.018x | 96.45 µs, 1.057x | 128.55 µs, 0.989x |
| [#36932716916](https://github.com/loadingalias/rscrypto/actions/runs/36932716916) | `129ea97a` (reverted) | `nightly-2026-09-30` | 127.97 µs, 1.072x | 98.48 µs, 1.079x | 128.89 µs, 0.990x |

`fce695f6` adds the affine window table (`b41e5e5c`) and the add-and-select field finish in the x86-64 doubling to `c2af568c`,
and moves the compiler pin; the run does not separate their effects.
`129ea97a` computed the doubling's squares as interleaved products;
it was slower on both x86-64 hosts and `4add945a` reverts it, restoring the `fce695f6` tree.
AWS-LC moved by at most 0.7% between runs.

Retained for v0.10.0 (the `fce695f6` tree):
P-384 agreement is ahead of AWS-LC on Graviton5 (0.989x) and behind it on x86-64,
by 1.8% on Intel and 5.7% on AMD.
In the same run rscrypto is 2.13x, 2.42x, and 2.15x faster than ring and 2.93x, 3.43x,
and 2.70x faster than RustCrypto `p384` (Intel, AMD, Graviton5).
The public-key row compares against AWS-LC's cached public key,
so it supports no key-derivation claim.

## 2026-09 allocator-adoption runs

GitHub Bench runs keep their Criterion artifacts.
Hosts: x86-64 Intel, x86-64 AMD, and AArch64 Linux.

- ML-KEM, `ebe24ea9` (run [#36483389850](https://github.com/loadingalias/rscrypto/actions/runs/36483389850)) versus `b672f572`
  (run [#36495823483](https://github.com/loadingalias/rscrypto/actions/runs/36495823483)):
  key preparation was 0.4–3.0% faster on all nine host and parameter-set pairs;
  every other row stayed within ±1.6% in both directions.
- Caller-provided work memory, `2c901505`
  (run [#36497076205](https://github.com/loadingalias/rscrypto/actions/runs/36497076205)), fresh versus reused memory within one run:

  | Host          | Argon2id 19 MiB                           | RustCrypto, reused | scrypt 128 MiB |
  | ------------- | ----------------------------------------- | ------------------ | -------------- |
  | x86-64 Intel  | 9.75 → 8.85 ms (−9.2%)                    | 13.82 ms           | 245.7 → 198.8 ms (−19.1%) |
  | x86-64 AMD    | 6.02 → 5.90 ms (−1.9%, intervals overlap) | 10.54 ms           | 327.4 → 295.1 ms (−9.8%) |
  | AArch64 Linux | 11.65 → 11.55 ms (−0.8%)                  | 11.40 ms           | 161.1 → 130.7 ms (−18.9%) |

  RustCrypto's reused row does not clear its buffer; rscrypto clears it on every call.
  Apple Silicon showed no reuse gain.
  The cost is keeping the buffer resident between calls.

No compiler-driven performance claim exists for Rust 1.100:
the comparison of the pre-change implementation on Rust 1.98.1,
the same implementation on the release compiler,
and the release candidate on that compiler has not run.

## P-256 ECDH development snapshot

The Apple M1 run measured complete API operations with Criterion.
Rscrypto medians were 3.5779 ns for caller-filled ephemeral generation,
7.8186 us for public derivation, 111.74 ns for canonical SEC1 parsing, 34.297 us for agreement,
and 85.106 us for a two-party TLS-shaped roundtrip.
The fastest equivalent competitors were RustCrypto at 4.9748 ns for generation,
`ring` at 10.579 us for public derivation, CRRL at 121.05 ns for parsing,
and AWS-LC at 34.990 us for agreement.
Under the repository's +/-5% classification these are three wins and one agreement tie.
AWS-LC's 1.2915 us cached-public row excludes key import/precomputation and is retained only
as a non-equivalent diagnostic; its equivalent import-plus-public row measured 15.769 us.

The physical Graviton4 run measured 7.0706 ns for caller-filled generation,
10.187 us for public derivation, 149.09 ns for canonical parsing, 48.241 us for agreement,
and 117.23 us for the TLS-shaped roundtrip.
Public derivation beat `ring` at 12.373 us and the equivalent AWS-LC import-plus-public row at 18.187 us.
Agreement tied the fastest native competitors while narrowly leading AWS-LC at 48.715 us
and `ring` at 49.509 us.
Parsing was within the repository's 5% tie band of CRRL at 141.76 ns
and ahead of RustCrypto at 206.73 ns.
AWS-LC's 1.4721 us cached-public row remains a non-equivalent diagnostic because it excludes import
and precomputation.

The physical Graviton3 run measured 8.7453 ns for caller-filled generation,
11.939 us for public derivation, 182.76 ns for canonical parsing, 56.296 us for agreement,
and 136.89 us for the TLS-shaped roundtrip.
Public derivation beat `ring` at 14.100 us and the equivalent AWS-LC import-plus-public row at 21.537 us.
Agreement tied AWS-LC at 56.262 us and was faster than `ring` at 58.263 us.
Generation tied RustCrypto at 9.0616 ns.
Parsing is a measured loss: CRRL completed the same operation in 166.98 ns, about 8.6% less time.
Under the repository's 5% classification, the G3 result is one win, two ties, and one loss.
AWS-LC's 1.7023 us cached-public row remains a non-equivalent diagnostic.
This is an intermediate-candidate result:
later shared parser and dispatch changes have not been rerun on Graviton3,
so the retained parsing loss is not a measurement of the exact final source.

The retained physical Linux Intel Granite Rapids run measured 3.9060 ns
for caller-filled generation, 8.5669 us for public derivation, 83.420 ns for canonical parsing,
36.130 us for agreement, and 90.004 us for the TLS-shaped roundtrip.
Generation beat RustCrypto at 10.297 ns,
and public derivation beat `ring` at 10.712 us
and the equivalent AWS-LC import-plus-public row at 16.109 us.
Agreement tied AWS-LC at 37.255 us while beating `ring` at 45.600 us.
Parsing narrowly led CRRL at 83.878 ns;
the repository's 5% classification treats that difference as a tie.
The result is two wins and two ties.
AWS-LC's 1.4492 us cached-public row remains a non-equivalent diagnostic.

The physical Windows x86-64 Intel Granite Rapids full run measured 4.2754 ns
for caller-filled generation, 8.6592 us for public derivation, 36.188 us for agreement,
and 90.309 us for the TLS-shaped roundtrip.
Generation beat RustCrypto at 18.837 ns;
public derivation beat `ring` at 9.9571 us and the equivalent AWS-LC import-plus-public row at 18.001 us;
agreement beat AWS-LC at 43.019 us and `ring` at 42.176 us.
After batching the five native public-field operations behind one Microsoft x64 ABI boundary,
the exact final parser-only run measured 82.507 ns against CRRL at 83.339 ns,
with non-overlapping Criterion intervals.
That is faster in the same run and a tie under the repository's conservative 5% classification.
The exact-final-source whole-operation hardware benchmark remains awaiting a future physical run.
The Windows qualification row is wired to retain exact-source P-256 timing and cleanup evidence,
but its first successful artifact is still pending.

This snapshot evaluates the independently proven safe Rust authority everywhere
and embedded s2n-bignum assembly for Apple/Linux AArch64 and Linux/Windows x86-64 public derivation
and agreement.
The candidate Linux and Apple assembly passed portable differential, NIST, Wycheproof,
native timing, cleanup, and deterministic provenance gates on M1 and physical G3/G4/Intel
as scoped above.
Those development bundles predate later shared-source edits
and are not exact-final release evidence; the final Windows backend has native differential,
independent-oracle, and performance evidence,
with exact-final-source qualification timing and cleanup still open until the wired job succeeds.
The retained G4 DudeCT maxima were 1.8903 for public derivation and 1.55471 for agreement;
the G3 maxima were 1.12000 and 2.59291 respectively, all against threshold 10.
Target qualification remains owned by [`docs/platforms.md`](../docs/platforms.md), [`docs/constant-time.md`](../docs/constant-time.md),
and `ct.toml`.
These results must be rerun from the exact candidate commit before publication.

Host coverage change: this run has eight Linux hosts.
The RISE RISC-V host did not contribute results in run #32185659553,
so every aggregate below is over eight platforms rather than the nine in the 2026-07-04 snapshot.
Row counts are therefore not directly comparable to that snapshot; ratios and geomeans are.

Equivalence correction resolved:
the historical RustCrypto HMAC-SHA-256 rows included key setup inside the timed loop.
The current benchmark source hoists `RustCryptoHmacSha256::new_from_slice` out of the timed loop and clones the keyed state per iteration,
matching the reusable-keyed-state treatment given to rscrypto, ring, and AWS-LC.
This artifact is a complete regenerated benchmark pass,
so the HMAC-SHA-256 rows and the aggregates
that include them are equivalent-work performance claims.

Surface change since 2026-07-04: the rapidhash benchmark surface was collapsed.
The former `rapidhash-64`, `rapidhash-128`, and `rapidhash-v3-128` primitives no longer exist; `rapidhash-v3-64`, `rapidhash-stream`, `rapidhash-buildhasher`, `rapidhash-hash-one`, and `rapidhash-hashmap` are the current rows.

Coverage note: this is a full Linux public benchmark pass.
It includes checksum, hash, XOF, MAC, KDF, password-hashing, BLAKE2/BLAKE3, RSA import/verification,
ECDSA P-256/P-384 signing and verification, Ed25519, X25519, AEAD, and ML-KEM-512/768/1024 keygen,
encapsulation, and decapsulation rows.
ML-KEM phase/arithmetic microbenches are present in the raw artifacts
and intentionally excluded from release-level competitor claims.

## 2026-07-28 Ed25519 Direct-Secret Diagnostic

This local diagnostic compares the exact 1 KiB `ed25519/sign/rscrypto-direct-secret/1024` Criterion case before and
after the maintenance remediation that removed duplicate secret expansion.
The baseline source is repository commit `c7338116bf8155566f9a028db1b28b5f0665e370` with only the identical benchmark row added.
The current source is that commit plus the maintenance working-tree diff.

Both runs used the pinned `rustc 1.97.0-nightly (ca9a134e0 2026-04-26)` toolchain on the same Apple Silicon macOS host.
Criterion used 50 samples, a 1-second warm-up, and a 3-second measurement window.

| Source   |    Median | 95% confidence interval | Mean |
| -------- | --------: | ----------------------: | ---: |
| Baseline | 21.892 µs |        21.874–21.919 µs | 21.885 µs |
| Current  | 21.754 µs |        21.704–21.799 µs | 21.757 µs |

The observed current/baseline median ratio is 0.9937.
This check found no regression.
It was not an interleaved release benchmark, so it does not support a speedup claim.

The repository policy retains only this curated overview.
The local Criterion metadata, estimates,
and raw 50-sample files were distinct and hashed before curation:

| Artifact | Baseline SHA-256 | Current SHA-256 |
| ---------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------ |
| `benchmark.json` | `6d27e19fd2a9563ecea5328345420c12b79f9924d3ecde179bc0166f5a62e6dd` | `6d27e19fd2a9563ecea5328345420c12b79f9924d3ecde179bc0166f5a62e6dd` |
| `estimates.json` | `728945652c3ec804ec064e9888fc431a5fa3528e885edf76e350392ae95ea2fc` | `3b987405f949847972740cb549826d46f2529caa1187bc13786f7d662ca63e03` |
| `sample.json` | `f36052bcf65362d6203a6be768e251822dc3182ce8fab75dd9bba20097db30f9` | `a92a7e9fcc1af048c2bb5dcfd8782d07b8727e46477a27bc7948cd02c7a8a6bc` |

## 2026-08-18 Linux snapshot (historical)

Scope: the 2026-08-18 eight-host Linux benchmark matrix for commit `7eb44e9`.
Ratios are `external_crate_time / rscrypto_time`; higher is better.
Wins are `>1.05x`, ties are `0.95x..1.05x`, and losses are `<0.95x`.
Fastest-external comparisons keep only the fastest external implementation for each platform,
primitive, operation, and input shape.
Internal kernel, scratch-buffer, padding-only, cold-path, PHC roundtrip, parallel-scaling,
threshold-selection, public-overhead,
and phase-attribution microbenches are parsed as raw rows
but excluded from external win/loss claims.
The macOS local run is listed separately and is not mixed into Linux claims.

This is a historical snapshot of commit `7eb44e9`, not an inventory of the current public API.
Primitive rows remain as measured even when a later commit changes or removes that surface.

The aggregates in the sections below include the withdrawn ML-KEM and Argon2 rows.
They are historical records, not current claims.

## Headline (2026-08-18, historical)

| Scope                                | Pairs | W/T/L           | Win % | Geomean | Median |
| ------------------------------------ | ----- | --------------- | ----- | ------- | ------ |
| Linux: all matched performance pairs | 9,674 | 6,831/2,085/758 | 71%   | 1.78x   | 1.24x  |
| Linux: fastest external per case     | 6,144 | 3,780/1,695/669 | 62%   | 1.62x   | 1.12x  |

Snapshot summary:

- **Headline:** 3,780 of 6,144 matched Linux fastest-external comparisons are wins;
  5,475 are wins or ties.
  Linux fastest-external geomean is 1.62x.
- **Checksums:** 6.18x geomean across 616 fastest-external rows; W/T/L is 476/118/22.
- **Hashes/MACs/XOFs:** 1.35x geomean across 3,456 fastest-external rows; W/T/L is 1,926/1,181/349.
- **Auth/KDF:** 1.28x geomean across 160 fastest-external rows; W/T/L is 140/20/0.
- **Password hashing:** 1.07x geomean across 120 fastest-external rows; W/T/L is 55/27/38.
- **Public-key:** 1.09x geomean across 296 fastest-external rows; W/T/L is 187/59/50.
- **RSA:** 1.65x geomean across 88 fastest-external rows; W/T/L is 86/2/0.
- **AEAD:** 1.61x geomean across 1,408 fastest-external rows; W/T/L is 910/288/210.
- **ML-KEM:** 1.55x geomean across 72 fastest-external rows; W/T/L is 64/0/8.
- **ECDSA P-256/P-384:** Linux 0.87x geomean across 128 fastest-external rows; W/T/L is 88/7/33.
- **Top current loss areas:** `ecdsa-p384` / `sign`: 0.70x geomean across 32 rows; W/T/L is 12/0/20; pressure `aws-lc-rs` 16, `rustcrypto-p384` 4;
  `ecdsa-p256` / `verify`: 0.84x geomean across 32 rows; W/T/L is 20/7/5; pressure `rustcrypto-p256` 4, `aws-lc-rs` 1; `rapidhash-stream` / `one-write`:
  0.87x geomean across 88 rows; W/T/L is 27/25/36; pressure `rapidhash` 36; `ecdsa-p256` / `sign`: 0.91x geomean across 32 rows;
  W/T/L is 28/0/4; pressure `ring` 4; `argon2id-owasp` / `hash`: 0.98x geomean across 8 rows; W/T/L is 3/1/4; pressure `rustcrypto` 3, `dryoc` 1.

## Coverage Matrix

| Platform | Raw Criterion rows | All pairs | Fastest rows | W/T/L | Win % | Geomean | Median |
| --------------------- | ------------------ | --------- | ------------ | ----------- | ----- | ------- | ------ |
| AMD Zen4 | 2,304 | 1,269 | 768 | 525/171/72 | 68% | 1.47x | 1.14x |
| AMD Zen5 | 2,304 | 1,269 | 768 | 447/245/76 | 58% | 1.47x | 1.10x |
| AWS Graviton3 | 2,308 | 1,269 | 768 | 367/287/114 | 48% | 1.36x | 1.04x |
| AWS Graviton4 | 2,308 | 1,269 | 768 | 366/337/65 | 48% | 1.37x | 1.04x |
| IBM Power10 | 2,055 | 1,030 | 768 | 400/302/66 | 52% | 1.83x | 1.06x |
| IBM z16/s390x | 2,055 | 1,030 | 768 | 620/67/81 | 81% | 2.77x | 2.19x |
| Intel Ice Lake | 2,304 | 1,269 | 768 | 517/137/114 | 67% | 1.45x | 1.17x |
| Intel Sapphire Rapids | 2,304 | 1,269 | 768 | 538/149/81 | 70% | 1.60x | 1.18x |

## Category Summary

| Category         | Rows  | W/T/L           | Win % | Geomean | Median |
| ---------------- | ----- | --------------- | ----- | ------- | ------ |
| Checksums        | 616   | 476/118/22      | 77%   | 6.18x   | 3.17x  |
| Hashes/MACs/XOFs | 3,456 | 1,926/1,181/349 | 56%   | 1.35x   | 1.08x  |
| Auth/KDF         | 160   | 140/20/0        | 88%   | 1.28x   | 1.13x  |
| Password hashing | 120   | 55/27/38        | 46%   | 1.07x   | 1.02x  |
| Public-key       | 296   | 187/59/50       | 63%   | 1.09x   | 1.14x  |
| RSA              | 88    | 86/2/0          | 98%   | 1.65x   | 1.20x  |
| AEAD             | 1,408 | 910/288/210     | 65%   | 1.61x   | 1.21x  |

## BLAKE3 Summary

BLAKE3 rows come from the Linux snapshot.
All-pair and fastest-external BLAKE3 metrics are identical
because official `blake3` is the only external implementation in this bench.

| Scope                 | Rows | W/T/L      | Geomean | Median |
| --------------------- | ---- | ---------- | ------- | ------ |
| All Linux BLAKE3 rows | 384  | 187/134/63 | 1.35x   | 1.04x  |
| x86_64                | 192  | 79/89/24   | 1.18x   | 1.02x  |
| AArch64               | 96   | 44/36/16   | 1.40x   | 1.04x  |

| Platform              | Rows | W/T/L    | Geomean | Median |
| --------------------- | ---- | -------- | ------- | ------ |
| AMD Zen4              | 48   | 20/22/6  | 1.24x   | 1.01x  |
| AMD Zen5              | 48   | 18/27/3  | 1.27x   | 1.02x  |
| AWS Graviton3         | 48   | 22/15/11 | 1.36x   | 0.98x  |
| AWS Graviton4         | 48   | 22/21/5  | 1.44x   | 1.04x  |
| IBM Power10           | 48   | 32/6/10  | 1.76x   | 1.12x  |
| IBM z16/s390x         | 48   | 32/3/13  | 1.69x   | 1.69x  |
| Intel Ice Lake        | 48   | 19/21/8  | 1.09x   | 1.00x  |
| Intel Sapphire Rapids | 48   | 22/19/7  | 1.13x   | 1.03x  |

| Operation    | Rows | W/T/L    | Geomean | Median |
| ------------ | ---- | -------- | ------- | ------ |
| `oneshot`    | 88   | 35/35/18 | 1.33x   | 1.00x  |
| `keyed`      | 88   | 27/21/40 | 1.20x   | 0.95x  |
| `derive-key` | 88   | 65/21/2  | 1.59x   | 1.53x  |
| `streaming`  | 32   | 10/21/1  | 1.21x   | 1.02x  |
| `xof`        | 88   | 50/36/2  | 1.37x   | 1.07x  |

## ML-KEM Summary

ML-KEM public coverage is complete for the selected primitive set: ML-KEM-512, ML-KEM-768,
and ML-KEM-1024 each include keygen, encapsulate, and decapsulate on all eight Linux platforms.
POWER10 and s390x do not have `aws-lc-rs` ML-KEM rows in this artifact set, but still have rscrypto plus `libcrux`, `fips203`,
and RustCrypto comparison rows for every public operation.

| Platform | Raw ML-KEM rows | Fastest rows | W/T/L | Geomean | Median | Fastest external split |
| --------------------- | --------------- | ------------ | ----- | ------- | ------ | -------------------------- |
| AMD Zen4 | 45 | 9 | 9/0/0 | 1.83x | 1.82x | `libcrux` 7, `aws-lc-rs` 2 |
| AMD Zen5 | 45 | 9 | 9/0/0 | 1.95x | 1.91x | `libcrux` 9 |
| AWS Graviton3 | 45 | 9 | 5/0/4 | 1.09x | 1.12x | `aws-lc-rs` 9 |
| AWS Graviton4 | 45 | 9 | 5/0/4 | 1.08x | 1.18x | `aws-lc-rs` 9 |
| IBM Power10 | 36 | 9 | 9/0/0 | 1.41x | 1.53x | `libcrux` 9 |
| IBM z16/s390x | 36 | 9 | 9/0/0 | 1.68x | 1.74x | `libcrux` 9 |
| Intel Ice Lake | 45 | 9 | 9/0/0 | 1.80x | 1.75x | `libcrux` 7, `aws-lc-rs` 2 |
| Intel Sapphire Rapids | 45 | 9 | 9/0/0 | 1.84x | 1.80x | `aws-lc-rs` 7, `libcrux` 2 |

| Primitive/op                | Rows | W/T/L | Win % | Geomean | Median | Pressure |
| --------------------------- | ---- | ----- | ----- | ------- | ------ | -------- |
| `mlkem1024` / `decapsulate` | 8    | 8/0/0 | 100%  | 1.70x   | 1.86x  | none     |
| `mlkem1024` / `encapsulate` | 8    | 8/0/0 | 100%  | 2.51x   | 2.63x  | none     |
| `mlkem1024` / `keygen`      | 8    | 6/0/2 | 75%   | 1.02x   | 1.13x  | `aws-lc-rs` 2 |
| `mlkem512` / `decapsulate`  | 8    | 6/0/2 | 75%   | 1.41x   | 1.59x  | `aws-lc-rs` 2 |
| `mlkem512` / `encapsulate`  | 8    | 8/0/0 | 100%  | 1.94x   | 2.17x  | none     |
| `mlkem512` / `keygen`       | 8    | 6/0/2 | 75%   | 1.09x   | 1.22x  | `aws-lc-rs` 2 |
| `mlkem768` / `decapsulate`  | 8    | 8/0/0 | 100%  | 1.58x   | 1.75x  | none     |
| `mlkem768` / `encapsulate`  | 8    | 8/0/0 | 100%  | 2.33x   | 2.54x  | none     |
| `mlkem768` / `keygen`       | 8    | 6/0/2 | 75%   | 1.06x   | 1.13x  | `aws-lc-rs` 2 |

## ECDSA Summary

ECDSA signing includes both deterministic and blinded rscrypto rows in raw results;
aggregate fastest-external comparisons use the fastest rscrypto row for the exact case.
Constant-time release evidence is tracked separately by `ct.toml` and CT artifacts.

Regression: every ECDSA aggregate in this snapshot is dominated by a single platform.
On IBM z16/s390x, P-256 signing went from 137.10 µs (2026-07-04) to 8,889.30 µs,
and P-384 signing from 562.91 µs to 34,557.00 µs,
while the external crates on the same host moved by less than 1.4x.
Excluding s390x, the seven-host geomeans are `ecdsa-p256` / `sign` 1.33x, `ecdsa-p256` / `verify` 1.19x, `ecdsa-p384` / `sign` 1.01x, and `ecdsa-p384` / `verify` 1.53x.

| Operation               | Rows | W/T/L   | Geomean | Median |
| ----------------------- | ---- | ------- | ------- | ------ |
| `ecdsa-p256` / `sign`   | 32   | 28/0/4  | 0.91x   | 1.30x  |
| `ecdsa-p256` / `verify` | 32   | 20/7/5  | 0.84x   | 1.08x  |
| `ecdsa-p384` / `sign`   | 32   | 12/0/20 | 0.70x   | 0.83x  |
| `ecdsa-p384` / `verify` | 32   | 28/0/4  | 1.08x   | 1.36x  |

## Primitive Summary

Linux primitives with matched exact `rscrypto` comparisons.
Fastest columns are strongest-external comparisons;
all-pair columns include every matched external implementation.

| Primitive | Fastest rows | Fastest W/T/L | Fastest geomean | All pairs | All W/T/L | All geomean |
| ----------------------- | ------------ | ------------- | --------------- | --------- | ---------- | ----------- |
| `ecdsa-p384` | 64 | 40/0/24 | 0.87x | 176 | 144/0/32 | 2.27x |
| `ecdsa-p256` | 64 | 48/7/9 | 0.87x | 176 | 148/11/17 | 1.57x |
| `rapidhash-stream` | 176 | 61/33/82 | 0.92x | 176 | 61/33/82 | 0.92x |
| `argon2id-owasp` | 8 | 3/1/4 | 0.98x | 16 | 7/4/5 | 1.25x |
| `xxh3-buildhasher` | 88 | 41/12/35 | 0.99x | 88 | 41/12/35 | 0.99x |
| `x25519` | 16 | 3/13/0 | 1.02x | 44 | 31/13/0 | 1.58x |
| `argon2i-small` | 24 | 10/3/11 | 1.03x | 40 | 26/3/11 | 1.34x |
| `argon2id-small` | 24 | 10/3/11 | 1.03x | 40 | 25/4/11 | 1.35x |
| `argon2d-small` | 24 | 10/5/9 | 1.04x | 24 | 10/5/9 | 1.04x |
| `rapidhash-v3-64` | 88 | 21/45/22 | 1.05x | 88 | 21/45/22 | 1.05x |
| `blake2b256` | 200 | 101/99/0 | 1.07x | 312 | 204/108/0 | 1.31x |
| `scrypt-owasp` | 8 | 4/2/2 | 1.08x | 8 | 4/2/2 | 1.08x |
| `blake2b512` | 176 | 106/69/1 | 1.08x | 264 | 194/69/1 | 1.33x |
| `blake2s256` | 200 | 114/86/0 | 1.11x | 200 | 114/86/0 | 1.11x |
| `chacha20-poly1305` | 176 | 75/101/0 | 1.12x | 484 | 304/180/0 | 1.32x |
| `xxh3-128` | 88 | 34/42/12 | 1.13x | 88 | 34/42/12 | 1.13x |
| `xxh3-64` | 88 | 34/34/20 | 1.13x | 88 | 34/34/20 | 1.13x |
| `blake2s128` | 176 | 113/63/0 | 1.13x | 176 | 113/63/0 | 1.13x |
| `ed25519` | 80 | 32/39/9 | 1.14x | 256 | 194/48/14 | 1.41x |
| `xxh3-hashmap` | 8 | 7/1/0 | 1.15x | 8 | 7/1/0 | 1.15x |
| `scrypt-small` | 32 | 18/13/1 | 1.18x | 32 | 18/13/1 | 1.18x |
| `rapidhash-buildhasher` | 88 | 44/29/15 | 1.19x | 88 | 44/29/15 | 1.19x |
| `aegis-256` | 176 | 81/65/30 | 1.23x | 176 | 81/65/30 | 1.23x |
| `hmac-sha256` | 104 | 42/36/26 | 1.24x | 258 | 144/78/36 | 1.60x |
| `hmac-sha384` | 88 | 28/49/11 | 1.24x | 242 | 133/93/16 | 1.29x |
| `hmac-sha512` | 88 | 32/44/12 | 1.27x | 242 | 137/88/17 | 1.31x |
| `sha256` | 104 | 44/46/14 | 1.27x | 258 | 143/89/26 | 1.60x |
| `hkdf-sha384` | 32 | 29/3/0 | 1.27x | 88 | 85/3/0 | 1.59x |
| `rsa-8192` | 16 | 14/2/0 | 1.28x | 28 | 26/2/0 | 1.33x |
| `hkdf-sha256` | 32 | 27/5/0 | 1.28x | 88 | 83/5/0 | 1.93x |
| `pbkdf2-sha256` | 48 | 43/5/0 | 1.28x | 132 | 127/5/0 | 1.71x |
| `pbkdf2-sha512` | 48 | 41/7/0 | 1.28x | 132 | 125/7/0 | 1.34x |
| `sha512` | 104 | 48/51/5 | 1.29x | 258 | 160/88/10 | 1.31x |
| `sha384` | 88 | 43/39/6 | 1.30x | 242 | 151/80/11 | 1.32x |
| `ascon-hash256` | 88 | 56/31/1 | 1.30x | 88 | 56/31/1 | 1.30x |
| `sha512-256` | 88 | 50/38/0 | 1.33x | 88 | 50/38/0 | 1.33x |
| `blake3` | 384 | 187/134/63 | 1.35x | 384 | 187/134/63 | 1.35x |
| `ascon-aead128` | 176 | 136/39/1 | 1.39x | 176 | 136/39/1 | 1.39x |
| `ascon-xof128` | 88 | 66/20/2 | 1.39x | 88 | 66/20/2 | 1.39x |
| `xchacha20-poly1305` | 176 | 173/3/0 | 1.43x | 176 | 173/3/0 | 1.43x |
| `mlkem512` | 24 | 20/0/4 | 1.44x | 90 | 86/0/4 | 2.90x |
| `rapidhash-hash-one` | 24 | 18/4/2 | 1.47x | 24 | 18/4/2 | 1.47x |
| `mlkem768` | 24 | 22/0/2 | 1.57x | 90 | 88/0/2 | 3.38x |
| `rapidhash-hashmap` | 24 | 24/0/0 | 1.61x | 24 | 24/0/0 | 1.61x |
| `mlkem1024` | 24 | 22/0/2 | 1.63x | 90 | 88/0/2 | 3.60x |
| `rsa-4096` | 24 | 24/0/0 | 1.70x | 52 | 52/0/0 | 2.69x |
| `crc32c` | 88 | 42/38/8 | 1.73x | 176 | 130/38/8 | 2.41x |
| `rsa-3072` | 24 | 24/0/0 | 1.75x | 52 | 52/0/0 | 2.73x |
| `rsa-2048` | 24 | 24/0/0 | 1.79x | 52 | 52/0/0 | 2.77x |
| `aes-128-gcm` | 176 | 96/42/38 | 1.80x | 484 | 390/50/44 | 2.01x |
| `crc32` | 88 | 47/33/8 | 1.80x | 176 | 133/35/8 | 2.51x |
| `aes-256-gcm` | 176 | 94/36/46 | 1.83x | 484 | 382/44/58 | 2.02x |
| `kmac256` | 88 | 58/19/11 | 1.86x | 88 | 58/19/11 | 1.86x |
| `cshake256` | 88 | 58/21/9 | 1.90x | 88 | 58/21/9 | 1.90x |
| `shake128` | 88 | 58/30/0 | 1.94x | 88 | 58/30/0 | 1.94x |
| `shake256` | 88 | 63/25/0 | 1.98x | 88 | 63/25/0 | 1.98x |
| `sha224` | 88 | 51/37/0 | 2.01x | 88 | 51/37/0 | 2.01x |
| `aes-128-gcm-siv` | 176 | 127/1/48 | 2.20x | 308 | 237/16/55 | 2.92x |
| `sha3-224` | 88 | 77/11/0 | 2.27x | 88 | 77/11/0 | 2.27x |
| `sha3-256` | 104 | 91/13/0 | 2.28x | 104 | 91/13/0 | 2.28x |
| `aes-256-gcm-siv` | 176 | 128/1/47 | 2.34x | 308 | 259/2/47 | 3.16x |
| `crc64-nvme` | 88 | 52/35/1 | 2.34x | 88 | 52/35/1 | 2.34x |
| `sha3-384` | 88 | 79/9/0 | 2.35x | 88 | 79/9/0 | 2.35x |
| `sha3-512` | 88 | 77/11/0 | 2.38x | 88 | 77/11/0 | 2.38x |
| `crc64-xz` | 88 | 73/12/3 | 2.78x | 88 | 73/12/3 | 2.78x |
| `crc24-openpgp` | 88 | 86/0/2 | 17.62x | 88 | 86/0/2 | 17.62x |
| `crc16-ccitt` | 88 | 88/0/0 | 30.24x | 88 | 88/0/0 | 30.24x |
| `crc16-ibm` | 88 | 88/0/0 | 32.07x | 88 | 88/0/0 | 32.07x |

## Linux Worst Individual Rows

| Platform      | Case                         | Fastest external  | Ratio |
| ------------- | ---------------------------- | ----------------- | ----- |
| IBM z16/s390x | `ecdsa-p256 / sign / 1024`   | `ring`            | 0.05x |
| IBM z16/s390x | `ecdsa-p384 / sign / 16384`  | `rustcrypto-p384` | 0.05x |
| IBM z16/s390x | `ecdsa-p384 / sign / 1024`   | `rustcrypto-p384` | 0.05x |
| IBM z16/s390x | `ecdsa-p384 / sign / 0`      | `rustcrypto-p384` | 0.06x |
| IBM z16/s390x | `ecdsa-p384 / sign / 32`     | `rustcrypto-p384` | 0.06x |
| IBM z16/s390x | `ecdsa-p256 / sign / 0`      | `ring`            | 0.06x |
| IBM z16/s390x | `ecdsa-p256 / sign / 32`     | `ring`            | 0.06x |
| IBM z16/s390x | `ecdsa-p256 / verify / 32`   | `rustcrypto-p256` | 0.06x |
| IBM z16/s390x | `ecdsa-p256 / verify / 1024` | `rustcrypto-p256` | 0.07x |
| IBM z16/s390x | `ecdsa-p256 / sign / 16384`  | `ring`            | 0.07x |
| IBM z16/s390x | `ecdsa-p256 / verify / 0`    | `rustcrypto-p256` | 0.07x |
| IBM z16/s390x | `ecdsa-p384 / verify / 1024` | `rustcrypto-p384` | 0.09x |

## Linux Strongest Individual Rows

| Platform              | Case                    | Fastest external | Ratio |
| --------------------- | ----------------------- | ---------------- | ----- |
| Intel Sapphire Rapids | `crc16-ibm / 262144`    | `crc`            | 212.60x |
| Intel Sapphire Rapids | `crc16-ccitt / 262144`  | `crc`            | 209.27x |
| Intel Sapphire Rapids | `crc16-ccitt / 16384`   | `crc`            | 206.40x |
| Intel Sapphire Rapids | `crc16-ibm / 16384`     | `crc`            | 198.48x |
| Intel Sapphire Rapids | `crc16-ibm / 1048576`   | `crc`            | 187.52x |
| Intel Sapphire Rapids | `crc16-ibm / 4096`      | `crc`            | 178.55x |
| Intel Sapphire Rapids | `crc16-ibm / 65536`     | `crc`            | 178.28x |
| Intel Sapphire Rapids | `crc16-ccitt / 4096`    | `crc`            | 178.15x |
| IBM Power10           | `crc16-ccitt / 1048576` | `crc`            | 176.67x |
| IBM Power10           | `crc16-ibm / 1048576`   | `crc`            | 176.60x |
| Intel Sapphire Rapids | `crc16-ccitt / 1048576` | `crc`            | 176.46x |
| IBM Power10           | `crc16-ccitt / 262144`  | `crc`            | 175.61x |

## Top Five Loss Areas

- `ecdsa-p384` / `sign`: 0.70x geomean across 32 rows; W/T/L 12/0/20; pressure `aws-lc-rs` 16, `rustcrypto-p384` 4.
- `ecdsa-p256` / `verify`: 0.84x geomean across 32 rows; W/T/L 20/7/5; pressure `rustcrypto-p256` 4, `aws-lc-rs` 1.
- `rapidhash-stream` / `one-write`: 0.87x geomean across 88 rows; W/T/L 27/25/36; pressure `rapidhash` 36.
- `ecdsa-p256` / `sign`: 0.91x geomean across 32 rows; W/T/L 28/0/4; pressure `ring` 4.
- `argon2id-owasp` / `hash`: 0.98x geomean across 8 rows; W/T/L 3/1/4; pressure `rustcrypto` 3, `dryoc` 1.

## External Pressure

| External          | Pairs | W/T/L         | Win % | Geomean | Median |
| ----------------- | ----- | ------------- | ----- | ------- | ------ |
| `rapidhash`       | 400   | 168/111/121   | 42%   | 1.07x   | 1.01x  |
| `xxhash-rust`     | 272   | 116/89/67     | 43%   | 1.08x   | 1.00x  |
| `aws-lc-rs`       | 1,434 | 896/343/195   | 62%   | 1.21x   | 1.13x  |
| `aegis-crate`     | 176   | 81/65/30      | 46%   | 1.23x   | 1.04x  |
| `ascon-hash`      | 176   | 122/51/3      | 69%   | 1.34x   | 1.32x  |
| `blake3`          | 384   | 187/134/63    | 49%   | 1.35x   | 1.04x  |
| `ascon-aead`      | 176   | 136/39/1      | 77%   | 1.39x   | 1.38x  |
| `dalek`           | 96    | 80/12/4       | 83%   | 1.52x   | 1.49x  |
| `sha2`            | 472   | 276/194/2     | 58%   | 1.60x   | 1.07x  |
| `ring`            | 1,472 | 1,154/237/81  | 78%   | 1.63x   | 1.28x  |
| `libcrux`         | 72    | 72/0/0        | 100%  | 1.79x   | 1.72x  |
| `dryoc`           | 320   | 293/22/5      | 92%   | 1.81x   | 1.85x  |
| `rustcrypto`      | 2,440 | 1,783/529/128 | 73%   | 1.87x   | 1.21x  |
| `tiny-keccak`     | 352   | 237/95/20     | 67%   | 1.92x   | 2.10x  |
| `crc-fast`        | 264   | 153/101/10    | 58%   | 2.15x   | 1.20x  |
| `sha3`            | 368   | 324/44/0      | 88%   | 2.32x   | 2.15x  |
| `crc32fast`       | 88    | 79/4/5        | 90%   | 2.75x   | 2.06x  |
| `crc64fast`       | 88    | 73/12/3       | 83%   | 2.78x   | 2.49x  |
| `rustcrypto-p256` | 64    | 56/0/8        | 88%   | 3.03x   | 3.10x  |
| `rustcrypto-p384` | 64    | 56/0/8        | 88%   | 3.06x   | 5.50x  |
| `crc32c`          | 88    | 83/3/2        | 94%   | 3.13x   | 2.29x  |
| `fips203`         | 72    | 72/0/0        | 100%  | 5.28x   | 6.07x  |
| `rustcrypto-rsa`  | 72    | 72/0/0        | 100%  | 6.07x   | 6.50x  |
| `crc`             | 264   | 262/0/2       | 99%   | 25.76x  | 46.98x |

## macOS Local Snapshot

The macOS Apple Silicon run is local evidence from the 2026-07-04 full benchmark at commit `596498f`,
carried forward unchanged in this refresh.
It is useful for Apple Silicon planning but is not folded into Linux release claims.
The ML-KEM row uses the same artifact's public ML-KEM rows.

| Scope                                      | Pairs | W/T/L      | Win % | Geomean | Median |
| ------------------------------------------ | ----- | ---------- | ----- | ------- | ------ |
| macOS local: all matched performance pairs | 1,297 | 815/404/78 | 63%   | 1.66x   | 1.16x  |
| macOS local: fastest external per case     | 774   | 382/326/66 | 49%   | 1.37x   | 1.05x  |
| macOS local: ML-KEM fastest external       | 9     | 6/1/2      | 67%   | 1.35x   | 1.39x  |

## Raw Results

| Platform              | Mode     | Date/time             | Parsed rows | Result |
| --------------------- | -------- | --------------------- | ----------- | ------ |
| AMD Zen4              | `remote` | `2026-08-18 21_03_07` | 2,304       | `benchmark_results/2026-08-18/linux/amd-zen4/results.txt` |
| AMD Zen5              | `remote` | `2026-08-18 21_03_07` | 2,304       | `benchmark_results/2026-08-18/linux/amd-zen5/results.txt` |
| AWS Graviton3         | `remote` | `2026-08-18 21_03_07` | 2,308       | `benchmark_results/2026-08-18/linux/graviton3/results.txt` |
| AWS Graviton4         | `remote` | `2026-08-18 21_03_07` | 2,308       | `benchmark_results/2026-08-18/linux/graviton4/results.txt` |
| IBM Power10           | `remote` | `2026-08-18 21_03_07` | 2,055       | `benchmark_results/2026-08-18/linux/ibm-power10/results.txt` |
| IBM z16/s390x         | `remote` | `2026-08-18 21_03_07` | 2,055       | `benchmark_results/2026-08-18/linux/ibm-s390x/results.txt` |
| Intel Ice Lake        | `remote` | `2026-08-18 21_03_07` | 2,304       | `benchmark_results/2026-08-18/linux/intel-icl/results.txt` |
| Intel Sapphire Rapids | `remote` | `2026-08-18 21_03_07` | 2,304       | `benchmark_results/2026-08-18/linux/intel-spr/results.txt` |
| macOS Apple Silicon   | `local`  | `2026-07-04 12_28_04` | 2,277       | `benchmark_results/2026-07-04/macos/aarch64/results.txt` |
