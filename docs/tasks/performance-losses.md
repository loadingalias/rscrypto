# Worst benchmark losses

## Outcome

Close the ten largest fastest-external performance gaps from benchmark run
[#34874736834](https://github.com/loadingalias/rscrypto/actions/runs/34874736834).
Do not treat these as ten independent optimizations. Establish the shared causes
in the s390x and RISC-V P-256/ECDSA production paths, then fix the smallest
number of production boundaries that closes the measured gaps.

The task is complete when every row either reaches the repository's 0.95x tie
boundary against the fastest equivalent external implementation or is removed
from fastest-equivalent claims with a documented, reviewed reason that the
security or lifecycle contracts differ materially.

## Baseline

- Source: `ae6f54afedaa652858fd2bcbd8f56f339e663a4f` on `main`.
- Measurement date: 2026-09-14.
- Target toolchain: `rustc 1.99.0-nightly (3d6c19bb9 2026-08-11)`.
- Profile: repository `bench` profile.
- Criterion settings: 20 samples, 100 ms warm-up, 400 ms measurement time,
  10,000 resamples, 95% confidence, and 1% noise threshold.
- Artifacts: `bench-s390x-linux-34874736834-1` and
  `bench-riscv64-linux-34874736834-1` from the linked workflow run.
- Ratio: `fastest_external_time / rscrypto_time`; lower is worse.
- Estimates: Criterion slope where available, otherwise the mean. Parentheses
  show the retained 95% confidence interval.

The s390x artifact identifies a four-CPU IBM/S390 machine type 8562. The
RISC-V artifact records four CPUs and the target architecture but no CPU model.
Preserve a complete native machine identity when reproducing these results.

## Ten worst exact cases

| Rank | Target | Exact rscrypto case | Rscrypto | Fastest external | External | Ratio |
| ---: | --- | --- | ---: | --- | ---: | ---: |
| 1 | s390x Linux | `p256-ecdh/public-key/rscrypto-selected` | 9.627 ms (9.139–10.131 ms) | `crrl-pure-rust` | 72.213 us (71.869–72.614 us) | 0.00750x |
| 2 | s390x Linux | `p256-ecdh/agreement/rscrypto-selected` | 19.075 ms (18.896–19.267 ms) | `crrl-pure-rust` | 176.205 us (173.562–180.180 us) | 0.00924x |
| 3 | RISC-V Linux | `p256-ecdh/public-key/rscrypto-selected` | 13.584 ms (13.568–13.602 ms) | `crrl-pure-rust` | 134.771 us (134.378–135.165 us) | 0.00992x |
| 4 | RISC-V Linux | `p256-ecdh/agreement/rscrypto-selected` | 30.860 ms (30.836–30.886 ms) | `crrl-pure-rust` | 332.052 us (331.697–332.584 us) | 0.01076x |
| 5 | RISC-V Linux | `ecdsa-p384/public-key/rscrypto-blinded` | 159.707 ms (159.108–160.723 ms) | `rustcrypto-p384` | 2.460 ms (2.452–2.470 ms) | 0.01540x |
| 6 | s390x Linux | `ecdsa-p384/public-key/rscrypto-blinded` | 96.934 ms (96.107–97.847 ms) | `rustcrypto-p384` | 1.512 ms (1.498–1.527 ms) | 0.01560x |
| 7 | s390x Linux | `p256-ecdh/parse/rscrypto` | 22.726 us (22.614–22.840 us) | `crrl-pure-rust` | 358.331 ns (339.375–380.387 ns) | 0.01577x |
| 8 | RISC-V Linux | `ecdsa-p256/public-key/rscrypto-blinded` | 36.110 ms (35.958–36.285 ms) | `rustcrypto-p256` | 591.334 us (590.168–592.512 us) | 0.01638x |
| 9 | RISC-V Linux | `ecdsa-p384/sign/rscrypto-deterministic/0` | 50.215 ms (50.161–50.275 ms) | `aws-lc-rs` | 854.838 us (853.000–856.784 us) | 0.01702x |
| 10 | RISC-V Linux | `ecdsa-p384/sign/rscrypto-deterministic/32` | 50.220 ms (50.188–50.258 ms) | `aws-lc-rs` | 858.742 us (854.988–862.990 us) | 0.01710x |

These are 58x–133x gaps. Their confidence intervals are narrow relative to the
observed differences, so the short Criterion window does not plausibly explain
the ranking. It does not identify the cause.

## Work

### 1. Reproduce before changing code

- [ ] Reproduce the exact cases on the same physical CPU families, toolchain,
  features, profile, and production dispatch paths.
- [ ] Retain complete host identity, source state, benchmark plan, estimates,
  samples, and selected backend diagnostics.
- [ ] Confirm the fastest-external match performs equivalent work. Pay special
  attention to ECDSA blinding, key preparation, validation, and destruction.
- [ ] Run the focused baseline through the repository front door:

  ```text
  just bench p256-ecdh 'filter=^p256-ecdh/(public-key|agreement|parse)/'
  just bench ecdsa-p256 ecdsa-p384 'filter=^ecdsa-p(256|384)/(public-key|sign)/'
  ```

### 2. Locate shared causes

- [ ] Profile `p256-ecdh/public-key/rscrypto-selected` and
  `p256-ecdh/agreement/rscrypto-selected` independently on both targets.
- [ ] Profile P-256 parsing separately. Determine whether inversion, field
  representation, validation, or target code generation owns the s390x and
  RISC-V gap.
- [ ] Profile P-256/P-384 public derivation and P-384 signing. Attribute time to
  field arithmetic, scalar multiplication, blinding, entropy, encoding, and
  cleanup without moving caller-paid work outside timing.
- [ ] Inspect generated code and structural counters after profiling identifies
  the hot production symbols. Do not infer the cause from portable source alone.
- [ ] Record whether one arithmetic or code-generation defect explains multiple
  rows before proposing target-specific backends.

Initial profiling front doors:

```text
just profile p256-ecdh 'p256-ecdh/public-key/rscrypto-selected' 10
just profile p256-ecdh 'p256-ecdh/agreement/rscrypto-selected' 10
just profile ecdsa-p384 'ecdsa-p384/sign/rscrypto-deterministic/0' 10
```

### 3. Fix production paths

- [ ] Prefer target-shaped safe Rust and better data layout before intrinsics or
  assembly. Keep portable Rust as the semantic authority.
- [ ] Change only production-reachable code. Do not add copied algorithms,
  benchmark-only implementations, hidden dispatch state, or weaker workloads.
- [ ] Preserve deterministic outputs, blinded-operation semantics, failure
  opacity, secret cleanup, constant-time boundaries, target fallback, and public
  API behavior.
- [ ] Route any unsafe, intrinsic, SIMD, assembly, or target-feature work through
  the `unsafe` skill and renew the required differential, ABI, code-generation,
  constant-time, and zeroization evidence.

### 4. Prove closure

- [ ] Rerun the exact baseline cases with longer measurement windows on the same
  machines. Report absolute estimates, confidence intervals, ratios, and
  selected backends.
- [ ] Run portable-versus-optimized differentials and independent vectors over
  representative lengths, encodings, state transitions, and failure cases.
- [ ] Run the target-native correctness, constant-time, cleanup, and generated-
  code checks required by the changed boundary.
- [ ] Rerun the full benchmark matrix and update
  `benchmark_results/OVERVIEW.md`; do not transfer results between CPU families.
- [ ] Delete a row from this task only after its evidence reaches at least 0.95x
  or a reviewed non-equivalence decision removes it from performance claims.

## Non-goals

- Hiding losses by averaging them with faster platforms or larger inputs.
- Weakening blinding, validation, cleanup, error opacity, or constant-time work.
- Selecting an external implementation as rscrypto's production algorithm.
- Claiming a target win from cross-compilation, source inspection, or structural
  counters without native wall-clock evidence.
