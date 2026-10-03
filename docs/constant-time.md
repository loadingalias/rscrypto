# Constant-time claims

In this project, constant time means that secret values do not change control flow,
memory addresses, or the operands of variable-latency instructions within a defined operation.
Public input lengths, algorithm parameters, target features, allocation, scheduling,
and external entropy sources can still change the time.

[`ct.toml`](../ct.toml) is the inventory of operations.
An operation has a claim only for the targets, features, compiler, linked binary,
and evidence that `ct.toml` names.
Code that `ct.toml` does not list has no constant-time claim.

## Evidence model

Release evidence combines:

- source review for secret-dependent branches, indexing, comparisons,
  and variable-latency instructions;
- differential tests that bind accelerated paths to portable Rust behavior;
- inspection of optimized linked binaries for the retained entry points;
- BINSEC proofs for declared fixed-shape kernels;
- DudeCT timing tests for declared end-to-end cases.

Source code that looks branchless is not proof.
Compiler lowering, inlining, target features, and linking can change machine behavior.
For example, a checked `strict_*` operation on a secret-derived value tests that value.
RV64GC has no conditional move, so the test can compile to a secret-dependent branch.
Bounded secret arithmetic therefore uses `wrapping_*`, with the bound written at the definition,
as the Curve25519 field and scalar code does.
Evidence for the release harness does not cover a downstream binary that is compiled differently.

Build and validate the local evidence artifacts:

```bash
just ct-artifacts
just ct-validate
```

`ct-validate` rejects generated artifacts that are missing or stale.
`just ct-full` builds the artifacts, runs the available timing checks, and writes the reports.
A claim for a target needs the evidence that `ct.toml` requires for that target.
A local host cannot stand in for another target.

For strict manifest and artifact coverage, build the artifacts, then run `just ct-validate --strict-coverage`.
This checks compiler output and coverage only.
Timing and formal evidence need their own target runs.

The bounded PBKDF2 proof hooks use `verify_primitive` with one iteration and a fixed salt.
They must reach key derivation and comparison.
The application password-policy verifier rejects these weak parameters before comparison,
so earlier proofs of the policy-rejecting hooks do not prove PBKDF2 verification coverage.
Regression tests for the internal hooks check both successful and failed verification.
Application password policies are separate and do not change.

## Public decisions and exclusions

Ordinary equality is allowed for public values, such as nonces, encoded public keys,
ciphertext lengths, and signature inputs.
Where a comparison must stay opaque until explicit declassification, secret owners return a `CtDecision`.

These operations are outside the constant-time claims on purpose:

- unkeyed hashes, checksums, and XOFs on public data;
- signature verification and public-key parsing;
- RSA prime generation;
- Argon2d, the data-dependent phase of Argon2id, and scrypt memory access;
- caller callbacks, operating-system allocation, thread scheduling, and entropy acquisition;
- external implementations of public traits;
- diagnostic APIs, which expose evidence values on purpose.

Authentication failures stay opaque, also when their inputs are public.

### P-256 ECDH

Scalar sampling and canonical SEC1 validation are public prelude operations.
They are outside the private-arithmetic claim.
When a valid scalar and peer point exist, public derivation and agreement use: fixed loop bounds,
full-table scans for secret digits, masked selection of exceptional points,
and no secret-dependent addresses.

`ct.toml` sets the required linked-binary and target evidence.
A source-level fixed-work design is not a release claim by itself.
The operation entry separates the portable implementation from the selected assembly for Apple
and Linux AArch64, Linux x86-64, and Windows x86-64.
The [September 2026 P-256 ECDH snapshot](../benchmark_results/OVERVIEW.md#p-256-ecdh-development-snapshot) keeps the historical Graviton3, Graviton4,
and Intel Granite Rapids results.
Those development bundles keep binaries and raw timing samples,
but later source changes need new evidence for the exact candidate.

### P-384 ECDH

P-384 ECDH has the same public prelude boundary.

- Public derivation uses a fixed 48-row comb with full 256-entry table scans.
- Agreement uses 77 fixed signed radix-32 windows with full 16-entry table scans,
  masked conditional negation, and masked handling of infinity and equal operands.
- AArch64 replaces the field arithmetic and the comb scan with inline assembly
  that keeps the same fixed schedule.
  It replaces Fermat inversion with a fixed 1116-step Bernstein–Yang divstep inversion.
- x86-64 uses the same divstep inversion and the inline-assembly field kernels.
  Runtime detection selects BMI2/ADX multiplication and fused point doubling.
  Window additions use the portable formula over those kernels.
- Other targets run the portable implementation.

`ct.toml` declares the DudeCT cases and the bounded BINSEC selector kernels.
The full Constant-Time workflow measures the DudeCT cases on every native platform.

### DudeCT decision

Each required DudeCT case is decided in two steps on the same prepared binary.

1. A screening measurement runs with the case's sample budget.
2. If its |t| reaches `review_fraction` of the threshold, or exceeds the threshold,
   a confirmation measurement runs with `sample_factor` times the samples.
   The confirmation decides the case.

`[dudect_confirmation]` in `ct.toml` sets both values.
Neither step changes a threshold.
A stationary timing difference grows |t| by about the square root of the sample ratio,
so four times the samples roughly doubles the |t| of a real leak.
Host noise does not grow that way, so a one-off outlier does not fail the gate.
The near-threshold rule also measures borderline passes again, so a real difference that screening
underestimated can fail instead of passing on one draw.
The report keeps the screening result under `screening`.
A screening failure that the confirmation does not reproduce is reported as a `dudect_unconfirmed` diagnostic.

### ECDSA

Recent ECDSA hardening keeps masked point selection on AArch64 and Windows,
masked selection in portable P-256, and table traversal with fixed bounds.
RISC-V generator-table loads stay unconditional under LLVM optimization.
These changes keep signature behavior the same.
They do not establish a timing claim for a release or a downstream build by themselves.

### ML-DSA

ML-DSA has a separate required boundary, `signature.mldsa.secret_kernels`.
Its timing cases, listed in `ct.toml`, cover transforms, products, accumulation, norm-rejection position,
rounding, secret samplers, and valid-key preparation for ML-DSA-44/65/87.

- Polynomial classes use canonical coefficients.
- Preparation compares one fixed key with a pool of 32 independently seeded valid keys.
- Key generation and input selection happen outside timing.
- Public matrix expansion and redundant-field validation are outside the preparation boundary.
- Retained roots support closure review of the linked binary, including direct tail transfers.
- The bounded BINSEC root analyzes the production portable Montgomery leaf over all operand pairs
  below 2q.
  It does not prove full transforms, accelerated arithmetic, samplers, preparation,
  or hardware instruction latency.

These are requirements for candidate qualification.
They do not state that every target has passed.
Standard signing retries until the first accepted candidate, so it stays variable-time.
Whole ML-DSA signing is best effort, with no strict constant-time claim.

## Native evidence by platform

Windows x86-64 needs the same evidence as the other selected native lanes:
the native timing campaign, the compiler API inventory, artifact validation,
and the cleanup sentinel.
BINSEC proofs for Windows (PE) binaries are not part of the release gate.
"Required" means that the evidence must be collected.
It does not mean that a candidate has passed.
Each release still needs passing native results for the exact source on every selected architecture.
Cross-compilation is not timing proof, and neither is a result from a different microarchitecture.

For RISC-V, POWER, and IBM Z, CI prepares the CT artifacts on x86-64,
then measures the transferred executables on matching native hardware.
The preparation bundle binds the exact source, compiler, binary, disassembly,
and validation evidence.
The timing reports keep both host identities.
Preparation alone gives no timing result.
Transferred execution keeps the same required cases and acceptance thresholds.
See [the transfer workflow](../scripts/README.md#constant-time-evidence).

The [Constant-Time workflow](../.github/workflows/ct.yml) runs on manual dispatch and in release qualification.
It does not run on each pull request or push.
Its AWS measurement profiles use fixed On-Demand instances.
They are sized separately from the benchmark profiles in [`.github/runs-on.yml`](../.github/runs-on.yml).
Preparation uses Spot instances, and its completion gives no timing evidence.
Diagnostic replay is separate from the full release campaign.

## Arm data-independent timing

On AArch64 cores with `FEAT_DIT`, the architecture guarantees data-independent timing
for its listed instructions only while `PSTATE.DIT` is set.
On Linux and macOS, it is clear by default.

`rscrypto` sets DIT for the full duration of every asymmetric, post-quantum, and password operation,
then restores the caller's state: X25519, Ed25519, ECDSA, P-256 and P-384 ECDH, ML-KEM, ML-DSA,
RSA private operations, Argon2, scrypt, and PBKDF2.
One toggle costs about 30 ns on Apple Silicon,
which is less than 1% of these operations at realistic parameters.

Short symmetric operations do not toggle DIT on each call: MACs and tag verification, AEADs,
keyed hashes, HKDF, and fixed-size `ct_eq`.
For them, a toggle would cost between 14% and 15×.
To use DIT there, wrap the work, or a whole worker loop, in `rscrypto::traits::ct::with_data_independent_timing`.
A unit test checks that every covered operation enters the guard.
DIT is hardening.
It does not replace the evidence in `ct.toml`.
Cores without `FEAT_DIT` run unchanged.

## Power and frequency channels

The constant-time claim covers control flow, memory addresses, and variable-latency operands.
Data-dependent power is outside it, as [`THREAT_MODEL.md`](../THREAT_MODEL.md) states for physical side channels.
Power can still affect timing.
On x86, frequency scaling reacts within milliseconds (Hertzbleed).
POWER9 and POWER10 slow the clock within nanoseconds when current causes a voltage droop.

The POWER CI hosts show this effect.
On some POWER10 hosts, DudeCT probes find vector Montgomery products faster
when every coefficient repeats one value than when the coefficients are random.
Random operands with the same distribution cannot be told apart,
and scalar Montgomery products show no difference.
This is a power effect on those hosts, not a secret-dependent instruction choice.
The `mldsa_probe_*` diagnostic cases in `ct.toml` reproduce it.

ML-DSA secret polynomials have small, heavily repeated coefficients.
On the POWER vector backend, their forward NTT is therefore blinded:
key generation and key preparation transform `s + r`, then subtract `NTT(r)`, where `r` comes from the secret seed `K`.
The outputs do not change.
Other targets transform directly.

The ML-DSA gate measures the operand distribution that production uses.
POWER sends four ML-DSA kernels to its vector unit: forward NTT, inverse NTT, product,
and accumulation.

- The `mldsa_*_dense_fixed_vs_random` cases compare one fixed dense polynomial with new ones, at 200,000 samples.
  They are required on every target,
  because NTT-domain secrets and the signing mask are dense residues.
- The all-zero cases of those four kernels are required on every target except POWER.
  On POWER, `diagnostic_targets` in `ct.toml` marks them as non-gating diagnostics and gives the reason:
  production never gives those kernels zero or small repeated secrets on POWER.
  Products, accumulations, and inverse NTTs take NTT-domain operands.
  Small secrets reach the vector NTT only blinded, and the required `mldsa_prepare*` cases measure that.
- The zero scalar Montgomery and scalar inverse NTT cases are required on every target.

ML-KEM decapsulation follows FIPS 203.
It decompresses the ciphertext polynomial before its NTT,
so an attacker cannot choose the sparse NTT inputs that frequency attacks on the inverse NTT need
(Yu et al., CHES 2024).
Prepared decapsulation keys add a second barrier.
Before the inverse NTT, decryption adds a secret dense polynomial,
derived once from the implicit-rejection secret `z`.
The fused final pass of the inverse NTT removes its transform.
The message and the work for each call do not change.
One-shot decapsulation does not mask yet.

See [`secret-ownership.md`](secret-ownership.md) for comparison capabilities,
and [`secret-lifecycle.md`](secret-lifecycle.md) for cleanup evidence.
