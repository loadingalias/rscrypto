# Constant-time claims

Constant time means secret values do not change control flow, memory addresses,
or variable-latency operands within a defined operation.
Public input lengths, algorithm parameters, target features, allocation, scheduling,
and external entropy sources may still affect time.

`ct.toml` is the authoritative operation inventory.
An operation is claimed only for the targets, features, compiler, linked binary,
and evidence named there.
Unlisted code is not claimed constant time.

## Evidence model

Release evidence combines:

- Source review of secret-dependent branches, indexing, comparisons, and
  variable-latency instructions.
- Differential tests that bind accelerated paths to portable Rust semantics.
- Optimized linked-binary inspection for retained entry points.
- BINSEC proofs for declared fixed-shape kernels.
- DudeCT timing tests for declared end-to-end cases.

Source that looks branchless is not proof.
Compiler lowering, inlining, target features, and linking can change machine behavior.
For example, a checked `strict_*` operation on a secret-derived value tests that value:
RV64GC has no conditional move, so the test can compile to a secret-dependent branch.
Bounded secret arithmetic uses `wrapping_*` with the bound written at the definition,
as the Curve25519 field and scalar code do.
Evidence for the release harness does not automatically cover a downstream binary compiled
differently.

Build and validate the local evidence artifacts with:

```bash
just ct-artifacts
just ct-validate
```

`ct-validate` rejects missing or stale generated artifacts.
`just ct-full` builds them, runs available timing checks, and emits reports.
A target-specific claim requires the evidence required by `ct.toml`;
a local host cannot stand in for another target.

For strict manifest and artifact coverage, run `just ct-validate --strict-coverage` after building the artifacts.
This checks compiler output and coverage;
timing and formal evidence require their respective target runs.

The bounded PBKDF2 proof hooks use `verify_primitive` with one iteration and a fixed salt.
They must reach key derivation and comparison; the application password-policy verifier rejects these
deliberately weak parameters before comparison. Earlier proofs of the policy-rejecting hooks do not establish
PBKDF2 verification coverage. Internal-hook regression tests check both successful and failed verification.
Application password policies remain separate and unchanged.

## Public decisions and exclusions

Ordinary equality is permitted for public values such as nonces, encoded public keys,
ciphertext lengths, and signature inputs.
Secret owners expose `CtDecision` where comparison must remain opaque until explicit declassification.

These operations are intentionally outside blanket constant-time claims:

- Unkeyed hashes, checksums, and XOFs processing public data.
- Signature verification and public-key parsing.
- RSA prime generation.
- Argon2d, the data-dependent phase of Argon2id, and scrypt memory access.
- Caller callbacks, OS allocation, thread scheduling, and entropy acquisition.
- External implementations of public traits.
- Diagnostic APIs, which deliberately expose evidence values.

P-256 ECDH scalar sampling and canonical SEC1 validation are public prelude operations outside the
private-arithmetic claim.
Once a valid scalar and peer point exist, public derivation and agreement use fixed loop bounds,
full-table secret-digit scans, masked exceptional-point selection,
and no secret-dependent addresses.
`ct.toml` scopes the required linked-binary and target evidence;
a source-level fixed-work design is not itself a release claim.
Its operation entry distinguishes the portable implementation from the selected Apple/Linux AArch64,
Linux x86-64, and Windows x86-64 assembly.
The [September 2026 P-256 ECDH snapshot](../benchmark_results/OVERVIEW.md#p-256-ecdh-development-snapshot) retains the historical Graviton3, Graviton4,
and Intel Granite Rapids results.
Those development bundles preserve binaries and raw timing samples,
but later source changes require new exact-candidate evidence.

P-384 ECDH has the same public prelude boundary.
Public derivation uses a fixed 48-row comb with full 256-entry table scans;
agreement uses 77 fixed signed radix-32 windows with full 16-entry table scans,
masked conditional negation, and masked handling of infinity and equal operands.
AArch64 builds replace the field arithmetic and the comb scan with inline assembly
that keeps the same fixed schedule, and replace the Fermat inversion with a fixed
1116-step Bernstein–Yang divstep inversion. x86-64 builds use the same divstep inversion
and inline-assembly field kernels, with BMI2/ADX multiplication and fused point doubling selected by runtime
detection; window additions use the portable formula over those kernels.
Other targets run the portable implementation.
`ct.toml` declares its DudeCT cases and bounded BINSEC selector kernels;
no native timing artifact has been recorded for P-384 ECDH yet.

Recent ECDSA hardening preserves masked point selection on AArch64 and Windows,
masked selection in portable P-256, and fixed-bound table traversal.
RISC-V generator-table loads remain unconditional under LLVM optimization.
These implementation changes preserve signature semantics;
they do not by themselves establish a timing claim for a release or a downstream build.

Windows x86-64 now requires the same native timing campaign, compiler API inventory,
artifact validation, and cleanup sentinel as the other selected native lanes.
Its BINSEC proof policy remains unsupported.
Required means the evidence must be collected, not that a candidate has passed:
each release still needs successful exact-source native results across all selected architectures.
Neither cross-compilation nor a different microarchitecture is timing proof.

RISC-V, POWER, and IBM Z CI prepare CT artifacts on x86-64
and measure the transferred executables on the matching native hardware.
The preparation bundle binds the exact source, compiler, binary, disassembly,
and validation evidence.
The timing reports retain both host identities.
Preparation alone supplies no timing result,
and transferred execution retains the same required cases and acceptance thresholds.
See [the transfer workflow](../scripts/README.md#constant-time-evidence).

The [Constant-Time workflow](../.github/workflows/ct.yml) runs through manual dispatch or release qualification,
not on each pull request or push.
Its AWS measurement profiles use fixed On-Demand instances
and are sized separately from benchmark profiles in [`.github/runs-on.yml`](../.github/runs-on.yml).
Preparation uses Spot instances; its completion supplies no timing evidence.
Diagnostic replay remains separate from the full release campaign.

ML-DSA has a separate required `signature.mldsa.secret_kernels` boundary.
Its 19 timing cases cover transforms, products, accumulation, norm rejection
position, rounding, secret samplers, and valid-key preparation for ML-DSA-44/65/87.
Polynomial classes use canonical coefficients; preparation uses one fixed key
against a pool of 32 independently seeded valid keys. Key generation and input
selection happen outside timing. Public matrix expansion and redundant-field
validation are outside the preparation boundary.
Retained roots support linked-binary closure review, including direct tail transfers.
The bounded BINSEC root analyzes the production portable Montgomery leaf over
all operand pairs below 2q; it does not prove full transforms, accelerated
arithmetic, samplers, preparation, or hardware instruction latency.
These are requirements for candidate qualification, not a statement that all
targets have passed. Standard first-accepted signing retries remain variable-time;
whole ML-DSA signing is best-effort and is not claimed strict constant time.

## Arm data-independent timing

On AArch64 cores with `FEAT_DIT`, the architecture guarantees data-independent
timing for its listed instructions only while `PSTATE.DIT` is set; the Linux and
macOS default is clear. rscrypto sets it for the duration of every asymmetric,
post-quantum, and password operation, and restores the caller's state afterwards:
X25519, Ed25519, ECDSA, P-256 and P-384 ECDH, ML-KEM, ML-DSA, RSA private
operations, Argon2, scrypt, and PBKDF2. One toggle costs about 30 ns on Apple
Silicon, well under 1% of these operations at realistic parameters.

Short symmetric operations do not toggle it per call: MACs and tag verification,
AEADs, keyed hashes, HKDF, and fixed-size `ct_eq`. For them a toggle would cost
14% to 15×. Callers that want DIT there wrap the work, or a whole worker loop, in
`rscrypto::traits::ct::with_data_independent_timing`. A unit test checks that every
covered operation enters the guard. DIT is hardening; it does not replace the
evidence in `ct.toml`, and cores without `FEAT_DIT` run unchanged.

## Power and frequency channels

The constant-time claim covers control flow, memory addresses, and
variable-latency operands. Data-dependent power is outside it, as
[`THREAT_MODEL.md`](../THREAT_MODEL.md) states for physical side channels.
Power can still reach timing: x86 frequency scaling reacts within milliseconds
(Hertzbleed), and POWER9 and POWER10 slow the clock within nanoseconds when
current causes a voltage droop.

The POWER CI hosts show that effect. On some POWER10 hosts, DudeCT probes find
vector Montgomery products faster when every coefficient repeats a value than
when the coefficients are random; random operands of equal distribution are
indistinguishable, and scalar Montgomery products show no difference. This
is a power effect on those hosts, not a secret-dependent instruction choice.
The `mldsa_probe_*` diagnostic cases in `ct.toml` reproduce it.

ML-DSA secret polynomials have small, heavily repeated coefficients, so on
POWER's vector backend their forward NTT is blinded: key generation and key
preparation transform `s + r` and subtract `NTT(r)`, where `r` is derived from
the secret seed `K`. Outputs are unchanged; other targets transform directly.

The ML-DSA gate measures the operand distribution production uses.
POWER dispatches four ML-DSA kernels to its vector unit: forward NTT, inverse
NTT, product, and accumulation. The `mldsa_*_dense_fixed_vs_random` cases
compare one fixed dense polynomial with fresh ones at 200,000 samples and are
required on every target: NTT-domain secrets and the signing mask are dense
residues. The all-zero cases of those four kernels stay required everywhere
except POWER, where `diagnostic_targets` in `ct.toml` records them as
non-gating diagnostics with the reason. On POWER, production never gives those
kernels zero or small repeated secrets. Products, accumulations, and inverse
NTTs take NTT-domain operands, and small secrets reach the vector NTT only
blinded, which the required `mldsa_prepare*` cases measure. The zero scalar
Montgomery and scalar inverse NTT cases stay required on every target.

ML-KEM decapsulation follows FIPS 203: the ciphertext polynomial is
decompressed before its NTT, so an attacker cannot choose the sparse NTT inputs
that frequency attacks on the inverse NTT require (Yu et al., CHES 2024).
Prepared decapsulation keys add a second barrier: decryption adds a secret dense
polynomial, derived once from the implicit-rejection secret `z`, before the
inverse NTT, and the inverse NTT's fused final pass removes its transform. The
message is unchanged and per-call work is unchanged. One-shot decapsulation does
not mask yet.

Authentication failures remain opaque even when their inputs are public.
See [`secret-ownership.md`](secret-ownership.md) for comparison capabilities and [`secret-lifecycle.md`](secret-lifecycle.md)
for cleanup evidence.
