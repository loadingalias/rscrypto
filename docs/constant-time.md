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

Authentication failures remain opaque even when their inputs are public.
See [`secret-ownership.md`](secret-ownership.md) for comparison capabilities and [`secret-lifecycle.md`](secret-lifecycle.md)
for cleanup evidence.
