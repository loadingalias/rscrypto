# Constant-time claims

Constant time means secret values do not change control flow, memory addresses,
or variable-latency operands within a defined operation. Public input lengths,
algorithm parameters, target features, allocation, scheduling, and external
entropy sources may still affect time.

`ct.toml` is the authoritative operation inventory. An operation is claimed
only for the targets, features, compiler, linked binary, and evidence named
there. Unlisted code is not claimed constant time.

## Evidence model

Release evidence combines:

- Source review of secret-dependent branches, indexing, comparisons, and
  variable-latency instructions.
- Differential tests that bind accelerated paths to portable Rust semantics.
- Optimized linked-binary inspection for retained entry points.
- BINSEC proofs for declared fixed-shape kernels.
- DudeCT timing tests for declared end-to-end cases.

Source that looks branchless is not proof. Compiler lowering, inlining, target
features, and linking can change machine behavior. Evidence for the release
harness does not automatically cover a downstream binary compiled differently.

Build and validate the local evidence artifacts with:

```sh
just ct-artifacts
just ct-validate
```

`ct-validate` rejects missing or stale generated artifacts. `just ct-full`
builds them, runs available timing checks, and emits reports. A release claim
requires the target-specific lanes required by `ct.toml`; a local host cannot
stand in for another target.

Affected pull requests run `just ct-structural` when Cargo Rail selects
`assurance.ct`. That bounded x86-64 gate builds the optimized release harness,
inspects its generated code, and validates strict manifest/artifact coverage.
It is an early compiler-regression gate, not timing or formal evidence.
Scheduled and release Qualification consume the same immutable plan and retain
the full `ct.yaml` physical DudeCT and BINSEC matrix.

## Public decisions and exclusions

Ordinary equality is permitted for public values such as nonces, encoded public
keys, ciphertext lengths, and signature inputs. Secret owners expose
`CtDecision` where comparison must remain opaque until explicit
declassification.

These operations are intentionally outside blanket constant-time claims:

- Unkeyed hashes, checksums, and XOFs processing public data.
- Signature verification and public-key parsing.
- RSA prime generation.
- Argon2d, the data-dependent phase of Argon2id, and scrypt memory access.
- Caller callbacks, OS allocation, thread scheduling, and entropy acquisition.
- External implementations of public traits.
- Diagnostic APIs, which deliberately expose evidence values.

Authentication failures remain opaque even when their inputs are public. See
[`secret-ownership.md`](secret-ownership.md) for comparison capabilities and
[`secret-lifecycle.md`](secret-lifecycle.md) for cleanup evidence.
