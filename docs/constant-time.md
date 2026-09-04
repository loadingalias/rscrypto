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
builds them, runs available timing checks, and emits reports. A target-specific
claim requires the evidence required by `ct.toml`; a local host cannot
stand in for another target.

`just ct-structural` builds the bounded x86-64 release harness, inspects its
generated code, and validates strict manifest and artifact coverage. It is a
compiler-regression check, not timing or formal evidence. Required target runs
must be performed and retained independently.

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

P-256 ECDH scalar sampling and canonical SEC1 validation are public prelude
operations outside the private-arithmetic claim. Once a valid scalar and peer
point exist, public derivation and agreement use fixed loop bounds, full-table
secret-digit scans, masked exceptional-point selection, and no
secret-dependent addresses. `ct.toml` scopes the required linked-binary and
target evidence; a source-level fixed-work design is not itself a release
claim. Its operation entry distinguishes the portable implementation from the
selected Apple/Linux AArch64, Linux x86-64, and Windows x86-64 assembly.
Physical Graviton3, Graviton4, and Intel Granite Rapids development runs cover
both Linux operation-level DudeCT cases and preserve the measured binary,
disassembly, symbols, linker command, and raw samples. Those bundles measure
intermediate Phase 4 candidates and do not replace exact-candidate evidence.
The retained G3 maxima are 1.12000 for public
derivation and 2.59291 for agreement; the Intel Granite Rapids maxima are
1.76752 and 1.33030, respectively, against the threshold of 10. Windows
x86-64 has direct native differential and performance evidence, but the final
batch-parser source identity does not yet have a complete retained timing
bundle. Exact-source Windows P-256 operation-level timing and optimized cleanup
artifacts remain unavailable, and dedicated physical timing is pending. Other
native rows likewise await their own target-specific evidence. Neither
cross-compilation nor a different microarchitecture is treated as timing proof.

Authentication failures remain opaque even when their inputs are public. See
[`secret-ownership.md`](secret-ownership.md) for comparison capabilities and
[`secret-lifecycle.md`](secret-lifecycle.md) for cleanup evidence.
