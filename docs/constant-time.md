# Constant-Time Policy

`rscrypto` constant-time claims are evidence claims, not marketing claims.

The project goal is to prove, for each release, that every primitive that needs
constant-time behavior has evidence for the exact configurations claimed. A
configuration is exact: crate version, commit, Rust compiler, LLVM codegen,
target triple, target CPU/features, linker, profile, panic mode, enabled
features, and dependency lockfile.

Unlisted configurations are not covered by a constant-time release claim.

The machine-readable source of truth is [`ct.toml`](../ct.toml). Documentation
may explain the policy, but release tooling must read the manifest.

## Threat Model

The constant-time policy targets software-observable timing leakage from secret
data through architectural control flow, memory access patterns, variable-time
operations, dispatch, allocation, panic/failure shape, and formatting/logging
paths.

The policy also treats cache-visible secret-dependent memory access as a timing
leak. A primitive that avoids early returns but performs secret-indexed loads is
not constant-time under this policy.

This policy does not claim resistance to physical side channels such as power,
electromagnetic leakage, acoustic leakage, fault injection, rowhammer, or
platform compromise. Those require separate hardware and operational evidence.

Speculation is handled by removing secret-dependent branches and addresses from
claimed code paths, not by claiming general Spectre-class resistance for the
whole process. If a primitive needs additional speculation barriers on a
specific target, that requirement must appear in the CT manifest and reviewed
artifact.

## Current Backend Boundary

Phase 1 claims are LLVM-only.

For now, `rscrypto` makes no constant-time claim for Cranelift, GCC codegen, or
other Rust codegen backends. Those backends can become claimed configurations
only after the same policy, manifest, artifact, and leakage checks run against
their generated binaries.

The policy is intentionally backend-explicit because constant-time behavior is
a binary property. Source that is written in a constant-time style is not enough
if the compiler, linker, target features, or build profile change the executed
machine code.

## Current Validation Engine

The release gate is `just ct-check`. It is intentionally the same hard gate as
`just ct-full`: build CT artifacts, validate the manifest/artifacts, run every
required DudeCT case, and run BINSEC where the target policy requires it.
Diagnostic DudeCT cases are available for investigation, but they do not satisfy
release coverage and do not block the required gate unless promoted in
[`ct.toml`](../ct.toml).

The current gates are:

| Gate | What it checks today | Applies to |
|---|---|---|
| Artifact and heuristic review | Build provenance, LLVM IR, assembly, object disassembly, symbol maps, reviewed hashes, and suspicious instruction/control-flow patterns. | Native LLVM targets in the CT matrix. |
| DudeCT timing evidence | Empirical timing tests for every required manifest-declared CT-critical primitive case. A failure or missing required case fails `ct-check`; diagnostic cases are reported separately. | Native host-executable targets. |
| BINSEC binary evidence | Binary-level symbolic checks for manifest-declared CT leaf kernels. A required non-`secure` kernel fails `ct-check`. | Linux ELF targets whose ISA/object path is supported by this workflow. |

This is an evidence pipeline. It is not a whole-crate formal proof, and it does
not claim that ergonomic public APIs with parsing, allocation, and conversion
logic are symbolically verified end to end. Public APIs are covered by the
manifest, artifacts, heuristics, and DudeCT cases; BINSEC is aimed at small
CT-critical kernels where binary symbolic execution is tractable and meaningful.

## Current Target Scope

Current release evidence is scoped as follows:

| Target class | Current CT gate |
|---|---|
| Linux `x86_64-unknown-linux-gnu` | Artifact/provenance review, LLVM IR/ASM/object heuristics, DudeCT, and BINSEC on the AMD Zen4, AMD Zen5, Intel Ice Lake, and Intel Sapphire Rapids release lanes. |
| Linux `aarch64-unknown-linux-gnu` | Artifact/provenance review, LLVM IR/ASM/object heuristics, DudeCT, and BINSEC on the AWS Graviton3 and Graviton4 release lanes. |
| Linux `riscv64gc-unknown-linux-gnu` | Artifact/provenance review, LLVM IR/ASM/object heuristics, DudeCT, and BINSEC on the RISE RISC-V release lane. |
| Linux `s390x-unknown-linux-gnu` | Artifact/provenance review, LLVM IR/ASM/object heuristics, and DudeCT on the IBM Z release lane. BINSEC is not claimed for this ISA in the current workflow. |
| Linux `powerpc64le-unknown-linux-gnu` | Artifact/provenance review, LLVM IR/ASM/object heuristics, and DudeCT on the IBM Power10 release lane. BINSEC is not claimed for little-endian POWER in the current workflow. |
| macOS `aarch64-apple-darwin` | Local artifact/provenance review, LLVM IR/ASM/object heuristics, and DudeCT through `just ct-full`. BINSEC is not claimed for Mach-O in the current workflow. |
| Linux MUSL, macOS `x86_64`, and Windows MSVC | `ct-intended` only until physical release lanes and object-format evidence are added. |
| `no_std` bare-metal targets | Not release-claimed today. Artifact-only coverage can be added, but timing evidence requires real hardware and BINSEC requires a supported ELF/ISA harness. |
| WASM targets | Not release-claimed today. WASM needs separate bytecode checks and engine-specific timing evidence. |

The native hot paths matter most for the current claim: accelerated ASM, SIMD,
hardware-instruction, and portable fallback code used by CT-critical primitives
must have manifest coverage and must pass the required gates for the target
class above. If a target class has no executable timing environment or no
supported BINSEC object/ISA path, it must remain outside the release CT claim
until that evidence exists.

## Definition

For this crate, "constant-time" means:

```text
For the declared secret inputs of a primitive, the claimed binary must not let
those secrets influence control flow, memory addresses, variable-time
operations, observable failure shape, allocation behavior, formatting/logging,
or dispatch behavior outside the leakage policy.
```

The claim is always relative to declared public inputs. Public inputs may affect
control flow, backend selection, lengths, and public error handling.

## Allowed Leakage

The following may leak unless a primitive manifest entry says otherwise:

- Public input lengths.
- Public algorithm/profile selection.
- Public key, nonce, AAD, salt, label, customization string, and domain
  separator values.
- Public ciphertext, plaintext length, message length, and output length.
- Public parsing shape errors for public material.
- Public CPU feature detection and backend dispatch.
- Public allocation failure or caller-provided buffer-size failure.
- Public success/failure of authentication or verification as a single opaque
  outcome.

Public length may leak.

## Forbidden Leakage

For declared secret inputs, the claimed binary must not contain
secret-dependent:

- Branches or conditional jumps.
- Table indices.
- Memory addresses.
- Division, remainder, or divisor operands.
- Variable loop bounds.
- Early error paths.
- Panic paths.
- Allocation behavior.
- Variable-latency instructions on the claimed architecture.
- Formatting, logging, tracing, or debug paths.
- Trait-object, vtable, function-pointer, or generic dispatch behavior.
- Heap capacity, allocation size, reallocation count, or drop path behavior.
- Parsing rejection timing for secret material.

The rule is strict: if a secret can choose the path, address, operation, or
observable failure shape, the primitive is not constant-time for that claim.

## CT-Critical Surfaces

Tier A surfaces require the strongest evidence:

- Private-key operations: RSA private sign/decrypt, Ed25519 signing, scalar
  multiplication, decapsulation, and any future KEM secret operation.
- Authentication comparisons: MAC/tag verification, keyed-hash verification,
  AEAD open authentication, password-hash verification, and shared-secret
  equality.
- Secret-dependent field, scalar, limb, padding, masking, select, swap,
  reduction, and blinding helpers used by Tier A entrypoints.

Tier B surfaces require evidence, but can be staged after Tier A:

- Symmetric-key encryption/decryption: AES, ChaCha20, Ascon, AEGIS, and
  related key schedules.
- Polynomial authenticators: GHASH and Poly1305.
- KDF and password-hashing internals where password or key material is live.

Tier C surfaces are not constant-time claims unless a manifest entry explicitly
promotes them:

- Hashing public messages.
- Checksums and non-cryptographic hashes.
- Encoding public data.
- Parsing public keys, signatures, ciphertext containers, PHC strings, DER, or
  protocol metadata.
- Canonical serialization/export of secret material, unless a manifest entry
  names a fixed-shape export API as CT-critical.
- Public-key verification as a public operation, except for opaque failure
  shape at the API boundary.

## Target And Platform Rule

Every shipped target triple must be classified in the CT manifest as one of:

- `ct-claimed`: release claims apply only after all required evidence passes.
- `ct-intended`: implementation is intended to be constant-time, but release
  evidence is incomplete.
- `best-effort`: no release CT claim.
- `unsupported`: no CT claim and no planned evidence for the current release.

A target triple is not claimed because it builds. It is claimed only when the
release has evidence for the exact compiler, backend, linker, profile, target
CPU/features, enabled features, and primitive set.

`portable-only` is an audit control, not a proof by itself. It can constrain
runtime dispatch to portable backends, but the portable binary still needs the
same constant-time artifact and leakage evidence for every claimed target.

## Configuration Rule

Each claimed configuration must record:

- Crate version and git commit.
- Rustc version and channel.
- Codegen backend.
- LLVM version for LLVM claims.
- Target triple.
- Target CPU and target features.
- Linker and link arguments.
- Build profile, opt level, LTO setting, codegen units, and panic mode.
- Enabled crate features.
- Dependency lockfile hash.
- Host runner and physical CPU where empirical timing evidence was collected.

Changing any of these creates a new configuration. A new configuration inherits
no constant-time claim until its required evidence passes.

## Evidence Rule

Required evidence is defined per primitive in the CT manifest. The default
shape for a current claimed native target is:

- Stable CT entrypoint harness.
- Build provenance capture.
- LLVM IR and assembly/object artifacts.
- Reviewed artifact hash.
- Automated assembly heuristics with explicit waivers.
- DudeCT-style statistical leakage run on native executable targets.
- Binary-level symbolic check for small high-risk kernels on supported Linux
  ELF targets.
- Miri and unsafe validation for code paths that rely on unsafe Rust.

Diagnostic cases must stay visibly separate from release evidence. They can be
run with `--dudect-gate diagnostic` or `--dudect-gate all`, but a primitive that
requires DudeCT still needs at least one non-diagnostic manifest case, and each
declared variant needs its own evidence unit when variants are listed.

A primitive/configuration pair may be marked `ct-claimed` only when all required
evidence for that manifest entry and target class passes. If physical timing
evidence is not available for a shipped target, the target may be `ct-intended`
or `best-effort`, but not `ct-claimed`.

Statistical checks must be worded correctly:

```text
No leakage detected for this configuration.
```

They must not be worded as proof by themselves.

## Invalidation Rule

A constant-time claim is invalidated until re-reviewed if any of the following
changes:

- CT-critical source code.
- Unsafe block, target-feature wrapper, dispatch table, or assembly wrapper used
  by a CT-critical primitive.
- Compiler version, codegen backend, LLVM version, target CPU/features, linker,
  LTO setting, panic mode, opt level, codegen units, or enabled features.
- Dependency implementation used by a CT-critical primitive.
- Manifest secret/public annotations.
- Assembly/object artifact hash for a reviewed CT-critical symbol.
- Waiver scope, reason, architecture, or reviewer.

CI must fail hard when a reviewed CT-critical artifact changes without renewed
review.

## Release Claim Language

Use this shape:

```text
This release makes constant-time claims only for the configurations listed in
ct-report.json.

Each claim is relative to declared secret inputs, declared public inputs, the
release leakage policy, crate version, rustc version, LLVM backend, target
triple, target CPU/features, linker, build profile, enabled features, and the
evidence recorded for that primitive.

Unlisted configurations are not covered by this release claim.
```

This keeps the project ambitious without pretending that unmeasured binaries are
proven.
