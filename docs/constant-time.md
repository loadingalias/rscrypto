# Constant-Time Policy

`rscrypto` treats constant-time behavior as an evidence-bound claim. Source code
style alone is not enough: the claim depends on the crate version, commit,
compiler, target, CPU features, enabled features, and generated binary.

Unlisted configurations are NOT covered by a constant-time release claim.

The machine-readable source of truth is [`ct.toml`](../ct.toml). The sections
below explain how to read that boundary.

## What Do I Mean By Constant-Time?

For the declared secret inputs of a claimed primitive, the generated binary must
not let those secrets influence:

- Branches or conditional jumps.
- Memory addresses or table indices.
- Variable-time operations.
- Loop bounds.
- Allocation behavior.
- Panic or early-error paths.
- Formatting, logging, or debug output.
- Observable authentication failure shape beyond one opaque success/failure bit.

Public inputs may still affect control flow, lengths, allocation size, backend
selection, and public error handling.

## Threat Model

The policy targets software-observable timing leakage from secret data through
control flow, memory access, dispatch, allocation, failure shape, and generated
machine code.

It does not claim resistance to physical side channels such as power analysis,
electromagnetic leakage, acoustic leakage, fault injection, rowhammer, or
platform compromise. Those need separate hardware and operational evidence.

Speculation is handled by avoiding secret-dependent branches and addresses in
claimed code paths. This is not a blanket Spectre-class guarantee for a whole
process.

## Claimed Surfaces

The highest-sensitivity surfaces are private-key and auth operations:

- MAC/tag verification and constant-time byte equality.
- AEAD authentication and failed-open cleanup.
- X25519 scalar multiplication.
- ML-KEM-512/768/1024 key generation secret noise, encapsulation coins,
  decapsulation secret-key material, implicit-rejection seed, and listed
  arithmetic diagnostics.
- Ed25519 signing and secret-key public derivation.
- ECDSA P-256/P-384 caller-blinded signing.
- RSA private sign/decrypt leaves.
- Password-verification comparisons.
- Secret-dependent field, scalar, limb, padding, select, swap, reduction, and
  blinding helpers used by those entrypoints.

Symmetric encryption, polynomial authenticators, KDF internals, and
password-hashing internals are CT-relevant when key or password material is live.

## Not Blanket Claims

The following are not constant-time claims unless a specific manifest entry says
otherwise:

- Raw hashes over public messages.
- Checksums and non-cryptographic hashes.
- Public-key verification math.
- Public key, signature, ciphertext-container, DER, PHC, and protocol parsing.
- Key generation and OS randomness.
- Serialization and export of secret material.
- Benchmark-only paths.
- Unmeasured targets, compilers, linkers, target features, or crate feature sets.

Public length may leak. Public algorithm/profile selection may leak. A single
opaque authentication success/failure result may leak.

## Target Scope

A target is not claimed because it builds. It is claimed only when the release
has evidence for the exact compiler, codegen backend, linker, target
CPU/features, profile, crate features, dependency lockfile, and primitive set.

Current native release evidence is centered on LLVM-generated binaries for:

| Target class | Current public claim shape |
|---|---|
| Linux `x86_64-unknown-linux-gnu` | Artifact review, generated-code heuristics, empirical timing tests, and binary checks where supported. |
| Linux `aarch64-unknown-linux-gnu` | Artifact review, generated-code heuristics, empirical timing tests, and binary checks where supported. |
| Linux `riscv64gc-unknown-linux-gnu` | Artifact review, generated-code heuristics, and empirical timing tests. |
| Linux `s390x-unknown-linux-gnu` | Artifact review, generated-code heuristics, and empirical timing tests. |
| Linux `powerpc64le-unknown-linux-gnu` | Artifact review, generated-code heuristics, and empirical timing tests. |
| macOS `aarch64-apple-darwin` | Local artifact review, generated-code heuristics, and empirical timing tests. |

For ML-KEM, the s390x claim covers the fixed-work z/Vector arithmetic kernels
present in the release evidence. It does not cover native scalar multiply or
divide substitutions for secret-fed ML-KEM arithmetic, and it does not cover
unreviewed hand-written assembly.

Linux MUSL, macOS `x86_64`, Windows MSVC, bare-metal `no_std`, and WASM builds
may compile and may be intended to follow the same coding rules, but they need
separate target-appropriate evidence before they should be described as covered
by a release CT claim.

`portable-only` constrains runtime dispatch to portable backends. It is useful
for audit-constrained builds, but it is not a proof by itself.

## Evidence

Release evidence is defined per primitive in `ct.toml`. The normal native
evidence set includes:

- Stable harness entrypoints.
- Build and host provenance.
- LLVM IR, assembly, object, and symbol artifacts.
- Automated checks for suspicious generated-code patterns.
- Empirical timing tests on native executable targets.
- ML-KEM DudeCT cases for key generation secret noise, encapsulation coins,
  decapsulation secret keys, implicit rejection, NTT, inverse NTT,
  product-domain conversion, basemul/dot products, and compress/decompress
  arithmetic.
- Binary-level checks for small high-risk kernels on supported Linux ELF/ISA
  paths.
- Miri and unsafe-code validation where the CT path uses unsafe Rust.

Statistical timing checks must be described precisely:

```text
No leakage detected for this configuration.
```

They are evidence, not a formal proof.
