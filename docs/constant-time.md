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

## Candidate Surfaces

`ct.toml` places the following highest-sensitivity surfaces inside the release
evidence gate. This is intent, not a standalone public claim:

- MAC/tag verification and fixed-size equality owned by concrete key, secret,
  tag, and keyed-output types.
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

The release workflow requires native evidence for these LLVM-generated target
classes. A row is covered only when it appears as a passing lane in the
matching release bundle:

| Target class | Required release evidence |
|---|---|
| Linux `x86_64-unknown-linux-gnu` | Artifact review, generated-code heuristics, empirical timing tests, and binary checks where supported. |
| Linux `aarch64-unknown-linux-gnu` | Artifact review, generated-code heuristics, empirical timing tests, and binary checks where supported. |
| Linux `riscv64gc-unknown-linux-gnu` | Artifact review, generated-code heuristics, and empirical timing tests. |
| Linux `s390x-unknown-linux-gnu` | Artifact review, generated-code heuristics, and empirical timing tests. |
| Linux `powerpc64le-unknown-linux-gnu` | Artifact review, generated-code heuristics, and empirical timing tests. |

Release CI exercises multiple x86_64 and AArch64 microarchitectures. The exact
CPU, target features, compiler, linker, tools, and artifact hashes are recorded
per lane rather than generalized to every CPU implementing the same triple.

ECDSA P-256/P-384 signing uses multiplication-free, fixed-work limb arithmetic
on s390x and RISC-V to avoid the variable-latency scalar multiply observed in
earlier native runs. That source and disassembly property is necessary, not
sufficient: those targets are promoted only after their native required
DudeCT cases pass in the matching release evidence.

For ML-KEM, the s390x claim covers the fixed-work z/Vector arithmetic kernels
present in the release evidence. It does not cover native scalar multiply or
divide substitutions for secret-fed ML-KEM arithmetic, and it does not cover
unreviewed hand-written assembly.

Linux MUSL, macOS `x86_64`, Windows MSVC, bare-metal `no_std`, and WASM builds
may compile and may follow the same coding rules, but physical timing evidence
is explicitly deferred. Apple Silicon macOS evidence is local rather than part
of the release bundle. Artifact and heuristic analysis for a deferred target
must never be represented as native physical timing evidence.

`portable-only` constrains runtime dispatch to portable backends. It is useful
for audit-constrained builds, but it is not a proof by itself.

## Evidence

Source inspection and `ct.toml` are not sufficient to establish a release claim. The matching signed GitHub release
must contain all of:

- The attested release manifest, source archive, crate, and `SHA256SUMS` binding the release tag, commit, toolchain,
  and artifacts.
- An attested `rscrypto-X.Y.Z-ct-evidence.tar.gz` built from the same release
  commit.
- `CT-EVIDENCE-BUNDLE.json`, naming the version, full commit, release profile,
  required lane set, toolchain, target CPU/features, and per-lane hashes.
- Raw generated-code artifacts matching each lane's provenance and artifact
  hash ledger.

The release packager fails closed on missing or extra lanes, dirty or
mismatched provenance, incomplete timing cases, required BINSEC kernels that
are absent or non-secure, and any compact/raw artifact hash mismatch. Releases
through `v0.6.4` predate this bundle and carry no release-bound constant-time
claim.

Release evidence is defined per primitive in `ct.toml`. The normal native
evidence set includes:

- Stable harness entrypoints.
- Build and host provenance.
- LLVM IR, assembly, pre-link objects, and symbol artifacts.
- A fat-LTO final linked equality evidence executable, its exact linker
  command and linker identity, and post-link disassembly, symbols, and size.
- Automated checks for suspicious generated-code patterns.
- Empirical timing tests on native executable targets, bound to the hashed
  timing executable, disassembly, symbol map, and linker command.
- ML-KEM DudeCT cases for key generation secret noise, encapsulation coins,
  decapsulation secret keys, implicit rejection, NTT, inverse NTT,
  product-domain conversion, basemul/dot products, and compress/decompress
  arithmetic.
- Binary-level checks for small high-risk kernels on supported Linux ELF/ISA
  paths.
- Miri and unsafe-code validation where the CT path uses unsafe Rust.

The linked equality executable retains production owner comparisons at the
distinct 16-, 28-, 32-, 48-, 64-, 1632-, 2400-, and 3168-byte owner widths.
The required owner timing cases remain the manifest-declared 16-, 32-, 48-,
and 64-byte cases. Public-length internal comparisons are mapped in `ct.toml`
to retained production entrypoints or to an explicit limitation; an uncovered
call is not silently treated as binary evidence.

This executable is an unpublished evidence surface. It proves only its exact
source, toolchain, backend, target, target features, feature set, profile, and
linker configuration. It is not the crate's public API, it is not a sealed
decision type, and it does not generalize to arbitrary downstream binaries.
Equality still returns `bool`; that declassification limitation remains until
T3.4.

Assembly triage is grouped by primitive, reachable symbol, finding kind, and
artifact. Register-indexed memory is presented first, then conditional control
flow, then indirect calls. `needs-binsec` means operand provenance remains
unproven; it is not a waiver, a proof, or a pass. An accepted waiver must bind
the exact primitive, symbol, kind, artifact, stable instruction locator,
function hash, source, public classification, rationale, reviewer, and review
date. Source or disassembly movement invalidates it.

BINSEC is required on the GNU Linux targets supported by the workflow. Every
manifest-required kernel must report `secure`. Other target reports record
BINSEC as `not_applicable` with the target policy reason; that status is not
binary proof. Each formal result is bound to its hashed proof driver,
disassembly, configuration, solver log, candidate identity, and toolchain.

Statistical timing checks must be described precisely:

```text
No leakage detected for this configuration.
```

They are evidence, not a formal proof.

## Consumer Verification

For release `vX.Y.Z`, download the crate, CT bundle, and checksums from that
exact GitHub release, then verify both attestations and hashes:

```bash
gh release download vX.Y.Z --repo loadingalias/rscrypto \
  -p 'rscrypto-X.Y.Z.crate' \
  -p 'rscrypto-X.Y.Z-source.tar.gz' \
  -p 'rscrypto-X.Y.Z-ct-evidence.tar.gz' \
  -p 'rscrypto-X.Y.Z-repository-controls.json' \
  -p 'rscrypto-X.Y.Z-release-manifest.json' \
  -p SHA256SUMS
sha256sum --check SHA256SUMS
gh release verify vX.Y.Z --repo loadingalias/rscrypto
gh release verify-asset vX.Y.Z rscrypto-X.Y.Z-ct-evidence.tar.gz --repo loadingalias/rscrypto
gh attestation verify rscrypto-X.Y.Z.crate --repo loadingalias/rscrypto
gh attestation verify rscrypto-X.Y.Z-source.tar.gz --repo loadingalias/rscrypto
gh attestation verify rscrypto-X.Y.Z-ct-evidence.tar.gz --repo loadingalias/rscrypto
gh attestation verify rscrypto-X.Y.Z-repository-controls.json --repo loadingalias/rscrypto
gh attestation verify rscrypto-X.Y.Z-release-manifest.json --repo loadingalias/rscrypto
gh attestation verify SHA256SUMS --repo loadingalias/rscrypto
mkdir ct-evidence && tar -xzf rscrypto-X.Y.Z-ct-evidence.tar.gz -C ct-evidence
(cd ct-evidence && sha256sum --check CT-EVIDENCE-MANIFEST.txt)
```

Inspect `CT-EVIDENCE-BUNDLE.json` and use only lanes whose exact target,
compiler, target CPU/features, profile, and primitive evidence match the
configuration being evaluated. A missing bundle, lane, or required pass means
there is no release-bound constant-time claim for that configuration.
