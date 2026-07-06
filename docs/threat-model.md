# Threat Model

This document defines the security boundary of `rscrypto`: what the crate
defends, against whom, and what the caller still owns. It is the starting
point for an external security review.

Related documents: [`SECURITY.md`](../SECURITY.md) for the reporting process,
[`constant-time.md`](constant-time.md) for the exact constant-time claim
model, [`compliance.md`](compliance.md) for the regulatory posture, and
[`ct.toml`](../ct.toml) for the machine-readable claim set.

## System Boundary

`rscrypto` is a primitives library. It computes hashes, MACs, KDFs, password
hashes, AEADs, signatures, key exchanges, and checksums on caller-provided
buffers. It does not open sockets, touch the filesystem, read the clock, or
spawn threads outside the opt-in `parallel` feature. Protocol design, key
storage, key rotation, and transport are the caller's responsibility.

Everything that crosses the boundary:

| Input | Source | Trust |
|---|---|---|
| Keys, passwords, seeds | Caller | Trusted for secrecy, not for validity. Imports are validated. |
| Messages, AAD, ciphertexts, tags, signatures, encoded keys (DER, SPKI, SEC1, PHC, hex) | Caller, usually relayed from a network peer | Untrusted. |
| Randomness | `getrandom` (opt-in) or caller-supplied fill closures | Trusted for entropy quality. Output lengths are fixed by the API. |
| CPU capability reports | CPUID, auxv, sysctl, OS APIs | Trusted. Forced-backend overrides are validated against detected capabilities before use. |
| Build configuration | Cargo features, target features | Trusted. |

Outputs are digests, tags, ciphertexts, signatures, derived keys, and opaque
errors. A failed verification returns one success/failure bit and nothing
else.

## Assets

1. Long-term secrets: private keys, passwords, master keys.
2. Session secrets: X25519 and ML-KEM shared secrets, AEAD keys, signing
   nonces, blinding factors.
3. Intermediate secret state: key schedules, scalars, limbs, DRBG state,
   sampler buffers.
4. Plaintext inside AEAD seal and open calls.
5. Integrity of the published crate artifacts.

## Adversaries

In scope:

1. **Network attacker.** Supplies malformed ciphertexts, signatures, tags,
   and encoded keys. Goals: memory corruption, reachable panics, oracle
   behavior beyond the single failure bit, acceptance of inputs other
   implementations reject.
2. **Co-located timing attacker.** Measures timing of secret-bearing
   operations. Scope is exactly the `ct_claimed` set in `ct.toml`, under the
   claim model in [`constant-time.md`](constant-time.md).
3. **Caller mistakes.** Nonce reuse, dropped verification results, weak
   parameters. Not an attacker, but the API is shaped against them: typed
   keys and nonces, `#[must_use]` on verification results, `NonceCounter`
   invocation budgets, opaque errors, zeroize on drop.
4. **Supply-chain attacker.** Targets the path between this repository and
   the artifact a downstream build consumes.

Out of scope:

- Physical side channels: power, electromagnetic, acoustic, fault injection,
  rowhammer.
- A compromised host, OS, hypervisor, or toolchain.
- Speculative-execution attacks, beyond keeping secret-dependent branches and
  memory addresses out of claimed paths.
- Entropy failure in the OS or in caller-supplied randomness.
- Protocol composition errors in downstream code.

## Attack Surface

Ordered by exposure to untrusted input:

| Surface | Entry points | Primary risks |
|---|---|---|
| Parsers | RSA DER/SPKI/PKCS#8 import, ECDSA DER signatures and SEC1 points, ML-KEM key and ciphertext parsing, PHC strings, hex | Memory safety, panics, accepting what should be rejected |
| Verification oracles | MAC `verify_tag`, AEAD open, signature `verify`, ML-KEM implicit rejection | Timing or error detail beyond the single failure bit |
| Secret-bearing compute | Sign, decrypt, decapsulate, derive; the `ct_claimed` set | Timing leakage, incorrect arithmetic |
| `unsafe` SIMD and assembly kernels | Per-architecture modules | Undefined behavior, divergence from the portable authority |
| Dispatch | `src/platform`, `src/backend` | Selecting a kernel the CPU cannot run, or one that produces wrong output |

## Mitigations and Evidence

| Risk | Mitigation | Evidence |
|---|---|---|
| Memory safety | `unsafe` confined to kernel and platform modules; `unsafe_op_in_unsafe_fn` and `undocumented_unsafe_blocks` deny | Miri (stacked and tree borrows) on portable paths in CI |
| Parser abuse | Strict imports, `strict_*` arithmetic, `overflow-checks` kept in release builds | Fuzz targets under `fuzz/` and `fuzz-packages/`, Wycheproof and official vectors under `tests/` |
| Wrong output from accelerated kernels | Portable path is the byte-for-byte authority | Differential portable-vs-accelerated tests, cross-architecture CI on native hardware |
| Timing leakage | Constant-time coding rules on claimed paths | Per-primitive evidence gate in `ct.toml`: timing tests, generated-code review, binary checks |
| Oracle behavior | Opaque errors, failed-open output wipe, single-bit failure shape | AEAD and verification tests, fuzz targets |
| Secret exposure at rest | Zeroize on drop, masked `Debug`, constant-time `PartialEq` on secret types | `src/secret.rs` and per-type tests |
| Supply chain | Three optional runtime dependencies, `cargo deny` with crates.io-only sources, SHA-pinned CI actions, committed lockfile | `deny.toml`, release provenance attestation in `.github/workflows/release.yaml` |

## Known Gaps

- No third-party audit yet. Single maintainer.
- Sanitizers do not run over the native SIMD and assembly kernels. Miri
  covers the portable paths only.
- The fuzz corpus is only partially published; third parties cannot yet
  reproduce fuzz coverage from the repository alone.
- Constant-time evidence is produced in CI but not yet published as a
  versioned bundle per release.

## Review Priorities

Where an external review buys the most, in order:

1. The `ct_claimed` core: X25519, Ed25519 signing, ECDSA blinded signing, RSA
   private operations, ML-KEM decapsulation, AEAD open.
2. RSA DER import and the PKCS#1 v1.5, PSS, and OAEP padding checks.
3. `unsafe` kernels with the weakest tool coverage: hand-written assembly is
   not visible to Miri.
4. Dispatch correctness under unusual CPU capability combinations.
