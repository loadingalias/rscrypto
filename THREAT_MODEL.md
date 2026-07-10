# Threat Model

This is the security-review entry point for `rscrypto`. It defines what the
crate defends, what the caller owns, and where an audit should focus.

Related documents: [`SECURITY.md`](SECURITY.md) for vulnerability reporting,
[`docs/constant-time.md`](docs/constant-time.md) for the exact constant-time
claim model, [`docs/compliance.md`](docs/compliance.md) for regulatory
positioning, and [`ct.toml`](ct.toml) for the machine-readable CT claim set.

## Audit Scope

Review the `ct-intended` candidate core before the rest of the repository:

1. X25519 scalar multiplication.
2. Ed25519 signing and secret-key public derivation.
3. ECDSA P-256/P-384 caller-blinded signing.
4. RSA private sign/decrypt leaves.
5. ML-KEM secret-noise key generation, encapsulation coins, decapsulation
   secret-key material, and implicit rejection.
6. AEAD authentication and failed-open cleanup.
7. MAC/tag verification, constant-time byte equality, and selected
   password-verification comparisons.

Public parsing, raw hashes, checksums, non-cryptographic hashes, public-key
verification math, benchmark paths, unlisted targets, unlisted feature sets, and
unlisted build configurations are not blanket constant-time claims.

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
| Messages, AAD, ciphertexts, tags, signatures, encoded keys | Caller, usually relayed from a network peer | Untrusted. |
| Randomness | `getrandom` or caller-supplied fill closures | Trusted for entropy quality. Output lengths are fixed by the API. |
| CPU capability reports | CPUID, auxv, sysctl, OS APIs | Trusted. Forced-backend overrides are validated before use. |
| Build configuration | Cargo features, target features | Trusted. |

Outputs are digests, tags, ciphertexts, signatures, derived keys, and opaque
errors. A failed verification returns one success/failure bit and nothing else.

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

1. **Network attacker.** Supplies malformed ciphertexts, signatures, tags, and
   encoded keys. Goals: memory corruption, reachable panics, oracle behavior
   beyond the single failure bit, and accepting inputs other implementations
   reject.
2. **Co-located timing attacker.** Measures timing of secret-bearing
   operations. `ct.toml` identifies candidate surfaces; a primitive/target
   configuration enters the release claim only when every required gate passes
   in the matching attested release bundle, under the model in
   [`docs/constant-time.md`](docs/constant-time.md).
3. **Caller mistakes.** Nonce reuse, dropped verification results, weak
   parameters. The API uses typed keys and nonces, `#[must_use]` verification
   results, `NonceCounter` invocation budgets, opaque errors, and zeroize on
   drop.
4. **Supply-chain attacker.** Targets the path between this repository and the
   artifact a downstream build consumes.

Out of scope:

- Physical side channels: power, electromagnetic, acoustic, fault injection,
  rowhammer.
- A compromised host, OS, hypervisor, or toolchain.
- Speculative-execution attacks, beyond avoiding secret-dependent branches and
  memory addresses in claimed paths.
- Entropy failure in the OS or in caller-supplied randomness.
- Protocol composition errors in downstream code.

## Attack Surface

Ordered by exposure to untrusted input:

| Surface | Entry points | Primary risks |
|---|---|---|
| Parsers | RSA DER/SPKI/PKCS#8 import, ECDSA DER signatures and SEC1 points, ML-KEM key and ciphertext parsing, PHC strings, hex | Memory safety, panics, accepting what should be rejected |
| Verification oracles | MAC `verify_tag`, AEAD open, signature `verify`, ML-KEM implicit rejection | Timing or error detail beyond the single failure bit |
| Secret-bearing compute | Sign, decrypt, decapsulate, derive; the release-evidenced subset of `ct.toml` | Timing leakage, incorrect arithmetic |
| `unsafe` SIMD and assembly kernels | Per-architecture modules | Undefined behavior, divergence from the portable authority |
| Dispatch | `src/platform`, `src/backend` | Selecting a kernel the CPU cannot run, or one that produces wrong output |

## Mitigations And Evidence

| Risk | Mitigation | Evidence |
|---|---|---|
| Memory safety | `unsafe` confined to kernel and platform modules; unsafe lint gates enabled | Miri on portable paths in CI |
| Parser abuse | Strict imports, `strict_*` arithmetic, release overflow checks | Fuzz targets, Wycheproof where mapped, official vectors |
| Wrong output from accelerated kernels | Portable path is the byte-for-byte authority | Portable-vs-accelerated differential tests and native CI |
| Timing leakage | Constant-time coding rules on claimed paths | `ct.toml` evidence gate: timing tests, generated-code review, binary checks where supported |
| Oracle behavior | Opaque errors, failed-open output wipe, single-bit failure shape | AEAD and verification tests, fuzz targets |
| Secret exposure at rest | Zeroize on drop, masked `Debug`, constant-time equality on secret types | `src/secret.rs` and per-type tests |
| Supply chain | Minimal optional runtime dependencies, `cargo deny`, `cargo audit`, signed tags, Trusted Publishing, release attestations | `deny.toml`, `.github/workflows/release.yaml`, `docs/release.md` |

## Known Gaps

- No third-party security audit yet.
- Miri covers portable paths only; sanitizer and interpreter coverage do not
  execute every native SIMD or assembly kernel.
- Constant-time evidence is produced by CI and release workflows. Consumers
  should use the versioned release bundle for the exact artifact they deploy.
  Releases through `v0.6.4` have no such bundle and carry no release-bound CT
  claim. Windows, Linux MUSL, Intel macOS, bare-metal, and WASM physical timing
  evidence remains explicitly deferred.

## Review Priorities

Where an external review buys the most, in order:

1. The candidate constant-time core listed above.
2. RSA DER import and the PKCS#1 v1.5, PSS, and OAEP padding checks.
3. `unsafe` kernels with the weakest tool coverage: hand-written assembly is
   not visible to Miri.
4. Dispatch correctness under unusual CPU capability combinations.
