# Threat Model

Use this document to scope a security review of `rscrypto`. It defines the
crate boundary, protected assets, caller responsibilities, threat assumptions,
evidence, and review priorities.

Related documents: [`SECURITY.md`](SECURITY.md) for vulnerability reporting,
[`docs/constant-time.md`](docs/constant-time.md) for the exact constant-time
claim model, [`docs/secret-ownership.md`](docs/secret-ownership.md) for the
secret-bearing type and heap inventory,
[`docs/secret-lifecycle.md`](docs/secret-lifecycle.md) for cleanup and redaction
evidence, [`docs/compliance.md`](docs/compliance.md) for regulatory positioning,
and [`ct.toml`](ct.toml) for the machine-readable CT claim set.

## Audit scope

Review the `ct_intended` candidate core before the rest of the repository:

1. X25519 scalar multiplication.
2. Ed25519 signing and secret-key public derivation.
3. ECDSA P-256/P-384 caller-blinded signing.
4. RSA private sign/decrypt leaves.
5. ML-KEM secret-noise key generation, encapsulation coins, decapsulation
   secret-key material, and implicit rejection.
6. AEAD authentication, including AES-SIV synthetic-IV derivation, and failed-open cleanup.
7. Header-protection mask generation.
8. MAC/tag verification, fixed-size owner comparison/declassification, and selected
   password-verification comparisons.

This order prioritizes secret-dependent computation; it does not remove public
parsers, dispatch, or unsafe kernels from the security boundary. Public parsing,
raw hashes, compatibility-only WebSocket accept digests, checksums,
non-cryptographic hashes, public-key verification math, benchmark paths, and
unlisted build configurations carry no blanket constant-time claim.

## System boundary

`rscrypto` is a primitives library. It computes hashes, MACs, KDFs, password
hashes, AEADs, signatures, key exchanges, and checksums on caller-provided
inputs. It does not open sockets, read the clock, or spawn production threads
outside the opt-in `parallel` feature. With `std`, runtime CPU detection may
read OS-exposed capability data such as `/proc/self/auxv`, `/proc/cpuinfo`, and
sysfs; `getrandom` constructors obtain randomness from the operating system.
The crate does not read application data or manage keys on disk. The caller
owns protocol design, key storage and rotation, entropy policy, nonce
lifecycle, transport, and access control.

Inputs crossing the boundary:

| Input                                                      | Source                                       | Assumption                                                                                                                                                                  |
| ---------------------------------------------------------- | -------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Keys, passwords, seeds                                     | Caller                                       | The caller protects confidentiality and supplies the required entropy. Imports enforce documented shape and algorithm constraints, not key quality.                         |
| Messages, AAD, ciphertexts, tags, signatures, encoded keys | Caller, usually relayed from a network peer  | Untrusted.                                                                                                                                                                  |
| Randomness                                                 | `getrandom` or caller-supplied fill closures | The operating system or caller provides the required entropy quality and fills every requested byte. Request lengths are fixed by the selected operation and public inputs. |
| CPU capability reports                                     | CPUID, auxv, sysctl, OS APIs                 | The host reports capabilities correctly. Forced-backend overrides are validated before use.                                                                                 |
| Build configuration                                        | Cargo features, target features              | The builder selects and records the intended configuration.                                                                                                                 |

RSA caller-random signing accepts bytes, not mathematical blinding values.
rscrypto owns bounded sampling, inversion, range validation, private
arithmetic, and the public fault check. A fill error is reduced to
`RsaPrivateOpError::EntropyUnavailable`; the callback's error payload does not
cross the primitive boundary. Accepted factors are inverted directly modulo
the public RSA modulus with a fixed public-width batched schedule; inversion
does not consume or branch on the private CRT factors. Candidate retry count
and the final opaque success/failure remain observable as documented.

Outputs are digests, tags, ciphertexts, signatures, derived keys, and opaque
errors. Direct comparison of fixed-size secret-bearing owners returns an opaque
`CtDecision`; only explicit, consuming declassification exposes the equality
bit. A failed verification exposes one public success/failure result; timing
claims remain limited to the release-evidenced configurations.

## Assets

1. Long-term secrets: private keys, passwords, master keys.
2. Session secrets: X25519 and ML-KEM shared secrets, AEAD and header-protection keys, signing
   nonces, blinding factors.
3. Intermediate secret state: key schedules, scalars, limbs, DRBG state,
   sampler buffers.
4. Plaintext inside AEAD seal and open calls.
5. Integrity of the published crate artifacts.

## Threats in scope

1. **Network attacker.** Supplies malformed ciphertexts, signatures, tags, and
   encoded keys. Goals: memory corruption, reachable panics, oracle behavior
   beyond the single failure bit, and accepting inputs other implementations
   reject.
2. **Co-located timing attacker.** Measures timing of secret-bearing
   operations. `ct.toml` identifies candidate surfaces; a primitive/target
   configuration enters the release claim only when every required gate passes
   in the matching attested release bundle, under the model in
   [`docs/constant-time.md`](docs/constant-time.md).
3. **Caller misuse.** Reuses nonces, drops verification results, or selects weak
   parameters. The API uses typed keys and nonces, `#[must_use]` verification
   results, `NonceCounter` invocation budgets, opaque errors, and explicit drop
   cleanup for the named secret owners.

   AES-SIV-CMAC-256 preserves authenticity when a nonce repeats, but it reveals
   equality when the complete key/nonce/AAD/plaintext tuple repeats. The nonce-based
   profile therefore still treats nonce uniqueness as the normal caller contract;
   misuse resistance is a containment property, not permission to omit nonce
   management.
4. **Supply-chain attacker.** Targets the path between this repository and the
   artifact a downstream build consumes.

The following threats are out of scope:

- Physical side channels: power, electromagnetic, acoustic, fault injection,
  rowhammer.
- A compromised host, OS, hypervisor, or toolchain.
- Speculative-execution attacks, beyond avoiding secret-dependent branches and
  memory addresses in claimed paths.
- Entropy failure in the OS or in caller-supplied randomness.
- Protocol composition errors in downstream code.

## Attack surface

Ordered by exposure to untrusted input:

| Surface                 | Entry points                                                                                                          | Primary risks                                                            |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| Parsers                 | RSA DER/SPKI/PKCS#8 import, ECDSA DER signatures and SEC1 points, ML-KEM key and ciphertext parsing, PHC strings, hex | Memory safety, panics, accepting what should be rejected                 |
| Verification oracles    | MAC `verify_tag`, AEAD open, signature `verify`, ML-KEM implicit rejection                                            | Timing or error detail beyond the single failure bit                     |
| Secret-bearing compute  | Sign, decrypt, decapsulate, derive, generate header masks; the release-evidenced subset of `ct.toml`                  | Timing leakage, incorrect arithmetic                                     |
| `unsafe` low-level code | SIMD/assembly kernels, raw buffer helpers, zeroization, and dispatch                                                  | Undefined behavior, divergence from the portable authority               |
| Dispatch                | `src/platform`, `src/backend`                                                                                         | Selecting a kernel the CPU cannot run, or one that produces wrong output |
| Compatibility operations | `hashes::legacy::WebSocketAcceptDigest::compute`                                                                     | Capability expansion or treating broken SHA-1 collision resistance as authentication |

## Mitigations and evidence

| Risk                                  | Mitigation                                                                                                                            | Evidence                                                                                                                                                                             |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Memory safety                         | Unsafe operations are lint-gated and require local `SAFETY` proofs; the portable Rust path remains authoritative                      | Miri on portable paths in CI                                                                                                                                                         |
| Parser abuse                          | Strict imports, `strict_*` arithmetic, release overflow checks                                                                        | Fuzz targets, Wycheproof where mapped, official vectors                                                                                                                              |
| Wrong output from accelerated kernels | Portable path is the byte-for-byte authority                                                                                          | Portable-vs-accelerated differential tests and native CI                                                                                                                             |
| Timing leakage                        | Constant-time coding rules on claimed paths                                                                                           | `ct.toml` evidence gate: timing tests, generated-code review, binary checks where supported                                                                                          |
| Oracle behavior                       | Opaque errors, failed-open output clearing, single-bit failure shape                                                                  | AEAD and verification tests, fuzz targets                                                                                                                                            |
| Secret exposure at rest               | Zeroize at the last owned use and on drop, masked `Debug` and errors, and sealed fixed-size comparison only on semantic secret owners | [`docs/secret-ownership.md`](docs/secret-ownership.md), [`docs/secret-lifecycle.md`](docs/secret-lifecycle.md), `scripts/check/zeroize-evidence.sh`, and `tests/secret_redaction.rs` |
| Supply chain                          | Minimal optional runtime dependencies, `cargo deny`, `cargo audit`, signed tags, Trusted Publishing, release attestations             | [`deny.toml`](deny.toml), [`.github/workflows/release.yaml`](.github/workflows/release.yaml), [`docs/release.md`](docs/release.md)                                                   |
| Legacy primitive misuse               | Semantic-only API, explicit leaf feature, no umbrella membership, no raw/streaming SHA-1                                                | Feature-boundary check, compile-fail root-surface doctest, RFC/oracle tests                                                                                                      |

## Known gaps

- No third-party security audit is claimed.
- Named secret owners and explicit temporaries use volatile source-level wipes,
  and `scripts/check/zeroize-evidence.sh` checks optimized lifecycle shapes in
  MIR, LLVM IR, and host assembly. This does not prove that every
  compiler-created register copy or spill is erased on every target. The crate
  does not lock pages, prevent swapping, or replace hardware-backed key storage.
- Miri covers portable paths only; sanitizer and interpreter coverage do not
  execute every native SIMD or assembly kernel.
- Constant-time evidence is produced by CI and release workflows. Consumers
  should use the versioned release bundle for the exact artifact they deploy.
  Releases through `v0.6.4` have no such bundle and carry no release-bound CT
  claim. Windows, Linux MUSL, Intel macOS, bare-metal, and WASM physical timing
  evidence remains explicitly deferred.

## Review priorities

Prioritize external review in this order:

1. The candidate constant-time core listed above.
2. RSA DER import and the PKCS#1 v1.5, PSS, and OAEP padding checks.
3. `unsafe` kernels with the weakest tool coverage: hand-written assembly is
   not visible to Miri.
4. Dispatch correctness under unusual CPU capability combinations.
