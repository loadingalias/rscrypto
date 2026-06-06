# Security Guidance

## Constant-Time Claim Boundary

Constant-time claims are scoped to secret-bearing comparisons and operations,
not to every function in the crate. The rulebook for release claims, allowed
leakage, target scope, and invalidation lives in
[`docs/constant-time.md`](constant-time.md).

The current release evidence gate is `just ct-check`, which runs the same hard
pipeline as `just ct-full` for the selected native target:

| Platform class | CT evidence currently claimed |
|---|---|
| Linux `x86_64`/`aarch64`, GNU and MUSL | Artifact/heuristic review, DudeCT, and BINSEC for manifest-declared CT kernels. |
| Linux `powerpc64le`/`riscv64gc` | Artifact/heuristic review, DudeCT, and BINSEC where the runner and BINSEC toolchain complete successfully. |
| Linux `s390x` | Artifact/heuristic review and DudeCT. BINSEC is not claimed for s390x today. |
| macOS `aarch64`/`x86_64` | Artifact/heuristic review and DudeCT. BINSEC is not claimed for Mach-O today. |
| Windows MSVC `x86_64`/`aarch64` | Artifact/heuristic review and DudeCT. BINSEC is not claimed for PE/COFF today. |
| `no_std` and WASM | Not part of the release CT claim today. They need separate hardware, bytecode, or engine-specific evidence before being claimed. |

For claimed native targets, the CT manifest must cover the hot paths that
actually execute: accelerated ASM, SIMD, hardware-instruction backends, and
portable fallbacks used by CT-critical primitives. A backend being fast or
selected at runtime does not exempt it from the manifest or evidence gates.

| Surface | Claim boundary | Evidence |
|---|---|---|
| MAC tag verification | Full-length HMAC and KMAC tag comparison avoids secret-dependent equality behavior. | CT manifest coverage, `just ct-check`, HMAC/KMAC vectors, Wycheproof where mapped, and mismatch tests in [`docs/test-vector-coverage.md`](test-vector-coverage.md). |
| AEAD open failure | Authentication checks avoid richer failure detail, and failed-open paths wipe output buffers. | CT manifest coverage, `just ct-check`, AEAD oracle tests, Wycheproof where mapped, and tamper tests in [`docs/test-vector-coverage.md`](test-vector-coverage.md). |
| Ed25519 signing and secret-key public derivation | Secret scalar paths must avoid secret-dependent branches, table indices, memory addresses, and failure shape. | CT manifest coverage, `just ct-check`, RFC 8032, oracle, Wycheproof, and malformed-encoding tests in [`docs/test-vector-coverage.md`](test-vector-coverage.md). |
| Ed25519 verification | Signature acceptance/rejection uses a single opaque verification error at the public API boundary. Public verification is not a private-key CT claim unless promoted by the manifest. | RFC 8032, oracle, Wycheproof, and malformed-encoding tests in [`docs/test-vector-coverage.md`](test-vector-coverage.md). |
| X25519 scalar multiplication | Scalar multiplication must avoid secret-dependent field behavior and rejects all-zero shared secrets. | CT manifest coverage, `just ct-check`, RFC/vector, oracle, and Wycheproof coverage in [`docs/test-vector-coverage.md`](test-vector-coverage.md). |
| RSA private sign/decrypt | Private-operation paths require same-width opaque failures and CT evidence for manifest-declared hot paths. | CT manifest coverage, `just ct-check`, and the RSA evidence boundary below. |

These are not global constant-time claims: parsers, DER/PHC decoding, algorithm
or profile negotiation, key generation, OS randomness paths, public RSA
verify/encrypt paths, raw hashes, checksums, and fast non-cryptographic hashes.
Test vectors, differential tests, Miri, fuzzing, and leakage tests are evidence,
not formal proofs.

## Verification Failures

- `VerificationError` and `OpenError::Verification` are intentionally opaque.
- Treat verification failure as a generic authentication failure.
- Do not map verification failures to richer protocol responses that recreate an oracle.
- Buffer-length failures are caller-public shape errors, not secret-bearing authentication outcomes.

## Password Hash Verification

- PHC strings encode their own cost parameters.
- Use `Argon2id::verify_string_with_policy`, `Argon2d::verify_string_with_policy`, `Argon2i::verify_string_with_policy`, or `Scrypt::verify_string_with_policy` when encoded hashes can come from untrusted storage, tenant-controlled rows, network peers, or migration input.
- `Argon2VerifyPolicy::default()` and `ScryptVerifyPolicy::default()` admit hashes produced by the default parameter constructors. Services with stronger configured parameters should set explicit policy ceilings that match their deployment budget.
- The unbounded `verify_string` helpers remain for compatibility with trusted local hash stores.
- Migration guides for Argon2 and scrypt live in [`docs/migration/RustCrypto/argon2.md`](migration/RustCrypto/argon2.md) and [`docs/migration/RustCrypto/scrypt.md`](migration/RustCrypto/scrypt.md).

## Nonces

- `Aes128Gcm`, `Aes256Gcm`, `ChaCha20Poly1305`, `XChaCha20Poly1305`, and `Aegis256` require nonce uniqueness per key.
- `Aes128GcmSiv` and `Aes256GcmSiv` are misuse-resistant, but nonce reuse is still not the normal operating model.
- Typed nonce wrappers prevent length mistakes, not lifecycle mistakes.
- Nonce wrappers intentionally do not implement `Default`; all-zero nonces must be constructed explicitly.
- For `Aes128Gcm` and `Aes256Gcm`, prefer monotonic counters or protocol sequence numbers over ad hoc random nonces.

## Random Constructors

- Prefer `try_random()` in services and long-running processes.
- `random()` is a convenience wrapper that panics if the platform entropy source fails.

## RSA Evidence Boundary

RSA release claims require explicit evidence. Treat local tests, hosted CI, and
benchmark results as separate evidence classes; do not substitute one for
another.

Mandatory local macOS evidence before RSA performance work:

- `just check-all && just test` passes on the current worktree.
- `just test --all` covers RSA API, parser, padding, private-operation, CAVP,
  Wycheproof, allocation, and protocol tests through the normal test lane.
- `just test-fuzz` covers RSA parser, protocol-mapping, import, and
  private-operation fuzz targets through the normal fuzz lane.

Mandatory hosted CI evidence before a release claim:

- The normal check and test jobs pass for the release commit.
- Any skipped external helper is reported as skipped support evidence, not as a
  passed release requirement.

Mandatory RSA release evidence:

- Public verify/encrypt and private sign/decrypt/keygen/import pass the
  normal check, test, fuzz, and benchmark lanes.
- Canonical PKCS#1/PKCS#8 private-key export is correctness-tested, but it is
  not a required constant-time claim. DER INTEGER encoding is value-shaped by
  standard; use a future fixed-shape export format if serialization itself must
  be constant-time.
- RSA key generation follows the crate's FIPS 186-5 Appendix A.1.3
  probable-prime contract in code. It uses `getrandom` only to seed an internal
  HMAC_DRBG for key generation; this is not a CMVP/FIPS 140-3 validation claim.
- Same-width failure opacity is covered for OAEP, RSAES-PKCS1-v1_5, PSS, and
  RSASSA-PKCS1-v1_5.
- `just ct-check` passes on every claimed native target. The CT gate covers
  manifest-declared RSA private-operation hot paths through artifacts,
  heuristics, DudeCT, and BINSEC where supported.
- `just test-rsa-leakage` remains a targeted RSA regression check on Linux
  x86_64 and Linux aarch64. It supports the CT gate; it does not replace it.
- Miri covers every feasible safe private-key parser, signing, decryption,
  scratch-width, padding-reject, and key-generation helper path through
  `just test-miri --rsa`.
- Optional OpenSSL CLI and AWS-LC checks may support review; skipped optional
  helpers never count as completed release evidence.

The RSA leakage gate is regression evidence, not a proof of constant time.
Parsing, public operations, DRBG-backed key generation, and OS-backed
blinding-factor rejection may branch on public data or fresh randomness. Online
private sign/decrypt paths must keep secret-dependent padding, exponentiation,
CRT, and failure-output behavior covered by tests, fuzzing, Miri, and the RSA
leakage workflow before a release claim.

## Secret Serialization

- Secret key and shared-secret types mask `Debug`, but raw bytes remain extractable by explicit API.
- The optional `serde` feature covers non-secret byte wrappers such as nonces, tags, public keys, and signatures.
- The optional `serde-secrets` feature serializes raw secret-key and shared-secret bytes. Enable it only for controlled key-material storage or protocol formats, not for logs, telemetry, or broad application DTOs.

## RISC-V AES And AEGIS

- Bare-scalar `riscv64` targets without `Zkne`, `Zvkned`, or `V` use the constant-time portable AES and AES-round fallback.
- Expect a large throughput drop on that path relative to hardware or vector backends.
- Secret-indexed AES lookup tables are not used on the fallback path.

## Post-Quantum Planning

- Ed25519 and X25519 are classical primitives.
- For systems with a long-lived trust horizon, plan a hybrid migration path instead of treating them as the final state.
- The repository roadmap tracks ML-DSA, ML-KEM, and other post-quantum primitives via [GitHub issues](https://github.com/loadingalias/rscrypto/issues).
