# Security Guidance

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

- Public verify/encrypt and private sign/decrypt/keygen/import/export pass the
  normal check, test, fuzz, and benchmark lanes.
- Constant-time audit findings are resolved by code, tests, or written
  rejection in `docs/security/rsa-side-channel-audit.md`.
- Same-width failure opacity is covered for OAEP, RSAES-PKCS1-v1_5, PSS, and
  RSASSA-PKCS1-v1_5.
- `just test-rsa-leakage` passes on Linux x86_64 and Linux aarch64.
- Miri covers every feasible safe private-key parser, signing, decryption,
  scratch-width, padding-reject, and key-generation helper path through
  `just test-miri --rsa`.
- Optional OpenSSL CLI and AWS-LC checks may support review; skipped optional
  helpers never count as completed release evidence.

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
