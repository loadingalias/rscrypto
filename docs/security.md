# Security Guidance

## Verification Failures

- `VerificationError` and `OpenError::Verification` are intentionally opaque.
- Treat verification failure as a generic authentication failure.
- Do not map verification failures to richer protocol responses that recreate an oracle.
- Buffer-length failures are caller-public shape errors, not secret-bearing authentication outcomes.

## Password Hash Verification

- PHC strings encode their own cost parameters.
- Use `Argon2id::verify_string_with_policy`, `Argon2d::verify_string_with_policy`,
  `Argon2i::verify_string_with_policy`, or `Scrypt::verify_string_with_policy` when encoded hashes can
  come from untrusted storage, tenant-controlled rows, network peers, or migration input.
- `Argon2VerifyPolicy::default()` and `ScryptVerifyPolicy::default()` admit hashes produced by the
  default parameter constructors. Services with stronger configured parameters should set explicit
  policy ceilings that match their deployment budget.
- The unbounded `verify_string` helpers remain for compatibility with trusted local hash stores.

## Nonces

- `Aes256Gcm`, `ChaCha20Poly1305`, `XChaCha20Poly1305`, and `Aegis256` require nonce uniqueness per key.
- `Aes256GcmSiv` is misuse-resistant, but nonce reuse is still not the normal operating model.
- Typed nonce wrappers prevent length mistakes, not lifecycle mistakes.
- For `Aes256Gcm`, prefer monotonic counters or protocol sequence numbers over ad hoc random nonces.

## Random Constructors

- Prefer `try_random()` in services and long-running processes.
- `random()` is a convenience wrapper that panics if the platform entropy source fails.

## RISC-V AES And AEGIS

- Bare-scalar `riscv64` targets without `Zkne`, `Zvkned`, or `V` use the constant-time portable AES and
  AES-round fallback.
- Expect a large throughput drop on that path relative to hardware or vector backends.
- Secret-indexed AES lookup tables are not used on the fallback path.
- The legacy `Riscv64Ttable` diagnostic label maps to portable code and is not selected as a live
  secret-indexed lookup-table backend.

## Post-Quantum Planning

- Ed25519 and X25519 are classical primitives.
- For systems with a long-lived trust horizon, plan a hybrid migration path instead of treating them as
  the final state.
- The repository roadmap tracks ML-DSA, ML-KEM, and other post-quantum primitives in `README.md`.
