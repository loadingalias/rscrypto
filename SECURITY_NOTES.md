# Security Notes

## Verification Failures

- `VerificationError` and `OpenError::Verification` are intentionally opaque.
- Treat verification failure as a generic authentication failure.
- Do not map verification failures to richer protocol responses that would recreate an oracle.
- Buffer-length failures are caller-public shape errors; they are not secret-bearing authentication outcomes.

## Nonces

- `Aes256Gcm`, `ChaCha20Poly1305`, `XChaCha20Poly1305`, and `Aegis256` require nonce uniqueness per key.
- `Aes256GcmSiv` is misuse-resistant, but nonce reuse is still not the normal operating model.
- Typed nonce wrappers prevent length mistakes, not lifecycle mistakes.
- For `Aes256Gcm`, prefer monotonic counters or protocol sequence numbers over ad hoc random nonces.

## Random Constructors

- Prefer `try_random()` in services and long-running processes.
- `random()` is a convenience wrapper that panics if the platform entropy source fails.

## RISC-V AES And AEGIS

- Bare-scalar `riscv64` targets without `Zkne`, `Zvkned`, or `V` now use the constant-time portable AES and AES-round fallback.
- Expect a large throughput drop on that path relative to hardware or vector backends.
- Secret-indexed AES lookup tables are not used on that fallback path.

## Post-Quantum Planning

- Ed25519 and X25519 are classical primitives.
- For systems with a long-lived trust horizon, plan a hybrid migration path instead of treating them as the final state.
- The current repository roadmap points toward `ML-DSA` and `ML-KEM`; see `docs/tasks/pqe_pqc.md`.
