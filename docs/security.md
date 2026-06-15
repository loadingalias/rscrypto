# Security Guidance

`rscrypto` protects specific primitive surfaces. Callers still own protocol
composition, key lifecycle, nonce lifecycle, deployment policy, and error
translation.

## Constant-Time Boundary

Constant-time claims are scoped to named secret-bearing operations. They are not
whole-crate claims and they are not portable to unmeasured build
configurations.

The exact release claim is the set of primitive/configuration pairs marked
`ct_claimed` in [`../ct.toml`](../ct.toml). The main user-facing surfaces are:

| Surface | User-facing boundary |
|---|---|
| MAC and keyed verification | Full-length HMAC and KMAC tag verification uses opaque success/failure. |
| AEAD open | Authentication failure is opaque, and failed-open paths wipe output buffers. |
| ECDSA P-256/P-384 | Caller-blinded signing is the private-key CT surface. Public verification is public-input work, with an opaque verification error. |
| Ed25519 | Signing and secret-key public derivation are private-key CT surfaces. Public verification is public-input work, with an opaque verification error. |
| X25519 | Scalar multiplication is a private-scalar CT surface; all-zero shared secrets are rejected. |
| RSA | Private sign/decrypt leaves are CT-critical. Public verify/encrypt, parsing, and key import are public-input work unless promoted by the CT manifest. |
| Password verification | Final secret comparisons are CT-critical. Argon2d, Argon2id, and scrypt still have algorithm-level data-dependent memory access, so do not describe them as globally constant-time. |

These are not blanket constant-time claims: DER/PHC parsing, public-key
verification math, key generation, OS randomness, serialization, raw hashes,
checksums, non-cryptographic hashes, benchmarks, and feature dispatch.

The exact evidence model is in [`constant-time.md`](constant-time.md). No
third-party audit, FIPS 140-3 validation, or formal proof is claimed today.

## Verification Failures

`VerificationError` and `OpenError::Verification` do not reveal failure detail.
Treat them as generic authentication failures.

Do not translate verification failures into richer protocol responses such as
"wrong tag", "bad padding", "unknown key", or "malformed signature" when the
peer can observe the distinction. That recreates an oracle at the application
layer.

Buffer-size and format errors are caller-public shape errors. They are separate
from secret-bearing authentication failures.

## Nonces

- `Aes128Gcm`, `Aes256Gcm`, `ChaCha20Poly1305`, `XChaCha20Poly1305`, and
  `Aegis256` require nonce uniqueness per key.
- `Aes128GcmSiv` and `Aes256GcmSiv` are misuse-resistant, but nonce reuse should
  still not be the normal operating model.
- Typed nonce wrappers prevent length mistakes, not lifecycle mistakes.
- Nonce wrappers do not implement `Default`; all-zero nonces must be constructed
  explicitly.
- For AES-GCM, prefer monotonic counters or protocol sequence numbers over ad
  hoc random nonces.

## Random Constructors

Prefer `try_random()` in services and long-running processes. It returns an
error if platform entropy fails.

`random()` is a convenience wrapper that panics on entropy failure. It is best
for tests, examples, and applications where panic-on-entropy-failure is an
acceptable policy.

ECDSA deterministic signing does not use OS randomness. ECDSA key generation,
caller-blinded signing, and no-std RSA encryption can use caller-supplied
byte-filling closures. RSA key generation and OS-backed RSA private-operation
blinding require the `getrandom` feature. RSA encryption seed hooks are hidden
diagnostic/test-vector APIs; production encryption should use `getrandom` or a
fresh random-fill closure.

## Password Hash Verification

PHC strings encode their own cost parameters. If encoded password hashes can
come from untrusted storage, tenant-controlled rows, network peers, or migration
input, use the bounded verification APIs:

- `Argon2id::verify_string`
- `Argon2id::verify_string_with_context`
- `Argon2id::verify_string_with_policy`
- `Argon2d::verify_string`
- `Argon2d::verify_string_with_context`
- `Argon2d::verify_string_with_policy`
- `Argon2i::verify_string`
- `Argon2i::verify_string_with_context`
- `Argon2i::verify_string_with_policy`
- `Scrypt::verify_string`
- `Scrypt::verify_string_with_policy`

The default policies admit hashes produced by the default parameter
constructors. Services with stronger local parameters should set explicit policy
ceilings that match their CPU and memory budget.

The `verify_string_unbounded` and `verify_string_with_context_unbounded` helpers
remain available for trusted local migration inputs and test-vector harnesses.

## Secret Serialization

Secret key and shared-secret types mask `Debug`, but raw bytes remain
extractable through explicit APIs.

The optional `serde` feature covers non-secret byte wrappers such as nonces,
tags, public keys, and signatures.

The optional `serde-secrets` feature serializes raw secret-key and shared-secret
bytes. Enable it only for controlled key-material storage or protocol formats,
not logs, telemetry, or broad application DTOs.

## Platform Notes

The portable implementation is always present and is the correctness reference
for accelerated backends.

Bare-scalar `riscv64` targets without `Zkne`, `Zvkned`, or `V` use the portable
AES and AES-round fallback. That path is designed to avoid secret-indexed AES
lookup tables, but it is much slower than hardware or vector backends.

Use [`platforms.md`](platforms.md) for target support and acceleration details.
Use [`test-vector-coverage.md`](test-vector-coverage.md) to inspect vector,
oracle, fuzz, and negative-input coverage.
