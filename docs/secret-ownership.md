# Secret ownership

This inventory tells users which public values retain secrets and which generic
operations can duplicate or expose them. It is a type-level contract, not proof
that every compiler-created copy is erased.

## Confidential owners

| Owner | Duplication | Exposure |
| --- | --- | --- |
| `SecretBytes<N>`, `SecretVec` | Not `Clone` or `Copy` | Consuming export transfers bytes and cleanup responsibility to the caller. |
| AEAD keys and contexts | Keys use explicit `duplicate_secret`; contexts do not duplicate | Key export is explicit; `Debug` is redacted. |
| Header-protection keys and contexts | No generic duplication | No public export; `Debug` is redacted. |
| ECDSA and Ed25519 secret keys and keypairs | Explicit `duplicate_secret` | Secret-key export is explicit; keypair `Debug` shows only public data. |
| X25519 secrets and ML-KEM decapsulation/shared secrets | Explicit `duplicate_secret` | Secret export is explicit; `Debug` is redacted. |
| `RsaPrivateKey`, `RsaPrivateScratch` | Not `Clone` or `Copy` | Private DER export returns `SecretVec`; `Debug` shows public metadata only. |
| HMAC, HKDF, KMAC, PBKDF2, and Poly1305 state | No generic duplication | `Debug` is redacted; keyed state is not serialized. |
| Keyed BLAKE2 state | `Clone` where required by the shared `Digest` contract | `Debug` is redacted; cloning duplicates keyed state. |
| `Blake3`, `Blake3XofReader` | `Clone` | In keyed or derive-key mode, cloning duplicates secret-derived state. |
| Password-hashing state and owned work memory | Borrowed contexts may be `Copy`; owned state is not | Borrowed copies duplicate references, not password or pepper bytes. |

Typed private keys, shared secrets, keyed states, expanded schedules, and
private-operation scratch follow the same confidential-owner rules even when
not named individually above.

## Public authentication values

AEAD tags, HMAC tags, `Poly1305Tag`, and `Blake3KeyedHash` are
protocol-visible. They may implement `Clone`, `Copy`, raw `Debug`, or public
serialization. Their verification still uses full-traversal comparison where
the concrete type provides it.

Public keys, signatures, nonces, ciphertexts, PHC records, unkeyed hash state,
and checksums are not secret owners. Callers can still place sensitive data in
their buffers; the crate cannot manage caller-owned memory.

## Explicit escape hatches

- `duplicate_secret()` creates another secret lifetime.
- `SecretBytes::expose()` returns ordinary bytes after clearing its source.
- `SecretVec::into_unprotected_vec()` transfers an allocation without clearing
  it; the caller becomes responsible for that memory.
- `serde-secrets` authorizes secret serialization.
- `expert::DisplaySecret` deliberately prints borrowed secret bytes.
- `as_bytes` and similar borrows expose bytes for the borrow's lifetime.

Do not log, format, or serialize secrets unless the integration requires that
exact transfer.

Changes to secret owners require review of `Clone`, `Copy`, `Debug`, Serde,
export, allocation, comparison, and cleanup behavior. See
[`secret-lifecycle.md`](secret-lifecycle.md) for cleanup evidence and
[`constant-time.md`](constant-time.md) for timing claims.
