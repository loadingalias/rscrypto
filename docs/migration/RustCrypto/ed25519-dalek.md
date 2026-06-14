# Migration: `ed25519-dalek` → `rscrypto`

> Replace `SigningKey` / `VerifyingKey` / `Signature` with `Ed25519SecretKey` / `Ed25519PublicKey` / `Ed25519Signature`. Same RFC 8032 algorithm, byte-identical signatures (Ed25519 is deterministic), strict verification on by default.

Verified against `ed25519-dalek = "2.2.0"` and the `rscrypto` 0.5.0 line.
Evidence: `tests/ed25519_rfc8032_vectors.rs`, `tests/ed25519_oracle.rs`, and `tests/ed25519_wycheproof.rs`.

## TL;DR

| | Before (`ed25519-dalek` 2.x) | After (`rscrypto` 0.5.0) |
|---|---|---|
| Cargo dep | `ed25519-dalek = "2.2"` | `rscrypto = { version = "0.5.0", features = ["ed25519"] }` |
| Import | `use ed25519_dalek::{SigningKey, Signer, Verifier};` | `use rscrypto::{Ed25519SecretKey, Ed25519PublicKey, Ed25519Signature};` |
| Sign | `signing_key.sign(msg)` | `secret.sign(msg)` |
| Verify | `verifying_key.verify_strict(msg, &sig)?` | `public_key.verify(msg, &sig)?` |

## Cargo.toml

```toml
# Before
[dependencies]
ed25519-dalek = "2.2"
```

```toml
# After
[dependencies]
rscrypto = { version = "0.5.0", features = ["ed25519"] }
```

The `ed25519` feature implies `sha2` (Ed25519 uses SHA-512 internally per RFC 8032).

## Type map

| `ed25519-dalek` type | rscrypto type | Bytes |
|---|---|---|
| `SigningKey` | `Ed25519SecretKey` | `[u8; 32]` (seed) |
| `VerifyingKey` | `Ed25519PublicKey` | `[u8; 32]` |
| `Signature` | `Ed25519Signature` | `[u8; 64]` |
| (implicit pair) | `Ed25519Keypair` | secret + public + expanded form |

## API patterns

### Construct from seed bytes

```rust
// Before
use ed25519_dalek::SigningKey;
let signing_key = SigningKey::from_bytes(&[0x42u8; 32]);    // takes &[u8; 32]
let verifying_key = signing_key.verifying_key();
```

```rust
// After
use rscrypto::{Ed25519SecretKey};
let secret = Ed25519SecretKey::from_bytes([0x42u8; 32]);     // takes [u8; 32] by value
let public = secret.public_key();
```

`Ed25519SecretKey::from_bytes` is a `const fn` — bake long-lived signing keys at compile time without runtime cost.

### Random key generation

```rust
// Before
use ed25519_dalek::SigningKey;
use rand_core::OsRng;
let signing_key = SigningKey::generate(&mut OsRng);
```

```rust
// After (with `getrandom` feature)
use rscrypto::Ed25519SecretKey;
let secret = Ed25519SecretKey::generate_random()?;       // returns Result<_, Error>

// Or supply your own RNG via a closure (no extra feature):
let secret = Ed25519SecretKey::generate(|buf| {
    // fill buf: &mut [u8; 32] from your CSPRNG
});
```

The closure form is the no-`getrandom` path — useful when you have a different entropy source (HSM, TPM, embedded TRNG).

### Sign

```rust
// Before
use ed25519_dalek::Signer;
let sig: ed25519_dalek::Signature = signing_key.sign(b"message");
```

```rust
// After
use rscrypto::Ed25519Signature;
let sig: Ed25519Signature = secret.sign(b"message");
```

`secret.sign(...)` is inherent in rscrypto — no `Signer` trait import needed. Ed25519 is deterministic (RFC 8032 §5.1.6); the same `(seed, message)` pair always produces the same signature in both crates (verified in the harness).

### Verify

```rust
// Before
use ed25519_dalek::Verifier;
verifying_key.verify_strict(b"message", &sig)?;             // strict mode: rejects small-order pks
// OR
verifying_key.verify(b"message", &sig)?;                    // legacy lax mode
```

```rust
// After
public_key.verify(b"message", &sig)?;                       // strict by default — there is no lax mode
```

rscrypto's `verify` is *always* strict (rejects small-order public keys, non-canonical S values per RFC 8032 §5.1.7). There is no separate `verify_strict` method — the only verifier is the strict one.

### Keypair (combined secret + public)

```rust
// Before — no first-class Keypair type in dalek 2.x; use SigningKey directly.
let signing_key = SigningKey::from_bytes(&seed);
let pk = signing_key.verifying_key();
```

```rust
// After
use rscrypto::Ed25519Keypair;
let kp = Ed25519Keypair::from_secret_key(Ed25519SecretKey::from_bytes(seed));
let sig = kp.sign(b"message");
```

`Ed25519Keypair` caches the expanded-secret form across calls — useful when you sign many messages with the same key.

### Cross-crate interoperability

You can sign with `ed25519-dalek` and verify with rscrypto (and vice versa) over the wire — the byte representation of `Signature` and `VerifyingKey` is identical. The harness verifies both directions.

```rust
// Sign with rscrypto, verify with dalek:
let us = rscrypto::Ed25519SecretKey::from_bytes(seed);
let sig = us.sign(msg);

let dalek_pk = ed25519_dalek::VerifyingKey::from_bytes(us.public_key().as_bytes())?;
let dalek_sig = ed25519_dalek::Signature::from_bytes(&sig.to_bytes());
dalek_pk.verify(msg, &dalek_sig)?;
```

## Notes

- **Strict verification is the default.** `ed25519-dalek::verify` (lax) accepts signatures using small-order public keys (which would let an attacker construct multi-key forgeries in some protocols). rscrypto removes this footgun — there is no lax verifier. If you previously called the lax `verify`, audit those call sites: they may be intentional (legacy compat) or a bug.
- **No batch verification.** `ed25519-dalek` ships `verify_batch` (with the `batch` feature) for 2–3× speedup when verifying many signatures at once. rscrypto does not expose batch verification yet. If you depend on it for transaction-validation throughput, file an issue or stay on `ed25519-dalek` for that path.
- **No prehashed (`Ed25519ph`) variant.** `ed25519-dalek` exposes the `Ed25519ph` prehashed variant for signing pre-hashed data. rscrypto only ships the standard `Ed25519` variant. If you need `Ed25519ph` for X.509 / TLS 1.3 hash-based signatures, file an issue.
- **`Signer` / `Verifier` trait imports not needed.** `ed25519-dalek` requires importing the `signature::Signer` and `Verifier` traits to call `sign` / `verify`. rscrypto's methods are inherent on the type — drop the trait imports.
- **`Drop` zeroizes the secret.** `Ed25519SecretKey: Drop` calls `ct::zeroize()` on the inner bytes. If you store a `SigningKey` by value, that storage is now scrubbed at drop without any extra `zeroize` annotation.
- **Byte-identical signatures.** Both crates implement RFC 8032 deterministically. The same `(seed, message)` pair produces the same 64-byte signature in either crate — your existing on-disk signatures verify under both implementations without re-signing.
- **`no_std`.** Both crates support `no_std`. rscrypto's Ed25519 path requires no `alloc`; signing and verification work on bare-metal targets with a stack-only allocator.
