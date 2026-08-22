# Migration: `ed25519-dalek` → `rscrypto`

> Replace `SigningKey` / `VerifyingKey` / `Signature` with
> `Ed25519SecretKey` / `Ed25519PublicKey` / `Ed25519Signature`. Signing
> preserves RFC 8032 output bytes, and the rscrypto verifier is always strict.

Verified against `ed25519-dalek = "3.0.0"` and the `rscrypto` 0.8.1 line.
Evidence: `tests/ed25519_rfc8032_vectors.rs`, `tests/ed25519_oracle.rs`, and `tests/ed25519_wycheproof.rs`.

## TL;DR

|           | Before (`ed25519-dalek` 3.x)                         | After (`rscrypto` 0.8.1)                                                |
| --------- | ---------------------------------------------------- | ----------------------------------------------------------------------- |
| Cargo dep | `ed25519-dalek = "3.0"`                              | `rscrypto = { version = "0.8.1", features = ["ed25519"] }`              |
| Import    | `use ed25519_dalek::{SigningKey, Signer, Verifier};` | `use rscrypto::{Ed25519SecretKey, Ed25519PublicKey, Ed25519Signature};` |
| Sign      | `signing_key.sign(msg)`                              | `secret.sign(msg)`                                                      |
| Verify    | `verifying_key.verify_strict(msg, &sig)?`            | `public_key.verify(msg, &sig)?`                                         |

## Cargo.toml

```toml
# Before
[dependencies]
ed25519-dalek = "3.0"
```

```toml
# After
[dependencies]
rscrypto = { version = "0.8.1", features = ["ed25519"] }
```

The `ed25519` feature implies `sha2` (Ed25519 uses SHA-512 internally per RFC 8032).

## Type map

| `ed25519-dalek` type | rscrypto type      | Bytes                           |
| -------------------- | ------------------ | ------------------------------- |
| `SigningKey`         | `Ed25519SecretKey` | `[u8; 32]` (seed)               |
| `VerifyingKey`       | `Ed25519PublicKey` | `[u8; 32]`                      |
| `Signature`          | `Ed25519Signature` | `[u8; 64]`                      |
| (implicit pair)      | `Ed25519Keypair`   | secret + public + expanded form |

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

`Ed25519SecretKey::from_bytes` is a `const fn` and can be used in const
contexts. That does not make embedding a long-lived signing key in a binary
safe.

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
let secret = Ed25519SecretKey::try_generate()?;          // returns Result<_, Error>

// Or supply your own fallible RNG via a closure (no extra feature):
let secret = Ed25519SecretKey::try_generate_with(|buf| fill_csprng(buf))?;
```

The closure form is the no-`getrandom` path: useful when you have a different entropy source (HSM, TPM, embedded TRNG).

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

`secret.sign(...)` is inherent in rscrypto: no `Signer` trait import needed. Ed25519 is deterministic (RFC 8032 §5.1.6); the same `(seed, message)` pair always produces the same signature in both crates (verified in the harness).

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
public_key.verify(b"message", &sig)?;                       // strict by default: there is no lax mode
```

rscrypto's `verify` is _always_ strict (rejects small-order public keys, non-canonical S values per RFC 8032 §5.1.7). There is no separate `verify_strict` method; the only verifier is the strict one.

### Keypair (combined secret + public)

```rust
// Before: no first-class Keypair type in dalek 3.x; use SigningKey directly.
let signing_key = SigningKey::from_bytes(&seed);
let pk = signing_key.verifying_key();
```

```rust
// After
use rscrypto::Ed25519Keypair;
let kp = Ed25519Keypair::from_secret_key(Ed25519SecretKey::from_bytes(seed));
let sig = kp.sign(b"message");
```

`Ed25519Keypair` caches the expanded-secret form across calls: useful when you sign many messages with the same key.

### Cross-crate interoperability

You can sign with `ed25519-dalek` and verify with rscrypto (and vice versa) over the wire; the byte representation of `Signature` and `VerifyingKey` is identical. The harness verifies both directions.

```rust
// Sign with rscrypto, verify with dalek:
let us = rscrypto::Ed25519SecretKey::from_bytes(seed);
let sig = us.sign(msg);

let dalek_pk = ed25519_dalek::VerifyingKey::from_bytes(us.public_key().as_bytes())?;
let dalek_sig = ed25519_dalek::Signature::from_bytes(&sig.to_bytes());
dalek_pk.verify(msg, &dalek_sig)?;
```

## Notes

- **Strict verification is the only policy.** rscrypto has no counterpart to
  `ed25519_dalek::VerifyingKey::verify`; it maps
  `VerifyingKey::verify_strict` to `Ed25519PublicKey::verify`. If the old code
  used `verify`, test existing signatures against the stricter acceptance
  boundary before switching.
- **No batch verification.** `ed25519-dalek` exposes `verify_batch` behind its
  `batch` feature. Keep `ed25519-dalek` for call sites that require a batch API.
- **No prehashed or contextual variant.** rscrypto exposes standard Ed25519,
  not Ed25519ph or its context-bearing API. Keep `ed25519-dalek` when the
  protocol specifically requires those RFC 8032 variants.
- **`Signer` / `Verifier` trait imports not needed.** `ed25519-dalek` requires importing the `signature::Signer` and `Verifier` traits to call `sign` / `verify`. rscrypto's methods are inherent on the type: drop the trait imports.
- **Secret drop behavior.** `Ed25519SecretKey::drop` overwrites its owned seed
  through rscrypto's zeroization primitive. The claim is limited to that owned
  storage and the evidence boundary in
  [`docs/secret-lifecycle.md`](../../secret-lifecycle.md).
- **Byte-identical signatures.** Both crates implement RFC 8032 deterministically. The same `(seed, message)` pair produces the same 64-byte signature in either crate: your existing on-disk signatures verify under both implementations without re-signing.
- **`no_std`.** Both crates support `no_std`. rscrypto's Ed25519 signing and verification paths do not require `alloc`.
