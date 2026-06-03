# Migration: `pbkdf2` (RustCrypto) → `rscrypto`

> Replace the free function `pbkdf2_hmac::<Sha256>(password, salt, iters, &mut out)` with `Pbkdf2Sha256::derive_key_array::<N>(password, salt, iters)?`. Adds a fallible result (rejects 0 iterations and over-long output), a stateful precompute when you derive multiple keys from the same password, and a constant-time `verify_password` helper.

Verified against `pbkdf2 = "0.13.0"` and the `rscrypto` 0.3.1 line.

## TL;DR

| | Before (`pbkdf2` 0.13.x) | After (`rscrypto` 0.3.1) |
|---|---|---|
| Cargo dep | `pbkdf2 = "0.13"` + `sha2 = "0.11"` | `rscrypto = { version = "0.3.1", features = ["pbkdf2"] }` |
| Import | `use pbkdf2::pbkdf2_hmac; use sha2::Sha256;` | `use rscrypto::Pbkdf2Sha256;` |
| Call | `pbkdf2_hmac::<Sha256>(pw, salt, iters, &mut okm)` | `Pbkdf2Sha256::derive_key(pw, salt, iters, &mut okm)?` |

## Cargo.toml

```toml
# Before
[dependencies]
pbkdf2 = "0.13"
sha2 = "0.11"
```

```toml
# After
[dependencies]
rscrypto = { version = "0.3.1", features = ["pbkdf2"] }
```

The `pbkdf2` feature implies `hmac` which implies `sha2`.

## Algorithm map

| `pbkdf2` instantiation | rscrypto type | OWASP min iters (2026) |
|---|---|---|
| `pbkdf2_hmac::<Sha256>` | `Pbkdf2Sha256` | `Pbkdf2Sha256::MIN_RECOMMENDED_ITERATIONS` (600,000) |
| `pbkdf2_hmac::<Sha512>` | `Pbkdf2Sha512` | `Pbkdf2Sha512::MIN_RECOMMENDED_ITERATIONS` (210,000) |
| `pbkdf2_hmac::<Sha1>` | not mapped — SHA-1 deprecated for KDF since 2010 |  |

## API patterns

### One-shot derivation

```rust
// Before
use pbkdf2::pbkdf2_hmac;
use sha2::Sha256;
let mut okm = [0u8; 32];
pbkdf2_hmac::<Sha256>(b"password", b"salt-16-bytes!!!", 600_000, &mut okm);
// no Result — silently produces output for any iteration count, including 0
```

```rust
// After
use rscrypto::Pbkdf2Sha256;
let mut okm = [0u8; 32];
Pbkdf2Sha256::derive_key(b"password", b"salt-16-bytes!!!", 600_000, &mut okm)?;
// rejects 0 iterations and outputs > (2^32 - 1) * hLen
```

Fixed-size convenience:

```rust
// After
use rscrypto::Pbkdf2Sha256;
let key: [u8; 32] = Pbkdf2Sha256::derive_key_array(b"password", b"salt", 600_000)?;
```

### Stateful: derive multiple keys from one password

When you derive several keys from the same password (e.g., encryption key + MAC key from the same vault password), the password's HMAC inner/outer key schedule is computed once and reused:

```rust
// After
use rscrypto::Pbkdf2Sha256;
let state = Pbkdf2Sha256::new(b"password");
let k_enc: [u8; 32] = state.derive_array(b"salt-enc", 600_000)?;
let k_mac: [u8; 32] = state.derive_array(b"salt-mac", 600_000)?;
```

`pbkdf2 = "0.13"` does not expose the precompute as a public type — every `pbkdf2_hmac` call rebuilds the schedule. Migrating to `Pbkdf2Sha256::new(...)` is a free perf win for multi-key derivations.

### Constant-time password verification

```rust
// Before
use pbkdf2::pbkdf2_hmac;
use sha2::Sha256;
use subtle::ConstantTimeEq;
let mut got = [0u8; 32];
pbkdf2_hmac::<Sha256>(submitted_password, &stored_salt, stored_iters, &mut got);
let ok: bool = got.ct_eq(&stored_hash).into();
```

```rust
// After
use rscrypto::Pbkdf2Sha256;
Pbkdf2Sha256::verify_password(submitted_password, &stored_salt, stored_iters, &stored_hash)?;
// Ok(()) on match, Err(VerificationError) on mismatch — both via constant-time compare
```

Drop the `subtle` dependency for the verify path. The stateful form is `state.verify(salt, iters, &expected)`.

## Notes

- **No PHC string format.** Both crates compute raw bytes. If you store `$pbkdf2-sha256$i=600000$salt$hash`-style PHC strings, use `pbkdf2::password_hash` (RustCrypto) or rscrypto's `phc-strings` feature with `Pbkdf2*` (forthcoming integration). For now, the raw-bytes path is byte-equivalent and storing the parts separately works.
- **Rejects zero iterations.** `pbkdf2 = "0.13"` will silently run zero rounds and return the salt-derived initial state. rscrypto returns `Err(Pbkdf2Error::InvalidIterations)`. If you have legacy callers that pass 0 (presumably as a typo), this is a hard catch — exactly what you want.
- **Output length cap (RFC 8018 §5.2 step 1).** PBKDF2 limits output to `(2^32 - 1) * hLen`. `pbkdf2` does not check; rscrypto returns `Err(Pbkdf2Error::OutputTooLong)`. The cap is in the gigabytes — only relevant for adversarial inputs.
- **Minimum salt length.** rscrypto exposes `Pbkdf2Sha256::MIN_SALT_LEN = 16` as a guidance constant; not enforced at runtime (RFC 8018 §4 recommends ≥ 8 bytes; OWASP 2026 recommends 16). Use it as a `const` check in your wrapper.
- **Iteration recommendation.** `MIN_RECOMMENDED_ITERATIONS` constants (600,000 for SHA-256, 210,000 for SHA-512) reflect OWASP 2026 guidance. Bump these as the OWASP cheat sheet bumps; rscrypto will track.
- **`no_std`.** Both crates work in `no_std`.
