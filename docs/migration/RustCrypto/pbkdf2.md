# Migration: `pbkdf2` (RustCrypto) → `rscrypto`

> Replace the free function `pbkdf2_hmac::<Sha256>(password, salt, iters, &mut out)` with `Pbkdf2Sha256::derive_key_array::<N>(password, salt, iters)?`. The password helpers enforce the current PBKDF2 iteration and salt floors by default, while `*_primitive` APIs remain available for RFC vectors and legacy compatibility.

Verified against `pbkdf2 = "0.13.0"` and the `rscrypto` 0.4.0 line.
Evidence: `tests/pbkdf2_kat_vectors.rs`, `tests/pbkdf2_differential.rs`, and `tests/pbkdf2_wycheproof.rs`.

## TL;DR

| | Before (`pbkdf2` 0.13.x) | After (`rscrypto` 0.4.0) |
|---|---|---|
| Cargo dep | `pbkdf2 = "0.13"` + `sha2 = "0.11"` | `rscrypto = { version = "0.4.0", features = ["pbkdf2"] }` |
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
rscrypto = { version = "0.4.0", features = ["pbkdf2"] }
```

The `pbkdf2` feature implies `hmac` which implies `sha2`.

## Algorithm map

| `pbkdf2` instantiation | rscrypto type | OWASP Password Storage Cheat Sheet minimum, checked 2026-06-09 |
|---|---|---|
| `pbkdf2_hmac::<Sha256>` | `Pbkdf2Sha256` | `Pbkdf2Sha256::MIN_RECOMMENDED_ITERATIONS` (600,000) |
| `pbkdf2_hmac::<Sha512>` | `Pbkdf2Sha512` | `Pbkdf2Sha512::MIN_RECOMMENDED_ITERATIONS` (220,000) |
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
// rejects weak iteration counts, short salts, and outputs > (2^32 - 1) * hLen
```

Fixed-size convenience:

```rust
// After
use rscrypto::Pbkdf2Sha256;
let key: [u8; 32] = Pbkdf2Sha256::derive_key_array(b"password", b"salt-16-bytes!!!", 600_000)?;
```

### Stateful: derive multiple keys from one password

When you derive several keys from the same password (e.g., encryption key + MAC key from the same vault password), the password's HMAC inner/outer key schedule is computed once and reused:

```rust
// After
use rscrypto::Pbkdf2Sha256;
let state = Pbkdf2Sha256::new(b"password");
let enc_params = Pbkdf2Sha256::params(b"salt-enc-16-byte", 600_000)?;
let mac_params = Pbkdf2Sha256::params(b"salt-mac-16-byte", 600_000)?;
let k_enc: [u8; 32] = state.derive_array_with_params(enc_params)?;
let k_mac: [u8; 32] = state.derive_array_with_params(mac_params)?;
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

Drop the `subtle` dependency for the verify path. The stateful form is `state.verify(salt, iters, &expected)`, which applies the same default policy.

## Notes

- **No PHC string format.** Both crates compute raw bytes. If you store `$pbkdf2-sha256$i=600000$salt$hash`-style PHC strings, use `pbkdf2::password_hash` (RustCrypto) or rscrypto's `phc-strings` feature with `Pbkdf2*` (forthcoming integration). For now, the raw-bytes path is byte-equivalent and storing the parts separately works.
- **Password helpers reject weak parameters.** `derive_key`, `derive_key_array`, `verify_password`, and stateful `verify` enforce the type-specific minimum iteration count and a 16-byte salt by default. `Pbkdf2Sha256::derive_key_primitive` / `verify_password_primitive` preserve raw PBKDF2 behavior for test vectors and explicit migrations.
- **Rejects zero iterations.** `pbkdf2 = "0.13"` will silently run zero rounds and return the salt-derived initial state. rscrypto returns `Err(Pbkdf2Error::InvalidIterations)`. If you have legacy callers that pass 0 (presumably as a typo), this is a hard catch — exactly what you want.
- **Output length cap (RFC 8018 §5.2 step 1).** PBKDF2 limits output to `(2^32 - 1) * hLen`. `pbkdf2` does not check; rscrypto returns `Err(Pbkdf2Error::OutputTooLong)`. The cap is in the gigabytes — only relevant for adversarial inputs.
- **Policy override.** Use `Pbkdf2VerifyPolicy` and `params_with_policy` only when you have a deliberate migration policy. This keeps legacy acceptance explicit instead of making low-cost password verification the default.
- **Iteration recommendation.** `MIN_RECOMMENDED_ITERATIONS` constants (600,000 for SHA-256, 220,000 for SHA-512) reflect the OWASP Password Storage Cheat Sheet as checked on 2026-06-09. Bump these as the OWASP cheat sheet bumps; rscrypto will track.
- **`no_std`.** Both crates work in `no_std`.
