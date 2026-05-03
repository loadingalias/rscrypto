# Migration: `scrypt` (RustCrypto) → `rscrypto`

> Replace the bare `scrypt(password, salt, &Params, &mut out)` function with `Scrypt::hash(&ScryptParams, password, salt, &mut out)`. Same RFC 7914 algorithm, byte-identical output, PHC string round-trip built in (no `password-hash` crate dependency).

Verified against `scrypt = "0.12.0"` and the `rscrypto` 0.1 line.

## TL;DR

| | Before (`scrypt` 0.12.x) | After (`rscrypto` 0.1) |
|---|---|---|
| Cargo dep | `scrypt = "0.12"` (+ `password-hash` for PHC) | `rscrypto = { version = "0.1", features = ["scrypt", "phc-strings"] }` |
| Import | `use scrypt::{scrypt, Params};` | `use rscrypto::{Scrypt, ScryptParams};` |
| Raw KDF | `scrypt(pw, salt, &Params::new(log_n, r, p)?, &mut out)?` | `Scrypt::hash(&ScryptParams::new()...build()?, pw, salt, &mut out)?` |

## Cargo.toml

```toml
# Before
[dependencies]
scrypt = "0.12"
password-hash = "0.5"            # only if you need PHC strings
```

```toml
# After
[dependencies]
rscrypto = { version = "0.1", features = ["scrypt", "phc-strings"] }
```

Drop `phc-strings` if you only need the raw KDF and not PHC `$scrypt$...$...` storage strings. Drop `getrandom` from the feature list if you supply your own salt.

## API patterns

### Raw KDF

```rust
// Before
use scrypt::{scrypt, Params};
let params = Params::new(15, 8, 1).unwrap();              // (log_n, r, p) — output len is the buffer's len
let mut out = [0u8; 32];
scrypt(password, salt, &params, &mut out).unwrap();
```

```rust
// After
use rscrypto::{Scrypt, ScryptParams};
let params = ScryptParams::new()
    .log_n(15)
    .r(8)
    .p(1)
    .output_len(32)                                          // explicit output length
    .build()?;                                               // validates ranges at build time
let mut out = [0u8; 32];
Scrypt::hash(&params, password, salt, &mut out)?;
```

Fixed-size convenience: `let out: [u8; 32] = Scrypt::hash_array(&params, password, salt)?;`.

### PHC encode (auto-generate salt)

```rust
// Before — requires the `password-hash` crate
use scrypt::{Scrypt, password_hash::{PasswordHasher, SaltString}};
use rand_core::OsRng;
let salt = SaltString::generate(&mut OsRng);
let phc = Scrypt
    .hash_password(password, &salt)?
    .to_string();
// phc: "$scrypt$ln=17,r=8,p=1$<salt>$<hash>"
```

```rust
// After (with `phc-strings` + `getrandom` features)
use rscrypto::{Scrypt, ScryptParams};
let phc: String = Scrypt::hash_string(&ScryptParams::default(), password)?;
```

### PHC encode (explicit salt)

```rust
// After (with `phc-strings`, no `getrandom` needed)
use rscrypto::{Scrypt, ScryptParams};
let phc = Scrypt::hash_string_with_salt(&ScryptParams::default(), password, salt)?;
```

### PHC verify (constant-time)

```rust
// Before — requires the `password-hash` crate
use scrypt::{Scrypt, password_hash::{PasswordHash, PasswordVerifier}};
let parsed = PasswordHash::new(stored_phc)?;
Scrypt.verify_password(password, &parsed)?;
```

```rust
// After
use rscrypto::Scrypt;
Scrypt::verify_string(password, stored_phc)?;
```

### PHC verify with policy bounds

When the stored PHC string came from an untrusted source, an attacker-supplied huge `ln=` (memory cost) would DoS your server. rscrypto exposes a runtime-policy check:

```rust
// After
use rscrypto::{Scrypt, ScryptVerifyPolicy};
let policy = ScryptVerifyPolicy::default();         // OWASP 2024 defaults
// or: ScryptVerifyPolicy::new(max_log_n, max_r, max_p, max_output_len)
Scrypt::verify_string_with_policy(password, stored_phc, &policy)?;
```

RustCrypto's `scrypt` has no equivalent policy gate — adding one externally requires parsing the PHC string with `password-hash`, rejecting high-cost cases, then re-validating. The built-in policy is a free hardening upgrade.

## Notes

- **`Params::new` is 3 args, not 4.** RustCrypto `scrypt = "0.12"` removed the explicit output-length parameter — output size is whatever buffer you pass to `scrypt(...)`. rscrypto restores it on the builder for cross-call safety (`output_len` is part of the `ScryptParams`, not the call site).
- **OWASP 2024 defaults.** `ScryptParams::default()` is `log_n=17 (N=131,072), r=8, p=1, output_len=32` — matches the OWASP Password Storage Cheat Sheet. RustCrypto's `Params::default()` matches.
- **`MIN_SALT_LEN` is policy, not enforcement.** rscrypto exposes `Scrypt::MIN_SALT_LEN = 16` as a guidance constant. Neither crate rejects shorter salts at hash time (RFC 7914 §12 test vectors include empty salts) — assert against the constant in your wrapper if you want enforcement.
- **PHC strings without `password-hash`.** rscrypto rolls its own PHC parser/encoder under `features = ["phc-strings"]`. The output format is identical and round-trips with `password-hash`-encoded strings.
- **Memory cost.** `Scrypt::hash` allocates `128 * r * N` bytes (default: ~128 MB at `log_n=17, r=8`). Use `ScryptVerifyPolicy::default()` to cap incoming PHC strings to OWASP-compliant bounds; otherwise an attacker submitting a `ln=24` PHC could request 16 GiB of allocation in your verify path.
- **No streaming API.** Scrypt is fundamentally one-shot — both crates expose only stateless functions.
- **`no_std` requires `alloc`.** Both crates need a heap for the working buffers (B, V, scratch). The deepest-embedded targets cannot run Scrypt — use `pbkdf2-hmac-sha256` with a high iteration count instead.
- **Algorithm is identical.** Bit-equivalent at every parameter set tested in the harness, including the RFC 7914 §12 first test vector (empty password, empty salt, N=16, r=1, p=1, output=64).
