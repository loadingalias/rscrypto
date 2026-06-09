# Migration: `argon2` (RustCrypto) → `rscrypto`

> Replace the runtime-variant `Argon2::new(Algorithm::*, Version, Params)` with type-level `Argon2id` / `Argon2i` / `Argon2d`. Same RFC 9106 spec, byte-identical output, PHC string round-trip built in (no `password-hash` crate dependency).

Verified against `argon2 = "0.6.0-rc.8"` and the `rscrypto` 0.4.0 line.
Evidence: `tests/argon2_vectors.rs`, `tests/argon2_differential.rs`, `tests/argon2_kernels.rs`, `tests/argon2_parallel.rs`, and `tests/phc_roundtrip.rs`.
The examples use the 0.5-style call shape because that is still common in
existing projects; the current oracle coverage in this repository is the 0.6
RC dev-dependency.

## TL;DR

| | Before (`argon2` 0.5.x) | After (`rscrypto` 0.4.0) |
|---|---|---|
| Cargo dep | `argon2 = "0.5"` (+ `password-hash` for PHC) | `rscrypto = { version = "0.4.0", features = ["argon2", "phc-strings"] }` |
| Import | `use argon2::{Argon2, Algorithm, Version, Params};` | `use rscrypto::{Argon2id, Argon2Params, Argon2Version};` |
| Raw KDF | `Argon2::new(Argon2id, V0x13, Params::new(m, t, p, Some(N))?).hash_password_into(pw, salt, &mut out)?` | `Argon2id::hash(&Argon2Params::new()...build()?, pw, salt, &mut out)?` |

## Cargo.toml

```toml
# Before
[dependencies]
argon2 = "0.5"
password-hash = "0.5"            # only if you need PHC strings
```

```toml
# After
[dependencies]
rscrypto = { version = "0.4.0", features = ["argon2", "phc-strings"] }
```

Drop `phc-strings` if you only need the raw KDF and not PHC `$argon2id$...$...` storage strings. Drop `getrandom` from the feature list if you supply your own salt.

## Variant map

| `argon2::Algorithm` value | rscrypto type | Notes |
|---|---|---|
| `Algorithm::Argon2id` | `Argon2id` | hybrid; recommended default |
| `Algorithm::Argon2i` | `Argon2i` | data-independent; for side-channel-sensitive deployments |
| `Algorithm::Argon2d` | `Argon2d` | data-dependent; legacy / Filecoin / niche |

## API patterns

### Raw KDF

```rust
// Before
use argon2::{Algorithm, Argon2, Params, Version};
let params = Params::new(19_456, 2, 1, Some(32)).unwrap();   // (mem KiB, time, parallel, output)
let argon2 = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);
let mut out = [0u8; 32];
argon2.hash_password_into(password, salt, &mut out).unwrap();
```

```rust
// After
use rscrypto::{Argon2Params, Argon2Version, Argon2id};
let params = Argon2Params::new()
    .memory_cost_kib(19_456)
    .time_cost(2)
    .parallelism(1)
    .output_len(32)
    .version(Argon2Version::V0x13)
    .build()?;                                                // validates ranges at build time
let mut out = [0u8; 32];
Argon2id::hash(&params, password, salt, &mut out)?;
```

For switching variants, change the type — the `params` value is the same:

```rust
Argon2id::hash(&params, password, salt, &mut out)?;
Argon2i ::hash(&params, password, salt, &mut out)?;
Argon2d ::hash(&params, password, salt, &mut out)?;
```

Fixed-size convenience: `let out: [u8; 32] = Argon2id::hash_array(&params, password, salt)?;`.

### PHC encode (auto-generate salt)

```rust
// Before — requires the `password-hash` crate
use argon2::{Argon2, password_hash::{PasswordHasher, SaltString}};
use rand_core::OsRng;
let salt = SaltString::generate(&mut OsRng);
let phc = Argon2::default()
    .hash_password(password, &salt)?
    .to_string();
// phc: String like "$argon2id$v=19$m=19456,t=2,p=1$..."
```

```rust
// After (with `phc-strings` + `getrandom` features)
use rscrypto::{Argon2Params, Argon2id};
let phc: String = Argon2id::hash_string(&Argon2Params::default(), password)?;
```

### PHC encode (explicit salt)

```rust
// After (with `phc-strings`, no `getrandom` needed)
use rscrypto::{Argon2Params, Argon2id};
let phc = Argon2id::hash_string_with_salt(&Argon2Params::default(), password, salt)?;
```

### PHC verify (constant-time)

```rust
// Before — requires the `password-hash` crate
use argon2::{Argon2, password_hash::{PasswordHash, PasswordVerifier}};
let parsed = PasswordHash::new(stored_phc)?;
Argon2::default().verify_password(password, &parsed)?;
```

```rust
// After
use rscrypto::Argon2id;
Argon2id::verify_string(password, stored_phc)?;
```

### PHC verify with policy bounds

When the stored PHC string came from an untrusted source (user input, distributed cache, federated identity), an attacker-supplied huge `m=` would let them DoS your server with one verify call. rscrypto exposes a runtime-policy check:

```rust
// After
use rscrypto::{Argon2id, Argon2VerifyPolicy};
let policy = Argon2VerifyPolicy::default();          // OWASP Password Storage Cheat Sheet defaults
// or: Argon2VerifyPolicy::new(max_m_kib, max_t, max_p, max_output_len)
Argon2id::verify_string_with_policy(password, stored_phc, &policy)?;
```

`x25519-dalek`'s `argon2` crate has no equivalent policy gate — you'd have to parse the PHC string, reject high-cost cases, then re-encode. The built-in policy is a free hardening upgrade.

### Verify with secret / associated data (pepper)

```rust
// After
use rscrypto::Argon2id;
Argon2id::verify_string_with_context(password, stored_phc, secret, associated_data)?;
```

The secret (pepper) and associated data are not embedded in the PHC string — they live in your application config / KMS. The `with_context` helpers thread them through verification. RustCrypto's `argon2` exposes the same on the `Argon2` builder; the rscrypto helper collapses it into one call.

## Notes

- **Variant as type, not enum value.** `Argon2id::hash(...)` instead of `argon2.algorithm(Algorithm::Argon2id).hash_password_into(...)`. Enum-driven dispatch becomes type-driven; in tests this catches accidental variant swaps at compile time.
- **`Argon2VerifyPolicy` is the migration win.** RustCrypto requires hand-rolling the cost-cap check before calling `verify_password`. rscrypto bakes it in. Use `Argon2VerifyPolicy::default()` (OWASP Password Storage Cheat Sheet caps, checked 2026-06-09) unless you have a specific reason to relax.
- **PHC strings without `password-hash`.** rscrypto rolls its own PHC parser/encoder under `features = ["phc-strings"]`. The output format is identical and round-trips with `password-hash`-encoded strings — your existing stored hashes still verify.
- **`Argon2Params::default()` matches OWASP Password Storage Cheat Sheet guidance.** `m=19456 KiB, t=2, p=1, output_len=32 bytes` per the sheet as checked on 2026-06-09. RustCrypto's `Params::default()` matches; both crates will track the cheat sheet over time.
- **`v=0x10` decode-only support.** Both crates can decode legacy `v=16` (Argon2 v1.0) PHC strings for compatibility; both produce `v=19` (v1.3) on encode.
- **Memory cost is real.** `Argon2id::hash` allocates `memory_cost_kib * 1024` bytes for the lane state. `params.build()?` enforces `m >= 8 * p`; allocate failures (out-of-memory) surface as `Argon2Error::AllocationFailed` at hash time.
- **Parallel lanes (`parallel` feature).** rscrypto's `parallel` umbrella feature enables Rayon-based lane parallelism for `p > 1`. RustCrypto's `argon2` is single-threaded only.
- **`no_std` requires `alloc`.** Both crates need a heap for the memory matrix. The deepest-embedded targets (no `alloc`) cannot run Argon2 — use `pbkdf2-hmac-sha256` with a high iteration count instead.
