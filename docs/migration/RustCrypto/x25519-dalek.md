# Migration: `x25519-dalek` → `rscrypto`

> Replace `StaticSecret` / `EphemeralSecret` / `PublicKey` / `SharedSecret` with rscrypto's unified `X25519SecretKey` / `X25519PublicKey` / `X25519SharedSecret`. Same RFC 7748 algorithm, byte-identical shared secrets, plus an explicit error on low-order peer input.

Verified against `x25519-dalek = "2.0.1"` and the `rscrypto` 0.5.0 line.
Evidence: `tests/x25519_vectors.rs`, `tests/x25519_oracle.rs`, and `tests/x25519_wycheproof.rs`.

## TL;DR

| | Before (`x25519-dalek` 2.x) | After (`rscrypto` 0.5.0) |
|---|---|---|
| Cargo dep | `x25519-dalek = { version = "2.0", features = ["static_secrets"] }` | `rscrypto = { version = "0.5.0", features = ["x25519"] }` |
| Import | `use x25519_dalek::{StaticSecret, PublicKey};` | `use rscrypto::{X25519SecretKey, X25519PublicKey};` |
| DH | `secret.diffie_hellman(&peer_pub)` (returns `SharedSecret`) | `secret.diffie_hellman(&peer_pub)?` (returns `Result<X25519SharedSecret, _>`) |

## Cargo.toml

```toml
# Before
[dependencies]
x25519-dalek = { version = "2.0", features = ["static_secrets"] }
```

```toml
# After
[dependencies]
rscrypto = { version = "0.5.0", features = ["x25519"] }
```

The `x25519` feature has no transitive dependencies: X25519 needs nothing beyond Curve25519 arithmetic.

## Type map

| `x25519-dalek` type | rscrypto type | Notes |
|---|---|---|
| `StaticSecret` | `X25519SecretKey` | reusable scalar; zeroizes on drop |
| `EphemeralSecret` | `X25519SecretKey` | rscrypto unifies the two: see "Ephemeral vs static" below |
| `PublicKey` | `X25519PublicKey` | 32-byte little-endian Montgomery u-coordinate |
| `SharedSecret` | `X25519SharedSecret` | zeroizes on drop |

## API patterns

### Construct a secret from seed bytes

```rust
// Before
use x25519_dalek::StaticSecret;
let secret = StaticSecret::from([0x42u8; 32]);              // From<[u8; 32]>; clamps internally
```

```rust
// After
use rscrypto::X25519SecretKey;
let secret = X25519SecretKey::from_bytes([0x42u8; 32]);     // const fn; clamps at use
```

Both crates accept any 32-byte input; both clamp the scalar per RFC 7748 §5 before performing the curve operation. Outputs are byte-identical (verified in the harness).

### Derive the public key

```rust
// Before
use x25519_dalek::{StaticSecret, PublicKey};
let public = PublicKey::from(&secret);
```

```rust
// After
use rscrypto::X25519SecretKey;
let public = secret.public_key();                            // inherent method; no PublicKey::from
```

### Diffie-Hellman exchange

```rust
// Before
let shared = secret.diffie_hellman(&peer_public);            // SharedSecret, infallible
let bytes: &[u8; 32] = shared.as_bytes();
```

```rust
// After
let shared = secret.diffie_hellman(&peer_public)?;           // Result<X25519SharedSecret, X25519Error>
let bytes: &[u8; 32] = shared.as_bytes();
```

rscrypto returns `Err(X25519Error)` if the computed shared secret is the all-zero point (RFC 7748 §6.1: indicates a low-order peer public key, often an attempted contributory-behaviour attack). `x25519-dalek` returns the all-zero `SharedSecret` and leaves the check to the caller. Re-audit your code for unchecked all-zero outputs and replace them with the `?`.

### Random key generation

```rust
// Before
use x25519_dalek::EphemeralSecret;
use rand_core::OsRng;
let secret = EphemeralSecret::random_from_rng(OsRng);
```

```rust
// After (with `getrandom` feature)
use rscrypto::X25519SecretKey;
let secret = X25519SecretKey::try_generate()?;               // Result<_, Error>

// Or supply your own fallible RNG via a closure (no extra feature):
let secret = X25519SecretKey::try_generate_with(|buf| fill_csprng(buf))?;
```

## Ephemeral vs static

`x25519-dalek` distinguishes `EphemeralSecret` (consumed by `diffie_hellman`, only one DH per secret) from `StaticSecret` (reusable, gated behind the `static_secrets` feature). The distinction is a type-level safety hint that ephemeral secrets should not be reused across handshakes.

rscrypto unifies both into `X25519SecretKey`. `diffie_hellman` borrows `&self`, so you can call it many times. `X25519SecretKey: Drop` zeroizes on drop, and you can model "one DH per secret" by wrapping in an `Option<X25519SecretKey>` and `.take()`-ing it in the handshake step.

If your protocol requires the ephemeral-only guarantee at the type level, file an issue requesting an `X25519EphemeralSecret` newtype.

## Notes

- **Clamping semantics match.** Both crates apply RFC 7748 §5 clamping (clear bits 0/1/2 of byte 0; clear bit 7 of byte 31; set bit 6 of byte 31) before each scalar multiplication. Outputs are bit-identical for any 32-byte input.
- **Low-order rejection: explicit vs implicit.** rscrypto's `Err(X25519Error)` on all-zero shared secret is the safer default because many protocols assume the shared secret is non-zero and skip the check, opening contributory-behaviour attacks. If you migrated *because* of an all-zero-shared-secret incident, the `?` propagation is exactly what you want.
- **`SharedSecret` zeroizes on drop.** Both crates do this. rscrypto exposes `expose_secret() -> SecretBytes<32>` if you need to feed the raw bytes into a KDF (and want the zeroization-on-drop guarantee preserved).
- **No HKDF integration.** Some protocols KDF the X25519 shared secret immediately (e.g., Noise, Signal). rscrypto's `HkdfSha256` (from `features = ["hkdf"]`) is the natural pairing; see `RustCrypto/hkdf.md`.
- **Constant-time scalar mult.** Both crates use constant-time field arithmetic. rscrypto's portable backend is constant-time on every target; SIMD acceleration (when available) is differential-tested against the portable path.
- **`no_std`.** Both crates support `no_std` with no `alloc` requirement. The `getrandom`-backed `try_generate()` requires `getrandom`; the closure form `try_generate_with(|buf| ...)` does not.
