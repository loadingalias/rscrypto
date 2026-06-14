# Migration: `dryoc` -> `rscrypto`

`dryoc` follows libsodium's API shape. Migrate the primitive calls directly and
keep `dryoc` for libsodium protocol helpers such as `secretbox`, `box`, sealed
boxes, and secret streams.

The direct BLAKE2b, Ed25519, and X25519 examples below are covered by
`tests/migration_dryoc.rs`.

## Cargo.toml

```toml
# Before
dryoc = "0.8"

# After: choose only the primitive features you use
rscrypto = { version = "0.5.0", default-features = false, features = ["blake2b", "ed25519", "x25519", "argon2"] }
```

## Map

| `dryoc` use | rscrypto | Status |
|---|---|---|
| `crypto_generichash` BLAKE2b | `Blake2b256`, `Blake2b512`, `Blake2bParams` | Supported for matching output/keyed modes |
| `crypto_sign_*` Ed25519 | `Ed25519SecretKey`, `Ed25519PublicKey` | Supported |
| `crypto_scalarmult*` X25519 | `X25519SecretKey`, `X25519PublicKey` | Supported |
| `crypto_pwhash` Argon2id / Argon2i | `Argon2id`, `Argon2i` | Partial; map parameters explicitly |
| `secretbox`, `box`, sealed boxes, secretstream | none | Keep `dryoc` or a protocol crate |

## BLAKE2b Generic Hash

```rust
// Before
let mut out = [0u8; 32];
dryoc::classic::crypto_generichash::crypto_generichash(&mut out, data, None)?;
```

```rust
// After
use rscrypto::Blake2b256;

let out = Blake2b256::digest(data);
```

For keyed BLAKE2b:

```rust
// Before
let mut tag = [0u8; 32];
dryoc::classic::crypto_generichash::crypto_generichash(&mut tag, data, Some(key))?;
```

```rust
// After
use rscrypto::Blake2b256;

let tag = Blake2b256::keyed_digest(key, data);
```

Use `Blake2b512` for 64-byte outputs. Use `Blake2bParams` only when you need
salt, personalization, or runtime output length.

## Ed25519

```rust
// Before
use dryoc::classic::crypto_sign::{
  crypto_sign_detached,
  crypto_sign_seed_keypair,
  crypto_sign_verify_detached,
};

let (public, secret) = crypto_sign_seed_keypair(seed);
let mut signature = [0u8; 64];
crypto_sign_detached(&mut signature, message, &secret)?;
crypto_sign_verify_detached(&signature, message, &public)?;
```

```rust
// After
use rscrypto::Ed25519SecretKey;

let secret = Ed25519SecretKey::from_bytes(*seed);
let public = secret.public_key();
let signature = secret.sign(message);
public.verify(message, &signature)?;
```

## X25519

```rust
// Before
use dryoc::classic::crypto_core::{crypto_scalarmult, crypto_scalarmult_base};

let mut public = [0u8; 32];
crypto_scalarmult_base(&mut public, secret);

let mut shared = [0u8; 32];
crypto_scalarmult(&mut shared, secret, peer_public);
```

```rust
// After
use rscrypto::{X25519PublicKey, X25519SecretKey};

let secret = X25519SecretKey::from_bytes(*secret);
let public = secret.public_key();

let peer = X25519PublicKey::from_bytes(*peer_public);
let shared = secret.diffie_hellman(&peer)?;
```

rscrypto returns an error for all-zero shared secrets. Handle that error instead
of accepting a zero output.

## Argon2

`dryoc::classic::crypto_pwhash` takes libsodium-style `opslimit`, `memlimit`
bytes, and algorithm constants. rscrypto takes explicit `Argon2Params`
(`memory_cost_kib`, `time_cost`, `parallelism`, `output_len`).

Do not copy parameters blindly. Convert `memlimit` from bytes to KiB, choose the
variant (`Argon2id` or `Argon2i`), and document the resulting policy at the call
site. Keep `dryoc` if your application depends on libsodium preset semantics.

## Keep `dryoc`

Keep `dryoc` or a protocol-specific crate for:

- `secretbox`, `box`, sealed boxes, and secret streams.
- libsodium-compatible password-hashing presets.
- Applications that need dryoc's higher-level keypair/container types.
