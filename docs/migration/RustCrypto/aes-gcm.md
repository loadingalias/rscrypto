# Migration: `aes-gcm` (RustCrypto) → `rscrypto`

> Replace the `Aes256Gcm` / `Key<Aes256Gcm>` / `Nonce` / `Payload { msg, aad }` builder with rscrypto's named types and a buffer-style `encrypt(&nonce, aad, plaintext, &mut out)`. Same algorithm, byte-identical ciphertext+tag, no `Vec` allocations on the hot path.

Verified against `aes-gcm = "0.10.3"` and the `rscrypto` 0.1 line.

## TL;DR

| | Before (`aes-gcm` 0.10.x) | After (`rscrypto` 0.1) |
|---|---|---|
| Cargo dep | `aes-gcm = "0.10"` | `rscrypto = { version = "0.1", features = ["aes-gcm"] }` |
| Import | `use aes_gcm::{Aes256Gcm, Key, Nonce, KeyInit, aead::{Aead, Payload}};` | `use rscrypto::{Aead, Aes256Gcm, Aes256GcmKey, aead::Nonce96};` |
| Encrypt | `cipher.encrypt(nonce, Payload { msg, aad })?` (returns `Vec<u8>`) | `cipher.encrypt(&nonce, aad, msg, &mut out)?` (writes into caller buffer) |

## Cargo.toml

```toml
# Before
[dependencies]
aes-gcm = "0.10"
```

```toml
# After
[dependencies]
rscrypto = { version = "0.1", features = ["aes-gcm"] }
```

## Algorithm map

| `aes-gcm` type | rscrypto type | Key bytes |
|---|---|---|
| `Aes128Gcm` | `Aes128Gcm` | 16 |
| `Aes256Gcm` | `Aes256Gcm` | 32 |

Both variants share the same typed surface (`Aes128GcmKey`/`Aes256GcmKey`,
`Nonce96`, `Aes128GcmTag`/`Aes256GcmTag`) and the same `Aead` trait.
The migration recipe below uses `Aes256Gcm` throughout; substitute
`Aes128Gcm` and the matching 16-byte key type for AES-128-GCM use.

## API patterns

### Combined encrypt (tag appended to ciphertext)

```rust
// Before
use aes_gcm::{Aes256Gcm, Key, Nonce, KeyInit};
use aes_gcm::aead::{Aead, Payload};

let key = Key::<Aes256Gcm>::from_slice(&[0u8; 32]);
let cipher = Aes256Gcm::new(key);
let nonce = Nonce::from_slice(&[0u8; 12]);
let ct = cipher.encrypt(nonce, Payload { msg: plaintext, aad }).unwrap();
// ct: Vec<u8>, last 16 bytes are the tag.
```

```rust
// After
use rscrypto::{Aead, Aes256Gcm, Aes256GcmKey, aead::Nonce96};

let key = Aes256GcmKey::from_bytes([0u8; 32]);
let cipher = Aes256Gcm::new(&key);
let nonce = Nonce96::from_bytes([0u8; 12]);
let mut ct = vec![0u8; plaintext.len() + 16];
cipher.encrypt(&nonce, aad, plaintext, &mut ct)?;
// ct[..plaintext.len()] is ciphertext, ct[plaintext.len()..] is the 16-byte tag.
```

The output layout is identical (`[ciphertext || tag]`), so on-the-wire compatibility is preserved. The shape change is who owns the buffer — `aes-gcm` allocates a `Vec`, rscrypto writes into a buffer you pre-sized.

### Combined decrypt

```rust
// Before
let plaintext = cipher
    .decrypt(nonce, Payload { msg: &ct, aad })
    .unwrap();
```

```rust
// After
let mut plaintext = vec![0u8; ct.len() - 16];
cipher.decrypt(&nonce, aad, &ct, &mut plaintext)?;
```

Decrypt returns `Err(OpenError::Verification(_))` on tag mismatch — opaque, no detail leaked.

### Detached (in-place) encrypt

```rust
// Before
use aes_gcm::aead::AeadInPlace;
let mut buffer = plaintext.to_vec();
let tag = cipher
    .encrypt_in_place_detached(nonce, aad, &mut buffer)
    .unwrap();
// buffer is now the ciphertext; tag is a 16-byte GenericArray.
```

```rust
// After
let mut buffer = plaintext.to_vec();
let tag = cipher.encrypt_in_place(&nonce, aad, &mut buffer)?;
// buffer is now the ciphertext; tag: Aes256GcmTag (Copy).
```

`encrypt_in_place` is the canonical name in rscrypto; `encrypt_in_place_detached` is also available as an alias matching RustCrypto's naming.

### Detached (in-place) decrypt

```rust
// Before
cipher
    .decrypt_in_place_detached(nonce, aad, &mut buffer, &tag)
    .unwrap();
```

```rust
// After
cipher.decrypt_in_place(&nonce, aad, &mut buffer, &tag)?;
// Returns Err(OpenError::Verification(_)) on tag mismatch; buffer state on
// failure is unspecified — treat the buffer as garbage and discard.
```

## Notes

- **`Key<T>` is gone, `Nonce` is gone.** Use the per-algorithm `Aes256GcmKey` and the size-specific `Nonce96`. No `KeyInit` trait import; `Aes256Gcm::new(&key)` is inherent.
- **`Payload { msg, aad }` is gone.** Pass `aad` and `msg` (or `aad` and the in-place buffer) as positional args. AAD is a plain `&[u8]`; pass `b""` for "no AAD".
- **`finalize` consumes vs. borrows: not applicable.** AEAD ciphers in both crates are stateless across calls; the cipher type is `Clone` and you can encrypt many messages with the same `cipher` value, supplying a fresh nonce each time.
- **Nonce reuse is catastrophic for AES-GCM.** Both crates accept any `Nonce96` — there's no enforcement against duplicate nonces. If you migrated from `aes-gcm` because of a nonce-reuse incident, switch to `Aes256GcmSiv` (see `aes-gcm-siv.md`); it tolerates accidental reuse.
- **Random nonces are unsafe at scale.** With 96-bit nonces, the collision probability after `2^32` messages is around `2^-32`. For random-nonce flows, switch to `XChaCha20Poly1305` (192-bit nonces) — see `chacha20poly1305.md`.
- **`AeadInPlace` trait import not needed.** RustCrypto requires importing `aead::AeadInPlace` separately to call the `_in_place_detached` methods. rscrypto exposes both shapes through the single `Aead` trait.
- **`generic-array` is gone.** rscrypto does not return `GenericArray` from any AEAD method. Tags are typed newtypes (`Aes256GcmTag`) wrapping `[u8; 16]`; key/nonce types wrap `[u8; N]` directly.
- **Hardware acceleration.** Both crates dispatch to AES-NI on x86_64 and AES-CE on aarch64. rscrypto adds VAES (AVX-512), s390x CPACF, and a portable constant-time bitsliced fallback. Force the portable kernel via `RSCRYPTO_AES_GCM_FORCE=portable` (std only) or the crate's `portable-only` feature.
- **`no_std`.** Both crates support `no_std`. rscrypto's combined API requires the caller to provide an output buffer — naturally fits stack-only embedded use. The `vec!` calls in the examples above are for std convenience; in `no_std` they become fixed-size arrays.
