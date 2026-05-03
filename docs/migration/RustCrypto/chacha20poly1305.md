# Migration: `chacha20poly1305` (RustCrypto) → `rscrypto`

> Covers both `ChaCha20Poly1305` (96-bit nonce, RFC 8439) and `XChaCha20Poly1305` (192-bit nonce). Same algorithm, byte-identical ciphertext+tag; replace `Key<T>` / `Nonce` / `XNonce` / `Payload { msg, aad }` with `ChaCha20Poly1305Key` + `Nonce96` / `Nonce192` and a buffer-style API.

Verified against `chacha20poly1305 = "0.10.1"` and the `rscrypto` 0.1 line.

## TL;DR

| | Before (`chacha20poly1305` 0.10.x) | After (`rscrypto` 0.1) |
|---|---|---|
| Cargo dep | `chacha20poly1305 = "0.10"` | `rscrypto = { version = "0.1", features = ["chacha20poly1305", "xchacha20poly1305"] }` |
| Import | `use chacha20poly1305::{ChaCha20Poly1305, Key, Nonce, KeyInit, aead::{Aead, Payload}};` | `use rscrypto::{Aead, ChaCha20Poly1305, ChaCha20Poly1305Key, aead::Nonce96};` |
| Encrypt | `cipher.encrypt(nonce, Payload { msg, aad })?` | `cipher.encrypt(&nonce, aad, msg, &mut out)?` |

Drop `xchacha20poly1305` from the feature list if you don't use the 192-bit-nonce variant.

## Cargo.toml

```toml
# Before
[dependencies]
chacha20poly1305 = "0.10"
```

```toml
# After
[dependencies]
rscrypto = { version = "0.1", features = ["chacha20poly1305", "xchacha20poly1305"] }
```

## Algorithm map

| `chacha20poly1305` type | rscrypto type | Nonce |
|---|---|---|
| `ChaCha20Poly1305` (RFC 8439, 96-bit nonce) | `ChaCha20Poly1305` | `Nonce96` |
| `XChaCha20Poly1305` (extended, 192-bit nonce) | `XChaCha20Poly1305` | `Nonce192` |
| `ChaCha8Poly1305`, `ChaCha12Poly1305` (reduced rounds) | not currently mapped — file an issue if you need them |  |

## API patterns

### `ChaCha20Poly1305` combined encrypt

```rust
// Before
use chacha20poly1305::{ChaCha20Poly1305, Key, Nonce, KeyInit};
use chacha20poly1305::aead::{Aead, Payload};

let key = Key::from_slice(&[0u8; 32]);                 // Key is non-generic
let cipher = ChaCha20Poly1305::new(key);
let nonce = Nonce::from_slice(&[0u8; 12]);
let ct = cipher.encrypt(nonce, Payload { msg: plaintext, aad }).unwrap();
```

```rust
// After
use rscrypto::{Aead, ChaCha20Poly1305, ChaCha20Poly1305Key, aead::Nonce96};

let key = ChaCha20Poly1305Key::from_bytes([0u8; 32]);
let cipher = ChaCha20Poly1305::new(&key);
let nonce = Nonce96::from_bytes([0u8; 12]);
let mut ct = vec![0u8; plaintext.len() + 16];
cipher.encrypt(&nonce, aad, plaintext, &mut ct)?;
```

### `XChaCha20Poly1305` combined encrypt

```rust
// Before
use chacha20poly1305::{XChaCha20Poly1305, Key, XNonce, KeyInit};
use chacha20poly1305::aead::{Aead, Payload};

let key = Key::from_slice(&[0u8; 32]);
let cipher = XChaCha20Poly1305::new(key);
let nonce = XNonce::from_slice(&[0u8; 24]);
let ct = cipher.encrypt(nonce, Payload { msg: plaintext, aad }).unwrap();
```

```rust
// After
use rscrypto::{Aead, XChaCha20Poly1305, XChaCha20Poly1305Key, aead::Nonce192};

let key = XChaCha20Poly1305Key::from_bytes([0u8; 32]);
let cipher = XChaCha20Poly1305::new(&key);
let nonce = Nonce192::from_bytes([0u8; 24]);
let mut ct = vec![0u8; plaintext.len() + 16];
cipher.encrypt(&nonce, aad, plaintext, &mut ct)?;
```

The XChaCha variant uses `Nonce192` (24 bytes) — the only structural change from the IETF-nonce variant.

### Decrypt + tamper-detection

```rust
// After
let mut plaintext = vec![0u8; ct.len() - 16];
cipher.decrypt(&nonce, aad, &ct, &mut plaintext)?;
// Err(OpenError::Verification(_)) on tag mismatch.
```

### Detached (in-place)

```rust
// After
let mut buffer = plaintext.to_vec();
let tag = cipher.encrypt_in_place(&nonce, aad, &mut buffer)?;
// later:
cipher.decrypt_in_place(&nonce, aad, &mut buffer, &tag)?;
```

## Notes

- **`Key` is non-generic in `chacha20poly1305`.** `chacha20poly1305::Key` is `pub type Key = GenericArray<u8, U32>;` — no `Key::<ChaCha20Poly1305>::from_slice` turbofish needed (or accepted). The same applies to `Nonce` and `XNonce`. rscrypto's per-algorithm key types collapse the choice.
- **XChaCha = ChaCha + HChaCha key derivation.** `XChaCha20Poly1305` derives a sub-key from the key + first-16-bytes-of-nonce via HChaCha20, then runs IETF ChaCha20-Poly1305 on the remaining 8 bytes of nonce. Outputs are bit-identical between the two crates (verified at random nonces in the harness).
- **96-bit nonce risk: nonce reuse is catastrophic.** With ChaCha20-Poly1305 (96-bit nonce), random-nonce selection gives a `2^-32` collision after `2^32` messages. Use a deterministic counter, or migrate to XChaCha20-Poly1305 (192-bit nonce) where random nonces are safe to ~2^96 messages.
- **No `Payload`.** Same simplification as `aes-gcm.md` — positional `aad` and `msg`/`buffer` args.
- **`AeadInPlace` import not needed.** rscrypto exposes both combined and in-place shapes through the single `Aead` trait.
- **Software-only acceleration.** ChaCha20 has no hardware AES; both crates use SIMD ChaCha20 implementations. rscrypto runtime-dispatches between SSE2/AVX2/AVX-512 on x86_64 and NEON on aarch64; portable scalar fallback is constant-time and always available. Force portable via `RSCRYPTO_CHACHA20_POLY1305_FORCE=portable` (std only).
- **`no_std`.** Both crates support `no_std`.
