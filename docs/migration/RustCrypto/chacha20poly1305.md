# Migration: `chacha20poly1305` (RustCrypto) → `rscrypto`

> Replace `Key<T>` / `Nonce` / `XNonce` / `Payload { msg, aad }` with named
> keys, `Nonce96` or `Nonce192`, and a caller-buffer API. The mapped
> ChaCha20-Poly1305 and XChaCha20-Poly1305 operations preserve ciphertext and
> tag bytes.

Verified against `chacha20poly1305 = "0.11.0"` and the `rscrypto` 0.8.1 line.
Evidence: `tests/chacha20poly1305.rs`, `tests/xchacha20poly1305.rs`, and `tests/aead_wycheproof.rs`.

## TL;DR

|           | Before (`chacha20poly1305` 0.11.x)                                                      | After (`rscrypto` 0.8.1)                                                                               |
| --------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| Cargo dep | `chacha20poly1305 = "0.11"`                                                             | `rscrypto = { version = "0.8.1", features = ["chacha20poly1305", "xchacha20poly1305"] }`               |
| Import    | `use chacha20poly1305::{ChaCha20Poly1305, Key, Nonce, KeyInit, aead::{Aead, Payload}};` | `use rscrypto::{Aead, ChaCha20Poly1305, ChaCha20Poly1305Key, aead::{Nonce96, expert::AeadWithNonce}};` |
| Encrypt   | `cipher.encrypt(nonce, Payload { msg, aad })?`                                          | `cipher.encrypt(&nonce, aad, msg, &mut out)?`                                                          |

Drop `xchacha20poly1305` from the feature list if you don't use the 192-bit-nonce variant.

## Cargo.toml

```toml
# Before
[dependencies]
chacha20poly1305 = "0.11"
```

```toml
# After
[dependencies]
rscrypto = { version = "0.8.1", features = ["chacha20poly1305", "xchacha20poly1305"] }
```

## Algorithm map

| `chacha20poly1305` type                                | rscrypto type                       | Nonce      |
| ------------------------------------------------------ | ----------------------------------- | ---------- |
| `ChaCha20Poly1305` (RFC 8439, 96-bit nonce)            | `ChaCha20Poly1305`                  | `Nonce96`  |
| `XChaCha20Poly1305` (extended, 192-bit nonce)          | `XChaCha20Poly1305`                 | `Nonce192` |
| `ChaCha8Poly1305`, `ChaCha12Poly1305` (reduced rounds) | not mapped: keep `chacha20poly1305` |            |

## API patterns

### `ChaCha20Poly1305` combined encrypt

```rust
// Before
use chacha20poly1305::{ChaCha20Poly1305, Key, Nonce, KeyInit};
use chacha20poly1305::aead::{Aead, Payload};

let key = Key::from([0u8; 32]);                        // Key is non-generic
let cipher = ChaCha20Poly1305::new(&key);
let nonce = Nonce::from([0u8; 12]);
let ct = cipher.encrypt(&nonce, Payload { msg: plaintext, aad }).unwrap();
```

```rust
// After
use rscrypto::{
  Aead, ChaCha20Poly1305, ChaCha20Poly1305Key,
  aead::{Nonce96, expert::AeadWithNonce},
};

let key = ChaCha20Poly1305Key::from_bytes([0u8; 32]);
let cipher = ChaCha20Poly1305::new(&key);
let nonce = Nonce96::from_bytes([0u8; 12]);
let mut ct = vec![0u8; ChaCha20Poly1305::ciphertext_len(plaintext.len())?];
cipher.encrypt(&nonce, aad, plaintext, &mut ct)?;
```

### `XChaCha20Poly1305` combined encrypt

```rust
// Before
use chacha20poly1305::{XChaCha20Poly1305, Key, XNonce, KeyInit};
use chacha20poly1305::aead::{Aead, Payload};

let key = Key::from([0u8; 32]);
let cipher = XChaCha20Poly1305::new(&key);
let nonce = XNonce::from([0u8; 24]);
let ct = cipher.encrypt(&nonce, Payload { msg: plaintext, aad }).unwrap();
```

```rust
// After
use rscrypto::{
  Aead, XChaCha20Poly1305, XChaCha20Poly1305Key,
  aead::{Nonce192, expert::AeadWithNonce},
};

let key = XChaCha20Poly1305Key::from_bytes([0u8; 32]);
let cipher = XChaCha20Poly1305::new(&key);
let nonce = Nonce192::from_bytes([0u8; 24]);
let mut ct = vec![0u8; XChaCha20Poly1305::ciphertext_len(plaintext.len())?];
cipher.encrypt(&nonce, aad, plaintext, &mut ct)?;
```

The XChaCha variant uses `Nonce192` (24 bytes). That is the only structural change from the IETF-nonce variant.
The expert trait import is required because these examples preserve an
existing caller-nonce protocol. Prefer `seal_random` for new protocols.

### ChaCha20-Poly1305 decrypt and tamper detection

```rust
// After
let mut plaintext = vec![0u8; ChaCha20Poly1305::plaintext_len(ct.len())?];
cipher.decrypt(&nonce, aad, &ct, &mut plaintext)?;
// Err(OpenError::Verification(_)) on tag mismatch.
```

Use `XChaCha20Poly1305::plaintext_len` for the XChaCha variant.

### Detached (in-place)

```rust
// After
let mut buffer = plaintext.to_vec();
let tag = cipher.encrypt_in_place(&nonce, aad, &mut buffer)?;
// later:
cipher.decrypt_in_place(&nonce, aad, &mut buffer, &tag)?;
```

## Notes

- **`Key` is non-generic in `chacha20poly1305`.** `chacha20poly1305::Key` is `pub type Key = GenericArray<u8, U32>;`: no `Key::<ChaCha20Poly1305>::from_slice` turbofish needed (or accepted). The same applies to `Nonce` and `XNonce`. rscrypto's per-algorithm key types collapse the choice.
- **XChaCha = ChaCha + HChaCha key derivation.** `XChaCha20Poly1305` derives a sub-key from the key + first-16-bytes-of-nonce via HChaCha20, then runs IETF ChaCha20-Poly1305 on the remaining 8 bytes of nonce. Outputs are bit-identical between the two crates (verified at random nonces in the harness).
- **Nonce reuse is catastrophic.** For uniformly random `n`-bit nonces and
  `q` encryptions, the birthday approximation is `q(q-1) / 2^(n+1)`.
  ChaCha20-Poly1305 therefore reaches about `2^-33` collision probability at
  `q = 2^32`. Prefer deterministic uniqueness for its 96-bit nonce.
  XChaCha20-Poly1305's 192-bit nonce space reduces random-collision risk, but
  the deployment must still define a message limit and acceptable error
  probability.
- **No `Payload`.** Same simplification as `aes-gcm.md`: positional `aad` and `msg`/`buffer` args.
- **Explicit-nonce sealing is expert-only.** Import `AeadWithNonce` for
  protocols that already prove nonce uniqueness. Decryption and fresh-random
  sealing remain on `Aead`.
- **Failed-open buffer semantics change.** RustCrypto keeps the in-place buffer
  unchanged on error. rscrypto clears it on authentication failure. Combined
  rscrypto decrypt also clears its output buffer on authentication failure.
- **Acceleration.** rscrypto selects eligible vector or assembly backends from
  detected CPU capabilities and retains a portable scalar fallback. The
  fallback has fixed-work source structure, but generated-code constant-time
  coverage is limited to the compiler, target, features, and binary in the
  matching [release evidence](../../constant-time.md). The `portable-only`
  feature constrains runtime dispatch as documented in
  [`docs/features.md`](../../features.md#portable-only).
- **`no_std`.** Both crates support `no_std`.
