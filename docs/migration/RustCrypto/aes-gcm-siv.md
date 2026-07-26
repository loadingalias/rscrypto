# Migration: `aes-gcm-siv` (RustCrypto) → `rscrypto`

> Same algorithm (RFC 8452), same nonce-misuse-resistant guarantees. Replace `Aes256GcmSiv` / `Key<Aes256GcmSiv>` / `Nonce` / `Payload { msg, aad }` with rscrypto's named types and a buffer-style API.

Verified against `aes-gcm-siv = "0.11.1"` and the `rscrypto` 0.7.8 line.
Evidence: `tests/aes128gcmsiv_oracle.rs`, `tests/aes256gcmsiv_oracle.rs`, and `tests/aead_wycheproof.rs`.

## TL;DR

| | Before (`aes-gcm-siv` 0.11.x) | After (`rscrypto` 0.7.8) |
|---|---|---|
| Cargo dep | `aes-gcm-siv = "0.11"` | `rscrypto = { version = "0.7.8", features = ["aes-gcm-siv"] }` |
| Import | `use aes_gcm_siv::{Aes256GcmSiv, Key, Nonce, KeyInit, aead::{Aead, Payload}};` | `use rscrypto::{Aead, Aes256GcmSiv, Aes256GcmSivKey, aead::Nonce96};` |
| Encrypt | `cipher.encrypt(nonce, Payload { msg, aad })?` | `cipher.encrypt(&nonce, aad, msg, &mut out)?` |

## Cargo.toml

```toml
# Before
[dependencies]
aes-gcm-siv = "0.11"
```

```toml
# After
[dependencies]
rscrypto = { version = "0.7.8", features = ["aes-gcm-siv"] }
```

## Algorithm map

| `aes-gcm-siv` type | rscrypto type | Key bytes |
|---|---|---|
| `Aes128GcmSiv` | `Aes128GcmSiv` | 16 |
| `Aes256GcmSiv` | `Aes256GcmSiv` | 32 |

Both variants share the same typed surface (`Aes128GcmSivKey`/`Aes256GcmSivKey`,
`Nonce96`, `Aes128GcmSivTag`/`Aes256GcmSivTag`) and the same `Aead` trait.
The migration recipe below uses `Aes256GcmSiv` throughout; substitute
`Aes128GcmSiv` and the matching 16-byte key type for AES-128-GCM-SIV use.

## API patterns

### Combined encrypt

```rust
// Before
use aes_gcm_siv::{Aes256GcmSiv, Key, Nonce, KeyInit};
use aes_gcm_siv::aead::{Aead, Payload};

let key = Key::<Aes256GcmSiv>::from_slice(&[0u8; 32]);
let cipher = Aes256GcmSiv::new(key);
let nonce = Nonce::from_slice(&[0u8; 12]);
let ct = cipher.encrypt(nonce, Payload { msg: plaintext, aad }).unwrap();
```

```rust
// After
use rscrypto::{Aead, Aes256GcmSiv, Aes256GcmSivKey, aead::Nonce96};

let key = Aes256GcmSivKey::from_bytes([0u8; 32]);
let cipher = Aes256GcmSiv::new(&key);
let nonce = Nonce96::from_bytes([0u8; 12]);
let mut ct = vec![0u8; plaintext.len() + 16];
cipher.encrypt(&nonce, aad, plaintext, &mut ct)?;
```

### Combined decrypt

```rust
// After
let mut plaintext = vec![0u8; ct.len() - 16];
cipher.decrypt(&nonce, aad, &ct, &mut plaintext)?;
```

### Detached (in-place)

```rust
// After
let mut buffer = plaintext.to_vec();
let tag = cipher.encrypt_in_place(&nonce, aad, &mut buffer)?;
// ...
cipher.decrypt_in_place(&nonce, aad, &mut buffer, &tag)?;
```

## Notes

- **Why GCM-SIV?** Per RFC 8452, AES-GCM-SIV is nonce-misuse-resistant.
  Reuse does not expose the authentication key as it does in AES-GCM, but it
  still makes repeated plaintext/AAD under the same nonce observable and
  weakens the construction's security bounds. Continue to prefer unique
  nonces.
- **The API-shape and failed-open notes from `aes-gcm.md` apply.**
  `Payload { msg, aad }` collapses to positional arguments; `Nonce` becomes
  `Nonce96`; `Key<T>` becomes `Aes256GcmSivKey`; and combined output is
  `[ciphertext || tag]`, byte-identical to `aes-gcm-siv`'s `Vec<u8>`. The
  AES-GCM nonce-reuse warning does not apply; use the GCM-SIV bounds above.
- **`no_std`.** Both crates work in `no_std`.
