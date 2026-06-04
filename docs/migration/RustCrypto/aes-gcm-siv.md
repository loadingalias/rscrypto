# Migration: `aes-gcm-siv` (RustCrypto) → `rscrypto`

> Same algorithm (RFC 8452), same nonce-misuse-resistant guarantees. Replace `Aes256GcmSiv` / `Key<Aes256GcmSiv>` / `Nonce` / `Payload { msg, aad }` with rscrypto's named types and a buffer-style API.

Verified against `aes-gcm-siv = "0.11.1"` and the `rscrypto` 0.3.1 line.
Evidence: `tests/aes128gcmsiv_oracle.rs`, `tests/aes256gcmsiv_oracle.rs`, and `tests/aead_wycheproof.rs`.

## TL;DR

| | Before (`aes-gcm-siv` 0.11.x) | After (`rscrypto` 0.3.1) |
|---|---|---|
| Cargo dep | `aes-gcm-siv = "0.11"` | `rscrypto = { version = "0.3.1", features = ["aes-gcm-siv"] }` |
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
rscrypto = { version = "0.3.1", features = ["aes-gcm-siv"] }
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

- **Why GCM-SIV?** Per RFC 8452, AES-GCM-SIV is misuse-resistant: accidentally reusing a `(key, nonce)` pair across two different messages reveals only that the messages are equal — it does *not* leak the GHASH key (which would otherwise enable universal forgery). Both crates implement the full RFC 8452 spec including the nonce-derived key schedule. Use it whenever your nonce source could plausibly produce duplicates (random selection, distributed counters across crash-restart, etc.).
- **All notes from `aes-gcm.md` apply.** `Payload { msg, aad }` collapses to positional args; `Nonce` becomes `Nonce96`; `Key<T>` becomes `Aes256GcmSivKey`; combined output is `[ciphertext || tag]` byte-identical to `aes-gcm-siv`'s `Vec<u8>`.
- **Performance.** GCM-SIV is roughly 1.5–2x slower than AES-GCM at large message sizes due to the polynomial-evaluation step. Both crates use AES-NI / AES-CE for the AES core; rscrypto adds VAES on AVX-512 hosts.
- **`no_std`.** Both crates work in `no_std`.
