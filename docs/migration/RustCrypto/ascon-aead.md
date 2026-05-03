# Migration: `ascon-aead` (RustCrypto) → `rscrypto`

> NIST SP 800-232 lightweight AEAD. Replace `AsconAead128` / `Key<T>` / `Nonce<T>` / `Payload { msg, aad }` with rscrypto's named types and a buffer-style API. 128-bit key, 128-bit nonce, 128-bit tag — all the bytes are 16.

Verified against `ascon-aead = "0.5.2"` and the `rscrypto` 0.1 line.

## TL;DR

| | Before (`ascon-aead` 0.5.x) | After (`rscrypto` 0.1) |
|---|---|---|
| Cargo dep | `ascon-aead = "0.5"` | `rscrypto = { version = "0.1", features = ["ascon-aead"] }` |
| Import | `use ascon_aead::{AsconAead128, Key, Nonce, aead::{Aead, KeyInit, Payload}};` | `use rscrypto::{Aead, AsconAead128, AsconAead128Key, aead::Nonce128};` |
| Encrypt | `cipher.encrypt(nonce, Payload { msg, aad })?` | `cipher.encrypt(&nonce, aad, msg, &mut out)?` |

## Cargo.toml

```toml
# Before
[dependencies]
ascon-aead = "0.5"
```

```toml
# After
[dependencies]
rscrypto = { version = "0.1", features = ["ascon-aead"] }
```

## Algorithm map

| `ascon-aead` type | rscrypto type | Key | Nonce | Tag |
|---|---|---|---|---|
| `AsconAead128` | `AsconAead128` | 16 bytes | 16 bytes | 16 bytes |
| `AsconAead128a` (legacy) | not mapped — superseded by NIST SP 800-232 |  |  |  |
| `AsconAead80pq` (post-quantum-flavored) | not mapped |  |  |  |

## API patterns

### Combined encrypt

```rust
// Before
use ascon_aead::{AsconAead128, Key, Nonce};
use ascon_aead::aead::{Aead, KeyInit, Payload};

let key = Key::<AsconAead128>::from_slice(&[0u8; 16]);
let cipher = AsconAead128::new(key);
let nonce = Nonce::<AsconAead128>::from_slice(&[0u8; 16]);
let ct = cipher.encrypt(nonce, Payload { msg: plaintext, aad }).unwrap();
```

```rust
// After
use rscrypto::{Aead, AsconAead128, AsconAead128Key, aead::Nonce128};

let key = AsconAead128Key::from_bytes([0u8; 16]);
let cipher = AsconAead128::new(&key);
let nonce = Nonce128::from_bytes([0u8; 16]);
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
cipher.decrypt_in_place(&nonce, aad, &mut buffer, &tag)?;
```

## Notes

- **NIST SP 800-232 standardised in 2026.** Both crates ship the final spec (the 16-byte key / 16-byte nonce / 16-byte tag layout). Outputs are bit-identical (verified in the harness).
- **Why Ascon?** Designed for severely resource-constrained targets (ATmega, Cortex-M0+, RFID). Code size ~3 KB, no AES tables, no `unsafe` / SIMD requirement, constant-time on every target. If you're embedded, prefer Ascon over AES-GCM where you can.
- **128-bit key is the only key length.** Ascon-AEAD does not have a 256-bit variant; the 128-bit spec is what NIST standardised.
- **Nonce reuse semantics.** Ascon-AEAD-128 is *not* nonce-misuse-resistant. Reusing `(key, nonce)` reveals plaintext XORs. Use a counter or a fresh random 128-bit nonce per message; the larger nonce space (128 bits vs. AES-GCM's 96) makes random nonces safer at high volume.
- **No `Payload`, no `KeyInit` import.** Same simplification as the rest of the AEAD lane.
- **Performance.** Pure Rust, no SIMD path expected. rscrypto's portable kernel is the only kernel — `portable-only` feature is a no-op for Ascon but does not break the build.
- **`no_std`.** Both crates support `no_std`. Ascon's tiny state (320 bits) plus the buffer-style rscrypto API works on the deepest-embedded targets without `alloc`.
