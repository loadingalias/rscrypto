# Migration: `ascon-aead` (RustCrypto) → `rscrypto`

> Replace `AsconAead128` / `Key<T>` / `Nonce<T>` /
> `Payload { msg, aad }` with rscrypto's named types and a caller-buffer API.
> NIST SP 800-232 specifies a 16-byte key, nonce, and tag.

Verified against `ascon-aead = "0.6.0"` and the `rscrypto` 0.7.8 line.
Evidence: `tests/ascon_aead_oracle.rs`.

## TL;DR

| | Before (`ascon-aead` 0.6.x) | After (`rscrypto` 0.7.8) |
|---|---|---|
| Cargo dep | `ascon-aead = "0.6"` | `rscrypto = { version = "0.7.8", features = ["ascon-aead"] }` |
| Import | `use ascon_aead::{AsconAead128, Key, Nonce, aead::{Aead, KeyInit, Payload}};` | `use rscrypto::{Aead, AsconAead128, AsconAead128Key, aead::{Nonce128, expert::AeadWithNonce}};` |
| Encrypt | `cipher.encrypt(nonce, Payload { msg, aad })?` | `cipher.encrypt(&nonce, aad, msg, &mut out)?` |

## Cargo.toml

```toml
# Before
[dependencies]
ascon-aead = "0.6"
```

```toml
# After
[dependencies]
rscrypto = { version = "0.7.8", features = ["ascon-aead"] }
```

## Algorithm map

| `ascon-aead` type | rscrypto type | Key | Nonce | Tag |
|---|---|---|---|---|
| `AsconAead128` | `AsconAead128` | 16 bytes | 16 bytes | 16 bytes |
| `AsconAead128a` (legacy) | not mapped: superseded by NIST SP 800-232 |  |  |  |
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
use rscrypto::{
  Aead, AsconAead128, AsconAead128Key,
  aead::{Nonce128, expert::AeadWithNonce},
};

let key = AsconAead128Key::from_bytes([0u8; 16]);
let cipher = AsconAead128::new(&key);
let nonce = Nonce128::from_bytes([0u8; 16]);
let mut ct = vec![0u8; AsconAead128::ciphertext_len(plaintext.len())?];
cipher.encrypt(&nonce, aad, plaintext, &mut ct)?;
```

The expert trait import preserves an existing caller-nonce protocol. Prefer
`seal_random` when the protocol does not already define nonce derivation.

### Combined decrypt

```rust
// After
let mut plaintext = vec![0u8; AsconAead128::plaintext_len(ct.len())?];
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

- **NIST SP 800-232.** NIST published the final standard on 2025-08-13.
  The harness verifies byte-identical output for the final Ascon-AEAD128
  parameter set.
- **Implementation boundary.** rscrypto currently uses a portable, table-free
  implementation. That source structure is not a machine-code timing proof;
  generated-code constant-time claims are limited to the compiler, target,
  features, and binary in the matching
  [release evidence](../../constant-time.md).
- **128-bit key is the only key length.** Ascon-AEAD does not have a 256-bit
  variant; SP 800-232 specifies the 128-bit parameter set.
- **Nonce reuse semantics.** Ascon-AEAD-128 is *not* nonce-misuse-resistant. Reusing `(key, nonce)` reveals plaintext XORs. Prefer deterministic uniqueness. A uniformly random 128-bit nonce has lower collision probability than a uniformly random 96-bit nonce at the same message count, but the deployment must still define a message limit.
- **No `Payload`, no `KeyInit` import.** Same simplification as the rest of the AEAD lane.
- **Failed-open buffer semantics change.** RustCrypto keeps the in-place buffer
  unchanged on error. rscrypto clears it on authentication failure. Combined
  rscrypto decrypt also clears its output buffer on authentication failure.
- **`no_std`.** Both crates support `no_std`; rscrypto's caller-buffer API does not require `alloc`.
