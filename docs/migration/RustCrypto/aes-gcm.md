# Migration: `aes-gcm` (RustCrypto) → `rscrypto`

Replace the `Aes256Gcm` / `Key<Aes256Gcm>` / `Nonce` /
`Payload { msg, aad }` builder with rscrypto's named types and a caller-buffer
API. The combined ciphertext-and-tag bytes remain interoperable.

Verified against `aes-gcm = "0.11.0"` and the `rscrypto` 0.7.8 line.
Evidence: `tests/aes128gcm_oracle.rs`, `tests/aes256gcm_oracle.rs`, and `tests/aead_wycheproof.rs`.

## TL;DR

| | Before (`aes-gcm` 0.11.x) | After (`rscrypto` 0.7.8) |
|---|---|---|
| Cargo dep | `aes-gcm = "0.11"` | `rscrypto = { version = "0.7.8", features = ["aes-gcm"] }` |
| Import | `use aes_gcm::{Aes256Gcm, Key, Nonce, KeyInit, aead::{Aead, Payload}};` | `use rscrypto::{Aead, Aes256Gcm, Aes256GcmKey, aead::{Nonce96, expert::AeadWithNonce}};` |
| Encrypt | `cipher.encrypt(nonce, Payload { msg, aad })?` (returns `Vec<u8>`) | `cipher.encrypt(&nonce, aad, msg, &mut out)?` (writes into caller buffer) |

## Cargo.toml

```toml
# Before
[dependencies]
aes-gcm = "0.11"
```

```toml
# After
[dependencies]
rscrypto = { version = "0.7.8", features = ["aes-gcm"] }
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
use rscrypto::{
  Aead, Aes256Gcm, Aes256GcmKey,
  aead::{Nonce96, expert::AeadWithNonce},
};

let key = Aes256GcmKey::from_bytes([0u8; 32]);
let cipher = Aes256Gcm::new(&key);
let nonce = Nonce96::from_bytes([0u8; 12]);
let mut ct = vec![0u8; Aes256Gcm::ciphertext_len(plaintext.len())?];
cipher.encrypt(&nonce, aad, plaintext, &mut ct)?;
// ct[..plaintext.len()] is ciphertext, ct[plaintext.len()..] is the 16-byte tag.
```

The output layout is identical (`[ciphertext || tag]`), so on-the-wire compatibility is preserved. The shape change is who owns the buffer: `aes-gcm` allocates a `Vec`, rscrypto writes into a buffer you pre-sized.
Use `Aes256Gcm::ciphertext_len` and `Aes256Gcm::plaintext_len` before
allocation; they reject length overflow and combined inputs shorter than the
tag.
The expert trait import is required because this migration preserves the
upstream caller-supplied nonce. New protocols should use `seal_random` or
`NonceCounter<Aes256Gcm>` so nonce issuance is not a normal call-site choice.

### Combined decrypt

```rust
// Before
let plaintext = cipher
    .decrypt(nonce, Payload { msg: &ct, aad })
    .unwrap();
```

```rust
// After
let mut plaintext = vec![0u8; Aes256Gcm::plaintext_len(ct.len())?];
cipher.decrypt(&nonce, aad, &ct, &mut plaintext)?;
```

Decrypt returns `Err(OpenError::Verification(_))` on tag mismatch: opaque, no detail leaked.

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

Both names require the explicit `AeadWithNonce` import.

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
// On tag mismatch, rscrypto zeroes buffer before returning
// Err(OpenError::Verification(_)).
```

## Notes

- **`Key<T>` is gone, `Nonce` is gone.** Use the per-algorithm `Aes256GcmKey` and the size-specific `Nonce96`. No `KeyInit` trait import; `Aes256Gcm::new(&key)` is inherent.
- **`Payload { msg, aad }` is gone.** Pass `aad` and `msg` (or `aad` and the in-place buffer) as positional args. AAD is a plain `&[u8]`; pass `b""` for "no AAD".
- **Cipher reuse does not require cloning.** AEAD cipher values are reusable but do not implement `Clone`, because cloning would silently duplicate secret state. Keep one cipher value and supply a fresh nonce for each encryption.
- **Nonce reuse is catastrophic for AES-GCM.** Both crates accept any
  `Nonce96`; neither prevents duplicates. If the design cannot guarantee
  uniqueness, evaluate `Aes256GcmSiv` and its misuse-resistance bounds in
  `aes-gcm-siv.md`.
- **Failed-open buffer semantics change.** RustCrypto's in-place AEAD contract
  keeps the buffer unchanged on error. rscrypto clears the caller's in-place
  buffer, and clears the combined-decrypt output buffer, on authentication
  failure. Do not rely on preserving the ciphertext after a failed rscrypto
  open.
- **Quantify random-nonce collision risk.** For uniformly random `n`-bit
  nonces and `q` encryptions, the birthday approximation is
  `q(q-1) / 2^(n+1)`. At `q = 2^32` with a 96-bit nonce, that is about
  `2^-33`. Prefer deterministic uniqueness for AES-GCM; if the protocol uses
  random nonces, set a message limit from its collision budget. XChaCha20-
  Poly1305 provides a 192-bit nonce space but is a different algorithm and wire
  format.
- **`AeadInPlace` trait import not needed.** RustCrypto requires importing `aead::AeadInPlace` separately to call the `_in_place_detached` methods. rscrypto exposes both shapes through the single `Aead` trait.
- **`generic-array` is gone.** rscrypto does not return `GenericArray` from any AEAD method. Tags are typed newtypes (`Aes256GcmTag`) wrapping `[u8; 16]`; key/nonce types wrap `[u8; N]` directly.
- **Hardware acceleration.** Both crates dispatch to AES-NI on x86_64 and
  AES-CE on aarch64. rscrypto adds VAES (AVX-512), s390x CPACF, and a portable
  bitsliced fallback that avoids secret-indexed tables. That source property is
  not a universal timing proof; constant-time coverage is limited to the
  compiler, target, features, and binary in the matching
  [release evidence](../../constant-time.md). The crate's `portable-only`
  feature makes runtime capability detection ignore host acceleration but does
  not override a compile-time backend; see
  [`docs/features.md`](../../features.md#portable-only).
- **`no_std`.** Both crates support `no_std`. rscrypto's combined API requires the caller to provide an output buffer, which fits stack-only embedded use. The `vec!` calls in the examples above are for std convenience; in `no_std` they become fixed-size arrays.
