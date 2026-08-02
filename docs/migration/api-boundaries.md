# Migration: misuse-resistant API boundaries

These API changes move invalid lengths and caller-controlled expert operations
to explicit boundaries. They do not change cryptographic outputs, buffer
layouts, or backend dispatch.

## BLAKE2

Keyed BLAKE2 accepts a validated borrowed key:

```rust
use rscrypto::{Blake2b256, Blake2bKey};

let key = Blake2bKey::new(b"key")?;
let tag = Blake2b256::keyed_digest(key, b"message");
let mut hasher = Blake2b256::new_keyed(key);
# Ok::<(), rscrypto::Blake2Error>(())
```

`Blake2bKey::new` and `Blake2sKey::new` reject empty or oversized keys at the
caller boundary. The borrowed key can then be reused without allocation or
copying. Omit the keyed method to select unkeyed hashing.

Variable-output BLAKE2b derives its output length from the destination:

```rust
use rscrypto::Blake2b;

let mut out = [0u8; 24];
Blake2b::digest_into(b"message", &mut out)?;
# Ok::<(), rscrypto::Blake2Error>(())
```

`Blake2bParams` uses exact `[u8; 16]` salt and personalization fields;
`Blake2sParams` uses `[u8; 8]`. Pad deliberately before the call when a
protocol defines a shorter value.

## Random generation

Panicking `random()` constructors were removed. Use `try_random()` or the
type-specific `try_generate()` method and propagate entropy failures.

## AEAD nonces

Normal `Aead` sealing generates a fresh OS nonce. AES-GCM also supports the
allocation-free `NonceCounter` stream. Caller-supplied nonce sealing is
available only after an explicit expert import:

```rust
use rscrypto::{
  Aes256Gcm, Aes256GcmKey,
  aead::{Nonce96, expert::AeadWithNonce},
};

let cipher = Aes256Gcm::new(&Aes256GcmKey::from_bytes([0x11; 32]));
let nonce = Nonce96::from_bytes([0x22; 12]);
let mut out = [0u8; 20];
cipher.encrypt(&nonce, b"aad", b"data", &mut out)?;
# Ok::<(), rscrypto::aead::SealError>(())
```

Use this extension only when a protocol or persistent counter already proves
nonce uniqueness for the key.

## Expert and diagnostic paths

- The implementation module `platform::detect` is private. Use `platform::get`,
  `platform::caps`, `platform::arch`, or `platform::caps_static` for normal
  detection.
- Detection overrides moved to `platform::expert`; use
  `try_set_override(Some(value))` to set and `try_set_override(None)` to clear.
  Uncached detection is `platform::expert::detect_uncached`.
- Explicit secret formatting is named `expert::DisplaySecret`.
- Diagnostic hooks remain under their owning modules such as `auth`, `aead`,
  and `hashes`; they are no longer re-exported from the crate root.

These namespace changes do not add a registry, lock, or dynamic dispatch.
