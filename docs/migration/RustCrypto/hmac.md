# Migration: `hmac` (RustCrypto) → `rscrypto`

> Replace `Hmac::<Sha256>` (generic over digest) with `HmacSha256` (named per hash). Key construction is now infallible, `finalize()` borrows, and the one-shot helper `HmacSha256::mac(key, data)` collapses the four-line idiom into one.

Verified against `hmac = "0.13.0"` and the `rscrypto` 0.3.1 line.
Evidence: `tests/hmac_sha256_vectors.rs`, `tests/hmac_sha2_family_vectors.rs`, the HMAC proptests, and `tests/hmac_wycheproof.rs`.

## TL;DR

| | Before (`hmac` 0.13.x) | After (`rscrypto` 0.3.1) |
|---|---|---|
| Cargo dep | `hmac = "0.13"` + `sha2 = "0.11"` | `rscrypto = { version = "0.3.1", features = ["hmac"] }` |
| Import | `use hmac::{Hmac, Mac, KeyInit}; use sha2::Sha256;` | `use rscrypto::{HmacSha256, Mac};` |
| Call | `Hmac::<Sha256>::new_from_slice(key).unwrap().chain_update(data).finalize().into_bytes()` | `HmacSha256::mac(key, data)` |

## Cargo.toml

```toml
# Before
[dependencies]
hmac = "0.13"
sha2 = "0.11"            # required as the generic parameter
```

```toml
# After
[dependencies]
rscrypto = { version = "0.3.1", features = ["hmac"] }
```

The `hmac` feature implies `sha2` — no second dep to manage.

## Algorithm map

| `hmac` instantiation | rscrypto type | Tag size |
|---|---|---|
| `Hmac<Sha256>` | `HmacSha256` | `[u8; 32]` |
| `Hmac<Sha384>` | `HmacSha384` | `[u8; 48]` |
| `Hmac<Sha512>` | `HmacSha512` | `[u8; 64]` |
| `Hmac<Sha224>` / `Hmac<Sha512_224>` / `Hmac<Sha512_256>` / `Hmac<Sha3_*>` | not currently mapped — file an issue |  |

## API patterns

### One-shot tag

```rust
// Before
use hmac::{Hmac, Mac, KeyInit};
use sha2::Sha256;
let mut mac = Hmac::<Sha256>::new_from_slice(key).expect("HMAC accepts any key length");
mac.update(data);
let tag = mac.finalize().into_bytes();         // GenericArray<u8, U32>
```

```rust
// After
use rscrypto::{HmacSha256, Mac};
let tag: [u8; 32] = HmacSha256::mac(key, data);
```

`HmacSha256::new(key)` is infallible (HMAC accepts any key length per RFC 2104; long keys are pre-hashed). Drop the `.unwrap()` / `.expect()`.

### Streaming

```rust
// Before
use hmac::{Hmac, Mac, KeyInit};
use sha2::Sha256;
let mut mac = Hmac::<Sha256>::new_from_slice(key).unwrap();
mac.update(b"foo");
mac.update(b"bar");
let tag = mac.finalize().into_bytes();         // consumes mac
```

```rust
// After
use rscrypto::{HmacSha256, Mac};
let mut mac = HmacSha256::new(key);
mac.update(b"foo");
mac.update(b"bar");
let tag: [u8; 32] = mac.finalize();            // borrows &self
```

`finalize()` borrows in rscrypto; you can read the running tag, keep feeding bytes, and finalize again. `mac.reset()` restores the keyed initial state without rebuilding.

### Constant-time verification

```rust
// Before
use hmac::{Hmac, Mac, KeyInit};
use sha2::Sha256;
let mut mac = Hmac::<Sha256>::new_from_slice(key).unwrap();
mac.update(data);
mac.verify_slice(&expected_tag)?;              // returns Result<(), MacError>
```

```rust
// After
use rscrypto::{HmacSha256, Mac};
HmacSha256::verify_tag(key, data, &expected_tag)?;   // Result<(), VerificationError>
```

Streaming form: `let mut mac = HmacSha256::new(key); mac.update(data); mac.verify(&expected_tag)?;`. Both helpers do constant-time comparison via `subtle`-equivalent internals — never use `==` on the raw tag.

## Notes

- **Generic parameter gone.** `Hmac<D>` becomes one of the named types (`HmacSha256`, `HmacSha384`, `HmacSha512`). Replace any `where D: Digest + BlockSizeUser` bound with the concrete rscrypto type. If you genuinely need to be generic over the underlying hash, file an issue.
- **`KeyInit` import not needed.** RustCrypto routes key construction through `KeyInit::new_from_slice`. rscrypto's `HmacSha256::new(&[u8])` is inherent — no extra trait import.
- **`finalize` consumes vs. borrows.** Same pattern as `sha2` / `sha3` migrations.
- **`Mac::chain_update` builder pattern.** RustCrypto's `chain_update(data) -> Self` lets you write `mac.chain_update(a).chain_update(b).finalize()`. rscrypto does not currently expose the chained variant; use `.update(a); .update(b);` then `.finalize()`. Same number of statements, no `Self` returns to thread.
- **Long keys are pre-hashed (RFC 2104).** Both crates pre-hash any key longer than the underlying hash's block size (64 bytes for SHA-256/SHA-384, 128 bytes for SHA-512). Outputs are byte-identical regardless of key length.
- **`MacError` → `VerificationError`.** RustCrypto's `MacError` is opaque (no detail). rscrypto's `VerificationError` is also opaque — both are designed not to leak timing information through error variants. The names differ; the contract is the same.
- **`no_std`.** Both crates support `no_std`. rscrypto's `mac_to_vec` / `finalize_to_vec` helpers are gated on `alloc`; the core fixed-array API works in pure `no_std`.
