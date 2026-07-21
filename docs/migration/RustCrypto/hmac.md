# Migration: `hmac` (RustCrypto) → `rscrypto`

> Replace `Hmac::<Sha256>` / `Hmac::<Sha3_256>` (generic over digest) with named rscrypto types such as `HmacSha256` and `HmacSha3_256`. Key construction is infallible, `finalize()` borrows, and one-shot helpers return typed tags with sealed comparison decisions.

Verified against `hmac = "0.13.0"` and the `rscrypto` 0.6.4 line.
Evidence: `tests/hmac_sha256_vectors.rs`, `tests/hmac_sha2_family_vectors.rs`, `tests/hmac_sha3_vectors.rs`, the HMAC proptests, and `tests/hmac_wycheproof.rs`.

## TL;DR

| | Before (`hmac` 0.13.x) | After (`rscrypto` 0.6.4) |
|---|---|---|
| Cargo dep | `hmac = "0.13"` + `sha2 = "0.11"` | `rscrypto = { version = "0.6.4", features = ["hmac"] }` |
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
rscrypto = { version = "0.6.4", features = ["hmac"] }
```

The `hmac` feature implies `sha2`: no second dep to manage for SHA-2 HMAC. Use `features = ["hmac-sha3"]` for HMAC-SHA3, or `features = ["macs"]` when you want both SHA-2 and SHA-3 HMAC families.

## Algorithm map

| `hmac` instantiation | rscrypto type | Tag size |
|---|---|---|
| `Hmac<Sha256>` | `HmacSha256` | `HmacSha256Tag` |
| `Hmac<Sha384>` | `HmacSha384` | `HmacSha384Tag` |
| `Hmac<Sha512>` | `HmacSha512` | `HmacSha512Tag` |
| `Hmac<Sha3_224>` | `HmacSha3_224` | `HmacSha3_224Tag` |
| `Hmac<Sha3_256>` | `HmacSha3_256` | `HmacSha3_256Tag` |
| `Hmac<Sha3_384>` | `HmacSha3_384` | `HmacSha3_384Tag` |
| `Hmac<Sha3_512>` | `HmacSha3_512` | `HmacSha3_512Tag` |
| `Hmac<Sha224>` / `Hmac<Sha512_224>` / `Hmac<Sha512_256>` | not currently mapped: file an issue |  |

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
let tag = HmacSha256::mac(key, data);
let bytes: [u8; 32] = tag.to_bytes();
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
let tag = mac.finalize();                      // borrows &self
```

`finalize()` borrows in rscrypto; you can read the running tag, keep feeding bytes, and finalize again. `mac.reset()` restores the keyed initial state without rebuilding.

### Opaque verification

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
use rscrypto::{HmacSha256, HmacSha256Tag, Mac};
let expected_tag = HmacSha256Tag::from_bytes(expected_tag_bytes);
HmacSha256::verify_tag(key, data, &expected_tag)?;   // Result<(), VerificationError>
```

Streaming form: `let mut mac = HmacSha256::new(key); mac.update(data); mac.verify(&expected_tag)?;`. Tags are typed (`HmacSha256Tag`, `HmacSha384Tag`, `HmacSha512Tag`, and the `HmacSha3_*Tag` family) and deliberately do not implement `PartialEq` or `Eq`. Prefer `verify` / `verify_tag` for authentication decisions. Direct tag comparison is explicit: `left.ct_eq(&right)` returns an opaque `CtDecision`, and only `.declassify()` exposes a branchable bit. This source-level boundary is not a universal timing proof; constant-time claims remain limited to the compiler, target, features, and binary in the matching [release evidence](../../constant-time.md). Use `as_bytes()` / `to_bytes()` only at protocol serialization boundaries.

## Notes

- **Generic parameter gone.** `Hmac<D>` becomes one of the named types (`HmacSha256`, `HmacSha384`, `HmacSha512`, `HmacSha3_224`, `HmacSha3_256`, `HmacSha3_384`, `HmacSha3_512`). Replace any `where D: Digest + BlockSizeUser` bound with the concrete rscrypto type. If you genuinely need to be generic over the underlying hash, file an issue.
- **`KeyInit` import not needed.** RustCrypto routes key construction through `KeyInit::new_from_slice`. rscrypto's `HmacSha256::new(&[u8])` is inherent: no extra trait import.
- **`finalize` consumes vs. borrows.** Same pattern as `sha2` / `sha3` migrations, except HMAC returns a typed tag rather than a raw byte array.
- **`Mac::chain_update` builder pattern.** RustCrypto's `chain_update(data) -> Self` lets you write `mac.chain_update(a).chain_update(b).finalize()`. rscrypto does not currently expose the chained variant; use `.update(a); .update(b);` then `.finalize()`. Same number of statements, no `Self` returns to thread.
- **Long keys are pre-hashed (RFC 2104).** Both crates pre-hash any key longer than the underlying hash's block size (64 bytes for SHA-256, 128 bytes for SHA-384/SHA-512, and SHA-3 rates of 144/136/104/72 bytes for SHA3-224/256/384/512). Outputs are byte-identical regardless of key length.
- **`MacError` → `VerificationError`.** RustCrypto's `MacError` is opaque (no detail). rscrypto's `VerificationError` is also opaque, so neither exposes an error-variant oracle. Error opacity alone does not prove identical timing. The names differ; the result contract is the same.
- **`no_std`.** Both crates support `no_std`. rscrypto's `mac_to_vec` / `finalize_to_vec` helpers are gated on `alloc`; the core fixed-array API works in pure `no_std`.
