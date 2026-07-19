# Migration: `blake2` (RustCrypto) → `rscrypto`

> Same algorithms (RFC 7693), with named convenience types replacing `Blake2b<U32>`-style generic instantiations and `Blake2b256::keyed_digest(key, data)` replacing the separate `Blake2bMac<U32>` MAC type.

Verified against `blake2 = "0.11.0-rc.6"` and the `rscrypto` 0.5.0 line.
Evidence: `tests/blake2_official_vectors.rs` and `tests/blake2_differential.rs`.
Code samples use the 0.10-style names where they remain the clearest migration
shape for existing projects.

## TL;DR

| | Before (`blake2` 0.10.x) | After (`rscrypto` 0.5.0) |
|---|---|---|
| Cargo dep | `blake2 = "0.10"` | `rscrypto = { version = "0.5.0", features = ["blake2b", "blake2s"] }` |
| Import | `use blake2::{Blake2b512, Digest};` | `use rscrypto::{Blake2b512, Digest};` |
| Call | `Blake2b512::digest(data)` | `Blake2b512::digest(data)` |

Drop one or both of `blake2b` / `blake2s` from the feature list if you don't use that family.

## Cargo.toml

```toml
# Before
[dependencies]
blake2 = "0.10"
```

```toml
# After
[dependencies]
rscrypto = { version = "0.5.0", features = ["blake2b", "blake2s"] }
```

## Algorithm map

| `blake2` type | rscrypto type | Output |
|---|---|---|
| `Blake2b<U32>` (generic) | `Blake2b256` | `[u8; 32]` |
| `Blake2b<U64>` or `Blake2b512` | `Blake2b512` | `[u8; 64]` |
| `Blake2b<UN>` (variable, runtime) | `Blake2b` | 1-64 bytes via `digest_into` |
| `Blake2s<U16>` | `Blake2s128` | `[u8; 16]` |
| `Blake2s<U32>` or `Blake2s256` | `Blake2s256` | `[u8; 32]` |
| `Blake2bMac<U32>` (keyed) | `Blake2b256::keyed_digest(...)` | `[u8; 32]` |
| `Blake2bMac512` (keyed, 512-bit) | `Blake2b512::keyed_digest(...)` | `[u8; 64]` |
| `Blake2sMac<U32>` (keyed) | `Blake2s256::keyed_digest(...)` | `[u8; 32]` |

`Blake2bp` / `Blake2sp` (parallel variants) are not currently mapped.

## API patterns

### Fixed-output one-shot

```rust
// Before
use blake2::{Blake2b512, Digest};
let out = Blake2b512::digest(b"123456789");        // Output<Blake2b512>
```

```rust
// After
use rscrypto::{Blake2b512, Digest};
let out: [u8; 64] = Blake2b512::digest(b"123456789");
```

### Custom-length Blake2b

```rust
// Before
use blake2::{Blake2b, Digest};
use blake2::digest::consts::U32;
let out = <Blake2b<U32> as Digest>::digest(b"123456789");   // [u8; 32] in an Output wrapper
```

```rust
// After
use rscrypto::{Blake2b256, Digest};
let out: [u8; 32] = Blake2b256::digest(b"123456789");
```

For runtime-variable output, use `Blake2b::digest_into(out_len, data, &mut buf)`: see the `Blake2b` rustdoc.

### Streaming

```rust
// Before
use blake2::{Blake2b512, Digest};
let mut hasher = Blake2b512::new();
hasher.update(b"foo");
hasher.update(b"bar");
let out = hasher.finalize();                       // consumes hasher
```

```rust
// After
use rscrypto::{Blake2b512, Digest};
let mut hasher = Blake2b512::new();
hasher.update(b"foo");
hasher.update(b"bar");
let out = hasher.finalize();                       // borrows &self
```

### Keyed mode (MAC)

```rust
// Before
use blake2::Blake2bMac512;
use blake2::digest::{KeyInit, Mac};
let key = [0x42u8; 32];
let mut mac = Blake2bMac512::new_from_slice(&key).unwrap();
mac.update(b"message");
let tag = mac.finalize().into_bytes();             // GenericArray<u8, U64>
```

```rust
// After
use rscrypto::Blake2b512;
let key = [0x42u8; 32];
let tag: [u8; 64] = Blake2b512::keyed_digest(&key, b"message");
```

For streaming keyed mode, use `Blake2b512::new_keyed(&key)`. The MAC type and the hash type are unified: no separate `Blake2bMac` import.

## Notes

- **Generic constants gone.** `Blake2b<U32>` and `Blake2s<U16>` style generics are replaced with named convenience types per output length. `generic-array` is no longer in your dependency tree if rscrypto is your only consumer.
- **MAC unification.** RustCrypto separates `Blake2bMac` from `Blake2b` because the MAC and the hash use different parameter blocks. rscrypto exposes both modes from the same type via `keyed_digest` / `new_keyed`. Personalisation, salt, and tree-hashing parameters are reachable through `Blake2bParams` / `Blake2sParams`.
- **`finalize` consumes vs. borrows.** Same as `sha2` / `sha3`: drop `.clone()`.
- **`Output<D>` → `[u8; N]`.** Same as `sha2` / `sha3`.
- **No generic keyed-output comparison.** rscrypto's Blake2 keyed digest
  returns raw bytes without a generic secret-comparison API. Use HMAC, KMAC,
  or typed `Blake3KeyedHash` verification when the protocol needs built-in
  verification.
- **`no_std`.** Both crates support `no_std`. rscrypto runtime-detects SIMD when `std` is enabled and falls back to compile-time `target_feature` selection in `no_std` builds.
