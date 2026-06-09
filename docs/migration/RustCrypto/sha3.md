# Migration: `sha3` (RustCrypto) → `rscrypto`

> Same algorithms (FIPS 202: SHA3-224/256/384/512, SHAKE128/256), `[u8; N]` outputs replace `Output<D>`, and the SHAKE chain `finalize_xof().read(&mut out)` becomes `finalize_xof().squeeze(&mut out)`.

Verified against `sha3 = "0.12.0"` and the `rscrypto` 0.4.0 line.
Evidence: `tests/sha3_official_vectors.rs`, `tests/sha3_differential.rs`, `tests/shake128_differential.rs`, and `tests/shake256_differential.rs`.

## TL;DR

| | Before (`sha3` 0.12.x) | After (`rscrypto` 0.4.0) |
|---|---|---|
| Cargo dep | `sha3 = "0.12"` | `rscrypto = { version = "0.4.0", features = ["sha3"] }` |
| Import | `use sha3::{Sha3_256, Digest};` | `use rscrypto::{Sha3_256, Digest};` |
| Call | `Sha3_256::digest(data)` | `Sha3_256::digest(data)` |

## Cargo.toml

```toml
# Before
[dependencies]
sha3 = "0.12"
```

```toml
# After
[dependencies]
rscrypto = { version = "0.4.0", features = ["sha3"] }
```

## Algorithm map

| `sha3` type | rscrypto type | Output |
|---|---|---|
| `sha3::Sha3_224` | `rscrypto::Sha3_224` | `[u8; 28]` |
| `sha3::Sha3_256` | `rscrypto::Sha3_256` | `[u8; 32]` |
| `sha3::Sha3_384` | `rscrypto::Sha3_384` | `[u8; 48]` |
| `sha3::Sha3_512` | `rscrypto::Sha3_512` | `[u8; 64]` |
| `sha3::Shake128` | `rscrypto::Shake128` | XOF (variable) |
| `sha3::Shake256` | `rscrypto::Shake256` | XOF (variable) |
| `sha3::CShake256` | `rscrypto::Cshake256` | XOF (variable, customizable) |

`sha3::Keccak*` (the original Keccak padding, distinct from FIPS 202) is not currently mapped.

## API patterns

### Fixed-output one-shot (SHA-3)

```rust
// Before
use sha3::{Digest, Sha3_256};
let out = Sha3_256::digest(b"123456789");      // Output<Sha3_256>
```

```rust
// After
use rscrypto::{Digest, Sha3_256};
let out: [u8; 32] = Sha3_256::digest(b"123456789");
```

### Fixed-output streaming

```rust
// Before
use sha3::{Digest, Sha3_256};
let mut hasher = Sha3_256::new();
hasher.update(b"foo");
hasher.update(b"bar");
let out = hasher.finalize();                   // consumes hasher
```

```rust
// After
use rscrypto::{Digest, Sha3_256};
let mut hasher = Sha3_256::new();
hasher.update(b"foo");
hasher.update(b"bar");
let out = hasher.finalize();                   // borrows &self
```

### SHAKE (extendable output)

```rust
// Before
use sha3::Shake128;
use sha3::digest::{ExtendableOutput, Update, XofReader};
let mut hasher = Shake128::default();
hasher.update(b"data");
let mut reader = hasher.finalize_xof();
let mut out = [0u8; 64];
reader.read(&mut out);
```

```rust
// After
use rscrypto::{Shake128, Xof};
let mut hasher = Shake128::new();
hasher.update(b"data");
let mut reader = hasher.finalize_xof();
let mut out = [0u8; 64];
reader.squeeze(&mut out);
```

Three changes:

| RustCrypto | rscrypto |
|---|---|
| `Shake128::default()` | `Shake128::new()` |
| `XofReader::read(&mut out)` | `Xof::squeeze(&mut out)` |
| imports `ExtendableOutput`, `Update`, `XofReader` separately | one trait `Xof` |

The one-shot form is `Shake128::xof(data)` — returns the reader directly, no `new`/`update`/`finalize_xof` chain.

### cSHAKE (customizable)

`sha3` exposes `CShake128` / `CShake256`. rscrypto currently ships only `Cshake256` (SP 800-185).

```rust
// After
use rscrypto::{Cshake256, Xof};
let mut reader = Cshake256::xof(b"function-name", b"customization-string", b"data");
let mut out = [0u8; 64];
reader.squeeze(&mut out);
```

If you depend on `CShake128`, file an issue.

## Notes

- **Trait consolidation.** RustCrypto SHAKE requires three trait imports (`ExtendableOutput`, `Update`, `XofReader`). rscrypto's `Xof` trait carries all the methods you need; combine with `Digest` if you also call `update()` on the sponge type.
- **`finalize` consumes vs. borrows.** Same as `sha2` — RustCrypto consumes, rscrypto borrows. Drop `.clone()` calls.
- **`Output<D>` → `[u8; N]`.** Same as `sha2` — drop `.as_slice()` / `.as_ref()` and use the array directly.
- **`Mac` trait via KMAC.** RustCrypto ships a separate `kmac` crate. rscrypto's KMAC lives under `rscrypto::auth` behind `features = ["kmac"]` (which implies `sha3`). See the [`sha3-kmac` migration guide](../sha3-kmac.md).
- **`no_std`.** Both crates support `no_std`. rscrypto adds runtime SIMD detection when `std` is enabled, plus a portable kernel always available.
