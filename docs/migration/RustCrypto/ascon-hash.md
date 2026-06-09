# Migration: `ascon-hash` (RustCrypto) → `rscrypto`

> Same algorithm (NIST LWC Ascon-Hash256). `ascon_hash::AsconHash256` becomes `rscrypto::AsconHash256`; everything else (trait shape, `update`, `finalize`) carries over.

Verified against `ascon-hash = "0.4.0"` and the `rscrypto` 0.4.0 line.
Evidence: `tests/ascon_official_vectors.rs`, `tests/ascon_hash_oracle.rs`, `tests/ascon_cxof_vectors.rs`, and `tests/ascon_differential.rs`.

## TL;DR

| | Before (`ascon-hash` 0.4.x) | After (`rscrypto` 0.4.0) |
|---|---|---|
| Cargo dep | `ascon-hash = "0.4"` | `rscrypto = { version = "0.4.0", features = ["ascon-hash"] }` |
| Import | `use ascon_hash::{AsconHash256, digest::Digest};` | `use rscrypto::{AsconHash256, Digest};` |
| Call | `AsconHash256::digest(data)` | `AsconHash256::digest(data)` |

## Cargo.toml

```toml
# Before
[dependencies]
ascon-hash = "0.4"
```

```toml
# After
[dependencies]
rscrypto = { version = "0.4.0", features = ["ascon-hash"] }
```

## Algorithm map

| `ascon-hash` type | rscrypto type | Output |
|---|---|---|
| `AsconHash256` | `AsconHash256` | `[u8; 32]` |
| `AsconXof128` | `AsconXof` (XOF) | variable |
| `AsconCxof128` | `AsconCxof128` (customizable XOF) | variable |

## API patterns

### One-shot

```rust
// Before
use ascon_hash::{AsconHash256, digest::Digest};
let out = AsconHash256::digest(b"123456789");      // Output<AsconHash256>
```

```rust
// After
use rscrypto::{AsconHash256, Digest};
let out: [u8; 32] = AsconHash256::digest(b"123456789");
```

### Streaming

```rust
// Before
use ascon_hash::{AsconHash256, digest::Digest};
let mut hasher = AsconHash256::new();
hasher.update(b"foo");
hasher.update(b"bar");
let out = hasher.finalize();                       // consumes hasher
```

```rust
// After
use rscrypto::{AsconHash256, Digest};
let mut hasher = AsconHash256::new();
hasher.update(b"foo");
hasher.update(b"bar");
let out = hasher.finalize();                       // borrows &self
```

### XOF

```rust
// After (rscrypto)
use rscrypto::{AsconXof, Xof};
let mut reader = AsconXof::xof(b"data");
let mut out = [0u8; 64];
reader.squeeze(&mut out);
```

### Customizable XOF (CXOF128, SP 800-232)

```rust
// After (rscrypto)
use rscrypto::{AsconCxof128, Xof};
let mut reader = AsconCxof128::xof(b"customization-string", b"data");
let mut out = [0u8; 64];
reader.squeeze(&mut out);
```

The `customization-string` is bounded at 256 bytes by the spec — passing longer returns an `AsconCxofCustomizationError` (use `try_xof` if you need fallible construction).

## Notes

- **`Output<D>` → `[u8; N]`.** Same as the rest of the RustCrypto migrations — drop `.as_slice()` / `.as_ref()`.
- **`finalize` consumes vs. borrows.** Same as the rest — drop any `.clone()`.
- **NIST LWC standard.** Ascon-Hash256 is the lightweight cryptography winner and standardised by NIST in SP 800-232. Both implementations track the final spec.
- **No SIMD.** Ascon's permutation is small enough that SIMD is not the dominant cost. rscrypto ships a portable-only implementation; the `portable-only` feature is a no-op for this algorithm but does not break the build.
- **`no_std`.** Both crates support `no_std`.
