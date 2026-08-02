# Migration: `ascon-hash` (RustCrypto) → `rscrypto`

> Replace `ascon_hash::AsconHash256` with `rscrypto::AsconHash256`. Both map to
> the SP 800-232 Ascon-Hash256 parameter set; the output bytes, `update`, and
> `finalize` flow are unchanged.

Verified against `ascon-hash = "0.4.0"` and the `rscrypto` 0.7.8 line.
Evidence: `tests/ascon_official_vectors.rs`, `tests/ascon_hash_oracle.rs`, `tests/ascon_cxof_vectors.rs`, and `tests/ascon_differential.rs`.

## TL;DR

| | Before (`ascon-hash` 0.4.x) | After (`rscrypto` 0.7.8) |
|---|---|---|
| Cargo dep | `ascon-hash = "0.4"` | `rscrypto = { version = "0.7.8", features = ["ascon-hash"] }` |
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
rscrypto = { version = "0.7.8", features = ["ascon-hash"] }
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
let mut reader = AsconCxof128::xof(b"customization-string", b"data")?;
let mut out = [0u8; 64];
reader.squeeze(&mut out);
```

The spec limits the customization string to 256 bytes.
`AsconCxof128::new`, `AsconCxof128::xof`, and
`AsconCxof128::hash_into` return `AsconCxofCustomizationError` when that limit
is exceeded.

## Notes

- **`Output<D>` → `[u8; N]`.** Same as the rest of the RustCrypto migrations: drop `.as_slice()` / `.as_ref()`.
- **`finalize` consumes vs. borrows.** Same as the rest: drop any `.clone()`.
- **NIST standard.** NIST published Ascon-Hash256, Ascon-XOF128, and
  Ascon-CXOF128 in SP 800-232 on 2025-08-13. The differential tests compare
  both implementations against the final parameter set.
- **Implementation boundary.** rscrypto currently exposes a portable
  implementation for these Ascon hash and XOF types. `portable-only` does not
  change their backend selection.
- **`no_std`.** Both crates support `no_std`.
