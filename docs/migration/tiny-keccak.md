# Migration: `tiny-keccak` → `rscrypto`

> Replace `tiny_keccak::Kmac::v256` and `tiny_keccak::CShake::v256` with `rscrypto::Kmac256` and `rscrypto::Cshake256`. Same SP 800-185 algorithms, byte-identical output, infallible construction, and a `verify` helper for KMAC tags.

Verified against `tiny-keccak = "2.0.2"` (with `kmac` and `cshake` features) and the `rscrypto` 0.5.0 line.
Evidence: `tests/kmac256_differential.rs`, `tests/cshake256_differential.rs`, `tests/cshake256_nist_vectors.rs`, and `tests/kmac_wycheproof.rs`.

## TL;DR

| | Before (`tiny-keccak` 2.x) | After (`rscrypto` 0.5.0) |
|---|---|---|
| Cargo dep | `tiny-keccak = { version = "2.0", features = ["kmac", "cshake"] }` | `rscrypto = { version = "0.5.0", features = ["kmac"] }` |
| KMAC import | `use tiny_keccak::{Hasher, Kmac};` | `use rscrypto::Kmac256;` |
| KMAC call | `let mut k = Kmac::v256(key, custom); k.update(data); k.finalize(&mut tag);` | `Kmac256::mac_into(key, custom, data, &mut tag);` |
| cSHAKE import | `use tiny_keccak::{Hasher, CShake};` | `use rscrypto::{Cshake256, Xof};` |
| cSHAKE call | `let mut x = CShake::v256(name, custom); x.update(data); x.finalize(&mut out);` | `Cshake256::xof(name, custom, data).squeeze(&mut out);` |

## Cargo.toml

```toml
# Before
[dependencies]
tiny-keccak = { version = "2.0", features = ["kmac", "cshake"] }
```

```toml
# After
[dependencies]
rscrypto = { version = "0.5.0", features = ["kmac"] }
```

The `kmac` feature implies `sha3` (which provides the underlying `Cshake256` sponge for both `Kmac256` and the standalone cSHAKE primitive).

If you only use cSHAKE and not KMAC, swap the feature for `sha3` alone. That exposes `Cshake256` without pulling in KMAC.

## Algorithm map

| `tiny-keccak` type | rscrypto type | Spec |
|---|---|---|
| `Kmac::v256(key, custom)` | `Kmac256` | NIST SP 800-185 §4.3 |
| `Kmac::v128(key, custom)` | not currently mapped: open an issue | NIST SP 800-185 §4.3 |
| `KmacXof::v256` (XOF mode) | `Kmac256` (variable-output via `mac_into` buffer length) | SP 800-185 §4.3 |
| `CShake::v256(name, custom)` | `Cshake256` | NIST SP 800-185 §3 |
| `CShake::v128(name, custom)` | not currently mapped | NIST SP 800-185 §3 |
| `Sha3*`, `Keccak*`, `Shake*`, `ParallelHash*`, `TupleHash*` | covered by `RustCrypto/sha3.md` (Sha3/Shake) or not yet mapped (ParallelHash, TupleHash) | FIPS 202 / SP 800-185 |

If you migrate from `tiny-keccak` for SHA-3 / SHAKE specifically (not KMAC / cSHAKE), follow `RustCrypto/sha3.md` instead: same destination types, slightly different upstream API.

## API patterns

### KMAC256: one-shot tag

```rust
// Before
use tiny_keccak::{Hasher, Kmac};
let mut k = Kmac::v256(key, custom);
k.update(data);
let mut tag = [0u8; 32];
k.finalize(&mut tag);                    // consumes k
```

```rust
// After
use rscrypto::Kmac256;
let tag: [u8; 32] = Kmac256::mac_array(key, custom, data);
```

Two changes:

| `tiny-keccak` | rscrypto |
|---|---|
| `Hasher` trait import required | inherent methods on `Kmac256` |
| `finalize(&mut tag)` consumes `self` | `mac_into` / `mac_array` are static; streaming `finalize_into` borrows `&mut self` |

### KMAC256: streaming

```rust
// Before
use tiny_keccak::{Hasher, Kmac};
let mut k = Kmac::v256(key, custom);
k.update(b"foo");
k.update(b"bar");
let mut tag = [0u8; 32];
k.finalize(&mut tag);
```

```rust
// After
use rscrypto::Kmac256;
let mut k = Kmac256::new(key, custom);
k.update(b"foo");
k.update(b"bar");
let mut tag = [0u8; 32];
k.finalize_into(&mut tag);               // borrows &mut self
```

`k.reset()` is available in rscrypto to start over without rebuilding the absorbed `(key, custom)` state.

### KMAC256: variable-length output

The output length is part of the KMAC tag derivation per SP 800-185: a 32-byte tag is *not* the prefix of a 64-byte tag. Both crates encode the length identically (verified at 32 and 64 bytes in the harness):

```rust
// After
use rscrypto::Kmac256;
let mut tag = [0u8; 64];
Kmac256::mac_into(key, custom, data, &mut tag);
```

### KMAC256: constant-time verification

```rust
// Before
// tiny-keccak has no verify helper; hand-roll with `subtle`:
use subtle::ConstantTimeEq;
use tiny_keccak::{Hasher, Kmac};

let mut k = Kmac::v256(key, custom);
k.update(data);
let mut got = [0u8; 32];
k.finalize(&mut got);
let ok: bool = got.ct_eq(&expected).into();
```

```rust
// After
use rscrypto::Kmac256;
Kmac256::verify_tag(key, custom, data, &expected)?;   // Result<(), VerificationError>
```

Drop the `subtle` dependency for the verify path. Streaming form: `let mut k = Kmac256::new(key, custom); k.update(data); k.verify(&expected)?;`.

### cSHAKE256: XOF streaming

```rust
// Before
use tiny_keccak::{Hasher, CShake};
let mut x = CShake::v256(function_name, customization);
x.update(data);
let mut out = [0u8; 64];
x.finalize(&mut out);                    // consumes x; uses Hasher::finalize for fixed length
```

```rust
// After
use rscrypto::{Cshake256, Xof};
let mut reader = Cshake256::xof(function_name, customization, data);
let mut out = [0u8; 64];
reader.squeeze(&mut out);
```

For the streaming form (data fed in chunks):

```rust
// After
use rscrypto::{Cshake256, Digest, Xof};
let mut x = Cshake256::new(function_name, customization);
x.update(b"foo");
x.update(b"bar");
let mut reader = x.finalize_xof();
let mut out = [0u8; 64];
reader.squeeze(&mut out);
```

Three changes from `tiny-keccak`:

| `tiny-keccak` | rscrypto |
|---|---|
| `Hasher::finalize` consumes the sponge and writes to a fixed buffer | `Cshake256` has a fused one-shot `xof(name, custom, data)` that returns a reader; streaming `finalize_xof()` returns the same reader |
| Cannot squeeze more bytes after `finalize` | Reader is a dedicated XOF type: call `squeeze` repeatedly for additional bytes |
| `Hasher::update` adds chunks | `Digest::update` plays the same role |

## Notes

- **`Kmac::v128` not yet mapped.** rscrypto ships only the 256-bit security level. If you depend on `Kmac128` for size-constrained protocols, file an issue.
- **`CShake::v128` not yet mapped.** Same situation. rscrypto ships `Cshake256` only.
- **`ParallelHash` and `TupleHash` (SP 800-185 §6 / §5) not yet mapped.** `tiny-keccak` ships these behind feature flags. rscrypto does not. If you depend on them, keep `tiny-keccak` for those primitives.
- **Spec conformance verified.** Outputs are byte-identical at every parameter set tested in the harness (KMAC at 32 and 64 bytes; cSHAKE at 64-byte squeeze). For your specific lengths, run the harness yourself.
- **Hand-rolled cSHAKE-based KMAC.** Some codebases implement KMAC by hand on top of `tiny_keccak::CShake`. The migration target is the same: `rscrypto::Kmac256`. Verify that your hand-rolled padding matches the SP 800-185 spec, specifically the `bytepad(encode_string(K))` step and the trailing `right_encode(L)`. If your output diverges from rscrypto's, your hand-rolled implementation has a spec bug; fix it before migrating.
- **`no_std`.** Both crates support `no_std`. rscrypto's `mac_to_vec` style helpers are gated on `alloc`; the fixed-array and user-supplied-buffer paths are pure `no_std`.
- **Performance.** Both crates use scalar Rust; SHA-3 / Keccak doesn't benefit from typical SIMD without dedicated SHA3-NI hardware (rare in 2026). rscrypto's portable kernel is the only kernel for this lane.
