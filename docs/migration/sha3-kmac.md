# Migration: `sha3-kmac` → `rscrypto`

Replace fixed-output `sha3_kmac::Kmac128` / `Kmac256` with
`rscrypto::Kmac128` / `Kmac256`. Construction becomes infallible and
`finalize_into` borrows the state. `KmacXof128` and `KmacXof256` do not have
equivalent rscrypto types.

KMAC128/256 fixed-output behavior is covered by NIST vectors and
`tiny-keccak` differential tests. KMAC256 also has Wycheproof coverage.
Evidence: `tests/kmac128_nist_vectors.rs`, `tests/kmac128_differential.rs`,
`tests/kmac256_nist_vectors.rs`, `tests/kmac256_differential.rs`, and
`tests/kmac_wycheproof.rs`.

## TL;DR

| | Before (`sha3-kmac` 0.3.x) | After (`rscrypto` 0.8.1) |
|---|---|---|
| Cargo dep | `sha3-kmac = "0.3"` | `rscrypto = { version = "0.8.1", features = ["kmac"] }` |
| Import | `use sha3_kmac::Kmac256;` | `use rscrypto::Kmac256;` |
| Call | `let mut k = Kmac256::new(key, custom)?; k.update(data); k.finalize_into(&mut tag);` | `Kmac256::mac_into(key, custom, data, &mut tag);` |

## Cargo.toml

```toml
# Before
[dependencies]
sha3-kmac = "0.3"
```

```toml
# After
[dependencies]
rscrypto = { version = "0.8.1", features = ["kmac"] }
```

The `kmac` feature implies `sha3`.

## Algorithm map

| `sha3-kmac` type | rscrypto type | Security |
|---|---|---|
| `Kmac128` | `Kmac128` | 128-bit |
| `Kmac256` | `Kmac256` | 256-bit |
| `KmacXof128` / `KmacXof256` | not mapped | Keep `sha3-kmac` for KMACXOF |

## API patterns

### One-shot tag

```rust
// Before
use sha3_kmac::Kmac256;
let mut k = Kmac256::new(key, custom).expect("key must be ≥ 32 bytes");
k.update(data);
let mut tag = [0u8; 32];
k.finalize_into(&mut tag);                      // consumes k
```

```rust
// After
use rscrypto::Kmac256;
let tag: [u8; 32] = Kmac256::mac_array(key, custom, data);
```

Two changes:

| `sha3-kmac` | rscrypto |
|---|---|
| `Kmac256::new(key, custom)` returns `Result<Self, InvalidLength>` (key < 32 bytes errors) | `Kmac256::new(key, custom)` is infallible (no minimum key size enforced; SP 800-185 leaves it to caller) |
| `k.finalize_into(&mut tag)` consumes `k` | `Kmac256::mac_into(...)` and the streaming `.finalize_into(&mut [u8])` borrow |

If your code relied on `sha3-kmac` rejecting keys shorter than 32 bytes, port
that application policy explicitly:
`assert!(key.len() >= 32, "KMAC key policy requires at least 32 bytes");`.
SP 800-185 does not impose that fixed constructor minimum; key length remains a
security parameter owned by the protocol.

### Streaming

```rust
// Before
use sha3_kmac::Kmac256;
let mut k = Kmac256::new(key, custom)?;
k.update(b"foo");
k.update(b"bar");
let mut tag = [0u8; 32];
k.finalize_into(&mut tag);
```

```rust
// After
use rscrypto::Kmac256;
let mut k = Kmac256::new(key, custom);
k.update(b"foo");
k.update(b"bar");
let mut tag = [0u8; 32];
k.finalize_into(&mut tag);
```

Same shape; rscrypto borrows `&mut self` instead of consuming `self`. Call `k.reset()` to start over without rebuilding the absorbed `(key, custom)` state.

### Variable-length output

KMAC's output length is part of the tag derivation: different lengths give different tags. Both crates encode the length identically (verified at 32 and 64 bytes in the harness):

```rust
// After
use rscrypto::Kmac256;
let mut tag = [0u8; 64];
Kmac256::mac_into(key, custom, data, &mut tag);   // 64-byte tag, distinct from 32-byte tag
```

### Opaque verification

```rust
// Before
// sha3-kmac has no verify helper; hand-roll with `subtle`:
use subtle::ConstantTimeEq;
let mut got = [0u8; 32];
let mut k = sha3_kmac::Kmac256::new(key, custom)?;
k.update(data);
k.finalize_into(&mut got);
let ok: bool = got.ct_eq(&expected).into();
```

```rust
// After
use rscrypto::Kmac256;
Kmac256::verify_tag(key, custom, data, &expected)?;   // Result<(), VerificationError>
```

For streaming verification, construct `Kmac256`, call `update`, then call
`verify(&expected)`. The authentication helpers require at least 16 bytes for
KMAC128 and 32 bytes for KMAC256, preserving the named security strength.

Use `verify_primitive` or `verify_tag_primitive` only when a protocol specifies
a shorter output and defines its forgery budget and failed-attempt limit.
Arbitrary-length `finalize_into` and `mac_into` remain available for PRF or KDF
use. Verification traverses the public-length expected tag before returning one
opaque result. Generated-code timing claims remain limited to the matching
[release evidence](../constant-time.md).

## Notes

- **Infallible `new` vs. fallible `new`.** `sha3-kmac` rejects keys shorter than
  32 bytes at construction; rscrypto leaves key-length policy at the call site.
  Preserve the upstream policy explicitly if your protocol depends on it.
- **Customization.** Both constructors accept a customization string. Pass
  `b""` only when the protocol specifies an empty customization string.
- **`no_std`.** Both crates work in `no_std`. rscrypto's `mac_to_vec` style helpers are gated on `alloc`; the fixed-array and user-supplied-buffer paths are pure `no_std`.
- **KMACXOF is not fixed-output KMAC with a longer buffer.** SP 800-185 KMAC
  appends `right_encode(L)`; KMACXOF appends `right_encode(0)`. Keep
  `sha3-kmac` for `KmacXof128` / `KmacXof256` until rscrypto exposes that mode.
- **NIST SP 800-185 conformance.** Both implementations track the spec including the `right_encode` length suffix and `bytepad` block alignment. Outputs are bit-identical at every length tested in the harness (32 and 64 bytes); for assurance, run the harness yourself with your specific lengths.
- **Hand-rolled cSHAKE-based KMAC.** Verify `bytepad(encode_string(K))`, the
  function-name/customization encoding, and the trailing `right_encode(L)`.
  Treat divergent output as a mode, encoding, or implementation mismatch that
  must be resolved before migration.
