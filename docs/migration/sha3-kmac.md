# Migration: `sha3-kmac` → `rscrypto`

> Replace `sha3_kmac::Kmac128` / `Kmac256` (fallible `new`, consuming `finalize_into`) with `rscrypto::Kmac128` / `Kmac256` (infallible `new`, borrowing `finalize_into`). Same NIST SP 800-185 algorithms; outputs are byte-identical at every length.

KMAC128/256 output is covered by NIST vectors and `tiny-keccak` differential
tests; KMAC256 also has Wycheproof coverage in `tests/kmac128_nist_vectors.rs`,
`tests/kmac128_differential.rs`, `tests/kmac256_nist_vectors.rs`,
`tests/kmac256_differential.rs`, and `tests/kmac_wycheproof.rs`.

## TL;DR

| | Before (`sha3-kmac` 0.3.x) | After (`rscrypto` 0.6.4) |
|---|---|---|
| Cargo dep | `sha3-kmac = "0.3"` | `rscrypto = { version = "0.6.4", features = ["kmac"] }` |
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
rscrypto = { version = "0.6.4", features = ["kmac"] }
```

The `kmac` feature implies `sha3`.

## Algorithm map

| `sha3-kmac` type | rscrypto type | Security |
|---|---|---|
| `Kmac128` | `Kmac128` | 128-bit |
| `KmacXof128` | `Kmac128` (variable-output is the same call: see below) | 128-bit XOF |
| `Kmac256` | `Kmac256` | 256-bit |
| `KmacXof256` | `Kmac256` (variable-output is the same call: see below) | 256-bit XOF |

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

If your codebase relied on `sha3-kmac`'s 32-byte key minimum to reject short keys, port that check explicitly: `assert!(key.len() >= 32, "KMAC256 key must be at least 32 bytes");`. The SP 800-185 spec gives no upper bound; SP 800-108 recommends ≥ 16 bytes for any KMAC.

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

### Constant-time verification

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

Streaming form: `let mut k = Kmac256::new(key, custom); k.update(data); k.verify(&expected)?;`. Drop the `subtle` dependency.

## Notes

- **Infallible `new` vs. fallible `new`.** `sha3-kmac` enforces SP 800-185's 32-byte key minimum at construction; rscrypto leaves the policy at the call site. If you want both: port the explicit check shown above.
- **Customization string is mandatory** in both crates. Pass `b""` if you want the unkeyed-customization form (which is rare; KMAC is almost always used with a domain-separation string).
- **`no_std`.** Both crates work in `no_std`. rscrypto's `mac_to_vec` style helpers are gated on `alloc`; the fixed-array and user-supplied-buffer paths are pure `no_std`.
- **`KmacXof*` (XOF mode)** in `sha3-kmac` is the variable-output form where the consumer streams arbitrary length out of the tag. rscrypto's `Kmac128` / `Kmac256` already handle variable output via the buffer length passed to `finalize_into` / `mac_into`. There is no separate `KmacXof128` or `KmacXof256` type: pass a longer buffer.
- **NIST SP 800-185 conformance.** Both implementations track the spec including the `right_encode` length suffix and `bytepad` block alignment. Outputs are bit-identical at every length tested in the harness (32 and 64 bytes); for assurance, run the harness yourself with your specific lengths.
- **Aside: hand-rolled cSHAKE-based KMAC.** Many existing codebases implement KMAC by hand on top of `sha3::CShake128` / `CShake256`. The migration target is `rscrypto::Kmac128` or `rscrypto::Kmac256`, matching your security level. Verify that your hand-rolled padding matches the SP 800-185 spec, specifically the `bytepad(encode_string(K))` step and the trailing `right_encode(L)`. If your output diverges from rscrypto's, your hand-rolled implementation has a spec bug; fix it before migrating.
