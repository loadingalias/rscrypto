# Migration: `rapidhash` → `rscrypto`

> Replace `rapidhash::v3::rapidhash_v3` with `rscrypto::RapidHash::hash`. Bit-equivalent, single feature flag, no version-pinning gymnastics.

Verified against `rapidhash = "4.5.1"` (V3 with avalanche, default secrets) and the `rscrypto` 0.6 line.
Evidence: `tests/rapidhash_differential.rs`.

## TL;DR

| | Before (`rapidhash` 4.x) | After (`rscrypto` 0.6) |
|---|---|---|
| Cargo dep | `rapidhash = "4.5.1"` | `rscrypto = { version = "0.6", features = ["rapidhash"] }` |
| Import | `use rapidhash::v3::rapidhash_v3;` | `use rscrypto::{FastHash, RapidHash};` |
| Call | `rapidhash_v3(data)` | `RapidHash::hash(data)` |

## Cargo.toml

```toml
# Before
[dependencies]
rapidhash = "4.5.1"
```

```toml
# After
[dependencies]
rscrypto = { version = "0.6", features = ["rapidhash"] }
```

## Algorithm map

| `rapidhash` function | rscrypto type | Output | Notes |
|---|---|---|---|
| `v3::rapidhash_v3` | `RapidHash` (alias for `RapidHash64`) | `u64` | V3 with avalanche, C++-compatible |
| `v3::rapidhash_v3_seeded` | `RapidHash::hash_with_seed` | `u64` | secrets are derived from the u64 seed |
| `v3::rapidhash_v3_inline::<true, _, _>` | `RapidHash::hash` (compile-time inlined where the optimiser can prove it) | `u64` | |
| `fast::RapidHasher::write` | `RapidHashFast64` | `u64` | native-endian, implementation-defined, not a portable V3 fingerprint |
| 128-bit `rapidhash_v3` variant | `RapidHash128` | `u128` | |
| `v1::*`, `v2::*` (legacy) | not mapped: keep `rapidhash` for V1/V2 | | |

## API patterns

### One-shot 64-bit

```rust
// Before
use rapidhash::v3::rapidhash_v3;
let h: u64 = rapidhash_v3(b"123456789");
```

```rust
// After
use rscrypto::{FastHash, RapidHash};
let h: u64 = RapidHash::hash(b"123456789");
```

### One-shot 64-bit, seeded

```rust
// Before
use rapidhash::v3::{rapidhash_v3_seeded, RapidSecrets};
const SECRETS: RapidSecrets = RapidSecrets::seed(0xDEADBEEF);
let h = rapidhash_v3_seeded(b"123456789", &SECRETS);
```

```rust
// After
use rscrypto::{FastHash, RapidHash};
let h = RapidHash::hash_with_seed(0xDEADBEEF, b"123456789");
```

`RapidSecrets` is collapsed into the seed: rscrypto derives the secret schedule internally. Pass the seed directly.

### One-shot collection-key algorithm

```rust
// Before
use core::hash::Hasher;
let mut hasher = rapidhash::fast::RapidHasher::new(0);
hasher.write(b"123456789");
let h = hasher.finish();
```

```rust
// After
use rscrypto::{FastHash, RapidHashFast64};
let h = RapidHashFast64::hash(b"123456789");
```

`RapidHashFast64` matches one upstream fast-hasher byte write with seed zero on little-endian targets. Its native-endian output is for in-process collection work, not persistence or interchange.

### Streaming (via `core::hash::Hasher`)

```rust
// Before (`rapidhash` 4.5)
use rapidhash::v3::{RapidSecrets, RapidStreamHasherV3};
let secrets = RapidSecrets::seed_cpp(0);
let mut hasher = RapidStreamHasherV3::new(&secrets);
hasher.write(b"foo");
hasher.write(b"bar");
let h = hasher.finish();
```

```rust
// After
use rscrypto::RapidStreamHasher;
use core::hash::Hasher;
let mut hasher = RapidStreamHasher::default();
hasher.write(b"foo");
hasher.write(b"bar");
let h = hasher.finish();
```

For `HashMap` / `HashSet` use, `RapidBuildHasher` slots into
`HashMap<K, V, RapidBuildHasher>` and produces `RapidHasher`. The builder uses a
fixed seed and is intended for trusted keys. Its output is implementation-defined,
not a C++-compatible V3 fingerprint. Use `RapidStreamHasher` when separate writes
must equal hashing their byte concatenation.

## Notes

- **Version selection.** `rapidhash` 4.x ships V1, V2, and V3 simultaneously. rscrypto targets V3 with avalanche (the C++-compatible default). If you depend on V1 or V2 outputs (e.g., for parity with an existing on-disk format), keep `rapidhash` as a sibling dependency.
- **Custom secrets are not exposed.** rscrypto's `hash_with_seed` derives the V3 secret schedule from one `u64`. A borrowed custom-secret API would add lifetimes and configuration states without improving the trusted-input paths this module targets. Keep `rapidhash` when exact custom-secret interoperability is required.
- **`RapidHashFast64` opt-in.** The fast variants use upstream's distinct native-endian in-memory algorithm. Do not persist or exchange their output.
- **Allocation-free state.** `RapidStreamHasher` keeps bounded inline V3 streaming state. `RapidHasher` is the smaller collection-key state produced by `RapidBuildHasher`. All three work without `alloc`, including in pure `no_std` builds.
- **Trusted collection keys only.** `RapidBuildHasher` is deterministic and does not draw entropy. Retain the standard library's randomized map hasher when an attacker can choose map or set keys.
- **Not cryptographic.** `rapidhash` and `RapidHash` are non-cryptographic. Do not use either for password hashing, MAC, or any verification step where an attacker controls input. Use `Blake3` or `Sha256` for those.
