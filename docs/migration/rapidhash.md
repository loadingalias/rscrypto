# Migration: `rapidhash` → `rscrypto`

> Replace `rapidhash::v3::rapidhash_v3` with `rscrypto::RapidHash::hash`. Bit-equivalent, single feature flag, no version-pinning gymnastics.

Verified against `rapidhash = "4.4.1"` (V3 with avalanche, default secrets) and the `rscrypto` 0.3.1 line.
Evidence: `tests/rapidhash_differential.rs`.

## TL;DR

| | Before (`rapidhash` 4.x) | After (`rscrypto` 0.3.1) |
|---|---|---|
| Cargo dep | `rapidhash = "4.4"` | `rscrypto = { version = "0.3.1", features = ["rapidhash"] }` |
| Import | `use rapidhash::v3::rapidhash_v3;` | `use rscrypto::{FastHash, RapidHash};` |
| Call | `rapidhash_v3(data)` | `RapidHash::hash(data)` |

## Cargo.toml

```toml
# Before
[dependencies]
rapidhash = "4.4"
```

```toml
# After
[dependencies]
rscrypto = { version = "0.3.1", features = ["rapidhash"] }
```

## Algorithm map

| `rapidhash` function | rscrypto type | Output | Notes |
|---|---|---|---|
| `v3::rapidhash_v3` | `RapidHash` (alias for `RapidHash64`) | `u64` | V3 with avalanche, C++-compatible |
| `v3::rapidhash_v3_seeded` | `RapidHash::hash_with_seed` | `u64` | secrets are derived from the u64 seed |
| `v3::rapidhash_v3_inline::<true, _, _>` | `RapidHash::hash` (compile-time inlined where the optimiser can prove it) | `u64` | |
| `v3::rapidhash_v3_inline::<false, _, _>` (no avalanche) | `RapidHashFast` (alias for `RapidHashFast64`) | `u64` | not C++-compatible |
| 128-bit `rapidhash_v3` variant | `RapidHash128` | `u128` | |
| `v1::*`, `v2::*` (legacy) | not mapped — keep `rapidhash` for V1/V2 | | |

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

`RapidSecrets` is collapsed into the seed — rscrypto derives the secret schedule internally. Pass the seed directly.

### One-shot, no avalanche (faster)

```rust
// Before
use rapidhash::v3::rapidhash_v3_inline;
let h = rapidhash_v3_inline::<false, false, false>(b"123456789", &rapidhash::v3::DEFAULT_RAPID_SECRETS);
```

```rust
// After
use rscrypto::{FastHash, RapidHashFast};
let h = RapidHashFast::hash(b"123456789");
```

`RapidHashFast` skips the avalanche finisher — same speed/quality trade-off, no const generics to spell out.

### Streaming (via `core::hash::Hasher`)

```rust
// Before
use rapidhash::v3::RapidStreamHasher;        // verify exact name in your version
use core::hash::Hasher;
let mut hasher = RapidStreamHasher::new();
hasher.write(b"foo");
hasher.write(b"bar");
let h = hasher.finish();
```

```rust
// After
use rscrypto::RapidHasher;
use core::hash::Hasher;
let mut hasher = RapidHasher::default();
hasher.write(b"foo");
hasher.write(b"bar");
let h = hasher.finish();
```

For `HashMap` / `HashSet` use, `RapidBuildHasher` slots into `HashMap<K, V, RapidBuildHasher>`.

## Notes

- **Version selection.** `rapidhash` 4.x ships V1, V2, and V3 simultaneously. rscrypto targets V3 with avalanche (the C++-compatible default). If you depend on V1 or V2 outputs (e.g., for parity with an existing on-disk format), keep `rapidhash` as a sibling dependency.
- **`RapidSecrets` is hidden.** rscrypto's `hash_with_seed` takes a single `u64` and derives the full secret schedule internally. If you need to inject custom secrets (e.g., for HashDoS-resistant deployments with a runtime-randomised secret block), file an issue.
- **`RapidHashFast` opt-in.** The `Fast` variants drop the avalanche finisher for ~2x throughput on small inputs at the cost of weaker bit avalanche. Only choose `RapidHashFast` if you are not interoperating with the C++ `rapidhash` reference.
- **Streaming requires `alloc`.** `RapidHasher` and `RapidBuildHasher` are gated on `alloc`. The one-shot `RapidHash::hash` is fully `no_std`.
- **Not cryptographic.** `rapidhash` and `RapidHash` are non-cryptographic. Do not use either for password hashing, MAC, or any verification step where an attacker controls input. Use `Blake3` or `Sha256` for those.
