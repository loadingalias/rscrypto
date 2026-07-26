# Migration: `rapidhash` → `rscrypto`

Use `RapidHash64` for portable V3 fingerprints. Its one-shot output is
bit-identical to `rapidhash` 4.5.1 V3 with the C++ seed derivation, and its
inherent methods can run at compile time.

## Dependency and one-shot hashing

```toml
[dependencies]
rscrypto = { version = "0.7.8", features = ["rapidhash"] }
```

```rust
use rscrypto::RapidHash64;

const EMPTY: u64 = RapidHash64::hash(b"");
let default = RapidHash64::hash(b"123456789");
let seeded = RapidHash64::hash_with_seed(0xDEADBEEF, b"123456789");
assert_ne!(default, seeded);
```

`RapidHash64::hash` replaces `rapidhash::v3::rapidhash_v3`.
`RapidHash64::hash_with_seed(seed, data)` matches
`rapidhash_v3_seeded(data, &RapidSecrets::seed_cpp(seed))`; arbitrary custom
secret schedules remain unsupported.

## Streaming

`RapidStreamHasher` is allocation-free. Separate writes produce the same
result as hashing their concatenation with `RapidHash64`.

```rust
use core::hash::Hasher;
use rscrypto::{RapidHash64, RapidStreamHasher};

let mut hasher = RapidStreamHasher::new();
hasher.write(b"foo");
hasher.write(b"bar");
assert_eq!(hasher.finish(), RapidHash64::hash(b"foobar"));
```

## Collections

Use explicit deterministic state only when keys are trusted and reproducible
hashes are required.

```rust
use std::collections::HashMap;
use rscrypto::RapidSeededState;

let state = RapidSeededState::new(42);
let mut map = HashMap::with_hasher(state);
map.insert("key", 7);
```

For non-interactive, untrusted-key workloads where RapidHash's limited HashDoS
hardening is sufficient, create `RapidRandomState` from a fallible entropy
source. Add the `getrandom` feature to use the OS-backed constructor:

```rust
use std::collections::HashMap;
use rscrypto::RapidRandomState;

let state = RapidRandomState::try_new()?;
let mut map = HashMap::with_hasher(state);
map.insert("key", 7);
# Ok::<(), getrandom::Error>(())
```

Constrained `no_std` integrations can omit `getrandom` and pass a
platform-specific CSPRNG callback that fills the entire buffer:

```rust
use rscrypto::RapidRandomState;

let state = RapidRandomState::try_new_with(platform_csprng_fill)?;
```

Randomized RapidHash state changes the collision structure between states; it
does not provide cryptographic collision resistance. RapidHash remains
unsuitable for MACs, password hashing, signatures, or fingerprints used as
authenticity checks.

## Deliberately unsupported surfaces

- V1 and V2 output compatibility;
- 128-bit and native-endian “fast” variants;
- public custom-secret schedules;
- deterministic default collection state;
- infallible ambient-entropy constructors.

Keep `rapidhash` as a sibling dependency if an existing format or protocol
requires one of those outputs. One-shot V3 and collection-state seed schedules
are checked against `rapidhash` in `tests/rapidhash_differential.rs`; streaming
chunk invariance is covered by the module tests.
