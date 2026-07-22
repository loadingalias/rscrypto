# Migration: `argon2` (RustCrypto) → `rscrypto`

rscrypto separates two jobs that should not share an API:

- `Argon2d`, `Argon2i`, and `Argon2id` are deterministic raw KDFs.
- `Argon2idPassword` generates canonical password records and verifies hostile PHC input under finite resource limits.

The raw implementations are checked against RFC 9106 vectors and the RustCrypto `argon2 0.6.0-rc.8` oracle in `tests/argon2_vectors.rs`, `tests/argon2_differential.rs`, `tests/argon2_kernels.rs`, and `tests/argon2_parallel.rs`.

## Cargo features

```toml
# Raw Argon2 KDF
rscrypto = { version = "0.7.8", default-features = false, features = ["argon2"] }

# Password-record generation and verification
rscrypto = { version = "0.7.8", default-features = false, features = [
  "argon2",
  "phc-strings",
  "getrandom",
] }
```

`getrandom` is needed only to generate password records. Verification needs `argon2` and `phc-strings`.

## Raw KDF

RustCrypto:

```rust
use argon2::{Algorithm, Argon2, Params, Version};

let params = Params::new(19_456, 2, 1, Some(32))?;
let argon2 = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);
let mut output = [0u8; 32];
argon2.hash_password_into(password, salt, &mut output)?;
```

rscrypto:

```rust
use rscrypto::{Argon2Params, Argon2id};

let params = Argon2Params::new(19_456, 2, 1)?;
let mut output = [0u8; 32];
Argon2id::derive(&params, password, salt, &mut output)?;
```

`Argon2Params::new(memory_kib, time_cost, parallelism)` returns a valid-by-construction Argon2 v1.3 profile. Output length comes from the destination slice. Switch variants by changing the type:

```rust
Argon2id::derive(&params, password, salt, &mut output)?;
Argon2i::derive(&params, password, salt, &mut output)?;
Argon2d::derive(&params, password, salt, &mut output)?;
```

Argon2's optional secret and associated data are borrowed for one operation:

```rust
use rscrypto::Argon2Context;

let context = Argon2Context::new(pepper, associated_data);
Argon2id::derive_with_context(&params, context, password, salt, &mut output)?;
```

The raw API deliberately accepts caller-provided salts and variable output lengths because those are required KDF capabilities. It does not encode PHC strings.

## Password records

`Argon2idPassword` owns the generation profile and the finite verification limits:

```rust
use rscrypto::{Argon2idPassword, PasswordStatus};

let passwords = Argon2idPassword::default();
let record = passwords.hash_password(password)?;

match passwords.verify_password(password, &record)? {
  PasswordStatus::Current => {}
  PasswordStatus::NeedsRehash => {
    let replacement = passwords.hash_password(password)?;
    // Store replacement atomically.
  }
}
```

Generation always uses Argon2id v1.3, a fresh 16-byte OS-random salt, a 32-byte verifier, and canonical `$argon2id$v=19$m=...,t=...,p=...$...$...` encoding. There is no caller-salt password helper.

For a custom generation profile:

```rust
use rscrypto::{Argon2Params, Argon2idPassword};

let generation = Argon2Params::new(64 * 1024, 3, 1)?;
let passwords = Argon2idPassword::new(generation)?;
```

To accept a broader finite migration envelope, derive limits from the largest deployment profile you intend to admit:

```rust
use rscrypto::{
  Argon2Params, Argon2VerificationLimits, Argon2idPassword,
};

let generation = Argon2Params::new(19_456, 2, 1)?;
let accepted_profile = Argon2Params::new(64 * 1024, 3, 1)?;
let limits = Argon2VerificationLimits::for_profile(accepted_profile);
let passwords = Argon2idPassword::with_limits(generation, limits)?;
```

Limits compare the actual rounded matrix bytes, total block work, and parallelism—not merely each encoded integer in isolation. Verification rejects over-budget parameters before base64 decoding, allocation, or KDF work.

Pepper and associated data remain external to the PHC record:

```rust
use rscrypto::{Argon2Context, Argon2idPassword};

let passwords = Argon2idPassword::default();
let context = Argon2Context::new(pepper, tenant_id);
let record = passwords.hash_password_with_context(password, context)?;
let status = passwords.verify_password_with_context(password, &record, context)?;
```

## Compatibility boundary

The password verifier accepts only the protocol rscrypto emits:

- Argon2id, not Argon2d or Argon2i password records;
- Argon2 v1.3 (`v=19`), not v1.0;
- canonical ordered `m,t,p` parameters;
- an 8–48-byte decoded salt;
- exactly 32 decoded verifier bytes;
- costs admitted by the configured finite limits.

Common canonical RustCrypto Argon2id v1.3 records with 32-byte outputs remain verifiable. Noncanonical encodings, v1.0 records, different output sizes, and out-of-envelope profiles must be rehashed through the old stack during migration. rscrypto intentionally exposes neither a public PHC decoder nor an unbounded verifier.

## Operational notes

- `Argon2Params::default()` is `m=19_456 KiB, t=2, p=1`, the current [OWASP Password Storage Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html) baseline.
- Argon2d and Argon2id use data-dependent memory access and are not local side-channel constant-time claims. Argon2i is the data-independent raw variant.
- The memory matrix is zeroized on drop. Target-size overflow and allocation failure are distinct errors on raw derivation.
- The `parallel` feature enables Rayon lane parallelism when the profile and workload justify it.
- Argon2 requires `alloc` and is not FIPS 140-3 approved.
