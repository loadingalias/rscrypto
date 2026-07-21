# Migration: `scrypt` (RustCrypto) → `rscrypto`

rscrypto separates deterministic key derivation from password-record handling:

- `Scrypt` is the raw RFC 7914 KDF.
- `ScryptPassword` generates canonical password records and verifies hostile PHC input under finite resource limits.

The implementation is checked against RFC 7914 vectors and the RustCrypto `scrypt 0.12.0` oracle in `tests/scrypt_vectors.rs` and `tests/scrypt_differential.rs`.

## Cargo features

```toml
# Raw scrypt KDF
rscrypto = { version = "0.7", default-features = false, features = ["scrypt"] }

# Password-record generation and verification
rscrypto = { version = "0.7", default-features = false, features = [
  "scrypt",
  "phc-strings",
  "getrandom",
] }
```

`getrandom` is needed only to generate password records. Verification needs `scrypt` and `phc-strings`.

## Raw KDF

RustCrypto:

```rust
use scrypt::{Params, scrypt};

let params = Params::new(15, 8, 1)?;
let mut output = [0u8; 32];
scrypt(password, salt, &params, &mut output)?;
```

rscrypto:

```rust
use rscrypto::{Scrypt, ScryptParams};

let params = ScryptParams::new(15, 8, 1)?;
let mut output = [0u8; 32];
Scrypt::derive(&params, password, salt, &mut output)?;
```

`ScryptParams::new(log_n, r, p)` rejects invalid profiles. As in RFC 7914 and RustCrypto, output length comes from the destination slice. The raw API accepts caller-provided salts, including the empty salt required by the RFC test vector; application salt policy belongs to the password-record API.

## Password records

`ScryptPassword` owns the generation profile and finite verification limits:

```rust
use rscrypto::{PasswordStatus, ScryptPassword};

let passwords = ScryptPassword::default();
let record = passwords.hash_password(password)?;

match passwords.verify_password(password, &record)? {
  PasswordStatus::Current => {}
  PasswordStatus::NeedsRehash => {
    let replacement = passwords.hash_password(password)?;
    // Store replacement atomically.
  }
}
```

Generation always uses a fresh 16-byte OS-random salt, a 32-byte verifier, and canonical `$scrypt$ln=...,r=...,p=...$...$...` encoding. There is no caller-salt password helper.

For a custom generation profile:

```rust
use rscrypto::{ScryptParams, ScryptPassword};

let generation = ScryptParams::new(17, 8, 1)?;
let passwords = ScryptPassword::new(generation)?;
```

To accept a broader finite migration envelope, derive limits from the largest deployment profile you intend to admit:

```rust
use rscrypto::{
  ScryptParams, ScryptPassword, ScryptVerificationLimits,
};

let generation = ScryptParams::new(17, 8, 1)?;
let accepted_profile = ScryptParams::new(18, 8, 1)?;
let limits = ScryptVerificationLimits::for_profile(accepted_profile);
let passwords = ScryptPassword::with_limits(generation, limits)?;
```

Limits compare the portable implementation's complete working-set bytes and `N × r × p` work. Verification rejects target-size overflow and over-budget parameters before base64 decoding, allocation, or KDF work.

## Compatibility boundary

The password verifier accepts only the protocol rscrypto emits:

- no PHC version segment;
- canonical ordered `ln,r,p` parameters;
- an 8–48-byte decoded salt;
- exactly 32 decoded verifier bytes;
- costs admitted by the configured finite limits.

Common canonical RustCrypto scrypt records with 32-byte outputs remain verifiable. Noncanonical encodings, different output sizes, and out-of-envelope profiles must be rehashed through the old stack during migration. rscrypto intentionally exposes neither a public PHC decoder nor an unbounded verifier.

## Operational notes

- `ScryptParams::default()` is `log_n=17, r=8, p=1`, the current [OWASP Password Storage Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html) baseline when Argon2id is unavailable.
- The raw algorithm accepts arbitrary salt lengths; generated password records use 16 bytes.
- Working buffers are zeroized on drop. Target-size overflow and allocation failure are distinct errors.
- scrypt ROMix uses password-derived, data-dependent memory access and is not a local side-channel constant-time claim. The final verifier traverses all expected bytes before returning one opaque result; any generated-code timing claim is limited to the exact configuration in the matching [release evidence](../../constant-time.md).
- scrypt requires `alloc` and is not FIPS 140-3 approved.
