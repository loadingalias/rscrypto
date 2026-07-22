# Migration: `rsa` (RustCrypto) -> `rscrypto`

rscrypto supports RSA key import/export, key generation, RSA-PSS,
RSASSA-PKCS1-v1_5, OAEP, RSAES-PKCS1-v1_5, and protocol profile helpers. This
is a close primitive migration, not a type-compatible one.

The RSA test lane checks these operations against RustCrypto `rsa`, `ring`,
AWS-LC, CAVP/Wycheproof vectors, and OpenSSL CLI when available.
Evidence: `tests/rsa_wycheproof.rs`, `tests/rsa_nist_cavp.rs`,
`tests/rsa_public_key.rs`, `tests/rsa_profile_confusion.rs`, and the RSA
interoperability tests used by the stack migration guides.

## Cargo.toml

```toml
# Before
rsa = { version = "0.9", features = ["sha2"] }

# After
rscrypto = { version = "0.7.8", default-features = false, features = ["rsa"] }

# After, when generating keys or using randomized private/encryption APIs
rscrypto = { version = "0.7.8", default-features = false, features = ["rsa", "getrandom"] }
```

## Map

| RustCrypto `rsa` | rscrypto |
|---|---|
| `RsaPublicKey` | `RsaPublicKey` |
| `RsaPrivateKey` | `RsaPrivateKey` |
| PKCS#1 / PKCS#8 / SPKI import-export traits | inherent DER import-export methods |
| `Pss`, `Pkcs1v15Sign`, `Oaep` | `RsaPssProfile`, `RsaPkcs1v15Profile`, `RsaOaepProfile` |
| caller-managed RNG | `getrandom` wrappers, or encryption `*_with_random_fill` methods for no-std callers |

## Import Keys

```rust
// Before
use rsa::{RsaPrivateKey, pkcs1::DecodeRsaPrivateKey};

let private = RsaPrivateKey::from_pkcs1_der(private_der)?;
let public = private.to_public_key();
```

```rust
// After
use rscrypto::RsaPrivateKey;

let private = RsaPrivateKey::from_pkcs1_der(private_der)?;
let public = private.public_key();
```

For public keys, use `RsaPublicKey::from_spki_der()` for SPKI DER and
`RsaPublicKey::from_pkcs1_der()` for PKCS#1 DER. Default imports enforce the
modern policy: RSA-3072 through RSA-8192 with exponent `65537`. For deployed
RSA-2048 compatibility keys, import with
`RsaPublicKeyPolicy::legacy_verification()` and the `*_with_policy` parser.

## Generate Keys

```rust
// After
use rscrypto::{RsaKeyGenerationContract, RsaPrivateKey};

assert_eq!(
  RsaPrivateKey::GENERATION_CONTRACT,
  RsaKeyGenerationContract::Fips1865A13ProbablePrime,
);

let private_key = RsaPrivateKey::generate(3072)?;
let public_key = private_key.public_key();
```

Enable `getrandom` for key generation. rscrypto does not take a caller-managed
RNG parameter for `RsaPrivateKey::generate`; it seeds an internal HMAC_DRBG from
OS entropy and uses that DRBG for the FIPS 186-5 Appendix A.1.3 probable-prime
generation path. This is a code-level key-generation contract, not a
CMVP/FIPS 140-3 validation claim.

## Verify RSA-PSS

```rust
// Before
use rsa::{RsaPublicKey, pss::VerifyingKey};
use rsa::signature::Verifier;
use sha2::Sha256;

let verifying_key = VerifyingKey::<Sha256>::new(public_key);
verifying_key.verify(message, &signature)?;
```

```rust
// After
use rscrypto::{RsaPssProfile, RsaPublicKey};

let public_key = RsaPublicKey::from_spki_der(public_key_der)?;
public_key.verify_pss(RsaPssProfile::Sha256, message, signature_bytes)?;
```

## Verify RSASSA-PKCS1-v1_5

```rust
// Before
use rsa::{RsaPublicKey, pkcs1v15::VerifyingKey};
use rsa::signature::Verifier;
use sha2::Sha256;

let verifying_key = VerifyingKey::<Sha256>::new(public_key);
verifying_key.verify(message, &signature)?;
```

```rust
// After
use rscrypto::{RsaPkcs1v15Profile, RsaPublicKey};

let public_key = RsaPublicKey::from_spki_der(public_key_der)?;
public_key.verify_pkcs1v15(RsaPkcs1v15Profile::Sha256, message, signature_bytes)?;
```

## Sign

```rust
// After
use rscrypto::{RsaPkcs1v15Profile, RsaPrivateKey, RsaPssProfile};

let private_key = RsaPrivateKey::from_pkcs1_der(private_key_der)?;
let mut pss_signature = vec![0u8; private_key.signature_len()];
private_key.sign_pss(RsaPssProfile::Sha256, message, &mut pss_signature)?;

let mut pkcs1v15_signature = vec![0u8; private_key.signature_len()];
private_key.sign_pkcs1v15(RsaPkcs1v15Profile::Sha256, message, &mut pkcs1v15_signature)?;
```

rscrypto signs with blinding. Prefer the high-level signing methods unless a
test needs deterministic explicit salt/blinding hooks.

## OAEP

```rust
// After
use rscrypto::{RsaOaepProfile, RsaPrivateKey};

let private_key = RsaPrivateKey::from_pkcs1_der(private_key_der)?;
let public_key = private_key.public_key();

let mut ciphertext = vec![0u8; public_key.modulus().len()];
public_key.encrypt_oaep(RsaOaepProfile::Sha256, b"label", plaintext, &mut ciphertext)?;

let mut plaintext_out = vec![0u8; ciphertext.len()];
let len = private_key.decrypt_oaep(RsaOaepProfile::Sha256, b"label", &ciphertext, &mut plaintext_out)?;
plaintext_out.truncate(len);
```

Without `getrandom`, use `RsaPublicKey::encrypt_oaep_with_random_fill` or
`encrypt_pkcs1v15_with_random_fill` and pass a closure backed by your platform
RNG. The closure must write fresh unpredictable bytes on every call. Raw
encryption seed hooks are hidden diagnostics for tests and protocol harnesses,
not normal application API.

## Notes

- Profile types make SHA-2 selection explicit at the call site.
- Generic RustCrypto trait abstractions are not drop-in compatible. Migrate
  concrete RSA call sites first.
- RSA private operations are blinded where applicable and covered by dedicated
  RSA Miri and leakage gates.
- Legacy or certificate-policy behavior should be reviewed separately; do not
  hide it behind a blanket crate replacement.
