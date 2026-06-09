# Migration: `aws-lc-rs` -> `rscrypto`

`aws-lc-rs` is a safe wrapper around AWS-LC. Migrate it
primitive-by-primitive. Do not treat it as a whole-crate swap.

The direct examples below are covered by `tests/migration_aws_lc_rs.rs`.

## Cargo.toml

```toml
# Before
aws-lc-rs = "1.17"

# After: enable only the primitives you use
rscrypto = { version = "0.4.0", default-features = false, features = ["sha2", "hmac", "hkdf", "pbkdf2", "aes-gcm", "chacha20poly1305", "ed25519", "x25519", "rsa"] }
```

## Map

| `aws-lc-rs` | rscrypto | Status |
|---|---|---|
| `digest::{SHA256, SHA384, SHA512}` | `Sha256`, `Sha384`, `Sha512` | Supported |
| `hmac::{HMAC_SHA256, HMAC_SHA384, HMAC_SHA512}` | `HmacSha256`, `HmacSha384`, `HmacSha512` | Supported |
| `hkdf::{HKDF_SHA256, HKDF_SHA384}` | `HkdfSha256`, `HkdfSha384` | Supported |
| `pbkdf2::{PBKDF2_HMAC_SHA256, PBKDF2_HMAC_SHA512}` | `Pbkdf2Sha256`, `Pbkdf2Sha512` | Supported |
| `aead::{AES_128_GCM, AES_256_GCM, CHACHA20_POLY1305}` | `Aes128Gcm`, `Aes256Gcm`, `ChaCha20Poly1305` | Supported |
| Ed25519 signing / verification | `Ed25519SecretKey`, `Ed25519PublicKey` | Supported |
| X25519 agreement | `X25519SecretKey`, `X25519PublicKey` | Supported |
| RSA-PSS / RSASSA-PKCS1-v1_5 verification | `RsaPublicKey`, `RsaPssProfile`, `RsaPkcs1v15Profile` | Supported |
| ECDSA, ML-DSA, TLS helpers, provider configuration, FIPS mode | none | Keep `aws-lc-rs` |

## Digest

```rust
// Before
let digest = aws_lc_rs::digest::digest(&aws_lc_rs::digest::SHA256, data);
let bytes = digest.as_ref();
```

```rust
// After
use rscrypto::Sha256;

let bytes = Sha256::digest(data);
```

## HMAC

```rust
// Before
let key = aws_lc_rs::hmac::Key::new(aws_lc_rs::hmac::HMAC_SHA256, key_bytes);
let tag = aws_lc_rs::hmac::sign(&key, data);
let bytes = tag.as_ref();
```

```rust
// After
use rscrypto::HmacSha256;

let bytes = HmacSha256::mac(key_bytes, data);
```

## HKDF

`aws-lc-rs` requires a `KeyType` for variable output lengths. rscrypto writes
directly into the caller's output buffer.

```rust
// Before
struct OutputLen(usize);

impl aws_lc_rs::hkdf::KeyType for OutputLen {
  fn len(&self) -> usize {
    self.0
  }
}

let salt = aws_lc_rs::hkdf::Salt::new(aws_lc_rs::hkdf::HKDF_SHA256, salt_bytes);
let prk = salt.extract(ikm);
let mut okm = [0u8; 42];
prk.expand(&[info], OutputLen(okm.len()))?.fill(&mut okm)?;
```

```rust
// After
use rscrypto::HkdfSha256;

let hkdf = HkdfSha256::new(salt_bytes, ikm);
let mut okm = [0u8; 42];
hkdf.expand(info, &mut okm)?;
```

## PBKDF2

```rust
// Before
let iterations = core::num::NonZeroU32::new(600_000).unwrap();
let mut key = [0u8; 32];
aws_lc_rs::pbkdf2::derive(
  aws_lc_rs::pbkdf2::PBKDF2_HMAC_SHA256,
  iterations,
  salt,
  password,
  &mut key,
);
```

```rust
// After
use rscrypto::Pbkdf2Sha256;

let mut key = [0u8; 32];
Pbkdf2Sha256::derive_key(password, salt, 600_000, &mut key)?;
```

## AEAD

`aws-lc-rs` appends the tag to the mutable ciphertext buffer. rscrypto can do
the same combined `ciphertext || tag` shape through `encrypt()`.

```rust
// Before
let key = aws_lc_rs::aead::LessSafeKey::new(
  aws_lc_rs::aead::UnboundKey::new(&aws_lc_rs::aead::AES_256_GCM, key_bytes)?,
);

let mut ciphertext_and_tag = plaintext.to_vec();
key.seal_in_place_append_tag(
  aws_lc_rs::aead::Nonce::assume_unique_for_key(*nonce_bytes),
  aws_lc_rs::aead::Aad::from(aad),
  &mut ciphertext_and_tag,
)?;
```

```rust
// After
use rscrypto::{Aes256Gcm, Aes256GcmKey, aead::Nonce96};

let cipher = Aes256Gcm::new(&Aes256GcmKey::from_bytes(*key_bytes));
let nonce = Nonce96::from_bytes(*nonce_bytes);

let mut ciphertext_and_tag = vec![0u8; plaintext.len() + 16];
cipher.encrypt(&nonce, aad, plaintext, &mut ciphertext_and_tag)?;
```

For `CHACHA20_POLY1305`, use `ChaCha20Poly1305` and
`ChaCha20Poly1305Key` with the same nonce and combined-output pattern.

## Ed25519

```rust
// Before
use aws_lc_rs::signature::KeyPair as _;

let keypair = aws_lc_rs::signature::Ed25519KeyPair::from_seed_unchecked(seed)?;
let public = keypair.public_key();
let signature = keypair.sign(message);

aws_lc_rs::signature::UnparsedPublicKey::new(
  &aws_lc_rs::signature::ED25519,
  public.as_ref(),
)
.verify(message, signature.as_ref())?;
```

```rust
// After
use rscrypto::Ed25519SecretKey;

let secret = Ed25519SecretKey::from_bytes(*seed);
let public = secret.public_key();
let signature = secret.sign(message);

public.verify(message, &signature)?;
```

## X25519

```rust
// Before
let private =
  aws_lc_rs::agreement::PrivateKey::from_private_key(&aws_lc_rs::agreement::X25519, secret_bytes)?;
let public = private.compute_public_key()?;

let peer = aws_lc_rs::agreement::UnparsedPublicKey::new(&aws_lc_rs::agreement::X25519, peer_public);
let shared = aws_lc_rs::agreement::agree(&private, &peer, (), |bytes| {
  let mut out = [0u8; 32];
  out.copy_from_slice(bytes);
  Ok::<[u8; 32], ()>(out)
})?;
```

```rust
// After
use rscrypto::{X25519PublicKey, X25519SecretKey};

let private = X25519SecretKey::from_bytes(*secret_bytes);
let public = private.public_key();

let peer = X25519PublicKey::from_bytes(*peer_public);
let shared = private.diffie_hellman(&peer)?;
let shared_bytes = shared.as_bytes();
```

## RSA Verification

```rust
// Before
let key = aws_lc_rs::signature::UnparsedPublicKey::new(
  &aws_lc_rs::signature::RSA_PSS_2048_8192_SHA256,
  public_key_der,
);
key.verify(message, signature)?;
```

```rust
// After
use rscrypto::{RsaPublicKey, RsaPssProfile};

let key = RsaPublicKey::from_spki_der(public_key_der)?;
key.verify_pss(RsaPssProfile::Sha256, message, signature)?;
```

For RSASSA-PKCS1-v1_5, use `RsaPkcs1v15Profile::Sha256` and
`verify_pkcs1v15()`.

## Unsupported AWS-LC Surfaces

Keep `aws-lc-rs` when you depend on:

- TLS provider integration or certificate/path validation.
- AWS-LC provider configuration, FIPS mode, or C library lifecycle behavior.
- ECDSA, ML-DSA, or other primitives that rscrypto does not expose.
- Direct access to AWS-LC internals through `aws-lc-sys`.

rscrypto is pure Rust cryptographic primitives with explicit feature flags. It
does not replace AWS-LC's C/ASM provider role in TLS stacks.
