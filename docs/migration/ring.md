# Migration: `ring` -> `rscrypto`

`ring` mixes primitive APIs with protocol-shaped helpers. Migrate the primitive
surfaces directly; keep `ring` where you need ECDH P-256/P-384, random, or
TLS-oriented wrappers.

The direct examples below are covered by `tests/migration_ring.rs`.

## Cargo.toml

```toml
# Before
ring = "0.17"

# After: choose only the primitive features you use
rscrypto = { version = "0.7.8", default-features = false, features = ["sha2", "hmac", "hkdf", "pbkdf2", "aes-gcm", "chacha20poly1305", "ecdsa", "ed25519", "rsa"] }
```

## Map

| `ring` | rscrypto | Status |
|---|---|---|
| `digest::{SHA256, SHA384, SHA512}` | `Sha256`, `Sha384`, `Sha512` | Supported |
| `hmac::{HMAC_SHA256, HMAC_SHA384, HMAC_SHA512}` | `HmacSha256`, `HmacSha384`, `HmacSha512` | Supported |
| `hkdf::{HKDF_SHA256, HKDF_SHA384}` | `HkdfSha256`, `HkdfSha384` | Supported |
| `pbkdf2::{PBKDF2_HMAC_SHA256, PBKDF2_HMAC_SHA512}` | `Pbkdf2Sha256`, `Pbkdf2Sha512` | Supported |
| `aead::{AES_128_GCM, AES_256_GCM, CHACHA20_POLY1305}` | `Aes128Gcm`, `Aes256Gcm`, `ChaCha20Poly1305` | Supported |
| `signature::Ed25519KeyPair` | `Ed25519SecretKey`, `Ed25519PublicKey` | Supported |
| ECDSA P-256/P-384 SHA-2 signing and verification | `EcdsaP256SecretKey`, `EcdsaP384SecretKey`, `EcdsaP256PublicKey`, `EcdsaP384PublicKey` | Supported for raw and DER signatures |
| RSA-PSS / RSASSA-PKCS1-v1_5 public verification | `RsaPublicKey` profiles | Supported |
| X25519 agreement | `X25519SecretKey` | Not a direct call-site migration; `ring` is ephemeral-agreement shaped |
| ECDH P-256/P-384, random, TLS wrappers | none | Keep `ring` or another protocol crate |

## Digest

```rust
// Before
let digest = ring::digest::digest(&ring::digest::SHA256, data);
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
let key = ring::hmac::Key::new(ring::hmac::HMAC_SHA256, key_bytes);
let tag = ring::hmac::sign(&key, data);
let bytes = tag.as_ref();
```

```rust
// After
use rscrypto::HmacSha256;

let bytes = HmacSha256::mac(key_bytes, data).to_bytes();
```

## HKDF

```rust
// Before
struct OutputLen(usize);

impl ring::hkdf::KeyType for OutputLen {
  fn len(&self) -> usize {
    self.0
  }
}

let salt = ring::hkdf::Salt::new(ring::hkdf::HKDF_SHA256, salt_bytes);
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
ring::pbkdf2::derive(
  ring::pbkdf2::PBKDF2_HMAC_SHA256,
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

```rust
// Before
let key = ring::aead::LessSafeKey::new(
  ring::aead::UnboundKey::new(&ring::aead::AES_256_GCM, key_bytes)?,
);

let mut ciphertext_and_tag = plaintext.to_vec();
key.seal_in_place_append_tag(
  ring::aead::Nonce::assume_unique_for_key(*nonce_bytes),
  ring::aead::Aad::from(aad),
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
use ring::signature::KeyPair as _;

let keypair = ring::signature::Ed25519KeyPair::from_seed_unchecked(seed)?;
let public = keypair.public_key();
let signature = keypair.sign(message);

ring::signature::UnparsedPublicKey::new(&ring::signature::ED25519, public.as_ref())
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

## RSA Verification

```rust
// Before
let key = ring::signature::UnparsedPublicKey::new(
  &ring::signature::RSA_PSS_2048_8192_SHA256,
  public_key_der,
);
key.verify(message, signature)?;
```

```rust
// After
use rscrypto::{RsaPublicKey, RsaPssProfile};

let key = RsaPublicKey::from_pkcs1_der(public_key_der)?;
key.verify_pss(RsaPssProfile::Sha256, message, signature)?;
```

For RSASSA-PKCS1-v1_5, use `RsaPkcs1v15Profile::Sha256` and
`verify_pkcs1v15()`. If your existing input is SPKI DER instead of PKCS#1
public-key DER, use `RsaPublicKey::from_spki_der()`.

## X25519

Do not treat `ring::agreement` as a mechanical migration. `ring` exposes
ephemeral private keys consumed by `agree_ephemeral`; rscrypto exposes typed
static/reusable `X25519SecretKey` and rejects all-zero shared secrets.

Migrate X25519 at the protocol boundary: identify the peer-public-key encoding,
the KDF applied to the raw shared secret, and whether reusable private keys are
acceptable for your protocol.

## Keep `ring`

Keep `ring` or another protocol crate for:

- ECDH P-256/P-384.
- Random generation APIs.
- TLS-oriented wrappers.
- `ring` verifier constants outside the SHA-2 RSA profiles listed above.
