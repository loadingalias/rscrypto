//! Authentication and key-derivation primitives.
//!
//! # Quick Start
//!
//! ```rust
//! # #[cfg(feature = "auth")]
//! # {
//! use rscrypto::{Ed25519Keypair, Ed25519SecretKey, HkdfSha256, HmacSha256, Mac};
//!
//! let key = b"shared-secret";
//! let data = b"hello world";
//!
//! let tag = HmacSha256::mac(key, data);
//!
//! let mut mac = HmacSha256::new(key);
//! mac.update(b"hello ");
//! mac.update(b"world");
//! assert!(mac.verify(&tag).is_ok());
//!
//! let mut okm = [0u8; 32];
//! HkdfSha256::new(b"salt", b"input key material").expand(b"context", &mut okm)?;
//! assert_ne!(okm, [0u8; 32]);
//!
//! let keypair = Ed25519Keypair::from_secret_key(Ed25519SecretKey::from_bytes([7u8; 32]));
//! let sig = keypair.sign(b"auth");
//! assert!(keypair.public_key().verify(b"auth", &sig).is_ok());
//!
//! let mut kmac = rscrypto::Kmac256::new(b"shared-secret", b"svc=v1");
//! kmac.update(b"auth");
//! let mut kmac_tag = [0u8; 32];
//! kmac.finalize_into(&mut kmac_tag);
//! assert!(rscrypto::Kmac256::verify_tag(b"shared-secret", b"svc=v1", b"auth", &kmac_tag).is_ok());
//!
//! let alice = rscrypto::X25519SecretKey::from_bytes([7u8; 32]);
//! let bob = rscrypto::X25519SecretKey::from_bytes([9u8; 32]);
//! let alice_shared = alice.diffie_hellman(&bob.public_key())?;
//! let bob_shared = bob.diffie_hellman(&alice.public_key())?;
//! assert!(alice_shared.ct_eq(&bob_shared).declassify());
//! # }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Feature Selection
//!
//! Use leaves for minimum size and bundles when you want the category:
//!
//! ```toml
//! [dependencies]
//! # MAC bundle: HMAC-SHA-2/SHA-3, KMAC128/256, standalone Poly1305, and SipHash
//! rscrypto = { version = "0.13", default-features = false, features = ["macs"] }
//!
//! # HKDF only
//! rscrypto = { version = "0.13", default-features = false, features = ["hkdf"] }
//!
//! # Ed25519 only
//! rscrypto = { version = "0.13", default-features = false, features = ["ed25519"] }
//!
//! # ECDSA P-256/P-384 signing and verification
//! rscrypto = { version = "0.13", default-features = false, features = ["ecdsa"] }
//!
//! # RSA
//! rscrypto = { version = "0.13", default-features = false, features = ["rsa"] }
//!
//! # Signature primitives
//! rscrypto = { version = "0.13", default-features = false, features = ["signatures"] }
//!
//! # X25519 only
//! rscrypto = { version = "0.13", default-features = false, features = ["x25519"] }
//!
//! # Everything in auth/key-derivation
//! rscrypto = { version = "0.13", default-features = false, features = ["auth"] }
//! ```
//!
//! # API Conventions
//!
//! - Fixed-output MACs use `Type::mac(key, data)` and `Type::verify_tag(key, data, tag)` for
//!   one-shot helpers, plus `new` / `update` / `finalize` / `reset` for streaming.
//! - KMAC is variable-output, so the streaming path uses `finalize_into`.
//! - Standalone Poly1305 consumes a `Poly1305OneTimeKey`; it intentionally does not implement the
//!   reusable-key `Mac` trait.
//! - HKDF uses `new(salt, ikm)` for extract state, then `expand` / `expand_array`; one-shot helpers
//!   are `derive` / `derive_array`.
//! - ML-KEM uses FIPS 203 names: encapsulation keys are public, decapsulation keys are secret.
//! - ML-DSA keeps context and deterministic or randomized signing explicit; its `Verifier`
//!   implementation selects pure ML-DSA with an empty context.
//! - SLH-DSA does the same, with pure and HashSLH-DSA keys as distinct types; signatures go
//!   into caller buffers and verification takes byte slices.
//! - Key generation uses `try_generate_with(fill)` for caller-owned entropy and `try_generate()` /
//!   `try_generate_keypair()` when `getrandom` is enabled.
//! - Native generic signing uses `TrySigner`, `TrySignerInto`, and `Verifier`; RSA generic
//!   signing/verifying goes through profile-bound wrappers.
//! - Public values use typed `from_bytes` / `to_bytes` / `as_bytes` wrappers.
//! - Secret values use typed `from_bytes` / `as_bytes` wrappers and require explicit
//!   `expose_secret()` opt-in for owned extraction.
//! - Verb-based operations such as `sign`, `verify`, and `diffie_hellman` stay on the typed
//!   wrappers.
//!
//! # Error Conventions
//!
//! - Authentication failures use [`crate::VerificationError`].
//! - HKDF oversized-output requests use `HkdfOutputLengthError`.
//! - X25519 low-order public inputs use `X25519Error`.
//!
//! # Modules
//!
//! - `argon2` - Argon2d/Argon2i/Argon2id password hashing (RFC 9106).
//! - `composite_mldsa` - composite ML-DSA-65 with Ed25519 or ECDSA P-256
//!   (draft-ietf-lamps-pq-composite-sigs-19).
//! - `ecdsa` - ECDSA P-256/SHA-256 and P-384/SHA-384 signing; SHA-256/SHA-384 verification on either curve.
//! - `ed25519` - Ed25519 key and signature types.
//! - `hmac` - HMAC-based authentication.
//! - `hmac_sha3` - HMAC-SHA3 authentication.
//! - `hkdf` - HKDF extract-then-expand key derivation.
//! - `hpke` - HPKE (RFC 9180) Base mode for qualified suites.
//! - `kmac` - KMAC128/KMAC256 variable-output MACs.
//! - `mldsa` - ML-DSA-44/65/87 signatures and HashML-DSA prehash profiles (FIPS 204).
//! - `mlkem` - ML-KEM typed key, ciphertext, and shared-secret foundations.
//! - `poly1305` - Standalone Poly1305 one-time authenticator.
//! - `siphash` - SipHash-1-3 and SipHash-2-4 keyed hashing for flood-resistant hash tables.
//! - `rsa` - RSA key import/export/generation, signing, verification, OAEP, and legacy
//!   RSAES-PKCS1-v1_5.
//! - `scrypt` - scrypt password hashing (RFC 7914).
//! - `slhdsa` - SLH-DSA signatures for all 12 FIPS 205 sets, pure and RFC 9909 HashSLH-DSA.
//! - `p256_ecdh` - ephemeral P-256 Diffie-Hellman key agreement.
//! - `p384_ecdh` - ephemeral P-384 Diffie-Hellman key agreement.
//! - `x25519` - X25519 Diffie-Hellman key agreement.
//! - `xwing` - X-Wing, the ML-KEM-768 and X25519 hybrid KEM
//!   (draft-connolly-cfrg-xwing-kem-11).

#[cfg(feature = "argon2")]
pub mod argon2;
#[cfg(any(feature = "mldsa65-ed25519", feature = "mldsa65-ecdsa-p256"))]
pub mod composite_mldsa;
#[cfg(any(
  feature = "ed25519",
  all(
    feature = "x25519",
    any(
      test,
      miri,
      not(any(
        all(
          target_arch = "aarch64",
          any(target_os = "macos", target_os = "linux"),
          not(feature = "portable-only")
        ),
        all(
          target_arch = "x86_64",
          any(target_os = "linux", target_os = "windows"),
          not(feature = "portable-only")
        )
      ))
    )
  )
))]
pub(crate) mod curve25519_edwards;
#[cfg(all(
  target_arch = "s390x",
  any(feature = "p256-ecdh", feature = "p384-ecdh"),
  not(feature = "portable-only"),
  not(miri)
))]
mod ecdh_s390x;
#[cfg(any(feature = "ecdsa-p256", feature = "ecdsa-p384"))]
pub mod ecdsa;
#[cfg(any(
  test,
  feature = "ecdsa-p256",
  feature = "ecdsa-p384",
  feature = "p384-ecdh",
  all(
    feature = "p256-ecdh",
    any(
      feature = "portable-only",
      miri,
      not(any(
        all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
        all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
      ))
    )
  )
))]
#[rustfmt::skip]
mod ecdsa_generator_tables;
#[cfg(feature = "ed25519")]
pub mod ed25519;
#[cfg(feature = "hkdf")]
pub mod hkdf;
#[cfg(feature = "hmac")]
pub mod hmac;
#[cfg(feature = "hmac-sha3")]
pub mod hmac_sha3;
#[cfg(any(feature = "hmac", feature = "hmac-sha3"))]
mod hmac_tag;
#[cfg(all(
  feature = "hpke",
  any(
    all(feature = "x25519", feature = "aes-gcm"),
    all(feature = "x-wing", feature = "chacha20poly1305")
  )
))]
pub mod hpke;
#[cfg(feature = "kmac")]
pub mod kmac;
#[cfg(feature = "ml-dsa")]
pub mod mldsa;
#[cfg(feature = "ml-kem")]
pub mod mlkem;
#[cfg(any(
  feature = "p256-ecdh",
  all(
    feature = "ecdsa-p256",
    any(
      all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
      all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
    )
  )
))]
mod p256_core;
#[cfg(feature = "p256-ecdh")]
pub mod p256_ecdh;
#[cfg(all(
  not(feature = "portable-only"),
  not(miri),
  any(
    all(
      feature = "ecdsa-p256",
      any(
        all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
        all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
      )
    ),
    all(
      feature = "p256-ecdh",
      any(
        all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
        all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
      )
    )
  )
))]
mod p256_platform;
#[cfg(any(
  feature = "p256-ecdh",
  all(
    feature = "ecdsa-p256",
    any(feature = "portable-only", miri),
    any(
      all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
      all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
    )
  )
))]
mod p256_portable;
#[cfg(feature = "p384-ecdh")]
pub mod p384_ecdh;
#[cfg(any(
  feature = "p384-ecdh",
  all(
    feature = "ecdsa-p384",
    target_arch = "x86_64",
    any(target_os = "linux", target_os = "windows"),
    not(feature = "portable-only"),
    not(miri)
  )
))]
mod p384_portable;
#[cfg(feature = "pbkdf2")]
pub mod pbkdf2;
#[cfg(all(feature = "phc-strings", any(feature = "argon2", feature = "scrypt")))]
pub(crate) mod phc;
#[cfg(feature = "poly1305")]
pub mod poly1305;
#[cfg(feature = "rsa")]
pub mod rsa;
#[cfg(feature = "scrypt")]
pub mod scrypt;
#[cfg(any(feature = "hkdf", feature = "pbkdf2"))]
mod sha2_words;
#[cfg(feature = "siphash")]
pub mod siphash;
#[cfg(feature = "slh-dsa")]
pub mod slhdsa;
#[cfg(feature = "x25519")]
pub mod x25519;
#[cfg(feature = "x-wing")]
pub mod xwing;

#[cfg(all(feature = "phc-strings", any(feature = "argon2", feature = "scrypt")))]
/// Result of a successful password verification.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PasswordStatus {
  /// The stored cost profile and salt length match the generation profile.
  Current,
  /// The password is correct, but the stored profile should be regenerated.
  NeedsRehash,
}

#[cfg(all(feature = "phc-strings", any(feature = "argon2", feature = "scrypt")))]
impl PasswordStatus {
  /// Whether the verified password should be hashed again with the current profile.
  #[must_use]
  pub const fn needs_rehash(self) -> bool {
    matches!(self, Self::NeedsRehash)
  }
}

/// Error from password-record generation with caller-provided entropy.
///
/// The entropy error is preserved for explicit recovery but omitted from
/// `Debug`, `Display`, and the standard error-source chain because entropy
/// providers may attach sensitive state to their errors.
#[cfg(all(feature = "phc-strings", any(feature = "argon2", feature = "scrypt")))]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum PasswordHashError<E, H> {
  /// The caller-provided entropy source failed.
  Entropy(E),
  /// Password derivation or record encoding failed.
  Hash(H),
}

#[cfg(all(feature = "phc-strings", any(feature = "argon2", feature = "scrypt")))]
impl<E, H> core::fmt::Debug for PasswordHashError<E, H> {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.write_str(match self {
      Self::Entropy(_) => "Entropy(..)",
      Self::Hash(_) => "Hash(..)",
    })
  }
}

#[cfg(all(feature = "phc-strings", any(feature = "argon2", feature = "scrypt")))]
impl<E, H> core::fmt::Display for PasswordHashError<E, H> {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.write_str(match self {
      Self::Entropy(_) => "password-record entropy source failed",
      Self::Hash(_) => "password-record hashing failed",
    })
  }
}

#[cfg(all(feature = "phc-strings", any(feature = "argon2", feature = "scrypt")))]
impl<E, H> core::error::Error for PasswordHashError<E, H>
where
  E: core::error::Error + 'static,
  H: core::error::Error + 'static,
{
}

#[cfg(feature = "argon2")]
pub use argon2::{Argon2Block, Argon2Context, Argon2Error, Argon2Params, Argon2d, Argon2i, Argon2id};
#[cfg(all(feature = "argon2", feature = "phc-strings"))]
pub use argon2::{Argon2VerificationLimits, Argon2idPassword};
#[cfg(any(feature = "mldsa65-ed25519", feature = "mldsa65-ecdsa-p256"))]
pub use composite_mldsa::{CompositeMlDsaError, CompositeMlDsaKeyError};
#[cfg(feature = "mldsa65-ecdsa-p256")]
pub use composite_mldsa::{
  MlDsa65EcdsaP256, MlDsa65EcdsaP256PreparedSecretKey, MlDsa65EcdsaP256PreparedSecretKeyStorage,
  MlDsa65EcdsaP256PublicKey, MlDsa65EcdsaP256SecretKey, MlDsa65EcdsaP256Signature,
};
#[cfg(feature = "mldsa65-ed25519")]
pub use composite_mldsa::{
  MlDsa65Ed25519, MlDsa65Ed25519PreparedSecretKey, MlDsa65Ed25519PreparedSecretKeyStorage, MlDsa65Ed25519PublicKey,
  MlDsa65Ed25519SecretKey, MlDsa65Ed25519Signature,
};

#[cfg(any(feature = "ecdsa-p256", feature = "ecdsa-p384"))]
pub use ecdsa::{EcdsaBlindedSigningError, EcdsaError, EcdsaKeyGenerationError};
#[cfg(feature = "ecdsa-p256")]
pub use ecdsa::{
  EcdsaP256Keypair, EcdsaP256PreparedPublicKey, EcdsaP256PublicKey, EcdsaP256SecretKey, EcdsaP256Signature,
};
#[cfg(feature = "ecdsa-p384")]
pub use ecdsa::{
  EcdsaP384Keypair, EcdsaP384PreparedPublicKey, EcdsaP384PublicKey, EcdsaP384SecretKey, EcdsaP384Signature,
};

#[cfg(feature = "ed25519")]
pub use ed25519::{Ed25519KeyError, Ed25519Keypair, Ed25519PublicKey, Ed25519SecretKey, Ed25519Signature};
#[cfg(feature = "hkdf")]
pub use hkdf::{HkdfOutputLengthError, HkdfSha256, HkdfSha384, HkdfSha512};
#[cfg(feature = "hmac")]
pub use hmac::{HmacSha256, HmacSha256Tag, HmacSha384, HmacSha384Tag, HmacSha512, HmacSha512Tag};
#[cfg(feature = "hmac-sha3")]
pub use hmac_sha3::{
  HmacSha3_224, HmacSha3_224Tag, HmacSha3_256, HmacSha3_256Tag, HmacSha3_384, HmacSha3_384Tag, HmacSha3_512,
  HmacSha3_512Tag,
};
#[cfg(all(feature = "hpke", feature = "x-wing", feature = "chacha20poly1305"))]
pub use hpke::HpkeXWingHkdfSha256ChaCha20Poly1305;
#[cfg(all(
  feature = "hpke",
  any(
    all(feature = "x25519", feature = "aes-gcm"),
    all(feature = "x-wing", feature = "chacha20poly1305")
  )
))]
pub use hpke::{HpkeError, HpkeReceiver, HpkeSender, HpkeSuite};
#[cfg(all(feature = "hpke", feature = "x25519", feature = "aes-gcm"))]
pub use hpke::{HpkeX25519HkdfSha256Aes128Gcm, HpkeX25519PrivateKey};
#[cfg(feature = "kmac")]
pub use kmac::{Kmac128, Kmac256};
#[cfg(feature = "ml-dsa")]
pub use mldsa::{
  MlDsa44, MlDsa44PreparedPublicKey, MlDsa44PreparedPublicKeyStorage, MlDsa44PreparedSecretKey,
  MlDsa44PreparedSecretKeyStorage, MlDsa44PublicKey, MlDsa44SecretKey, MlDsa44Seed, MlDsa44Signature, MlDsa65,
  MlDsa65PreparedPublicKey, MlDsa65PreparedPublicKeyStorage, MlDsa65PreparedSecretKey, MlDsa65PreparedSecretKeyStorage,
  MlDsa65PublicKey, MlDsa65SecretKey, MlDsa65Seed, MlDsa65Signature, MlDsa87, MlDsa87PreparedPublicKey,
  MlDsa87PreparedPublicKeyStorage, MlDsa87PreparedSecretKey, MlDsa87PreparedSecretKeyStorage, MlDsa87PublicKey,
  MlDsa87SecretKey, MlDsa87Seed, MlDsa87Signature, MlDsaError, MlDsaKeyError, MlDsaPrehash, MlDsaPrehashAlgorithm,
};
#[cfg(feature = "ml-kem")]
pub use mlkem::{
  MlKem512, MlKem512Ciphertext, MlKem512DecapsulationKey, MlKem512EncapsulationKey, MlKem512PreparedDecapsulationKey,
  MlKem512PreparedEncapsulationKey, MlKem512Seed, MlKem512SharedSecret, MlKem768, MlKem768Ciphertext,
  MlKem768DecapsulationKey, MlKem768EncapsulationKey, MlKem768PreparedDecapsulationKey,
  MlKem768PreparedEncapsulationKey, MlKem768Seed, MlKem768SharedSecret, MlKem1024, MlKem1024Ciphertext,
  MlKem1024DecapsulationKey, MlKem1024EncapsulationKey, MlKem1024PreparedDecapsulationKey,
  MlKem1024PreparedEncapsulationKey, MlKem1024Seed, MlKem1024SharedSecret, MlKemError, MlKemKeyError,
};

#[cfg(feature = "p256-ecdh")]
pub use p256_ecdh::{P256EphemeralSecret, P256KeyGenerationError, P256PublicKey, P256PublicKeyError, P256SharedSecret};

#[cfg(feature = "p384-ecdh")]
pub use p384_ecdh::{P384EphemeralSecret, P384KeyGenerationError, P384PublicKey, P384PublicKeyError, P384SharedSecret};

#[cfg(feature = "pbkdf2")]
pub use pbkdf2::{Pbkdf2Error, Pbkdf2Params, Pbkdf2Sha256, Pbkdf2Sha512, Pbkdf2VerifyPolicy};
#[cfg(feature = "poly1305")]
pub use poly1305::{Poly1305, Poly1305OneTimeKey, Poly1305Tag};
#[cfg(feature = "rsa")]
pub use rsa::{
  RsaBlindingPair, RsaEncryptionError, RsaJwtAlgorithm, RsaJwtVerifier, RsaKeyError, RsaKeyGenerationContract,
  RsaKeyGenerationError, RsaOaepProfile, RsaPkcs1v15Profile, RsaPrivateKey, RsaPrivateKeyParts, RsaPrivateOpError,
  RsaPrivateScratch, RsaProtocolAlgorithmError, RsaPssProfile, RsaPublicExponent, RsaPublicExponentPolicy,
  RsaPublicKey, RsaPublicKeyPolicy, RsaPublicOpError, RsaPublicScratch, RsaSignatureProfile, RsaSignatureSigner,
  RsaSignatureVerifier, RsaSigningWorkspace, RsaTlsSignatureSchemes, RsaX509PublicKey, RsaX509PublicKeyAlgorithm,
};

#[cfg(feature = "scrypt")]
pub use scrypt::{Scrypt, ScryptBlock, ScryptError, ScryptParams};
#[cfg(all(feature = "scrypt", feature = "phc-strings"))]
pub use scrypt::{ScryptPassword, ScryptVerificationLimits};
#[cfg(feature = "siphash")]
pub use siphash::{SipHash13, SipHash24, SipHashKey, SipRandomState};
#[cfg(feature = "slh-dsa")]
pub use slhdsa::{
  HashSlhDsaSha2_128fWithSha256, HashSlhDsaSha2_128fWithSha256PublicKey, HashSlhDsaSha2_128fWithSha256SecretKey,
  HashSlhDsaSha2_128sWithSha256, HashSlhDsaSha2_128sWithSha256PublicKey, HashSlhDsaSha2_128sWithSha256SecretKey,
  HashSlhDsaSha2_192fWithSha512, HashSlhDsaSha2_192fWithSha512PublicKey, HashSlhDsaSha2_192fWithSha512SecretKey,
  HashSlhDsaSha2_192sWithSha512, HashSlhDsaSha2_192sWithSha512PublicKey, HashSlhDsaSha2_192sWithSha512SecretKey,
  HashSlhDsaSha2_256fWithSha512, HashSlhDsaSha2_256fWithSha512PublicKey, HashSlhDsaSha2_256fWithSha512SecretKey,
  HashSlhDsaSha2_256sWithSha512, HashSlhDsaSha2_256sWithSha512PublicKey, HashSlhDsaSha2_256sWithSha512SecretKey,
  HashSlhDsaShake128fWithShake128, HashSlhDsaShake128fWithShake128PublicKey, HashSlhDsaShake128fWithShake128SecretKey,
  HashSlhDsaShake128sWithShake128, HashSlhDsaShake128sWithShake128PublicKey, HashSlhDsaShake128sWithShake128SecretKey,
  HashSlhDsaShake192fWithShake256, HashSlhDsaShake192fWithShake256PublicKey, HashSlhDsaShake192fWithShake256SecretKey,
  HashSlhDsaShake192sWithShake256, HashSlhDsaShake192sWithShake256PublicKey, HashSlhDsaShake192sWithShake256SecretKey,
  HashSlhDsaShake256fWithShake256, HashSlhDsaShake256fWithShake256PublicKey, HashSlhDsaShake256fWithShake256SecretKey,
  HashSlhDsaShake256sWithShake256, HashSlhDsaShake256sWithShake256PublicKey, HashSlhDsaShake256sWithShake256SecretKey,
  SlhDsaError, SlhDsaKeyError, SlhDsaSha2_128f, SlhDsaSha2_128fPublicKey, SlhDsaSha2_128fSecretKey, SlhDsaSha2_128s,
  SlhDsaSha2_128sPublicKey, SlhDsaSha2_128sSecretKey, SlhDsaSha2_192f, SlhDsaSha2_192fPublicKey,
  SlhDsaSha2_192fSecretKey, SlhDsaSha2_192s, SlhDsaSha2_192sPublicKey, SlhDsaSha2_192sSecretKey, SlhDsaSha2_256f,
  SlhDsaSha2_256fPublicKey, SlhDsaSha2_256fSecretKey, SlhDsaSha2_256s, SlhDsaSha2_256sPublicKey,
  SlhDsaSha2_256sSecretKey, SlhDsaShake128f, SlhDsaShake128fPublicKey, SlhDsaShake128fSecretKey, SlhDsaShake128s,
  SlhDsaShake128sPublicKey, SlhDsaShake128sSecretKey, SlhDsaShake192f, SlhDsaShake192fPublicKey,
  SlhDsaShake192fSecretKey, SlhDsaShake192s, SlhDsaShake192sPublicKey, SlhDsaShake192sSecretKey, SlhDsaShake256f,
  SlhDsaShake256fPublicKey, SlhDsaShake256fSecretKey, SlhDsaShake256s, SlhDsaShake256sPublicKey,
  SlhDsaShake256sSecretKey,
};
#[cfg(feature = "x25519")]
pub use x25519::{X25519Error, X25519PublicKey, X25519SecretKey, X25519SharedSecret};
#[cfg(feature = "x-wing")]
pub use xwing::{
  XWing, XWingCiphertext, XWingDecapsulationKey, XWingEncapsulationKey, XWingError, XWingPreparedDecapsulationKey,
  XWingPreparedEncapsulationKey, XWingSharedSecret,
};

pub use crate::traits::Mac;
