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
//! assert_eq!(alice_shared, bob_shared);
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
//! # MAC bundle: HMAC-SHA-2/SHA-3, KMAC128/256, and standalone Poly1305
//! rscrypto = { version = "0.6.4", default-features = false, features = ["macs"] }
//!
//! # HKDF only
//! rscrypto = { version = "0.6.4", default-features = false, features = ["hkdf"] }
//!
//! # Ed25519 only
//! rscrypto = { version = "0.6.4", default-features = false, features = ["ed25519"] }
//!
//! # ECDSA P-256/P-384 signing and verification
//! rscrypto = { version = "0.6.4", default-features = false, features = ["ecdsa"] }
//!
//! # RSA
//! rscrypto = { version = "0.6.4", default-features = false, features = ["rsa"] }
//!
//! # Signature primitives
//! rscrypto = { version = "0.6.4", default-features = false, features = ["signatures"] }
//!
//! # X25519 only
//! rscrypto = { version = "0.6.4", default-features = false, features = ["key-exchange"] }
//!
//! # Everything in auth/key-derivation
//! rscrypto = { version = "0.6.4", default-features = false, features = ["auth"] }
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
//! - `ecdsa` - ECDSA P-256/SHA-256 and P-384/SHA-384 signing and verification.
//! - `ed25519` - Ed25519 key and signature types.
//! - `hmac` - HMAC-based authentication.
//! - `hmac_sha3` - HMAC-SHA3 authentication.
//! - `hkdf` - HKDF extract-then-expand key derivation.
//! - `kmac` - KMAC128/KMAC256 variable-output MACs.
//! - `mlkem` - ML-KEM typed key, ciphertext, and shared-secret foundations.
//! - `poly1305` - Standalone Poly1305 one-time authenticator.
//! - `rsa` - RSA key import/export/generation, signing, verification, OAEP, and legacy
//!   RSAES-PKCS1-v1_5.
//! - `scrypt` - scrypt password hashing (RFC 7914).
//! - `x25519` - X25519 Diffie-Hellman key agreement.

#[cfg(feature = "argon2")]
pub mod argon2;
#[cfg(any(feature = "ed25519", feature = "x25519"))]
pub(crate) mod curve25519_edwards;
#[cfg(any(feature = "ecdsa-p256", feature = "ecdsa-p384"))]
pub mod ecdsa;
#[cfg(feature = "ed25519")]
pub mod ed25519;
#[cfg(feature = "hkdf")]
pub mod hkdf;
#[cfg(feature = "hmac")]
pub mod hmac;
#[cfg(feature = "hmac-sha3")]
pub mod hmac_sha3;
#[cfg(feature = "kmac")]
pub mod kmac;
#[cfg(feature = "ml-kem")]
pub mod mlkem;
#[cfg(feature = "pbkdf2")]
pub mod pbkdf2;
#[cfg(feature = "phc-strings")]
pub(crate) mod phc;
#[cfg(feature = "poly1305")]
pub mod poly1305;
#[cfg(feature = "rsa")]
pub mod rsa;
#[cfg(feature = "scrypt")]
pub mod scrypt;
#[cfg(feature = "x25519")]
pub mod x25519;

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

#[cfg(feature = "argon2")]
pub use argon2::{Argon2Context, Argon2Error, Argon2Params, Argon2d, Argon2i, Argon2id};
#[cfg(all(feature = "argon2", feature = "phc-strings"))]
pub use argon2::{Argon2VerificationLimits, Argon2idPassword};
#[cfg(all(feature = "diag", feature = "ed25519"))]
pub use curve25519_edwards::diag_ed25519_select_basepoint_cached_limb_digest;
#[cfg(all(feature = "diag", feature = "ed25519", target_arch = "x86_64"))]
pub use curve25519_edwards::{
  diag_ed25519_select_basepoint_cached_avx2_limb_digest, diag_ed25519_select_basepoint_cached_ifma_limb_digest,
};
#[cfg(any(feature = "ecdsa-p256", feature = "ecdsa-p384"))]
pub use ecdsa::{EcdsaError, EcdsaKeyGenerationError};
#[cfg(feature = "ecdsa-p256")]
pub use ecdsa::{EcdsaP256Keypair, EcdsaP256PublicKey, EcdsaP256SecretKey, EcdsaP256Signature};
#[cfg(feature = "ecdsa-p384")]
pub use ecdsa::{EcdsaP384Keypair, EcdsaP384PublicKey, EcdsaP384SecretKey, EcdsaP384Signature};
#[cfg(all(feature = "diag", feature = "ecdsa-p256"))]
pub use ecdsa::{
  diag_ecdsa_p256_basepoint_blinded_limb_digest, diag_ecdsa_p256_final_multiply_limb_digest,
  diag_ecdsa_p256_nonce_inverse_limb_digest, diag_ecdsa_p256_nonce_reduce_limb_digest,
  diag_ecdsa_p256_order_mul_blinded_fixed_r_limb_digest, diag_ecdsa_p256_order_mul_fixed_r_limb_digest,
  diag_ecdsa_p256_reduce_wide_order_limb_digest, diag_ecdsa_p256_scalar_finish_limb_digest,
  diag_ecdsa_p256_select_signing_generator_affine_limb_digest,
};
#[cfg(all(feature = "diag", feature = "ecdsa-p384"))]
pub use ecdsa::{
  diag_ecdsa_p384_basepoint_blinded_limb_digest, diag_ecdsa_p384_basepoint_r_limb_digest,
  diag_ecdsa_p384_final_multiply_limb_digest, diag_ecdsa_p384_nonce_inverse_limb_digest,
  diag_ecdsa_p384_nonce_reduce_limb_digest, diag_ecdsa_p384_order_mul_fixed_r_limb_digest,
  diag_ecdsa_p384_reduce_wide_order_limb_digest, diag_ecdsa_p384_scalar_finish_limb_digest,
  diag_ecdsa_p384_select_signing_generator_affine_limb_digest,
};
#[cfg(all(
  feature = "diag",
  feature = "ed25519",
  target_arch = "aarch64",
  any(target_os = "macos", target_os = "linux"),
  not(feature = "portable-only"),
  not(miri)
))]
pub use ed25519::diag_ed25519_verify_aarch64_asm_double_scalar_digest;
#[cfg(all(feature = "diag", feature = "ed25519"))]
pub use ed25519::{
  DiagEd25519VerifyScalars, diag_ed25519_verify_challenge_reduce_digest,
  diag_ed25519_verify_portable_double_scalar_digest, diag_ed25519_verify_public_decode_digest,
  diag_ed25519_verify_r_decode_digest, diag_ed25519_verify_scalars,
};
#[cfg(feature = "ed25519")]
pub use ed25519::{Ed25519Keypair, Ed25519PublicKey, Ed25519SecretKey, Ed25519Signature};
#[cfg(feature = "hkdf")]
pub use hkdf::{HkdfOutputLengthError, HkdfSha256, HkdfSha384, HkdfSha512};
#[cfg(all(feature = "diag", feature = "hkdf"))]
pub use hkdf::{diag_hkdf_sha256_derive_portable, diag_hkdf_sha384_derive_portable, diag_hkdf_sha512_derive_portable};
#[cfg(feature = "hmac")]
pub use hmac::{HmacSha256, HmacSha256Tag, HmacSha384, HmacSha384Tag, HmacSha512, HmacSha512Tag};
#[cfg(all(feature = "diag", feature = "hmac"))]
pub use hmac::{diag_hmac_sha256_verify_portable, diag_hmac_sha384_verify_portable, diag_hmac_sha512_verify_portable};
#[cfg(feature = "hmac-sha3")]
pub use hmac_sha3::{
  HmacSha3_224, HmacSha3_224Tag, HmacSha3_256, HmacSha3_256Tag, HmacSha3_384, HmacSha3_384Tag, HmacSha3_512,
  HmacSha3_512Tag,
};
#[cfg(feature = "kmac")]
pub use kmac::{Kmac128, Kmac256};
#[cfg(feature = "ml-kem")]
pub use mlkem::{
  MlKem512, MlKem512Ciphertext, MlKem512DecapsulationKey, MlKem512EncapsulationKey, MlKem512PreparedDecapsulationKey,
  MlKem512PreparedEncapsulationKey, MlKem512SharedSecret, MlKem768, MlKem768Ciphertext, MlKem768DecapsulationKey,
  MlKem768EncapsulationKey, MlKem768PreparedDecapsulationKey, MlKem768PreparedEncapsulationKey, MlKem768SharedSecret,
  MlKem1024, MlKem1024Ciphertext, MlKem1024DecapsulationKey, MlKem1024EncapsulationKey,
  MlKem1024PreparedDecapsulationKey, MlKem1024PreparedEncapsulationKey, MlKem1024SharedSecret, MlKemError,
};
#[cfg(all(
  feature = "diag",
  feature = "ml-kem",
  target_arch = "aarch64",
  any(target_os = "macos", target_os = "linux"),
  not(miri),
  not(feature = "portable-only")
))]
pub use mlkem::{
  diag_mlkem_aarch64_multiply_ntts_add_assign_asm_digest, diag_mlkem_aarch64_multiply_ntts_add_assign_asm_input_digest,
  diag_mlkem768_aarch64_multiply_ntts_accumulate_asm_digest,
  diag_mlkem768_aarch64_multiply_ntts_accumulate_asm_input_digest,
  diag_mlkem1024_aarch64_multiply_ntts_accumulate_asm_digest,
  diag_mlkem1024_aarch64_multiply_ntts_accumulate_asm_input_digest,
};
#[cfg(all(
  feature = "diag",
  feature = "ml-kem",
  target_arch = "aarch64",
  not(miri),
  not(feature = "portable-only")
))]
pub use mlkem::{diag_mlkem_aarch64_ntt_neon_digest, diag_mlkem_aarch64_ntt_neon_input_digest};
#[cfg(all(feature = "diag", feature = "ml-kem"))]
pub use mlkem::{
  diag_mlkem_compress_decompress_values_digest, diag_mlkem_from_montgomery_product_domain_input_digest,
  diag_mlkem_inverse_ntt_montgomery_product_add_assign_input_digest,
  diag_mlkem_inverse_ntt_montgomery_product_input_digest, diag_mlkem_multiply_ntts_add_assign_input_digest,
  diag_mlkem_ntt_input_digest, diag_mlkem_to_montgomery_product_domain_input_digest,
  diag_mlkem512_keygen_secret_noise_digest, diag_mlkem512_multiply_ntts_accumulate_digest,
  diag_mlkem512_multiply_ntts_accumulate_input_digest, diag_mlkem768_keygen_secret_noise_digest,
  diag_mlkem768_multiply_ntts_accumulate_digest, diag_mlkem1024_keygen_secret_noise_digest,
  diag_mlkem1024_multiply_ntts_accumulate_digest, diag_mlkem1024_multiply_ntts_accumulate_input_digest,
};
#[cfg(feature = "pbkdf2")]
pub use pbkdf2::{Pbkdf2Error, Pbkdf2Params, Pbkdf2Sha256, Pbkdf2Sha512, Pbkdf2VerifyPolicy};
#[cfg(all(feature = "diag", feature = "pbkdf2"))]
pub use pbkdf2::{diag_pbkdf2_sha256_verify_portable, diag_pbkdf2_sha512_verify_portable};
#[cfg(feature = "poly1305")]
pub use poly1305::{Poly1305, Poly1305OneTimeKey, Poly1305Tag};
#[cfg(feature = "rsa")]
pub use rsa::{
  RsaEncryptionError, RsaKeyError, RsaKeyGenerationContract, RsaKeyGenerationError, RsaOaepProfile, RsaPkcs1v15Profile,
  RsaPrivateKey, RsaPrivateKeyParts, RsaPrivateOpError, RsaPrivateScratch, RsaProtocolAlgorithmError, RsaPssProfile,
  RsaPublicExponent, RsaPublicExponentPolicy, RsaPublicKey, RsaPublicKeyPolicy, RsaPublicOpError, RsaPublicScratch,
  RsaSignatureProfile, RsaSignatureSigner, RsaSignatureVerifier, RsaTlsSignatureSchemes, RsaX509PublicKey,
  RsaX509PublicKeyAlgorithm,
};
#[cfg(all(feature = "rsa", feature = "diag"))]
pub use rsa::{
  diag_rsa_import_pkcs8_private_key_der_stage, diag_rsa_private_component_validation_32,
  diag_rsa_private_select_window_power_4, diag_rsa_validate_pkcs8_private_key_der,
  diag_rsa_validate_pkcs8_private_key_der_stage,
};
#[cfg(feature = "scrypt")]
pub use scrypt::{Scrypt, ScryptError, ScryptParams};
#[cfg(all(feature = "scrypt", feature = "phc-strings"))]
pub use scrypt::{ScryptPassword, ScryptVerificationLimits};
#[cfg(feature = "x25519")]
pub use x25519::{X25519Error, X25519PublicKey, X25519SecretKey, X25519SharedSecret};

pub use crate::traits::Mac;
