//! Authentication and key-derivation primitives.
//!
//! # Quick Start
//!
//! ```rust
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
//! assert_eq!(mac.finalize(), tag);
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
//! assert!(rscrypto::Kmac256::verify(b"shared-secret", b"svc=v1", b"auth", &kmac_tag).is_ok());
//!
//! let alice = rscrypto::X25519SecretKey::from_bytes([7u8; 32]);
//! let bob = rscrypto::X25519SecretKey::from_bytes([9u8; 32]);
//! let alice_shared = alice.diffie_hellman(&bob.public_key())?;
//! let bob_shared = bob.diffie_hellman(&alice.public_key())?;
//! assert_eq!(alice_shared, bob_shared);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Feature Selection
//!
//! Use leaves for minimum size and bundles when you want the category:
//!
//! ```toml
//! [dependencies]
//! # HMAC + KMAC only
//! rscrypto = { version = "0.1", default-features = false, features = ["macs"] }
//!
//! # HKDF only
//! rscrypto = { version = "0.1", default-features = false, features = ["hkdf"] }
//!
//! # Ed25519 only
//! rscrypto = { version = "0.1", default-features = false, features = ["signatures"] }
//!
//! # X25519 only
//! rscrypto = { version = "0.1", default-features = false, features = ["key-exchange"] }
//!
//! # Everything in auth/key-derivation
//! rscrypto = { version = "0.1", default-features = false, features = ["auth"] }
//! ```
//!
//! # API Conventions
//!
//! - MACs use `Type::mac(key, data)` and `Type::verify_tag(key, data, tag)` for one-shot helpers,
//!   plus `new` / `update` / `finalize` / `reset` for streaming.
//! - KMAC is variable-output, so the streaming path uses `finalize_into`.
//! - HKDF uses `new(salt, ikm)` for extract state, then `expand` / `expand_array`; one-shot helpers
//!   are `derive` / `derive_array`.
//! - Signature and key-exchange types use typed `from_bytes` / `to_bytes` / `as_bytes` wrappers
//!   plus verb-based operations such as `sign`, `verify`, and `diffie_hellman`.
//!
//! # Error Conventions
//!
//! - Authentication failures use [`crate::VerificationError`].
//! - HKDF oversized-output requests use [`HkdfOutputLengthError`].
//! - X25519 low-order public inputs use [`X25519Error`].
//!
//! # Modules
//!
//! - [`ed25519`] - Ed25519 key and signature types.
//! - [`hmac`] - HMAC-based authentication.
//! - [`hkdf`] - HKDF extract-then-expand key derivation.
//! - [`kmac`] - KMAC256 variable-output MAC.
//! - [`x25519`] - X25519 Diffie-Hellman key agreement.

#[cfg(feature = "ed25519")]
pub mod ed25519;
#[cfg(feature = "hkdf")]
pub mod hkdf;
#[cfg(feature = "hmac")]
pub mod hmac;
#[cfg(feature = "pbkdf2")]
pub mod pbkdf2;
#[cfg(feature = "kmac")]
pub mod kmac;
#[cfg(feature = "x25519")]
pub mod x25519;

#[cfg(feature = "ed25519")]
pub use ed25519::{Ed25519Keypair, Ed25519PublicKey, Ed25519SecretKey, Ed25519Signature, verify as verify_ed25519};
#[cfg(feature = "hkdf")]
pub use hkdf::{HkdfOutputLengthError, HkdfSha256, HkdfSha384};
#[cfg(feature = "hmac")]
pub use hmac::{HmacSha256, HmacSha384, HmacSha512};
#[cfg(feature = "pbkdf2")]
pub use pbkdf2::{Pbkdf2Error, Pbkdf2Sha256, Pbkdf2Sha512};
#[cfg(feature = "kmac")]
pub use kmac::Kmac256;
#[cfg(feature = "x25519")]
pub use x25519::{X25519Error, X25519PublicKey, X25519SecretKey, X25519SharedSecret};

pub use crate::traits::Mac;
