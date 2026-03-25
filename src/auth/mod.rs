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
//! # Ok::<(), rscrypto::auth::HkdfOutputLengthError>(())
//! ```
//!
//! # Modules
//!
//! - [`ed25519`] - Ed25519 key and signature types.
//! - [`hmac`] - HMAC-based authentication.
//! - [`hkdf`] - HKDF extract-then-expand key derivation.

pub mod ed25519;
pub mod hkdf;
pub mod hmac;

pub use ed25519::{Ed25519Keypair, Ed25519PublicKey, Ed25519SecretKey, Ed25519Signature, verify as verify_ed25519};
pub use hkdf::{HkdfOutputLengthError, HkdfSha256};
pub use hmac::HmacSha256;

pub use crate::traits::Mac;
