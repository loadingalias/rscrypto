//! Modern, high-performance cryptography for Rust.
//!
//! rscrypto provides fast, safe cryptographic primitives that automatically
//! use the best available CPU instructions (AVX-512, AVX2, SSE4.2, NEON, etc.).
//!
//! # Crates
//!
//! - `traits` - Core cryptographic traits
//! - `checksum` - CRC32-C, CRC32, CRC64
//! - `hash` - Blake3, SHA2, SHA3
//!
//! # Future
//!
//! - `aead` - AES-GCM, ChaCha20-Poly1305, AEGIS, Ascon
//! - `cipher` - AES, `ChaCha20`
//! - `mac` - HMAC, Poly1305, CMAC
//! - `kdf` - HKDF, Argon2, scrypt
//! - `x25519`, `ed25519`, `ecdsa` - Asymmetric crypto
//! - `mlkem`, `mldsa`, `slhdsa` - Post-quantum crypto

#![no_std]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

pub use checksum;
pub use hash;
pub use traits;
