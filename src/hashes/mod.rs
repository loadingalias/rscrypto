//! Cryptographic digests and fast non-cryptographic hashes.
//!
//! This module is `no_std` compatible and has zero library dependencies outside
//! the rscrypto workspace. Dev-only dependencies are used for oracle testing
//! and benchmarking.
//!
//! # Quick Start
//!
//! ```rust
//! # #[cfg(feature = "sha2")]
//! # {
//! use rscrypto::{Digest, Sha256};
//!
//! let digest = Sha256::digest(b"hello");
//!
//! let mut streaming = Sha256::new();
//! streaming.update(b"he");
//! streaming.update(b"llo");
//! assert_eq!(streaming.finalize(), digest);
//! # }
//!
//! # #[cfg(feature = "sha3")]
//! # {
//! use rscrypto::{Shake256, Xof};
//!
//! let mut xof = Shake256::xof(b"hello");
//! let mut out = [0u8; 32];
//! xof.squeeze(&mut out);
//! assert_ne!(out, [0u8; 32]);
//! # }
//!
//! # #[cfg(feature = "xxh3")]
//! # {
//! use rscrypto::{FastHash, Xxh3};
//!
//! let fast = Xxh3::hash(b"hello");
//! assert_ne!(fast, 0);
//! # }
//! ```
//!
//! # Feature Selection
//!
//! Pick leaves for minimum size and bundles for category intent:
//!
//! ```toml
//! [dependencies]
//! # Smallest SHA-2-only build
//! rscrypto = { version = "0.10", default-features = false, features = ["sha2"] }
//!
//! # All cryptographic hashes
//! rscrypto = { version = "0.10", default-features = false, features = ["crypto-hashes"] }
//!
//! # Fast non-cryptographic hashes only
//! rscrypto = { version = "0.10", default-features = false, features = ["fast-hashes"] }
//!
//! # Everything hash-related
//! rscrypto = { version = "0.10", default-features = false, features = ["hashes"] }
//! ```
//!
//! The compatibility-only `websocket-sha1` feature is deliberately absent
//! from `hashes`, `crypto-hashes`, and `full`. It exposes only the complete
//! RFC 6455 accept-digest operation under [`legacy`].
//!
//! # API Conventions
//!
//! - Fixed-output digests use `Type::digest(data)` for one-shot and `new` / `update` / `finalize` /
//!   `reset` for streaming.
//! - XOFs use `Type::xof(data)` for one-shot and `finalize_xof()` for streaming squeeze readers.
//! - Fast hashes implement the one-shot [`crate::traits::FastHash`] trait and also expose
//!   algorithm-specific streaming hashers.
//!
//! # Modules
//!
//! - `crypto` - Cryptographic hash functions (safe by default).
//! - `fast` - Non-cryptographic hashes (**NOT CRYPTO**).
//! - `legacy` - Capability-confined compatibility operations; never general-purpose hashes.
//! - `introspect` (requires `diag` feature) - Advanced kernel selection reporting.
//!
//! # Advanced
//!
//! Dispatch is automatic by default.
//!
//! - Use `crate::hashes::introspect` (requires `diag` feature) for kernel reporting and size-based
//!   dispatch details.
//! - Use `crate::hashes::fast` for explicit fast-hash family access.
#[doc(hidden)]
pub(crate) mod common;
#[cfg(any(
  feature = "sha2",
  feature = "sha3",
  feature = "blake2b",
  feature = "blake2s",
  feature = "blake3",
  feature = "ascon-hash"
))]
pub mod crypto;
#[cfg(any(feature = "xxh3", feature = "xxh32", feature = "xxh64", feature = "rapidhash"))]
pub mod fast;
#[cfg(feature = "diag")]
pub mod introspect;

/// Hash capabilities whose inputs callers must structure correctly.
///
/// Normal hashing needs none of these; their placement is an intentional opt-in.
#[cfg(feature = "blake3")]
pub mod expert {
  /// BLAKE3 subtree hashing.
  ///
  /// Hash parts of one input independently, on your own threads or machines,
  /// then merge their chaining values into the same root that one-shot
  /// hashing produces. This enables caller-scheduled parallel hashing,
  /// verified streaming, and incremental rehashing of large inputs.
  ///
  /// Every [`Blake3ChainingValue`](blake3_tree::Blake3ChainingValue) records
  /// its input range and mode. A merge that does not form a valid BLAKE3 tree
  /// returns an error instead of an unrelated hash.
  pub mod blake3_tree {
    pub use crate::hashes::crypto::blake3::tree::{Blake3ChainingValue, Blake3Subtree, Blake3SubtreeError, Blake3Tree};
  }

  /// Verified reading of Bao combined encodings against a trusted BLAKE3 root.
  ///
  /// Requires `std`. Verification uses the same subtree and parent operations
  /// as [`blake3_tree`]. The caller must obtain the expected root through a
  /// trusted channel; a root supplied with untrusted content authenticates nothing.
  #[cfg(feature = "std")]
  pub mod bao {
    pub use crate::hashes::crypto::blake3::bao::Decoder;
  }

  /// Source text of the portable BLAKE3 compression function.
  ///
  /// This is the exact file that rscrypto compiles as its Portable BLAKE3 kernel. Embed it in
  /// code that cannot link rscrypto, for example a `no_std` cdylib that plain `rustc` builds.
  ///
  /// The text holds these `pub(crate)` items:
  ///
  /// - The flags `CHUNK_START`, `CHUNK_END`, `PARENT`, `ROOT`, `KEYED_HASH`, `DERIVE_KEY_CONTEXT`,
  ///   and `DERIVE_KEY_MATERIAL`, each a `u32`.
  /// - `IV: [u32; 8]` and `MSG_SCHEDULE: [[usize; 16]; 7]`.
  /// - `const fn block_words(block: &[u8; 64]) -> [u32; 16]`, which decodes the 16
  ///   little-endian message words of a block.
  /// - `const fn compress_pre`, the seven rounds, and `fn compress_words`, the compression
  ///   function of decoded message words.
  /// - The compression function of one block:
  ///   `fn compress(cv: &[u32; 8], block: &[u8; 64], counter: u64, block_len: u32, flags: u32) -> [u32; 16]`.
  ///
  /// # Build contract
  ///
  /// The text builds with Rust 1.85 or later, edition 2024. It uses only `core`, so it builds in
  /// a `no_std` crate. It has no `unsafe` code, no conditional compilation, and no inner
  /// attribute. Put it inline in a module, use it as the file of a `mod`, or use it with
  /// `include!`. An embedder that uses only some of the items gets `dead_code` warnings.
  ///
  /// The BLAKE3 vectors and kernel tests of rscrypto run this text. Test the output of your
  /// embedded copy against [`Blake3::keyed_xof`](crate::Blake3::keyed_xof) or another
  /// [`Blake3`](crate::Blake3) function.
  ///
  /// The names, values, and signatures of these items are public API under semantic versioning.
  pub const BLAKE3_PORTABLE_SOURCE: &str = include_str!("crypto/blake3/portable.rs");
}
#[cfg(all(
  feature = "std",
  any(
    feature = "sha2",
    feature = "sha3",
    feature = "blake2b",
    feature = "blake2s",
    feature = "blake3",
    feature = "ascon-hash"
  )
))]
pub mod io;
#[cfg(feature = "websocket-sha1")]
pub mod legacy;

#[cfg(feature = "sha2")]
mod util;

// Re-export I/O adapters (requires std)
#[cfg(all(
  feature = "std",
  any(
    feature = "sha2",
    feature = "sha3",
    feature = "blake2b",
    feature = "blake2s",
    feature = "blake3",
    feature = "ascon-hash"
  )
))]
pub use io::{DigestReader, DigestWriter};

pub use crate::traits::{Digest, FastHash, Xof};
