//! Internal backend support for rscrypto.
//!
//! The public crate surface exposes platform detection and algorithm-level
//! introspection. This module holds internal caching and shared backend kernels
//! so algorithms can reuse implementation building blocks without creating
//! public-surface or feature-graph coupling.
#[cfg(any(feature = "ascon-hash", feature = "ascon-aead"))]
pub(crate) mod ascon;
#[cfg(any(
  feature = "crc16",
  feature = "crc24",
  feature = "crc32",
  feature = "crc64",
  feature = "argon2",
  feature = "sha2",
  feature = "websocket-sha1",
  all(feature = "sha3", any(test, feature = "diag")),
  all(
    any(feature = "blake2b", feature = "blake2s"),
    not(all(target_arch = "aarch64", target_os = "macos"))
  ),
  feature = "blake3",
  all(feature = "ascon-hash", feature = "diag"),
  feature = "xxh3",
  feature = "aes-gcm",
  feature = "aes-gcm-siv",
  feature = "chacha20poly1305",
  feature = "xchacha20poly1305"
))]
pub(crate) mod cache;
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
        all(target_arch = "x86_64", target_os = "linux", not(feature = "portable-only"))
      ))
    )
  )
))]
pub(crate) mod curve25519;
