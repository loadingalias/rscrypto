//! CPU detection and microarchitecture identification.
//!
//! Provides runtime detection of CPU features and microarchitectures,
//! enabling optimal algorithm selection across the rscrypto workspace.
//!
//! # Architecture
//!
//! Detection is organized by target architecture:
//! - `x86_64`: Intel and AMD processors with AVX-512, VPCLMULQDQ, etc.
//! - `aarch64`: ARM processors with NEON, PMULL, CRC, SHA3, etc.
//!
//! # Usage Pattern
//!
//! ```text
//! use platform::x86_64::{MicroArch, detect_microarch};
//!
//! let arch = detect_microarch();
//! match arch {
//!     MicroArch::SapphireRapids => { /* use v3s1_s3 config */ }
//!     MicroArch::IceLake => { /* use v4s5x3 config */ }
//!     MicroArch::Zen4 | MicroArch::Zen5 => { /* AMD path */ }
//!     _ => { /* generic fallback */ }
//! }
//! ```
//!
//! # Caching
//!
//! Detection results are cached using `OnceLock` (requires `std` feature).
//! Without `std`, detection runs on each call but is still fast (~100 cycles).

#![no_std]

#[cfg(feature = "std")]
extern crate std;

#[cfg(target_arch = "x86_64")]
pub mod x86_64;

#[cfg(target_arch = "aarch64")]
pub mod aarch64;
