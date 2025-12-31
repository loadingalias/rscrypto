//! Backend crate: dispatch and SIMD acceleration primitives for rscrypto.
//!
//! This crate provides the foundation for the rscrypto acceleration subsystem:
//!
//! - **Dispatch**: Zero-cost (compile-time) or cached (runtime) kernel selection
//! - **Capabilities**: Re-exports from `platform` for capability-based dispatch
//! - **Primitives**: Shared SIMD primitives (future: carryless multiply, AES rounds, etc.)
//!
//! # Architecture
//!
//! The dispatch system has two paths:
//!
//! 1. **Compile-time selection** (zero-cost): When target features are known at compile time (`-C
//!    target-feature=...`), the dispatcher can resolve to a direct function call with no overhead.
//!
//! 2. **Runtime selection** (cached): For generic binaries, the dispatcher detects CPU features
//!    once and caches the selected kernel. Subsequent calls are a single indirect call.
//!
//! # Usage
//!
//! Algorithm crates register kernels as an ordered list of `Candidate`s.
//! Use the [`candidates!`] macro for concise syntax:
//!
//! ```ignore
//! use backend::dispatch::{candidates, Candidate, Selected, select};
//! use backend::caps::{Caps, x86};
//!
//! fn select_crc32c() -> Selected<fn(u32, &[u8]) -> u32> {
//!     let caps = platform::caps();
//!     select(caps, candidates![
//!         "x86_64/vpclmul" => x86::VPCLMUL_READY => vpclmul_kernel,
//!         "x86_64/pclmul"  => x86::PCLMUL_READY  => pclmul_kernel,
//!         "portable"       => Caps::NONE         => portable_kernel,
//!     ])
//! }
//! ```
//!
//! The macro expands to `&[Candidate::new(...), ...]`, providing a cleaner
//! syntax while maintaining zero runtime overhead.
//!
//! // Fallibility discipline: deny unwrap/expect in production, allow in tests.
#![cfg_attr(not(test), deny(clippy::unwrap_used))]
#![cfg_attr(not(test), deny(clippy::expect_used))]
#![cfg_attr(not(test), deny(clippy::indexing_slicing))]
#![no_std]

#[cfg(feature = "std")]
extern crate std;

pub mod cache;
pub mod caps;
pub mod dispatch;
pub mod family;
pub mod policy;
pub mod tier;

// Re-export core dispatch types for convenience.
pub use cache::{OnceCache, PolicyCache};
pub use family::{KernelFamily, KernelSubfamily};
// Re-export platform types for convenience.
pub use platform;
pub use policy::{ForceMode, SelectionPolicy};
pub use tier::KernelTier;
