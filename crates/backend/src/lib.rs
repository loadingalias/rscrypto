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
//! Algorithm crates register kernels as an ordered list of `Candidate`s:
//!
//! ```ignore
//! use backend::dispatch::{Candidate, Selected, select};
//! use platform::caps::{CpuCaps, X86_64_VPCLMUL, X86_64_PCLMUL};
//!
//! fn select_crc32c() -> Selected<fn(u32, &[u8]) -> u32> {
//!     let caps = platform::get().0;
//!     let candidates = &[
//!         Candidate::new("x86_64/vpclmul", X86_64_VPCLMUL, vpclmul_kernel),
//!         Candidate::new("x86_64/pclmul", X86_64_PCLMUL, pclmul_kernel),
//!         Candidate::new("portable", CpuCaps::NONE, portable_kernel),
//!     ];
//!     select(caps, candidates)
//! }
//! ```

#![no_std]

#[cfg(feature = "std")]
extern crate std;

pub mod caps;
pub mod dispatch;

// Re-export platform types for convenience.
pub use platform;
