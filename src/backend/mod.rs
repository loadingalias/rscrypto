//! Dispatch and SIMD acceleration primitives for rscrypto.
//!
//! This crate provides the foundation for the rscrypto acceleration subsystem:
//!
//! - **Dispatch**: Zero-cost (compile-time) or cached (runtime) kernel selection
//! - **Capabilities**: Re-exports from `platform` for capability-based dispatch
//! - **Policy**: Pre-computed selection policies with explicit algorithm tuning
//!
//! # Architecture
//!
//! The dispatch system has two paths:
//!
//! 1. **Compile-time** (zero-cost): When target features are known at compile time (`-C
//!    target-feature=...`), the dispatcher resolves to a direct function call.
//!
//! 2. **Runtime** (cached): For generic binaries, the dispatcher detects CPU features once and
//!    caches the selected kernel. Subsequent calls are a single indirect call.
//!
//! # Key Types
//!
//! - [`KernelTier`]: Acceleration levels (Reference, Portable, HwCrc, Folding, Wide)
//! - [`KernelFamily`]: Specific backend implementations (X86Pclmul, ArmPmull, etc.)
//! - [`SelectionPolicy`]: Pre-computed dispatch structure with explicit algorithm tuning
//! - [`PolicyTunables`]: Algorithm-owned thresholds layered onto a structural policy
//! - [`OnceCache`]: Thread-safe lazy initialization for kernel selection
//!
//! # Usage
//!
//! Algorithm crates register kernels via the `candidates!` macro.
//! See [`dispatch`] module for details.
pub mod cache;
pub mod caps;
pub mod dispatch;
pub mod family;
pub mod policy;
pub mod tier;

// Re-export core dispatch types for convenience.
pub use cache::{OnceCache, PolicyCache};
pub use family::{KernelFamily, KernelSubfamily};
pub use policy::{ForceMode, PolicyTunables, SelectionPolicy};
pub use tier::KernelTier;

// Re-export platform types for convenience.
pub use crate::platform;
