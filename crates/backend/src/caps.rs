//! Re-exported capability constants and masks.
//!
//! This module re-exports the capability types and constants from `platform`
//! for convenience. Algorithm crates can import everything they need from
//! `backend::caps` instead of reaching into `platform` directly.
//!
//! # Example
//!
//! ```ignore
//! use backend::caps::{Caps, x86, aarch64};
//!
//! fn select_kernel(caps: Caps) {
//!     if caps.has(x86::VPCLMUL_READY) {
//!         // Use AVX-512 VPCLMULQDQ kernel
//!     } else if caps.has(x86::PCLMUL_READY) {
//!         // Use PCLMULQDQ kernel
//!     }
//! }
//! ```

// Re-export core types
pub use platform::caps::{Arch, Caps};

// Re-export x86 feature constants
pub mod x86 {
  pub use platform::caps::x86::*;
}

// Re-export aarch64 feature constants
pub mod aarch64 {
  pub use platform::caps::aarch64::*;
}

// Re-export RISC-V feature constants
pub mod riscv {
  pub use platform::caps::riscv::*;
}

// Re-export wasm feature constants
pub mod wasm {
  pub use platform::caps::wasm::*;
}

// Re-export s390x feature constants
pub mod s390x {
  pub use platform::caps::s390x::*;
}

// Re-export powerpc64 feature constants
pub mod powerpc64 {
  pub use platform::caps::powerpc64::*;
}
