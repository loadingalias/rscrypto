//! Re-exported capability constants and masks.
//!
//! This module re-exports the capability types and constants from `platform`
//! for convenience. Algorithm crates can import everything they need from
//! `backend::caps` instead of reaching into `platform` directly.
//!
//! # Example
//!
//! ```ignore
//! use backend::caps::{CpuCaps, Bits256, x86, aarch64};
//!
//! fn select_kernel(caps: CpuCaps) {
//!     if caps.has(x86::VPCLMUL_READY) {
//!         // Use AVX-512 VPCLMULQDQ kernel
//!     } else if caps.has(x86::PCLMUL_READY) {
//!         // Use PCLMULQDQ kernel
//!     }
//! }
//! ```

// Re-export core types
pub use platform::caps::{Arch, Bits256, CpuCaps};

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
