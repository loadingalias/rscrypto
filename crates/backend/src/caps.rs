//! Re-exported capability constants and masks.
//!
//! This module re-exports the capability types and constants from `platform`
//! for convenience. Algorithm crates can import everything they need from
//! `backend::caps` instead of reaching into `platform` directly.
//!
//! # Examples
//!
//! ```
//! use backend::caps::Caps;
//! let caps = platform::caps();
//! assert!(caps.has(Caps::NONE));
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

// Re-export Power feature constants
pub mod power {
  pub use platform::caps::power::*;
}
