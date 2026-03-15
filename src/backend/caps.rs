//! Re-exported capability constants and masks from `platform`.
//!
//! Algorithm crates can import capabilities from `backend::caps` instead
//! of reaching into `platform` directly.
//!
//! # Example
//!
//! ```
//! use rscrypto::backend::caps::{Caps, x86};
//!
//! let caps = rscrypto::platform::caps();
//! if caps.has(x86::PCLMUL_READY) {
//!   // Use PCLMUL kernel
//! }
//! ```

// Re-export core types
pub use crate::platform::caps::{Arch, Caps};

// Re-export x86 feature constants
pub mod x86 {
  pub use crate::platform::caps::x86::*;
}

// Re-export aarch64 feature constants
pub mod aarch64 {
  pub use crate::platform::caps::aarch64::*;
}

// Re-export RISC-V feature constants
pub mod riscv {
  pub use crate::platform::caps::riscv::*;
}

// Re-export wasm feature constants
pub mod wasm {
  pub use crate::platform::caps::wasm::*;
}

// Re-export s390x feature constants
pub mod s390x {
  pub use crate::platform::caps::s390x::*;
}

// Re-export Power feature constants
pub mod power {
  pub use crate::platform::caps::power::*;
}
