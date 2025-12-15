//! Shared SIMD engines and dispatch helpers.
//!
//! This module is internal to the crate and contains ISA-specific SIMD code
//! organized by architecture.

#[cfg(target_arch = "aarch64")]
pub(crate) mod aarch64;

#[cfg(target_arch = "x86_64")]
pub mod x86_64;

#[cfg(target_arch = "wasm32")]
pub(crate) mod wasm32;
