//! SHA-256 RISC-V Zknh kernel.
//!
//! Uses the RISC-V Scalar Cryptographic Hash extension (Zknh) instructions
//! `sha256sum0`, `sha256sum1`, `sha256sig0`, `sha256sig1` to accelerate the
//! sigma/sum operations in SHA-256 compression. Each replaces 5-10 base
//! instructions (shifts, rotates, XORs) with a single dedicated instruction.
//!
//! Expected speedup: ~1.4-2x over the portable scalar implementation.
//!
//! # Safety
//!
//! All functions require the `zknh` target feature.

#![allow(clippy::indexing_slicing)] // Fixed-size arrays + compression schedule

#[cfg(target_arch = "riscv32")]
use core::arch::riscv32::{sha256sig0, sha256sig1, sha256sum0, sha256sum1};
#[cfg(target_arch = "riscv64")]
use core::arch::riscv64::{sha256sig0, sha256sig1, sha256sum0, sha256sum1};

use super::BLOCK_LEN;

// Safe wrappers for the unsafe Zknh intrinsics.
// SAFETY: only called from within `compress_blocks_zknh` which has
// `#[target_feature(enable = "zknh")]`, guaranteeing the feature is available.

#[inline(always)]
fn sum0(x: u32) -> u32 {
  // SAFETY: zknh target feature guaranteed by caller.
  unsafe { sha256sum0(x) }
}

#[inline(always)]
fn sum1(x: u32) -> u32 {
  // SAFETY: zknh target feature guaranteed by caller.
  unsafe { sha256sum1(x) }
}

#[inline(always)]
fn sig0(x: u32) -> u32 {
  // SAFETY: zknh target feature guaranteed by caller.
  unsafe { sha256sig0(x) }
}

#[inline(always)]
fn sig1(x: u32) -> u32 {
  // SAFETY: zknh target feature guaranteed by caller.
  unsafe { sha256sig1(x) }
}

/// SHA-256 multi-block compression using Zknh scalar crypto instructions.
///
/// # Safety
///
/// Caller must ensure `zknh` CPU feature is available.
#[target_feature(enable = "zknh")]
pub(crate) unsafe fn compress_blocks_zknh(state: &mut [u32; 8], blocks: &[u8]) {
  debug_assert_eq!(blocks.len() % BLOCK_LEN, 0);
  let mut chunks = blocks.chunks_exact(BLOCK_LEN);
  for chunk in &mut chunks {
    // SAFETY: `chunks_exact(BLOCK_LEN)` yields slices of exactly `BLOCK_LEN` bytes.
    let block = unsafe { &*(chunk.as_ptr() as *const [u8; BLOCK_LEN]) };
    super::compress_block_with(state, block, sum0, sum1, sig0, sig1);
  }
  debug_assert!(chunks.remainder().is_empty());
}
