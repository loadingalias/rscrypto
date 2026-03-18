//! SHA-512 RISC-V Zknh kernel.
//!
//! Uses the RISC-V Scalar Cryptographic Hash extension (Zknh) instructions
//! `sha512sig0`, `sha512sig1`, `sha512sum0`, `sha512sum1` to accelerate the
//! sigma/sum operations in SHA-512 compression. Each replaces a 3-instruction
//! rotate-shift-xor sequence with a single dedicated instruction.
//!
//! Expected speedup: ~1.4-1.75x over the portable scalar implementation.
//!
//! # Safety
//!
//! All functions require the `zknh` target feature.

#![allow(clippy::indexing_slicing)] // Fixed-size arrays + compression schedule

#[cfg(target_arch = "riscv64")]
use core::arch::riscv64::{sha512sig0, sha512sig1, sha512sum0, sha512sum1};

use super::BLOCK_LEN;

// Safe wrappers for the unsafe Zknh intrinsics.
// SAFETY: only called from within `compress_blocks_zknh` which has
// `#[target_feature(enable = "zknh")]`, guaranteeing the feature is available.

#[inline(always)]
fn sum0(x: u64) -> u64 {
  // SAFETY: zknh target feature guaranteed by caller.
  unsafe { sha512sum0(x) }
}

#[inline(always)]
fn sum1(x: u64) -> u64 {
  // SAFETY: zknh target feature guaranteed by caller.
  unsafe { sha512sum1(x) }
}

#[inline(always)]
fn sig0(x: u64) -> u64 {
  // SAFETY: zknh target feature guaranteed by caller.
  unsafe { sha512sig0(x) }
}

#[inline(always)]
fn sig1(x: u64) -> u64 {
  // SAFETY: zknh target feature guaranteed by caller.
  unsafe { sha512sig1(x) }
}

/// SHA-512 multi-block compression using Zknh scalar crypto instructions.
///
/// # Safety
///
/// Caller must ensure `zknh` CPU feature is available.
#[target_feature(enable = "zknh")]
pub(crate) unsafe fn compress_blocks_zknh(state: &mut [u64; 8], blocks: &[u8]) {
  debug_assert_eq!(blocks.len() % BLOCK_LEN, 0);
  let mut chunks = blocks.chunks_exact(BLOCK_LEN);
  for chunk in &mut chunks {
    // SAFETY: `chunks_exact(BLOCK_LEN)` yields slices of exactly `BLOCK_LEN` bytes.
    let block = unsafe { &*(chunk.as_ptr() as *const [u8; BLOCK_LEN]) };
    super::compress_block_with(state, block, sum0, sum1, sig0, sig1);
  }
  debug_assert!(chunks.remainder().is_empty());
}
