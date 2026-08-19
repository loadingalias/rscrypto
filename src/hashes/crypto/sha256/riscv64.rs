//! SHA-256 RISC-V Zknh kernel.
//!
//! Uses the RISC-V Scalar Cryptographic Hash extension (Zknh) instructions
//! `sha256sum0`, `sha256sum1`, `sha256sig0`, `sha256sig1` to accelerate the
//! sigma/sum operations in SHA-256 compression. Each replaces 5-10 base
//! instructions (shifts, rotates, XORs) with a single dedicated instruction.
//! # Safety
//!
//! All functions require the `zknh` target feature.

#[cfg(target_arch = "riscv32")]
use core::arch::riscv32::{sha256sig0, sha256sig1, sha256sum0, sha256sum1};
#[cfg(target_arch = "riscv64")]
use core::arch::riscv64::{sha256sig0, sha256sig1, sha256sum0, sha256sum1};

use super::BLOCK_LEN;

#[inline(always)]
fn sum0(x: u32) -> u32 {
  // SAFETY: `compress_blocks_zknh` calls this wrapper only inside its Zknh target-feature scope.
  unsafe { sha256sum0(x) }
}

#[inline(always)]
fn sum1(x: u32) -> u32 {
  // SAFETY: `compress_blocks_zknh` calls this wrapper only inside its Zknh target-feature scope.
  unsafe { sha256sum1(x) }
}

#[inline(always)]
fn sig0(x: u32) -> u32 {
  // SAFETY: `compress_blocks_zknh` calls this wrapper only inside its Zknh target-feature scope.
  unsafe { sha256sig0(x) }
}

#[inline(always)]
fn sig1(x: u32) -> u32 {
  // SAFETY: `compress_blocks_zknh` calls this wrapper only inside its Zknh target-feature scope.
  unsafe { sha256sig1(x) }
}

/// SHA-256 multi-block compression using Zknh scalar crypto instructions.
///
/// # Safety
///
/// Caller must ensure `zknh` CPU feature is available.
#[target_feature(enable = "zknh")]
pub(crate) unsafe fn compress_blocks_zknh(state: &mut [u32; 8], blocks: &[u8]) {
  let (chunks, remainder) = blocks.as_chunks::<BLOCK_LEN>();
  debug_assert!(remainder.is_empty());
  for block in chunks {
    super::compress_block_with(state, block, sum0, sum1, sig0, sig1);
  }
}
