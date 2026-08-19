//! SHA-512 RISC-V Zknh kernel.
//!
//! Uses the RISC-V Scalar Cryptographic Hash extension (Zknh) instructions
//! `sha512sig0`, `sha512sig1`, `sha512sum0`, `sha512sum1` to accelerate the
//! sigma/sum operations in SHA-512 compression. Each replaces a 3-instruction
//! rotate-shift-xor sequence with a single dedicated instruction.
//! # Safety
//!
//! All functions require the `zknh` target feature.

#[cfg(target_arch = "riscv64")]
use core::arch::riscv64::{sha512sig0, sha512sig1, sha512sum0, sha512sum1};

use super::BLOCK_LEN;

#[inline(always)]
fn sum0(x: u64) -> u64 {
  // SAFETY: `compress_blocks_zknh` calls this wrapper only inside its Zknh target-feature scope.
  unsafe { sha512sum0(x) }
}

#[inline(always)]
fn sum1(x: u64) -> u64 {
  // SAFETY: `compress_blocks_zknh` calls this wrapper only inside its Zknh target-feature scope.
  unsafe { sha512sum1(x) }
}

#[inline(always)]
fn sig0(x: u64) -> u64 {
  // SAFETY: `compress_blocks_zknh` calls this wrapper only inside its Zknh target-feature scope.
  unsafe { sha512sig0(x) }
}

#[inline(always)]
fn sig1(x: u64) -> u64 {
  // SAFETY: `compress_blocks_zknh` calls this wrapper only inside its Zknh target-feature scope.
  unsafe { sha512sig1(x) }
}

/// SHA-512 multi-block compression using Zknh scalar crypto instructions.
///
/// # Safety
///
/// Caller must ensure `zknh` CPU feature is available.
#[target_feature(enable = "zknh")]
pub(crate) unsafe fn compress_blocks_zknh(state: &mut [u64; 8], blocks: &[u8]) {
  let (chunks, remainder) = blocks.as_chunks::<BLOCK_LEN>();
  debug_assert!(remainder.is_empty());
  for block in chunks {
    super::compress_block_with(state, block, sum0, sum1, sig0, sig1);
  }
}
