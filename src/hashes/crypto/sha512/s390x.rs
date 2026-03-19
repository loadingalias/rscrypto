//! SHA-512 s390x KIMD hardware kernel.
//!
//! The KIMD (Compute Intermediate Message Digest) instruction executes the
//! complete SHA-512 compression function in hardware — rounds, message
//! schedule, and state update — for one or more 128-byte blocks.
//!
//! Function code 3 selects SHA-512. The 64-byte parameter block (at GR1)
//! holds the running state in big-endian (native on s390x). GR2/GR3 form
//! an even/odd pair for data address/length. CC=2 means partial completion
//! (interrupted by the kernel) — retry the instruction.
//!
//! # Safety
//!
//! Requires the MSA facility (CPACF). Caller must verify `s390x::MSA`.

#![allow(unsafe_code)]
#![allow(clippy::indexing_slicing)]

use super::BLOCK_LEN;

/// SHA-512 block compression via KIMD instruction.
///
/// # Safety
///
/// Caller must ensure the MSA (CPACF) facility is available.
pub(crate) unsafe fn compress_blocks_kimd(state: &mut [u64; 8], blocks: &[u8]) {
  debug_assert_eq!(blocks.len() % BLOCK_LEN, 0);
  if blocks.is_empty() {
    return;
  }

  let parm = state.as_mut_ptr();
  let data = blocks.as_ptr();
  let len = blocks.len();

  // KIMD function code 3 = SHA-512.
  //
  // Instruction: .insn rre, 0xB93E0000, 0, 2
  //   -> KIMD with R1-field=0 (unused), R2-field=2 (r2/r3 pair)
  //
  // Register assignment:
  //   r0  = function code (3)
  //   r1  = parameter block pointer (64-byte state, updated in place)
  //   r2  = data pointer (updated after each block)
  //   r3  = remaining data length (decremented after each block)
  //
  // CC=0: complete. CC=2: partial (kernel preemption) — retry.
  //
  // SAFETY: MSA verified by caller. Parameter block is &mut [u64; 8]
  // (64 bytes, 8-aligned >= required 8-byte alignment). Data from valid slice.
  unsafe {
    core::arch::asm!(
      "0:",
      ".insn rre, 0xB93E0000, 0, 2",
      "jo 0b",
      inout("r0") 3u64 => _,
      inout("r1") parm => _,
      inout("r2") data => _,
      inout("r3") len => _,
      options(nostack),
    );
  }
}
