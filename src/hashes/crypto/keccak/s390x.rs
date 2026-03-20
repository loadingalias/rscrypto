//! Keccak-f[1600] s390x KIMD hardware kernel.
//!
//! The KIMD (Compute Intermediate Message Digest) instruction processes one or
//! more rate-sized blocks: XOR data into the Keccak state and apply the
//! Keccak-f[1600] permutation in hardware. This replaces both the absorb-XOR
//! and the permutation steps of the sponge core.
//!
//! SHA-3/SHAKE function codes (z14+ with MSA8):
//!
//! | Code | Algorithm  | Rate (bytes) |
//! |------|-----------|-------------|
//! |  32  | SHA3-224  | 144         |
//! |  33  | SHA3-256  | 136         |
//! |  34  | SHA3-384  | 104         |
//! |  35  | SHA3-512  |  72         |
//! |  36  | SHAKE-128 | 168         |
//! |  37  | SHAKE-256 | 136         |
//!
//! The 200-byte parameter block at GR1 holds the Keccak state as 25 big-endian
//! double-words (native on s390x). GR2/GR3 form an even/odd pair for
//! data address/length. CC=2 means partial completion (kernel preemption) —
//! retry the instruction.
//!
//! # Safety
//!
//! Requires MSA8 facility (CPACF, z14+). Caller must verify `s390x::MSA8`.

#![allow(unsafe_code)]

/// Map Keccak rate (bytes) to KIMD function code.
///
/// For rate=136 (shared by SHA3-256 and SHAKE-256), returns SHA3-256's code (33).
/// Both produce identical results for intermediate (non-final) blocks since KIMD
/// only XORs rate bytes into state + permutes.
#[inline]
#[must_use]
pub(crate) const fn kimd_fc_for_rate(rate: usize) -> Option<u64> {
  match rate {
    144 => Some(32), // SHA3-224
    136 => Some(33), // SHA3-256 (identical to SHAKE-256 for intermediate blocks)
    104 => Some(34), // SHA3-384
    72 => Some(35),  // SHA3-512
    168 => Some(36), // SHAKE-128
    _ => None,
  }
}

/// Absorb one or more rate-sized blocks into the Keccak state via KIMD.
///
/// `blocks` must be a multiple of the rate corresponding to `func_code`.
/// The state is updated in place.
///
/// # Safety
///
/// - Caller must ensure the MSA8 (CPACF) facility is available.
/// - `blocks.len()` must be a multiple of the algorithm's rate.
#[cfg(target_arch = "s390x")]
pub(crate) unsafe fn absorb_blocks_kimd(state: &mut [u64; 25], blocks: &[u8], func_code: u64) {
  if blocks.is_empty() {
    return;
  }

  let parm = state.as_mut_ptr();
  let data = blocks.as_ptr();
  let len = blocks.len();

  // KIMD with SHA-3/SHAKE function code.
  //
  // Instruction: .insn rre, 0xB93E0000, 0, 2
  //   -> KIMD with R1-field=0 (unused), R2-field=2 (r2/r3 pair)
  //
  // Register assignment:
  //   r0  = function code (32-37 for SHA3/SHAKE)
  //   r1  = parameter block pointer (200-byte Keccak state, 8-aligned)
  //   r2  = data pointer (advanced after each rate-sized block)
  //   r3  = remaining data length (decremented after each block)
  //
  // CC=0: complete. CC=2: partial (kernel preemption) — retry via jo.
  //
  // SAFETY: MSA8 verified by caller. Parameter block is &mut [u64; 25]
  // (200 bytes, 8-aligned >= required alignment). Data from valid slice.
  // State is big-endian on s390x, matching KIMD's expected format.
  unsafe {
    core::arch::asm!(
      "0:",
      ".insn rre, 0xB93E0000, 0, 2",
      "jo 0b",
      inout("r0") func_code => _,
      inout("r1") parm => _,
      inout("r2") data => _,
      inout("r3") len => _,
      options(nostack),
    );
  }
}
