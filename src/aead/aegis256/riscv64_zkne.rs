use core::arch::riscv64::aes64esm;

use super::{BLOCK_SIZE, Block, C0, C1, KEY_SIZE, NONCE_SIZE, TAG_SIZE, and_block, split_halves, xor_block};

type State = [Block; 6];

/// Split a block into the two registers of the RV64 AES instructions.
///
/// Each register holds eight state bytes in little-endian order. The ratified scalar crypto
/// specification (v1.0.1) defines `aes64esm` by the Sail model, in which
/// `aes_rv64_shiftrows_fwd` selects state byte `i` of a register as
/// `getbyte(x, i) = x[8i+7..8i]` (sail-riscv `model/extensions/K/types_kext.sail`).
#[inline(always)]
fn halves(block: &Block) -> (u64, u64) {
  let (lo, hi) = block.split_at(8);
  (
    u64::from_le_bytes(lo.try_into().expect("eight-byte half")),
    u64::from_le_bytes(hi.try_into().expect("eight-byte half")),
  )
}

#[target_feature(enable = "zkne")]
/// # Safety
///
/// Caller must ensure the CPU supports the RISC-V `zkne` extension.
unsafe fn aes_round(block: &Block, round_key: &Block) -> Block {
  let (lo, hi) = halves(block);
  let (rk_lo, rk_hi) = halves(round_key);

  let next_lo = aes64esm(lo, hi) ^ rk_lo;
  let next_hi = aes64esm(hi, lo) ^ rk_hi;

  let mut out = [0u8; BLOCK_SIZE];
  out[0..8].copy_from_slice(&next_lo.to_le_bytes());
  out[8..16].copy_from_slice(&next_hi.to_le_bytes());
  out
}

#[target_feature(enable = "zkne")]
/// # Safety
///
/// Caller must ensure the CPU supports the RISC-V `zkne` extension.
unsafe fn update(s: &mut State, m: &Block) {
  // SAFETY: callers only reach this helper from `zkne`-gated entry points.
  unsafe {
    let tmp = s[5];
    s[5] = aes_round(&s[4], &s[5]);
    s[4] = aes_round(&s[3], &s[4]);
    s[3] = aes_round(&s[2], &s[3]);
    s[2] = aes_round(&s[1], &s[2]);
    s[1] = aes_round(&s[0], &s[1]);
    s[0] = xor_block(&aes_round(&tmp, &s[0]), m);
  }
}

impl_riscv_aegis!("zkne");
