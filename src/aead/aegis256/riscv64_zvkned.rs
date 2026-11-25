use core::arch::asm;

use super::{BLOCK_SIZE, Block, C0, C1, KEY_SIZE, NONCE_SIZE, TAG_SIZE, and_block, split_halves, xor_block};

type State = [Block; 6];

/// Single AES round via Zvkned: SubBytes + ShiftRows + MixColumns + AddRoundKey.
#[target_feature(enable = "v", enable = "zvkned")]
#[inline]
/// # Safety
///
/// Caller must ensure the CPU supports the RISC-V `v` and `zvkned`
/// extensions.
unsafe fn aes_round(block: &Block, round_key: &Block) -> Block {
  let mut out = [0u8; BLOCK_SIZE];

  // SAFETY: caller guarantees the RISC-V vector AES extension is available.
  unsafe {
    asm!(
      "vsetivli zero, 4, e32, m1, ta, ma",
      "vle32.v v1, ({block})",
      "vle32.v v2, ({rk})",
      "vaesem.vs v1, v2",
      "vse32.v v1, ({out})",
      block = in(reg) block.as_ptr(),
      rk = in(reg) round_key.as_ptr(),
      out = in(reg) out.as_mut_ptr(),
      out("v1") _,
      out("v2") _,
      options(nostack),
    );
  }

  out
}

#[target_feature(enable = "v", enable = "zvkned")]
#[inline]
/// # Safety
///
/// Caller must ensure the CPU supports the RISC-V `v` and `zvkned`
/// extensions.
unsafe fn update(s: &mut State, m: &Block) {
  // SAFETY: callers only reach this helper from `zvkned`-gated entry points.
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

impl_riscv_aegis!("v,zvkned");
