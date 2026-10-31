//! Register-resident Montgomery reduction for a 16-limb RSA CRT prime.
//!
//! Each row adds the low product halves in one carry chain, then the high halves in a second.
//! The sixteen-word running window stays in registers. Four rows rotate it by four registers;
//! four such blocks discard the low sixteen limbs. Every row reads all sixteen modulus words
//! twice and one upper input word, independent of their values.

use core::arch::asm;

use super::super::super::subtract_modulus_if_needed;

macro_rules! low {
  ($word:literal, $offset:literal, $add:literal) => {
    concat!(
      "ldr {m}, [{modulus}, #",
      $offset,
      "]\n",
      "mul {product}, {q}, {m}\n",
      $add,
      " ",
      $word,
      ", ",
      $word,
      ", {product}\n",
    )
  };
}

macro_rules! high {
  ($word:literal, $offset:literal, $add:literal) => {
    concat!(
      "ldr {m}, [{modulus}, #",
      $offset,
      "]\n",
      "umulh {product}, {q}, {m}\n",
      $add,
      " ",
      $word,
      ", ",
      $word,
      ", {product}\n",
    )
  };
}

macro_rules! row {
  ($w0:literal, $w1:literal, $w2:literal, $w3:literal, $w4:literal, $w5:literal,
   $w6:literal, $w7:literal, $w8:literal, $w9:literal, $w10:literal, $w11:literal,
   $w12:literal, $w13:literal, $w14:literal, $w15:literal) => {
    concat!(
      "mov {q}, {next_q}\n",
      low!($w0, "0", "adds"),
      low!($w1, "8", "adcs"),
      low!($w2, "16", "adcs"),
      low!($w3, "24", "adcs"),
      low!($w4, "32", "adcs"),
      low!($w5, "40", "adcs"),
      low!($w6, "48", "adcs"),
      low!($w7, "56", "adcs"),
      low!($w8, "64", "adcs"),
      low!($w9, "72", "adcs"),
      low!($w10, "80", "adcs"),
      low!($w11, "88", "adcs"),
      low!($w12, "96", "adcs"),
      low!($w13, "104", "adcs"),
      low!($w14, "112", "adcs"),
      low!($w15, "120", "adcs"),
      "cset {low_carry}, cs\n",
      // The low word is zero. The next row's multiplier is available as soon as word 1 is final.
      high!($w1, "0", "adds"),
      "mul {next_q}, ",
      $w1,
      ", {n0}\n",
      high!($w2, "8", "adcs"),
      high!($w3, "16", "adcs"),
      high!($w4, "24", "adcs"),
      high!($w5, "32", "adcs"),
      high!($w6, "40", "adcs"),
      high!($w7, "48", "adcs"),
      high!($w8, "56", "adcs"),
      high!($w9, "64", "adcs"),
      high!($w10, "72", "adcs"),
      high!($w11, "80", "adcs"),
      high!($w12, "88", "adcs"),
      high!($w13, "96", "adcs"),
      high!($w14, "104", "adcs"),
      high!($w15, "112", "adcs"),
      "ldr {m}, [{modulus}, #120]\n",
      "umulh {product}, {q}, {m}\n",
      "adc {product}, {product}, {low_carry}\n",
      // q*n + the running window fits seventeen words, so this carry fits one word.
      "ldr {m}, [{upper}], #8\n",
      "adds {product}, {product}, {m}\n",
      "cset {low_carry}, cs\n",
      "adds ",
      $w0,
      ", {product}, {top}\n",
      "cinc {top}, {low_carry}, cs\n",
    )
  };
}

pub(super) fn reduce(out: &mut [u64], wide: &mut [u64], modulus: &[u64], n0: u64) {
  assert!(out.len() == 16 && wide.len() >= 32 && modulus.len() == 16);
  let top: u64;
  // SAFETY: The input window reads wide[0..16]. Four blocks of four rows each read one word
  // from wide[16..32]; the final pointer is one-past and is never dereferenced. Each row reads
  // modulus[0..16] at fixed eight-byte offsets. The final stores write out[0..16], which the
  // distinct mutable borrows keep disjoint from wide and modulus. All pointers have u64
  // alignment, all registers are declared, and the block uses no stack or calls. Its only
  // branch counts four public blocks. MUL, UMULH and carry arithmetic require baseline AArch64.
  unsafe {
    asm!(
      "ldp {w0}, {w1}, [{upper}, #-128]",
      "ldp {w2}, {w3}, [{upper}, #-112]",
      "ldp {w4}, {w5}, [{upper}, #-96]",
      "ldp {w6}, {w7}, [{upper}, #-80]",
      "ldp {w8}, {w9}, [{upper}, #-64]",
      "ldp {w10}, {w11}, [{upper}, #-48]",
      "ldp {w12}, {w13}, [{upper}, #-32]",
      "ldp {w14}, {w15}, [{upper}, #-16]",
      "mov {top}, xzr",
      "mov {blocks}, #4",
      "mul {next_q}, {w0}, {n0}",
      "2:",
      row!("{w0}", "{w1}", "{w2}", "{w3}", "{w4}", "{w5}", "{w6}", "{w7}",
           "{w8}", "{w9}", "{w10}", "{w11}", "{w12}", "{w13}", "{w14}", "{w15}"),
      row!("{w1}", "{w2}", "{w3}", "{w4}", "{w5}", "{w6}", "{w7}", "{w8}",
           "{w9}", "{w10}", "{w11}", "{w12}", "{w13}", "{w14}", "{w15}", "{w0}"),
      row!("{w2}", "{w3}", "{w4}", "{w5}", "{w6}", "{w7}", "{w8}", "{w9}",
           "{w10}", "{w11}", "{w12}", "{w13}", "{w14}", "{w15}", "{w0}", "{w1}"),
      row!("{w3}", "{w4}", "{w5}", "{w6}", "{w7}", "{w8}", "{w9}", "{w10}",
           "{w11}", "{w12}", "{w13}", "{w14}", "{w15}", "{w0}", "{w1}", "{w2}"),
      // Reuse dead product temporaries to rotate the logical window for the next block.
      "mov {q}, {w0}", "mov {product}, {w1}", "mov {m}, {w2}", "mov {low_carry}, {w3}",
      "mov {w0}, {w4}", "mov {w1}, {w5}", "mov {w2}, {w6}", "mov {w3}, {w7}",
      "mov {w4}, {w8}", "mov {w5}, {w9}", "mov {w6}, {w10}", "mov {w7}, {w11}",
      "mov {w8}, {w12}", "mov {w9}, {w13}", "mov {w10}, {w14}", "mov {w11}, {w15}",
      "mov {w12}, {q}", "mov {w13}, {product}", "mov {w14}, {m}", "mov {w15}, {low_carry}",
      "subs {blocks}, {blocks}, #1",
      "b.ne 2b",
      "stp {w0}, {w1}, [{out}]",
      "stp {w2}, {w3}, [{out}, #16]",
      "stp {w4}, {w5}, [{out}, #32]",
      "stp {w6}, {w7}, [{out}, #48]",
      "stp {w8}, {w9}, [{out}, #64]",
      "stp {w10}, {w11}, [{out}, #80]",
      "stp {w12}, {w13}, [{out}, #96]",
      "stp {w14}, {w15}, [{out}, #112]",
      upper = inout(reg) wide[16..].as_ptr() => _,
      modulus = in(reg) modulus.as_ptr(),
      out = in(reg) out.as_mut_ptr(),
      n0 = in(reg) n0,
      top = out(reg) top,
      blocks = out(reg) _, q = out(reg) _, next_q = out(reg) _,
      product = out(reg) _, m = out(reg) _, low_carry = out(reg) _,
      w0 = out(reg) _, w1 = out(reg) _, w2 = out(reg) _, w3 = out(reg) _,
      w4 = out(reg) _, w5 = out(reg) _, w6 = out(reg) _, w7 = out(reg) _,
      w8 = out(reg) _, w9 = out(reg) _, w10 = out(reg) _, w11 = out(reg) _,
      w12 = out(reg) _, w13 = out(reg) _, w14 = out(reg) _, w15 = out(reg) _,
      options(nostack),
    );
  }
  subtract_modulus_if_needed(out, modulus, top);
}
