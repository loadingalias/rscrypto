//! Keccak-f[1600] with lazy lane rotations, for AArch64.
//!
//! The ρ step rotates 24 of the 25 lanes every round. AArch64 `EOR` and `BIC`
//! rotate their second operand at no cost, so this permutation never rotates
//! a lane by itself. Each stored lane `s` carries a pending left rotation `p`;
//! its value is `s.rotate_left(p)`. Every operation that reads a lane applies
//! the rotation that operation needs through its rotated operand, and each
//! result lives in the frame of the operand it was not able to rotate. ρ then
//! changes only the pending rotations, and a round costs 101 logical
//! instructions: 20 and 5 for the θ parities, 25 to apply θ, 50 for χ, and one
//! for ι. The last step restores the 25 lanes.
//!
//! The pending rotations depend only on the round number, so `SCHEDULE`
//! computes them at compile time and the rounds are unrolled. Lane `x + 5y`
//! holds `A[x, y]`.
//!
//! The code is portable Rust; tests compile it on every target and check it
//! with the other registered Keccak kernels.

use super::{KECCAKF_ROUNDS, RC};

/// ρ offsets, indexed by lane `x + 5y`.
const RHO: [u32; 25] = [
  0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56, 14,
];

/// π destination of lane `x + 5y`: lane `y + 5 * ((2x + 3y) mod 5)`.
const PI: [usize; 25] = pi();

const fn pi() -> [usize; 25] {
  let mut out = [0usize; 25];
  let mut lane = 0usize;
  while lane < 25 {
    let x = lane.strict_rem(5);
    let y = lane.strict_div(5);
    out[lane] = y.strict_add(x.strict_mul(2).strict_add(y.strict_mul(3)).strict_rem(5).strict_mul(5));
    lane = lane.strict_add(1);
  }
  out
}

/// Rotations for one round. Every entry is a left rotation in `0..64`.
#[derive(Clone, Copy)]
struct Round {
  /// θ: rotation of lane `x + 5y` (`y >= 1`) into the frame of lane `x`.
  column: [u32; 25],
  /// θ: rotation of parity `x + 1` into the frame of parity `x - 1`.
  d: [u32; 5],
  /// θ: rotation of `d[x]` into the frame of lane `x + 5y`.
  apply: [u32; 25],
  /// χ for output lane `i` in row `y`: rotation of `B[x + 1, y]` into the
  /// frame of `B[x + 2, y]`.
  chi_next: [u32; 25],
  /// χ for output lane `i`: rotation of `B[x, y]` into the frame of
  /// `B[x + 2, y]`.
  chi_self: [u32; 25],
  /// ι: the round constant in the frame of output lane 0.
  iota: u64,
}

struct Schedule {
  rounds: [Round; KECCAKF_ROUNDS],
  /// Pending rotation of each lane after the last round.
  restore: [u32; 25],
}

const SCHEDULE: Schedule = schedule();

/// Returns `(a - b) mod 64`.
const fn relative(a: u32, b: u32) -> u32 {
  a.wrapping_sub(b) & 63
}

/// Tracks the pending rotation of every lane through the 24 rounds.
const fn schedule() -> Schedule {
  let empty = Round {
    column: [0; 25],
    d: [0; 5],
    apply: [0; 25],
    chi_next: [0; 25],
    chi_self: [0; 25],
    iota: 0,
  };
  let mut rounds = [empty; KECCAKF_ROUNDS];
  let mut pending = [0u32; 25];
  let mut r = 0usize;
  while r < KECCAKF_ROUNDS {
    let mut round = empty;
    // θ parities: column x is computed in the frame of lane x.
    let mut lane = 5usize;
    while lane < 25 {
      round.column[lane] = relative(pending[lane], pending[lane.strict_rem(5)]);
      lane = lane.strict_add(1);
    }
    // d[x] is computed in the frame of parity x - 1, which is that of lane x - 1.
    let mut x = 0usize;
    let mut d_frame = [0u32; 5];
    while x < 5 {
      let below = x.strict_add(4).strict_rem(5);
      let above = x.strict_add(1).strict_rem(5);
      d_frame[x] = pending[below];
      round.d[x] = relative(pending[above].strict_add(1), pending[below]);
      x = x.strict_add(1);
    }
    // Applying θ keeps the lane's frame; ρ adds its offset at π's destination.
    let mut b_frame = [0u32; 25];
    lane = 0;
    while lane < 25 {
      round.apply[lane] = relative(d_frame[lane.strict_rem(5)], pending[lane]);
      b_frame[PI[lane]] = pending[lane].strict_add(RHO[lane]) & 63;
      lane = lane.strict_add(1);
    }
    // χ for each output lane is computed in the frame of B[x + 2, y].
    lane = 0;
    while lane < 25 {
      let row = lane.strict_sub(lane.strict_rem(5));
      let next = row.strict_add(lane.strict_add(1).strict_rem(5));
      let after = row.strict_add(lane.strict_add(2).strict_rem(5));
      round.chi_next[lane] = relative(b_frame[next], b_frame[after]);
      round.chi_self[lane] = relative(b_frame[lane], b_frame[after]);
      pending[lane] = b_frame[after];
      lane = lane.strict_add(1);
    }
    round.iota = RC[r].rotate_right(pending[0]);
    rounds[r] = round;
    r = r.strict_add(1);
  }
  Schedule {
    rounds,
    restore: pending,
  }
}

/// Returns `x` unchanged, hidden from LLVM's reassociation.
///
/// LLVM otherwise reorders a parity's XOR chain so that two rotated lanes
/// meet, which costs a separate rotation.
#[inline(always)]
fn opaque(x: u64) -> u64 {
  #[cfg(all(target_arch = "aarch64", not(miri)))]
  let x = {
    let mut x = x;
    // SAFETY: the template emits no instruction. Its one operand is the general
    // register holding `x` on entry and exit; it touches no memory, stack, or
    // flags.
    unsafe {
      core::arch::asm!("/* {0} */", inout(reg) x, options(pure, nomem, nostack, preserves_flags));
    }
    x
  };
  x
}

/// One round. Each index is a literal and each rotation a schedule constant,
/// so every rotation folds into the operand of its `EOR` or `BIC`.
macro_rules! round {
  ($s:ident, $r:literal) => {{
    const R: Round = SCHEDULE.rounds[$r];
    // θ: column parities, each in the frame of its row-0 lane. The barriers
    // keep the XOR chain in order, so each step rotates one operand.
    let c = [
      opaque(
        opaque(opaque($s[0] ^ $s[5].rotate_left(R.column[5])) ^ $s[10].rotate_left(R.column[10]))
          ^ $s[15].rotate_left(R.column[15]),
      ) ^ $s[20].rotate_left(R.column[20]),
      opaque(
        opaque(opaque($s[1] ^ $s[6].rotate_left(R.column[6])) ^ $s[11].rotate_left(R.column[11]))
          ^ $s[16].rotate_left(R.column[16]),
      ) ^ $s[21].rotate_left(R.column[21]),
      opaque(
        opaque(opaque($s[2] ^ $s[7].rotate_left(R.column[7])) ^ $s[12].rotate_left(R.column[12]))
          ^ $s[17].rotate_left(R.column[17]),
      ) ^ $s[22].rotate_left(R.column[22]),
      opaque(
        opaque(opaque($s[3] ^ $s[8].rotate_left(R.column[8])) ^ $s[13].rotate_left(R.column[13]))
          ^ $s[18].rotate_left(R.column[18]),
      ) ^ $s[23].rotate_left(R.column[23]),
      opaque(
        opaque(opaque($s[4] ^ $s[9].rotate_left(R.column[9])) ^ $s[14].rotate_left(R.column[14]))
          ^ $s[19].rotate_left(R.column[19]),
      ) ^ $s[24].rotate_left(R.column[24]),
    ];
    // θ: d[x] = c[x - 1] ^ c[x + 1].rotate_left(1), in the frame of c[x - 1].
    let d = [
      c[4] ^ c[1].rotate_left(R.d[0]),
      c[0] ^ c[2].rotate_left(R.d[1]),
      c[1] ^ c[3].rotate_left(R.d[2]),
      c[2] ^ c[4].rotate_left(R.d[3]),
      c[3] ^ c[0].rotate_left(R.d[4]),
    ];
    // θ and ρ+π, one χ row at a time: B[j] = lane PI⁻¹(j) ^ d. ρ only changes the
    // pending rotation. Each row consumes five lanes, so building a row's B
    // values next to its χ keeps the live set small.
    let row0 = {
      let b0 = $s[0] ^ d[0].rotate_left(R.apply[0]);
      let b1 = $s[6] ^ d[1].rotate_left(R.apply[6]);
      let b2 = $s[12] ^ d[2].rotate_left(R.apply[12]);
      let b3 = $s[18] ^ d[3].rotate_left(R.apply[18]);
      let b4 = $s[24] ^ d[4].rotate_left(R.apply[24]);
      [
        (b2 & !b1.rotate_left(R.chi_next[0])) ^ b0.rotate_left(R.chi_self[0]) ^ R.iota,
        (b3 & !b2.rotate_left(R.chi_next[1])) ^ b1.rotate_left(R.chi_self[1]),
        (b4 & !b3.rotate_left(R.chi_next[2])) ^ b2.rotate_left(R.chi_self[2]),
        (b0 & !b4.rotate_left(R.chi_next[3])) ^ b3.rotate_left(R.chi_self[3]),
        (b1 & !b0.rotate_left(R.chi_next[4])) ^ b4.rotate_left(R.chi_self[4]),
      ]
    };
    let row1 = {
      let b5 = $s[3] ^ d[3].rotate_left(R.apply[3]);
      let b6 = $s[9] ^ d[4].rotate_left(R.apply[9]);
      let b7 = $s[10] ^ d[0].rotate_left(R.apply[10]);
      let b8 = $s[16] ^ d[1].rotate_left(R.apply[16]);
      let b9 = $s[22] ^ d[2].rotate_left(R.apply[22]);
      [
        (b7 & !b6.rotate_left(R.chi_next[5])) ^ b5.rotate_left(R.chi_self[5]),
        (b8 & !b7.rotate_left(R.chi_next[6])) ^ b6.rotate_left(R.chi_self[6]),
        (b9 & !b8.rotate_left(R.chi_next[7])) ^ b7.rotate_left(R.chi_self[7]),
        (b5 & !b9.rotate_left(R.chi_next[8])) ^ b8.rotate_left(R.chi_self[8]),
        (b6 & !b5.rotate_left(R.chi_next[9])) ^ b9.rotate_left(R.chi_self[9]),
      ]
    };
    let row2 = {
      let b10 = $s[1] ^ d[1].rotate_left(R.apply[1]);
      let b11 = $s[7] ^ d[2].rotate_left(R.apply[7]);
      let b12 = $s[13] ^ d[3].rotate_left(R.apply[13]);
      let b13 = $s[19] ^ d[4].rotate_left(R.apply[19]);
      let b14 = $s[20] ^ d[0].rotate_left(R.apply[20]);
      [
        (b12 & !b11.rotate_left(R.chi_next[10])) ^ b10.rotate_left(R.chi_self[10]),
        (b13 & !b12.rotate_left(R.chi_next[11])) ^ b11.rotate_left(R.chi_self[11]),
        (b14 & !b13.rotate_left(R.chi_next[12])) ^ b12.rotate_left(R.chi_self[12]),
        (b10 & !b14.rotate_left(R.chi_next[13])) ^ b13.rotate_left(R.chi_self[13]),
        (b11 & !b10.rotate_left(R.chi_next[14])) ^ b14.rotate_left(R.chi_self[14]),
      ]
    };
    let row3 = {
      let b15 = $s[4] ^ d[4].rotate_left(R.apply[4]);
      let b16 = $s[5] ^ d[0].rotate_left(R.apply[5]);
      let b17 = $s[11] ^ d[1].rotate_left(R.apply[11]);
      let b18 = $s[17] ^ d[2].rotate_left(R.apply[17]);
      let b19 = $s[23] ^ d[3].rotate_left(R.apply[23]);
      [
        (b17 & !b16.rotate_left(R.chi_next[15])) ^ b15.rotate_left(R.chi_self[15]),
        (b18 & !b17.rotate_left(R.chi_next[16])) ^ b16.rotate_left(R.chi_self[16]),
        (b19 & !b18.rotate_left(R.chi_next[17])) ^ b17.rotate_left(R.chi_self[17]),
        (b15 & !b19.rotate_left(R.chi_next[18])) ^ b18.rotate_left(R.chi_self[18]),
        (b16 & !b15.rotate_left(R.chi_next[19])) ^ b19.rotate_left(R.chi_self[19]),
      ]
    };
    let row4 = {
      let b20 = $s[2] ^ d[2].rotate_left(R.apply[2]);
      let b21 = $s[8] ^ d[3].rotate_left(R.apply[8]);
      let b22 = $s[14] ^ d[4].rotate_left(R.apply[14]);
      let b23 = $s[15] ^ d[0].rotate_left(R.apply[15]);
      let b24 = $s[21] ^ d[1].rotate_left(R.apply[21]);
      [
        (b22 & !b21.rotate_left(R.chi_next[20])) ^ b20.rotate_left(R.chi_self[20]),
        (b23 & !b22.rotate_left(R.chi_next[21])) ^ b21.rotate_left(R.chi_self[21]),
        (b24 & !b23.rotate_left(R.chi_next[22])) ^ b22.rotate_left(R.chi_self[22]),
        (b20 & !b24.rotate_left(R.chi_next[23])) ^ b23.rotate_left(R.chi_self[23]),
        (b21 & !b20.rotate_left(R.chi_next[24])) ^ b24.rotate_left(R.chi_self[24]),
      ]
    };
    // χ in the frame of B[x + 2, y], then ι.
    $s = [
      row0[0], row0[1], row0[2], row0[3], row0[4], row1[0], row1[1], row1[2], row1[3], row1[4], row2[0], row2[1],
      row2[2], row2[3], row2[4], row3[0], row3[1], row3[2], row3[3], row3[4], row4[0], row4[1], row4[2], row4[3],
      row4[4],
    ];
  }};
}

/// Applies Keccak-f[1600] to `state`.
#[inline(never)]
pub(crate) fn keccakf(state: &mut [u64; 25]) {
  let mut s = *state;
  round!(s, 0);
  round!(s, 1);
  round!(s, 2);
  round!(s, 3);
  round!(s, 4);
  round!(s, 5);
  round!(s, 6);
  round!(s, 7);
  round!(s, 8);
  round!(s, 9);
  round!(s, 10);
  round!(s, 11);
  round!(s, 12);
  round!(s, 13);
  round!(s, 14);
  round!(s, 15);
  round!(s, 16);
  round!(s, 17);
  round!(s, 18);
  round!(s, 19);
  round!(s, 20);
  round!(s, 21);
  round!(s, 22);
  round!(s, 23);
  for (lane, (out, value)) in state.iter_mut().zip(s).enumerate() {
    *out = value.rotate_left(SCHEDULE.restore[lane]);
  }
}
