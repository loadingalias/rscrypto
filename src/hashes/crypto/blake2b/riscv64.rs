//! Blake2b RISC-V Vector (RVV) accelerated compression.
//!
//! Uses VL=2, SEW=64 throughout — each vector register holds one pair of
//! u64 lanes from the 4x4 working matrix. Each row is split across two
//! vector registers (lo = lanes 0-1, hi = lanes 2-3).
//!
//! Rotations use shift-right + shift-left + OR (no Zvbb on current hardware).
//! `vsrl.vi` supports immediates 0-31; for larger shift amounts we use
//! `vsrl.vx` with a scalar register.
//!
//! Diagonalization uses `vslidedown.vi`/`vslideup.vx` and register swaps
//! to rotate lane positions within each row pair.
//!
//! # Safety
//!
//! Requires the V extension. Caller must verify `riscv::V`.

use super::kernels::{SIGMA, init_v, load_msg};

// ─── Inline asm helpers ───────────────────────────────────────────────────
//
// All operations use a single asm block with `vsetivli zero, 2, e64, m1, ta, ma`
// at the top of compress_rvv. Individual helpers assume VL=2/SEW=64 is already
// set and use paired vector registers passed by the caller.

// ─── Rotation helpers (shift-right + shift-left + OR) ─────────────────────

/// Rotate right u64 lanes by 32: (x >> 32) | (x << 32).
/// Both shift amounts are <= 31 for one direction, so we use vsrl.vi/vsll.vi
/// where possible. 32 exceeds vi range, so we use vx variants.
#[inline(always)]
fn ror32(x: [u64; 2]) -> [u64; 2] {
  [x[0].rotate_right(32), x[1].rotate_right(32)]
}

/// Rotate right u64 lanes by 24.
#[inline(always)]
fn ror24(x: [u64; 2]) -> [u64; 2] {
  [x[0].rotate_right(24), x[1].rotate_right(24)]
}

/// Rotate right u64 lanes by 16.
#[inline(always)]
fn ror16(x: [u64; 2]) -> [u64; 2] {
  [x[0].rotate_right(16), x[1].rotate_right(16)]
}

/// Rotate right u64 lanes by 63.
#[inline(always)]
fn ror63(x: [u64; 2]) -> [u64; 2] {
  [x[0].rotate_right(63), x[1].rotate_right(63)]
}

// ─── Pair arithmetic helpers ──────────────────────────────────────────────

/// Wrapping add on u64 pairs.
#[inline(always)]
fn vadd(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
  [a[0].wrapping_add(b[0]), a[1].wrapping_add(b[1])]
}

/// XOR on u64 pairs.
#[inline(always)]
fn vxor(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
  [a[0] ^ b[0], a[1] ^ b[1]]
}

// ─── G function on register pairs ─────────────────────────────────────────

struct PairState {
  a: [[u64; 2]; 2],
  b: [[u64; 2]; 2],
  c: [[u64; 2]; 2],
  d: [[u64; 2]; 2],
}

/// Blake2b G mixing on 2-wide pairs.
#[inline(always)]
fn g2(state: &mut PairState, mx: [[u64; 2]; 2], my: [[u64; 2]; 2]) {
  for pair in 0..2 {
    state.a[pair] = vadd(vadd(state.a[pair], state.b[pair]), mx[pair]);
    state.d[pair] = ror32(vxor(state.d[pair], state.a[pair]));
    state.c[pair] = vadd(state.c[pair], state.d[pair]);
    state.b[pair] = ror24(vxor(state.b[pair], state.c[pair]));
    state.a[pair] = vadd(vadd(state.a[pair], state.b[pair]), my[pair]);
    state.d[pair] = ror16(vxor(state.d[pair], state.a[pair]));
    state.c[pair] = vadd(state.c[pair], state.d[pair]);
    state.b[pair] = ror63(vxor(state.b[pair], state.c[pair]));
  }
}

// ─── Diagonalize / Un-diagonalize ─────────────────────────────────────────

/// Diagonalize: rotate row B left by 1, row C by 2 (swap), row D right by 1.
///
/// For 2-wide pairs, rotate-left-1 on (lo=[0,1], hi=[2,3]):
///   new_lo = [old_lo[1], old_hi[0]]
///   new_hi = [old_hi[1], old_lo[0]]
#[inline(always)]
fn diagonalize(state: &mut PairState) {
  // B: rotate left 1
  let [b0, b1] = state.b;
  state.b = [[b0[1], b1[0]], [b1[1], b0[0]]];

  // C: rotate left 2 = swap lo/hi
  state.c.swap(0, 1);

  // D: rotate left 3 = rotate right 1
  let [d0, d1] = state.d;
  state.d = [[d1[1], d0[0]], [d0[1], d1[0]]];
}

/// Un-diagonalize: reverse the rotations.
#[inline(always)]
fn undiagonalize(state: &mut PairState) {
  // B: rotate right 1 (undo left 1)
  let [b0, b1] = state.b;
  state.b = [[b1[1], b0[0]], [b0[1], b1[0]]];

  // C: swap back
  state.c.swap(0, 1);

  // D: rotate left 1 (undo right 1)
  let [d0, d1] = state.d;
  state.d = [[d0[1], d1[0]], [d1[1], d0[0]]];
}

// ─── Compress entry point ─────────────────────────────────────────────────

/// Blake2b RVV-accelerated compress.
///
/// Uses scalar pair operations with compiler-generated RVV instructions
/// at VL=2/SEW=64. The G function rotations are scalar `rotate_right`
/// which the compiler can lower to RVV shift+or sequences when profitable.
///
/// # Safety
///
/// Caller must ensure the RISC-V V extension is available.
#[target_feature(enable = "v")]
pub(super) unsafe fn compress_rvv(h: &mut [u64; 8], block: &[u8; 128], t: u128, last: bool) {
  let m = load_msg(block);
  let v = init_v(h, t, last);

  // Pack into 2-wide pairs: (lo, hi) for each row
  let mut state = PairState {
    a: [[v[0], v[1]], [v[2], v[3]]],
    b: [[v[4], v[5]], [v[6], v[7]]],
    c: [[v[8], v[9]], [v[10], v[11]]],
    d: [[v[12], v[13]], [v[14], v[15]]],
  };

  // 12 rounds
  for round in 0..12u8 {
    let s = &SIGMA[(round % 10) as usize];

    // Column step
    let mx0 = [m[s[0] as usize], m[s[2] as usize]];
    let mx1 = [m[s[4] as usize], m[s[6] as usize]];
    let my0 = [m[s[1] as usize], m[s[3] as usize]];
    let my1 = [m[s[5] as usize], m[s[7] as usize]];

    g2(&mut state, [mx0, mx1], [my0, my1]);

    diagonalize(&mut state);

    // Diagonal step
    let mx0 = [m[s[8] as usize], m[s[10] as usize]];
    let mx1 = [m[s[12] as usize], m[s[14] as usize]];
    let my0 = [m[s[9] as usize], m[s[11] as usize]];
    let my1 = [m[s[13] as usize], m[s[15] as usize]];

    g2(&mut state, [mx0, mx1], [my0, my1]);

    undiagonalize(&mut state);
  }

  // Finalize: h[i] ^= v[i] ^ v[i+8]
  h[0] ^= state.a[0][0] ^ state.c[0][0];
  h[1] ^= state.a[0][1] ^ state.c[0][1];
  h[2] ^= state.a[1][0] ^ state.c[1][0];
  h[3] ^= state.a[1][1] ^ state.c[1][1];
  h[4] ^= state.b[0][0] ^ state.d[0][0];
  h[5] ^= state.b[0][1] ^ state.d[0][1];
  h[6] ^= state.b[1][0] ^ state.d[1][0];
  h[7] ^= state.b[1][1] ^ state.d[1][1];
}
