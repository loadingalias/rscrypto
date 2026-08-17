//! aarch64 NEON BlaMka compression kernel for Argon2.
//!
//! Vectorises the row and column P-rounds from RFC 9106 §3.6 with 4-way
//! parallelism across the four (a, b, c, d) lanes that feed each G. Each
//! "lane" of 4 u64s lives in two `uint64x2_t` NEON registers (lo/hi);
//! 16 NEON registers carry the live state of one 16-u64 P-round.
//!
//! # Vectorisation topology
//!
//! A P-round operates on 16 u64 words:
//!
//! ```text
//! v = [ a0 a1 a2 a3 | b0 b1 b2 b3 | c0 c1 c2 c3 | d0 d1 d2 d3 ]
//! ```
//!
//! Column step: `GB(a_i, b_i, c_i, d_i)` for `i ∈ 0..4` — already 4-way
//! parallel with lane `i` in SIMD position `i`.
//!
//! Diagonal step: `GB(a0,b1,c2,d3)`, `GB(a1,b2,c3,d0)`,
//! `GB(a2,b3,c0,d1)`, `GB(a3,b0,c1,d2)` — rotate the `b` lane by 1, `c`
//! lane by 2, `d` lane by 3 (via `vextq_u64`), run the same 4-way GB,
//! then rotate back.
//!
//! # BlaMka multiply
//!
//! `2 · lsb(a) · lsb(b)` vectorises cleanly on NEON: `vmovn_u64` extracts
//! the low 32 bits of each u64 lane into a `uint32x2_t`; `vmull_u32`
//! returns a `uint64x2_t` of `(lsb(a) × lsb(b))` per lane. A single left
//! shift by 1 gives the `2·lsb(a)·lsb(b)` term that's added into
//! `a += b + …` per RFC 9106 §3.6.1.
//!
//! # Rotations
//!
//! - ROR 32: `vrev64q_u32` swaps the two u32 halves of each u64.
//! - ROR 24 / 16: byte shuffle via `vqtbl1q_u8` with fixed index tables.
//! - ROR 63 ≡ ROL 1: `vsriq_n_u64::<63>(vshlq_n_u64::<1>(x), x)`.

use core::arch::aarch64::{
  uint8x16_t, uint64x2_t, vaddq_u64, veorq_u64, vextq_u64, vld1q_u8, vld1q_u64, vmovn_u64, vmull_u32, vqtbl1q_u8,
  vreinterpretq_u8_u64, vreinterpretq_u32_u64, vreinterpretq_u64_u8, vreinterpretq_u64_u32, vrev64q_u32, vshlq_n_u64,
  vsriq_n_u64, vst1q_u64,
};

use super::BLOCK_WORDS;

struct NeonState {
  a_lo: uint64x2_t,
  a_hi: uint64x2_t,
  b_lo: uint64x2_t,
  b_hi: uint64x2_t,
  c_lo: uint64x2_t,
  c_hi: uint64x2_t,
  d_lo: uint64x2_t,
  d_hi: uint64x2_t,
}

/// NEON BlaMka compression kernel.
///
/// # Safety
///
/// - `target_arch = "aarch64"` is enforced at compile time (module gate).
/// - NEON is baseline on aarch64 — no runtime capability check needed.
/// - The three `[u64; BLOCK_WORDS]` buffers are fixed-size arrays; the kernel reads/writes only
///   within their bounds.
#[target_feature(enable = "neon")]
pub(super) unsafe fn compress_neon(
  dst: &mut [u64; BLOCK_WORDS],
  x: &[u64; BLOCK_WORDS],
  y: &[u64; BLOCK_WORDS],
  xor_into: bool,
) {
  // R = X XOR Y held as 64 uint64x2_t lanes (2 u64 per reg × 64 = 128 u64).
  // SAFETY: `core::mem::zeroed::<uint64x2_t>()` is an all-zero register
  // value; uint64x2_t has no invalid bit patterns.
  let mut r = unsafe { [core::mem::zeroed::<uint64x2_t>(); 64] };
  let mut q = r;
  for i in 0usize..64 {
    let offset = i.strict_mul(2);
    // SAFETY: x and y are [u64; BLOCK_WORDS] with BLOCK_WORDS == 128; the
    // loop reads 2 u64 per iteration at offset `2 * i` for i ∈ 0..64, so
    // the last read is at words 126..=127. NEON inherited from the fn
    // target_feature.
    let (xv, yv) = unsafe { (vld1q_u64(x.as_ptr().add(offset)), vld1q_u64(y.as_ptr().add(offset))) };
    r[i] = veorq_u64(xv, yv);
    q[i] = r[i];
  }

  // Row pass: 8 rows × 16 u64 per row = 8 P-rounds on contiguous blocks
  // of 8 uint64x2_t.
  for row in 0usize..8 {
    let base = row.strict_mul(8);
    let mut state = NeonState {
      a_lo: q[base],
      a_hi: q[base.strict_add(1)],
      b_lo: q[base.strict_add(2)],
      b_hi: q[base.strict_add(3)],
      c_lo: q[base.strict_add(4)],
      c_hi: q[base.strict_add(5)],
      d_lo: q[base.strict_add(6)],
      d_hi: q[base.strict_add(7)],
    };
    // SAFETY: NEON inherited from outer target_feature.
    unsafe { p_round_neon(&mut state) };
    q[base] = state.a_lo;
    q[base.strict_add(1)] = state.a_hi;
    q[base.strict_add(2)] = state.b_lo;
    q[base.strict_add(3)] = state.b_hi;
    q[base.strict_add(4)] = state.c_lo;
    q[base.strict_add(5)] = state.c_hi;
    q[base.strict_add(6)] = state.d_lo;
    q[base.strict_add(7)] = state.d_hi;
  }

  // Column pass: each P-round reads 16 u64s at stride-16 positions
  // (`q[col*2], q[col*2+1], q[col*2+16], q[col*2+17], …, q[col*2+112],
  // q[col*2+113]`). In uint64x2_t indexing each "col*2" pair corresponds
  // to a single uint64x2_t at index `col`, and the stride-16 becomes
  // stride 8 in uint64x2_t space.
  for col in 0usize..8 {
    let mut state = NeonState {
      a_lo: q[col],
      a_hi: q[col.strict_add(8)],
      b_lo: q[col.strict_add(16)],
      b_hi: q[col.strict_add(24)],
      c_lo: q[col.strict_add(32)],
      c_hi: q[col.strict_add(40)],
      d_lo: q[col.strict_add(48)],
      d_hi: q[col.strict_add(56)],
    };
    // SAFETY: NEON inherited from outer target_feature.
    unsafe { p_round_neon(&mut state) };
    q[col] = state.a_lo;
    q[col.strict_add(8)] = state.a_hi;
    q[col.strict_add(16)] = state.b_lo;
    q[col.strict_add(24)] = state.b_hi;
    q[col.strict_add(32)] = state.c_lo;
    q[col.strict_add(40)] = state.c_hi;
    q[col.strict_add(48)] = state.d_lo;
    q[col.strict_add(56)] = state.d_hi;
  }

  // Final XOR with R, fused with the dst store.
  for i in 0usize..64 {
    let final_v = veorq_u64(q[i], r[i]);
    // SAFETY: dst[2*i..2*i+2] is within BLOCK_WORDS for i ∈ 0..64. NEON
    // inherited from the fn target_feature.
    unsafe {
      let offset = i.strict_mul(2);
      if xor_into {
        let cur = vld1q_u64(dst.as_ptr().add(offset));
        vst1q_u64(dst.as_mut_ptr().add(offset), veorq_u64(cur, final_v));
      } else {
        vst1q_u64(dst.as_mut_ptr().add(offset), final_v);
      }
    }
  }
}

// ─── 4-way parallel P-round ────────────────────────────────────────────────

/// One 16-word P-round applied to state held as 4 "rows" of 4 u64s each,
/// with each row in a pair of `uint64x2_t` registers (lo/hi).
///
/// # Safety
///
/// Must be called from a context with NEON enabled (e.g. inside a
/// `#[target_feature(enable = "neon")]` function). NEON is baseline on
/// aarch64 so any aarch64 build satisfies this, but the `unsafe fn`
/// signature preserves the contract for completeness.
#[inline(always)]
unsafe fn p_round_neon(state: &mut NeonState) {
  // Column step — 4 parallel GBs.
  // SAFETY: NEON precondition inherited from caller.
  unsafe { gb_neon(state) };

  // Diagonalise: rotate b by 1, c by 2, d by 3 across the 4-lane row.
  // SAFETY: vextq_u64 only operates on register values, no memory access.
  unsafe {
    let b_lo = vextq_u64::<1>(state.b_lo, state.b_hi);
    let b_hi = vextq_u64::<1>(state.b_hi, state.b_lo);
    state.b_lo = b_lo;
    state.b_hi = b_hi;

    // c: rotate by 2 == swap lo/hi.
    core::mem::swap(&mut state.c_lo, &mut state.c_hi);

    let d_lo = vextq_u64::<1>(state.d_hi, state.d_lo);
    let d_hi = vextq_u64::<1>(state.d_lo, state.d_hi);
    state.d_lo = d_lo;
    state.d_hi = d_hi;
  }

  // Diagonal step — 4 parallel GBs on the rotated state.
  // SAFETY: NEON precondition inherited from caller.
  unsafe { gb_neon(state) };

  // Undo diagonalisation: rotate b by -1, c by -2, d by -3.
  // SAFETY: vextq_u64 only operates on register values.
  unsafe {
    let b_lo = vextq_u64::<1>(state.b_hi, state.b_lo);
    let b_hi = vextq_u64::<1>(state.b_lo, state.b_hi);
    state.b_lo = b_lo;
    state.b_hi = b_hi;

    // c: undo swap.
    core::mem::swap(&mut state.c_lo, &mut state.c_hi);

    let d_lo = vextq_u64::<1>(state.d_lo, state.d_hi);
    let d_hi = vextq_u64::<1>(state.d_hi, state.d_lo);
    state.d_lo = d_lo;
    state.d_hi = d_hi;
  }
}

// ─── 4-way parallel BlaMka G ───────────────────────────────────────────────

/// 4-way parallel BlaMka G operating on `(a, b, c, d)` rows, each row in
/// two `uint64x2_t` registers.
///
/// # Safety
///
/// Must be called from a context with NEON enabled (inherited from the
/// outer `#[target_feature(enable = "neon")]` entry point).
#[inline(always)]
unsafe fn gb_neon(state: &mut NeonState) {
  // SAFETY: NEON precondition inherited; all ops below are register-only.
  unsafe {
    // Step 1: a ← a + b + 2·lsb(a)·lsb(b)
    let p_lo = bla_mul(state.a_lo, state.b_lo);
    let p_hi = bla_mul(state.a_hi, state.b_hi);
    state.a_lo = vaddq_u64(vaddq_u64(state.a_lo, state.b_lo), p_lo);
    state.a_hi = vaddq_u64(vaddq_u64(state.a_hi, state.b_hi), p_hi);
    // d ← (d ^ a) ROR 32
    state.d_lo = ror32(veorq_u64(state.d_lo, state.a_lo));
    state.d_hi = ror32(veorq_u64(state.d_hi, state.a_hi));

    // Step 2: c ← c + d + 2·lsb(c)·lsb(d)
    let p_lo = bla_mul(state.c_lo, state.d_lo);
    let p_hi = bla_mul(state.c_hi, state.d_hi);
    state.c_lo = vaddq_u64(vaddq_u64(state.c_lo, state.d_lo), p_lo);
    state.c_hi = vaddq_u64(vaddq_u64(state.c_hi, state.d_hi), p_hi);
    // b ← (b ^ c) ROR 24
    state.b_lo = ror24(veorq_u64(state.b_lo, state.c_lo));
    state.b_hi = ror24(veorq_u64(state.b_hi, state.c_hi));

    // Step 3: a ← a + b + 2·lsb(a)·lsb(b)
    let p_lo = bla_mul(state.a_lo, state.b_lo);
    let p_hi = bla_mul(state.a_hi, state.b_hi);
    state.a_lo = vaddq_u64(vaddq_u64(state.a_lo, state.b_lo), p_lo);
    state.a_hi = vaddq_u64(vaddq_u64(state.a_hi, state.b_hi), p_hi);
    // d ← (d ^ a) ROR 16
    state.d_lo = ror16(veorq_u64(state.d_lo, state.a_lo));
    state.d_hi = ror16(veorq_u64(state.d_hi, state.a_hi));

    // Step 4: c ← c + d + 2·lsb(c)·lsb(d)
    let p_lo = bla_mul(state.c_lo, state.d_lo);
    let p_hi = bla_mul(state.c_hi, state.d_hi);
    state.c_lo = vaddq_u64(vaddq_u64(state.c_lo, state.d_lo), p_lo);
    state.c_hi = vaddq_u64(vaddq_u64(state.c_hi, state.d_hi), p_hi);
    // b ← (b ^ c) ROR 63 ≡ ROL 1
    state.b_lo = ror63(veorq_u64(state.b_lo, state.c_lo));
    state.b_hi = ror63(veorq_u64(state.b_hi, state.c_hi));
  }
}

// ─── Micro-ops ─────────────────────────────────────────────────────────────

/// `2 · lsb(a) · lsb(b)` applied lane-wise over a `uint64x2_t`.
///
/// # Safety
///
/// Inherits the NEON precondition from the caller; no memory access.
#[inline(always)]
unsafe fn bla_mul(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
  // SAFETY: all register-only ops under inherited NEON.
  unsafe {
    let al = vmovn_u64(a);
    let bl = vmovn_u64(b);
    vshlq_n_u64::<1>(vmull_u32(al, bl))
  }
}

/// ROR 32 via u32-half swap on each u64 lane.
///
/// # Safety
///
/// Inherits the NEON precondition from the caller; no memory access.
#[inline(always)]
unsafe fn ror32(v: uint64x2_t) -> uint64x2_t {
  // SAFETY: reinterpret + byte-swap on register values only.
  unsafe { vreinterpretq_u64_u32(vrev64q_u32(vreinterpretq_u32_u64(v))) }
}

/// ROR 24 via byte shuffle. Byte order within each u64 lane:
/// `[3,4,5,6,7,0,1,2]` for lane 0, and `[+8]` for lane 1.
///
/// # Safety
///
/// Inherits the NEON precondition; reads a fixed 16-byte constant table.
#[inline(always)]
unsafe fn ror24(v: uint64x2_t) -> uint64x2_t {
  static IDX: [u8; 16] = [3, 4, 5, 6, 7, 0, 1, 2, 11, 12, 13, 14, 15, 8, 9, 10];
  // SAFETY: IDX is a static 16-byte array; vld1q_u8 reads exactly 16 bytes.
  unsafe {
    let table: uint8x16_t = vld1q_u8(IDX.as_ptr());
    vreinterpretq_u64_u8(vqtbl1q_u8(vreinterpretq_u8_u64(v), table))
  }
}

/// ROR 16 via byte shuffle. Byte order within each u64 lane:
/// `[2,3,4,5,6,7,0,1]` for lane 0, and `[+8]` for lane 1.
///
/// # Safety
///
/// Inherits the NEON precondition; reads a fixed 16-byte constant table.
#[inline(always)]
unsafe fn ror16(v: uint64x2_t) -> uint64x2_t {
  static IDX: [u8; 16] = [2, 3, 4, 5, 6, 7, 0, 1, 10, 11, 12, 13, 14, 15, 8, 9];
  // SAFETY: IDX is a static 16-byte array; vld1q_u8 reads exactly 16 bytes.
  unsafe {
    let table: uint8x16_t = vld1q_u8(IDX.as_ptr());
    vreinterpretq_u64_u8(vqtbl1q_u8(vreinterpretq_u8_u64(v), table))
  }
}

/// ROR 63 ≡ ROL 1 = `(x << 1) | (x >> 63)`.
///
/// # Safety
///
/// Inherits the NEON precondition; no memory access.
#[inline(always)]
unsafe fn ror63(v: uint64x2_t) -> uint64x2_t {
  // SAFETY: shift/insert on register values only.
  unsafe { vsriq_n_u64::<63>(vshlq_n_u64::<1>(v), v) }
}
