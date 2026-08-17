//! IBM Z (s390x) z/Vector BlaMka compression kernel for Argon2.
//!
//! Mirrors the Blake2b s390x kernel layout: u64 add via `vag`, XOR via
//! `vx`, and the four BlaMka rotations (32, 24, 16, 63) via `verllg`
//! immediates (`32`, `40`, `48`, `1` for ROR-equivalent ROL counts).
//!
//! # BlaMka multiply
//!
//! `2 · lsb(a) · lsb(b)` lane-wise. The base z13 vector facility has
//! no native `u64 × u64 → u64` instruction (only word-width
//! multiply-even/odd at 32×32→64). Rather than encode the cross-lane
//! VPMULUDQ-style pattern, the multiply step extracts each lane to
//! scalar and reuses the host's 64-bit `mlgr`/`msgr` integer multiply
//! — the dependency chain is short and the resulting code stays
//! readable. If a future z16+ kernel needs the SIMD multiply, replace
//! `bla_mul` only.
//!
//! # Endianness
//!
//! s390x is big-endian. Memory layout for `[u64; 2]` aligned through
//! `core::ptr::read_unaligned`/`write_unaligned` already accounts for
//! native byte order, so no explicit byte swap is needed.
//!
//! # Safety
//!
//! Requires the z13+ vector facility. Caller must verify
//! `s390x::VECTOR`.

#![cfg(target_arch = "s390x")]

use core::simd::i64x2;

use super::BLOCK_WORDS;

struct VectorState {
  a_lo: i64x2,
  a_hi: i64x2,
  b_lo: i64x2,
  b_hi: i64x2,
  c_lo: i64x2,
  c_hi: i64x2,
  d_lo: i64x2,
  d_hi: i64x2,
}

// ─── Inline-asm primitives (z13+ vector facility) ──────────────────────────

#[target_feature(enable = "vector")]
/// # Safety
///
/// The current CPU must support the z13+ vector facility.
unsafe fn vag(a: i64x2, b: i64x2) -> i64x2 {
  let out: i64x2;
  // SAFETY: z13+ vector facility via target_feature.
  unsafe {
    core::arch::asm!(
      "vag {out}, {a}, {b}",
      out = lateout(vreg) out,
      a = in(vreg) a,
      b = in(vreg) b,
      options(nomem, nostack, pure)
    );
  }
  out
}

#[target_feature(enable = "vector")]
/// # Safety
///
/// The current CPU must support the z13+ vector facility.
unsafe fn vx(a: i64x2, b: i64x2) -> i64x2 {
  let out: i64x2;
  // SAFETY: z13+ vector facility via target_feature.
  unsafe {
    core::arch::asm!(
      "vx {out}, {a}, {b}",
      out = lateout(vreg) out,
      a = in(vreg) a,
      b = in(vreg) b,
      options(nomem, nostack, pure)
    );
  }
  out
}

/// `verllg` ROL by 32 = ROR 32.
///
/// # Safety
///
/// The current CPU must support the z13+ vector facility.
#[target_feature(enable = "vector")]
unsafe fn verllg_32(x: i64x2) -> i64x2 {
  let out: i64x2;
  // SAFETY: z13+ vector facility via target_feature.
  unsafe {
    core::arch::asm!(
      "verllg {out}, {x}, 32",
      out = lateout(vreg) out,
      x = in(vreg) x,
      options(nomem, nostack, pure)
    );
  }
  out
}

/// `verllg` ROL by 40 = ROR 24.
///
/// # Safety
///
/// The current CPU must support the z13+ vector facility.
#[target_feature(enable = "vector")]
unsafe fn verllg_40(x: i64x2) -> i64x2 {
  let out: i64x2;
  // SAFETY: z13+ vector facility via target_feature.
  unsafe {
    core::arch::asm!(
      "verllg {out}, {x}, 40",
      out = lateout(vreg) out,
      x = in(vreg) x,
      options(nomem, nostack, pure)
    );
  }
  out
}

/// `verllg` ROL by 48 = ROR 16.
///
/// # Safety
///
/// The current CPU must support the z13+ vector facility.
#[target_feature(enable = "vector")]
unsafe fn verllg_48(x: i64x2) -> i64x2 {
  let out: i64x2;
  // SAFETY: z13+ vector facility via target_feature.
  unsafe {
    core::arch::asm!(
      "verllg {out}, {x}, 48",
      out = lateout(vreg) out,
      x = in(vreg) x,
      options(nomem, nostack, pure)
    );
  }
  out
}

/// `verllg` ROL by 1 = ROR 63.
///
/// # Safety
///
/// The current CPU must support the z13+ vector facility.
#[target_feature(enable = "vector")]
unsafe fn verllg_1(x: i64x2) -> i64x2 {
  let out: i64x2;
  // SAFETY: z13+ vector facility via target_feature.
  unsafe {
    core::arch::asm!(
      "verllg {out}, {x}, 1",
      out = lateout(vreg) out,
      x = in(vreg) x,
      options(nomem, nostack, pure)
    );
  }
  out
}

#[inline(always)]
fn pair_a1_b0(a: i64x2, b: i64x2) -> i64x2 {
  core::simd::simd_swizzle!(a, b, [1, 2])
}

#[inline(always)]
fn pair_b1_a0(a: i64x2, b: i64x2) -> i64x2 {
  core::simd::simd_swizzle!(a, b, [3, 0])
}

#[inline(always)]
/// # Safety
///
/// `p` must remain valid to read two initialized `u64` values from one
/// allocation.
unsafe fn vload_pair(p: *const u64) -> i64x2 {
  // SAFETY: caller ensures p is valid for 2 × u64.
  unsafe { core::ptr::read_unaligned(p.cast()) }
}

#[inline(always)]
/// # Safety
///
/// `p` must remain valid to write two `u64` values into one allocation.
unsafe fn vstore_pair(p: *mut u64, v: i64x2) {
  // SAFETY: caller ensures p is valid for 2 × u64.
  unsafe { core::ptr::write_unaligned(p.cast(), v) }
}

/// `2 · lsb(a) · lsb(b)` lane-wise — scalar fallback (z13 has no native
/// u64×u64→u64 vector multiply; the masked u32 product fits in u64 so
/// the math is exact).
#[inline(always)]
fn bla_mul(a: i64x2, b: i64x2) -> i64x2 {
  let aa = a.to_array();
  let bb = b.to_array();
  const MASK: u64 = 0xffff_ffff;
  let r0 = (aa[0].cast_unsigned() & MASK)
    .wrapping_mul(bb[0].cast_unsigned() & MASK)
    .wrapping_shl(1);
  let r1 = (aa[1].cast_unsigned() & MASK)
    .wrapping_mul(bb[1].cast_unsigned() & MASK)
    .wrapping_shl(1);
  i64x2::from_array([r0.cast_signed(), r1.cast_signed()])
}

// ─── 4-way P-round ─────────────────────────────────────────────────────────

/// # Safety
///
/// The current CPU must support the z13+ vector facility.
#[target_feature(enable = "vector")]
unsafe fn p_round(state: &mut VectorState) {
  // SAFETY: vector facility inherited.
  unsafe {
    gb(state);

    let b_lo = state.b_lo;
    let b_hi = state.b_hi;
    state.b_lo = pair_a1_b0(b_lo, b_hi);
    state.b_hi = pair_b1_a0(b_lo, b_hi);

    core::mem::swap(&mut state.c_lo, &mut state.c_hi);

    let d_lo = state.d_lo;
    let d_hi = state.d_hi;
    state.d_lo = pair_b1_a0(d_lo, d_hi);
    state.d_hi = pair_a1_b0(d_lo, d_hi);

    gb(state);

    let b_lo = state.b_lo;
    let b_hi = state.b_hi;
    state.b_lo = pair_b1_a0(b_lo, b_hi);
    state.b_hi = pair_a1_b0(b_lo, b_hi);

    core::mem::swap(&mut state.c_lo, &mut state.c_hi);

    let d_lo = state.d_lo;
    let d_hi = state.d_hi;
    state.d_lo = pair_a1_b0(d_lo, d_hi);
    state.d_hi = pair_b1_a0(d_lo, d_hi);
  }
}

// ─── 4-way BlaMka G ────────────────────────────────────────────────────────

/// # Safety
///
/// The current CPU must support the z13+ vector facility.
#[target_feature(enable = "vector")]
unsafe fn gb(state: &mut VectorState) {
  // SAFETY: vector facility inherited.
  unsafe {
    // Step 1
    let p_lo = bla_mul(state.a_lo, state.b_lo);
    let p_hi = bla_mul(state.a_hi, state.b_hi);
    state.a_lo = vag(vag(state.a_lo, state.b_lo), p_lo);
    state.a_hi = vag(vag(state.a_hi, state.b_hi), p_hi);
    state.d_lo = verllg_32(vx(state.d_lo, state.a_lo));
    state.d_hi = verllg_32(vx(state.d_hi, state.a_hi));

    // Step 2
    let p_lo = bla_mul(state.c_lo, state.d_lo);
    let p_hi = bla_mul(state.c_hi, state.d_hi);
    state.c_lo = vag(vag(state.c_lo, state.d_lo), p_lo);
    state.c_hi = vag(vag(state.c_hi, state.d_hi), p_hi);
    state.b_lo = verllg_40(vx(state.b_lo, state.c_lo));
    state.b_hi = verllg_40(vx(state.b_hi, state.c_hi));

    // Step 3
    let p_lo = bla_mul(state.a_lo, state.b_lo);
    let p_hi = bla_mul(state.a_hi, state.b_hi);
    state.a_lo = vag(vag(state.a_lo, state.b_lo), p_lo);
    state.a_hi = vag(vag(state.a_hi, state.b_hi), p_hi);
    state.d_lo = verllg_48(vx(state.d_lo, state.a_lo));
    state.d_hi = verllg_48(vx(state.d_hi, state.a_hi));

    // Step 4
    let p_lo = bla_mul(state.c_lo, state.d_lo);
    let p_hi = bla_mul(state.c_hi, state.d_hi);
    state.c_lo = vag(vag(state.c_lo, state.d_lo), p_lo);
    state.c_hi = vag(vag(state.c_hi, state.d_hi), p_hi);
    state.b_lo = verllg_1(vx(state.b_lo, state.c_lo));
    state.b_hi = verllg_1(vx(state.b_hi, state.c_hi));
  }
}

// ─── Compress entry point ──────────────────────────────────────────────────

/// IBM Z z/Vector BlaMka compression kernel.
///
/// # Safety
///
/// - `target_arch = "s390x"` enforced at compile time (module gate).
/// - Caller must have the z13+ vector facility — enforced via `#[target_feature(enable =
///   "vector")]`.
/// - The three `[u64; BLOCK_WORDS]` buffers are fixed-size arrays.
#[target_feature(enable = "vector")]
pub(super) unsafe fn compress_vector(
  dst: &mut [u64; BLOCK_WORDS],
  x: &[u64; BLOCK_WORDS],
  y: &[u64; BLOCK_WORDS],
  xor_into: bool,
) {
  // SAFETY: vector facility enabled by target_feature; pointer-based
  // loads / stores stay within fixed-size arrays.
  unsafe {
    let mut r = [0u64; BLOCK_WORDS];
    let mut q = [0u64; BLOCK_WORDS];
    let mut i = 0usize;
    while i < BLOCK_WORDS {
      let xv = vload_pair(x.as_ptr().add(i));
      let yv = vload_pair(y.as_ptr().add(i));
      let rv = vx(xv, yv);
      vstore_pair(r.as_mut_ptr().add(i), rv);
      vstore_pair(q.as_mut_ptr().add(i), rv);
      i = i.strict_add(2);
    }

    // Row pass.
    let mut row = 0usize;
    while row < 8 {
      let base = row.strict_mul(16);
      let mut state = VectorState {
        a_lo: vload_pair(q.as_ptr().add(base)),
        a_hi: vload_pair(q.as_ptr().add(base.strict_add(2))),
        b_lo: vload_pair(q.as_ptr().add(base.strict_add(4))),
        b_hi: vload_pair(q.as_ptr().add(base.strict_add(6))),
        c_lo: vload_pair(q.as_ptr().add(base.strict_add(8))),
        c_hi: vload_pair(q.as_ptr().add(base.strict_add(10))),
        d_lo: vload_pair(q.as_ptr().add(base.strict_add(12))),
        d_hi: vload_pair(q.as_ptr().add(base.strict_add(14))),
      };

      p_round(&mut state);

      vstore_pair(q.as_mut_ptr().add(base), state.a_lo);
      vstore_pair(q.as_mut_ptr().add(base.strict_add(2)), state.a_hi);
      vstore_pair(q.as_mut_ptr().add(base.strict_add(4)), state.b_lo);
      vstore_pair(q.as_mut_ptr().add(base.strict_add(6)), state.b_hi);
      vstore_pair(q.as_mut_ptr().add(base.strict_add(8)), state.c_lo);
      vstore_pair(q.as_mut_ptr().add(base.strict_add(10)), state.c_hi);
      vstore_pair(q.as_mut_ptr().add(base.strict_add(12)), state.d_lo);
      vstore_pair(q.as_mut_ptr().add(base.strict_add(14)), state.d_hi);
      row = row.strict_add(1);
    }

    // Column pass.
    let mut col = 0usize;
    while col < 8 {
      let base = col.strict_mul(2);
      let mut state = VectorState {
        a_lo: vload_pair(q.as_ptr().add(base)),
        a_hi: vload_pair(q.as_ptr().add(base.strict_add(16))),
        b_lo: vload_pair(q.as_ptr().add(base.strict_add(32))),
        b_hi: vload_pair(q.as_ptr().add(base.strict_add(48))),
        c_lo: vload_pair(q.as_ptr().add(base.strict_add(64))),
        c_hi: vload_pair(q.as_ptr().add(base.strict_add(80))),
        d_lo: vload_pair(q.as_ptr().add(base.strict_add(96))),
        d_hi: vload_pair(q.as_ptr().add(base.strict_add(112))),
      };

      p_round(&mut state);

      vstore_pair(q.as_mut_ptr().add(base), state.a_lo);
      vstore_pair(q.as_mut_ptr().add(base.strict_add(16)), state.a_hi);
      vstore_pair(q.as_mut_ptr().add(base.strict_add(32)), state.b_lo);
      vstore_pair(q.as_mut_ptr().add(base.strict_add(48)), state.b_hi);
      vstore_pair(q.as_mut_ptr().add(base.strict_add(64)), state.c_lo);
      vstore_pair(q.as_mut_ptr().add(base.strict_add(80)), state.c_hi);
      vstore_pair(q.as_mut_ptr().add(base.strict_add(96)), state.d_lo);
      vstore_pair(q.as_mut_ptr().add(base.strict_add(112)), state.d_hi);
      col = col.strict_add(1);
    }

    // Final XOR with R, fused with dst store/xor.
    let mut i = 0usize;
    while i < BLOCK_WORDS {
      let qv = vload_pair(q.as_ptr().add(i));
      let rv = vload_pair(r.as_ptr().add(i));
      let f = vx(qv, rv);
      if xor_into {
        let cur = vload_pair(dst.as_ptr().add(i));
        vstore_pair(dst.as_mut_ptr().add(i), vx(cur, f));
      } else {
        vstore_pair(dst.as_mut_ptr().add(i), f);
      }
      i = i.strict_add(2);
    }
  }
}
