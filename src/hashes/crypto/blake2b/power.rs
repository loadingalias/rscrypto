//! Blake2b POWER VSX-accelerated compression for ppc64.
//!
//! Each row of the 4x4 u64 working matrix is split across two `i64x2`
//! registers (lo = lanes 0-1, hi = lanes 2-3). Diagonalization uses
//! `vperm` for lane rearrangement.
//!
//! All four Blake2b rotations (32, 24, 16, 63) use `vrld` (vector rotate
//! left doubleword), which takes the shift amount from a vector register.
//! Pre-built shift vectors eliminate repeated setup.
//!
//! # Endianness
//!
//! POWER in this codebase targets big-endian (`powerpc64`). The `m: [u64; 16]`
//! array from `load_msg` contains native-endian u64 values (already decoded
//! from little-endian wire format). We load pairs directly — no byte-swap.
//!
//! # Safety
//!
//! Requires POWER8+ with VSX. Caller must verify `power::VSX`.

#![allow(unsafe_code)]
#![allow(clippy::cast_possible_truncation, clippy::indexing_slicing)]

use core::simd::i64x2;

use super::kernels::{SIGMA, init_v, load_msg};

// ─── VSX primitive operations (inline asm, POWER8+) ───────────────────────

/// Vector add u64 lanes: `vaddudm`.
#[inline(always)]
#[target_feature(enable = "altivec", enable = "vsx", enable = "power8-vector")]
unsafe fn vaddudm(a: i64x2, b: i64x2) -> i64x2 {
  let out: i64x2;
  // SAFETY: POWER8+ VSX available via target_feature.
  unsafe {
    core::arch::asm!(
      "vaddudm {out}, {a}, {b}",
      out = lateout(vreg) out,
      a = in(vreg) a,
      b = in(vreg) b,
      options(nomem, nostack, pure)
    );
  }
  out
}

/// Vector XOR: `vxor`.
#[inline(always)]
#[target_feature(enable = "altivec", enable = "vsx", enable = "power8-vector")]
unsafe fn vxor(a: i64x2, b: i64x2) -> i64x2 {
  let out: i64x2;
  // SAFETY: POWER8+ VSX available via target_feature.
  unsafe {
    core::arch::asm!(
      "vxor {out}, {a}, {b}",
      out = lateout(vreg) out,
      a = in(vreg) a,
      b = in(vreg) b,
      options(nomem, nostack, pure)
    );
  }
  out
}

/// Vector rotate left doubleword: `vrld`.
///
/// Each u64 lane of `x` is rotated left by the corresponding u64 lane of `shift`.
#[inline(always)]
#[target_feature(enable = "altivec", enable = "vsx", enable = "power8-vector")]
unsafe fn vrld(x: i64x2, shift: i64x2) -> i64x2 {
  let out: i64x2;
  // SAFETY: POWER8+ VSX available via target_feature.
  unsafe {
    core::arch::asm!(
      "vrld {out}, {x}, {shift}",
      out = lateout(vreg) out,
      x = in(vreg) x,
      shift = in(vreg) shift,
      options(nomem, nostack, pure)
    );
  }
  out
}

/// Vector permute: `vperm`.
///
/// Selects bytes from the concatenation of `a:b` according to `mask`.
#[inline(always)]
#[target_feature(enable = "altivec", enable = "vsx", enable = "power8-vector")]
unsafe fn vperm(a: i64x2, b: i64x2, mask: i64x2) -> i64x2 {
  let out: i64x2;
  // SAFETY: POWER8+ VSX available via target_feature.
  unsafe {
    core::arch::asm!(
      "vperm {out}, {a}, {b}, {mask}",
      out = lateout(vreg) out,
      a = in(vreg) a,
      b = in(vreg) b,
      mask = in(vreg) mask,
      options(nomem, nostack, pure)
    );
  }
  out
}

// ─── Rotation shift vectors ──────────────────────────────────────────────

/// Pre-built rotation amounts for `vrld`.
/// Each is an i64x2 with both lanes set to the ROL amount.
///
/// Blake2b ROR 32 = ROL 32, ROR 24 = ROL 40, ROR 16 = ROL 48, ROR 63 = ROL 1.
const ROL_32: i64x2 = i64x2::from_array([32, 32]);
const ROL_40: i64x2 = i64x2::from_array([40, 40]);
const ROL_48: i64x2 = i64x2::from_array([48, 48]);
const ROL_1: i64x2 = i64x2::from_array([1, 1]);

// ─── Diagonalization permute masks ───────────────────────────────────────

/// `vperm` mask: extract [a[1], b[0]] from concatenation a:b.
///
/// On big-endian POWER, `vperm` indexes bytes 0..31 across the concatenation
/// of the first and second source registers:
///   bytes 0-15 = first source, bytes 16-31 = second source.
///
/// Element layout in each 128-bit register (BE):
///   element 0 = bytes 0-7, element 1 = bytes 8-15.
///
/// To get [a[1], b[0]]: take bytes 8-15 from a, then bytes 16-23 from b.
const VPERM_ROT_LEFT_1_LO: [u8; 16] = [
  8, 9, 10, 11, 12, 13, 14, 15, // a[1]
  16, 17, 18, 19, 20, 21, 22, 23, // b[0]
];

/// `vperm` mask: extract [b[1], a[0]] from concatenation a:b.
///
/// To get [b[1], a[0]]: take bytes 24-31 from b, then bytes 0-7 from a.
const VPERM_ROT_LEFT_1_HI: [u8; 16] = [
  24, 25, 26, 27, 28, 29, 30, 31, // b[1]
  0, 1, 2, 3, 4, 5, 6, 7, // a[0]
];

/// `vperm` mask: extract [b[1], a[0]] — used for rotate-right-1 lo.
/// Same as ROT_LEFT_1_HI.
const VPERM_ROT_RIGHT_1_LO: [u8; 16] = VPERM_ROT_LEFT_1_HI;

/// `vperm` mask: extract [a[1], b[0]] — used for rotate-right-1 hi.
/// Same as ROT_LEFT_1_LO.
const VPERM_ROT_RIGHT_1_HI: [u8; 16] = VPERM_ROT_LEFT_1_LO;

// ─── Load helpers ─────────────────────────────────────────────────────────

/// Load a pair of u64 values from the message array into a vector register.
#[inline(always)]
unsafe fn load_msg_pair(m: &[u64; 16], i0: u8, i1: u8) -> i64x2 {
  i64x2::from_array([m[i0 as usize] as i64, m[i1 as usize] as i64])
}

/// Load 2 consecutive u64 values from a pointer.
#[inline(always)]
unsafe fn vload_u64_pair(p: *const u64) -> i64x2 {
  // SAFETY: caller ensures p is valid for 2 x u64.
  unsafe { core::ptr::read_unaligned(p as *const i64x2) }
}

/// Load a permute mask constant into a vector register.
#[inline(always)]
unsafe fn load_perm_mask(mask: &[u8; 16]) -> i64x2 {
  // SAFETY: mask is a 16-byte array.
  unsafe { core::ptr::read_unaligned(mask.as_ptr() as *const i64x2) }
}

// ─── G function on SIMD register pairs ────────────────────────────────────

/// Blake2b G mixing on SIMD rows (2-wide).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "altivec", enable = "vsx", enable = "power8-vector")]
unsafe fn g2(
  a0: &mut i64x2,
  a1: &mut i64x2,
  b0: &mut i64x2,
  b1: &mut i64x2,
  c0: &mut i64x2,
  c1: &mut i64x2,
  d0: &mut i64x2,
  d1: &mut i64x2,
  mx0: i64x2,
  mx1: i64x2,
  my0: i64x2,
  my1: i64x2,
) {
  // a += b + mx
  *a0 = vaddudm(vaddudm(*a0, *b0), mx0);
  *a1 = vaddudm(vaddudm(*a1, *b1), mx1);
  // d = (d ^ a) >>> 32  (ROL 32)
  *d0 = vrld(vxor(*d0, *a0), ROL_32);
  *d1 = vrld(vxor(*d1, *a1), ROL_32);
  // c += d
  *c0 = vaddudm(*c0, *d0);
  *c1 = vaddudm(*c1, *d1);
  // b = (b ^ c) >>> 24  (ROL 40)
  *b0 = vrld(vxor(*b0, *c0), ROL_40);
  *b1 = vrld(vxor(*b1, *c1), ROL_40);
  // a += b + my
  *a0 = vaddudm(vaddudm(*a0, *b0), my0);
  *a1 = vaddudm(vaddudm(*a1, *b1), my1);
  // d = (d ^ a) >>> 16  (ROL 48)
  *d0 = vrld(vxor(*d0, *a0), ROL_48);
  *d1 = vrld(vxor(*d1, *a1), ROL_48);
  // c += d
  *c0 = vaddudm(*c0, *d0);
  *c1 = vaddudm(*c1, *d1);
  // b = (b ^ c) >>> 63  (ROL 1)
  *b0 = vrld(vxor(*b0, *c0), ROL_1);
  *b1 = vrld(vxor(*b1, *c1), ROL_1);
}

// ─── Diagonalize / Un-diagonalize ─────────────────────────────────────────

/// Diagonalize: rotate row B left by 1, row C by 2 (swap), row D right by 1.
#[inline(always)]
#[target_feature(enable = "altivec", enable = "vsx", enable = "power8-vector")]
unsafe fn diagonalize(b0: &mut i64x2, b1: &mut i64x2, c0: &mut i64x2, c1: &mut i64x2, d0: &mut i64x2, d1: &mut i64x2) {
  // SAFETY: permute masks are 16-byte aligned constants.
  let perm_l1_lo = unsafe { load_perm_mask(&VPERM_ROT_LEFT_1_LO) };
  let perm_l1_hi = unsafe { load_perm_mask(&VPERM_ROT_LEFT_1_HI) };
  let perm_r1_lo = unsafe { load_perm_mask(&VPERM_ROT_RIGHT_1_LO) };
  let perm_r1_hi = unsafe { load_perm_mask(&VPERM_ROT_RIGHT_1_HI) };

  // B: rotate left 1: (v4,v5,v6,v7) -> (v5,v6,v7,v4)
  let tb0 = *b0;
  let tb1 = *b1;
  *b0 = vperm(tb0, tb1, perm_l1_lo); // [b0[1], b1[0]] = [v5, v6]
  *b1 = vperm(tb0, tb1, perm_l1_hi); // [b1[1], b0[0]] = [v7, v4]

  // C: rotate left 2 = swap lo/hi
  core::mem::swap(c0, c1);

  // D: rotate left 3 = rotate right 1: (v12,v13,v14,v15) -> (v15,v12,v13,v14)
  let td0 = *d0;
  let td1 = *d1;
  *d0 = vperm(td0, td1, perm_r1_lo); // [d1[1], d0[0]] = [v15, v12]
  *d1 = vperm(td0, td1, perm_r1_hi); // [d0[1], d1[0]] = [v13, v14]
}

/// Un-diagonalize: reverse the rotations.
#[inline(always)]
#[target_feature(enable = "altivec", enable = "vsx", enable = "power8-vector")]
unsafe fn undiagonalize(
  b0: &mut i64x2,
  b1: &mut i64x2,
  c0: &mut i64x2,
  c1: &mut i64x2,
  d0: &mut i64x2,
  d1: &mut i64x2,
) {
  // SAFETY: permute masks are 16-byte aligned constants.
  let perm_r1_lo = unsafe { load_perm_mask(&VPERM_ROT_RIGHT_1_LO) };
  let perm_r1_hi = unsafe { load_perm_mask(&VPERM_ROT_RIGHT_1_HI) };
  let perm_l1_lo = unsafe { load_perm_mask(&VPERM_ROT_LEFT_1_LO) };
  let perm_l1_hi = unsafe { load_perm_mask(&VPERM_ROT_LEFT_1_HI) };

  // B: rotate right 1 (undo left 1)
  let tb0 = *b0;
  let tb1 = *b1;
  *b0 = vperm(tb0, tb1, perm_r1_lo); // [b1[1], b0[0]] = undo
  *b1 = vperm(tb0, tb1, perm_r1_hi); // [b0[1], b1[0]] = undo

  // C: swap back
  core::mem::swap(c0, c1);

  // D: rotate left 1 (undo right 1)
  let td0 = *d0;
  let td1 = *d1;
  *d0 = vperm(td0, td1, perm_l1_lo); // [d0[1], d1[0]]
  *d1 = vperm(td0, td1, perm_l1_hi); // [d1[1], d0[0]]
}

// ─── Compress entry point ─────────────────────────────────────────────────

/// Blake2b POWER VSX-accelerated compress.
///
/// # Safety
///
/// Caller must ensure POWER8+ with VSX is available.
#[target_feature(enable = "altivec", enable = "vsx", enable = "power8-vector")]
pub(super) unsafe fn compress_vsx(h: &mut [u64; 8], block: &[u8; 128], t: u128, last: bool) {
  let m = load_msg(block);
  let v = init_v(h, t, last);

  // Pack into 2-wide SIMD rows: (lo, hi) for each row
  // SAFETY: v is a [u64; 16] — pointer arithmetic is within bounds.
  let mut a0 = unsafe { vload_u64_pair(v.as_ptr()) }; // v[0], v[1]
  let mut a1 = unsafe { vload_u64_pair(v.as_ptr().add(2)) }; // v[2], v[3]
  let mut b0 = unsafe { vload_u64_pair(v.as_ptr().add(4)) }; // v[4], v[5]
  let mut b1 = unsafe { vload_u64_pair(v.as_ptr().add(6)) }; // v[6], v[7]
  let mut c0 = unsafe { vload_u64_pair(v.as_ptr().add(8)) }; // v[8], v[9]
  let mut c1 = unsafe { vload_u64_pair(v.as_ptr().add(10)) }; // v[10], v[11]
  let mut d0 = unsafe { vload_u64_pair(v.as_ptr().add(12)) }; // v[12], v[13]
  let mut d1 = unsafe { vload_u64_pair(v.as_ptr().add(14)) }; // v[14], v[15]

  // 12 rounds
  for round in 0..12u8 {
    let s = &SIGMA[(round % 10) as usize];

    // Column step
    let mx0 = load_msg_pair(&m, s[0], s[2]);
    let mx1 = load_msg_pair(&m, s[4], s[6]);
    let my0 = load_msg_pair(&m, s[1], s[3]);
    let my1 = load_msg_pair(&m, s[5], s[7]);

    g2(
      &mut a0, &mut a1, &mut b0, &mut b1, &mut c0, &mut c1, &mut d0, &mut d1, mx0, mx1, my0, my1,
    );

    diagonalize(&mut b0, &mut b1, &mut c0, &mut c1, &mut d0, &mut d1);

    // Diagonal step
    let mx0 = load_msg_pair(&m, s[8], s[10]);
    let mx1 = load_msg_pair(&m, s[12], s[14]);
    let my0 = load_msg_pair(&m, s[9], s[11]);
    let my1 = load_msg_pair(&m, s[13], s[15]);

    g2(
      &mut a0, &mut a1, &mut b0, &mut b1, &mut c0, &mut c1, &mut d0, &mut d1, mx0, mx1, my0, my1,
    );

    undiagonalize(&mut b0, &mut b1, &mut c0, &mut c1, &mut d0, &mut d1);
  }

  // Finalize: h[i] ^= v[i] ^ v[i+8]
  // SAFETY: h is a [u64; 8] — pointer arithmetic is within bounds.
  let h0 = unsafe { vload_u64_pair(h.as_ptr()) };
  let h1 = unsafe { vload_u64_pair(h.as_ptr().add(2)) };
  let h2 = unsafe { vload_u64_pair(h.as_ptr().add(4)) };
  let h3 = unsafe { vload_u64_pair(h.as_ptr().add(6)) };

  let r0 = vxor(h0, vxor(a0, c0));
  let r1 = vxor(h1, vxor(a1, c1));
  let r2 = vxor(h2, vxor(b0, d0));
  let r3 = vxor(h3, vxor(b1, d1));

  // SAFETY: h is a [u64; 8] — pointer arithmetic is within bounds.
  unsafe {
    core::ptr::write_unaligned(h.as_mut_ptr() as *mut i64x2, r0);
    core::ptr::write_unaligned(h.as_mut_ptr().add(2) as *mut i64x2, r1);
    core::ptr::write_unaligned(h.as_mut_ptr().add(4) as *mut i64x2, r2);
    core::ptr::write_unaligned(h.as_mut_ptr().add(6) as *mut i64x2, r3);
  }
}
