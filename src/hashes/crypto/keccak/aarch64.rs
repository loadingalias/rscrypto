//! Keccak-f[1600] aarch64 SHA3 Crypto Extension kernels.
//!
//! Uses ARMv8.2-SHA3 instructions for hardware-accelerated permutation:
//! - `EOR3`: 3-input XOR (θ column parity)
//! - `RAX1`: rotate-and-XOR (θ diffusion)
//! - `BCAX`: bit-clear-and-XOR (χ)
//!
//! Two kernel variants:
//! - **1-state scalar**: scalar `u64` state with selective SHA3 CE acceleration for θ and χ. Uses
//!   `EOR3` for 3-input column parity, `RAX1` for rotate-and-XOR diffusion, and `BCAX` for the χ
//!   step — each saving 1–2 instructions vs the scalar equivalent. The ρ+π step uses scalar
//!   `rotate_left` which compiles to a single `ROR` instruction.
//! - **2-state interleaved**: lane 0 = state A, lane 1 = state B. Processes two independent Keccak
//!   states in parallel for ~2× aggregate throughput using full-width NEON SHA3 CE.
//!
//! # Safety
//!
//! All functions require the `sha3` target feature (ARMv8.2-SHA3).

#![allow(unsafe_code)]
#![allow(clippy::inline_always)]

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

// ---------------------------------------------------------------------------
// SHA3 CE inline asm helpers for scalar u64
// ---------------------------------------------------------------------------
//
// These helpers use `vreg` register constraints to move scalar u64 values
// into NEON registers for a single SHA3 CE instruction, then back to scalar.
// When chained (e.g., eor3 → rax1 in θ), LLVM's register allocator can
// keep intermediate values in NEON registers, amortizing the transfer cost.

/// 3-input XOR: `a ^ b ^ c`.
///
/// Uses the ARMv8.2-SHA3 `EOR3` instruction. Only the lower 64 bits of each
/// 128-bit NEON register are meaningful; upper bits are don't-care.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn eor3_u64(a: u64, b: u64, c: u64) -> u64 {
  // SAFETY: SHA3 CE inline asm requires the sha3 target feature, ensured by callers'
  // #[target_feature].
  unsafe {
    let result: u64;
    core::arch::asm!(
      "eor3 {out:v}.16b, {a:v}.16b, {b:v}.16b, {c:v}.16b",
      a = in(vreg) a,
      b = in(vreg) b,
      c = in(vreg) c,
      out = lateout(vreg) result,
      options(pure, nomem, nostack, preserves_flags),
    );
    result
  }
}

/// Rotate-and-XOR: `a ^ ROL(b, 1)`.
///
/// Uses the ARMv8.2-SHA3 `RAX1` instruction on the lower 64-bit element.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn rax1_u64(a: u64, b: u64) -> u64 {
  // SAFETY: SHA3 CE inline asm requires the sha3 target feature, ensured by callers'
  // #[target_feature].
  unsafe {
    let result: u64;
    core::arch::asm!(
      "rax1 {out:v}.2d, {a:v}.2d, {b:v}.2d",
      a = in(vreg) a,
      b = in(vreg) b,
      out = lateout(vreg) result,
      options(pure, nomem, nostack, preserves_flags),
    );
    result
  }
}

/// Bit-clear and XOR: `a ^ (b & ~c)`.
///
/// Uses the ARMv8.2-SHA3 `BCAX` instruction. Implements the Keccak χ step
/// as `b[i] ^ (b[i+2] & ~b[i+1])` when called as `bcax_u64(b_i, b_{i+2}, b_{i+1})`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn bcax_u64(a: u64, b: u64, c: u64) -> u64 {
  // SAFETY: SHA3 CE inline asm requires the sha3 target feature, ensured by callers'
  // #[target_feature].
  unsafe {
    let result: u64;
    core::arch::asm!(
      "bcax {out:v}.16b, {a:v}.16b, {b:v}.16b, {c:v}.16b",
      a = in(vreg) a,
      b = in(vreg) b,
      c = in(vreg) c,
      out = lateout(vreg) result,
      options(pure, nomem, nostack, preserves_flags),
    );
    result
  }
}

// ---------------------------------------------------------------------------
// 1-state scalar round macro with SHA3 CE acceleration
// ---------------------------------------------------------------------------

/// One round of Keccak-f[1600] using scalar u64 state + SHA3 CE helpers.
///
/// - θ parity: `EOR3` (3-input XOR, saves 1 insn per column vs 2× EOR)
/// - θ diffusion: `RAX1` (rotate-and-XOR, saves 1 insn vs ROR+EOR)
/// - θ XOR-back: scalar XOR
/// - ρ+π: scalar `rotate_left` (compiles to single `ROR`)
/// - χ: `BCAX` (bit-clear-and-XOR, saves 1–2 insns vs BIC+EOR)
/// - ι: scalar XOR
#[cfg(target_arch = "aarch64")]
macro_rules! keccakf_sha3_scalar_round {
  ($a0:ident, $a1:ident, $a2:ident, $a3:ident, $a4:ident,
   $a5:ident, $a6:ident, $a7:ident, $a8:ident, $a9:ident,
   $a10:ident, $a11:ident, $a12:ident, $a13:ident, $a14:ident,
   $a15:ident, $a16:ident, $a17:ident, $a18:ident, $a19:ident,
   $a20:ident, $a21:ident, $a22:ident, $a23:ident, $a24:ident,
   $rc:expr) => {{
    // ---- θ: column parity via EOR3 ----
    let c0 = eor3_u64(eor3_u64($a0, $a5, $a10), $a15, $a20);
    let c1 = eor3_u64(eor3_u64($a1, $a6, $a11), $a16, $a21);
    let c2 = eor3_u64(eor3_u64($a2, $a7, $a12), $a17, $a22);
    let c3 = eor3_u64(eor3_u64($a3, $a8, $a13), $a18, $a23);
    let c4 = eor3_u64(eor3_u64($a4, $a9, $a14), $a19, $a24);

    // ---- θ: diffusion via RAX1 ----
    let d0 = rax1_u64(c4, c1);
    let d1 = rax1_u64(c0, c2);
    let d2 = rax1_u64(c1, c3);
    let d3 = rax1_u64(c2, c4);
    let d4 = rax1_u64(c3, c0);

    // ---- θ: XOR-back (scalar) ----
    $a0 ^= d0;
    $a5 ^= d0;
    $a10 ^= d0;
    $a15 ^= d0;
    $a20 ^= d0;

    $a1 ^= d1;
    $a6 ^= d1;
    $a11 ^= d1;
    $a16 ^= d1;
    $a21 ^= d1;

    $a2 ^= d2;
    $a7 ^= d2;
    $a12 ^= d2;
    $a17 ^= d2;
    $a22 ^= d2;

    $a3 ^= d3;
    $a8 ^= d3;
    $a13 ^= d3;
    $a18 ^= d3;
    $a23 ^= d3;

    $a4 ^= d4;
    $a9 ^= d4;
    $a14 ^= d4;
    $a19 ^= d4;
    $a24 ^= d4;

    // ---- ρ + π (scalar rotate_left → single ROR instruction) ----
    let b0 = $a0;
    let b10 = $a1.rotate_left(1);
    let b20 = $a2.rotate_left(62);
    let b5 = $a3.rotate_left(28);
    let b15 = $a4.rotate_left(27);

    let b16 = $a5.rotate_left(36);
    let b1 = $a6.rotate_left(44);
    let b11 = $a7.rotate_left(6);
    let b21 = $a8.rotate_left(55);
    let b6 = $a9.rotate_left(20);

    let b7 = $a10.rotate_left(3);
    let b17 = $a11.rotate_left(10);
    let b2 = $a12.rotate_left(43);
    let b12 = $a13.rotate_left(25);
    let b22 = $a14.rotate_left(39);

    let b23 = $a15.rotate_left(41);
    let b8 = $a16.rotate_left(45);
    let b18 = $a17.rotate_left(15);
    let b3 = $a18.rotate_left(21);
    let b13 = $a19.rotate_left(8);

    let b14 = $a20.rotate_left(18);
    let b24 = $a21.rotate_left(2);
    let b9 = $a22.rotate_left(61);
    let b19 = $a23.rotate_left(56);
    let b4 = $a24.rotate_left(14);

    // ---- χ: BCAX(x, z, y) = x ^ (z & ~y) ----
    $a0 = bcax_u64(b0, b2, b1);
    $a1 = bcax_u64(b1, b3, b2);
    $a2 = bcax_u64(b2, b4, b3);
    $a3 = bcax_u64(b3, b0, b4);
    $a4 = bcax_u64(b4, b1, b0);

    $a5 = bcax_u64(b5, b7, b6);
    $a6 = bcax_u64(b6, b8, b7);
    $a7 = bcax_u64(b7, b9, b8);
    $a8 = bcax_u64(b8, b5, b9);
    $a9 = bcax_u64(b9, b6, b5);

    $a10 = bcax_u64(b10, b12, b11);
    $a11 = bcax_u64(b11, b13, b12);
    $a12 = bcax_u64(b12, b14, b13);
    $a13 = bcax_u64(b13, b10, b14);
    $a14 = bcax_u64(b14, b11, b10);

    $a15 = bcax_u64(b15, b17, b16);
    $a16 = bcax_u64(b16, b18, b17);
    $a17 = bcax_u64(b17, b19, b18);
    $a18 = bcax_u64(b18, b15, b19);
    $a19 = bcax_u64(b19, b16, b15);

    $a20 = bcax_u64(b20, b22, b21);
    $a21 = bcax_u64(b21, b23, b22);
    $a22 = bcax_u64(b22, b24, b23);
    $a23 = bcax_u64(b23, b20, b24);
    $a24 = bcax_u64(b24, b21, b20);

    // ---- ι ----
    $a0 ^= $rc;
  }};
}

// ---------------------------------------------------------------------------
// 1-state scalar kernel with SHA3 CE acceleration
// ---------------------------------------------------------------------------

/// Keccak-f[1600] permutation — single state, scalar u64 + SHA3 CE.
///
/// State lives in scalar `u64` variables (GPRs). SHA3 CE instructions are
/// used for θ (EOR3, RAX1) and χ (BCAX) where they reduce instruction count.
/// The ρ+π step uses scalar `rotate_left` which compiles to a single `ROR`.
///
/// # Safety
///
/// Caller must ensure `sha3` target feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "sha3")]
#[inline]
unsafe fn keccakf_sha3_impl(state: &mut [u64; 25]) {
  // SAFETY: SHA3 CE helpers (eor3_u64, rax1_u64, bcax_u64) called via the
  // macro require the sha3 target feature, ensured by this function's
  // #[target_feature(enable = "sha3")] attribute.
  unsafe {
    let mut a0 = state[0];
    let mut a1 = state[1];
    let mut a2 = state[2];
    let mut a3 = state[3];
    let mut a4 = state[4];
    let mut a5 = state[5];
    let mut a6 = state[6];
    let mut a7 = state[7];
    let mut a8 = state[8];
    let mut a9 = state[9];
    let mut a10 = state[10];
    let mut a11 = state[11];
    let mut a12 = state[12];
    let mut a13 = state[13];
    let mut a14 = state[14];
    let mut a15 = state[15];
    let mut a16 = state[16];
    let mut a17 = state[17];
    let mut a18 = state[18];
    let mut a19 = state[19];
    let mut a20 = state[20];
    let mut a21 = state[21];
    let mut a22 = state[22];
    let mut a23 = state[23];
    let mut a24 = state[24];

    for &rc in &super::RC {
      keccakf_sha3_scalar_round!(
        a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23,
        a24, rc
      );
    }

    state[0] = a0;
    state[1] = a1;
    state[2] = a2;
    state[3] = a3;
    state[4] = a4;
    state[5] = a5;
    state[6] = a6;
    state[7] = a7;
    state[8] = a8;
    state[9] = a9;
    state[10] = a10;
    state[11] = a11;
    state[12] = a12;
    state[13] = a13;
    state[14] = a14;
    state[15] = a15;
    state[16] = a16;
    state[17] = a17;
    state[18] = a18;
    state[19] = a19;
    state[20] = a20;
    state[21] = a21;
    state[22] = a22;
    state[23] = a23;
    state[24] = a24;
  } // unsafe
}

/// Keccak-f[1600] permutation using ARMv8.2 SHA3 Crypto Extensions.
///
/// Requires `aarch64::SHA3` capability (verified by dispatch before selection).
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) fn keccakf_aarch64_sha3(state: &mut [u64; 25]) {
  // SAFETY: Dispatch verifies aarch64::SHA3 capability before selecting this kernel.
  unsafe { keccakf_sha3_impl(state) }
}

// ---------------------------------------------------------------------------
// 2-state interleaved kernel (lane 0 = state A, lane 1 = state B)
// ---------------------------------------------------------------------------

/// One round of Keccak-f[1600] using full-width SHA3 CE on uint64x2_t.
///
/// All instructions (EOR3, RAX1, XAR, BCAX, EOR) are lane-wise on uint64x2_t,
/// processing both lanes simultaneously for the 2-state interleaved kernel.
#[cfg(target_arch = "aarch64")]
macro_rules! keccakf_sha3_x2_round {
  ($a0:ident, $a1:ident, $a2:ident, $a3:ident, $a4:ident,
   $a5:ident, $a6:ident, $a7:ident, $a8:ident, $a9:ident,
   $a10:ident, $a11:ident, $a12:ident, $a13:ident, $a14:ident,
   $a15:ident, $a16:ident, $a17:ident, $a18:ident, $a19:ident,
   $a20:ident, $a21:ident, $a22:ident, $a23:ident, $a24:ident,
   $rc:expr) => {{
    // ---- θ: column parity ----
    let c0 = veor3q_u64(veor3q_u64($a0, $a5, $a10), $a15, $a20);
    let c1 = veor3q_u64(veor3q_u64($a1, $a6, $a11), $a16, $a21);
    let c2 = veor3q_u64(veor3q_u64($a2, $a7, $a12), $a17, $a22);
    let c3 = veor3q_u64(veor3q_u64($a3, $a8, $a13), $a18, $a23);
    let c4 = veor3q_u64(veor3q_u64($a4, $a9, $a14), $a19, $a24);

    // ---- θ: diffusion ----
    let d0 = vrax1q_u64(c4, c1);
    let d1 = vrax1q_u64(c0, c2);
    let d2 = vrax1q_u64(c1, c3);
    let d3 = vrax1q_u64(c2, c4);
    let d4 = vrax1q_u64(c3, c0);

    // ---- θ XOR-back + ρ + π (fused via XAR) ----
    // XAR(a, d, imm) = ROR(a ^ d, imm) = ROL(a ^ d, 64-imm)
    // imm = (64 - rho_rotation) % 64

    // Column 0
    let b0 = vxarq_u64::<0>($a0, d0);
    let b16 = vxarq_u64::<28>($a5, d0);
    let b7 = vxarq_u64::<61>($a10, d0);
    let b23 = vxarq_u64::<23>($a15, d0);
    let b14 = vxarq_u64::<46>($a20, d0);

    // Column 1
    let b10 = vxarq_u64::<63>($a1, d1);
    let b1 = vxarq_u64::<20>($a6, d1);
    let b17 = vxarq_u64::<54>($a11, d1);
    let b8 = vxarq_u64::<19>($a16, d1);
    let b24 = vxarq_u64::<62>($a21, d1);

    // Column 2
    let b20 = vxarq_u64::<2>($a2, d2);
    let b11 = vxarq_u64::<58>($a7, d2);
    let b2 = vxarq_u64::<21>($a12, d2);
    let b18 = vxarq_u64::<49>($a17, d2);
    let b9 = vxarq_u64::<3>($a22, d2);

    // Column 3
    let b5 = vxarq_u64::<36>($a3, d3);
    let b21 = vxarq_u64::<9>($a8, d3);
    let b12 = vxarq_u64::<39>($a13, d3);
    let b3 = vxarq_u64::<43>($a18, d3);
    let b19 = vxarq_u64::<8>($a23, d3);

    // Column 4
    let b15 = vxarq_u64::<37>($a4, d4);
    let b6 = vxarq_u64::<44>($a9, d4);
    let b22 = vxarq_u64::<25>($a14, d4);
    let b13 = vxarq_u64::<56>($a19, d4);
    let b4 = vxarq_u64::<50>($a24, d4);

    // ---- χ: BCAX(x, z, y) = x ^ (z & ~y) ----
    $a0 = vbcaxq_u64(b0, b2, b1);
    $a1 = vbcaxq_u64(b1, b3, b2);
    $a2 = vbcaxq_u64(b2, b4, b3);
    $a3 = vbcaxq_u64(b3, b0, b4);
    $a4 = vbcaxq_u64(b4, b1, b0);

    $a5 = vbcaxq_u64(b5, b7, b6);
    $a6 = vbcaxq_u64(b6, b8, b7);
    $a7 = vbcaxq_u64(b7, b9, b8);
    $a8 = vbcaxq_u64(b8, b5, b9);
    $a9 = vbcaxq_u64(b9, b6, b5);

    $a10 = vbcaxq_u64(b10, b12, b11);
    $a11 = vbcaxq_u64(b11, b13, b12);
    $a12 = vbcaxq_u64(b12, b14, b13);
    $a13 = vbcaxq_u64(b13, b10, b14);
    $a14 = vbcaxq_u64(b14, b11, b10);

    $a15 = vbcaxq_u64(b15, b17, b16);
    $a16 = vbcaxq_u64(b16, b18, b17);
    $a17 = vbcaxq_u64(b17, b19, b18);
    $a18 = vbcaxq_u64(b18, b15, b19);
    $a19 = vbcaxq_u64(b19, b16, b15);

    $a20 = vbcaxq_u64(b20, b22, b21);
    $a21 = vbcaxq_u64(b21, b23, b22);
    $a22 = vbcaxq_u64(b22, b24, b23);
    $a23 = vbcaxq_u64(b23, b20, b24);
    $a24 = vbcaxq_u64(b24, b21, b20);

    // ---- ι ----
    $a0 = veorq_u64($a0, vdupq_n_u64($rc));
  }};
}

/// Combine lane 0 from `state_a[i]` and lane 1 from `state_b[i]` into one
/// uint64x2_t register.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn combine_lanes(a: u64, b: u64) -> uint64x2_t {
  // SAFETY: NEON intrinsics are available on all aarch64 targets.
  unsafe { vcombine_u64(vcreate_u64(a), vcreate_u64(b)) }
}

/// Keccak-f[1600] permutation — two independent states in parallel.
///
/// Lane 0 of each NEON register holds state_a, lane 1 holds state_b.
/// All SHA3 CE instructions are lane-wise, so both states are permuted
/// simultaneously with zero additional round instructions.
///
/// # Safety
///
/// Caller must ensure `sha3` target feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "sha3")]
unsafe fn keccakf_sha3_x2_impl(state_a: &mut [u64; 25], state_b: &mut [u64; 25]) {
  // SAFETY: NEON + SHA3 CE intrinsics (combine_lanes, veor3q_u64, vrax1q_u64,
  // vxarq_u64, vbcaxq_u64, vgetq_lane_u64, etc.) are available via this
  // function's #[target_feature(enable = "sha3")] attribute.
  unsafe {
    // Load: lane 0 = state_a, lane 1 = state_b
    let mut a0 = combine_lanes(state_a[0], state_b[0]);
    let mut a1 = combine_lanes(state_a[1], state_b[1]);
    let mut a2 = combine_lanes(state_a[2], state_b[2]);
    let mut a3 = combine_lanes(state_a[3], state_b[3]);
    let mut a4 = combine_lanes(state_a[4], state_b[4]);
    let mut a5 = combine_lanes(state_a[5], state_b[5]);
    let mut a6 = combine_lanes(state_a[6], state_b[6]);
    let mut a7 = combine_lanes(state_a[7], state_b[7]);
    let mut a8 = combine_lanes(state_a[8], state_b[8]);
    let mut a9 = combine_lanes(state_a[9], state_b[9]);
    let mut a10 = combine_lanes(state_a[10], state_b[10]);
    let mut a11 = combine_lanes(state_a[11], state_b[11]);
    let mut a12 = combine_lanes(state_a[12], state_b[12]);
    let mut a13 = combine_lanes(state_a[13], state_b[13]);
    let mut a14 = combine_lanes(state_a[14], state_b[14]);
    let mut a15 = combine_lanes(state_a[15], state_b[15]);
    let mut a16 = combine_lanes(state_a[16], state_b[16]);
    let mut a17 = combine_lanes(state_a[17], state_b[17]);
    let mut a18 = combine_lanes(state_a[18], state_b[18]);
    let mut a19 = combine_lanes(state_a[19], state_b[19]);
    let mut a20 = combine_lanes(state_a[20], state_b[20]);
    let mut a21 = combine_lanes(state_a[21], state_b[21]);
    let mut a22 = combine_lanes(state_a[22], state_b[22]);
    let mut a23 = combine_lanes(state_a[23], state_b[23]);
    let mut a24 = combine_lanes(state_a[24], state_b[24]);

    for &rc in &super::RC {
      keccakf_sha3_x2_round!(
        a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23,
        a24, rc
      );
    }

    // Store: extract lane 0 → state_a, lane 1 → state_b
    state_a[0] = vgetq_lane_u64(a0, 0);
    state_b[0] = vgetq_lane_u64(a0, 1);
    state_a[1] = vgetq_lane_u64(a1, 0);
    state_b[1] = vgetq_lane_u64(a1, 1);
    state_a[2] = vgetq_lane_u64(a2, 0);
    state_b[2] = vgetq_lane_u64(a2, 1);
    state_a[3] = vgetq_lane_u64(a3, 0);
    state_b[3] = vgetq_lane_u64(a3, 1);
    state_a[4] = vgetq_lane_u64(a4, 0);
    state_b[4] = vgetq_lane_u64(a4, 1);
    state_a[5] = vgetq_lane_u64(a5, 0);
    state_b[5] = vgetq_lane_u64(a5, 1);
    state_a[6] = vgetq_lane_u64(a6, 0);
    state_b[6] = vgetq_lane_u64(a6, 1);
    state_a[7] = vgetq_lane_u64(a7, 0);
    state_b[7] = vgetq_lane_u64(a7, 1);
    state_a[8] = vgetq_lane_u64(a8, 0);
    state_b[8] = vgetq_lane_u64(a8, 1);
    state_a[9] = vgetq_lane_u64(a9, 0);
    state_b[9] = vgetq_lane_u64(a9, 1);
    state_a[10] = vgetq_lane_u64(a10, 0);
    state_b[10] = vgetq_lane_u64(a10, 1);
    state_a[11] = vgetq_lane_u64(a11, 0);
    state_b[11] = vgetq_lane_u64(a11, 1);
    state_a[12] = vgetq_lane_u64(a12, 0);
    state_b[12] = vgetq_lane_u64(a12, 1);
    state_a[13] = vgetq_lane_u64(a13, 0);
    state_b[13] = vgetq_lane_u64(a13, 1);
    state_a[14] = vgetq_lane_u64(a14, 0);
    state_b[14] = vgetq_lane_u64(a14, 1);
    state_a[15] = vgetq_lane_u64(a15, 0);
    state_b[15] = vgetq_lane_u64(a15, 1);
    state_a[16] = vgetq_lane_u64(a16, 0);
    state_b[16] = vgetq_lane_u64(a16, 1);
    state_a[17] = vgetq_lane_u64(a17, 0);
    state_b[17] = vgetq_lane_u64(a17, 1);
    state_a[18] = vgetq_lane_u64(a18, 0);
    state_b[18] = vgetq_lane_u64(a18, 1);
    state_a[19] = vgetq_lane_u64(a19, 0);
    state_b[19] = vgetq_lane_u64(a19, 1);
    state_a[20] = vgetq_lane_u64(a20, 0);
    state_b[20] = vgetq_lane_u64(a20, 1);
    state_a[21] = vgetq_lane_u64(a21, 0);
    state_b[21] = vgetq_lane_u64(a21, 1);
    state_a[22] = vgetq_lane_u64(a22, 0);
    state_b[22] = vgetq_lane_u64(a22, 1);
    state_a[23] = vgetq_lane_u64(a23, 0);
    state_b[23] = vgetq_lane_u64(a23, 1);
    state_a[24] = vgetq_lane_u64(a24, 0);
    state_b[24] = vgetq_lane_u64(a24, 1);
  } // unsafe
}

/// Permute two independent Keccak-f[1600] states in parallel using 2-state
/// NEON interleaving via ARMv8.2 SHA3 Crypto Extensions.
///
/// Both states are permuted independently for 24 rounds. The only overhead
/// vs. a single permutation is the load/store interleave (~100 instructions).
///
/// Requires `aarch64::SHA3` capability (verified by dispatch before calling).
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) fn keccakf_aarch64_sha3_x2(state_a: &mut [u64; 25], state_b: &mut [u64; 25]) {
  // SAFETY: Dispatch verifies aarch64::SHA3 capability before calling.
  unsafe { keccakf_sha3_x2_impl(state_a, state_b) }
}
