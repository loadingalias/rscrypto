//! Keccak-f[1600] aarch64 SHA3 Crypto Extension kernel.
//!
//! Uses ARMv8.2-SHA3 instructions for hardware-accelerated permutation:
//! - `EOR3`: 3-input XOR (θ column parity)
//! - `RAX1`: rotate-and-XOR (θ diffusion)
//! - `XAR`: XOR-and-rotate (fused θ+ρ+π)
//! - `BCAX`: bit-clear-and-XOR (χ)
//!
//! Expected ~2-3× throughput vs. portable scalar on Apple Silicon / Graviton3+.
//!
//! # Safety
//!
//! All functions require the `sha3` target feature (ARMv8.2-SHA3).

#![allow(unsafe_code)]
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::inline_always)]
#![allow(clippy::undocumented_unsafe_blocks)]

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

/// Keccak-f[1600] permutation using ARMv8.2 SHA3 Crypto Extension.
///
/// State lives in 25 NEON registers (one u64 lane per register, duplicated).
/// Each round: θ via EOR3+RAX1, ρ+π via XAR, χ via BCAX, ι via EOR.
///
/// # Safety
///
/// Caller must ensure `sha3` target feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "sha3")]
#[inline]
unsafe fn keccakf_sha3_impl(state: &mut [u64; 25]) {
  // Load 25-lane state into NEON registers.
  // Each lane is duplicated across both u64 positions; only lane 0 is meaningful.
  let mut a0 = vdupq_n_u64(state[0]);
  let mut a1 = vdupq_n_u64(state[1]);
  let mut a2 = vdupq_n_u64(state[2]);
  let mut a3 = vdupq_n_u64(state[3]);
  let mut a4 = vdupq_n_u64(state[4]);
  let mut a5 = vdupq_n_u64(state[5]);
  let mut a6 = vdupq_n_u64(state[6]);
  let mut a7 = vdupq_n_u64(state[7]);
  let mut a8 = vdupq_n_u64(state[8]);
  let mut a9 = vdupq_n_u64(state[9]);
  let mut a10 = vdupq_n_u64(state[10]);
  let mut a11 = vdupq_n_u64(state[11]);
  let mut a12 = vdupq_n_u64(state[12]);
  let mut a13 = vdupq_n_u64(state[13]);
  let mut a14 = vdupq_n_u64(state[14]);
  let mut a15 = vdupq_n_u64(state[15]);
  let mut a16 = vdupq_n_u64(state[16]);
  let mut a17 = vdupq_n_u64(state[17]);
  let mut a18 = vdupq_n_u64(state[18]);
  let mut a19 = vdupq_n_u64(state[19]);
  let mut a20 = vdupq_n_u64(state[20]);
  let mut a21 = vdupq_n_u64(state[21]);
  let mut a22 = vdupq_n_u64(state[22]);
  let mut a23 = vdupq_n_u64(state[23]);
  let mut a24 = vdupq_n_u64(state[24]);

  for &rc in &super::RC {
    // ---- θ: column parity ----
    // C[x] = A[x,0] ^ A[x,1] ^ A[x,2] ^ A[x,3] ^ A[x,4]
    // Two EOR3 per column: EOR3(EOR3(a,b,c), d, e)
    let c0 = veor3q_u64(veor3q_u64(a0, a5, a10), a15, a20);
    let c1 = veor3q_u64(veor3q_u64(a1, a6, a11), a16, a21);
    let c2 = veor3q_u64(veor3q_u64(a2, a7, a12), a17, a22);
    let c3 = veor3q_u64(veor3q_u64(a3, a8, a13), a18, a23);
    let c4 = veor3q_u64(veor3q_u64(a4, a9, a14), a19, a24);

    // ---- θ: diffusion ----
    // D[x] = C[x-1] ^ ROL(C[x+1], 1)
    // RAX1(a, b) = a ^ ROL(b, 1)
    let d0 = vrax1q_u64(c4, c1);
    let d1 = vrax1q_u64(c0, c2);
    let d2 = vrax1q_u64(c1, c3);
    let d3 = vrax1q_u64(c2, c4);
    let d4 = vrax1q_u64(c3, c0);

    // ---- θ XOR-back + ρ + π (fused via XAR) ----
    // XAR(a, d, imm) = ROR(a ^ d, imm) = ROL(a ^ d, 64-imm)
    // imm = (64 - rho_rotation) % 64
    //
    // Reference table: src → (rho, π dest, XAR imm)
    //  a0→( 0, b0,  0)  a5→(36,b16,28)  a10→( 3, b7,61)  a15→(41,b23,23)  a20→(18,b14,46)
    //  a1→( 1,b10,63)  a6→(44, b1,20)  a11→(10,b17,54)  a16→(45, b8,19)  a21→( 2,b24,62)
    //  a2→(62,b20, 2)  a7→( 6,b11,58)  a12→(43, b2,21)  a17→(15,b18,49)  a22→(61, b9, 3)
    //  a3→(28, b5,36)  a8→(55,b21, 9)  a13→(25,b12,39)  a18→(21, b3,43)  a23→(56,b19, 8)
    //  a4→(27,b15,37)  a9→(20, b6,44)  a14→(39,b22,25)  a19→( 8,b13,56)  a24→(14, b4,50)

    // Column 0: a0, a5, a10, a15, a20 use d0
    let b0 = vxarq_u64::<0>(a0, d0); // ROL 0
    let b16 = vxarq_u64::<28>(a5, d0); // ROL 36
    let b7 = vxarq_u64::<61>(a10, d0); // ROL 3
    let b23 = vxarq_u64::<23>(a15, d0); // ROL 41
    let b14 = vxarq_u64::<46>(a20, d0); // ROL 18

    // Column 1: a1, a6, a11, a16, a21 use d1
    let b10 = vxarq_u64::<63>(a1, d1); // ROL 1
    let b1 = vxarq_u64::<20>(a6, d1); // ROL 44
    let b17 = vxarq_u64::<54>(a11, d1); // ROL 10
    let b8 = vxarq_u64::<19>(a16, d1); // ROL 45
    let b24 = vxarq_u64::<62>(a21, d1); // ROL 2

    // Column 2: a2, a7, a12, a17, a22 use d2
    let b20 = vxarq_u64::<2>(a2, d2); // ROL 62
    let b11 = vxarq_u64::<58>(a7, d2); // ROL 6
    let b2 = vxarq_u64::<21>(a12, d2); // ROL 43
    let b18 = vxarq_u64::<49>(a17, d2); // ROL 15
    let b9 = vxarq_u64::<3>(a22, d2); // ROL 61

    // Column 3: a3, a8, a13, a18, a23 use d3
    let b5 = vxarq_u64::<36>(a3, d3); // ROL 28
    let b21 = vxarq_u64::<9>(a8, d3); // ROL 55
    let b12 = vxarq_u64::<39>(a13, d3); // ROL 25
    let b3 = vxarq_u64::<43>(a18, d3); // ROL 21
    let b19 = vxarq_u64::<8>(a23, d3); // ROL 56

    // Column 4: a4, a9, a14, a19, a24 use d4
    let b15 = vxarq_u64::<37>(a4, d4); // ROL 27
    let b6 = vxarq_u64::<44>(a9, d4); // ROL 20
    let b22 = vxarq_u64::<25>(a14, d4); // ROL 39
    let b13 = vxarq_u64::<56>(a19, d4); // ROL 8
    let b4 = vxarq_u64::<50>(a24, d4); // ROL 14

    // ---- χ ----
    // BCAX(x, z, y) = x ^ (z & ~y)
    // χ formula: a[i] = b[i] ^ (~b[i+1] & b[i+2])
    //          = BCAX(b[i], b[i+2], b[i+1])

    // Row 0
    a0 = vbcaxq_u64(b0, b2, b1);
    a1 = vbcaxq_u64(b1, b3, b2);
    a2 = vbcaxq_u64(b2, b4, b3);
    a3 = vbcaxq_u64(b3, b0, b4);
    a4 = vbcaxq_u64(b4, b1, b0);

    // Row 1
    a5 = vbcaxq_u64(b5, b7, b6);
    a6 = vbcaxq_u64(b6, b8, b7);
    a7 = vbcaxq_u64(b7, b9, b8);
    a8 = vbcaxq_u64(b8, b5, b9);
    a9 = vbcaxq_u64(b9, b6, b5);

    // Row 2
    a10 = vbcaxq_u64(b10, b12, b11);
    a11 = vbcaxq_u64(b11, b13, b12);
    a12 = vbcaxq_u64(b12, b14, b13);
    a13 = vbcaxq_u64(b13, b10, b14);
    a14 = vbcaxq_u64(b14, b11, b10);

    // Row 3
    a15 = vbcaxq_u64(b15, b17, b16);
    a16 = vbcaxq_u64(b16, b18, b17);
    a17 = vbcaxq_u64(b17, b19, b18);
    a18 = vbcaxq_u64(b18, b15, b19);
    a19 = vbcaxq_u64(b19, b16, b15);

    // Row 4
    a20 = vbcaxq_u64(b20, b22, b21);
    a21 = vbcaxq_u64(b21, b23, b22);
    a22 = vbcaxq_u64(b22, b24, b23);
    a23 = vbcaxq_u64(b23, b20, b24);
    a24 = vbcaxq_u64(b24, b21, b20);

    // ---- ι ----
    a0 = veorq_u64(a0, vdupq_n_u64(rc));
  }

  // Store state back (extract lane 0 from each NEON register).
  state[0] = vgetq_lane_u64(a0, 0);
  state[1] = vgetq_lane_u64(a1, 0);
  state[2] = vgetq_lane_u64(a2, 0);
  state[3] = vgetq_lane_u64(a3, 0);
  state[4] = vgetq_lane_u64(a4, 0);
  state[5] = vgetq_lane_u64(a5, 0);
  state[6] = vgetq_lane_u64(a6, 0);
  state[7] = vgetq_lane_u64(a7, 0);
  state[8] = vgetq_lane_u64(a8, 0);
  state[9] = vgetq_lane_u64(a9, 0);
  state[10] = vgetq_lane_u64(a10, 0);
  state[11] = vgetq_lane_u64(a11, 0);
  state[12] = vgetq_lane_u64(a12, 0);
  state[13] = vgetq_lane_u64(a13, 0);
  state[14] = vgetq_lane_u64(a14, 0);
  state[15] = vgetq_lane_u64(a15, 0);
  state[16] = vgetq_lane_u64(a16, 0);
  state[17] = vgetq_lane_u64(a17, 0);
  state[18] = vgetq_lane_u64(a18, 0);
  state[19] = vgetq_lane_u64(a19, 0);
  state[20] = vgetq_lane_u64(a20, 0);
  state[21] = vgetq_lane_u64(a21, 0);
  state[22] = vgetq_lane_u64(a22, 0);
  state[23] = vgetq_lane_u64(a23, 0);
  state[24] = vgetq_lane_u64(a24, 0);
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
