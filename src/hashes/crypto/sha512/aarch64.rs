//! aarch64 FEAT_SHA512 kernel for SHA-512.
//!
//! Uses the SHA-512 Crypto Extension instructions (grouped under `sha3` in Rust):
//! `vsha512hq_u64`, `vsha512h2q_u64`, `vsha512su0q_u64`, `vsha512su1q_u64`.
//!
//! Available on Apple Silicon (M1+), Graviton2+, Ampere Altra, Cortex-A76+.

#![allow(unsafe_code)]
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::inline_always)]
#![allow(clippy::undocumented_unsafe_blocks)]
#![allow(clippy::indexing_slicing)]

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

/// K constants packed as 40 pairs of `[K[2i], K[2i+1]]` for `vld1q_u64` loads.
///
/// Cache-line aligned (64 bytes) so the 640-byte table starts at an L1 boundary.
#[repr(C, align(64))]
struct AlignedKPairs([[u64; 2]; 40]);

impl core::ops::Deref for AlignedKPairs {
  type Target = [[u64; 2]; 40];
  #[inline(always)]
  fn deref(&self) -> &[[u64; 2]; 40] {
    &self.0
  }
}

#[cfg(target_arch = "aarch64")]
static K_PAIRS: AlignedKPairs = AlignedKPairs([
  [0x428a_2f98_d728_ae22, 0x7137_4491_23ef_65cd],
  [0xb5c0_fbcf_ec4d_3b2f, 0xe9b5_dba5_8189_dbbc],
  [0x3956_c25b_f348_b538, 0x59f1_11f1_b605_d019],
  [0x923f_82a4_af19_4f9b, 0xab1c_5ed5_da6d_8118],
  [0xd807_aa98_a303_0242, 0x1283_5b01_4570_6fbe],
  [0x2431_85be_4ee4_b28c, 0x550c_7dc3_d5ff_b4e2],
  [0x72be_5d74_f27b_896f, 0x80de_b1fe_3b16_96b1],
  [0x9bdc_06a7_25c7_1235, 0xc19b_f174_cf69_2694],
  [0xe49b_69c1_9ef1_4ad2, 0xefbe_4786_384f_25e3],
  [0x0fc1_9dc6_8b8c_d5b5, 0x240c_a1cc_77ac_9c65],
  [0x2de9_2c6f_592b_0275, 0x4a74_84aa_6ea6_e483],
  [0x5cb0_a9dc_bd41_fbd4, 0x76f9_88da_8311_53b5],
  [0x983e_5152_ee66_dfab, 0xa831_c66d_2db4_3210],
  [0xb003_27c8_98fb_213f, 0xbf59_7fc7_beef_0ee4],
  [0xc6e0_0bf3_3da8_8fc2, 0xd5a7_9147_930a_a725],
  [0x06ca_6351_e003_826f, 0x1429_2967_0a0e_6e70],
  [0x27b7_0a85_46d2_2ffc, 0x2e1b_2138_5c26_c926],
  [0x4d2c_6dfc_5ac4_2aed, 0x5338_0d13_9d95_b3df],
  [0x650a_7354_8baf_63de, 0x766a_0abb_3c77_b2a8],
  [0x81c2_c92e_47ed_aee6, 0x9272_2c85_1482_353b],
  [0xa2bf_e8a1_4cf1_0364, 0xa81a_664b_bc42_3001],
  [0xc24b_8b70_d0f8_9791, 0xc76c_51a3_0654_be30],
  [0xd192_e819_d6ef_5218, 0xd699_0624_5565_a910],
  [0xf40e_3585_5771_202a, 0x106a_a070_32bb_d1b8],
  [0x19a4_c116_b8d2_d0c8, 0x1e37_6c08_5141_ab53],
  [0x2748_774c_df8e_eb99, 0x34b0_bcb5_e19b_48a8],
  [0x391c_0cb3_c5c9_5a63, 0x4ed8_aa4a_e341_8acb],
  [0x5b9c_ca4f_7763_e373, 0x682e_6ff3_d6b2_b8a3],
  [0x748f_82ee_5def_b2fc, 0x78a5_636f_4317_2f60],
  [0x84c8_7814_a1f0_ab72, 0x8cc7_0208_1a64_39ec],
  [0x90be_fffa_2363_1e28, 0xa450_6ceb_de82_bde9],
  [0xbef9_a3f7_b2c6_7915, 0xc671_78f2_e372_532b],
  [0xca27_3ece_ea26_619c, 0xd186_b8c7_21c0_c207],
  [0xeada_7dd6_cde0_eb1e, 0xf57d_4f7f_ee6e_d178],
  [0x06f0_67aa_7217_6fba, 0x0a63_7dc5_a2c8_98a6],
  [0x113f_9804_bef9_0dae, 0x1b71_0b35_131c_471b],
  [0x28db_77f5_2304_7d84, 0x32ca_ab7b_40c7_2493],
  [0x3c9e_be0a_15c9_bebc, 0x431d_67c4_9c10_0d4c],
  [0x4cc5_d4be_cb3e_42b6, 0x597f_299c_fc65_7e2a],
  [0x5fcb_6fab_3ad6_faec, 0x6c44_198c_4a47_5817],
]);

/// Compress one or more 128-byte blocks using FEAT_SHA512 instructions.
///
/// # Safety
///
/// Caller must verify that `aarch64::SHA512` capability is available.
/// `blocks.len()` must be a multiple of 128.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "sha3")]
pub(crate) unsafe fn compress_blocks_aarch64_sha512(state: &mut [u64; 8], blocks: &[u8]) {
  debug_assert_eq!(blocks.len() % 128, 0);
  if blocks.is_empty() {
    return;
  }

  // State: s0=(a,b), s1=(c,d), s2=(e,f), s3=(g,h), s4=scratch.
  let mut s0 = vld1q_u64(state.as_ptr());
  let mut s1 = vld1q_u64(state.as_ptr().add(2));
  let mut s2 = vld1q_u64(state.as_ptr().add(4));
  let mut s3 = vld1q_u64(state.as_ptr().add(6));
  let mut s4: uint64x2_t;

  let mut ptr = blocks.as_ptr();
  let end = ptr.add(blocks.len());

  while ptr < end {
    let s0_save = s0;
    let s1_save = s1;
    let s2_save = s2;
    let s3_save = s3;

    // Load and byte-swap 8 message pairs (128 bytes).
    let mut w0 = vreinterpretq_u64_u8(vrev64q_u8(vld1q_u8(ptr)));
    let mut w1 = vreinterpretq_u64_u8(vrev64q_u8(vld1q_u8(ptr.add(16))));
    let mut w2 = vreinterpretq_u64_u8(vrev64q_u8(vld1q_u8(ptr.add(32))));
    let mut w3 = vreinterpretq_u64_u8(vrev64q_u8(vld1q_u8(ptr.add(48))));
    let mut w4 = vreinterpretq_u64_u8(vrev64q_u8(vld1q_u8(ptr.add(64))));
    let mut w5 = vreinterpretq_u64_u8(vrev64q_u8(vld1q_u8(ptr.add(80))));
    let mut w6 = vreinterpretq_u64_u8(vrev64q_u8(vld1q_u8(ptr.add(96))));
    let mut w7 = vreinterpretq_u64_u8(vrev64q_u8(vld1q_u8(ptr.add(112))));

    let k = |i: usize| -> uint64x2_t { vld1q_u64(K_PAIRS[i].as_ptr()) };

    // Two-round step using 5-register rotation (Linux kernel pattern).
    //
    // $i0=ab, $i1=cd, $i2=ef, $i3=gh (input), $i4=scratch (output).
    // After: $i3 = new ab (SHA512H + SHA512H2), $i4 = new ef (cd + T1).
    // Register roles rotate: the caller permutes arguments for the next step.
    macro_rules! dround {
      ($i0:ident, $i1:ident, $i2:ident, $i3:ident, $i4:ident, $ki:expr, $wi:expr) => {{
        let kw = vaddq_u64(k($ki), $wi);
        let fg = vextq_u64($i2, $i3, 1);
        let de = vextq_u64($i1, $i2, 1);
        let kw_swap = vextq_u64(kw, kw, 1);
        let t = vaddq_u64($i3, kw_swap);
        let t = vsha512hq_u64(t, fg, de);
        $i4 = vaddq_u64($i1, t);
        $i3 = vsha512h2q_u64(t, $i1, $i0);
      }};
    }

    // Message schedule update: W[t] = sigma1(W[t-2]) + W[t-7] + sigma0(W[t-15]) + W[t-16].
    macro_rules! sched {
      ($wn:ident, $wn1:expr, $wn4:expr, $wn5:expr, $wn7:expr) => {{
        $wn = vsha512su1q_u64(vsha512su0q_u64($wn, $wn1), $wn7, vextq_u64($wn4, $wn5, 1));
      }};
    }

    // 40 drounds (80 rounds). 5-register rotation cycle has period 5,
    // and 40 = 8 × 5, so state returns to s0=ab after all drounds.
    //
    // Rotation pattern per cycle of 5 drounds:
    //   pos 0: dround!(s0, s1, s2, s3, s4)
    //   pos 1: dround!(s3, s0, s4, s2, s1)
    //   pos 2: dround!(s2, s3, s1, s4, s0)
    //   pos 3: dround!(s4, s2, s0, s1, s3)
    //   pos 4: dround!(s1, s4, s3, s0, s2)  → back to s0=ab

    // ---- Rounds 0-15 (drounds 0-7, no schedule update) ----
    dround!(s0, s1, s2, s3, s4, 0, w0); // pos 0
    dround!(s3, s0, s4, s2, s1, 1, w1); // pos 1
    dround!(s2, s3, s1, s4, s0, 2, w2); // pos 2
    dround!(s4, s2, s0, s1, s3, 3, w3); // pos 3
    dround!(s1, s4, s3, s0, s2, 4, w4); // pos 4 [original]
    dround!(s0, s1, s2, s3, s4, 5, w5); // pos 0
    dround!(s3, s0, s4, s2, s1, 6, w6); // pos 1
    dround!(s2, s3, s1, s4, s0, 7, w7); // pos 2

    // ---- Rounds 16-79 (drounds 8-39, with schedule update) ----
    sched!(w0, w1, w4, w5, w7);
    dround!(s4, s2, s0, s1, s3, 8, w0); // pos 3
    sched!(w1, w2, w5, w6, w0);
    dround!(s1, s4, s3, s0, s2, 9, w1); // pos 4 [original]

    sched!(w2, w3, w6, w7, w1);
    dround!(s0, s1, s2, s3, s4, 10, w2); // pos 0
    sched!(w3, w4, w7, w0, w2);
    dround!(s3, s0, s4, s2, s1, 11, w3); // pos 1
    sched!(w4, w5, w0, w1, w3);
    dround!(s2, s3, s1, s4, s0, 12, w4); // pos 2
    sched!(w5, w6, w1, w2, w4);
    dround!(s4, s2, s0, s1, s3, 13, w5); // pos 3
    sched!(w6, w7, w2, w3, w5);
    dround!(s1, s4, s3, s0, s2, 14, w6); // pos 4 [original]

    sched!(w7, w0, w3, w4, w6);
    dround!(s0, s1, s2, s3, s4, 15, w7); // pos 0
    sched!(w0, w1, w4, w5, w7);
    dround!(s3, s0, s4, s2, s1, 16, w0); // pos 1
    sched!(w1, w2, w5, w6, w0);
    dround!(s2, s3, s1, s4, s0, 17, w1); // pos 2
    sched!(w2, w3, w6, w7, w1);
    dround!(s4, s2, s0, s1, s3, 18, w2); // pos 3
    sched!(w3, w4, w7, w0, w2);
    dround!(s1, s4, s3, s0, s2, 19, w3); // pos 4 [original]

    sched!(w4, w5, w0, w1, w3);
    dround!(s0, s1, s2, s3, s4, 20, w4); // pos 0
    sched!(w5, w6, w1, w2, w4);
    dround!(s3, s0, s4, s2, s1, 21, w5); // pos 1
    sched!(w6, w7, w2, w3, w5);
    dround!(s2, s3, s1, s4, s0, 22, w6); // pos 2
    sched!(w7, w0, w3, w4, w6);
    dround!(s4, s2, s0, s1, s3, 23, w7); // pos 3

    sched!(w0, w1, w4, w5, w7);
    dround!(s1, s4, s3, s0, s2, 24, w0); // pos 4 [original]
    sched!(w1, w2, w5, w6, w0);
    dround!(s0, s1, s2, s3, s4, 25, w1); // pos 0
    sched!(w2, w3, w6, w7, w1);
    dround!(s3, s0, s4, s2, s1, 26, w2); // pos 1
    sched!(w3, w4, w7, w0, w2);
    dround!(s2, s3, s1, s4, s0, 27, w3); // pos 2
    sched!(w4, w5, w0, w1, w3);
    dround!(s4, s2, s0, s1, s3, 28, w4); // pos 3
    sched!(w5, w6, w1, w2, w4);
    dround!(s1, s4, s3, s0, s2, 29, w5); // pos 4 [original]

    sched!(w6, w7, w2, w3, w5);
    dround!(s0, s1, s2, s3, s4, 30, w6); // pos 0
    sched!(w7, w0, w3, w4, w6);
    dround!(s3, s0, s4, s2, s1, 31, w7); // pos 1
    sched!(w0, w1, w4, w5, w7);
    dround!(s2, s3, s1, s4, s0, 32, w0); // pos 2
    sched!(w1, w2, w5, w6, w0);
    dround!(s4, s2, s0, s1, s3, 33, w1); // pos 3
    sched!(w2, w3, w6, w7, w1);
    dround!(s1, s4, s3, s0, s2, 34, w2); // pos 4 [original]

    sched!(w3, w4, w7, w0, w2);
    dround!(s0, s1, s2, s3, s4, 35, w3); // pos 0
    sched!(w4, w5, w0, w1, w3);
    dround!(s3, s0, s4, s2, s1, 36, w4); // pos 1
    sched!(w5, w6, w1, w2, w4);
    dround!(s2, s3, s1, s4, s0, 37, w5); // pos 2
    sched!(w6, w7, w2, w3, w5);
    dround!(s4, s2, s0, s1, s3, 38, w6); // pos 3
    sched!(w7, w0, w3, w4, w6);
    dround!(s1, s4, s3, s0, s2, 39, w7); // pos 4 [original]

    // Add saved state (s0..s3 are back to original ab/cd/ef/gh roles).
    s0 = vaddq_u64(s0, s0_save);
    s1 = vaddq_u64(s1, s1_save);
    s2 = vaddq_u64(s2, s2_save);
    s3 = vaddq_u64(s3, s3_save);

    ptr = ptr.add(128);
  }

  // Store final state.
  vst1q_u64(state.as_mut_ptr(), s0);
  vst1q_u64(state.as_mut_ptr().add(2), s1);
  vst1q_u64(state.as_mut_ptr().add(4), s2);
  vst1q_u64(state.as_mut_ptr().add(6), s3);
}
