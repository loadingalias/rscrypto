//! aarch64 FEAT_SHA512 kernel for SHA-512.
//!
//! Uses the SHA-512 Crypto Extension instructions (grouped under `sha3` in Rust):
//! `vsha512hq_u64`, `vsha512h2q_u64`, `vsha512su0q_u64`, `vsha512su1q_u64`.
//!
//! Available on Apple Silicon (M1+), Graviton2+, Ampere Altra, Cortex-A76+.

#![allow(unsafe_code)]
#![allow(clippy::inline_always)]
#![allow(clippy::indexing_slicing)]

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

use crate::hashes::util::Aligned64;

/// K constants packed as 40 pairs of `[K[2i], K[2i+1]]` for `vld1q_u64` loads.
///
/// Cache-line aligned (64 bytes) so the 640-byte table starts at an L1 boundary.
#[cfg(target_arch = "aarch64")]
static K_PAIRS: Aligned64<[[u64; 2]; 40]> = Aligned64([
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

  // SAFETY: NEON/SHA512 intrinsics (via sha3 target feature) are available via this
  // function's #[target_feature] attribute. Pointer arithmetic is bounded by blocks.len().
  unsafe {
    let mut ab = vld1q_u64(state.as_ptr());
    let mut cd = vld1q_u64(state.as_ptr().add(2));
    let mut ef = vld1q_u64(state.as_ptr().add(4));
    let mut gh = vld1q_u64(state.as_ptr().add(6));

    let mut ptr = blocks.as_ptr();
    let end = ptr.add(blocks.len());

    while ptr < end {
      let ab_orig = ab;
      let cd_orig = cd;
      let ef_orig = ef;
      let gh_orig = gh;

      // Load and byte-swap 8 message pairs (128 bytes).
      let mut s0 = vreinterpretq_u64_u8(vrev64q_u8(vld1q_u8(ptr)));
      let mut s1 = vreinterpretq_u64_u8(vrev64q_u8(vld1q_u8(ptr.add(16))));
      let mut s2 = vreinterpretq_u64_u8(vrev64q_u8(vld1q_u8(ptr.add(32))));
      let mut s3 = vreinterpretq_u64_u8(vrev64q_u8(vld1q_u8(ptr.add(48))));
      let mut s4 = vreinterpretq_u64_u8(vrev64q_u8(vld1q_u8(ptr.add(64))));
      let mut s5 = vreinterpretq_u64_u8(vrev64q_u8(vld1q_u8(ptr.add(80))));
      let mut s6 = vreinterpretq_u64_u8(vrev64q_u8(vld1q_u8(ptr.add(96))));
      let mut s7 = vreinterpretq_u64_u8(vrev64q_u8(vld1q_u8(ptr.add(112))));

      let k = |pair: usize| -> uint64x2_t { vld1q_u64(K_PAIRS[pair].as_ptr()) };

      let mut initial_sum = vaddq_u64(s0, k(0));
      let mut sum = vaddq_u64(vextq_u64(initial_sum, initial_sum, 1), gh);
      let mut intermed = vsha512hq_u64(sum, vextq_u64(ef, gh, 1), vextq_u64(cd, ef, 1));
      gh = vsha512h2q_u64(intermed, cd, ab);
      cd = vaddq_u64(cd, intermed);

      initial_sum = vaddq_u64(s1, k(1));
      sum = vaddq_u64(vextq_u64(initial_sum, initial_sum, 1), ef);
      intermed = vsha512hq_u64(sum, vextq_u64(cd, ef, 1), vextq_u64(ab, cd, 1));
      ef = vsha512h2q_u64(intermed, ab, gh);
      ab = vaddq_u64(ab, intermed);

      initial_sum = vaddq_u64(s2, k(2));
      sum = vaddq_u64(vextq_u64(initial_sum, initial_sum, 1), cd);
      intermed = vsha512hq_u64(sum, vextq_u64(ab, cd, 1), vextq_u64(gh, ab, 1));
      cd = vsha512h2q_u64(intermed, gh, ef);
      gh = vaddq_u64(gh, intermed);

      initial_sum = vaddq_u64(s3, k(3));
      sum = vaddq_u64(vextq_u64(initial_sum, initial_sum, 1), ab);
      intermed = vsha512hq_u64(sum, vextq_u64(gh, ab, 1), vextq_u64(ef, gh, 1));
      ab = vsha512h2q_u64(intermed, ef, cd);
      ef = vaddq_u64(ef, intermed);

      initial_sum = vaddq_u64(s4, k(4));
      sum = vaddq_u64(vextq_u64(initial_sum, initial_sum, 1), gh);
      intermed = vsha512hq_u64(sum, vextq_u64(ef, gh, 1), vextq_u64(cd, ef, 1));
      gh = vsha512h2q_u64(intermed, cd, ab);
      cd = vaddq_u64(cd, intermed);

      initial_sum = vaddq_u64(s5, k(5));
      sum = vaddq_u64(vextq_u64(initial_sum, initial_sum, 1), ef);
      intermed = vsha512hq_u64(sum, vextq_u64(cd, ef, 1), vextq_u64(ab, cd, 1));
      ef = vsha512h2q_u64(intermed, ab, gh);
      ab = vaddq_u64(ab, intermed);

      initial_sum = vaddq_u64(s6, k(6));
      sum = vaddq_u64(vextq_u64(initial_sum, initial_sum, 1), cd);
      intermed = vsha512hq_u64(sum, vextq_u64(ab, cd, 1), vextq_u64(gh, ab, 1));
      cd = vsha512h2q_u64(intermed, gh, ef);
      gh = vaddq_u64(gh, intermed);

      initial_sum = vaddq_u64(s7, k(7));
      sum = vaddq_u64(vextq_u64(initial_sum, initial_sum, 1), ab);
      intermed = vsha512hq_u64(sum, vextq_u64(gh, ab, 1), vextq_u64(ef, gh, 1));
      ab = vsha512h2q_u64(intermed, ef, cd);
      ef = vaddq_u64(ef, intermed);

      for pair in (8..40).step_by(8) {
        s0 = vsha512su1q_u64(vsha512su0q_u64(s0, s1), s7, vextq_u64(s4, s5, 1));
        initial_sum = vaddq_u64(s0, k(pair));
        sum = vaddq_u64(vextq_u64(initial_sum, initial_sum, 1), gh);
        intermed = vsha512hq_u64(sum, vextq_u64(ef, gh, 1), vextq_u64(cd, ef, 1));
        gh = vsha512h2q_u64(intermed, cd, ab);
        cd = vaddq_u64(cd, intermed);

        s1 = vsha512su1q_u64(vsha512su0q_u64(s1, s2), s0, vextq_u64(s5, s6, 1));
        initial_sum = vaddq_u64(s1, k(pair + 1));
        sum = vaddq_u64(vextq_u64(initial_sum, initial_sum, 1), ef);
        intermed = vsha512hq_u64(sum, vextq_u64(cd, ef, 1), vextq_u64(ab, cd, 1));
        ef = vsha512h2q_u64(intermed, ab, gh);
        ab = vaddq_u64(ab, intermed);

        s2 = vsha512su1q_u64(vsha512su0q_u64(s2, s3), s1, vextq_u64(s6, s7, 1));
        initial_sum = vaddq_u64(s2, k(pair + 2));
        sum = vaddq_u64(vextq_u64(initial_sum, initial_sum, 1), cd);
        intermed = vsha512hq_u64(sum, vextq_u64(ab, cd, 1), vextq_u64(gh, ab, 1));
        cd = vsha512h2q_u64(intermed, gh, ef);
        gh = vaddq_u64(gh, intermed);

        s3 = vsha512su1q_u64(vsha512su0q_u64(s3, s4), s2, vextq_u64(s7, s0, 1));
        initial_sum = vaddq_u64(s3, k(pair + 3));
        sum = vaddq_u64(vextq_u64(initial_sum, initial_sum, 1), ab);
        intermed = vsha512hq_u64(sum, vextq_u64(gh, ab, 1), vextq_u64(ef, gh, 1));
        ab = vsha512h2q_u64(intermed, ef, cd);
        ef = vaddq_u64(ef, intermed);

        s4 = vsha512su1q_u64(vsha512su0q_u64(s4, s5), s3, vextq_u64(s0, s1, 1));
        initial_sum = vaddq_u64(s4, k(pair + 4));
        sum = vaddq_u64(vextq_u64(initial_sum, initial_sum, 1), gh);
        intermed = vsha512hq_u64(sum, vextq_u64(ef, gh, 1), vextq_u64(cd, ef, 1));
        gh = vsha512h2q_u64(intermed, cd, ab);
        cd = vaddq_u64(cd, intermed);

        s5 = vsha512su1q_u64(vsha512su0q_u64(s5, s6), s4, vextq_u64(s1, s2, 1));
        initial_sum = vaddq_u64(s5, k(pair + 5));
        sum = vaddq_u64(vextq_u64(initial_sum, initial_sum, 1), ef);
        intermed = vsha512hq_u64(sum, vextq_u64(cd, ef, 1), vextq_u64(ab, cd, 1));
        ef = vsha512h2q_u64(intermed, ab, gh);
        ab = vaddq_u64(ab, intermed);

        s6 = vsha512su1q_u64(vsha512su0q_u64(s6, s7), s5, vextq_u64(s2, s3, 1));
        initial_sum = vaddq_u64(s6, k(pair + 6));
        sum = vaddq_u64(vextq_u64(initial_sum, initial_sum, 1), cd);
        intermed = vsha512hq_u64(sum, vextq_u64(ab, cd, 1), vextq_u64(gh, ab, 1));
        cd = vsha512h2q_u64(intermed, gh, ef);
        gh = vaddq_u64(gh, intermed);

        s7 = vsha512su1q_u64(vsha512su0q_u64(s7, s0), s6, vextq_u64(s3, s4, 1));
        initial_sum = vaddq_u64(s7, k(pair + 7));
        sum = vaddq_u64(vextq_u64(initial_sum, initial_sum, 1), ab);
        intermed = vsha512hq_u64(sum, vextq_u64(gh, ab, 1), vextq_u64(ef, gh, 1));
        ab = vsha512h2q_u64(intermed, ef, cd);
        ef = vaddq_u64(ef, intermed);
      }

      ab = vaddq_u64(ab, ab_orig);
      cd = vaddq_u64(cd, cd_orig);
      ef = vaddq_u64(ef, ef_orig);
      gh = vaddq_u64(gh, gh_orig);

      ptr = ptr.add(128);
    }

    vst1q_u64(state.as_mut_ptr(), ab);
    vst1q_u64(state.as_mut_ptr().add(2), cd);
    vst1q_u64(state.as_mut_ptr().add(4), ef);
    vst1q_u64(state.as_mut_ptr().add(6), gh);
  } // unsafe
}
