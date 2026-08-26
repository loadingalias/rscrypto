//! AArch64 SHA-1 Crypto Extension compression.
//!
//! The instruction schedule follows RustCrypto `sha1` 0.11.0's AArch64
//! backend (MIT OR Apache-2.0), narrowed to one exact 64-byte block.

use core::arch::aarch64::*;

const ROUND_CONSTANTS: [u32; 4] = [0x5a82_7999, 0x6ed9_eba1, 0x8f1b_bcdc, 0xca62_c1d6];

/// Compresses one SHA-1 block with AArch64 SHA instructions.
///
/// # Safety
///
/// The caller must establish the AArch64 `sha2` target feature before entry.
#[target_feature(enable = "sha2")]
pub(super) unsafe fn compress(state: &mut [u32; 5], block: &[u8; 64]) {
  // SAFETY: The target-feature contract above makes every SHA/NEON intrinsic
  // legal. `state` contains five initialized u32 words: the vector load/store
  // touches exactly the first four. The four unaligned byte loads cover
  // block[0..64] in 16-byte increments, so all pointer additions remain inside
  // the exact 64-byte array. No pointer or reference escapes this call.
  unsafe {
    let mut abcd = vld1q_u32(state.as_ptr());
    let mut e_zero = state[4];
    let [constant_zero, constant_one, constant_two, constant_three] =
      ROUND_CONSTANTS.map(|constant| vdupq_n_u32(constant));
    let saved_abcd = abcd;
    let saved_e = e_zero;
    let pointer = block.as_ptr();

    let mut message_zero = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(pointer)));
    let mut message_one = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(pointer.add(16))));
    let mut message_two = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(pointer.add(32))));
    let mut message_three = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(pointer.add(48))));

    let mut work_zero = vaddq_u32(message_zero, constant_zero);
    let mut work_one = vaddq_u32(message_one, constant_zero);

    let mut e_one = vsha1h_u32(vgetq_lane_u32(abcd, 0));
    abcd = vsha1cq_u32(abcd, e_zero, work_zero);
    work_zero = vaddq_u32(message_two, constant_zero);
    message_zero = vsha1su0q_u32(message_zero, message_one, message_two);

    e_zero = vsha1h_u32(vgetq_lane_u32(abcd, 0));
    abcd = vsha1cq_u32(abcd, e_one, work_one);
    work_one = vaddq_u32(message_three, constant_zero);
    message_zero = vsha1su1q_u32(message_zero, message_three);
    message_one = vsha1su0q_u32(message_one, message_two, message_three);

    e_one = vsha1h_u32(vgetq_lane_u32(abcd, 0));
    abcd = vsha1cq_u32(abcd, e_zero, work_zero);
    work_zero = vaddq_u32(message_zero, constant_zero);
    message_one = vsha1su1q_u32(message_one, message_zero);
    message_two = vsha1su0q_u32(message_two, message_three, message_zero);

    e_zero = vsha1h_u32(vgetq_lane_u32(abcd, 0));
    abcd = vsha1cq_u32(abcd, e_one, work_one);
    work_one = vaddq_u32(message_one, constant_one);
    message_two = vsha1su1q_u32(message_two, message_one);
    message_three = vsha1su0q_u32(message_three, message_zero, message_one);

    e_one = vsha1h_u32(vgetq_lane_u32(abcd, 0));
    abcd = vsha1cq_u32(abcd, e_zero, work_zero);
    work_zero = vaddq_u32(message_two, constant_one);
    message_three = vsha1su1q_u32(message_three, message_two);
    message_zero = vsha1su0q_u32(message_zero, message_one, message_two);

    e_zero = vsha1h_u32(vgetq_lane_u32(abcd, 0));
    abcd = vsha1pq_u32(abcd, e_one, work_one);
    work_one = vaddq_u32(message_three, constant_one);
    message_zero = vsha1su1q_u32(message_zero, message_three);
    message_one = vsha1su0q_u32(message_one, message_two, message_three);

    e_one = vsha1h_u32(vgetq_lane_u32(abcd, 0));
    abcd = vsha1pq_u32(abcd, e_zero, work_zero);
    work_zero = vaddq_u32(message_zero, constant_one);
    message_one = vsha1su1q_u32(message_one, message_zero);
    message_two = vsha1su0q_u32(message_two, message_three, message_zero);

    e_zero = vsha1h_u32(vgetq_lane_u32(abcd, 0));
    abcd = vsha1pq_u32(abcd, e_one, work_one);
    work_one = vaddq_u32(message_one, constant_one);
    message_two = vsha1su1q_u32(message_two, message_one);
    message_three = vsha1su0q_u32(message_three, message_zero, message_one);

    e_one = vsha1h_u32(vgetq_lane_u32(abcd, 0));
    abcd = vsha1pq_u32(abcd, e_zero, work_zero);
    work_zero = vaddq_u32(message_two, constant_two);
    message_three = vsha1su1q_u32(message_three, message_two);
    message_zero = vsha1su0q_u32(message_zero, message_one, message_two);

    e_zero = vsha1h_u32(vgetq_lane_u32(abcd, 0));
    abcd = vsha1pq_u32(abcd, e_one, work_one);
    work_one = vaddq_u32(message_three, constant_two);
    message_zero = vsha1su1q_u32(message_zero, message_three);
    message_one = vsha1su0q_u32(message_one, message_two, message_three);

    e_one = vsha1h_u32(vgetq_lane_u32(abcd, 0));
    abcd = vsha1mq_u32(abcd, e_zero, work_zero);
    work_zero = vaddq_u32(message_zero, constant_two);
    message_one = vsha1su1q_u32(message_one, message_zero);
    message_two = vsha1su0q_u32(message_two, message_three, message_zero);

    e_zero = vsha1h_u32(vgetq_lane_u32(abcd, 0));
    abcd = vsha1mq_u32(abcd, e_one, work_one);
    work_one = vaddq_u32(message_one, constant_two);
    message_two = vsha1su1q_u32(message_two, message_one);
    message_three = vsha1su0q_u32(message_three, message_zero, message_one);

    e_one = vsha1h_u32(vgetq_lane_u32(abcd, 0));
    abcd = vsha1mq_u32(abcd, e_zero, work_zero);
    work_zero = vaddq_u32(message_two, constant_two);
    message_three = vsha1su1q_u32(message_three, message_two);
    message_zero = vsha1su0q_u32(message_zero, message_one, message_two);

    e_zero = vsha1h_u32(vgetq_lane_u32(abcd, 0));
    abcd = vsha1mq_u32(abcd, e_one, work_one);
    work_one = vaddq_u32(message_three, constant_three);
    message_zero = vsha1su1q_u32(message_zero, message_three);
    message_one = vsha1su0q_u32(message_one, message_two, message_three);

    e_one = vsha1h_u32(vgetq_lane_u32(abcd, 0));
    abcd = vsha1mq_u32(abcd, e_zero, work_zero);
    work_zero = vaddq_u32(message_zero, constant_three);
    message_one = vsha1su1q_u32(message_one, message_zero);
    message_two = vsha1su0q_u32(message_two, message_three, message_zero);

    e_zero = vsha1h_u32(vgetq_lane_u32(abcd, 0));
    abcd = vsha1pq_u32(abcd, e_one, work_one);
    work_one = vaddq_u32(message_one, constant_three);
    message_two = vsha1su1q_u32(message_two, message_one);
    message_three = vsha1su0q_u32(message_three, message_zero, message_one);

    e_one = vsha1h_u32(vgetq_lane_u32(abcd, 0));
    abcd = vsha1pq_u32(abcd, e_zero, work_zero);
    work_zero = vaddq_u32(message_two, constant_three);
    message_three = vsha1su1q_u32(message_three, message_two);

    e_zero = vsha1h_u32(vgetq_lane_u32(abcd, 0));
    abcd = vsha1pq_u32(abcd, e_one, work_one);
    work_one = vaddq_u32(message_three, constant_three);

    e_one = vsha1h_u32(vgetq_lane_u32(abcd, 0));
    abcd = vsha1pq_u32(abcd, e_zero, work_zero);

    e_zero = vsha1h_u32(vgetq_lane_u32(abcd, 0));
    abcd = vsha1pq_u32(abcd, e_one, work_one);

    abcd = vaddq_u32(saved_abcd, abcd);
    e_zero = e_zero.wrapping_add(saved_e);
    vst1q_u32(state.as_mut_ptr(), abcd);
    state[4] = e_zero;
  }
}
