//! aarch64 CRC-32/CRC-32C kernels.
//!
//! This module provides:
//! - Hardware CRC extension kernels (`crc` target_feature)
//! - Fusion kernels that combine CRC instructions with PMULL folding (`aes`)
//! - Optional SHA3/EOR3-enhanced fusion kernels (`sha3`)
//!
//! # Safety
//!
//! Uses `unsafe` for aarch64 intrinsics. Callers must ensure required target
//! features are available before selecting a kernel (the dispatcher does this).

#![allow(unsafe_code)]
// SAFETY: This module is intrinsics-heavy and uses tight, invariant-driven indexing
// (e.g. fixed-size lanes and chunked processing) where bounds are proven by
// control flow; Clippy cannot always see these invariants.
#![allow(clippy::indexing_slicing)]
#![allow(unsafe_op_in_unsafe_fn)]

use core::{arch::aarch64::*, ptr};

use crate::common::{
  combine::{Gf2Matrix32, generate_shift8_matrix_32},
  tables::{CRC32_IEEE_POLY, CRC32C_POLY},
};

const CRC32_SHIFT8_MATRIX: Gf2Matrix32 = generate_shift8_matrix_32(CRC32_IEEE_POLY);
const CRC32C_SHIFT8_MATRIX: Gf2Matrix32 = generate_shift8_matrix_32(CRC32C_POLY);

// ─────────────────────────────────────────────────────────────────────────────
// Hardware CRC extension (CRC-only)
// ─────────────────────────────────────────────────────────────────────────────

/// CRC-32 (IEEE) update using ARMv8 CRC extension.
///
/// `crc` is the current state (pre-inverted).
#[inline]
#[target_feature(enable = "crc")]
unsafe fn crc32_armv8(crc: u32, data: &[u8]) -> u32 {
  let mut state = crc;
  let mut buf = data.as_ptr();
  let mut len = data.len();

  // Align to 8-byte boundary for the hot loop.
  while len > 0 && (buf as usize & 7) != 0 {
    state = __crc32b(state, *buf);
    buf = buf.add(1);
    len = len.strict_sub(1);
  }

  while len >= 64 {
    state = __crc32d(state, ptr::read_unaligned(buf as *const u64));
    state = __crc32d(state, ptr::read_unaligned(buf.add(8) as *const u64));
    state = __crc32d(state, ptr::read_unaligned(buf.add(16) as *const u64));
    state = __crc32d(state, ptr::read_unaligned(buf.add(24) as *const u64));
    state = __crc32d(state, ptr::read_unaligned(buf.add(32) as *const u64));
    state = __crc32d(state, ptr::read_unaligned(buf.add(40) as *const u64));
    state = __crc32d(state, ptr::read_unaligned(buf.add(48) as *const u64));
    state = __crc32d(state, ptr::read_unaligned(buf.add(56) as *const u64));
    buf = buf.add(64);
    len = len.strict_sub(64);
  }

  while len >= 32 {
    state = __crc32d(state, ptr::read_unaligned(buf as *const u64));
    state = __crc32d(state, ptr::read_unaligned(buf.add(8) as *const u64));
    state = __crc32d(state, ptr::read_unaligned(buf.add(16) as *const u64));
    state = __crc32d(state, ptr::read_unaligned(buf.add(24) as *const u64));
    buf = buf.add(32);
    len = len.strict_sub(32);
  }

  while len >= 8 {
    state = __crc32d(state, ptr::read_unaligned(buf as *const u64));
    buf = buf.add(8);
    len = len.strict_sub(8);
  }

  if len >= 4 {
    state = __crc32w(state, ptr::read_unaligned(buf as *const u32));
    buf = buf.add(4);
    len = len.strict_sub(4);
  }

  if len >= 2 {
    state = __crc32h(state, ptr::read_unaligned(buf as *const u16));
    buf = buf.add(2);
    len = len.strict_sub(2);
  }

  if len > 0 {
    state = __crc32b(state, *buf);
  }

  state
}

/// CRC-32C (Castagnoli) update using ARMv8 CRC extension.
///
/// `crc` is the current state (pre-inverted).
#[inline]
#[target_feature(enable = "crc")]
unsafe fn crc32c_armv8(crc: u32, data: &[u8]) -> u32 {
  let mut state = crc;
  let mut buf = data.as_ptr();
  let mut len = data.len();

  // Align to 8-byte boundary for the hot loop.
  while len > 0 && (buf as usize & 7) != 0 {
    state = __crc32cb(state, *buf);
    buf = buf.add(1);
    len = len.strict_sub(1);
  }

  while len >= 64 {
    state = __crc32cd(state, ptr::read_unaligned(buf as *const u64));
    state = __crc32cd(state, ptr::read_unaligned(buf.add(8) as *const u64));
    state = __crc32cd(state, ptr::read_unaligned(buf.add(16) as *const u64));
    state = __crc32cd(state, ptr::read_unaligned(buf.add(24) as *const u64));
    state = __crc32cd(state, ptr::read_unaligned(buf.add(32) as *const u64));
    state = __crc32cd(state, ptr::read_unaligned(buf.add(40) as *const u64));
    state = __crc32cd(state, ptr::read_unaligned(buf.add(48) as *const u64));
    state = __crc32cd(state, ptr::read_unaligned(buf.add(56) as *const u64));
    buf = buf.add(64);
    len = len.strict_sub(64);
  }

  while len >= 32 {
    state = __crc32cd(state, ptr::read_unaligned(buf as *const u64));
    state = __crc32cd(state, ptr::read_unaligned(buf.add(8) as *const u64));
    state = __crc32cd(state, ptr::read_unaligned(buf.add(16) as *const u64));
    state = __crc32cd(state, ptr::read_unaligned(buf.add(24) as *const u64));
    buf = buf.add(32);
    len = len.strict_sub(32);
  }

  while len >= 8 {
    state = __crc32cd(state, ptr::read_unaligned(buf as *const u64));
    buf = buf.add(8);
    len = len.strict_sub(8);
  }

  if len >= 4 {
    state = __crc32cw(state, ptr::read_unaligned(buf as *const u32));
    buf = buf.add(4);
    len = len.strict_sub(4);
  }

  if len >= 2 {
    state = __crc32ch(state, ptr::read_unaligned(buf as *const u16));
    buf = buf.add(2);
    len = len.strict_sub(2);
  }

  if len > 0 {
    state = __crc32cb(state, *buf);
  }

  state
}

/// Safe wrapper for CRC-32 ARMv8 CRC extension kernel.
#[inline]
pub fn crc32_armv8_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies CRC extension before selecting this kernel.
  unsafe { crc32_armv8(crc, data) }
}

/// Safe wrapper for CRC-32C ARMv8 CRC extension kernel.
#[inline]
pub fn crc32c_armv8_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies CRC extension before selecting this kernel.
  unsafe { crc32c_armv8(crc, data) }
}

// ─────────────────────────────────────────────────────────────────────────────
// Hardware CRC extension (multi-stream wrappers)
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
#[target_feature(enable = "crc")]
unsafe fn crc32_armv8_nway<const N: usize>(crc: u32, data: &[u8]) -> u32 {
  debug_assert!(N == 2 || N == 3);

  let len = data.len();
  if len < N.strict_mul(256) {
    return crc32_armv8(crc, data);
  }

  let chunk_len = len / N;
  let mut lanes = [!0u32; N];

  let mut i: usize = 0;
  while i < chunk_len {
    if i.strict_add(8) <= chunk_len {
      let mut lane_idx: usize = 0;
      while lane_idx < N {
        let base = lane_idx.strict_mul(chunk_len).strict_add(i);
        lanes[lane_idx] = __crc32d(
          lanes[lane_idx],
          ptr::read_unaligned(data.as_ptr().add(base) as *const u64),
        );
        lane_idx = lane_idx.strict_add(1);
      }
      i = i.strict_add(8);
    } else {
      let mut lane_idx: usize = 0;
      while lane_idx < N {
        let base = lane_idx.strict_mul(chunk_len).strict_add(i);
        lanes[lane_idx] = __crc32b(lanes[lane_idx], *data.get_unchecked(base));
        lane_idx = lane_idx.strict_add(1);
      }
      i = i.strict_add(1);
    }
  }

  let tail_start = chunk_len.strict_mul(N);
  if tail_start < len {
    lanes[N - 1] = crc32_armv8(lanes[N - 1], data.get_unchecked(tail_start..));
  }

  let mut data_crc_final: u32 = 0;
  let mut lane_idx: usize = 0;
  while lane_idx < N {
    let lane_len = if lane_idx.strict_add(1) == N {
      len.strict_sub(chunk_len.strict_mul(lane_idx))
    } else {
      chunk_len
    };
    data_crc_final =
      crate::common::combine::combine_crc32(data_crc_final, lanes[lane_idx] ^ !0, lane_len, CRC32_SHIFT8_MATRIX);
    lane_idx = lane_idx.strict_add(1);
  }

  let boundary_final = crc ^ !0;
  let combined_final = crate::common::combine::combine_crc32(boundary_final, data_crc_final, len, CRC32_SHIFT8_MATRIX);
  combined_final ^ !0
}

#[inline]
#[target_feature(enable = "crc")]
unsafe fn crc32c_armv8_nway<const N: usize>(crc: u32, data: &[u8]) -> u32 {
  debug_assert!(N == 2 || N == 3);

  let len = data.len();
  if len < N.strict_mul(256) {
    return crc32c_armv8(crc, data);
  }

  let chunk_len = len / N;
  let mut lanes = [!0u32; N];

  let mut i: usize = 0;
  while i < chunk_len {
    if i.strict_add(8) <= chunk_len {
      let mut lane_idx: usize = 0;
      while lane_idx < N {
        let base = lane_idx.strict_mul(chunk_len).strict_add(i);
        lanes[lane_idx] = __crc32cd(
          lanes[lane_idx],
          ptr::read_unaligned(data.as_ptr().add(base) as *const u64),
        );
        lane_idx = lane_idx.strict_add(1);
      }
      i = i.strict_add(8);
    } else {
      let mut lane_idx: usize = 0;
      while lane_idx < N {
        let base = lane_idx.strict_mul(chunk_len).strict_add(i);
        lanes[lane_idx] = __crc32cb(lanes[lane_idx], *data.get_unchecked(base));
        lane_idx = lane_idx.strict_add(1);
      }
      i = i.strict_add(1);
    }
  }

  let tail_start = chunk_len.strict_mul(N);
  if tail_start < len {
    lanes[N - 1] = crc32c_armv8(lanes[N - 1], data.get_unchecked(tail_start..));
  }

  let mut data_crc_final: u32 = 0;
  let mut lane_idx: usize = 0;
  while lane_idx < N {
    let lane_len = if lane_idx.strict_add(1) == N {
      len.strict_sub(chunk_len.strict_mul(lane_idx))
    } else {
      chunk_len
    };
    data_crc_final =
      crate::common::combine::combine_crc32(data_crc_final, lanes[lane_idx] ^ !0, lane_len, CRC32C_SHIFT8_MATRIX);
    lane_idx = lane_idx.strict_add(1);
  }

  let boundary_final = crc ^ !0;
  let combined_final = crate::common::combine::combine_crc32(boundary_final, data_crc_final, len, CRC32C_SHIFT8_MATRIX);
  combined_final ^ !0
}

#[inline]
pub fn crc32_armv8_2way_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies CRC extension before selecting this kernel.
  unsafe { crc32_armv8_nway::<2>(crc, data) }
}

#[inline]
pub fn crc32_armv8_3way_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies CRC extension before selecting this kernel.
  unsafe { crc32_armv8_nway::<3>(crc, data) }
}

#[inline]
pub fn crc32c_armv8_2way_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies CRC extension before selecting this kernel.
  unsafe { crc32c_armv8_nway::<2>(crc, data) }
}

#[inline]
pub fn crc32c_armv8_3way_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies CRC extension before selecting this kernel.
  unsafe { crc32c_armv8_nway::<3>(crc, data) }
}

// ─────────────────────────────────────────────────────────────────────────────
// "SVE2 PMULL" Tier (ILP-oriented split + combine)
// ─────────────────────────────────────────────────────────────────────────────

// Round-robin scheduling across independent CRC states to expose ILP.
const CRC32_SVE2_PMULL_STRIPE_BYTES: usize = 16 * 1024;

#[inline]
fn crc32_sve2_pmull_nway<const N: usize>(
  crc: u32,
  data: &[u8],
  update: fn(u32, &[u8]) -> u32,
  shift8: Gf2Matrix32,
) -> u32 {
  debug_assert!(N == 2 || N == 3);

  let len = data.len();
  if len < N.strict_mul(192) {
    return update(crc, data);
  }

  let chunk_len = len.strict_div(N);
  let mut lanes = [!0u32; N];

  let mut offset: usize = 0;
  while offset < chunk_len {
    let remaining = chunk_len.strict_sub(offset);
    let step = remaining.min(CRC32_SVE2_PMULL_STRIPE_BYTES);

    let mut lane_idx: usize = 0;
    while lane_idx < N {
      let start = lane_idx.strict_mul(chunk_len).strict_add(offset);
      lanes[lane_idx] = update(lanes[lane_idx], &data[start..start.strict_add(step)]);
      lane_idx = lane_idx.strict_add(1);
    }

    offset = offset.strict_add(step);
  }

  let tail_start = chunk_len.strict_mul(N);
  if tail_start < len {
    lanes[N - 1] = update(lanes[N - 1], &data[tail_start..]);
  }

  // Combine the N independent CRCs with precomputed matrices to avoid
  // per-lane exponentiation overhead (which can dominate at mid sizes).
  let mat_chunk = crate::common::combine::pow_shift8_matrix_32(chunk_len, shift8);
  let mat_total = crate::common::combine::pow_shift8_matrix_32(len, shift8);

  let mut data_crc_final: u32 = 0;
  let mut lane_idx: usize = 0;
  while lane_idx < N {
    let start = lane_idx.strict_mul(chunk_len);
    let lane_len = if lane_idx.strict_add(1) == N {
      len.strict_sub(start)
    } else {
      chunk_len
    };
    let mat_lane = if lane_len == chunk_len {
      mat_chunk
    } else {
      crate::common::combine::pow_shift8_matrix_32(lane_len, shift8)
    };
    data_crc_final = mat_lane.mul_vec(data_crc_final) ^ (lanes[lane_idx] ^ !0);
    lane_idx = lane_idx.strict_add(1);
  }

  let boundary_final = crc ^ !0;
  (mat_total.mul_vec(boundary_final) ^ data_crc_final) ^ !0
}

#[inline]
pub fn crc32_iso_hdlc_sve2_pmull_2way_safe(crc: u32, data: &[u8]) -> u32 {
  crc32_sve2_pmull_nway::<2>(crc, data, crc32_iso_hdlc_pmull_v12e_v1_safe, CRC32_SHIFT8_MATRIX)
}

#[inline]
pub fn crc32_iso_hdlc_sve2_pmull_3way_safe(crc: u32, data: &[u8]) -> u32 {
  crc32_sve2_pmull_nway::<3>(crc, data, crc32_iso_hdlc_pmull_v12e_v1_safe, CRC32_SHIFT8_MATRIX)
}

#[inline]
pub fn crc32c_iscsi_sve2_pmull_2way_safe(crc: u32, data: &[u8]) -> u32 {
  crc32_sve2_pmull_nway::<2>(crc, data, crc32c_iscsi_pmull_v12e_v1_safe, CRC32C_SHIFT8_MATRIX)
}

#[inline]
pub fn crc32c_iscsi_sve2_pmull_3way_safe(crc: u32, data: &[u8]) -> u32 {
  crc32_sve2_pmull_nway::<3>(crc, data, crc32c_iscsi_pmull_v12e_v1_safe, CRC32C_SHIFT8_MATRIX)
}

// ─────────────────────────────────────────────────────────────────────────────
// PMULL tier multi-stream wrappers (2/3-way)
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
pub fn crc32_iso_hdlc_pmull_2way_safe(crc: u32, data: &[u8]) -> u32 {
  crc32_sve2_pmull_nway::<2>(crc, data, crc32_iso_hdlc_pmull_v9s3x2e_s3_safe, CRC32_SHIFT8_MATRIX)
}

#[inline]
pub fn crc32_iso_hdlc_pmull_3way_safe(crc: u32, data: &[u8]) -> u32 {
  crc32_sve2_pmull_nway::<3>(crc, data, crc32_iso_hdlc_pmull_v9s3x2e_s3_safe, CRC32_SHIFT8_MATRIX)
}

#[inline]
pub fn crc32c_iscsi_pmull_2way_safe(crc: u32, data: &[u8]) -> u32 {
  crc32_sve2_pmull_nway::<2>(crc, data, crc32c_iscsi_pmull_v9s3x2e_s3_safe, CRC32C_SHIFT8_MATRIX)
}

#[inline]
pub fn crc32c_iscsi_pmull_3way_safe(crc: u32, data: &[u8]) -> u32 {
  crc32_sve2_pmull_nway::<3>(crc, data, crc32c_iscsi_pmull_v9s3x2e_s3_safe, CRC32C_SHIFT8_MATRIX)
}

// ─────────────────────────────────────────────────────────────────────────────
// EOR3 tier multi-stream wrappers (2/3-way)
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
pub fn crc32_iso_hdlc_pmull_eor3_2way_safe(crc: u32, data: &[u8]) -> u32 {
  crc32_sve2_pmull_nway::<2>(
    crc,
    data,
    crc32_iso_hdlc_pmull_eor3_v9s3x2e_s3_safe,
    CRC32_SHIFT8_MATRIX,
  )
}

#[inline]
pub fn crc32_iso_hdlc_pmull_eor3_3way_safe(crc: u32, data: &[u8]) -> u32 {
  crc32_sve2_pmull_nway::<3>(
    crc,
    data,
    crc32_iso_hdlc_pmull_eor3_v9s3x2e_s3_safe,
    CRC32_SHIFT8_MATRIX,
  )
}

#[inline]
pub fn crc32c_iscsi_pmull_eor3_2way_safe(crc: u32, data: &[u8]) -> u32 {
  crc32_sve2_pmull_nway::<2>(crc, data, crc32c_iscsi_pmull_eor3_v9s3x2e_s3_safe, CRC32C_SHIFT8_MATRIX)
}

#[inline]
pub fn crc32c_iscsi_pmull_eor3_3way_safe(crc: u32, data: &[u8]) -> u32 {
  crc32_sve2_pmull_nway::<3>(crc, data, crc32c_iscsi_pmull_eor3_v9s3x2e_s3_safe, CRC32C_SHIFT8_MATRIX)
}

// ─────────────────────────────────────────────────────────────────────────────
// Small-buffer wrappers (selected for len < fold block)
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
pub fn crc32_iso_hdlc_pmull_small_safe(crc: u32, data: &[u8]) -> u32 {
  crc32_iso_hdlc_pmull_v12e_v1_safe(crc, data)
}

#[inline]
pub fn crc32c_iscsi_pmull_small_safe(crc: u32, data: &[u8]) -> u32 {
  crc32c_iscsi_pmull_v12e_v1_safe(crc, data)
}

#[inline]
pub fn crc32_iso_hdlc_sve2_pmull_small_safe(crc: u32, data: &[u8]) -> u32 {
  crc32_iso_hdlc_pmull_v12e_v1_safe(crc, data)
}

#[inline]
pub fn crc32c_iscsi_sve2_pmull_small_safe(crc: u32, data: &[u8]) -> u32 {
  crc32c_iscsi_pmull_v12e_v1_safe(crc, data)
}

// ─────────────────────────────────────────────────────────────────────────────
// PMULL helpers (fusion kernels)
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
#[target_feature(enable = "aes")]
unsafe fn clmul_lo(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
  let result = vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0));
  vreinterpretq_u64_p128(result)
}

#[inline]
#[target_feature(enable = "aes")]
unsafe fn clmul_hi(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
  let result = vmull_p64(vgetq_lane_u64(a, 1), vgetq_lane_u64(b, 1));
  vreinterpretq_u64_p128(result)
}

#[inline]
#[target_feature(enable = "aes")]
unsafe fn clmul_scalar(a: u32, b: u32) -> uint64x2_t {
  let result = vmull_p64(a as u64, b as u64);
  vreinterpretq_u64_p128(result)
}

#[inline]
#[target_feature(enable = "aes")]
unsafe fn clmul_lo_and_xor(a: uint64x2_t, b: uint64x2_t, c: uint64x2_t) -> uint64x2_t {
  veorq_u64(clmul_lo(a, b), c)
}

#[inline]
#[target_feature(enable = "aes")]
unsafe fn clmul_hi_and_xor(a: uint64x2_t, b: uint64x2_t, c: uint64x2_t) -> uint64x2_t {
  veorq_u64(clmul_hi(a, b), c)
}

// ─────────────────────────────────────────────────────────────────────────────
// Fusion: CRC-32C (iSCSI) - PMULL v12e_v1
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
#[target_feature(enable = "crc,aes")]
unsafe fn crc32c_iscsi_pmull_v12e_v1(mut crc0: u32, mut buf: *const u8, mut len: usize) -> u32 {
  while len > 0 && (buf as usize & 7) != 0 {
    crc0 = __crc32cb(crc0, *buf);
    buf = buf.add(1);
    len = len.strict_sub(1);
  }

  if (buf as usize & 8) != 0 && len >= 8 {
    crc0 = __crc32cd(crc0, ptr::read_unaligned(buf as *const u64));
    buf = buf.add(8);
    len = len.strict_sub(8);
  }

  if len >= 192 {
    let end = buf.add(len);
    let limit = buf.add(len.strict_sub(192));

    let mut x0 = vld1q_u64(buf as *const u64);
    let mut x1 = vld1q_u64(buf.add(16) as *const u64);
    let mut x2 = vld1q_u64(buf.add(32) as *const u64);
    let mut x3 = vld1q_u64(buf.add(48) as *const u64);
    let mut x4 = vld1q_u64(buf.add(64) as *const u64);
    let mut x5 = vld1q_u64(buf.add(80) as *const u64);
    let mut x6 = vld1q_u64(buf.add(96) as *const u64);
    let mut x7 = vld1q_u64(buf.add(112) as *const u64);
    let mut x8 = vld1q_u64(buf.add(128) as *const u64);
    let mut x9 = vld1q_u64(buf.add(144) as *const u64);
    let mut x10 = vld1q_u64(buf.add(160) as *const u64);
    let mut x11 = vld1q_u64(buf.add(176) as *const u64);

    let k_vals: [u64; 2] = [0xa87ab8a8, 0xab7aff2a];
    let mut k = vld1q_u64(k_vals.as_ptr());

    let crc_vec = vsetq_lane_u64(crc0 as u64, vmovq_n_u64(0), 0);
    x0 = veorq_u64(crc_vec, x0);
    buf = buf.add(192);

    while buf <= limit {
      let y0 = clmul_lo_and_xor(x0, k, vld1q_u64(buf as *const u64));
      x0 = clmul_hi_and_xor(x0, k, y0);
      let y1 = clmul_lo_and_xor(x1, k, vld1q_u64(buf.add(16) as *const u64));
      x1 = clmul_hi_and_xor(x1, k, y1);
      let y2 = clmul_lo_and_xor(x2, k, vld1q_u64(buf.add(32) as *const u64));
      x2 = clmul_hi_and_xor(x2, k, y2);
      let y3 = clmul_lo_and_xor(x3, k, vld1q_u64(buf.add(48) as *const u64));
      x3 = clmul_hi_and_xor(x3, k, y3);
      let y4 = clmul_lo_and_xor(x4, k, vld1q_u64(buf.add(64) as *const u64));
      x4 = clmul_hi_and_xor(x4, k, y4);
      let y5 = clmul_lo_and_xor(x5, k, vld1q_u64(buf.add(80) as *const u64));
      x5 = clmul_hi_and_xor(x5, k, y5);
      let y6 = clmul_lo_and_xor(x6, k, vld1q_u64(buf.add(96) as *const u64));
      x6 = clmul_hi_and_xor(x6, k, y6);
      let y7 = clmul_lo_and_xor(x7, k, vld1q_u64(buf.add(112) as *const u64));
      x7 = clmul_hi_and_xor(x7, k, y7);
      let y8 = clmul_lo_and_xor(x8, k, vld1q_u64(buf.add(128) as *const u64));
      x8 = clmul_hi_and_xor(x8, k, y8);
      let y9 = clmul_lo_and_xor(x9, k, vld1q_u64(buf.add(144) as *const u64));
      x9 = clmul_hi_and_xor(x9, k, y9);
      let y10 = clmul_lo_and_xor(x10, k, vld1q_u64(buf.add(160) as *const u64));
      x10 = clmul_hi_and_xor(x10, k, y10);
      let y11 = clmul_lo_and_xor(x11, k, vld1q_u64(buf.add(176) as *const u64));
      x11 = clmul_hi_and_xor(x11, k, y11);
      buf = buf.add(192);
    }

    let k_vals: [u64; 2] = [0xf20c0dfe, 0x493c7d27];
    k = vld1q_u64(k_vals.as_ptr());

    let y0 = clmul_lo_and_xor(x0, k, x1);
    x0 = clmul_hi_and_xor(x0, k, y0);
    let y2 = clmul_lo_and_xor(x2, k, x3);
    x2 = clmul_hi_and_xor(x2, k, y2);
    let y4 = clmul_lo_and_xor(x4, k, x5);
    x4 = clmul_hi_and_xor(x4, k, y4);
    let y6 = clmul_lo_and_xor(x6, k, x7);
    x6 = clmul_hi_and_xor(x6, k, y6);
    let y8 = clmul_lo_and_xor(x8, k, x9);
    x8 = clmul_hi_and_xor(x8, k, y8);
    let y10 = clmul_lo_and_xor(x10, k, x11);
    x10 = clmul_hi_and_xor(x10, k, y10);

    let k_vals: [u64; 2] = [0x3da6d0cb, 0xba4fc28e];
    k = vld1q_u64(k_vals.as_ptr());

    let y0 = clmul_lo_and_xor(x0, k, x2);
    x0 = clmul_hi_and_xor(x0, k, y0);
    let y4 = clmul_lo_and_xor(x4, k, x6);
    x4 = clmul_hi_and_xor(x4, k, y4);
    let y8 = clmul_lo_and_xor(x8, k, x10);
    x8 = clmul_hi_and_xor(x8, k, y8);

    let k_vals: [u64; 2] = [0x740eef02, 0x9e4addf8];
    k = vld1q_u64(k_vals.as_ptr());

    let y0 = clmul_lo_and_xor(x0, k, x4);
    x0 = clmul_hi_and_xor(x0, k, y0);
    x4 = x8;
    let y0 = clmul_lo_and_xor(x0, k, x4);
    x0 = clmul_hi_and_xor(x0, k, y0);

    crc0 = __crc32cd(0, vgetq_lane_u64(x0, 0));
    crc0 = __crc32cd(crc0, vgetq_lane_u64(x0, 1));
    len = end.offset_from(buf) as usize;
  }

  if len >= 16 {
    let mut x0 = vld1q_u64(buf as *const u64);

    let k_vals: [u64; 2] = [0xf20c0dfe, 0x493c7d27];
    let k = vld1q_u64(k_vals.as_ptr());

    let crc_vec = vsetq_lane_u64(crc0 as u64, vmovq_n_u64(0), 0);
    x0 = veorq_u64(crc_vec, x0);
    buf = buf.add(16);
    len = len.strict_sub(16);

    while len >= 16 {
      let y0 = clmul_lo_and_xor(x0, k, vld1q_u64(buf as *const u64));
      x0 = clmul_hi_and_xor(x0, k, y0);
      buf = buf.add(16);
      len = len.strict_sub(16);
    }

    crc0 = __crc32cd(0, vgetq_lane_u64(x0, 0));
    crc0 = __crc32cd(crc0, vgetq_lane_u64(x0, 1));
  }

  while len >= 8 {
    crc0 = __crc32cd(crc0, ptr::read_unaligned(buf as *const u64));
    buf = buf.add(8);
    len = len.strict_sub(8);
  }

  while len > 0 {
    crc0 = __crc32cb(crc0, *buf);
    buf = buf.add(1);
    len = len.strict_sub(1);
  }

  crc0
}

/// Safe wrapper for CRC-32C fusion kernel (CRC+PMULL v12e_v1).
#[inline]
pub fn crc32c_iscsi_pmull_v12e_v1_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies CRC + PMULL before selecting this kernel.
  unsafe { crc32c_iscsi_pmull_v12e_v1(crc, data.as_ptr(), data.len()) }
}

// ─────────────────────────────────────────────────────────────────────────────
// Fusion: CRC-32 (ISO-HDLC / IEEE) - PMULL v12e_v1
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
#[target_feature(enable = "crc,aes")]
unsafe fn crc32_iso_hdlc_pmull_v12e_v1(mut crc0: u32, mut buf: *const u8, mut len: usize) -> u32 {
  while len > 0 && (buf as usize & 7) != 0 {
    crc0 = __crc32b(crc0, *buf);
    buf = buf.add(1);
    len = len.strict_sub(1);
  }

  if (buf as usize & 8) != 0 && len >= 8 {
    crc0 = __crc32d(crc0, ptr::read_unaligned(buf as *const u64));
    buf = buf.add(8);
    len = len.strict_sub(8);
  }

  if len >= 192 {
    let end = buf.add(len);
    let limit = buf.add(len.strict_sub(192));

    let mut x0 = vld1q_u64(buf as *const u64);
    let mut x1 = vld1q_u64(buf.add(16) as *const u64);
    let mut x2 = vld1q_u64(buf.add(32) as *const u64);
    let mut x3 = vld1q_u64(buf.add(48) as *const u64);
    let mut x4 = vld1q_u64(buf.add(64) as *const u64);
    let mut x5 = vld1q_u64(buf.add(80) as *const u64);
    let mut x6 = vld1q_u64(buf.add(96) as *const u64);
    let mut x7 = vld1q_u64(buf.add(112) as *const u64);
    let mut x8 = vld1q_u64(buf.add(128) as *const u64);
    let mut x9 = vld1q_u64(buf.add(144) as *const u64);
    let mut x10 = vld1q_u64(buf.add(160) as *const u64);
    let mut x11 = vld1q_u64(buf.add(176) as *const u64);

    let k_vals: [u64; 2] = [0x596c8d81, 0xf5e48c85];
    let mut k = vld1q_u64(k_vals.as_ptr());

    let crc_vec = vsetq_lane_u64(crc0 as u64, vmovq_n_u64(0), 0);
    x0 = veorq_u64(crc_vec, x0);
    buf = buf.add(192);

    while buf <= limit {
      let y0 = clmul_lo_and_xor(x0, k, vld1q_u64(buf as *const u64));
      x0 = clmul_hi_and_xor(x0, k, y0);
      let y1 = clmul_lo_and_xor(x1, k, vld1q_u64(buf.add(16) as *const u64));
      x1 = clmul_hi_and_xor(x1, k, y1);
      let y2 = clmul_lo_and_xor(x2, k, vld1q_u64(buf.add(32) as *const u64));
      x2 = clmul_hi_and_xor(x2, k, y2);
      let y3 = clmul_lo_and_xor(x3, k, vld1q_u64(buf.add(48) as *const u64));
      x3 = clmul_hi_and_xor(x3, k, y3);
      let y4 = clmul_lo_and_xor(x4, k, vld1q_u64(buf.add(64) as *const u64));
      x4 = clmul_hi_and_xor(x4, k, y4);
      let y5 = clmul_lo_and_xor(x5, k, vld1q_u64(buf.add(80) as *const u64));
      x5 = clmul_hi_and_xor(x5, k, y5);
      let y6 = clmul_lo_and_xor(x6, k, vld1q_u64(buf.add(96) as *const u64));
      x6 = clmul_hi_and_xor(x6, k, y6);
      let y7 = clmul_lo_and_xor(x7, k, vld1q_u64(buf.add(112) as *const u64));
      x7 = clmul_hi_and_xor(x7, k, y7);
      let y8 = clmul_lo_and_xor(x8, k, vld1q_u64(buf.add(128) as *const u64));
      x8 = clmul_hi_and_xor(x8, k, y8);
      let y9 = clmul_lo_and_xor(x9, k, vld1q_u64(buf.add(144) as *const u64));
      x9 = clmul_hi_and_xor(x9, k, y9);
      let y10 = clmul_lo_and_xor(x10, k, vld1q_u64(buf.add(160) as *const u64));
      x10 = clmul_hi_and_xor(x10, k, y10);
      let y11 = clmul_lo_and_xor(x11, k, vld1q_u64(buf.add(176) as *const u64));
      x11 = clmul_hi_and_xor(x11, k, y11);
      buf = buf.add(192);
    }

    let k_vals: [u64; 2] = [0xae689191, 0xccaa009e];
    k = vld1q_u64(k_vals.as_ptr());

    let y0 = clmul_lo_and_xor(x0, k, x1);
    x0 = clmul_hi_and_xor(x0, k, y0);
    let y2 = clmul_lo_and_xor(x2, k, x3);
    x2 = clmul_hi_and_xor(x2, k, y2);
    let y4 = clmul_lo_and_xor(x4, k, x5);
    x4 = clmul_hi_and_xor(x4, k, y4);
    let y6 = clmul_lo_and_xor(x6, k, x7);
    x6 = clmul_hi_and_xor(x6, k, y6);
    let y8 = clmul_lo_and_xor(x8, k, x9);
    x8 = clmul_hi_and_xor(x8, k, y8);
    let y10 = clmul_lo_and_xor(x10, k, x11);
    x10 = clmul_hi_and_xor(x10, k, y10);

    let k_vals: [u64; 2] = [0xf1da05aa, 0x81256527];
    k = vld1q_u64(k_vals.as_ptr());

    let y0 = clmul_lo_and_xor(x0, k, x2);
    x0 = clmul_hi_and_xor(x0, k, y0);
    let y4 = clmul_lo_and_xor(x4, k, x6);
    x4 = clmul_hi_and_xor(x4, k, y4);
    let y8 = clmul_lo_and_xor(x8, k, x10);
    x8 = clmul_hi_and_xor(x8, k, y8);

    let k_vals: [u64; 2] = [0x8f352d95, 0x1d9513d7];
    k = vld1q_u64(k_vals.as_ptr());

    let y0 = clmul_lo_and_xor(x0, k, x4);
    x0 = clmul_hi_and_xor(x0, k, y0);
    x4 = x8;
    let y0 = clmul_lo_and_xor(x0, k, x4);
    x0 = clmul_hi_and_xor(x0, k, y0);

    crc0 = __crc32d(0, vgetq_lane_u64(x0, 0));
    crc0 = __crc32d(crc0, vgetq_lane_u64(x0, 1));
    len = end.offset_from(buf) as usize;
  }

  if len >= 16 {
    let mut x0 = vld1q_u64(buf as *const u64);

    let k_vals: [u64; 2] = [0xae689191, 0xccaa009e];
    let k = vld1q_u64(k_vals.as_ptr());

    let crc_vec = vsetq_lane_u64(crc0 as u64, vmovq_n_u64(0), 0);
    x0 = veorq_u64(crc_vec, x0);
    buf = buf.add(16);
    len = len.strict_sub(16);

    while len >= 16 {
      let y0 = clmul_lo_and_xor(x0, k, vld1q_u64(buf as *const u64));
      x0 = clmul_hi_and_xor(x0, k, y0);
      buf = buf.add(16);
      len = len.strict_sub(16);
    }

    crc0 = __crc32d(0, vgetq_lane_u64(x0, 0));
    crc0 = __crc32d(crc0, vgetq_lane_u64(x0, 1));
  }

  while len >= 8 {
    crc0 = __crc32d(crc0, ptr::read_unaligned(buf as *const u64));
    buf = buf.add(8);
    len = len.strict_sub(8);
  }

  while len > 0 {
    crc0 = __crc32b(crc0, *buf);
    buf = buf.add(1);
    len = len.strict_sub(1);
  }

  crc0
}

/// Safe wrapper for CRC-32 fusion kernel (CRC+PMULL v12e_v1).
#[inline]
pub fn crc32_iso_hdlc_pmull_v12e_v1_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies CRC + PMULL before selecting this kernel.
  unsafe { crc32_iso_hdlc_pmull_v12e_v1(crc, data.as_ptr(), data.len()) }
}

// ─────────────────────────────────────────────────────────────────────────────
// Fusion: PMULL v9s3x2e_s3 (no EOR3)
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
unsafe fn xor3_u64x2(a: uint64x2_t, b: uint64x2_t, c: uint64x2_t) -> uint64x2_t {
  veorq_u64(veorq_u64(a, b), c)
}

#[inline]
#[target_feature(enable = "crc,aes")]
unsafe fn crc32c_iscsi_pmull_v9s3x2e_s3(mut crc0: u32, mut buf: *const u8, mut len: usize) -> u32 {
  // Non-EOR3 equivalent of fast-crc32 neon_eor3 CRC32C (v9s3x2e_s3).
  while len > 0 && (buf as usize & 7) != 0 {
    crc0 = __crc32cb(crc0, *buf);
    buf = buf.add(1);
    len = len.strict_sub(1);
  }

  if (buf as usize & 8) != 0 && len >= 8 {
    crc0 = __crc32cd(crc0, ptr::read_unaligned(buf as *const u64));
    buf = buf.add(8);
    len = len.strict_sub(8);
  }

  if len >= 192 {
    let end = buf.add(len);
    let blk = len.strict_div(192);
    let klen = blk.strict_mul(16);
    let mut buf2 = buf.add(klen.strict_mul(3));
    let limit = buf.add(klen).sub(32);
    let mut crc1 = 0u32;
    let mut crc2 = 0u32;

    let mut x0 = vld1q_u64(buf2 as *const u64);
    let mut x1 = vld1q_u64(buf2.add(16) as *const u64);
    let mut x2 = vld1q_u64(buf2.add(32) as *const u64);
    let mut x3 = vld1q_u64(buf2.add(48) as *const u64);
    let mut x4 = vld1q_u64(buf2.add(64) as *const u64);
    let mut x5 = vld1q_u64(buf2.add(80) as *const u64);
    let mut x6 = vld1q_u64(buf2.add(96) as *const u64);
    let mut x7 = vld1q_u64(buf2.add(112) as *const u64);
    let mut x8 = vld1q_u64(buf2.add(128) as *const u64);

    let k_vals: [u64; 2] = [0x7e908048, 0xc96cfdc0];
    let mut k = vld1q_u64(k_vals.as_ptr());
    buf2 = buf2.add(144);

    while buf <= limit {
      let y0 = clmul_lo(x0, k);
      x0 = clmul_hi(x0, k);
      let y1 = clmul_lo(x1, k);
      x1 = clmul_hi(x1, k);
      let y2 = clmul_lo(x2, k);
      x2 = clmul_hi(x2, k);
      let y3 = clmul_lo(x3, k);
      x3 = clmul_hi(x3, k);
      let y4 = clmul_lo(x4, k);
      x4 = clmul_hi(x4, k);
      let y5 = clmul_lo(x5, k);
      x5 = clmul_hi(x5, k);
      let y6 = clmul_lo(x6, k);
      x6 = clmul_hi(x6, k);
      let y7 = clmul_lo(x7, k);
      x7 = clmul_hi(x7, k);
      let y8 = clmul_lo(x8, k);
      x8 = clmul_hi(x8, k);

      x0 = xor3_u64x2(x0, y0, vld1q_u64(buf2 as *const u64));
      x1 = xor3_u64x2(x1, y1, vld1q_u64(buf2.add(16) as *const u64));
      x2 = xor3_u64x2(x2, y2, vld1q_u64(buf2.add(32) as *const u64));
      x3 = xor3_u64x2(x3, y3, vld1q_u64(buf2.add(48) as *const u64));
      x4 = xor3_u64x2(x4, y4, vld1q_u64(buf2.add(64) as *const u64));
      x5 = xor3_u64x2(x5, y5, vld1q_u64(buf2.add(80) as *const u64));
      x6 = xor3_u64x2(x6, y6, vld1q_u64(buf2.add(96) as *const u64));
      x7 = xor3_u64x2(x7, y7, vld1q_u64(buf2.add(112) as *const u64));
      x8 = xor3_u64x2(x8, y8, vld1q_u64(buf2.add(128) as *const u64));

      crc0 = __crc32cd(crc0, ptr::read_unaligned(buf as *const u64));
      crc1 = __crc32cd(crc1, ptr::read_unaligned(buf.add(klen) as *const u64));
      crc2 = __crc32cd(crc2, ptr::read_unaligned(buf.add(klen.strict_mul(2)) as *const u64));
      crc0 = __crc32cd(crc0, ptr::read_unaligned(buf.add(8) as *const u64));
      crc1 = __crc32cd(crc1, ptr::read_unaligned(buf.add(klen.strict_add(8)) as *const u64));
      crc2 = __crc32cd(
        crc2,
        ptr::read_unaligned(buf.add(klen.strict_mul(2).strict_add(8)) as *const u64),
      );

      buf = buf.add(16);
      buf2 = buf2.add(144);
    }

    let k_vals: [u64; 2] = [0xf20c0dfe, 0x493c7d27];
    k = vld1q_u64(k_vals.as_ptr());

    let y0 = clmul_lo(x0, k);
    x0 = clmul_hi(x0, k);
    x0 = xor3_u64x2(x0, y0, x1);
    x1 = x2;
    x2 = x3;
    x3 = x4;
    x4 = x5;
    x5 = x6;
    x6 = x7;
    x7 = x8;

    let y0 = clmul_lo(x0, k);
    x0 = clmul_hi(x0, k);
    let y2 = clmul_lo(x2, k);
    x2 = clmul_hi(x2, k);
    let y4 = clmul_lo(x4, k);
    x4 = clmul_hi(x4, k);
    let y6 = clmul_lo(x6, k);
    x6 = clmul_hi(x6, k);

    x0 = xor3_u64x2(x0, y0, x1);
    x2 = xor3_u64x2(x2, y2, x3);
    x4 = xor3_u64x2(x4, y4, x5);
    x6 = xor3_u64x2(x6, y6, x7);

    let k_vals: [u64; 2] = [0x3da6d0cb, 0xba4fc28e];
    k = vld1q_u64(k_vals.as_ptr());

    let y0 = clmul_lo(x0, k);
    x0 = clmul_hi(x0, k);
    let y4 = clmul_lo(x4, k);
    x4 = clmul_hi(x4, k);

    x0 = xor3_u64x2(x0, y0, x2);
    x4 = xor3_u64x2(x4, y4, x6);

    let k_vals: [u64; 2] = [0x740eef02, 0x9e4addf8];
    k = vld1q_u64(k_vals.as_ptr());

    let y0 = clmul_lo(x0, k);
    x0 = clmul_hi(x0, k);
    x0 = xor3_u64x2(x0, y0, x4);

    crc0 = __crc32cd(crc0, ptr::read_unaligned(buf as *const u64));
    crc1 = __crc32cd(crc1, ptr::read_unaligned(buf.add(klen) as *const u64));
    crc2 = __crc32cd(crc2, ptr::read_unaligned(buf.add(klen.strict_mul(2)) as *const u64));
    crc0 = __crc32cd(crc0, ptr::read_unaligned(buf.add(8) as *const u64));
    crc1 = __crc32cd(crc1, ptr::read_unaligned(buf.add(klen.strict_add(8)) as *const u64));
    crc2 = __crc32cd(
      crc2,
      ptr::read_unaligned(buf.add(klen.strict_mul(2).strict_add(8)) as *const u64),
    );

    let vc0 = crc_shift_iscsi(crc0, klen.strict_mul(2).strict_add(blk.strict_mul(144)));
    let vc1 = crc_shift_iscsi(crc1, klen.strict_add(blk.strict_mul(144)));
    let vc2 = crc_shift_iscsi(crc2, blk.strict_mul(144));
    let vc = vgetq_lane_u64(xor3_u64x2(vc0, vc1, vc2), 0);

    crc0 = __crc32cd(0, vgetq_lane_u64(x0, 0));
    crc0 = __crc32cd(crc0, vc ^ vgetq_lane_u64(x0, 1));

    buf = buf2;
    len = end.offset_from(buf) as usize;
  }

  if len >= 32 {
    let klen = ((len.strict_sub(8)).strict_div(24)).strict_mul(8);
    let mut crc1 = 0u32;
    let mut crc2 = 0u32;

    loop {
      crc0 = __crc32cd(crc0, ptr::read_unaligned(buf as *const u64));
      crc1 = __crc32cd(crc1, ptr::read_unaligned(buf.add(klen) as *const u64));
      crc2 = __crc32cd(crc2, ptr::read_unaligned(buf.add(klen.strict_mul(2)) as *const u64));
      buf = buf.add(8);
      len = len.strict_sub(24);
      if len < 32 {
        break;
      }
    }

    let vc0 = crc_shift_iscsi(crc0, klen.strict_mul(2).strict_add(8));
    let vc1 = crc_shift_iscsi(crc1, klen.strict_add(8));
    let vc = vgetq_lane_u64(veorq_u64(vc0, vc1), 0);

    buf = buf.add(klen.strict_mul(2));
    crc0 = crc2;
    crc0 = __crc32cd(crc0, ptr::read_unaligned(buf as *const u64) ^ vc);
    buf = buf.add(8);
    len = len.strict_sub(8);
  }

  while len >= 8 {
    crc0 = __crc32cd(crc0, ptr::read_unaligned(buf as *const u64));
    buf = buf.add(8);
    len = len.strict_sub(8);
  }

  while len > 0 {
    crc0 = __crc32cb(crc0, *buf);
    buf = buf.add(1);
    len = len.strict_sub(1);
  }

  crc0
}

/// Safe wrapper for CRC-32C fusion kernel (CRC+PMULL, no EOR3).
#[inline]
pub fn crc32c_iscsi_pmull_v9s3x2e_s3_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies CRC + PMULL before selecting this kernel.
  unsafe { crc32c_iscsi_pmull_v9s3x2e_s3(crc, data.as_ptr(), data.len()) }
}

#[inline]
#[target_feature(enable = "crc,aes")]
unsafe fn crc32_iso_hdlc_pmull_v9s3x2e_s3(mut crc0: u32, mut buf: *const u8, mut len: usize) -> u32 {
  // Non-EOR3 equivalent of fast-crc32 neon_eor3 ISO-HDLC (v9s3x2e_s3).
  while len > 0 && (buf as usize & 7) != 0 {
    crc0 = __crc32b(crc0, *buf);
    buf = buf.add(1);
    len = len.strict_sub(1);
  }

  if (buf as usize & 8) != 0 && len >= 8 {
    crc0 = __crc32d(crc0, ptr::read_unaligned(buf as *const u64));
    buf = buf.add(8);
    len = len.strict_sub(8);
  }

  if len >= 192 {
    let end = buf.add(len);
    let blk = len.strict_div(192);
    let klen = blk.strict_mul(16);
    let mut buf2 = buf.add(klen.strict_mul(3));
    let limit = buf.add(klen).sub(32);
    let mut crc1 = 0u32;
    let mut crc2 = 0u32;

    let mut x0 = vld1q_u64(buf2 as *const u64);
    let mut x1 = vld1q_u64(buf2.add(16) as *const u64);
    let mut x2 = vld1q_u64(buf2.add(32) as *const u64);
    let mut x3 = vld1q_u64(buf2.add(48) as *const u64);
    let mut x4 = vld1q_u64(buf2.add(64) as *const u64);
    let mut x5 = vld1q_u64(buf2.add(80) as *const u64);
    let mut x6 = vld1q_u64(buf2.add(96) as *const u64);
    let mut x7 = vld1q_u64(buf2.add(112) as *const u64);
    let mut x8 = vld1q_u64(buf2.add(128) as *const u64);

    let k_vals: [u64; 2] = [0x26b70c3d, 0x3f41287a];
    let mut k = vld1q_u64(k_vals.as_ptr());
    buf2 = buf2.add(144);

    while buf <= limit {
      let y0 = clmul_lo(x0, k);
      x0 = clmul_hi(x0, k);
      let y1 = clmul_lo(x1, k);
      x1 = clmul_hi(x1, k);
      let y2 = clmul_lo(x2, k);
      x2 = clmul_hi(x2, k);
      let y3 = clmul_lo(x3, k);
      x3 = clmul_hi(x3, k);
      let y4 = clmul_lo(x4, k);
      x4 = clmul_hi(x4, k);
      let y5 = clmul_lo(x5, k);
      x5 = clmul_hi(x5, k);
      let y6 = clmul_lo(x6, k);
      x6 = clmul_hi(x6, k);
      let y7 = clmul_lo(x7, k);
      x7 = clmul_hi(x7, k);
      let y8 = clmul_lo(x8, k);
      x8 = clmul_hi(x8, k);

      x0 = xor3_u64x2(x0, y0, vld1q_u64(buf2 as *const u64));
      x1 = xor3_u64x2(x1, y1, vld1q_u64(buf2.add(16) as *const u64));
      x2 = xor3_u64x2(x2, y2, vld1q_u64(buf2.add(32) as *const u64));
      x3 = xor3_u64x2(x3, y3, vld1q_u64(buf2.add(48) as *const u64));
      x4 = xor3_u64x2(x4, y4, vld1q_u64(buf2.add(64) as *const u64));
      x5 = xor3_u64x2(x5, y5, vld1q_u64(buf2.add(80) as *const u64));
      x6 = xor3_u64x2(x6, y6, vld1q_u64(buf2.add(96) as *const u64));
      x7 = xor3_u64x2(x7, y7, vld1q_u64(buf2.add(112) as *const u64));
      x8 = xor3_u64x2(x8, y8, vld1q_u64(buf2.add(128) as *const u64));

      crc0 = __crc32d(crc0, ptr::read_unaligned(buf as *const u64));
      crc1 = __crc32d(crc1, ptr::read_unaligned(buf.add(klen) as *const u64));
      crc2 = __crc32d(crc2, ptr::read_unaligned(buf.add(klen.strict_mul(2)) as *const u64));
      crc0 = __crc32d(crc0, ptr::read_unaligned(buf.add(8) as *const u64));
      crc1 = __crc32d(crc1, ptr::read_unaligned(buf.add(klen.strict_add(8)) as *const u64));
      crc2 = __crc32d(
        crc2,
        ptr::read_unaligned(buf.add(klen.strict_mul(2).strict_add(8)) as *const u64),
      );

      buf = buf.add(16);
      buf2 = buf2.add(144);
    }

    let k_vals: [u64; 2] = [0xae689191, 0xccaa009e];
    k = vld1q_u64(k_vals.as_ptr());

    let y0 = clmul_lo(x0, k);
    x0 = clmul_hi(x0, k);
    x0 = xor3_u64x2(x0, y0, x1);
    x1 = x2;
    x2 = x3;
    x3 = x4;
    x4 = x5;
    x5 = x6;
    x6 = x7;
    x7 = x8;

    let y0 = clmul_lo(x0, k);
    x0 = clmul_hi(x0, k);
    let y2 = clmul_lo(x2, k);
    x2 = clmul_hi(x2, k);
    let y4 = clmul_lo(x4, k);
    x4 = clmul_hi(x4, k);
    let y6 = clmul_lo(x6, k);
    x6 = clmul_hi(x6, k);

    x0 = xor3_u64x2(x0, y0, x1);
    x2 = xor3_u64x2(x2, y2, x3);
    x4 = xor3_u64x2(x4, y4, x5);
    x6 = xor3_u64x2(x6, y6, x7);

    let k_vals: [u64; 2] = [0xf1da05aa, 0x81256527];
    k = vld1q_u64(k_vals.as_ptr());

    let y0 = clmul_lo(x0, k);
    x0 = clmul_hi(x0, k);
    let y4 = clmul_lo(x4, k);
    x4 = clmul_hi(x4, k);

    x0 = xor3_u64x2(x0, y0, x2);
    x4 = xor3_u64x2(x4, y4, x6);

    let k_vals: [u64; 2] = [0x8f352d95, 0x1d9513d7];
    k = vld1q_u64(k_vals.as_ptr());

    let y0 = clmul_lo(x0, k);
    x0 = clmul_hi(x0, k);
    x0 = xor3_u64x2(x0, y0, x4);

    crc0 = __crc32d(crc0, ptr::read_unaligned(buf as *const u64));
    crc1 = __crc32d(crc1, ptr::read_unaligned(buf.add(klen) as *const u64));
    crc2 = __crc32d(crc2, ptr::read_unaligned(buf.add(klen.strict_mul(2)) as *const u64));
    crc0 = __crc32d(crc0, ptr::read_unaligned(buf.add(8) as *const u64));
    crc1 = __crc32d(crc1, ptr::read_unaligned(buf.add(klen.strict_add(8)) as *const u64));
    crc2 = __crc32d(
      crc2,
      ptr::read_unaligned(buf.add(klen.strict_mul(2).strict_add(8)) as *const u64),
    );

    let vc0 = crc_shift_iso_hdlc(crc0, klen.strict_mul(2).strict_add(blk.strict_mul(144)));
    let vc1 = crc_shift_iso_hdlc(crc1, klen.strict_add(blk.strict_mul(144)));
    let vc2 = crc_shift_iso_hdlc(crc2, blk.strict_mul(144));
    let vc = vgetq_lane_u64(xor3_u64x2(vc0, vc1, vc2), 0);

    crc0 = __crc32d(0, vgetq_lane_u64(x0, 0));
    crc0 = __crc32d(crc0, vc ^ vgetq_lane_u64(x0, 1));

    buf = buf2;
    len = end.offset_from(buf) as usize;
  }

  if len >= 32 {
    let klen = ((len.strict_sub(8)).strict_div(24)).strict_mul(8);
    let mut crc1 = 0u32;
    let mut crc2 = 0u32;

    loop {
      crc0 = __crc32d(crc0, ptr::read_unaligned(buf as *const u64));
      crc1 = __crc32d(crc1, ptr::read_unaligned(buf.add(klen) as *const u64));
      crc2 = __crc32d(crc2, ptr::read_unaligned(buf.add(klen.strict_mul(2)) as *const u64));
      buf = buf.add(8);
      len = len.strict_sub(24);
      if len < 32 {
        break;
      }
    }

    let vc0 = crc_shift_iso_hdlc(crc0, klen.strict_mul(2).strict_add(8));
    let vc1 = crc_shift_iso_hdlc(crc1, klen.strict_add(8));
    let vc = vgetq_lane_u64(veorq_u64(vc0, vc1), 0);

    buf = buf.add(klen.strict_mul(2));
    crc0 = crc2;
    crc0 = __crc32d(crc0, ptr::read_unaligned(buf as *const u64) ^ vc);
    buf = buf.add(8);
    len = len.strict_sub(8);
  }

  while len >= 8 {
    crc0 = __crc32d(crc0, ptr::read_unaligned(buf as *const u64));
    buf = buf.add(8);
    len = len.strict_sub(8);
  }

  while len > 0 {
    crc0 = __crc32b(crc0, *buf);
    buf = buf.add(1);
    len = len.strict_sub(1);
  }

  crc0
}

/// Safe wrapper for CRC-32 fusion kernel (CRC+PMULL, no EOR3).
#[inline]
pub fn crc32_iso_hdlc_pmull_v9s3x2e_s3_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies CRC + PMULL before selecting this kernel.
  unsafe { crc32_iso_hdlc_pmull_v9s3x2e_s3(crc, data.as_ptr(), data.len()) }
}

// ─────────────────────────────────────────────────────────────────────────────
// Fusion: EOR3 variants (CRC + PMULL + SHA3)
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
#[target_feature(enable = "crc,aes,sha3")]
unsafe fn crc32c_iscsi_pmull_eor3_v9s3x2e_s3(mut crc0: u32, mut buf: *const u8, mut len: usize) -> u32 {
  // Ported from fast-crc32 neon_eor3 CRC32C (v9s3x2e_s3).
  while len > 0 && (buf as usize & 7) != 0 {
    crc0 = __crc32cb(crc0, *buf);
    buf = buf.add(1);
    len = len.strict_sub(1);
  }

  if (buf as usize & 8) != 0 && len >= 8 {
    crc0 = __crc32cd(crc0, ptr::read_unaligned(buf as *const u64));
    buf = buf.add(8);
    len = len.strict_sub(8);
  }

  if len >= 192 {
    let end = buf.add(len);
    let blk = len / 192;
    let klen = blk.strict_mul(16);
    let mut buf2 = buf.add(klen.strict_mul(3));
    let limit = buf.add(klen).sub(32);
    let mut crc1 = 0u32;
    let mut crc2 = 0u32;

    let mut x0 = vld1q_u64(buf2 as *const u64);
    let mut x1 = vld1q_u64(buf2.add(16) as *const u64);
    let mut x2 = vld1q_u64(buf2.add(32) as *const u64);
    let mut x3 = vld1q_u64(buf2.add(48) as *const u64);
    let mut x4 = vld1q_u64(buf2.add(64) as *const u64);
    let mut x5 = vld1q_u64(buf2.add(80) as *const u64);
    let mut x6 = vld1q_u64(buf2.add(96) as *const u64);
    let mut x7 = vld1q_u64(buf2.add(112) as *const u64);
    let mut x8 = vld1q_u64(buf2.add(128) as *const u64);

    let k_vals: [u64; 2] = [0x7e908048, 0xc96cfdc0];
    let mut k = vld1q_u64(k_vals.as_ptr());
    buf2 = buf2.add(144);

    while buf <= limit {
      let y0 = clmul_lo(x0, k);
      x0 = clmul_hi(x0, k);
      let y1 = clmul_lo(x1, k);
      x1 = clmul_hi(x1, k);
      let y2 = clmul_lo(x2, k);
      x2 = clmul_hi(x2, k);
      let y3 = clmul_lo(x3, k);
      x3 = clmul_hi(x3, k);
      let y4 = clmul_lo(x4, k);
      x4 = clmul_hi(x4, k);
      let y5 = clmul_lo(x5, k);
      x5 = clmul_hi(x5, k);
      let y6 = clmul_lo(x6, k);
      x6 = clmul_hi(x6, k);
      let y7 = clmul_lo(x7, k);
      x7 = clmul_hi(x7, k);
      let y8 = clmul_lo(x8, k);
      x8 = clmul_hi(x8, k);

      x0 = veor3q_u64(x0, y0, vld1q_u64(buf2 as *const u64));
      x1 = veor3q_u64(x1, y1, vld1q_u64(buf2.add(16) as *const u64));
      x2 = veor3q_u64(x2, y2, vld1q_u64(buf2.add(32) as *const u64));
      x3 = veor3q_u64(x3, y3, vld1q_u64(buf2.add(48) as *const u64));
      x4 = veor3q_u64(x4, y4, vld1q_u64(buf2.add(64) as *const u64));
      x5 = veor3q_u64(x5, y5, vld1q_u64(buf2.add(80) as *const u64));
      x6 = veor3q_u64(x6, y6, vld1q_u64(buf2.add(96) as *const u64));
      x7 = veor3q_u64(x7, y7, vld1q_u64(buf2.add(112) as *const u64));
      x8 = veor3q_u64(x8, y8, vld1q_u64(buf2.add(128) as *const u64));

      crc0 = __crc32cd(crc0, ptr::read_unaligned(buf as *const u64));
      crc1 = __crc32cd(crc1, ptr::read_unaligned(buf.add(klen) as *const u64));
      crc2 = __crc32cd(crc2, ptr::read_unaligned(buf.add(klen.strict_mul(2)) as *const u64));
      crc0 = __crc32cd(crc0, ptr::read_unaligned(buf.add(8) as *const u64));
      crc1 = __crc32cd(crc1, ptr::read_unaligned(buf.add(klen + 8) as *const u64));
      crc2 = __crc32cd(crc2, ptr::read_unaligned(buf.add(klen.strict_mul(2) + 8) as *const u64));

      buf = buf.add(16);
      buf2 = buf2.add(144);
    }

    let k_vals: [u64; 2] = [0xf20c0dfe, 0x493c7d27];
    k = vld1q_u64(k_vals.as_ptr());

    let y0 = clmul_lo(x0, k);
    x0 = clmul_hi(x0, k);
    x0 = veor3q_u64(x0, y0, x1);
    x1 = x2;
    x2 = x3;
    x3 = x4;
    x4 = x5;
    x5 = x6;
    x6 = x7;
    x7 = x8;

    let y0 = clmul_lo(x0, k);
    x0 = clmul_hi(x0, k);
    let y2 = clmul_lo(x2, k);
    x2 = clmul_hi(x2, k);
    let y4 = clmul_lo(x4, k);
    x4 = clmul_hi(x4, k);
    let y6 = clmul_lo(x6, k);
    x6 = clmul_hi(x6, k);

    x0 = veor3q_u64(x0, y0, x1);
    x2 = veor3q_u64(x2, y2, x3);
    x4 = veor3q_u64(x4, y4, x5);
    x6 = veor3q_u64(x6, y6, x7);

    let k_vals: [u64; 2] = [0x3da6d0cb, 0xba4fc28e];
    k = vld1q_u64(k_vals.as_ptr());

    let y0 = clmul_lo(x0, k);
    x0 = clmul_hi(x0, k);
    let y4 = clmul_lo(x4, k);
    x4 = clmul_hi(x4, k);

    x0 = veor3q_u64(x0, y0, x2);
    x4 = veor3q_u64(x4, y4, x6);

    let k_vals: [u64; 2] = [0x740eef02, 0x9e4addf8];
    k = vld1q_u64(k_vals.as_ptr());

    let y0 = clmul_lo(x0, k);
    x0 = clmul_hi(x0, k);
    x0 = veor3q_u64(x0, y0, x4);

    crc0 = __crc32cd(crc0, ptr::read_unaligned(buf as *const u64));
    crc1 = __crc32cd(crc1, ptr::read_unaligned(buf.add(klen) as *const u64));
    crc2 = __crc32cd(crc2, ptr::read_unaligned(buf.add(klen.strict_mul(2)) as *const u64));
    crc0 = __crc32cd(crc0, ptr::read_unaligned(buf.add(8) as *const u64));
    crc1 = __crc32cd(crc1, ptr::read_unaligned(buf.add(klen + 8) as *const u64));
    crc2 = __crc32cd(crc2, ptr::read_unaligned(buf.add(klen.strict_mul(2) + 8) as *const u64));

    let vc0 = crc_shift_iscsi(crc0, klen.strict_mul(2) + blk.strict_mul(144));
    let vc1 = crc_shift_iscsi(crc1, klen + blk.strict_mul(144));
    let vc2 = crc_shift_iscsi(crc2, blk.strict_mul(144));
    let vc = vgetq_lane_u64(veor3q_u64(vc0, vc1, vc2), 0);

    crc0 = __crc32cd(0, vgetq_lane_u64(x0, 0));
    crc0 = __crc32cd(crc0, vc ^ vgetq_lane_u64(x0, 1));

    buf = buf2;
    len = end.offset_from(buf) as usize;
  }

  if len >= 32 {
    let klen = ((len.strict_sub(8)) / 24).strict_mul(8);
    let mut crc1 = 0u32;
    let mut crc2 = 0u32;

    loop {
      crc0 = __crc32cd(crc0, ptr::read_unaligned(buf as *const u64));
      crc1 = __crc32cd(crc1, ptr::read_unaligned(buf.add(klen) as *const u64));
      crc2 = __crc32cd(crc2, ptr::read_unaligned(buf.add(klen.strict_mul(2)) as *const u64));
      buf = buf.add(8);
      len = len.strict_sub(24);
      if len < 32 {
        break;
      }
    }

    let vc0 = crc_shift_iscsi(crc0, klen.strict_mul(2) + 8);
    let vc1 = crc_shift_iscsi(crc1, klen + 8);
    let vc = vgetq_lane_u64(veorq_u64(vc0, vc1), 0);

    buf = buf.add(klen.strict_mul(2));
    crc0 = crc2;
    crc0 = __crc32cd(crc0, ptr::read_unaligned(buf as *const u64) ^ vc);
    buf = buf.add(8);
    len = len.strict_sub(8);
  }

  while len >= 8 {
    crc0 = __crc32cd(crc0, ptr::read_unaligned(buf as *const u64));
    buf = buf.add(8);
    len = len.strict_sub(8);
  }

  while len > 0 {
    crc0 = __crc32cb(crc0, *buf);
    buf = buf.add(1);
    len = len.strict_sub(1);
  }

  crc0
}

#[inline]
#[target_feature(enable = "aes")]
unsafe fn crc_shift_iscsi(crc: u32, nbytes: usize) -> uint64x2_t {
  clmul_scalar(crc, xnmodp_crc32_iscsi((nbytes.strict_mul(8).strict_sub(33)) as u64))
}

#[inline]
#[target_feature(enable = "crc,aes")]
unsafe fn xnmodp_crc32_iscsi(mut n: u64) -> u32 {
  let mut stack = !1u64;
  let mut acc: u32;
  let mut low: u32;

  while n > 191 {
    stack = (stack << 1) + (n & 1);
    n = (n >> 1).strict_sub(16);
  }
  stack = !stack;
  acc = 0x8000_0000u32 >> (n & 31);
  n >>= 5;

  while n > 0 {
    acc = __crc32cw(acc, 0);
    n = n.strict_sub(1);
  }

  while {
    low = (stack & 1) as u32;
    stack >>= 1;
    stack != 0
  } {
    let x = vreinterpret_p8_u64(vmov_n_u64(acc as u64));
    let squared = vmull_p8(x, x);
    let y = vgetq_lane_u64(vreinterpretq_u64_p16(squared), 0);
    acc = __crc32cd(0, y << low);
  }

  acc
}

/// Safe wrapper for CRC-32C fusion kernel (CRC+PMULL+EOR3 v9s3x2e_s3).
#[inline]
pub fn crc32c_iscsi_pmull_eor3_v9s3x2e_s3_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies CRC + PMULL + SHA3/EOR3 before selecting this kernel.
  unsafe { crc32c_iscsi_pmull_eor3_v9s3x2e_s3(crc, data.as_ptr(), data.len()) }
}

#[inline]
#[target_feature(enable = "crc,aes,sha3")]
unsafe fn crc32_iso_hdlc_pmull_eor3_v9s3x2e_s3(mut crc0: u32, mut buf: *const u8, mut len: usize) -> u32 {
  // Ported from fast-crc32 neon_eor3 ISO-HDLC (v9s3x2e_s3).
  while len > 0 && (buf as usize & 7) != 0 {
    crc0 = __crc32b(crc0, *buf);
    buf = buf.add(1);
    len = len.strict_sub(1);
  }

  if (buf as usize & 8) != 0 && len >= 8 {
    crc0 = __crc32d(crc0, ptr::read_unaligned(buf as *const u64));
    buf = buf.add(8);
    len = len.strict_sub(8);
  }

  if len >= 192 {
    let end = buf.add(len);
    let blk = len / 192;
    let klen = blk.strict_mul(16);
    let mut buf2 = buf.add(klen.strict_mul(3));
    let limit = buf.add(klen).sub(32);
    let mut crc1 = 0u32;
    let mut crc2 = 0u32;

    let mut x0 = vld1q_u64(buf2 as *const u64);
    let mut x1 = vld1q_u64(buf2.add(16) as *const u64);
    let mut x2 = vld1q_u64(buf2.add(32) as *const u64);
    let mut x3 = vld1q_u64(buf2.add(48) as *const u64);
    let mut x4 = vld1q_u64(buf2.add(64) as *const u64);
    let mut x5 = vld1q_u64(buf2.add(80) as *const u64);
    let mut x6 = vld1q_u64(buf2.add(96) as *const u64);
    let mut x7 = vld1q_u64(buf2.add(112) as *const u64);
    let mut x8 = vld1q_u64(buf2.add(128) as *const u64);

    let k_vals: [u64; 2] = [0x26b70c3d, 0x3f41287a];
    let mut k = vld1q_u64(k_vals.as_ptr());
    buf2 = buf2.add(144);

    while buf <= limit {
      let y0 = clmul_lo(x0, k);
      x0 = clmul_hi(x0, k);
      let y1 = clmul_lo(x1, k);
      x1 = clmul_hi(x1, k);
      let y2 = clmul_lo(x2, k);
      x2 = clmul_hi(x2, k);
      let y3 = clmul_lo(x3, k);
      x3 = clmul_hi(x3, k);
      let y4 = clmul_lo(x4, k);
      x4 = clmul_hi(x4, k);
      let y5 = clmul_lo(x5, k);
      x5 = clmul_hi(x5, k);
      let y6 = clmul_lo(x6, k);
      x6 = clmul_hi(x6, k);
      let y7 = clmul_lo(x7, k);
      x7 = clmul_hi(x7, k);
      let y8 = clmul_lo(x8, k);
      x8 = clmul_hi(x8, k);

      x0 = veor3q_u64(x0, y0, vld1q_u64(buf2 as *const u64));
      x1 = veor3q_u64(x1, y1, vld1q_u64(buf2.add(16) as *const u64));
      x2 = veor3q_u64(x2, y2, vld1q_u64(buf2.add(32) as *const u64));
      x3 = veor3q_u64(x3, y3, vld1q_u64(buf2.add(48) as *const u64));
      x4 = veor3q_u64(x4, y4, vld1q_u64(buf2.add(64) as *const u64));
      x5 = veor3q_u64(x5, y5, vld1q_u64(buf2.add(80) as *const u64));
      x6 = veor3q_u64(x6, y6, vld1q_u64(buf2.add(96) as *const u64));
      x7 = veor3q_u64(x7, y7, vld1q_u64(buf2.add(112) as *const u64));
      x8 = veor3q_u64(x8, y8, vld1q_u64(buf2.add(128) as *const u64));

      crc0 = __crc32d(crc0, ptr::read_unaligned(buf as *const u64));
      crc1 = __crc32d(crc1, ptr::read_unaligned(buf.add(klen) as *const u64));
      crc2 = __crc32d(crc2, ptr::read_unaligned(buf.add(klen.strict_mul(2)) as *const u64));
      crc0 = __crc32d(crc0, ptr::read_unaligned(buf.add(8) as *const u64));
      crc1 = __crc32d(crc1, ptr::read_unaligned(buf.add(klen + 8) as *const u64));
      crc2 = __crc32d(crc2, ptr::read_unaligned(buf.add(klen.strict_mul(2) + 8) as *const u64));

      buf = buf.add(16);
      buf2 = buf2.add(144);
    }

    let k_vals: [u64; 2] = [0xae689191, 0xccaa009e];
    k = vld1q_u64(k_vals.as_ptr());

    let y0 = clmul_lo(x0, k);
    x0 = clmul_hi(x0, k);
    x0 = veor3q_u64(x0, y0, x1);
    x1 = x2;
    x2 = x3;
    x3 = x4;
    x4 = x5;
    x5 = x6;
    x6 = x7;
    x7 = x8;

    let y0 = clmul_lo(x0, k);
    x0 = clmul_hi(x0, k);
    let y2 = clmul_lo(x2, k);
    x2 = clmul_hi(x2, k);
    let y4 = clmul_lo(x4, k);
    x4 = clmul_hi(x4, k);
    let y6 = clmul_lo(x6, k);
    x6 = clmul_hi(x6, k);

    x0 = veor3q_u64(x0, y0, x1);
    x2 = veor3q_u64(x2, y2, x3);
    x4 = veor3q_u64(x4, y4, x5);
    x6 = veor3q_u64(x6, y6, x7);

    let k_vals: [u64; 2] = [0xf1da05aa, 0x81256527];
    k = vld1q_u64(k_vals.as_ptr());

    let y0 = clmul_lo(x0, k);
    x0 = clmul_hi(x0, k);
    let y4 = clmul_lo(x4, k);
    x4 = clmul_hi(x4, k);

    x0 = veor3q_u64(x0, y0, x2);
    x4 = veor3q_u64(x4, y4, x6);

    let k_vals: [u64; 2] = [0x8f352d95, 0x1d9513d7];
    k = vld1q_u64(k_vals.as_ptr());

    let y0 = clmul_lo(x0, k);
    x0 = clmul_hi(x0, k);
    x0 = veor3q_u64(x0, y0, x4);

    crc0 = __crc32d(crc0, ptr::read_unaligned(buf as *const u64));
    crc1 = __crc32d(crc1, ptr::read_unaligned(buf.add(klen) as *const u64));
    crc2 = __crc32d(crc2, ptr::read_unaligned(buf.add(klen.strict_mul(2)) as *const u64));
    crc0 = __crc32d(crc0, ptr::read_unaligned(buf.add(8) as *const u64));
    crc1 = __crc32d(crc1, ptr::read_unaligned(buf.add(klen + 8) as *const u64));
    crc2 = __crc32d(crc2, ptr::read_unaligned(buf.add(klen.strict_mul(2) + 8) as *const u64));

    let vc0 = crc_shift_iso_hdlc(crc0, klen.strict_mul(2) + blk.strict_mul(144));
    let vc1 = crc_shift_iso_hdlc(crc1, klen + blk.strict_mul(144));
    let vc2 = crc_shift_iso_hdlc(crc2, blk.strict_mul(144));
    let vc = vgetq_lane_u64(veor3q_u64(vc0, vc1, vc2), 0);

    crc0 = __crc32d(0, vgetq_lane_u64(x0, 0));
    crc0 = __crc32d(crc0, vc ^ vgetq_lane_u64(x0, 1));

    buf = buf2;
    len = end.offset_from(buf) as usize;
  }

  if len >= 32 {
    let klen = ((len.strict_sub(8)) / 24).strict_mul(8);
    let mut crc1 = 0u32;
    let mut crc2 = 0u32;

    loop {
      crc0 = __crc32d(crc0, ptr::read_unaligned(buf as *const u64));
      crc1 = __crc32d(crc1, ptr::read_unaligned(buf.add(klen) as *const u64));
      crc2 = __crc32d(crc2, ptr::read_unaligned(buf.add(klen.strict_mul(2)) as *const u64));
      buf = buf.add(8);
      len = len.strict_sub(24);
      if len < 32 {
        break;
      }
    }

    let vc0 = crc_shift_iso_hdlc(crc0, klen.strict_mul(2) + 8);
    let vc1 = crc_shift_iso_hdlc(crc1, klen + 8);
    let vc = vgetq_lane_u64(veorq_u64(vc0, vc1), 0);

    buf = buf.add(klen.strict_mul(2));
    crc0 = crc2;
    crc0 = __crc32d(crc0, ptr::read_unaligned(buf as *const u64) ^ vc);
    buf = buf.add(8);
    len = len.strict_sub(8);
  }

  while len >= 8 {
    crc0 = __crc32d(crc0, ptr::read_unaligned(buf as *const u64));
    buf = buf.add(8);
    len = len.strict_sub(8);
  }

  while len > 0 {
    crc0 = __crc32b(crc0, *buf);
    buf = buf.add(1);
    len = len.strict_sub(1);
  }

  crc0
}

#[inline]
#[target_feature(enable = "aes")]
unsafe fn crc_shift_iso_hdlc(crc: u32, nbytes: usize) -> uint64x2_t {
  clmul_scalar(crc, xnmodp_iso_hdlc((nbytes.strict_mul(8).strict_sub(33)) as u64))
}

#[inline]
#[target_feature(enable = "crc,aes")]
unsafe fn xnmodp_iso_hdlc(mut n: u64) -> u32 {
  let mut stack = !1u64;
  let mut acc: u32;
  let mut low: u32;

  while n > 191 {
    stack = (stack << 1) + (n & 1);
    n = (n >> 1).strict_sub(16);
  }
  stack = !stack;
  acc = 0x8000_0000u32 >> (n & 31);
  n >>= 5;

  while n > 0 {
    acc = __crc32w(acc, 0);
    n = n.strict_sub(1);
  }

  while {
    low = (stack & 1) as u32;
    stack >>= 1;
    stack != 0
  } {
    let x = vreinterpret_p8_u64(vmov_n_u64(acc as u64));
    let squared = vmull_p8(x, x);
    let y = vgetq_lane_u64(vreinterpretq_u64_p16(squared), 0);
    acc = __crc32d(0, y << low);
  }

  acc
}

/// Safe wrapper for CRC-32 fusion kernel (CRC+PMULL+EOR3 v9s3x2e_s3).
#[inline]
pub fn crc32_iso_hdlc_pmull_eor3_v9s3x2e_s3_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies CRC + PMULL + SHA3/EOR3 before selecting this kernel.
  unsafe { crc32_iso_hdlc_pmull_eor3_v9s3x2e_s3(crc, data.as_ptr(), data.len()) }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
  extern crate std;

  use alloc::vec::Vec;

  use super::*;

  const LENS: &[usize] = &[0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 1024, 4096];

  const SMALL_LENS: &[usize] = &[0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 127];

  fn make_data(len: usize) -> Vec<u8> {
    (0..len)
      .map(|i| (i as u8).wrapping_mul(31).wrapping_add((i >> 8) as u8))
      .collect()
  }

  fn assert_crc32_kernel(name: &str, kernel: fn(u32, &[u8]) -> u32, lens: &[usize]) {
    for &len in lens {
      let data = make_data(len);
      let expected = super::super::portable::crc32_slice16_ieee(!0, &data) ^ !0;
      let got = kernel(!0, &data) ^ !0;
      assert_eq!(got, expected, "{name} len={len}");
    }
  }

  fn assert_crc32c_kernel(name: &str, kernel: fn(u32, &[u8]) -> u32, lens: &[usize]) {
    for &len in lens {
      let data = make_data(len);
      let expected = super::super::portable::crc32c_slice16(!0, &data) ^ !0;
      let got = kernel(!0, &data) ^ !0;
      assert_eq!(got, expected, "{name} len={len}");
    }
  }

  #[test]
  fn test_crc32_hwcrc_matches_portable_various_lengths() {
    let caps = platform::caps();
    if !caps.has(platform::caps::aarch64::CRC_READY) {
      return;
    }

    assert_crc32_kernel("crc32/hwcrc", crc32_armv8_safe, LENS);
    assert_crc32_kernel("crc32/hwcrc-2way", crc32_armv8_2way_safe, LENS);
    assert_crc32_kernel("crc32/hwcrc-3way", crc32_armv8_3way_safe, LENS);

    assert_crc32c_kernel("crc32c/hwcrc", crc32c_armv8_safe, LENS);
    assert_crc32c_kernel("crc32c/hwcrc-2way", crc32c_armv8_2way_safe, LENS);
    assert_crc32c_kernel("crc32c/hwcrc-3way", crc32c_armv8_3way_safe, LENS);
  }

  #[test]
  fn test_crc32_pmull_v9s3x2e_s3_matches_portable_various_lengths() {
    let caps = platform::caps();
    if !(caps.has(platform::caps::aarch64::CRC_READY) && caps.has(platform::caps::aarch64::PMULL_READY)) {
      return;
    }

    assert_crc32_kernel("crc32/pmull-v9s3x2e-s3", crc32_iso_hdlc_pmull_v9s3x2e_s3_safe, LENS);
    assert_crc32_kernel("crc32/pmull-v9s3x2e-s3-2way", crc32_iso_hdlc_pmull_2way_safe, LENS);
    assert_crc32_kernel("crc32/pmull-v9s3x2e-s3-3way", crc32_iso_hdlc_pmull_3way_safe, LENS);
    assert_crc32_kernel("crc32/pmull-small", crc32_iso_hdlc_pmull_small_safe, SMALL_LENS);

    assert_crc32c_kernel("crc32c/pmull-v9s3x2e-s3", crc32c_iscsi_pmull_v9s3x2e_s3_safe, LENS);
    assert_crc32c_kernel("crc32c/pmull-v9s3x2e-s3-2way", crc32c_iscsi_pmull_2way_safe, LENS);
    assert_crc32c_kernel("crc32c/pmull-v9s3x2e-s3-3way", crc32c_iscsi_pmull_3way_safe, LENS);
    assert_crc32c_kernel("crc32c/pmull-small", crc32c_iscsi_pmull_small_safe, SMALL_LENS);
  }

  #[test]
  fn test_crc32_pmull_v12e_v1_matches_portable_various_lengths() {
    let caps = platform::caps();
    if !(caps.has(platform::caps::aarch64::CRC_READY) && caps.has(platform::caps::aarch64::PMULL_READY)) {
      return;
    }

    assert_crc32_kernel("crc32/pmull-v12e-v1", crc32_iso_hdlc_pmull_v12e_v1_safe, LENS);
    assert_crc32c_kernel("crc32c/pmull-v12e-v1", crc32c_iscsi_pmull_v12e_v1_safe, LENS);
  }

  #[test]
  fn test_crc32_pmull_eor3_v9s3x2e_s3_matches_portable_various_lengths() {
    let caps = platform::caps();
    if !(caps.has(platform::caps::aarch64::CRC_READY) && caps.has(platform::caps::aarch64::PMULL_EOR3_READY)) {
      return;
    }

    assert_crc32_kernel(
      "crc32/pmull-eor3-v9s3x2e-s3",
      crc32_iso_hdlc_pmull_eor3_v9s3x2e_s3_safe,
      LENS,
    );
    assert_crc32_kernel("crc32/pmull-eor3-2way", crc32_iso_hdlc_pmull_eor3_2way_safe, LENS);
    assert_crc32_kernel("crc32/pmull-eor3-3way", crc32_iso_hdlc_pmull_eor3_3way_safe, LENS);

    assert_crc32c_kernel(
      "crc32c/pmull-eor3-v9s3x2e-s3",
      crc32c_iscsi_pmull_eor3_v9s3x2e_s3_safe,
      LENS,
    );
    assert_crc32c_kernel("crc32c/pmull-eor3-2way", crc32c_iscsi_pmull_eor3_2way_safe, LENS);
    assert_crc32c_kernel("crc32c/pmull-eor3-3way", crc32c_iscsi_pmull_eor3_3way_safe, LENS);
  }

  #[test]
  fn test_crc32_sve2_pmull_matches_portable_various_lengths() {
    let caps = platform::caps();
    if !(caps.has(platform::caps::aarch64::CRC_READY)
      && caps.has(platform::caps::aarch64::PMULL_READY)
      && caps.has(platform::caps::aarch64::SVE2_PMULL))
    {
      return;
    }

    assert_crc32_kernel("crc32/sve2-pmull-2way", crc32_iso_hdlc_sve2_pmull_2way_safe, LENS);
    assert_crc32_kernel("crc32/sve2-pmull-3way", crc32_iso_hdlc_sve2_pmull_3way_safe, LENS);
    assert_crc32_kernel(
      "crc32/sve2-pmull-small",
      crc32_iso_hdlc_sve2_pmull_small_safe,
      SMALL_LENS,
    );

    assert_crc32c_kernel("crc32c/sve2-pmull-2way", crc32c_iscsi_sve2_pmull_2way_safe, LENS);
    assert_crc32c_kernel("crc32c/sve2-pmull-3way", crc32c_iscsi_sve2_pmull_3way_safe, LENS);
    assert_crc32c_kernel(
      "crc32c/sve2-pmull-small",
      crc32c_iscsi_sve2_pmull_small_safe,
      SMALL_LENS,
    );
  }
}
