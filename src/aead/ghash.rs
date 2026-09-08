//! Fixed-schedule, table-free GHASH universal hash (NIST SP 800-38D).
//!
//! Generated-code timing claims remain configuration- and
//! release-evidence-bound; see `ct.toml`.
//!
//! GHASH operates in GF(2^128) with the irreducible polynomial
//! x^128 + x^7 + x^2 + x + 1 (big-endian / MSB-first convention).
//!
//! This implementation uses the GHASH↔POLYVAL relationship from RFC 8452 § 3:
//! blocks are loaded as big-endian u128 (equivalent to ByteReverse + LE load),
//! the hash key gets a `mulX_POLYVAL` correction, and the same Karatsuba
//! multiplication and Montgomery reduction used by POLYVAL apply unchanged
//! because the reflected GHASH polynomial equals POLYVAL's polynomial.

use super::polyval;
use crate::traits::ct;

/// GHASH key size in bytes.
pub(crate) const KEY_SIZE: usize = 16;

/// POLYVAL feedback constant: x^127 + x^126 + x^121 + 1.
///
/// When multiplying by x in POLYVAL's field and the high bit (x^127) is set,
/// x^128 reduces to this value.
const POLYVAL_FEEDBACK: u128 = (1u128 << 127) | (1u128 << 126) | (1u128 << 121) | 1;

/// Multiply a field element by x in the POLYVAL field.
///
/// This is the `mulX_POLYVAL` operation from RFC 8452 § 3: left-shift by 1,
/// with conditional XOR of the reduction polynomial if the top bit was set.
#[inline]
fn mul_x_polyval(v: u128) -> u128 {
  let carry = v >> 127;
  let shifted = v << 1;
  shifted ^ (0u128.wrapping_sub(carry) & POLYVAL_FEEDBACK)
}

/// Convert a raw GHASH key (big-endian bytes) into the POLYVAL domain.
///
/// Loads as big-endian u128 then applies `mulX_POLYVAL` for precomputation.
#[inline]
pub(crate) fn h_to_polyval(h_bytes: &[u8; KEY_SIZE]) -> u128 {
  let h = u128::from_be_bytes(*h_bytes);
  mul_x_polyval(h)
}

/// Computes one GHASH block with the portable POLYVAL-domain reduction.
#[cfg(feature = "diag")]
#[must_use]
pub fn diag_ghash_block_portable(h_bytes: &[u8; KEY_SIZE], block: &[u8; KEY_SIZE]) -> [u8; KEY_SIZE] {
  let h = h_to_polyval(h_bytes);
  let block = u128::from_be_bytes(*block);
  super::polyval::clmul128_reduce_portable(block, h).to_be_bytes()
}

pub(super) struct GhashAccumulator(pub(super) u128);

impl Drop for GhashAccumulator {
  fn drop(&mut self) {
    ct::zeroize_words(core::slice::from_mut(&mut self.0));
  }
}

#[inline]
pub(super) fn ghash_update_padded(mut acc: u128, h_polyval: u128, data: &[u8]) -> u128 {
  let (blocks, remainder) = data.as_chunks::<16>();
  for block in blocks {
    acc ^= u128::from_be_bytes(*block);
    acc = polyval::clmul128_reduce(acc, h_polyval);
  }

  if !remainder.is_empty() {
    let mut block = [0u8; 16];
    block[..remainder.len()].copy_from_slice(remainder);
    acc ^= u128::from_be_bytes(block);
    acc = polyval::clmul128_reduce(acc, h_polyval);
  }

  acc
}

#[inline]
pub(super) fn ghash_update_padded_wide(mut acc: u128, h_polyval: u128, h_powers_rev: &[u128; 4], data: &[u8]) -> u128 {
  let (full_blocks, remainder) = data.as_chunks::<16>();
  let (chunks, trailing_blocks) = full_blocks.as_chunks::<4>();

  for chunk in chunks {
    let blocks = [
      u128::from_be_bytes(chunk[0]),
      u128::from_be_bytes(chunk[1]),
      u128::from_be_bytes(chunk[2]),
      u128::from_be_bytes(chunk[3]),
    ];
    acc = polyval::accumulate_4blocks(acc, h_polyval, h_powers_rev, &blocks);
  }

  for block in trailing_blocks {
    acc ^= u128::from_be_bytes(*block);
    acc = polyval::clmul128_reduce(acc, h_polyval);
  }

  if !remainder.is_empty() {
    let mut block = [0u8; 16];
    block[..remainder.len()].copy_from_slice(remainder);
    acc ^= u128::from_be_bytes(block);
    acc = polyval::clmul128_reduce(acc, h_polyval);
  }

  acc
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub(super) fn ghash_collect_padded_block(blocks: &mut [u128; 4], block_count: &mut usize, block: u128) -> bool {
  if *block_count == 4 {
    return false;
  }
  blocks[*block_count] = block;
  *block_count = (*block_count).strict_add(1);
  true
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub(super) fn ghash_collect_padded(blocks: &mut [u128; 4], block_count: &mut usize, data: &[u8]) -> bool {
  let (full_blocks, remainder) = data.as_chunks::<16>();
  for block in full_blocks {
    if !ghash_collect_padded_block(blocks, block_count, u128::from_be_bytes(*block)) {
      return false;
    }
  }

  if !remainder.is_empty() {
    let mut block = [0u8; 16];
    block[..remainder.len()].copy_from_slice(remainder);
    return ghash_collect_padded_block(blocks, block_count, u128::from_be_bytes(block));
  }

  true
}

/// Update GHASH using VPCLMUL without scalar block packing.
///
/// # Safety
/// Caller must ensure VPCLMULQDQ, PCLMULQDQ, AVX-512F/VL/BW/DQ, and SSE2 are available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq,pclmulqdq,sse2")]
pub(super) unsafe fn ghash_update_padded_wide_x86(
  mut acc: u128,
  h_polyval: u128,
  h_powers_rev: &[u128; 4],
  data: &[u8],
) -> u128 {
  let (chunks, tail) = data.as_chunks::<64>();
  for chunk in chunks {
    // SAFETY: direct-byte VPCLMUL GHASH aggregation because:
    // 1. This function's caller guarantees all required x86 target features.
    // 2. `chunk` is exactly four initialized 16-byte GHASH blocks.
    acc = unsafe { polyval::x86_aggregate_4blocks_be_bytes_inline(acc, h_powers_rev, chunk) };
  }

  let (full_blocks, remainder) = tail.as_chunks::<16>();
  for block in full_blocks {
    acc ^= u128::from_be_bytes(*block);
    // SAFETY: x86 carryless multiply because:
    // 1. This function's caller guarantees PCLMULQDQ and SSE2 availability.
    // 2. `acc` and `h_polyval` are initialized GHASH field elements.
    acc = unsafe { polyval::x86_clmul128_reduce_inline(acc, h_polyval) };
  }

  if !remainder.is_empty() {
    let mut block = [0u8; 16];
    block[..remainder.len()].copy_from_slice(remainder);
    acc ^= u128::from_be_bytes(block);
    // SAFETY: x86 carryless multiply because:
    // 1. This function's caller guarantees PCLMULQDQ and SSE2 availability.
    // 2. `acc` and `h_polyval` are initialized GHASH field elements.
    acc = unsafe { polyval::x86_clmul128_reduce_inline(acc, h_polyval) };
  }

  acc
}

/// Update GHASH using PMULL without leaving the aarch64 target-feature scope.
///
/// # Safety
/// Caller must ensure AES-CE and PMULL are available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes,neon")]
pub(super) unsafe fn ghash_update_padded_wide_aarch64(
  mut acc: u128,
  h_polyval: u128,
  h_powers_rev: &[u128; 4],
  data: &[u8],
) -> u128 {
  let (full_blocks, remainder) = data.as_chunks::<16>();
  let (chunks, trailing_blocks) = full_blocks.as_chunks::<4>();

  for chunk in chunks {
    let blocks = [
      u128::from_be_bytes(chunk[0]),
      u128::from_be_bytes(chunk[1]),
      u128::from_be_bytes(chunk[2]),
      u128::from_be_bytes(chunk[3]),
    ];
    // SAFETY: PMULL 4-block GHASH aggregation because:
    // 1. This function's caller must guarantee AES-CE/PMULL availability.
    // 2. `h_powers_rev` and `blocks` are fixed 4-lane arrays with valid initialized values.
    acc = unsafe { polyval::aarch64_aggregate_4blocks_inline(acc, h_powers_rev, &blocks) };
  }

  for block in trailing_blocks {
    acc ^= u128::from_be_bytes(*block);
    // SAFETY: PMULL carryless multiply because:
    // 1. This function's caller must guarantee AES-CE/PMULL availability.
    // 2. `acc` and `h_polyval` are initialized GHASH field elements.
    acc = unsafe { polyval::aarch64_clmul128_reduce_inline(acc, h_polyval) };
  }

  if !remainder.is_empty() {
    let mut block = [0u8; 16];
    block[..remainder.len()].copy_from_slice(remainder);
    acc ^= u128::from_be_bytes(block);
    // SAFETY: PMULL carryless multiply because:
    // 1. This function's caller must guarantee AES-CE/PMULL availability.
    // 2. `acc` and `h_polyval` are initialized GHASH field elements.
    acc = unsafe { polyval::aarch64_clmul128_reduce_inline(acc, h_polyval) };
  }

  acc
}

/// Update GHASH using POWER8 crypto without leaving the target-feature scope.
///
/// # Safety
/// Caller must ensure POWER8 crypto is available.
#[cfg(target_arch = "powerpc64")]
#[target_feature(enable = "altivec,vsx,power8-vector,power8-crypto")]
pub(super) unsafe fn ghash_update_padded_wide_ppc(
  mut acc: u128,
  h_polyval: u128,
  h_powers_rev: &[u128; 4],
  data: &[u8],
) -> u128 {
  let (full_blocks, remainder) = data.as_chunks::<16>();
  let (chunks, trailing_blocks) = full_blocks.as_chunks::<4>();

  for chunk in chunks {
    let blocks = [
      u128::from_be_bytes(chunk[0]),
      u128::from_be_bytes(chunk[1]),
      u128::from_be_bytes(chunk[2]),
      u128::from_be_bytes(chunk[3]),
    ];
    // SAFETY: POWER8 4-block GHASH aggregation because:
    // 1. This function's caller must guarantee POWER8 crypto availability.
    // 2. `h_powers_rev` and `blocks` are fixed 4-lane arrays with valid initialized values.
    acc = unsafe { polyval::ppc_aggregate_4blocks_inline(acc, h_powers_rev, &blocks) };
  }

  for block in trailing_blocks {
    acc ^= u128::from_be_bytes(*block);
    // SAFETY: POWER8 carryless multiply because:
    // 1. This function's caller must guarantee POWER8 crypto availability.
    // 2. `acc` and `h_polyval` are initialized GHASH field elements.
    acc = unsafe { polyval::ppc_clmul128_reduce_inline(acc, h_polyval) };
  }

  if !remainder.is_empty() {
    let mut block = [0u8; 16];
    block[..remainder.len()].copy_from_slice(remainder);
    acc ^= u128::from_be_bytes(block);
    // SAFETY: POWER8 carryless multiply because:
    // 1. This function's caller must guarantee POWER8 crypto availability.
    // 2. `acc` and `h_polyval` are initialized GHASH field elements.
    acc = unsafe { polyval::ppc_clmul128_reduce_inline(acc, h_polyval) };
  }

  acc
}

// Tests

#[cfg(test)]
mod tests {
  use super::*;

  const TEST_H: [u8; 16] = 0x66e9_4bd4_ef8a_2c3b_884c_fa59_ca34_2b2eu128.to_be_bytes();

  #[test]
  fn ghash_empty() {
    let h = h_to_polyval(&TEST_H);
    let powers = polyval::precompute_powers(h);
    let reversed = [powers[3], powers[2], powers[1], powers[0]];
    for initial in [0, 0x1234] {
      assert_eq!(ghash_update_padded(initial, h, &[]), initial);
      assert_eq!(ghash_update_padded_wide(initial, h, &reversed, &[]), initial);
    }
  }

  #[test]
  fn ghash_zero_key() {
    for len in [1, 15, 16, 17, 63, 64, 65] {
      let data = [0xff; 65];
      assert_eq!(ghash_update_padded(0, 0, &data[..len]), 0);
      assert_eq!(ghash_update_padded_wide(0, 0, &[0; 4], &data[..len]), 0);
    }
  }

  #[test]
  fn padded_helpers_match_portable_reduction() {
    let input: [u8; 288] = core::array::from_fn(|i| i.to_le_bytes()[0].wrapping_mul(17));
    let h = h_to_polyval(&TEST_H);
    let powers = polyval::precompute_powers(h);
    let reversed = [powers[3], powers[2], powers[1], powers[0]];
    let _caps = crate::platform::caps();
    for len in [
      0usize, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257,
    ] {
      for offset in [0usize, 1, 15] {
        let data = &input[offset..offset.strict_add(len)];
        let mut padded = [0u8; 272];
        padded[..len].copy_from_slice(data);
        for initial in [0, u128::MAX] {
          let mut expected = initial;
          for block in padded[..len.next_multiple_of(16)].as_chunks::<16>().0 {
            expected = polyval::clmul128_reduce_portable(expected ^ u128::from_be_bytes(*block), h);
          }
          assert_eq!(
            ghash_update_padded(initial, h, data),
            expected,
            "scalar len={len} offset={offset}"
          );
          assert_eq!(
            ghash_update_padded_wide(initial, h, &reversed, data),
            expected,
            "wide len={len} offset={offset}"
          );
          #[cfg(target_arch = "aarch64")]
          if _caps.has(crate::platform::caps::aarch64::AES | crate::platform::caps::aarch64::PMULL) {
            // SAFETY: AES/PMULL were checked above; slices and key powers are initialized.
            let actual = unsafe { ghash_update_padded_wide_aarch64(initial, h, &reversed, data) };
            assert_eq!(actual, expected, "aarch64 len={len} offset={offset}");
          }
          #[cfg(target_arch = "x86_64")]
          if _caps.has(crate::platform::caps::x86::VPCLMUL_READY) {
            // SAFETY: VPCLMUL_READY establishes the helper's target features; inputs are initialized.
            let actual = unsafe { ghash_update_padded_wide_x86(initial, h, &reversed, data) };
            assert_eq!(actual, expected, "x86 len={len} offset={offset}");
          }
          #[cfg(target_arch = "powerpc64")]
          if _caps.has(crate::platform::caps::power::POWER8_CRYPTO) {
            // SAFETY: POWER8 crypto was checked above; slices and key powers are initialized.
            let actual = unsafe { ghash_update_padded_wide_ppc(initial, h, &reversed, data) };
            assert_eq!(actual, expected, "power len={len} offset={offset}");
          }
        }
      }
    }
  }

  #[test]
  #[cfg(target_arch = "x86_64")]
  fn padded_collection_preserves_field_boundaries_and_capacity() {
    let mut blocks = [0u128; 4];
    let mut count = 0;
    assert!(ghash_collect_padded(&mut blocks, &mut count, &[]));
    assert_eq!(count, 0);
    assert!(ghash_collect_padded(&mut blocks, &mut count, &[0x11; 1]));
    assert!(ghash_collect_padded(&mut blocks, &mut count, &[0x22; 17]));
    assert_eq!(count, 3);
    assert_eq!(
      blocks[..3],
      [0x11u128 << 120, u128::from_be_bytes([0x22; 16]), 0x22u128 << 120]
    );
    assert!(ghash_collect_padded_block(&mut blocks, &mut count, 7));
    let full = blocks;
    assert!(!ghash_collect_padded_block(&mut blocks, &mut count, 8));
    assert!(!ghash_collect_padded(&mut blocks, &mut count, &[0x33; 1]));
    assert!(!ghash_collect_padded(&mut blocks, &mut count, &[0x33; 16]));
    assert!(ghash_collect_padded(&mut blocks, &mut count, &[]));
    assert_eq!(count, 4);
    assert_eq!(blocks, full);
  }

  /// mulX_POLYVAL: zero input.
  #[test]
  fn mul_x_zero() {
    assert_eq!(mul_x_polyval(0), 0);
  }

  /// mulX_POLYVAL: 1 → 2 (no reduction).
  #[test]
  fn mul_x_one() {
    assert_eq!(mul_x_polyval(1), 2);
  }

  /// mulX_POLYVAL: high bit set triggers reduction.
  #[test]
  fn mul_x_high_bit() {
    let v = 1u128 << 127;
    let result = mul_x_polyval(v);
    assert_eq!(
      result, POLYVAL_FEEDBACK,
      "mulX(x^127) should reduce to feedback polynomial"
    );
  }
}
