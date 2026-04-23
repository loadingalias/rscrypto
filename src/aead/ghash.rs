#![allow(clippy::indexing_slicing)]

//! Constant-time GHASH universal hash (NIST SP 800-38D).
//!
//! GHASH operates in GF(2^128) with the irreducible polynomial
//! x^128 + x^7 + x^2 + x + 1 (big-endian / MSB-first convention).
//!
//! This implementation uses the GHASH↔POLYVAL relationship from RFC 8452 § 3:
//! blocks are loaded as big-endian u128 (equivalent to ByteReverse + LE load),
//! the hash key gets a `mulX_POLYVAL` correction, and the same Karatsuba
//! multiplication and Montgomery reduction used by POLYVAL apply unchanged
//! because the reflected GHASH polynomial equals POLYVAL's polynomial.

use super::polyval::clmul128_reduce;

/// GHASH block size in bytes (128 bits).
pub(crate) const BLOCK_SIZE: usize = 16;

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
/// Loads as big-endian u128 then applies `mulX_POLYVAL`. This is the same
/// key transformation used internally by `Ghash::new`, exposed for callers
/// that need the POLYVAL-domain key for precomputation.
#[inline]
pub(crate) fn h_to_polyval(h_bytes: &[u8; KEY_SIZE]) -> u128 {
  let h = u128::from_be_bytes(*h_bytes);
  mul_x_polyval(h)
}

/// GHASH accumulator state.
///
/// Internally operates in the "reflected" domain (identical to POLYVAL's
/// representation) so the same `clmul128` + `mont_reduce` pipeline applies.
/// The GHASH↔POLYVAL bridge is:
///
/// - Load blocks as big-endian u128 (= ByteReverse + LE load)
/// - Apply `mulX_POLYVAL` to the hash key
/// - Accumulate with XOR + field multiply + Montgomery reduce
/// - Finalize as big-endian bytes
pub(crate) struct Ghash {
  /// Hash key H with `mulX_POLYVAL` applied.
  h: u128,
  /// Running accumulator.
  acc: u128,
}

impl Ghash {
  /// Create a new GHASH instance with the given 128-bit hash key.
  ///
  /// The key `H` must be in GHASH convention (big-endian byte order).
  #[inline]
  pub(crate) fn new(h_bytes: &[u8; KEY_SIZE]) -> Self {
    // Loading as BE is equivalent to ByteReverse + LE load,
    // which maps the GHASH element into POLYVAL's internal domain.
    let h = u128::from_be_bytes(*h_bytes);
    // Apply the mulX_POLYVAL correction per RFC 8452 § 3.
    let h = mul_x_polyval(h);
    Self { h, acc: 0 }
  }

  /// Feed a single 16-byte block into the accumulator.
  ///
  /// Computes: acc = (acc XOR block) * H in GF(2^128).
  #[inline]
  pub(crate) fn update_block(&mut self, block: &[u8; BLOCK_SIZE]) {
    let block = u128::from_be_bytes(*block);
    self.acc ^= block;
    self.acc = clmul128_reduce(self.acc, self.h);
  }

  /// Feed arbitrary-length data, padding the last block with zeros.
  pub(crate) fn update_padded(&mut self, data: &[u8]) {
    let (blocks, remainder) = data.as_chunks::<BLOCK_SIZE>();
    for block in blocks {
      self.update_block(block);
    }

    if !remainder.is_empty() {
      let mut block = [0u8; BLOCK_SIZE];
      block[..remainder.len()].copy_from_slice(remainder);
      self.update_block(&block);
    }
  }

  /// Finalize and return the 16-byte GHASH digest.
  #[inline]
  pub(crate) fn finalize(self) -> [u8; BLOCK_SIZE] {
    // Converting back to GHASH convention: to_be_bytes reverses the
    // ByteReverse that was applied at input time.
    self.acc.to_be_bytes()
  }

  /// Finalize and return the raw accumulator as u128 (POLYVAL domain).
  ///
  /// Used by the wide GCM path to continue accumulation outside the
  /// `Ghash` struct with precomputed H powers.
  #[cfg(target_arch = "x86_64")]
  #[inline]
  pub(crate) fn finalize_u128(self) -> u128 {
    self.acc
  }
}

impl Drop for Ghash {
  fn drop(&mut self) {
    // SAFETY: self.acc/self.h are valid, aligned, dereferenceable pointers to initialized memory.
    unsafe {
      core::ptr::write_volatile(&mut self.acc, 0);
      core::ptr::write_volatile(&mut self.h, 0);
    }
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
  }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
  use super::*;

  /// GHASH with empty input should return zero.
  #[test]
  fn ghash_empty() {
    let h = [0x42u8; 16];
    let gh = Ghash::new(&h);
    assert_eq!(gh.finalize(), [0u8; 16]);
  }

  /// GHASH with zero key should always return zero.
  #[test]
  fn ghash_zero_key() {
    let h = [0u8; 16];
    let x = [0xffu8; 16];
    let mut gh = Ghash::new(&h);
    gh.update_block(&x);
    assert_eq!(gh.finalize(), [0u8; 16]);
  }

  /// Verify update_padded matches manual block-by-block.
  #[test]
  fn ghash_padded_matches_manual() {
    let h = hex_to_16("66e94bd4ef8a2c3b884cfa59ca342b2e");
    let data = b"Hello, World! This is test data for GHASH padding.";

    // Manual: split into 16-byte blocks, pad last one.
    let mut manual = Ghash::new(&h);
    let mut offset = 0;
    while offset + 16 <= data.len() {
      let block: [u8; 16] = data[offset..offset + 16].try_into().unwrap();
      manual.update_block(&block);
      offset += 16;
    }
    if offset < data.len() {
      let mut block = [0u8; 16];
      block[..data.len() - offset].copy_from_slice(&data[offset..]);
      manual.update_block(&block);
    }
    let manual_result = manual.finalize();

    // Padded API.
    let mut padded = Ghash::new(&h);
    padded.update_padded(data);
    let padded_result = padded.finalize();

    assert_eq!(manual_result, padded_result);
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

  fn hex_to_16(hex: &str) -> [u8; 16] {
    let mut out = [0u8; 16];
    let mut i = 0;
    while i < 16 {
      out[i] = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap();
      i = i.strict_add(1);
    }
    out
  }
}
