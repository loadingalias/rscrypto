#[cfg(feature = "aes-gcm")]
use core::arch::riscv64::{aes64ds, aes64dsm, aes64im};
use core::arch::riscv64::{aes64es, aes64esm};

/// AES-256 round keys stored as 15 pairs of 64-bit halves for scalar AES.
///
/// Each round key is split into the two register halves that [`load_halves`]
/// describes.
#[repr(C, align(64))]
pub(super) struct RvScalarRoundKeys {
  rk: [(u64, u64); 15],
}

impl RvScalarRoundKeys {
  /// Zeroize all round keys via volatile writes.
  pub(super) fn zeroize(&mut self) {
    // SAFETY: [(u64, u64); 15] is layout-compatible with [u8; 240].
    let bytes = unsafe { core::slice::from_raw_parts_mut(self.rk.as_mut_ptr().cast::<u8>(), 15usize.strict_mul(16)) };
    crate::traits::ct::zeroize(bytes);
  }
}

/// Convert portable round keys (60 × big-endian u32) to scalar-Zkne format.
pub(super) fn from_portable(rk: &[u32; 60]) -> RvScalarRoundKeys {
  let mut keys = [(0u64, 0u64); 15];
  for (key, words) in keys.iter_mut().zip(rk.as_chunks::<4>().0) {
    *key = round_key_halves(words);
  }
  RvScalarRoundKeys { rk: keys }
}

/// Split one round key of the portable schedule into the registers that [`load_halves`] describes.
///
/// Word `w` holds round-key bytes `4w..4w + 4`, most significant first (FIPS 197). The halves come
/// from the words in registers, so no byte copy of the round key exists.
#[inline(always)]
fn round_key_halves(words: &[u32; 4]) -> (u64, u64) {
  let half = |first: u32, second: u32| u64::from(first.swap_bytes()) | (u64::from(second.swap_bytes()) << 32);
  (half(words[0], words[1]), half(words[2], words[3]))
}

/// Split an AES state or round key into the two registers of the RV64 AES instructions.
///
/// Each register holds eight state bytes in little-endian order: bytes 0..8 form the low register
/// and bytes 8..16 the high one. The ratified scalar crypto specification (v1.0.1) defines
/// `aes64es` and its siblings by the Sail model, in which `aes_rv64_shiftrows_fwd` selects state
/// byte `i` of a register as `getbyte(x, i) = x[8i+7..8i]` (sail-riscv
/// `model/extensions/K/types_kext.sail`).
#[inline(always)]
fn load_halves(block: &[u8; 16]) -> (u64, u64) {
  let (lo, hi) = block.split_at(8);
  (
    u64::from_le_bytes(lo.try_into().expect("eight-byte half")),
    u64::from_le_bytes(hi.try_into().expect("eight-byte half")),
  )
}

/// Inverse of [`load_halves`].
#[inline(always)]
fn store_halves(block: &mut [u8; 16], lo: u64, hi: u64) {
  block[0..8].copy_from_slice(&lo.to_le_bytes());
  block[8..16].copy_from_slice(&hi.to_le_bytes());
}

/// Encrypt a single 16-byte block using AES-256 with scalar RV64 AES.
///
/// The scalar AES instructions transform one 64-bit half of the AES state at
/// a time. The second instruction for each round reverses the source register
/// order, exactly as described in the RISC-V scalar crypto specification.
///
/// # Safety
/// Caller must ensure the CPU supports Zkne.
#[target_feature(enable = "zkne")]
pub(super) unsafe fn encrypt_block(keys: &RvScalarRoundKeys, block: &mut [u8; 16]) {
  let (mut lo, mut hi) = load_halves(block);

  let (rk0_lo, rk0_hi) = keys.rk[0];
  lo ^= rk0_lo;
  hi ^= rk0_hi;

  let mut round = 1usize;
  while round < 14 {
    let next_lo = aes64esm(lo, hi);
    let next_hi = aes64esm(hi, lo);
    let (rk_lo, rk_hi) = keys.rk[round];
    lo = next_lo ^ rk_lo;
    hi = next_hi ^ rk_hi;
    round = round.strict_add(1);
  }

  let next_lo = aes64es(lo, hi);
  let next_hi = aes64es(hi, lo);
  let (rk_lo, rk_hi) = keys.rk[14];
  lo = next_lo ^ rk_lo;
  hi = next_hi ^ rk_hi;

  store_halves(block, lo, hi);
}

/// Encrypt four independent 16-byte blocks using AES-256 with scalar RV64 AES.
///
/// # Safety
/// Caller must ensure the CPU supports Zkne.
#[target_feature(enable = "zkne")]
pub(super) unsafe fn encrypt_4blocks(keys: &RvScalarRoundKeys, blocks: &mut [[u8; 16]; 4]) {
  let (mut lo0, mut hi0) = load_halves(&blocks[0]);
  let (mut lo1, mut hi1) = load_halves(&blocks[1]);
  let (mut lo2, mut hi2) = load_halves(&blocks[2]);
  let (mut lo3, mut hi3) = load_halves(&blocks[3]);

  let (rk0_lo, rk0_hi) = keys.rk[0];
  lo0 ^= rk0_lo;
  hi0 ^= rk0_hi;
  lo1 ^= rk0_lo;
  hi1 ^= rk0_hi;
  lo2 ^= rk0_lo;
  hi2 ^= rk0_hi;
  lo3 ^= rk0_lo;
  hi3 ^= rk0_hi;

  let mut round = 1usize;
  while round < 14 {
    let (rk_lo, rk_hi) = keys.rk[round];
    let next_lo0 = aes64esm(lo0, hi0) ^ rk_lo;
    let next_hi0 = aes64esm(hi0, lo0) ^ rk_hi;
    let next_lo1 = aes64esm(lo1, hi1) ^ rk_lo;
    let next_hi1 = aes64esm(hi1, lo1) ^ rk_hi;
    let next_lo2 = aes64esm(lo2, hi2) ^ rk_lo;
    let next_hi2 = aes64esm(hi2, lo2) ^ rk_hi;
    let next_lo3 = aes64esm(lo3, hi3) ^ rk_lo;
    let next_hi3 = aes64esm(hi3, lo3) ^ rk_hi;
    lo0 = next_lo0;
    hi0 = next_hi0;
    lo1 = next_lo1;
    hi1 = next_hi1;
    lo2 = next_lo2;
    hi2 = next_hi2;
    lo3 = next_lo3;
    hi3 = next_hi3;
    round = round.strict_add(1);
  }

  let (rk_lo, rk_hi) = keys.rk[14];
  let next_lo0 = aes64es(lo0, hi0) ^ rk_lo;
  let next_hi0 = aes64es(hi0, lo0) ^ rk_hi;
  let next_lo1 = aes64es(lo1, hi1) ^ rk_lo;
  let next_hi1 = aes64es(hi1, lo1) ^ rk_hi;
  let next_lo2 = aes64es(lo2, hi2) ^ rk_lo;
  let next_hi2 = aes64es(hi2, lo2) ^ rk_hi;
  let next_lo3 = aes64es(lo3, hi3) ^ rk_lo;
  let next_hi3 = aes64es(hi3, lo3) ^ rk_hi;

  store_halves(&mut blocks[0], next_lo0, next_hi0);
  store_halves(&mut blocks[1], next_lo1, next_hi1);
  store_halves(&mut blocks[2], next_lo2, next_hi2);
  store_halves(&mut blocks[3], next_lo3, next_hi3);
}

// AES-128 (11 round keys, 10 rounds)

/// AES-128 round keys stored as 11 pairs of 64-bit halves for scalar AES.
#[repr(C, align(64))]
pub(super) struct RvScalar128RoundKeys {
  rk: [(u64, u64); 11],
}

impl RvScalar128RoundKeys {
  /// Zeroize all round keys via volatile writes.
  pub(super) fn zeroize(&mut self) {
    // SAFETY: [(u64, u64); 11] is layout-compatible with [u8; 176].
    let bytes = unsafe { core::slice::from_raw_parts_mut(self.rk.as_mut_ptr().cast::<u8>(), 11usize.strict_mul(16)) };
    crate::traits::ct::zeroize(bytes);
  }
}

/// Convert portable round keys (44 × big-endian u32) to scalar-Zkne format.
pub(super) fn from_portable_128(rk: &[u32; 44]) -> RvScalar128RoundKeys {
  let mut keys = [(0u64, 0u64); 11];
  for (key, words) in keys.iter_mut().zip(rk.as_chunks::<4>().0) {
    *key = round_key_halves(words);
  }
  RvScalar128RoundKeys { rk: keys }
}

/// Convert the portable AES-128 schedule into the equivalent inverse cipher schedule
/// (FIPS 197 section 5.3.5) for Zknd.
///
/// The schedule is the encryption schedule reversed, with `aes64im` (InvMixColumns) applied to the
/// nine middle round keys, because `aes64dsm` applies InvMixColumns before the key is added.
///
/// # Safety
/// Caller must ensure the CPU supports Zknd.
#[cfg(feature = "aes-gcm")]
#[target_feature(enable = "zknd")]
pub(super) unsafe fn decrypt_keys_from_portable_128(rk: &[u32; 44]) -> RvScalar128RoundKeys {
  let mut keys = [(0u64, 0u64); 11];
  for (round, (key, words)) in keys.iter_mut().zip(rk.as_chunks::<4>().0.iter().rev()).enumerate() {
    let (lo, hi) = round_key_halves(words);
    *key = if round == 0 || round == 10 {
      (lo, hi)
    } else {
      (aes64im(lo), aes64im(hi))
    };
  }
  RvScalar128RoundKeys { rk: keys }
}

/// Decrypt a single 16-byte block using AES-128 with scalar RV64 AES.
///
/// # Safety
/// Caller must ensure the CPU supports Zknd, and `keys` must come from
/// [`decrypt_keys_from_portable_128`].
#[cfg(feature = "aes-gcm")]
#[target_feature(enable = "zknd")]
pub(super) unsafe fn decrypt_block_128(keys: &RvScalar128RoundKeys, block: &mut [u8; 16]) {
  let (mut lo, mut hi) = load_halves(block);

  let (rk0_lo, rk0_hi) = keys.rk[0];
  lo ^= rk0_lo;
  hi ^= rk0_hi;

  let mut round = 1usize;
  while round < 10 {
    let next_lo = aes64dsm(lo, hi);
    let next_hi = aes64dsm(hi, lo);
    let (rk_lo, rk_hi) = keys.rk[round];
    lo = next_lo ^ rk_lo;
    hi = next_hi ^ rk_hi;
    round = round.strict_add(1);
  }

  let next_lo = aes64ds(lo, hi);
  let next_hi = aes64ds(hi, lo);
  let (rk_lo, rk_hi) = keys.rk[10];
  store_halves(block, next_lo ^ rk_lo, next_hi ^ rk_hi);
}

/// Encrypt a single 16-byte block using AES-128 with scalar RV64 AES.
///
/// # Safety
/// Caller must ensure the CPU supports Zkne.
#[target_feature(enable = "zkne")]
pub(super) unsafe fn encrypt_block_128(keys: &RvScalar128RoundKeys, block: &mut [u8; 16]) {
  let (mut lo, mut hi) = load_halves(block);

  let (rk0_lo, rk0_hi) = keys.rk[0];
  lo ^= rk0_lo;
  hi ^= rk0_hi;

  let mut round = 1usize;
  while round < 10 {
    let next_lo = aes64esm(lo, hi);
    let next_hi = aes64esm(hi, lo);
    let (rk_lo, rk_hi) = keys.rk[round];
    lo = next_lo ^ rk_lo;
    hi = next_hi ^ rk_hi;
    round = round.strict_add(1);
  }

  let next_lo = aes64es(lo, hi);
  let next_hi = aes64es(hi, lo);
  let (rk_lo, rk_hi) = keys.rk[10];
  lo = next_lo ^ rk_lo;
  hi = next_hi ^ rk_hi;

  store_halves(block, lo, hi);
}

/// Encrypt four independent 16-byte blocks using AES-128 with scalar RV64 AES.
///
/// # Safety
/// Caller must ensure the CPU supports Zkne.
#[target_feature(enable = "zkne")]
pub(super) unsafe fn encrypt_4blocks_128(keys: &RvScalar128RoundKeys, blocks: &mut [[u8; 16]; 4]) {
  let (mut lo0, mut hi0) = load_halves(&blocks[0]);
  let (mut lo1, mut hi1) = load_halves(&blocks[1]);
  let (mut lo2, mut hi2) = load_halves(&blocks[2]);
  let (mut lo3, mut hi3) = load_halves(&blocks[3]);

  let (rk0_lo, rk0_hi) = keys.rk[0];
  lo0 ^= rk0_lo;
  hi0 ^= rk0_hi;
  lo1 ^= rk0_lo;
  hi1 ^= rk0_hi;
  lo2 ^= rk0_lo;
  hi2 ^= rk0_hi;
  lo3 ^= rk0_lo;
  hi3 ^= rk0_hi;

  let mut round = 1usize;
  while round < 10 {
    let (rk_lo, rk_hi) = keys.rk[round];
    let next_lo0 = aes64esm(lo0, hi0) ^ rk_lo;
    let next_hi0 = aes64esm(hi0, lo0) ^ rk_hi;
    let next_lo1 = aes64esm(lo1, hi1) ^ rk_lo;
    let next_hi1 = aes64esm(hi1, lo1) ^ rk_hi;
    let next_lo2 = aes64esm(lo2, hi2) ^ rk_lo;
    let next_hi2 = aes64esm(hi2, lo2) ^ rk_hi;
    let next_lo3 = aes64esm(lo3, hi3) ^ rk_lo;
    let next_hi3 = aes64esm(hi3, lo3) ^ rk_hi;
    lo0 = next_lo0;
    hi0 = next_hi0;
    lo1 = next_lo1;
    hi1 = next_hi1;
    lo2 = next_lo2;
    hi2 = next_hi2;
    lo3 = next_lo3;
    hi3 = next_hi3;
    round = round.strict_add(1);
  }

  let (rk_lo, rk_hi) = keys.rk[10];
  let next_lo0 = aes64es(lo0, hi0) ^ rk_lo;
  let next_hi0 = aes64es(hi0, lo0) ^ rk_hi;
  let next_lo1 = aes64es(lo1, hi1) ^ rk_lo;
  let next_hi1 = aes64es(hi1, lo1) ^ rk_hi;
  let next_lo2 = aes64es(lo2, hi2) ^ rk_lo;
  let next_hi2 = aes64es(hi2, lo2) ^ rk_hi;
  let next_lo3 = aes64es(lo3, hi3) ^ rk_lo;
  let next_hi3 = aes64es(hi3, lo3) ^ rk_hi;

  store_halves(&mut blocks[0], next_lo0, next_hi0);
  store_halves(&mut blocks[1], next_lo1, next_hi1);
  store_halves(&mut blocks[2], next_lo2, next_hi2);
  store_halves(&mut blocks[3], next_lo3, next_hi3);
}

#[cfg(test)]
mod tests {
  /// The Sail model reads state byte `i` of a register from bits `8i + 7..8i` (`getbyte` in sail-riscv
  /// `model/extensions/K/types_kext.sail`), so round-key bytes 0..8 are the low register, least
  /// significant first. The words are the FIPS 197 schedule words of round-key bytes 0x00..0x0f.
  #[test]
  fn round_key_halves_place_byte_i_in_bits_8i_upward() {
    let words = [0x0001_0203, 0x0405_0607, 0x0809_0a0b, 0x0c0d_0e0f];
    assert_eq!(
      super::round_key_halves(&words),
      (0x0706_0504_0302_0100, 0x0f0e_0d0c_0b0a_0908)
    );
  }
}
