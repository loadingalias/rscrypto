use core::arch::riscv64::{aes64es, aes64esm};

/// AES-256 round keys stored as 15 pairs of 64-bit halves for scalar AES.
///
/// Each round key is kept in canonical AES byte order and split into the
/// upper/lower 64-bit halves consumed by the RV64 scalar AES instructions.
#[derive(Clone)]
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
  let mut i = 0usize;
  while i < 15 {
    let base = i.strict_mul(4);
    let mut bytes = [0u8; 16];
    bytes[0..4].copy_from_slice(&rk[base].to_be_bytes());
    bytes[4..8].copy_from_slice(&rk[base.strict_add(1)].to_be_bytes());
    bytes[8..12].copy_from_slice(&rk[base.strict_add(2)].to_be_bytes());
    bytes[12..16].copy_from_slice(&rk[base.strict_add(3)].to_be_bytes());
    let mut lo_bytes = [0u8; 8];
    lo_bytes.copy_from_slice(&bytes[0..8]);
    let mut hi_bytes = [0u8; 8];
    hi_bytes.copy_from_slice(&bytes[8..16]);
    keys[i] = (u64::from_be_bytes(lo_bytes), u64::from_be_bytes(hi_bytes));
    i = i.strict_add(1);
  }
  RvScalarRoundKeys { rk: keys }
}

#[inline(always)]
fn load_halves(block: &[u8; 16]) -> (u64, u64) {
  let mut lo_bytes = [0u8; 8];
  lo_bytes.copy_from_slice(&block[0..8]);
  let mut hi_bytes = [0u8; 8];
  hi_bytes.copy_from_slice(&block[8..16]);
  (u64::from_be_bytes(lo_bytes), u64::from_be_bytes(hi_bytes))
}

#[inline(always)]
fn store_halves(block: &mut [u8; 16], lo: u64, hi: u64) {
  block[0..8].copy_from_slice(&lo.to_be_bytes());
  block[8..16].copy_from_slice(&hi.to_be_bytes());
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

// ---------------------------------------------------------------------------
// AES-128 (11 round keys, 10 rounds)
// ---------------------------------------------------------------------------

/// AES-128 round keys stored as 11 pairs of 64-bit halves for scalar AES.
#[derive(Clone)]
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
  let mut i = 0usize;
  while i < 11 {
    let base = i.strict_mul(4);
    let mut bytes = [0u8; 16];
    bytes[0..4].copy_from_slice(&rk[base].to_be_bytes());
    bytes[4..8].copy_from_slice(&rk[base.strict_add(1)].to_be_bytes());
    bytes[8..12].copy_from_slice(&rk[base.strict_add(2)].to_be_bytes());
    bytes[12..16].copy_from_slice(&rk[base.strict_add(3)].to_be_bytes());
    let mut lo_bytes = [0u8; 8];
    lo_bytes.copy_from_slice(&bytes[0..8]);
    let mut hi_bytes = [0u8; 8];
    hi_bytes.copy_from_slice(&bytes[8..16]);
    keys[i] = (u64::from_be_bytes(lo_bytes), u64::from_be_bytes(hi_bytes));
    i = i.strict_add(1);
  }
  RvScalar128RoundKeys { rk: keys }
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
