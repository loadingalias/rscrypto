use core::arch::riscv64::{aes64es, aes64esm};

/// AES-256 round keys stored as 15 pairs of 64-bit halves for scalar AES.
///
/// Each round key is kept in canonical AES byte order and split into the
/// upper/lower 64-bit halves consumed by the RV64 scalar AES instructions.
#[derive(Clone)]
#[repr(C, align(16))]
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
  let mut lo_bytes = [0u8; 8];
  lo_bytes.copy_from_slice(&block[0..8]);
  let mut lo = u64::from_be_bytes(lo_bytes);

  let mut hi_bytes = [0u8; 8];
  hi_bytes.copy_from_slice(&block[8..16]);
  let mut hi = u64::from_be_bytes(hi_bytes);

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

  block[0..8].copy_from_slice(&lo.to_be_bytes());
  block[8..16].copy_from_slice(&hi.to_be_bytes());
}
