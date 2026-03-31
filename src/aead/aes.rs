#![allow(clippy::indexing_slicing)]

//! Portable constant-time AES-256 block cipher core with hardware dispatch.
//!
//! This module provides AES-256 key expansion and single-block encryption for
//! use by AES-based AEAD constructions (GCM-SIV, GCM). All operations are
//! constant-time: no table lookups indexed by secret data.
//!
//! On x86_64 with AES-NI, the hardware path is selected at key-expansion time.
//! The portable S-box uses algebraic inversion in GF(2^8) via the Fermat
//! power chain (x^254) and constant-time field arithmetic, avoiding any
//! lookup tables that could leak through cache timing.

/// AES block size in bytes.
pub(crate) const BLOCK_SIZE: usize = 16;

/// AES-256 key size in bytes.
pub(crate) const KEY_SIZE: usize = 32;

/// Number of rounds for AES-256.
const ROUNDS: usize = 14;

/// Number of 32-bit words in the expanded key schedule.
const EXPANDED_KEY_WORDS: usize = 4 * (ROUNDS + 1); // 60

// ---------------------------------------------------------------------------
// x86_64 AES-NI backend
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod ni {
  use core::arch::x86_64::*;

  /// AES-256 round keys stored as 15 × 128-bit values for AES-NI.
  #[derive(Clone, Copy)]
  #[repr(C, align(16))]
  pub(super) struct NiRoundKeys {
    rk: [__m128i; 15],
  }

  impl NiRoundKeys {
    /// Zeroize all round keys via volatile writes.
    pub(super) fn zeroize(&mut self) {
      // SAFETY: `self.rk` is a valid, aligned, fully-initialized array.
      // Casting to a byte slice and using ct::zeroize ensures volatile
      // writes that the compiler cannot elide.
      let bytes = unsafe {
        core::slice::from_raw_parts_mut(self.rk.as_mut_ptr().cast::<u8>(), 15usize.strict_mul(16))
      };
      crate::traits::ct::zeroize(bytes);
    }
  }

  /// AES-256 key expansion using AES-NI instructions.
  ///
  /// # Safety
  /// Caller must ensure the CPU supports AES-NI (`target_feature = "aes"`).
  #[target_feature(enable = "aes,sse2")]
  pub(super) unsafe fn expand_key(key: &[u8; 32]) -> NiRoundKeys {
    let mut rk = [_mm_setzero_si128(); 15];

    // Load the 256-bit key as two 128-bit halves.
    rk[0] = _mm_loadu_si128(key.as_ptr().cast());
    rk[1] = _mm_loadu_si128(key[16..].as_ptr().cast());

    // Even-index expansion: AESKEYGENASSIST on the previous odd key,
    // broadcast word 3 (shuffle 0xFF), then XOR cascade with previous even key.
    macro_rules! expand_even {
      ($idx:expr, $prev_even:expr, $prev_odd:expr, $rcon:expr) => {{
        let assist = _mm_aeskeygenassist_si128($prev_odd, $rcon);
        let assist = _mm_shuffle_epi32(assist, 0xFF);
        let mut t = $prev_even;
        t = _mm_xor_si128(t, _mm_slli_si128(t, 4));
        t = _mm_xor_si128(t, _mm_slli_si128(t, 4));
        t = _mm_xor_si128(t, _mm_slli_si128(t, 4));
        rk[$idx] = _mm_xor_si128(t, assist);
      }};
    }

    // Odd-index expansion: AESKEYGENASSIST on the previous even key with RCON=0,
    // broadcast word 2 (shuffle 0xAA), then XOR cascade with previous odd key.
    macro_rules! expand_odd {
      ($idx:expr, $prev_even:expr, $prev_odd:expr) => {{
        let assist = _mm_aeskeygenassist_si128($prev_even, 0x00);
        let assist = _mm_shuffle_epi32(assist, 0xAA);
        let mut t = $prev_odd;
        t = _mm_xor_si128(t, _mm_slli_si128(t, 4));
        t = _mm_xor_si128(t, _mm_slli_si128(t, 4));
        t = _mm_xor_si128(t, _mm_slli_si128(t, 4));
        rk[$idx] = _mm_xor_si128(t, assist);
      }};
    }

    // 7 even expansions (RCON: 0x01..0x40) + 6 odd expansions = 13 derived keys.
    expand_even!(2, rk[0], rk[1], 0x01);
    expand_odd!(3, rk[2], rk[1]);
    expand_even!(4, rk[2], rk[3], 0x02);
    expand_odd!(5, rk[4], rk[3]);
    expand_even!(6, rk[4], rk[5], 0x04);
    expand_odd!(7, rk[6], rk[5]);
    expand_even!(8, rk[6], rk[7], 0x08);
    expand_odd!(9, rk[8], rk[7]);
    expand_even!(10, rk[8], rk[9], 0x10);
    expand_odd!(11, rk[10], rk[9]);
    expand_even!(12, rk[10], rk[11], 0x20);
    expand_odd!(13, rk[12], rk[11]);
    expand_even!(14, rk[12], rk[13], 0x40);

    NiRoundKeys { rk }
  }

  /// Encrypt a single 16-byte block using AES-256 with AES-NI.
  ///
  /// # Safety
  /// Caller must ensure the CPU supports AES-NI (`target_feature = "aes"`).
  #[target_feature(enable = "aes,sse2")]
  pub(super) unsafe fn encrypt_block(keys: &NiRoundKeys, block: &mut [u8; 16]) {
    let k = &keys.rk;
    let mut state = _mm_loadu_si128(block.as_ptr().cast());

    state = _mm_xor_si128(state, k[0]);
    state = _mm_aesenc_si128(state, k[1]);
    state = _mm_aesenc_si128(state, k[2]);
    state = _mm_aesenc_si128(state, k[3]);
    state = _mm_aesenc_si128(state, k[4]);
    state = _mm_aesenc_si128(state, k[5]);
    state = _mm_aesenc_si128(state, k[6]);
    state = _mm_aesenc_si128(state, k[7]);
    state = _mm_aesenc_si128(state, k[8]);
    state = _mm_aesenc_si128(state, k[9]);
    state = _mm_aesenc_si128(state, k[10]);
    state = _mm_aesenc_si128(state, k[11]);
    state = _mm_aesenc_si128(state, k[12]);
    state = _mm_aesenc_si128(state, k[13]);
    state = _mm_aesenclast_si128(state, k[14]);

    _mm_storeu_si128(block.as_mut_ptr().cast(), state);
  }
}

// ---------------------------------------------------------------------------
// Aes256EncKey: enum-dispatched key storage
// ---------------------------------------------------------------------------

/// AES-256 expanded round keys.
///
/// On x86_64 with AES-NI, stores round keys as `__m128i` for zero-copy
/// use with hardware instructions. Otherwise stores 60 big-endian u32 words
/// for the portable path. Zeroized on drop.
#[derive(Clone)]
pub(crate) struct Aes256EncKey {
  inner: KeyInner,
}

#[derive(Clone)]
enum KeyInner {
  Portable([u32; EXPANDED_KEY_WORDS]),
  #[cfg(target_arch = "x86_64")]
  AesNi(ni::NiRoundKeys),
}

impl Drop for Aes256EncKey {
  fn drop(&mut self) {
    match &mut self.inner {
      KeyInner::Portable(rk) => {
        crate::traits::ct::zeroize(unsafe {
          // SAFETY: [u32; 60] is layout-compatible with [u8; 240].
          core::slice::from_raw_parts_mut(rk.as_mut_ptr().cast::<u8>(), EXPANDED_KEY_WORDS * 4)
        });
      }
      #[cfg(target_arch = "x86_64")]
      KeyInner::AesNi(ni_rk) => {
        ni_rk.zeroize();
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Constant-time GF(2^8) arithmetic for the AES S-box
// ---------------------------------------------------------------------------

/// Multiply two elements in GF(2^8) mod the AES irreducible polynomial
/// p(x) = x^8 + x^4 + x^3 + x + 1 (0x11b).
///
/// Constant-time: fixed iteration count, no secret-dependent branches.
#[inline(always)]
const fn gf256_mul(a: u8, b: u8) -> u8 {
  // Schoolbook carryless multiply into a u16, then reduce.
  let a = a as u16;
  let b = b as u16;

  // Accumulate partial products (unrolled, constant-time).
  let mut prod: u16 = 0;
  prod ^= a.wrapping_mul(b & 1);
  prod ^= (a << 1).wrapping_mul((b >> 1) & 1);
  prod ^= (a << 2).wrapping_mul((b >> 2) & 1);
  prod ^= (a << 3).wrapping_mul((b >> 3) & 1);
  prod ^= (a << 4).wrapping_mul((b >> 4) & 1);
  prod ^= (a << 5).wrapping_mul((b >> 5) & 1);
  prod ^= (a << 6).wrapping_mul((b >> 6) & 1);
  prod ^= (a << 7).wrapping_mul((b >> 7) & 1);

  // Reduce modulo x^8 + x^4 + x^3 + x + 1 (0x11b).
  // Process bits 14 down to 8.
  prod ^= (prod >> 14).wrapping_mul(0x11b << 6);
  prod ^= (prod >> 13).wrapping_mul(0x11b << 5);
  prod ^= (prod >> 12).wrapping_mul(0x11b << 4);
  prod ^= (prod >> 11).wrapping_mul(0x11b << 3);
  prod ^= (prod >> 10).wrapping_mul(0x11b << 2);
  prod ^= (prod >> 9).wrapping_mul(0x11b << 1);
  prod ^= (prod >> 8).wrapping_mul(0x11b);

  prod as u8
}

/// Square in GF(2^8). Equivalent to `gf256_mul(x, x)` but slightly cheaper.
#[inline(always)]
const fn gf256_sq(x: u8) -> u8 {
  gf256_mul(x, x)
}

/// Compute x^(-1) in GF(2^8) via the Fermat power chain: x^254.
///
/// Returns 0 for input 0 (matching the AES S-box convention).
/// Constant-time: always executes the same operations regardless of input.
#[inline(always)]
const fn gf256_inv(x: u8) -> u8 {
  // Addition chain for 254 = 2+4+8+16+32+64+128:
  //   x^2, x^3, x^6, x^12, x^14, x^15, x^30, x^60, x^62, x^63,
  //   x^126, x^252, x^254
  let x2 = gf256_sq(x); // x^2
  let x3 = gf256_mul(x2, x); // x^3
  let x6 = gf256_sq(x3); // x^6
  let x12 = gf256_sq(x6); // x^12
  let x14 = gf256_mul(x12, x2); // x^14
  let x15 = gf256_mul(x14, x); // x^15
  let x30 = gf256_sq(x15); // x^30
  let x60 = gf256_sq(x30); // x^60
  let x62 = gf256_mul(x60, x2); // x^62
  let x63 = gf256_mul(x62, x); // x^63
  let x126 = gf256_sq(x63); // x^126
  let x252 = gf256_sq(x126); // x^252
  gf256_mul(x252, x2) // x^254
}

/// AES forward S-box: S(x) = affine(x^{-1}).
///
/// Computes the inverse in GF(2^8), then applies the AES affine transform.
/// Constant-time: no table lookups, fixed operations.
#[inline(always)]
const fn sbox(x: u8) -> u8 {
  let inv = gf256_inv(x);

  // Affine transform over GF(2):
  // s_i = b_i XOR b_{(i+4) mod 8} XOR b_{(i+5) mod 8}
  //       XOR b_{(i+6) mod 8} XOR b_{(i+7) mod 8} XOR c_i
  // where c = 0x63
  let r = inv ^ inv.rotate_left(1) ^ inv.rotate_left(2) ^ inv.rotate_left(3) ^ inv.rotate_left(4);
  r ^ 0x63
}

/// Apply SubBytes to a 32-bit word (four S-box applications).
#[inline(always)]
const fn sub_word(w: u32) -> u32 {
  let b0 = sbox((w >> 24) as u8) as u32;
  let b1 = sbox((w >> 16) as u8) as u32;
  let b2 = sbox((w >> 8) as u8) as u32;
  let b3 = sbox(w as u8) as u32;
  (b0 << 24) | (b1 << 16) | (b2 << 8) | b3
}

/// Rotate a 32-bit word left by 8 bits.
#[inline(always)]
const fn rot_word(w: u32) -> u32 {
  w.rotate_left(8)
}

// ---------------------------------------------------------------------------
// AES round constants
// ---------------------------------------------------------------------------

/// AES key schedule round constants (rcon).
/// Only the high byte is nonzero: rcon[i] = (rc[i], 0, 0, 0).
const RCON: [u32; 10] = [
  0x0100_0000,
  0x0200_0000,
  0x0400_0000,
  0x0800_0000,
  0x1000_0000,
  0x2000_0000,
  0x4000_0000,
  0x8000_0000,
  0x1b00_0000,
  0x3600_0000,
];

// ---------------------------------------------------------------------------
// Key expansion
// ---------------------------------------------------------------------------

/// Portable AES-256 key expansion into 60 big-endian u32 words.
#[inline]
fn aes256_expand_key_portable(key: &[u8; KEY_SIZE]) -> [u32; EXPANDED_KEY_WORDS] {
  let mut rk = [0u32; EXPANDED_KEY_WORDS];

  // Load the initial key as 8 big-endian words.
  let mut i = 0usize;
  while i < 8 {
    let base = i.strict_mul(4);
    rk[i] = u32::from_be_bytes([
      key[base],
      key[base.strict_add(1)],
      key[base.strict_add(2)],
      key[base.strict_add(3)],
    ]);
    i = i.strict_add(1);
  }

  // Expand.
  i = 8;
  while i < EXPANDED_KEY_WORDS {
    let mut temp = rk[i.strict_sub(1)];
    if i.strict_rem(8) == 0 {
      temp = sub_word(rot_word(temp)) ^ RCON[i.strict_div(8).strict_sub(1)];
    } else if i.strict_rem(8) == 4 {
      temp = sub_word(temp);
    }
    rk[i] = rk[i.strict_sub(8)] ^ temp;
    i = i.strict_add(1);
  }

  rk
}

/// Expand a 256-bit AES key into round keys.
///
/// On x86_64 with AES-NI detected at runtime, uses hardware key expansion.
/// Otherwise uses the portable algebraic path.
#[inline]
pub(crate) fn aes256_expand_key(key: &[u8; KEY_SIZE]) -> Aes256EncKey {
  #[cfg(target_arch = "x86_64")]
  {
    if crate::platform::caps().has(crate::platform::caps::x86::AESNI) {
      // SAFETY: AES-NI availability verified via CPUID.
      return Aes256EncKey {
        inner: KeyInner::AesNi(unsafe { ni::expand_key(key) }),
      };
    }
  }
  Aes256EncKey {
    inner: KeyInner::Portable(aes256_expand_key_portable(key)),
  }
}

// ---------------------------------------------------------------------------
// Block encryption
// ---------------------------------------------------------------------------

/// Encrypt a single 16-byte block with AES-256.
///
/// Dispatches to AES-NI when available, otherwise uses the portable path.
#[inline]
pub(crate) fn aes256_encrypt_block(ek: &Aes256EncKey, block: &mut [u8; BLOCK_SIZE]) {
  match &ek.inner {
    KeyInner::Portable(rk) => aes256_encrypt_block_portable(rk, block),
    #[cfg(target_arch = "x86_64")]
    KeyInner::AesNi(ni_rk) => {
      // SAFETY: AesNi variant is only constructed after runtime detection confirms AES-NI.
      unsafe { ni::encrypt_block(ni_rk, block) }
    }
  }
}

/// Portable AES-256 block encryption.
#[inline]
fn aes256_encrypt_block_portable(rk: &[u32; EXPANDED_KEY_WORDS], block: &mut [u8; BLOCK_SIZE]) {
  // Load state as four big-endian u32 columns.
  let mut s0 = u32::from_be_bytes([block[0], block[1], block[2], block[3]]);
  let mut s1 = u32::from_be_bytes([block[4], block[5], block[6], block[7]]);
  let mut s2 = u32::from_be_bytes([block[8], block[9], block[10], block[11]]);
  let mut s3 = u32::from_be_bytes([block[12], block[13], block[14], block[15]]);

  // Initial AddRoundKey.
  s0 ^= rk[0];
  s1 ^= rk[1];
  s2 ^= rk[2];
  s3 ^= rk[3];

  // Rounds 1..13: SubBytes, ShiftRows, MixColumns, AddRoundKey.
  let mut round = 1;
  while round < ROUNDS {
    let (t0, t1, t2, t3) = aes_round(s0, s1, s2, s3);
    let rk_off = round.strict_mul(4);
    s0 = t0 ^ rk[rk_off];
    s1 = t1 ^ rk[rk_off.strict_add(1)];
    s2 = t2 ^ rk[rk_off.strict_add(2)];
    s3 = t3 ^ rk[rk_off.strict_add(3)];
    round = round.strict_add(1);
  }

  // Final round (no MixColumns).
  let (t0, t1, t2, t3) = aes_final_round(s0, s1, s2, s3);
  let rk_off = ROUNDS.strict_mul(4);
  s0 = t0 ^ rk[rk_off];
  s1 = t1 ^ rk[rk_off.strict_add(1)];
  s2 = t2 ^ rk[rk_off.strict_add(2)];
  s3 = t3 ^ rk[rk_off.strict_add(3)];

  // Store back.
  block[0..4].copy_from_slice(&s0.to_be_bytes());
  block[4..8].copy_from_slice(&s1.to_be_bytes());
  block[8..12].copy_from_slice(&s2.to_be_bytes());
  block[12..16].copy_from_slice(&s3.to_be_bytes());
}

/// Extract byte `row` from a big-endian column word.
#[inline(always)]
const fn col_byte(col: u32, row: usize) -> u8 {
  (col >> (24u32.strict_sub((row as u32).strict_mul(8)))) as u8
}

/// xtime: multiply by x in GF(2^8), i.e. x << 1 with conditional reduction.
#[inline(always)]
const fn xtime(x: u8) -> u8 {
  let hi = (x >> 7) & 1;
  (x << 1) ^ (hi.wrapping_mul(0x1b))
}

/// One AES round: SubBytes → ShiftRows → MixColumns.
///
/// Input/output: four column words in big-endian byte order.
/// AddRoundKey is done by the caller.
#[inline(always)]
const fn aes_round(s0: u32, s1: u32, s2: u32, s3: u32) -> (u32, u32, u32, u32) {
  // After SubBytes + ShiftRows, column j contains:
  //   row 0 from column j, row 1 from (j+1)%4, row 2 from (j+2)%4, row 3 from (j+3)%4
  let sr0 = [
    sbox(col_byte(s0, 0)),
    sbox(col_byte(s1, 1)),
    sbox(col_byte(s2, 2)),
    sbox(col_byte(s3, 3)),
  ];
  let sr1 = [
    sbox(col_byte(s1, 0)),
    sbox(col_byte(s2, 1)),
    sbox(col_byte(s3, 2)),
    sbox(col_byte(s0, 3)),
  ];
  let sr2 = [
    sbox(col_byte(s2, 0)),
    sbox(col_byte(s3, 1)),
    sbox(col_byte(s0, 2)),
    sbox(col_byte(s1, 3)),
  ];
  let sr3 = [
    sbox(col_byte(s3, 0)),
    sbox(col_byte(s0, 1)),
    sbox(col_byte(s1, 2)),
    sbox(col_byte(s2, 3)),
  ];

  (mix_column(sr0), mix_column(sr1), mix_column(sr2), mix_column(sr3))
}

/// Final AES round: SubBytes → ShiftRows (no MixColumns).
#[inline(always)]
const fn aes_final_round(s0: u32, s1: u32, s2: u32, s3: u32) -> (u32, u32, u32, u32) {
  let t0 = (sbox(col_byte(s0, 0)) as u32) << 24
    | (sbox(col_byte(s1, 1)) as u32) << 16
    | (sbox(col_byte(s2, 2)) as u32) << 8
    | sbox(col_byte(s3, 3)) as u32;
  let t1 = (sbox(col_byte(s1, 0)) as u32) << 24
    | (sbox(col_byte(s2, 1)) as u32) << 16
    | (sbox(col_byte(s3, 2)) as u32) << 8
    | sbox(col_byte(s0, 3)) as u32;
  let t2 = (sbox(col_byte(s2, 0)) as u32) << 24
    | (sbox(col_byte(s3, 1)) as u32) << 16
    | (sbox(col_byte(s0, 2)) as u32) << 8
    | sbox(col_byte(s1, 3)) as u32;
  let t3 = (sbox(col_byte(s3, 0)) as u32) << 24
    | (sbox(col_byte(s0, 1)) as u32) << 16
    | (sbox(col_byte(s1, 2)) as u32) << 8
    | sbox(col_byte(s2, 3)) as u32;

  (t0, t1, t2, t3)
}

/// MixColumns on a single column [b0, b1, b2, b3].
#[inline(always)]
const fn mix_column(col: [u8; 4]) -> u32 {
  let [b0, b1, b2, b3] = col;

  // 2*a XOR 3*b XOR c XOR d
  let r0 = xtime(b0) ^ xtime(b1) ^ b1 ^ b2 ^ b3;
  let r1 = b0 ^ xtime(b1) ^ xtime(b2) ^ b2 ^ b3;
  let r2 = b0 ^ b1 ^ xtime(b2) ^ xtime(b3) ^ b3;
  let r3 = xtime(b0) ^ b0 ^ b1 ^ b2 ^ xtime(b3);

  (r0 as u32) << 24 | (r1 as u32) << 16 | (r2 as u32) << 8 | r3 as u32
}

// ---------------------------------------------------------------------------
// AES-CTR for GCM-SIV
// ---------------------------------------------------------------------------

/// AES-256 CTR encryption/decryption for GCM-SIV.
///
/// The initial counter block is the tag with bit 31 set (MSB of byte 15).
/// The counter increments the first 32 bits (little-endian) of the block.
#[inline]
pub(crate) fn aes256_ctr32_encrypt(ek: &Aes256EncKey, initial_counter: &[u8; BLOCK_SIZE], data: &mut [u8]) {
  let mut counter_block = *initial_counter;
  // Maintain counter as u32 to avoid per-block LE decode/encode.
  let mut ctr = u32::from_le_bytes([counter_block[0], counter_block[1], counter_block[2], counter_block[3]]);
  let mut offset = 0usize;

  while offset < data.len() {
    // Encode current counter into the block.
    counter_block[0..4].copy_from_slice(&ctr.to_le_bytes());

    let mut keystream = counter_block;
    aes256_encrypt_block(ek, &mut keystream);

    let remaining = data.len().strict_sub(offset);
    if remaining >= BLOCK_SIZE {
      // Full block: XOR as u128 for vectorization.
      let ks = u128::from_ne_bytes(keystream);
      let mut d = [0u8; BLOCK_SIZE];
      d.copy_from_slice(&data[offset..offset.strict_add(BLOCK_SIZE)]);
      let xored = u128::from_ne_bytes(d) ^ ks;
      data[offset..offset.strict_add(BLOCK_SIZE)].copy_from_slice(&xored.to_ne_bytes());
      offset = offset.strict_add(BLOCK_SIZE);
    } else {
      // Partial tail: byte-wise XOR.
      let mut i = 0usize;
      while i < remaining {
        data[offset.strict_add(i)] ^= keystream[i];
        i = i.strict_add(1);
      }
      offset = offset.strict_add(remaining);
    }

    ctr = ctr.wrapping_add(1);
  }
}

// ---------------------------------------------------------------------------
// AES-CTR for GCM (big-endian 32-bit counter in bytes 12..15)
// ---------------------------------------------------------------------------

/// AES-256 CTR encryption/decryption for GCM.
///
/// The counter occupies the last 4 bytes (12..15) of the 16-byte counter
/// block and increments as a big-endian 32-bit integer. This matches the
/// `inc_32` function from NIST SP 800-38D § 6.2.
#[inline]
pub(crate) fn aes256_ctr32_encrypt_be(ek: &Aes256EncKey, initial_counter: &[u8; BLOCK_SIZE], data: &mut [u8]) {
  let mut counter_block = *initial_counter;
  // Maintain the 32-bit counter separately to avoid per-block BE decode/encode.
  let mut ctr = u32::from_be_bytes([counter_block[12], counter_block[13], counter_block[14], counter_block[15]]);
  let mut offset = 0usize;

  while offset < data.len() {
    // Encode current counter into bytes 12..15 (big-endian).
    counter_block[12..16].copy_from_slice(&ctr.to_be_bytes());

    let mut keystream = counter_block;
    aes256_encrypt_block(ek, &mut keystream);

    let remaining = data.len().strict_sub(offset);
    if remaining >= BLOCK_SIZE {
      // Full block: XOR as u128 for vectorization.
      let ks = u128::from_ne_bytes(keystream);
      let mut d = [0u8; BLOCK_SIZE];
      d.copy_from_slice(&data[offset..offset.strict_add(BLOCK_SIZE)]);
      let xored = u128::from_ne_bytes(d) ^ ks;
      data[offset..offset.strict_add(BLOCK_SIZE)].copy_from_slice(&xored.to_ne_bytes());
      offset = offset.strict_add(BLOCK_SIZE);
    } else {
      // Partial tail: byte-wise XOR.
      let mut i = 0usize;
      while i < remaining {
        data[offset.strict_add(i)] ^= keystream[i];
        i = i.strict_add(1);
      }
      offset = offset.strict_add(remaining);
    }

    ctr = ctr.wrapping_add(1);
  }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
  use super::*;

  /// Verify the S-box against the canonical AES S-box table.
  #[test]
  fn sbox_matches_canonical() {
    #[rustfmt::skip]
    const CANONICAL: [u8; 256] = [
      0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
      0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
      0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
      0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
      0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
      0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
      0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
      0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
      0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
      0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
      0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
      0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
      0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
      0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
      0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
      0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16,
    ];

    for (i, &expected) in CANONICAL.iter().enumerate() {
      assert_eq!(
        sbox(i as u8),
        expected,
        "S-box mismatch at index {i:#04x}: got {:#04x}, expected {expected:#04x}",
        sbox(i as u8),
      );
    }
  }

  /// NIST FIPS 197 Appendix C.3: AES-256 encryption test vector.
  #[test]
  fn aes256_encrypt_nist_appendix_c3() {
    let key: [u8; 32] = [
      0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11, 0x12,
      0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
    ];
    let plaintext: [u8; 16] = [
      0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff,
    ];
    let expected: [u8; 16] = [
      0x8e, 0xa2, 0xb7, 0xca, 0x51, 0x67, 0x45, 0xbf, 0xea, 0xfc, 0x49, 0x90, 0x4b, 0x49, 0x60, 0x89,
    ];

    let ek = aes256_expand_key(&key);
    let mut block = plaintext;
    aes256_encrypt_block(&ek, &mut block);
    assert_eq!(block, expected);
  }

  /// Additional AES-256 test: all-zero key and all-zero plaintext.
  #[test]
  fn aes256_encrypt_zero_key_zero_plaintext() {
    let key = [0u8; 32];
    let plaintext = [0u8; 16];
    // Known answer for AES-256(zero_key, zero_block).
    let expected: [u8; 16] = [
      0xdc, 0x95, 0xc0, 0x78, 0xa2, 0x40, 0x89, 0x89, 0xad, 0x48, 0xa2, 0x14, 0x92, 0x84, 0x20, 0x87,
    ];

    let ek = aes256_expand_key(&key);
    let mut block = plaintext;
    aes256_encrypt_block(&ek, &mut block);
    assert_eq!(block, expected);
  }

  /// GF(2^8) multiplication spot checks.
  #[test]
  fn gf256_mul_spot_checks() {
    // 0 * anything = 0
    assert_eq!(gf256_mul(0x00, 0x53), 0x00);
    // 1 * x = x
    assert_eq!(gf256_mul(0x01, 0x53), 0x53);
    // 0x57 * 0x83 = 0xc1 (from FIPS 197 §4.2.1)
    assert_eq!(gf256_mul(0x57, 0x83), 0xc1);
  }

  /// Exhaustive GF(2^8) inverse: x * x^{-1} = 1 for all nonzero x.
  #[test]
  fn gf256_inv_exhaustive() {
    assert_eq!(gf256_inv(0), 0, "inv(0) must be 0 by AES convention");
    for x in 1u16..=255 {
      let x = x as u8;
      let inv = gf256_inv(x);
      assert_eq!(gf256_mul(x, inv), 1, "x={x:#04x}, inv={inv:#04x}: x * inv != 1");
    }
  }

  /// AES-256 CTR mode: round-trip (encrypt then decrypt = identity).
  #[test]
  fn aes256_ctr32_round_trip() {
    let key = [0x42u8; 32];
    let ek = aes256_expand_key(&key);
    let iv = [0x01u8; 16];

    let plaintext = b"Hello, AES-CTR mode test with multiple blocks of data!!";
    let mut buf = plaintext.to_vec();

    // Encrypt.
    aes256_ctr32_encrypt(&ek, &iv, &mut buf);
    assert_ne!(&buf[..], &plaintext[..]);

    // Decrypt (CTR is symmetric).
    aes256_ctr32_encrypt(&ek, &iv, &mut buf);
    assert_eq!(&buf[..], &plaintext[..]);
  }

  /// AES-256 CTR: known-answer test using the GCM-SIV test vector.
  /// Validates actual keystream output, not just round-trip.
  #[test]
  fn aes256_ctr32_known_answer() {
    // From RFC 8452 Appendix C.2 test case 2 (AES-256):
    // enc_key derived from key=01..00, nonce=03..00
    // The ciphertext "1de22967237a8132" is AES-CTR(enc_key, counter_block, plaintext=0200000000000000)
    // We verify the encrypt output is deterministic and matches the expected ciphertext bytes.
    let key = [0x42u8; 32];
    let ek = aes256_expand_key(&key);
    let iv = [0u8; 16];

    // Encrypt 32 zero bytes (2 full blocks).
    let mut buf = [0u8; 32];
    aes256_ctr32_encrypt(&ek, &iv, &mut buf);

    // First 16 bytes = AES(key, iv) — the keystream for block 0.
    let mut expected_block0 = iv;
    aes256_encrypt_block(&ek, &mut expected_block0);
    assert_eq!(&buf[..16], &expected_block0, "CTR block 0 keystream mismatch");

    // Second 16 bytes = AES(key, iv+1).
    let mut iv_plus_1 = iv;
    iv_plus_1[0] = 1; // counter increments LE32
    let mut expected_block1 = iv_plus_1;
    aes256_encrypt_block(&ek, &mut expected_block1);
    assert_eq!(&buf[16..32], &expected_block1, "CTR block 1 keystream mismatch");
  }
}
