use core::arch::x86_64::*;

/// AES-256 round keys stored as 15 × 128-bit values for AES-NI.
#[repr(C, align(16))]
pub(super) struct NiRoundKeys {
  rk: [__m128i; 15],
}

impl NiRoundKeys {
  #[cfg(all(target_os = "linux", any(feature = "aes-gcm", feature = "aes-gcm-siv")))]
  #[inline]
  pub(super) fn as_ptr(&self) -> *const u8 {
    self.rk.as_ptr().cast()
  }

  /// Zeroize all round keys via volatile writes.
  pub(super) fn zeroize(&mut self) {
    // SAFETY: `self.rk` is a valid, aligned, fully-initialized [__m128i; 15].
    // Reinterpreting as a byte slice for volatile zeroization is sound.
    let bytes = unsafe { core::slice::from_raw_parts_mut(self.rk.as_mut_ptr().cast::<u8>(), 15usize.strict_mul(16)) };
    crate::traits::ct::zeroize(bytes);
  }
}

impl Drop for NiRoundKeys {
  fn drop(&mut self) {
    self.zeroize();
  }
}

/// AES-256 key expansion using AES-NI instructions.
///
/// # Safety
/// Caller must ensure the CPU supports AES-NI (`target_feature = "aes"`).
#[target_feature(enable = "aes,sse2")]
pub(super) unsafe fn expand_key(key: &[u8; 32]) -> NiRoundKeys {
  // SAFETY: target_feature gate guarantees AES-NI + SSE2.
  unsafe {
    let mut rk = [_mm_setzero_si128(); 15];

    rk[0] = _mm_loadu_si128(key.as_ptr().cast());
    rk[1] = _mm_loadu_si128(key[16..].as_ptr().cast());

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
}

/// Encrypt 4 blocks in parallel using VAES-512 (14-round AES-256).
///
/// Takes 4 pre-loaded 128-bit blocks in a `__m512i`, broadcasts each
/// round key, and applies 14 VAES rounds. Returns the encrypted blocks.
///
/// # Safety
/// Caller must ensure AVX-512F + AVX-512VL + VAES + AES + SSE2.
#[cfg(any(feature = "aes-gcm", feature = "aes-gcm-siv"))]
#[target_feature(enable = "aes,sse2,avx512f,avx512vl,vaes")]
#[inline]
pub(super) unsafe fn encrypt_4blocks(keys: &NiRoundKeys, blocks: __m512i) -> __m512i {
  let k = &keys.rk;
  let mut state = _mm512_xor_si512(blocks, _mm512_broadcast_i32x4(k[0]));
  state = _mm512_aesenc_epi128(state, _mm512_broadcast_i32x4(k[1]));
  state = _mm512_aesenc_epi128(state, _mm512_broadcast_i32x4(k[2]));
  state = _mm512_aesenc_epi128(state, _mm512_broadcast_i32x4(k[3]));
  state = _mm512_aesenc_epi128(state, _mm512_broadcast_i32x4(k[4]));
  state = _mm512_aesenc_epi128(state, _mm512_broadcast_i32x4(k[5]));
  state = _mm512_aesenc_epi128(state, _mm512_broadcast_i32x4(k[6]));
  state = _mm512_aesenc_epi128(state, _mm512_broadcast_i32x4(k[7]));
  state = _mm512_aesenc_epi128(state, _mm512_broadcast_i32x4(k[8]));
  state = _mm512_aesenc_epi128(state, _mm512_broadcast_i32x4(k[9]));
  state = _mm512_aesenc_epi128(state, _mm512_broadcast_i32x4(k[10]));
  state = _mm512_aesenc_epi128(state, _mm512_broadcast_i32x4(k[11]));
  state = _mm512_aesenc_epi128(state, _mm512_broadcast_i32x4(k[12]));
  state = _mm512_aesenc_epi128(state, _mm512_broadcast_i32x4(k[13]));
  _mm512_aesenclast_epi128(state, _mm512_broadcast_i32x4(k[14]))
}

/// Encrypt 4 blocks in parallel using classic AES-NI (14-round AES-256).
///
/// # Safety
/// Caller must ensure AES-NI and SSE2 are available.
#[cfg(feature = "aes-gcm")]
#[target_feature(enable = "aes,sse2")]
#[inline]
pub(super) unsafe fn encrypt_4blocks_aesni(
  keys: &NiRoundKeys,
  b0: __m128i,
  b1: __m128i,
  b2: __m128i,
  b3: __m128i,
) -> (__m128i, __m128i, __m128i, __m128i) {
  let k = &keys.rk;
  let rk0 = k[0];
  let mut s0 = _mm_xor_si128(b0, rk0);
  let mut s1 = _mm_xor_si128(b1, rk0);
  let mut s2 = _mm_xor_si128(b2, rk0);
  let mut s3 = _mm_xor_si128(b3, rk0);

  macro_rules! round {
    ($idx:expr) => {{
      let rk = k[$idx];
      s0 = _mm_aesenc_si128(s0, rk);
      s1 = _mm_aesenc_si128(s1, rk);
      s2 = _mm_aesenc_si128(s2, rk);
      s3 = _mm_aesenc_si128(s3, rk);
    }};
  }

  round!(1);
  round!(2);
  round!(3);
  round!(4);
  round!(5);
  round!(6);
  round!(7);
  round!(8);
  round!(9);
  round!(10);
  round!(11);
  round!(12);
  round!(13);

  let rk14 = k[14];
  (
    _mm_aesenclast_si128(s0, rk14),
    _mm_aesenclast_si128(s1, rk14),
    _mm_aesenclast_si128(s2, rk14),
    _mm_aesenclast_si128(s3, rk14),
  )
}

/// Encrypt 16 blocks as four independent VAES-512 dependency chains.
///
/// # Safety
/// Caller must ensure AVX-512F + AVX-512VL + VAES + AES + SSE2.
#[cfg(any(feature = "aes-gcm", feature = "aes-gcm-siv"))]
#[target_feature(enable = "aes,sse2,avx512f,avx512vl,vaes")]
#[inline]
pub(super) unsafe fn encrypt_16blocks(
  keys: &NiRoundKeys,
  b0: __m512i,
  b1: __m512i,
  b2: __m512i,
  b3: __m512i,
) -> (__m512i, __m512i, __m512i, __m512i) {
  let k = &keys.rk;
  let rk0 = _mm512_broadcast_i32x4(k[0]);
  let mut s0 = _mm512_xor_si512(b0, rk0);
  let mut s1 = _mm512_xor_si512(b1, rk0);
  let mut s2 = _mm512_xor_si512(b2, rk0);
  let mut s3 = _mm512_xor_si512(b3, rk0);

  macro_rules! round {
    ($idx:expr) => {{
      let rk = _mm512_broadcast_i32x4(k[$idx]);
      s0 = _mm512_aesenc_epi128(s0, rk);
      s1 = _mm512_aesenc_epi128(s1, rk);
      s2 = _mm512_aesenc_epi128(s2, rk);
      s3 = _mm512_aesenc_epi128(s3, rk);
    }};
  }

  round!(1);
  round!(2);
  round!(3);
  round!(4);
  round!(5);
  round!(6);
  round!(7);
  round!(8);
  round!(9);
  round!(10);
  round!(11);
  round!(12);
  round!(13);

  let rk14 = _mm512_broadcast_i32x4(k[14]);
  (
    _mm512_aesenclast_epi128(s0, rk14),
    _mm512_aesenclast_epi128(s1, rk14),
    _mm512_aesenclast_epi128(s2, rk14),
    _mm512_aesenclast_epi128(s3, rk14),
  )
}

/// Encrypts one AES-256 state with the expanded AES-NI schedule.
///
/// # Safety
///
/// The caller must ensure the CPU supports the x86 `aes` and `sse2` features.
#[target_feature(enable = "aes,sse2")]
#[inline]
unsafe fn encrypt_state(keys: &NiRoundKeys, mut state: __m128i) -> __m128i {
  let k = &keys.rk;
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
  _mm_aesenclast_si128(state, k[14])
}

/// Encrypt a single 16-byte block using AES-256 with AES-NI.
///
/// # Safety
/// Caller must ensure the CPU supports AES-NI (`target_feature = "aes"`).
#[target_feature(enable = "aes,sse2")]
pub(super) unsafe fn encrypt_block(keys: &NiRoundKeys, block: &mut [u8; 16]) {
  // SAFETY: target_feature gate guarantees AES-NI + SSE2. The fixed-size block
  // supplies one complete unaligned load and store.
  unsafe {
    let state = encrypt_state(keys, _mm_loadu_si128(block.as_ptr().cast()));
    _mm_storeu_si128(block.as_mut_ptr().cast(), state);
  }
}

#[cfg(feature = "aes-gcm")]
/// Encrypt one block and return only its first five bytes.
///
/// The unused eleven output bytes remain in the SIMD state and are never
/// materialized in addressable memory.
///
/// # Safety
/// Caller must ensure the CPU supports AES-NI (`target_feature = "aes"`).
#[target_feature(enable = "aes,sse2")]
pub(super) unsafe fn encrypt_block_prefix_5(keys: &NiRoundKeys, block: &[u8; 16]) -> [u8; 5] {
  // SAFETY: target_feature gate guarantees AES-NI + SSE2. The fixed-size input
  // supplies one complete unaligned block load; lane extraction reads only the
  // low eight bytes of the initialized encrypted state.
  unsafe {
    let state = encrypt_state(keys, _mm_loadu_si128(block.as_ptr().cast()));
    let prefix = _mm_cvtsi128_si64(state).cast_unsigned().to_le_bytes();
    [prefix[0], prefix[1], prefix[2], prefix[3], prefix[4]]
  }
}

// AES-128 (11 round keys, 10 rounds)

/// AES-128 round keys stored as 11 × 128-bit values for AES-NI.
#[repr(C, align(16))]
pub(super) struct Ni128RoundKeys {
  rk: [__m128i; 11],
}

impl Ni128RoundKeys {
  #[cfg(all(target_os = "linux", any(feature = "aes-gcm", feature = "aes-gcm-siv")))]
  #[inline]
  pub(super) fn as_ptr(&self) -> *const u8 {
    self.rk.as_ptr().cast()
  }

  /// Zeroize all round keys via volatile writes.
  pub(super) fn zeroize(&mut self) {
    // SAFETY: `self.rk` is a valid, aligned, fully-initialized [__m128i; 11].
    let bytes = unsafe { core::slice::from_raw_parts_mut(self.rk.as_mut_ptr().cast::<u8>(), 11usize.strict_mul(16)) };
    crate::traits::ct::zeroize(bytes);
  }
}

impl Drop for Ni128RoundKeys {
  fn drop(&mut self) {
    self.zeroize();
  }
}

/// AES-128 key expansion using AES-NI instructions.
///
/// # Safety
/// Caller must ensure the CPU supports AES-NI (`target_feature = "aes"`).
#[target_feature(enable = "aes,sse2")]
pub(super) unsafe fn expand_key_128(key: &[u8; 16]) -> Ni128RoundKeys {
  // SAFETY: target_feature gate guarantees AES-NI + SSE2.
  unsafe {
    let mut rk = [_mm_setzero_si128(); 11];

    rk[0] = _mm_loadu_si128(key.as_ptr().cast());

    // AES-128 key schedule expands each round key from the previous one
    // using AESKEYGENASSIST with the rcon for that round.
    macro_rules! expand_step {
      ($idx:expr, $prev:expr, $rcon:expr) => {{
        let assist = _mm_aeskeygenassist_si128($prev, $rcon);
        let assist = _mm_shuffle_epi32(assist, 0xFF);
        let mut t = $prev;
        t = _mm_xor_si128(t, _mm_slli_si128(t, 4));
        t = _mm_xor_si128(t, _mm_slli_si128(t, 4));
        t = _mm_xor_si128(t, _mm_slli_si128(t, 4));
        rk[$idx] = _mm_xor_si128(t, assist);
      }};
    }

    expand_step!(1, rk[0], 0x01);
    expand_step!(2, rk[1], 0x02);
    expand_step!(3, rk[2], 0x04);
    expand_step!(4, rk[3], 0x08);
    expand_step!(5, rk[4], 0x10);
    expand_step!(6, rk[5], 0x20);
    expand_step!(7, rk[6], 0x40);
    expand_step!(8, rk[7], 0x80);
    expand_step!(9, rk[8], 0x1b);
    expand_step!(10, rk[9], 0x36);

    Ni128RoundKeys { rk }
  }
}

/// Encrypt 4 blocks in parallel using VAES-512 (10-round AES-128).
///
/// # Safety
/// Caller must ensure AVX-512F + AVX-512VL + VAES + AES + SSE2.
#[cfg(any(feature = "aes-gcm", feature = "aes-gcm-siv"))]
#[target_feature(enable = "aes,sse2,avx512f,avx512vl,vaes")]
#[inline]
pub(super) unsafe fn encrypt_4blocks_128(keys: &Ni128RoundKeys, blocks: __m512i) -> __m512i {
  let k = &keys.rk;
  let mut state = _mm512_xor_si512(blocks, _mm512_broadcast_i32x4(k[0]));
  state = _mm512_aesenc_epi128(state, _mm512_broadcast_i32x4(k[1]));
  state = _mm512_aesenc_epi128(state, _mm512_broadcast_i32x4(k[2]));
  state = _mm512_aesenc_epi128(state, _mm512_broadcast_i32x4(k[3]));
  state = _mm512_aesenc_epi128(state, _mm512_broadcast_i32x4(k[4]));
  state = _mm512_aesenc_epi128(state, _mm512_broadcast_i32x4(k[5]));
  state = _mm512_aesenc_epi128(state, _mm512_broadcast_i32x4(k[6]));
  state = _mm512_aesenc_epi128(state, _mm512_broadcast_i32x4(k[7]));
  state = _mm512_aesenc_epi128(state, _mm512_broadcast_i32x4(k[8]));
  state = _mm512_aesenc_epi128(state, _mm512_broadcast_i32x4(k[9]));
  _mm512_aesenclast_epi128(state, _mm512_broadcast_i32x4(k[10]))
}

/// Encrypt 4 blocks in parallel using classic AES-NI (10-round AES-128).
///
/// # Safety
/// Caller must ensure AES-NI and SSE2 are available.
#[cfg(any(test, feature = "aes-gcm", feature = "aes-gcm-siv", feature = "aes-siv"))]
#[target_feature(enable = "aes,sse2")]
#[inline]
pub(super) unsafe fn encrypt_4blocks_128_aesni(
  keys: &Ni128RoundKeys,
  b0: __m128i,
  b1: __m128i,
  b2: __m128i,
  b3: __m128i,
) -> (__m128i, __m128i, __m128i, __m128i) {
  let k = &keys.rk;
  let rk0 = k[0];
  let mut s0 = _mm_xor_si128(b0, rk0);
  let mut s1 = _mm_xor_si128(b1, rk0);
  let mut s2 = _mm_xor_si128(b2, rk0);
  let mut s3 = _mm_xor_si128(b3, rk0);

  macro_rules! round {
    ($idx:expr) => {{
      let rk = k[$idx];
      s0 = _mm_aesenc_si128(s0, rk);
      s1 = _mm_aesenc_si128(s1, rk);
      s2 = _mm_aesenc_si128(s2, rk);
      s3 = _mm_aesenc_si128(s3, rk);
    }};
  }

  round!(1);
  round!(2);
  round!(3);
  round!(4);
  round!(5);
  round!(6);
  round!(7);
  round!(8);
  round!(9);

  let rk10 = k[10];
  (
    _mm_aesenclast_si128(s0, rk10),
    _mm_aesenclast_si128(s1, rk10),
    _mm_aesenclast_si128(s2, rk10),
    _mm_aesenclast_si128(s3, rk10),
  )
}

/// Encrypt 16 AES-128 blocks as four independent VAES-512 dependency chains.
///
/// # Safety
/// Caller must ensure AVX-512F + AVX-512VL + VAES + AES + SSE2.
#[cfg(any(test, feature = "aes-gcm", feature = "aes-gcm-siv", feature = "aes-siv"))]
#[target_feature(enable = "aes,sse2,avx512f,avx512vl,vaes")]
#[inline]
pub(super) unsafe fn encrypt_16blocks_128(
  keys: &Ni128RoundKeys,
  b0: __m512i,
  b1: __m512i,
  b2: __m512i,
  b3: __m512i,
) -> (__m512i, __m512i, __m512i, __m512i) {
  let k = &keys.rk;
  let rk0 = _mm512_broadcast_i32x4(k[0]);
  let mut s0 = _mm512_xor_si512(b0, rk0);
  let mut s1 = _mm512_xor_si512(b1, rk0);
  let mut s2 = _mm512_xor_si512(b2, rk0);
  let mut s3 = _mm512_xor_si512(b3, rk0);

  macro_rules! round {
    ($idx:expr) => {{
      let rk = _mm512_broadcast_i32x4(k[$idx]);
      s0 = _mm512_aesenc_epi128(s0, rk);
      s1 = _mm512_aesenc_epi128(s1, rk);
      s2 = _mm512_aesenc_epi128(s2, rk);
      s3 = _mm512_aesenc_epi128(s3, rk);
    }};
  }

  round!(1);
  round!(2);
  round!(3);
  round!(4);
  round!(5);
  round!(6);
  round!(7);
  round!(8);
  round!(9);

  let rk10 = _mm512_broadcast_i32x4(k[10]);
  (
    _mm512_aesenclast_epi128(s0, rk10),
    _mm512_aesenclast_epi128(s1, rk10),
    _mm512_aesenclast_epi128(s2, rk10),
    _mm512_aesenclast_epi128(s3, rk10),
  )
}

/// Encrypts one AES-128 state with the expanded AES-NI schedule.
///
/// # Safety
///
/// The caller must ensure the CPU supports the x86 `aes` and `sse2` features.
#[target_feature(enable = "aes,sse2")]
#[inline]
unsafe fn encrypt_state_128(keys: &Ni128RoundKeys, mut state: __m128i) -> __m128i {
  let k = &keys.rk;
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
  _mm_aesenclast_si128(state, k[10])
}

/// Encrypt a single 16-byte block using AES-128 with AES-NI.
///
/// # Safety
/// Caller must ensure the CPU supports AES-NI (`target_feature = "aes"`).
#[target_feature(enable = "aes,sse2")]
pub(super) unsafe fn encrypt_block_128(keys: &Ni128RoundKeys, block: &mut [u8; 16]) {
  // SAFETY: target_feature gate guarantees AES-NI + SSE2. The fixed-size block
  // supplies one complete unaligned load and store.
  unsafe {
    let state = encrypt_state_128(keys, _mm_loadu_si128(block.as_ptr().cast()));
    _mm_storeu_si128(block.as_mut_ptr().cast(), state);
  }
}

#[cfg(feature = "aes-gcm")]
/// Encrypt one block and return only its first five bytes.
///
/// The unused eleven output bytes remain in the SIMD state and are never
/// materialized in addressable memory.
///
/// # Safety
/// Caller must ensure the CPU supports AES-NI (`target_feature = "aes"`).
#[target_feature(enable = "aes,sse2")]
pub(super) unsafe fn encrypt_block_prefix_5_128(keys: &Ni128RoundKeys, block: &[u8; 16]) -> [u8; 5] {
  // SAFETY: target_feature gate guarantees AES-NI + SSE2. The fixed-size input
  // supplies one complete unaligned block load; lane extraction reads only the
  // low eight bytes of the initialized encrypted state.
  unsafe {
    let state = encrypt_state_128(keys, _mm_loadu_si128(block.as_ptr().cast()));
    let prefix = _mm_cvtsi128_si64(state).cast_unsigned().to_le_bytes();
    [prefix[0], prefix[1], prefix[2], prefix[3], prefix[4]]
  }
}

/// Encrypt independent AES-128 blocks in four-way AES-NI batches.
///
/// # Safety
/// Caller must ensure the CPU supports AES-NI (`target_feature = "aes"`).
#[cfg(any(test, feature = "aes-gcm-siv", feature = "aes-siv"))]
#[target_feature(enable = "aes,sse2")]
pub(super) unsafe fn encrypt_blocks_128(keys: &Ni128RoundKeys, blocks: &mut [[u8; 16]]) {
  // SAFETY: AES-NI + SSE2 are enabled. Every load and store is through a complete initialized
  // block reference, and four-block batches hold disjoint mutable elements.
  unsafe {
    let (batches, remainder) = blocks.as_chunks_mut::<4>();
    for batch in batches {
      let b0 = _mm_loadu_si128(batch[0].as_ptr().cast());
      let b1 = _mm_loadu_si128(batch[1].as_ptr().cast());
      let b2 = _mm_loadu_si128(batch[2].as_ptr().cast());
      let b3 = _mm_loadu_si128(batch[3].as_ptr().cast());
      let (b0, b1, b2, b3) = encrypt_4blocks_128_aesni(keys, b0, b1, b2, b3);
      _mm_storeu_si128(batch[0].as_mut_ptr().cast(), b0);
      _mm_storeu_si128(batch[1].as_mut_ptr().cast(), b1);
      _mm_storeu_si128(batch[2].as_mut_ptr().cast(), b2);
      _mm_storeu_si128(batch[3].as_mut_ptr().cast(), b3);
    }
    for block in remainder {
      let state = encrypt_state_128(keys, _mm_loadu_si128(block.as_ptr().cast()));
      _mm_storeu_si128(block.as_mut_ptr().cast(), state);
    }
  }
}

/// Encrypt independent AES-128 blocks in sixteen-way VAES-512 batches.
///
/// # Safety
/// Caller must ensure the CPU and OS support AES-NI, VAES, AVX-512F, and AVX-512VL.
#[cfg(any(test, feature = "aes-gcm-siv", feature = "aes-siv"))]
#[target_feature(enable = "aes,sse2,avx512f,avx512vl,vaes")]
pub(super) unsafe fn encrypt_blocks_128_vaes512(keys: &Ni128RoundKeys, blocks: &mut [[u8; 16]]) {
  // SAFETY: required target features are enabled. Each 256-byte batch contains sixteen complete,
  // initialized, disjoint blocks; the four unaligned vector loads and stores stay within it.
  unsafe {
    let (batches, remainder) = blocks.as_chunks_mut::<16>();
    for batch in batches {
      let ptr = batch.as_mut_ptr().cast::<u8>();
      let b0 = _mm512_loadu_si512(ptr.cast());
      let b1 = _mm512_loadu_si512(ptr.add(64).cast());
      let b2 = _mm512_loadu_si512(ptr.add(128).cast());
      let b3 = _mm512_loadu_si512(ptr.add(192).cast());
      let (b0, b1, b2, b3) = encrypt_16blocks_128(keys, b0, b1, b2, b3);
      _mm512_storeu_si512(ptr.cast(), b0);
      _mm512_storeu_si512(ptr.add(64).cast(), b1);
      _mm512_storeu_si512(ptr.add(128).cast(), b2);
      _mm512_storeu_si512(ptr.add(192).cast(), b3);
    }
    encrypt_blocks_128(keys, remainder);
  }
}

/// XOR a serial block stream into `state` and AES-128-encrypt after every block.
///
/// # Safety
/// Caller must ensure the CPU supports AES-NI (`target_feature = "aes"`).
#[cfg(feature = "aes-siv")]
#[target_feature(enable = "aes,sse2")]
pub(super) unsafe fn xor_encrypt_blocks_128(keys: &Ni128RoundKeys, state: &mut [u8; 16], blocks: &[[u8; 16]]) {
  // SAFETY: AES-NI + SSE2 are enabled. Inputs are complete initialized blocks; the only write is
  // the final fixed-size state store. The loop count depends solely on public input length.
  unsafe {
    let mut chain = _mm_loadu_si128(state.as_ptr().cast());
    for block in blocks {
      chain = _mm_xor_si128(chain, _mm_loadu_si128(block.as_ptr().cast()));
      chain = encrypt_state_128(keys, chain);
    }
    _mm_storeu_si128(state.as_mut_ptr().cast(), chain);
  }
}
