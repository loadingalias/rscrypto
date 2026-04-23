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
    // SAFETY: `self.rk` is a valid, aligned, fully-initialized [__m128i; 15].
    // Reinterpreting as a byte slice for volatile zeroization is sound.
    let bytes = unsafe { core::slice::from_raw_parts_mut(self.rk.as_mut_ptr().cast::<u8>(), 15usize.strict_mul(16)) };
    crate::traits::ct::zeroize(bytes);
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
#[target_feature(enable = "aes,sse2,avx512f,avx512vl,vaes")]
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

/// Encrypt a single 16-byte block using AES-256 with AES-NI.
///
/// # Safety
/// Caller must ensure the CPU supports AES-NI (`target_feature = "aes"`).
#[target_feature(enable = "aes,sse2")]
pub(super) unsafe fn encrypt_block(keys: &NiRoundKeys, block: &mut [u8; 16]) {
  // SAFETY: target_feature gate guarantees AES-NI + SSE2.
  unsafe {
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
