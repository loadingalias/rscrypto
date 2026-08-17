use core::arch::x86_64::*;

use super::{BLOCK_SIZE, C0, C1, KEY_SIZE, NONCE_SIZE, TAG_SIZE};

#[inline]
fn load(bytes: &[u8; BLOCK_SIZE]) -> __m128i {
  // SAFETY: SSE2 is an x86_64 baseline feature; `bytes` provides 16 readable bytes, and the load is unaligned.
  unsafe { _mm_loadu_si128(bytes.as_ptr().cast()) }
}

#[inline]
fn store(value: __m128i, out: &mut [u8; BLOCK_SIZE]) {
  // SAFETY: SSE2 is an x86_64 baseline feature; `out` provides 16 writable bytes, and the store is unaligned.
  unsafe { _mm_storeu_si128(out.as_mut_ptr().cast(), value) };
}

type StateMut<'a> = (
  &'a mut __m128i,
  &'a mut __m128i,
  &'a mut __m128i,
  &'a mut __m128i,
  &'a mut __m128i,
  &'a mut __m128i,
);

#[derive(Clone, Copy)]
struct AesNi {
  _private: (),
}

impl AesNi {
  /// Creates an AES-NI capability token.
  ///
  /// # Safety
  ///
  /// The current CPU must support AES-NI and AVX for the token's entire lifetime.
  #[inline]
  unsafe fn new() -> Self {
    Self { _private: () }
  }

  #[inline(always)]
  fn update(self, (s0, s1, s2, s3, s4, s5): StateMut<'_>, message: __m128i) {
    let old_s5 = *s5;
    // SAFETY: this module constructs `AesNi` only through `new`, whose caller guarantees AES-NI and AVX support.
    unsafe {
      *s5 = _mm_aesenc_si128(*s4, *s5);
      *s4 = _mm_aesenc_si128(*s3, *s4);
      *s3 = _mm_aesenc_si128(*s2, *s3);
      *s2 = _mm_aesenc_si128(*s1, *s2);
      *s1 = _mm_aesenc_si128(*s0, *s1);
      *s0 = _mm_xor_si128(_mm_aesenc_si128(old_s5, *s0), message);
    }
  }
}

#[inline]
fn keystream(s1: __m128i, s2: __m128i, s3: __m128i, s4: __m128i, s5: __m128i) -> __m128i {
  // SAFETY: SSE2 is part of the x86_64 baseline.
  unsafe { _mm_xor_si128(_mm_xor_si128(s1, s4), _mm_xor_si128(s5, _mm_and_si128(s2, s3))) }
}

#[target_feature(enable = "aes,avx")]
/// # Safety
///
/// Caller must ensure the CPU supports `aes` and `avx`.
pub(super) unsafe fn encrypt_fused(
  key: &[u8; KEY_SIZE],
  nonce: &[u8; NONCE_SIZE],
  aad: &[u8],
  buffer: &mut [u8],
) -> [u8; TAG_SIZE] {
  // SAFETY: this function's caller guarantees AES-NI and AVX support.
  let aesni = unsafe { AesNi::new() };
  let (kh0, kh1) = super::split_halves(key);
  let (nh0, nh1) = super::split_halves(nonce);
  let k0 = load(kh0);
  let k1 = load(kh1);
  let n0 = load(nh0);
  let n1 = load(nh1);
  let c0 = load(&C0);
  let c1 = load(&C1);
  let k0_xor_n0 = _mm_xor_si128(k0, n0);
  let k1_xor_n1 = _mm_xor_si128(k1, n1);
  let (mut s0, mut s1, mut s2, mut s3, mut s4, mut s5) = (
    k0_xor_n0,
    k1_xor_n1,
    c1,
    c0,
    _mm_xor_si128(k0, c0),
    _mm_xor_si128(k1, c1),
  );
  for _ in 0..4 {
    aesni.update((&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5), k0);
    aesni.update((&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5), k1);
    aesni.update((&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5), k0_xor_n0);
    aesni.update((&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5), k1_xor_n1);
  }

  let (aad_blocks, aad_tail) = aad.as_chunks::<BLOCK_SIZE>();
  for block in aad_blocks {
    aesni.update((&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5), load(block));
  }
  if !aad_tail.is_empty() {
    let mut pad = [0u8; BLOCK_SIZE];
    let (pad_tail, _) = pad.split_at_mut(aad_tail.len());
    pad_tail.copy_from_slice(aad_tail);
    aesni.update((&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5), load(&pad));
  }

  let msg_len = buffer.len();
  let (quads, remainder) = buffer.as_chunks_mut::<64>();
  for quad in quads {
    let (blocks, _) = quad.as_chunks_mut::<BLOCK_SIZE>();
    for block in blocks {
      let stream = keystream(s1, s2, s3, s4, s5);
      let plaintext = load(block);
      aesni.update((&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5), plaintext);
      store(_mm_xor_si128(plaintext, stream), block);
    }
  }
  let (blocks, tail) = remainder.as_chunks_mut::<BLOCK_SIZE>();
  for block in blocks {
    let stream = keystream(s1, s2, s3, s4, s5);
    let plaintext = load(block);
    aesni.update((&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5), plaintext);
    store(_mm_xor_si128(plaintext, stream), block);
  }
  if !tail.is_empty() {
    let stream = keystream(s1, s2, s3, s4, s5);
    let mut pad = [0u8; BLOCK_SIZE];
    let (pad_tail, _) = pad.split_at_mut(tail.len());
    pad_tail.copy_from_slice(tail);
    let plaintext = load(&pad);
    aesni.update((&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5), plaintext);
    let mut ciphertext = [0u8; BLOCK_SIZE];
    store(_mm_xor_si128(plaintext, stream), &mut ciphertext);
    let (ciphertext_tail, _) = ciphertext.split_at(tail.len());
    tail.copy_from_slice(ciphertext_tail);
  }

  let ad_bits = (aad.len() as u64).strict_mul(8);
  let msg_bits = (msg_len as u64).strict_mul(8);
  let len_block = _mm_set_epi64x(
    i64::from_ne_bytes(msg_bits.to_ne_bytes()),
    i64::from_ne_bytes(ad_bits.to_ne_bytes()),
  );
  let t = _mm_xor_si128(s3, len_block);
  for _ in 0..7 {
    aesni.update((&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5), t);
  }
  let tag_vec = _mm_xor_si128(
    _mm_xor_si128(_mm_xor_si128(s0, s1), _mm_xor_si128(s2, s3)),
    _mm_xor_si128(s4, s5),
  );
  let mut tag = [0u8; TAG_SIZE];
  store(tag_vec, &mut tag);
  tag
}

#[target_feature(enable = "aes,avx")]
/// # Safety
///
/// Caller must ensure the CPU supports `aes` and `avx`.
pub(super) unsafe fn decrypt_fused(
  key: &[u8; KEY_SIZE],
  nonce: &[u8; NONCE_SIZE],
  aad: &[u8],
  buffer: &mut [u8],
) -> [u8; TAG_SIZE] {
  // SAFETY: this function's caller guarantees AES-NI and AVX support.
  let aesni = unsafe { AesNi::new() };
  let (kh0, kh1) = super::split_halves(key);
  let (nh0, nh1) = super::split_halves(nonce);
  let k0 = load(kh0);
  let k1 = load(kh1);
  let n0 = load(nh0);
  let n1 = load(nh1);
  let c0 = load(&C0);
  let c1 = load(&C1);
  let k0_xor_n0 = _mm_xor_si128(k0, n0);
  let k1_xor_n1 = _mm_xor_si128(k1, n1);
  let (mut s0, mut s1, mut s2, mut s3, mut s4, mut s5) = (
    k0_xor_n0,
    k1_xor_n1,
    c1,
    c0,
    _mm_xor_si128(k0, c0),
    _mm_xor_si128(k1, c1),
  );
  for _ in 0..4 {
    aesni.update((&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5), k0);
    aesni.update((&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5), k1);
    aesni.update((&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5), k0_xor_n0);
    aesni.update((&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5), k1_xor_n1);
  }

  let (aad_blocks, aad_tail) = aad.as_chunks::<BLOCK_SIZE>();
  for block in aad_blocks {
    aesni.update((&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5), load(block));
  }
  if !aad_tail.is_empty() {
    let mut pad = [0u8; BLOCK_SIZE];
    let (pad_tail, _) = pad.split_at_mut(aad_tail.len());
    pad_tail.copy_from_slice(aad_tail);
    aesni.update((&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5), load(&pad));
  }

  let ct_len = buffer.len();
  let (quads, remainder) = buffer.as_chunks_mut::<64>();
  for quad in quads {
    let (blocks, _) = quad.as_chunks_mut::<BLOCK_SIZE>();
    for block in blocks {
      let stream = keystream(s1, s2, s3, s4, s5);
      let plaintext = _mm_xor_si128(load(block), stream);
      aesni.update((&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5), plaintext);
      store(plaintext, block);
    }
  }
  let (blocks, tail) = remainder.as_chunks_mut::<BLOCK_SIZE>();
  for block in blocks {
    let stream = keystream(s1, s2, s3, s4, s5);
    let plaintext = _mm_xor_si128(load(block), stream);
    aesni.update((&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5), plaintext);
    store(plaintext, block);
  }
  if !tail.is_empty() {
    let stream = keystream(s1, s2, s3, s4, s5);
    let mut stream_bytes = [0u8; BLOCK_SIZE];
    store(stream, &mut stream_bytes);
    let mut plaintext = [0u8; BLOCK_SIZE];
    for ((out, ciphertext), mask) in plaintext.iter_mut().zip(tail.iter()).zip(stream_bytes) {
      *out = *ciphertext ^ mask;
    }
    aesni.update((&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5), load(&plaintext));
    let (plaintext_tail, _) = plaintext.split_at(tail.len());
    tail.copy_from_slice(plaintext_tail);
  }

  let ad_bits = (aad.len() as u64).strict_mul(8);
  let ct_bits = (ct_len as u64).strict_mul(8);
  let len_block = _mm_set_epi64x(
    i64::from_ne_bytes(ct_bits.to_ne_bytes()),
    i64::from_ne_bytes(ad_bits.to_ne_bytes()),
  );
  let t = _mm_xor_si128(s3, len_block);
  for _ in 0..7 {
    aesni.update((&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5), t);
  }
  let tag_vec = _mm_xor_si128(
    _mm_xor_si128(_mm_xor_si128(s0, s1), _mm_xor_si128(s2, s3)),
    _mm_xor_si128(s4, s5),
  );
  let mut tag = [0u8; TAG_SIZE];
  store(tag_vec, &mut tag);
  tag
}
