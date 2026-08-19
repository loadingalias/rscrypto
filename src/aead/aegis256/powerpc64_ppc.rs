use core::{arch::asm, simd::i64x2};

use super::{BLOCK_SIZE, C0, C1, KEY_SIZE, NONCE_SIZE, TAG_SIZE};

/// Load a 16-byte block into a POWER vector register (big-endian byte order).
#[inline]
fn load_be(bytes: &[u8; 16]) -> i64x2 {
  let elems = [
    i64::from_be_bytes([
      bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
    ]),
    i64::from_be_bytes([
      bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]),
  ];
  #[cfg(target_endian = "little")]
  {
    i64x2::from_array(elems)
  }
}

/// Store a POWER vector register back to a 16-byte block.
#[inline(always)]
fn store_be(v: i64x2, out: &mut [u8; 16]) {
  let arr = v.to_array();
  #[cfg(target_endian = "little")]
  {
    let hi = u64::from_ne_bytes(arr[1].to_ne_bytes()).to_be_bytes();
    let lo = u64::from_ne_bytes(arr[0].to_ne_bytes()).to_be_bytes();
    out[0..8].copy_from_slice(&hi);
    out[8..16].copy_from_slice(&lo);
  }
}

/// Single AES round: vcipher(state, round_key).
///
/// vcipher computes ShiftRows → SubBytes → MixColumns → XOR(round_key),
/// matching x86 AESENC / aarch64 AESE+AESMC+EOR semantics.
#[target_feature(enable = "altivec,vsx,power8-vector,power8-crypto")]
#[inline]
/// # Safety
///
/// Caller must ensure POWER8 vector crypto support is available.
unsafe fn aes_round(block: i64x2, round_key: i64x2) -> i64x2 {
  let out: i64x2;
  // SAFETY: the caller guarantees POWER8 vector crypto support; the instruction only reads its register operands.
  unsafe {
    asm!(
      "vcipher {out}, {block}, {rk}",
      out = lateout(vreg) out,
      block = in(vreg) block,
      rk = in(vreg) round_key,
      options(nomem, nostack),
    )
  };
  out
}

#[inline(always)]
fn xor_vec(a: i64x2, b: i64x2) -> i64x2 {
  let aa = a.to_array();
  let ba = b.to_array();
  i64x2::from_array([aa[0] ^ ba[0], aa[1] ^ ba[1]])
}

#[inline(always)]
fn and_vec(a: i64x2, b: i64x2) -> i64x2 {
  let aa = a.to_array();
  let ba = b.to_array();
  i64x2::from_array([aa[0] & ba[0], aa[1] & ba[1]])
}

type StateMut<'a> = (
  &'a mut i64x2,
  &'a mut i64x2,
  &'a mut i64x2,
  &'a mut i64x2,
  &'a mut i64x2,
  &'a mut i64x2,
);

#[derive(Clone, Copy)]
struct Power8 {
  _private: (),
}

impl Power8 {
  /// Creates a POWER8 vector-crypto capability token.
  ///
  /// # Safety
  ///
  /// The current CPU must support POWER8 vector crypto for the token's entire lifetime.
  #[inline]
  unsafe fn new() -> Self {
    Self { _private: () }
  }

  #[inline(always)]
  fn update(self, (s0, s1, s2, s3, s4, s5): StateMut<'_>, message: i64x2) {
    let old_s5 = *s5;
    // SAFETY: this module constructs `Power8` only through `new`, whose caller guarantees POWER8 vector crypto.
    unsafe {
      *s5 = aes_round(*s4, *s5);
      *s4 = aes_round(*s3, *s4);
      *s3 = aes_round(*s2, *s3);
      *s2 = aes_round(*s1, *s2);
      *s1 = aes_round(*s0, *s1);
      *s0 = xor_vec(aes_round(old_s5, *s0), message);
    }
  }
}

#[inline(always)]
fn keystream_regs(s1: i64x2, s2: i64x2, s3: i64x2, s4: i64x2, s5: i64x2) -> i64x2 {
  xor_vec(xor_vec(s1, s4), xor_vec(s5, and_vec(s2, s3)))
}

#[target_feature(enable = "altivec,vsx,power8-vector,power8-crypto")]
/// # Safety
///
/// Caller must ensure POWER8 vector crypto support is available.
pub(super) unsafe fn encrypt_fused(
  key: &[u8; KEY_SIZE],
  nonce: &[u8; NONCE_SIZE],
  aad: &[u8],
  buffer: &mut [u8],
) -> [u8; TAG_SIZE] {
  // SAFETY: this function's caller guarantees POWER8 vector crypto support.
  let power8 = unsafe { Power8::new() };
  let (kh0, kh1) = super::split_halves(key);
  let (nh0, nh1) = super::split_halves(nonce);
  let k0 = load_be(kh0);
  let k1 = load_be(kh1);
  let n0 = load_be(nh0);
  let n1 = load_be(nh1);
  let c0 = load_be(&C0);
  let c1 = load_be(&C1);
  let k0_xor_n0 = xor_vec(k0, n0);
  let k1_xor_n1 = xor_vec(k1, n1);
  let (mut s0, mut s1, mut s2, mut s3, mut s4, mut s5) =
    (k0_xor_n0, k1_xor_n1, c1, c0, xor_vec(k0, c0), xor_vec(k1, c1));
  for _ in 0..4 {
    power8.update((&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5), k0);
    power8.update((&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5), k1);
    power8.update((&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5), k0_xor_n0);
    power8.update((&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5), k1_xor_n1);
  }

  let (aad_blocks, aad_tail) = aad.as_chunks::<BLOCK_SIZE>();
  for block in aad_blocks {
    power8.update((&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5), load_be(block));
  }
  if !aad_tail.is_empty() {
    let mut pad = [0u8; BLOCK_SIZE];
    let (pad_tail, _) = pad.split_at_mut(aad_tail.len());
    pad_tail.copy_from_slice(aad_tail);
    power8.update((&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5), load_be(&pad));
  }

  let msg_len = buffer.len();
  let (quads, remainder) = buffer.as_chunks_mut::<64>();
  for quad in quads {
    let (blocks, _) = quad.as_chunks_mut::<BLOCK_SIZE>();
    for block in blocks {
      let stream = keystream_regs(s1, s2, s3, s4, s5);
      let plaintext = load_be(block);
      power8.update((&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5), plaintext);
      store_be(xor_vec(plaintext, stream), block);
    }
  }
  let (blocks, tail) = remainder.as_chunks_mut::<BLOCK_SIZE>();
  for block in blocks {
    let stream = keystream_regs(s1, s2, s3, s4, s5);
    let plaintext = load_be(block);
    power8.update((&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5), plaintext);
    store_be(xor_vec(plaintext, stream), block);
  }
  if !tail.is_empty() {
    let stream = keystream_regs(s1, s2, s3, s4, s5);
    let mut pad = [0u8; BLOCK_SIZE];
    let (pad_tail, _) = pad.split_at_mut(tail.len());
    pad_tail.copy_from_slice(tail);
    let plaintext = load_be(&pad);
    power8.update((&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5), plaintext);
    let mut ciphertext = [0u8; BLOCK_SIZE];
    store_be(xor_vec(plaintext, stream), &mut ciphertext);
    let (ciphertext_tail, _) = ciphertext.split_at(tail.len());
    tail.copy_from_slice(ciphertext_tail);
  }

  let ad_bits = (aad.len() as u64).strict_mul(8);
  let msg_bits = (msg_len as u64).strict_mul(8);
  let mut len_bytes = [0u8; BLOCK_SIZE];
  let (ad_len_bytes, msg_len_bytes) = len_bytes.split_at_mut(8);
  ad_len_bytes.copy_from_slice(&ad_bits.to_le_bytes());
  msg_len_bytes.copy_from_slice(&msg_bits.to_le_bytes());
  let t = xor_vec(s3, load_be(&len_bytes));
  for _ in 0..7 {
    power8.update((&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5), t);
  }
  let tag_vec = xor_vec(xor_vec(xor_vec(s0, s1), xor_vec(s2, s3)), xor_vec(s4, s5));
  let mut tag = [0u8; TAG_SIZE];
  store_be(tag_vec, &mut tag);
  tag
}

#[target_feature(enable = "altivec,vsx,power8-vector,power8-crypto")]
/// # Safety
///
/// Caller must ensure POWER8 vector crypto support is available.
pub(super) unsafe fn decrypt_fused(
  key: &[u8; KEY_SIZE],
  nonce: &[u8; NONCE_SIZE],
  aad: &[u8],
  buffer: &mut [u8],
) -> [u8; TAG_SIZE] {
  // SAFETY: this function's caller guarantees POWER8 vector crypto support.
  let power8 = unsafe { Power8::new() };
  let (kh0, kh1) = super::split_halves(key);
  let (nh0, nh1) = super::split_halves(nonce);
  let k0 = load_be(kh0);
  let k1 = load_be(kh1);
  let n0 = load_be(nh0);
  let n1 = load_be(nh1);
  let c0 = load_be(&C0);
  let c1 = load_be(&C1);
  let k0_xor_n0 = xor_vec(k0, n0);
  let k1_xor_n1 = xor_vec(k1, n1);
  let (mut s0, mut s1, mut s2, mut s3, mut s4, mut s5) =
    (k0_xor_n0, k1_xor_n1, c1, c0, xor_vec(k0, c0), xor_vec(k1, c1));
  for _ in 0..4 {
    power8.update((&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5), k0);
    power8.update((&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5), k1);
    power8.update((&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5), k0_xor_n0);
    power8.update((&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5), k1_xor_n1);
  }

  let (aad_blocks, aad_tail) = aad.as_chunks::<BLOCK_SIZE>();
  for block in aad_blocks {
    power8.update((&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5), load_be(block));
  }
  if !aad_tail.is_empty() {
    let mut pad = [0u8; BLOCK_SIZE];
    let (pad_tail, _) = pad.split_at_mut(aad_tail.len());
    pad_tail.copy_from_slice(aad_tail);
    power8.update((&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5), load_be(&pad));
  }

  let ct_len = buffer.len();
  let (quads, remainder) = buffer.as_chunks_mut::<64>();
  for quad in quads {
    let (blocks, _) = quad.as_chunks_mut::<BLOCK_SIZE>();
    for block in blocks {
      let stream = keystream_regs(s1, s2, s3, s4, s5);
      let plaintext = xor_vec(load_be(block), stream);
      power8.update((&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5), plaintext);
      store_be(plaintext, block);
    }
  }
  let (blocks, tail) = remainder.as_chunks_mut::<BLOCK_SIZE>();
  for block in blocks {
    let stream = keystream_regs(s1, s2, s3, s4, s5);
    let plaintext = xor_vec(load_be(block), stream);
    power8.update((&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5), plaintext);
    store_be(plaintext, block);
  }
  if !tail.is_empty() {
    let stream = keystream_regs(s1, s2, s3, s4, s5);
    let mut stream_bytes = [0u8; BLOCK_SIZE];
    store_be(stream, &mut stream_bytes);
    let mut plaintext = [0u8; BLOCK_SIZE];
    for ((out, ciphertext), mask) in plaintext.iter_mut().zip(tail.iter()).zip(stream_bytes) {
      *out = *ciphertext ^ mask;
    }
    power8.update(
      (&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5),
      load_be(&plaintext),
    );
    let (plaintext_tail, _) = plaintext.split_at(tail.len());
    tail.copy_from_slice(plaintext_tail);
  }

  let ad_bits = (aad.len() as u64).strict_mul(8);
  let ct_bits = (ct_len as u64).strict_mul(8);
  let mut len_bytes = [0u8; BLOCK_SIZE];
  let (ad_len_bytes, ct_len_bytes) = len_bytes.split_at_mut(8);
  ad_len_bytes.copy_from_slice(&ad_bits.to_le_bytes());
  ct_len_bytes.copy_from_slice(&ct_bits.to_le_bytes());
  let t = xor_vec(s3, load_be(&len_bytes));
  for _ in 0..7 {
    power8.update((&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5), t);
  }
  let tag_vec = xor_vec(xor_vec(xor_vec(s0, s1), xor_vec(s2, s3)), xor_vec(s4, s5));
  let mut tag = [0u8; TAG_SIZE];
  store_be(tag_vec, &mut tag);
  tag
}
