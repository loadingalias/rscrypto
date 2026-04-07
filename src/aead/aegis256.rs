#![allow(clippy::indexing_slicing)]

//! AEGIS-256 authenticated encryption (draft-irtf-cfrg-aegis-aead).
//!
//! High-performance AES-based AEAD with a 256-bit key, 256-bit nonce,
//! and 128-bit authentication tag. Uses raw AES round functions (not full
//! AES encryption), achieving ~2-3x the throughput of AES-256-GCM on
//! hardware with AES-NI or AES-CE.

use core::fmt;

use super::{AeadBufferError, Nonce256, OpenError};
use crate::traits::{Aead, VerificationError, ct};

const KEY_SIZE: usize = 32;
const NONCE_SIZE: usize = Nonce256::LENGTH;
const TAG_SIZE: usize = 16;
const BLOCK_SIZE: usize = 16;

/// Fibonacci-derived constant C0.
const C0: [u8; 16] = [
  0x00, 0x01, 0x01, 0x02, 0x03, 0x05, 0x08, 0x0d, 0x15, 0x22, 0x37, 0x59, 0x90, 0xe9, 0x79, 0x62,
];

/// Fibonacci-derived constant C1.
const C1: [u8; 16] = [
  0xdb, 0x3d, 0x18, 0x55, 0x6d, 0xc2, 0x2f, 0xf1, 0x20, 0x11, 0x31, 0x42, 0x73, 0xb5, 0x28, 0xdd,
];

// ---------------------------------------------------------------------------
// Block helpers
// ---------------------------------------------------------------------------

type Block = [u8; BLOCK_SIZE];

/// Split a 32-byte array into two 16-byte halves.
#[inline(always)]
fn split_halves(bytes: &[u8; 32]) -> (&[u8; 16], &[u8; 16]) {
  // Infallible: split_first_chunk on [u8; 32] always yields a [u8; 16] prefix.
  let (lo, hi) = bytes.split_first_chunk::<16>().unwrap_or((&[0; 16], &[]));
  // `hi` is &[u8; 16] since 32 - 16 = 16. Use first_chunk for the second half.
  let hi: &[u8; 16] = hi.first_chunk().unwrap_or(&[0; 16]);
  (lo, hi)
}

#[inline(always)]
fn xor_block(a: &Block, b: &Block) -> Block {
  let mut out = [0u8; BLOCK_SIZE];
  for i in 0..BLOCK_SIZE {
    out[i] = a[i] ^ b[i];
  }
  out
}

#[inline(always)]
fn and_block(a: &Block, b: &Block) -> Block {
  let mut out = [0u8; BLOCK_SIZE];
  for i in 0..BLOCK_SIZE {
    out[i] = a[i] & b[i];
  }
  out
}

#[inline(always)]
fn zero_block() -> Block {
  [0u8; BLOCK_SIZE]
}

// ---------------------------------------------------------------------------
// Portable backend
// ---------------------------------------------------------------------------

/// AEGIS-256 state: six 128-bit AES blocks.
type State = [Block; 6];

/// Single AES round: SubBytes + ShiftRows + MixColumns + XOR(round_key).
#[inline(always)]
fn aes_round(block: &Block, round_key: &Block) -> Block {
  super::aes::aes_enc_round_portable(block, round_key)
}

/// AEGIS-256 Update function: absorb one 128-bit message block into the state.
///
/// Each step applies a single AES round to rotate the state pipeline.
/// The message block is XORed into S0 after the round.
#[inline]
fn update(s: &mut State, m: &Block) {
  let tmp = s[5];
  s[5] = aes_round(&s[4], &s[5]);
  s[4] = aes_round(&s[3], &s[4]);
  s[3] = aes_round(&s[2], &s[3]);
  s[2] = aes_round(&s[1], &s[2]);
  s[1] = aes_round(&s[0], &s[1]);
  s[0] = xor_block(&aes_round(&tmp, &s[0]), m);
}

/// Initialize AEGIS-256 state from key and nonce.
///
/// Splits key and nonce into 128-bit halves, seeds the 6-block state,
/// then runs 16 Update calls (4 iterations of 4 Updates).
fn init(key: &[u8; KEY_SIZE], nonce: &[u8; NONCE_SIZE]) -> State {
  let (k0_ref, k1_ref) = split_halves(key);
  let (n0_ref, n1_ref) = split_halves(nonce);
  let k0 = *k0_ref;
  let k1 = *k1_ref;
  let n0 = *n0_ref;
  let n1 = *n1_ref;

  let k0_xor_n0 = xor_block(&k0, &n0);
  let k1_xor_n1 = xor_block(&k1, &n1);

  let mut s: State = [k0_xor_n0, k1_xor_n1, C1, C0, xor_block(&k0, &C0), xor_block(&k1, &C1)];

  for _ in 0..4 {
    update(&mut s, &k0);
    update(&mut s, &k1);
    update(&mut s, &k0_xor_n0);
    update(&mut s, &k1_xor_n1);
  }

  s
}

/// Absorb associated data into the state.
fn process_aad(s: &mut State, aad: &[u8]) {
  let mut offset = 0usize;

  // Full 16-byte blocks.
  while offset.strict_add(BLOCK_SIZE) <= aad.len() {
    let mut block = [0u8; BLOCK_SIZE];
    block.copy_from_slice(&aad[offset..offset.strict_add(BLOCK_SIZE)]);
    update(s, &block);
    offset = offset.strict_add(BLOCK_SIZE);
  }

  // Last partial block (zero-padded).
  if offset < aad.len() {
    let mut block = zero_block();
    block[..aad.len().strict_sub(offset)].copy_from_slice(&aad[offset..]);
    update(s, &block);
  }
}

/// Compute the AEGIS-256 keystream word from the current state.
#[inline(always)]
fn keystream(s: &State) -> Block {
  // z = S1 ^ S4 ^ S5 ^ (S2 & S3)
  let s2_and_s3 = and_block(&s[2], &s[3]);
  let mut z = xor_block(&s[1], &s[4]);
  z = xor_block(&z, &s[5]);
  z = xor_block(&z, &s2_and_s3);
  z
}

/// Finalize the state and extract the 128-bit authentication tag.
///
/// XORs the bit-lengths of AAD and message into S3, then runs 7 Update
/// rounds and XORs all six state blocks together.
fn finalize(s: &mut State, ad_len: usize, msg_len: usize) -> [u8; TAG_SIZE] {
  // t = S3 ^ (LE64(ad_len_bits) || LE64(msg_len_bits))
  let ad_bits = (ad_len as u64).strict_mul(8);
  let msg_bits = (msg_len as u64).strict_mul(8);
  let mut t = s[3];
  let ad_bytes = ad_bits.to_le_bytes();
  let msg_bytes = msg_bits.to_le_bytes();
  for i in 0..8 {
    t[i] ^= ad_bytes[i];
    t[i.strict_add(8)] ^= msg_bytes[i];
  }

  for _ in 0..7 {
    update(s, &t);
  }

  // tag = S0 ^ S1 ^ S2 ^ S3 ^ S4 ^ S5
  let mut tag = xor_block(&s[0], &s[1]);
  tag = xor_block(&tag, &s[2]);
  tag = xor_block(&tag, &s[3]);
  tag = xor_block(&tag, &s[4]);
  tag = xor_block(&tag, &s[5]);
  tag
}

// ---------------------------------------------------------------------------
// x86_64 AES-NI backend
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_op_in_unsafe_fn)]
mod ni {
  use core::arch::x86_64::*;

  use super::{BLOCK_SIZE, C0, C1, KEY_SIZE, NONCE_SIZE, TAG_SIZE};

  #[inline(always)]
  unsafe fn load(bytes: &[u8; BLOCK_SIZE]) -> __m128i {
    _mm_loadu_si128(bytes.as_ptr().cast())
  }

  #[inline(always)]
  unsafe fn store(v: __m128i, out: &mut [u8; BLOCK_SIZE]) {
    _mm_storeu_si128(out.as_mut_ptr().cast(), v);
  }

  // ── Register-based helpers ──────────────────────────────────────────────
  //
  // All 6 AES rounds in an AEGIS-256 update read from the OLD state and are
  // mutually independent. Keeping state in 6 local `__m128i` values (rather
  // than indexing an array through `&mut State`) lets the register allocator
  // pin them to XMM registers across loop iterations, eliminating store-to-
  // load round-trips that serialize the OOO pipeline.

  #[inline(always)]
  unsafe fn update_regs(
    s0: &mut __m128i,
    s1: &mut __m128i,
    s2: &mut __m128i,
    s3: &mut __m128i,
    s4: &mut __m128i,
    s5: &mut __m128i,
    m: __m128i,
  ) {
    let tmp = *s5;
    *s5 = _mm_aesenc_si128(*s4, *s5);
    *s4 = _mm_aesenc_si128(*s3, *s4);
    *s3 = _mm_aesenc_si128(*s2, *s3);
    *s2 = _mm_aesenc_si128(*s1, *s2);
    *s1 = _mm_aesenc_si128(*s0, *s1);
    *s0 = _mm_xor_si128(_mm_aesenc_si128(tmp, *s0), m);
  }

  #[inline(always)]
  unsafe fn keystream_regs(s1: __m128i, s2: __m128i, s3: __m128i, s4: __m128i, s5: __m128i) -> __m128i {
    _mm_xor_si128(_mm_xor_si128(s1, s4), _mm_xor_si128(s5, _mm_and_si128(s2, s3)))
  }

  // ── Fused encrypt/decrypt ───────────────────────────────────────────────
  //
  // Single `#[target_feature]` entry points that keep state in XMM registers
  // from init through finalize, eliminating ~15 cycles of stack spills that
  // occur when init/aad/encrypt/finalize are separate function calls.

  #[target_feature(enable = "aes,sse2")]
  pub(super) unsafe fn encrypt_fused(
    key: &[u8; KEY_SIZE],
    nonce: &[u8; NONCE_SIZE],
    aad: &[u8],
    buffer: &mut [u8],
  ) -> [u8; TAG_SIZE] {
    // ── init ──
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
      k0_xor_n0, k1_xor_n1, c1, c0,
      _mm_xor_si128(k0, c0), _mm_xor_si128(k1, c1),
    );
    for _ in 0..4 {
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, k0);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, k1);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, k0_xor_n0);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, k1_xor_n1);
    }

    // ── aad ──
    let mut offset = 0usize;
    while offset.strict_add(BLOCK_SIZE) <= aad.len() {
      let block = _mm_loadu_si128(aad.as_ptr().add(offset).cast());
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, block);
      offset = offset.strict_add(BLOCK_SIZE);
    }
    if offset < aad.len() {
      let mut pad = [0u8; BLOCK_SIZE];
      pad[..aad.len().strict_sub(offset)].copy_from_slice(&aad[offset..]);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, load(&pad));
    }

    // ── encrypt ──
    let msg_len = buffer.len();
    let ptr = buffer.as_mut_ptr();
    let len = buffer.len();
    offset = 0;
    let two_blocks = BLOCK_SIZE.strict_mul(2);
    while offset.strict_add(two_blocks) <= len {
      let z_a = keystream_regs(s1, s2, s3, s4, s5);
      let xi_a = _mm_loadu_si128(ptr.add(offset).cast());
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi_a);
      _mm_storeu_si128(ptr.add(offset).cast(), _mm_xor_si128(xi_a, z_a));
      let z_b = keystream_regs(s1, s2, s3, s4, s5);
      let xi_b = _mm_loadu_si128(ptr.add(offset.strict_add(BLOCK_SIZE)).cast());
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi_b);
      _mm_storeu_si128(ptr.add(offset.strict_add(BLOCK_SIZE)).cast(), _mm_xor_si128(xi_b, z_b));
      offset = offset.strict_add(two_blocks);
    }
    if offset.strict_add(BLOCK_SIZE) <= len {
      let z = keystream_regs(s1, s2, s3, s4, s5);
      let xi = _mm_loadu_si128(ptr.add(offset).cast());
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi);
      _mm_storeu_si128(ptr.add(offset).cast(), _mm_xor_si128(xi, z));
      offset = offset.strict_add(BLOCK_SIZE);
    }
    if offset < len {
      let z = keystream_regs(s1, s2, s3, s4, s5);
      let tail_len = len.strict_sub(offset);
      let mut pad = [0u8; BLOCK_SIZE];
      pad[..tail_len].copy_from_slice(&buffer[offset..]);
      let xi = load(&pad);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi);
      let mut ct_bytes = [0u8; BLOCK_SIZE];
      store(_mm_xor_si128(xi, z), &mut ct_bytes);
      buffer[offset..].copy_from_slice(&ct_bytes[..tail_len]);
    }

    // ── finalize ──
    let ad_bits = (aad.len() as u64).strict_mul(8);
    let msg_bits = (msg_len as u64).strict_mul(8);
    let len_block = _mm_set_epi64x(msg_bits as i64, ad_bits as i64);
    let t = _mm_xor_si128(s3, len_block);
    for _ in 0..7 {
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, t);
    }
    let tag_vec = _mm_xor_si128(
      _mm_xor_si128(_mm_xor_si128(s0, s1), _mm_xor_si128(s2, s3)),
      _mm_xor_si128(s4, s5),
    );
    let mut tag = [0u8; TAG_SIZE];
    store(tag_vec, &mut tag);
    tag
  }

  #[target_feature(enable = "aes,sse2")]
  pub(super) unsafe fn decrypt_fused(
    key: &[u8; KEY_SIZE],
    nonce: &[u8; NONCE_SIZE],
    aad: &[u8],
    buffer: &mut [u8],
  ) -> [u8; TAG_SIZE] {
    // ── init ──
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
      k0_xor_n0, k1_xor_n1, c1, c0,
      _mm_xor_si128(k0, c0), _mm_xor_si128(k1, c1),
    );
    for _ in 0..4 {
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, k0);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, k1);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, k0_xor_n0);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, k1_xor_n1);
    }

    // ── aad ──
    let mut offset = 0usize;
    while offset.strict_add(BLOCK_SIZE) <= aad.len() {
      let block = _mm_loadu_si128(aad.as_ptr().add(offset).cast());
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, block);
      offset = offset.strict_add(BLOCK_SIZE);
    }
    if offset < aad.len() {
      let mut pad = [0u8; BLOCK_SIZE];
      pad[..aad.len().strict_sub(offset)].copy_from_slice(&aad[offset..]);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, load(&pad));
    }

    // ── decrypt ──
    let ct_len = buffer.len();
    let ptr = buffer.as_mut_ptr();
    let len = buffer.len();
    offset = 0;
    let two_blocks = BLOCK_SIZE.strict_mul(2);
    while offset.strict_add(two_blocks) <= len {
      let z_a = keystream_regs(s1, s2, s3, s4, s5);
      let ci_a = _mm_loadu_si128(ptr.add(offset).cast());
      let xi_a = _mm_xor_si128(ci_a, z_a);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi_a);
      _mm_storeu_si128(ptr.add(offset).cast(), xi_a);
      let z_b = keystream_regs(s1, s2, s3, s4, s5);
      let ci_b = _mm_loadu_si128(ptr.add(offset.strict_add(BLOCK_SIZE)).cast());
      let xi_b = _mm_xor_si128(ci_b, z_b);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi_b);
      _mm_storeu_si128(ptr.add(offset.strict_add(BLOCK_SIZE)).cast(), xi_b);
      offset = offset.strict_add(two_blocks);
    }
    if offset.strict_add(BLOCK_SIZE) <= len {
      let z = keystream_regs(s1, s2, s3, s4, s5);
      let ci = _mm_loadu_si128(ptr.add(offset).cast());
      let xi = _mm_xor_si128(ci, z);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi);
      _mm_storeu_si128(ptr.add(offset).cast(), xi);
      offset = offset.strict_add(BLOCK_SIZE);
    }
    if offset < len {
      let z = keystream_regs(s1, s2, s3, s4, s5);
      let tail_len = len.strict_sub(offset);
      let mut pad = [0u8; BLOCK_SIZE];
      pad[..tail_len].copy_from_slice(&buffer[offset..]);
      let mut z_bytes = [0u8; BLOCK_SIZE];
      store(z, &mut z_bytes);
      let mut pt_pad = [0u8; BLOCK_SIZE];
      for i in 0..tail_len {
        pt_pad[i] = pad[i] ^ z_bytes[i];
      }
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, load(&pt_pad));
      buffer[offset..].copy_from_slice(&pt_pad[..tail_len]);
    }

    // ── finalize ──
    let ad_bits = (aad.len() as u64).strict_mul(8);
    let ct_bits = (ct_len as u64).strict_mul(8);
    let len_block = _mm_set_epi64x(ct_bits as i64, ad_bits as i64);
    let t = _mm_xor_si128(s3, len_block);
    for _ in 0..7 {
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, t);
    }
    let tag_vec = _mm_xor_si128(
      _mm_xor_si128(_mm_xor_si128(s0, s1), _mm_xor_si128(s2, s3)),
      _mm_xor_si128(s4, s5),
    );
    let mut tag = [0u8; TAG_SIZE];
    store(tag_vec, &mut tag);
    tag
  }
}

// NOTE: A VAES-256 backend was previously implemented here, packing 6 states
// into 3 YMM registers with 3 VAESENC per update. It was removed because the
// 3 cross-lane shuffles (vperm2i128, 3-cycle latency on Zen4) that feed each
// VAESENC dominate the critical path, making VAES ~8 cyc/block vs AES-NI's
// ~5 cyc/block on 2-port machines. See git history for the implementation.
// ---------------------------------------------------------------------------
// aarch64 AES-CE backend
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_op_in_unsafe_fn)]
mod ce {
  use core::arch::aarch64::*;

  use super::{BLOCK_SIZE, C0, C1, KEY_SIZE, NONCE_SIZE, TAG_SIZE};

  #[inline(always)]
  unsafe fn load(bytes: &[u8; BLOCK_SIZE]) -> uint8x16_t {
    vld1q_u8(bytes.as_ptr())
  }

  #[inline(always)]
  unsafe fn store(v: uint8x16_t, out: &mut [u8; BLOCK_SIZE]) {
    vst1q_u8(out.as_mut_ptr(), v);
  }

  // ── Register-based helpers ──────────────────────────────────────────────
  //
  // ARM AESE applies AddRoundKey *before* SubBytes (opposite of x86 AESENC),
  // so we pass zero as the AESE key and XOR the actual round key after
  // MixColumns. The zero XOR is NOT redundant — removing it would break
  // correctness because SubBytes is non-linear.
  //
  // Neoverse V1/V2 has 2 crypto pipelines; register-based code gives the
  // OOO engine maximum scheduling freedom across both pipes.

  #[target_feature(enable = "aes,neon")]
  #[inline]
  unsafe fn update_regs(
    s0: &mut uint8x16_t,
    s1: &mut uint8x16_t,
    s2: &mut uint8x16_t,
    s3: &mut uint8x16_t,
    s4: &mut uint8x16_t,
    s5: &mut uint8x16_t,
    m: uint8x16_t,
  ) {
    let zero = vdupq_n_u8(0);
    let tmp = *s5;
    *s5 = veorq_u8(vaesmcq_u8(vaeseq_u8(*s4, zero)), *s5);
    *s4 = veorq_u8(vaesmcq_u8(vaeseq_u8(*s3, zero)), *s4);
    *s3 = veorq_u8(vaesmcq_u8(vaeseq_u8(*s2, zero)), *s3);
    *s2 = veorq_u8(vaesmcq_u8(vaeseq_u8(*s1, zero)), *s2);
    *s1 = veorq_u8(vaesmcq_u8(vaeseq_u8(*s0, zero)), *s1);
    *s0 = veorq_u8(veorq_u8(vaesmcq_u8(vaeseq_u8(tmp, zero)), *s0), m);
  }

  #[inline(always)]
  unsafe fn keystream_regs(
    s1: uint8x16_t,
    s2: uint8x16_t,
    s3: uint8x16_t,
    s4: uint8x16_t,
    s5: uint8x16_t,
  ) -> uint8x16_t {
    veorq_u8(veorq_u8(s1, s4), veorq_u8(s5, vandq_u8(s2, s3)))
  }

  // ── Fused encrypt/decrypt ───────────────────────────────────────────────

  #[target_feature(enable = "aes,neon")]
  pub(super) unsafe fn encrypt_fused(
    key: &[u8; KEY_SIZE], nonce: &[u8; NONCE_SIZE], aad: &[u8], buffer: &mut [u8],
  ) -> [u8; TAG_SIZE] {
    let (kh0, kh1) = super::split_halves(key);
    let (nh0, nh1) = super::split_halves(nonce);
    let k0 = load(kh0); let k1 = load(kh1);
    let n0 = load(nh0); let n1 = load(nh1);
    let c0 = load(&C0); let c1 = load(&C1);
    let k0_xor_n0 = veorq_u8(k0, n0);
    let k1_xor_n1 = veorq_u8(k1, n1);
    let (mut s0, mut s1, mut s2, mut s3, mut s4, mut s5) =
      (k0_xor_n0, k1_xor_n1, c1, c0, veorq_u8(k0, c0), veorq_u8(k1, c1));
    for _ in 0..4 {
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, k0);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, k1);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, k0_xor_n0);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, k1_xor_n1);
    }
    let mut offset = 0usize;
    while offset.strict_add(BLOCK_SIZE) <= aad.len() {
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, vld1q_u8(aad.as_ptr().add(offset)));
      offset = offset.strict_add(BLOCK_SIZE);
    }
    if offset < aad.len() {
      let mut pad = [0u8; BLOCK_SIZE];
      pad[..aad.len().strict_sub(offset)].copy_from_slice(&aad[offset..]);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, load(&pad));
    }
    let msg_len = buffer.len();
    let ptr = buffer.as_mut_ptr();
    let len = buffer.len();
    offset = 0;
    let two_blocks = BLOCK_SIZE.strict_mul(2);
    while offset.strict_add(two_blocks) <= len {
      let z_a = keystream_regs(s1, s2, s3, s4, s5);
      let xi_a = vld1q_u8(ptr.add(offset));
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi_a);
      vst1q_u8(ptr.add(offset), veorq_u8(xi_a, z_a));
      let z_b = keystream_regs(s1, s2, s3, s4, s5);
      let xi_b = vld1q_u8(ptr.add(offset.strict_add(BLOCK_SIZE)));
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi_b);
      vst1q_u8(ptr.add(offset.strict_add(BLOCK_SIZE)), veorq_u8(xi_b, z_b));
      offset = offset.strict_add(two_blocks);
    }
    if offset.strict_add(BLOCK_SIZE) <= len {
      let z = keystream_regs(s1, s2, s3, s4, s5);
      let xi = vld1q_u8(ptr.add(offset));
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi);
      vst1q_u8(ptr.add(offset), veorq_u8(xi, z));
      offset = offset.strict_add(BLOCK_SIZE);
    }
    if offset < len {
      let z = keystream_regs(s1, s2, s3, s4, s5);
      let tail_len = len.strict_sub(offset);
      let mut pad = [0u8; BLOCK_SIZE];
      pad[..tail_len].copy_from_slice(&buffer[offset..]);
      let xi = load(&pad);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi);
      let mut ct_bytes = [0u8; BLOCK_SIZE];
      store(veorq_u8(xi, z), &mut ct_bytes);
      buffer[offset..].copy_from_slice(&ct_bytes[..tail_len]);
    }
    let ad_bits = (aad.len() as u64).strict_mul(8);
    let msg_bits = (msg_len as u64).strict_mul(8);
    let mut len_bytes = [0u8; BLOCK_SIZE];
    len_bytes[..8].copy_from_slice(&ad_bits.to_le_bytes());
    len_bytes[8..].copy_from_slice(&msg_bits.to_le_bytes());
    let t = veorq_u8(s3, load(&len_bytes));
    for _ in 0..7 { update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, t); }
    let tag_vec = veorq_u8(veorq_u8(veorq_u8(s0, s1), veorq_u8(s2, s3)), veorq_u8(s4, s5));
    let mut tag = [0u8; TAG_SIZE];
    store(tag_vec, &mut tag);
    tag
  }

  #[target_feature(enable = "aes,neon")]
  pub(super) unsafe fn decrypt_fused(
    key: &[u8; KEY_SIZE], nonce: &[u8; NONCE_SIZE], aad: &[u8], buffer: &mut [u8],
  ) -> [u8; TAG_SIZE] {
    let (kh0, kh1) = super::split_halves(key);
    let (nh0, nh1) = super::split_halves(nonce);
    let k0 = load(kh0); let k1 = load(kh1);
    let n0 = load(nh0); let n1 = load(nh1);
    let c0 = load(&C0); let c1 = load(&C1);
    let k0_xor_n0 = veorq_u8(k0, n0);
    let k1_xor_n1 = veorq_u8(k1, n1);
    let (mut s0, mut s1, mut s2, mut s3, mut s4, mut s5) =
      (k0_xor_n0, k1_xor_n1, c1, c0, veorq_u8(k0, c0), veorq_u8(k1, c1));
    for _ in 0..4 {
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, k0);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, k1);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, k0_xor_n0);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, k1_xor_n1);
    }
    let mut offset = 0usize;
    while offset.strict_add(BLOCK_SIZE) <= aad.len() {
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, vld1q_u8(aad.as_ptr().add(offset)));
      offset = offset.strict_add(BLOCK_SIZE);
    }
    if offset < aad.len() {
      let mut pad = [0u8; BLOCK_SIZE];
      pad[..aad.len().strict_sub(offset)].copy_from_slice(&aad[offset..]);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, load(&pad));
    }
    let ct_len = buffer.len();
    let ptr = buffer.as_mut_ptr();
    let len = buffer.len();
    offset = 0;
    let two_blocks = BLOCK_SIZE.strict_mul(2);
    while offset.strict_add(two_blocks) <= len {
      let z_a = keystream_regs(s1, s2, s3, s4, s5);
      let ci_a = vld1q_u8(ptr.add(offset));
      let xi_a = veorq_u8(ci_a, z_a);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi_a);
      vst1q_u8(ptr.add(offset), xi_a);
      let z_b = keystream_regs(s1, s2, s3, s4, s5);
      let ci_b = vld1q_u8(ptr.add(offset.strict_add(BLOCK_SIZE)));
      let xi_b = veorq_u8(ci_b, z_b);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi_b);
      vst1q_u8(ptr.add(offset.strict_add(BLOCK_SIZE)), xi_b);
      offset = offset.strict_add(two_blocks);
    }
    if offset.strict_add(BLOCK_SIZE) <= len {
      let z = keystream_regs(s1, s2, s3, s4, s5);
      let ci = vld1q_u8(ptr.add(offset));
      let xi = veorq_u8(ci, z);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi);
      vst1q_u8(ptr.add(offset), xi);
      offset = offset.strict_add(BLOCK_SIZE);
    }
    if offset < len {
      let z = keystream_regs(s1, s2, s3, s4, s5);
      let tail_len = len.strict_sub(offset);
      let mut pad = [0u8; BLOCK_SIZE];
      pad[..tail_len].copy_from_slice(&buffer[offset..]);
      let mut z_bytes = [0u8; BLOCK_SIZE];
      store(z, &mut z_bytes);
      let mut pt_pad = [0u8; BLOCK_SIZE];
      for i in 0..tail_len { pt_pad[i] = pad[i] ^ z_bytes[i]; }
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, load(&pt_pad));
      buffer[offset..].copy_from_slice(&pt_pad[..tail_len]);
    }
    let ad_bits = (aad.len() as u64).strict_mul(8);
    let ct_bits = (ct_len as u64).strict_mul(8);
    let mut len_bytes = [0u8; BLOCK_SIZE];
    len_bytes[..8].copy_from_slice(&ad_bits.to_le_bytes());
    len_bytes[8..].copy_from_slice(&ct_bits.to_le_bytes());
    let t = veorq_u8(s3, load(&len_bytes));
    for _ in 0..7 { update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, t); }
    let tag_vec = veorq_u8(veorq_u8(veorq_u8(s0, s1), veorq_u8(s2, s3)), veorq_u8(s4, s5));
    let mut tag = [0u8; TAG_SIZE];
    store(tag_vec, &mut tag);
    tag
  }
}

// ---------------------------------------------------------------------------
// powerpc64 vcipher backend (POWER8 Crypto)
// ---------------------------------------------------------------------------

#[cfg(all(target_arch = "powerpc64", target_endian = "little"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
mod ppc {
  use core::{arch::asm, simd::i64x2};

  use super::{BLOCK_SIZE, C0, C1, KEY_SIZE, NONCE_SIZE, TAG_SIZE};

  /// Load a 16-byte block into a POWER vector register (big-endian byte order).
  #[inline(always)]
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
      let hi = (arr[1] as u64).to_be_bytes();
      let lo = (arr[0] as u64).to_be_bytes();
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
  unsafe fn aes_round(block: i64x2, round_key: i64x2) -> i64x2 {
    let out: i64x2;
    asm!(
      "vcipher {out}, {block}, {rk}",
      out = lateout(vreg) out,
      block = in(vreg) block,
      rk = in(vreg) round_key,
      options(nomem, nostack),
    );
    out
  }

  // ── Register-based helpers ──────────────────────────────────────────────

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

  #[inline(always)]
  unsafe fn update_regs(
    s0: &mut i64x2,
    s1: &mut i64x2,
    s2: &mut i64x2,
    s3: &mut i64x2,
    s4: &mut i64x2,
    s5: &mut i64x2,
    m: i64x2,
  ) {
    let tmp = *s5;
    *s5 = aes_round(*s4, *s5);
    *s4 = aes_round(*s3, *s4);
    *s3 = aes_round(*s2, *s3);
    *s2 = aes_round(*s1, *s2);
    *s1 = aes_round(*s0, *s1);
    *s0 = xor_vec(aes_round(tmp, *s0), m);
  }

  #[inline(always)]
  fn keystream_regs(s1: i64x2, s2: i64x2, s3: i64x2, s4: i64x2, s5: i64x2) -> i64x2 {
    xor_vec(xor_vec(s1, s4), xor_vec(s5, and_vec(s2, s3)))
  }

  // ── Fused encrypt/decrypt ───────────────────────────────────────────────

  #[target_feature(enable = "altivec,vsx,power8-vector,power8-crypto")]
  pub(super) unsafe fn encrypt_fused(
    key: &[u8; KEY_SIZE], nonce: &[u8; NONCE_SIZE], aad: &[u8], buffer: &mut [u8],
  ) -> [u8; TAG_SIZE] {
    let (kh0, kh1) = super::split_halves(key);
    let (nh0, nh1) = super::split_halves(nonce);
    let k0 = load_be(kh0); let k1 = load_be(kh1);
    let n0 = load_be(nh0); let n1 = load_be(nh1);
    let c0 = load_be(&C0); let c1 = load_be(&C1);
    let k0_xor_n0 = xor_vec(k0, n0);
    let k1_xor_n1 = xor_vec(k1, n1);
    let (mut s0, mut s1, mut s2, mut s3, mut s4, mut s5) =
      (k0_xor_n0, k1_xor_n1, c1, c0, xor_vec(k0, c0), xor_vec(k1, c1));
    for _ in 0..4 {
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, k0);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, k1);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, k0_xor_n0);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, k1_xor_n1);
    }
    let mut offset = 0usize;
    while offset.strict_add(BLOCK_SIZE) <= aad.len() {
      let mut tmp = [0u8; 16];
      tmp.copy_from_slice(&aad[offset..offset.strict_add(BLOCK_SIZE)]);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, load_be(&tmp));
      offset = offset.strict_add(BLOCK_SIZE);
    }
    if offset < aad.len() {
      let mut pad = [0u8; BLOCK_SIZE];
      pad[..aad.len().strict_sub(offset)].copy_from_slice(&aad[offset..]);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, load_be(&pad));
    }
    let msg_len = buffer.len();
    let len = buffer.len();
    offset = 0;
    let two_blocks = BLOCK_SIZE.strict_mul(2);
    while offset.strict_add(two_blocks) <= len {
      let z_a = keystream_regs(s1, s2, s3, s4, s5);
      let mut tmp_a = [0u8; 16];
      tmp_a.copy_from_slice(&buffer[offset..offset.strict_add(BLOCK_SIZE)]);
      let xi_a = load_be(&tmp_a);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi_a);
      store_be(xor_vec(xi_a, z_a), &mut tmp_a);
      buffer[offset..offset.strict_add(BLOCK_SIZE)].copy_from_slice(&tmp_a);
      let z_b = keystream_regs(s1, s2, s3, s4, s5);
      let off_b = offset.strict_add(BLOCK_SIZE);
      let mut tmp_b = [0u8; 16];
      tmp_b.copy_from_slice(&buffer[off_b..off_b.strict_add(BLOCK_SIZE)]);
      let xi_b = load_be(&tmp_b);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi_b);
      store_be(xor_vec(xi_b, z_b), &mut tmp_b);
      buffer[off_b..off_b.strict_add(BLOCK_SIZE)].copy_from_slice(&tmp_b);
      offset = offset.strict_add(two_blocks);
    }
    if offset.strict_add(BLOCK_SIZE) <= len {
      let z = keystream_regs(s1, s2, s3, s4, s5);
      let mut tmp = [0u8; 16];
      tmp.copy_from_slice(&buffer[offset..offset.strict_add(BLOCK_SIZE)]);
      let xi = load_be(&tmp);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi);
      store_be(xor_vec(xi, z), &mut tmp);
      buffer[offset..offset.strict_add(BLOCK_SIZE)].copy_from_slice(&tmp);
      offset = offset.strict_add(BLOCK_SIZE);
    }
    if offset < len {
      let z = keystream_regs(s1, s2, s3, s4, s5);
      let tail_len = len.strict_sub(offset);
      let mut pad = [0u8; BLOCK_SIZE];
      pad[..tail_len].copy_from_slice(&buffer[offset..]);
      let xi = load_be(&pad);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi);
      let mut ct_bytes = [0u8; BLOCK_SIZE];
      store_be(xor_vec(xi, z), &mut ct_bytes);
      buffer[offset..].copy_from_slice(&ct_bytes[..tail_len]);
    }
    let ad_bits = (aad.len() as u64).strict_mul(8);
    let msg_bits = (msg_len as u64).strict_mul(8);
    let mut len_bytes = [0u8; BLOCK_SIZE];
    len_bytes[..8].copy_from_slice(&ad_bits.to_le_bytes());
    len_bytes[8..].copy_from_slice(&msg_bits.to_le_bytes());
    let t = xor_vec(s3, load_be(&len_bytes));
    for _ in 0..7 { update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, t); }
    let tag_vec = xor_vec(xor_vec(xor_vec(s0, s1), xor_vec(s2, s3)), xor_vec(s4, s5));
    let mut tag = [0u8; TAG_SIZE];
    store_be(tag_vec, &mut tag);
    tag
  }

  #[target_feature(enable = "altivec,vsx,power8-vector,power8-crypto")]
  pub(super) unsafe fn decrypt_fused(
    key: &[u8; KEY_SIZE], nonce: &[u8; NONCE_SIZE], aad: &[u8], buffer: &mut [u8],
  ) -> [u8; TAG_SIZE] {
    let (kh0, kh1) = super::split_halves(key);
    let (nh0, nh1) = super::split_halves(nonce);
    let k0 = load_be(kh0); let k1 = load_be(kh1);
    let n0 = load_be(nh0); let n1 = load_be(nh1);
    let c0 = load_be(&C0); let c1 = load_be(&C1);
    let k0_xor_n0 = xor_vec(k0, n0);
    let k1_xor_n1 = xor_vec(k1, n1);
    let (mut s0, mut s1, mut s2, mut s3, mut s4, mut s5) =
      (k0_xor_n0, k1_xor_n1, c1, c0, xor_vec(k0, c0), xor_vec(k1, c1));
    for _ in 0..4 {
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, k0);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, k1);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, k0_xor_n0);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, k1_xor_n1);
    }
    let mut offset = 0usize;
    while offset.strict_add(BLOCK_SIZE) <= aad.len() {
      let mut tmp = [0u8; 16];
      tmp.copy_from_slice(&aad[offset..offset.strict_add(BLOCK_SIZE)]);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, load_be(&tmp));
      offset = offset.strict_add(BLOCK_SIZE);
    }
    if offset < aad.len() {
      let mut pad = [0u8; BLOCK_SIZE];
      pad[..aad.len().strict_sub(offset)].copy_from_slice(&aad[offset..]);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, load_be(&pad));
    }
    let ct_len = buffer.len();
    let len = buffer.len();
    offset = 0;
    let two_blocks = BLOCK_SIZE.strict_mul(2);
    while offset.strict_add(two_blocks) <= len {
      let z_a = keystream_regs(s1, s2, s3, s4, s5);
      let mut tmp_a = [0u8; 16];
      tmp_a.copy_from_slice(&buffer[offset..offset.strict_add(BLOCK_SIZE)]);
      let xi_a = xor_vec(load_be(&tmp_a), z_a);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi_a);
      store_be(xi_a, &mut tmp_a);
      buffer[offset..offset.strict_add(BLOCK_SIZE)].copy_from_slice(&tmp_a);
      let z_b = keystream_regs(s1, s2, s3, s4, s5);
      let off_b = offset.strict_add(BLOCK_SIZE);
      let mut tmp_b = [0u8; 16];
      tmp_b.copy_from_slice(&buffer[off_b..off_b.strict_add(BLOCK_SIZE)]);
      let xi_b = xor_vec(load_be(&tmp_b), z_b);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi_b);
      store_be(xi_b, &mut tmp_b);
      buffer[off_b..off_b.strict_add(BLOCK_SIZE)].copy_from_slice(&tmp_b);
      offset = offset.strict_add(two_blocks);
    }
    if offset.strict_add(BLOCK_SIZE) <= len {
      let z = keystream_regs(s1, s2, s3, s4, s5);
      let mut tmp = [0u8; 16];
      tmp.copy_from_slice(&buffer[offset..offset.strict_add(BLOCK_SIZE)]);
      let xi = xor_vec(load_be(&tmp), z);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi);
      store_be(xi, &mut tmp);
      buffer[offset..offset.strict_add(BLOCK_SIZE)].copy_from_slice(&tmp);
      offset = offset.strict_add(BLOCK_SIZE);
    }
    if offset < len {
      let z = keystream_regs(s1, s2, s3, s4, s5);
      let tail_len = len.strict_sub(offset);
      let mut pad = [0u8; BLOCK_SIZE];
      pad[..tail_len].copy_from_slice(&buffer[offset..]);
      let mut z_bytes = [0u8; BLOCK_SIZE];
      store_be(z, &mut z_bytes);
      let mut pt_pad = [0u8; BLOCK_SIZE];
      for i in 0..tail_len { pt_pad[i] = pad[i] ^ z_bytes[i]; }
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, load_be(&pt_pad));
      buffer[offset..].copy_from_slice(&pt_pad[..tail_len]);
    }
    let ad_bits = (aad.len() as u64).strict_mul(8);
    let ct_bits = (ct_len as u64).strict_mul(8);
    let mut len_bytes = [0u8; BLOCK_SIZE];
    len_bytes[..8].copy_from_slice(&ad_bits.to_le_bytes());
    len_bytes[8..].copy_from_slice(&ct_bits.to_le_bytes());
    let t = xor_vec(s3, load_be(&len_bytes));
    for _ in 0..7 { update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, t); }
    let tag_vec = xor_vec(xor_vec(xor_vec(s0, s1), xor_vec(s2, s3)), xor_vec(s4, s5));
    let mut tag = [0u8; TAG_SIZE];
    store_be(tag_vec, &mut tag);
    tag
  }
}

// ---------------------------------------------------------------------------
// Backend dispatch
// ---------------------------------------------------------------------------

/// Whether hardware AES acceleration is available.
#[cfg(any(
  target_arch = "x86_64",
  target_arch = "aarch64",
  all(target_arch = "powerpc64", target_endian = "little"),
))]
#[inline]
fn has_hw_aes() -> bool {
  #[cfg(target_arch = "x86_64")]
  {
    use crate::platform::caps::x86;
    crate::platform::caps().has(x86::AESNI)
  }
  #[cfg(target_arch = "aarch64")]
  {
    use crate::platform::caps::aarch64;
    crate::platform::caps().has(aarch64::AES)
  }
  #[cfg(all(target_arch = "powerpc64", target_endian = "little"))]
  {
    use crate::platform::caps::power;
    crate::platform::caps().has(power::POWER8_CRYPTO)
  }
}

// ---------------------------------------------------------------------------
// Key
// ---------------------------------------------------------------------------

/// AEGIS-256 256-bit secret key.
#[derive(Clone)]
pub struct Aegis256Key([u8; Self::LENGTH]);

impl Aegis256Key {
  /// Key length in bytes.
  pub const LENGTH: usize = KEY_SIZE;

  /// Construct a key from raw bytes.
  #[inline]
  #[must_use]
  pub const fn from_bytes(bytes: [u8; Self::LENGTH]) -> Self {
    Self(bytes)
  }

  /// Return the raw key bytes.
  #[inline]
  #[must_use]
  pub fn to_bytes(&self) -> [u8; Self::LENGTH] {
    self.0
  }

  /// Borrow the raw key bytes.
  #[inline]
  #[must_use]
  pub const fn as_bytes(&self) -> &[u8; Self::LENGTH] {
    &self.0
  }
}

impl PartialEq for Aegis256Key {
  fn eq(&self, other: &Self) -> bool {
    ct::constant_time_eq(&self.0, &other.0)
  }
}

impl Eq for Aegis256Key {}

impl Default for Aegis256Key {
  #[inline]
  fn default() -> Self {
    Self([0u8; Self::LENGTH])
  }
}

impl AsRef<[u8]> for Aegis256Key {
  #[inline]
  fn as_ref(&self) -> &[u8] {
    &self.0
  }
}

impl fmt::Debug for Aegis256Key {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str("Aegis256Key(****)")
  }
}

impl Aegis256Key {
  /// Construct a key by filling bytes from the provided closure.
  ///
  /// ```ignore
  /// let key = Aegis256Key::generate(|buf| getrandom::fill(buf).unwrap());
  /// ```
  #[inline]
  #[must_use]
  pub fn generate(fill: impl FnOnce(&mut [u8; Self::LENGTH])) -> Self {
    let mut bytes = [0u8; Self::LENGTH];
    fill(&mut bytes);
    Self(bytes)
  }
}

impl_hex_fmt_secret!(Aegis256Key);

impl Drop for Aegis256Key {
  fn drop(&mut self) {
    ct::zeroize(&mut self.0);
  }
}

// ---------------------------------------------------------------------------
// Tag
// ---------------------------------------------------------------------------

/// AEGIS-256 128-bit authentication tag.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Aegis256Tag([u8; Self::LENGTH]);

impl Aegis256Tag {
  /// Tag length in bytes.
  pub const LENGTH: usize = TAG_SIZE;

  /// Construct a tag from raw bytes.
  #[inline]
  #[must_use]
  pub const fn from_bytes(bytes: [u8; Self::LENGTH]) -> Self {
    Self(bytes)
  }

  /// Return the raw tag bytes.
  #[inline]
  #[must_use]
  pub const fn to_bytes(self) -> [u8; Self::LENGTH] {
    self.0
  }

  /// Borrow the raw tag bytes.
  #[inline]
  #[must_use]
  pub const fn as_bytes(&self) -> &[u8; Self::LENGTH] {
    &self.0
  }
}

impl Default for Aegis256Tag {
  #[inline]
  fn default() -> Self {
    Self([0u8; Self::LENGTH])
  }
}

impl AsRef<[u8]> for Aegis256Tag {
  #[inline]
  fn as_ref(&self) -> &[u8] {
    &self.0
  }
}

impl fmt::Debug for Aegis256Tag {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "Aegis256Tag(")?;
    crate::hex::fmt_hex_lower(&self.0, f)?;
    write!(f, ")")
  }
}

impl_hex_fmt!(Aegis256Tag);

// ---------------------------------------------------------------------------
// AEAD
// ---------------------------------------------------------------------------

/// AEGIS-256 authenticated encryption with associated data.
///
/// High-performance AES-based AEAD with a 256-bit key, 256-bit nonce,
/// and 128-bit authentication tag. On hardware with AES round instructions
/// (AES-NI, AES-CE, POWER8 vcipher), AEGIS-256 achieves
/// ~2-3x the throughput of AES-256-GCM.
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), rscrypto::VerificationError> {
/// use rscrypto::{Aead, Aegis256, Aegis256Key, Aegis256Tag, aead::Nonce256};
///
/// let key = Aegis256Key::from_bytes([0u8; 32]);
/// let nonce = Nonce256::from_bytes([0u8; 32]);
/// let aead = Aegis256::new(&key);
///
/// let mut buf = *b"hello";
/// let tag = aead.encrypt_in_place(&nonce, b"", &mut buf);
/// aead.decrypt_in_place(&nonce, b"", &mut buf, &tag)?;
/// assert_eq!(&buf, b"hello");
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct Aegis256 {
  key: Aegis256Key,
}

impl fmt::Debug for Aegis256 {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("Aegis256").finish_non_exhaustive()
  }
}

impl Aegis256 {
  /// Key length in bytes.
  pub const KEY_SIZE: usize = KEY_SIZE;

  /// Nonce length in bytes.
  pub const NONCE_SIZE: usize = NONCE_SIZE;

  /// Tag length in bytes.
  pub const TAG_SIZE: usize = TAG_SIZE;

  /// Construct a new AEGIS-256 instance from `key`.
  #[inline]
  #[must_use]
  pub fn new(key: &Aegis256Key) -> Self {
    <Self as Aead>::new(key)
  }

  /// Rebuild a typed tag from raw tag bytes.
  #[inline]
  pub fn tag_from_slice(bytes: &[u8]) -> Result<Aegis256Tag, AeadBufferError> {
    <Self as Aead>::tag_from_slice(bytes)
  }

  /// Encrypt `buffer` in place and return the detached authentication tag.
  #[inline]
  #[must_use]
  pub fn encrypt_in_place(&self, nonce: &Nonce256, aad: &[u8], buffer: &mut [u8]) -> Aegis256Tag {
    <Self as Aead>::encrypt_in_place(self, nonce, aad, buffer)
  }

  /// Decrypt `buffer` in place and verify the detached authentication tag.
  #[inline]
  pub fn decrypt_in_place(
    &self,
    nonce: &Nonce256,
    aad: &[u8],
    buffer: &mut [u8],
    tag: &Aegis256Tag,
  ) -> Result<(), VerificationError> {
    <Self as Aead>::decrypt_in_place(self, nonce, aad, buffer, tag)
  }

  /// Encrypt `plaintext` into `out` as `ciphertext || tag`.
  #[inline]
  pub fn encrypt(&self, nonce: &Nonce256, aad: &[u8], plaintext: &[u8], out: &mut [u8]) -> Result<(), AeadBufferError> {
    <Self as Aead>::encrypt(self, nonce, aad, plaintext, out)
  }

  /// Decrypt a combined `ciphertext || tag` into `out`.
  #[inline]
  pub fn decrypt(
    &self,
    nonce: &Nonce256,
    aad: &[u8],
    ciphertext_and_tag: &[u8],
    out: &mut [u8],
  ) -> Result<(), OpenError> {
    <Self as Aead>::decrypt(self, nonce, aad, ciphertext_and_tag, out)
  }
}

// ---------------------------------------------------------------------------
// Portable encrypt/decrypt helpers
// ---------------------------------------------------------------------------

fn encrypt_portable(key: &[u8; KEY_SIZE], nonce: &[u8; NONCE_SIZE], aad: &[u8], buffer: &mut [u8]) -> [u8; TAG_SIZE] {
  let mut s = init(key, nonce);
  process_aad(&mut s, aad);
  let msg_len = buffer.len();
  let mut offset = 0usize;

  // Full blocks.
  while offset.strict_add(BLOCK_SIZE) <= buffer.len() {
    let z = keystream(&s);
    let mut xi = [0u8; BLOCK_SIZE];
    xi.copy_from_slice(&buffer[offset..offset.strict_add(BLOCK_SIZE)]);
    update(&mut s, &xi);
    buffer[offset..offset.strict_add(BLOCK_SIZE)].copy_from_slice(&xor_block(&xi, &z));
    offset = offset.strict_add(BLOCK_SIZE);
  }

  // Partial tail.
  if offset < buffer.len() {
    let z = keystream(&s);
    let tail_len = buffer.len().strict_sub(offset);
    let mut pad = zero_block();
    pad[..tail_len].copy_from_slice(&buffer[offset..]);
    update(&mut s, &pad);
    let ct = xor_block(&pad, &z);
    buffer[offset..].copy_from_slice(&ct[..tail_len]);
  }

  finalize(&mut s, aad.len(), msg_len)
}

fn decrypt_portable(key: &[u8; KEY_SIZE], nonce: &[u8; NONCE_SIZE], aad: &[u8], buffer: &mut [u8]) -> [u8; TAG_SIZE] {
  let mut s = init(key, nonce);
  process_aad(&mut s, aad);
  let ct_len = buffer.len();
  let mut offset = 0usize;

  // Full blocks.
  while offset.strict_add(BLOCK_SIZE) <= buffer.len() {
    let z = keystream(&s);
    let mut ci = [0u8; BLOCK_SIZE];
    ci.copy_from_slice(&buffer[offset..offset.strict_add(BLOCK_SIZE)]);
    let xi = xor_block(&ci, &z);
    update(&mut s, &xi);
    buffer[offset..offset.strict_add(BLOCK_SIZE)].copy_from_slice(&xi);
    offset = offset.strict_add(BLOCK_SIZE);
  }

  // Partial tail.
  if offset < buffer.len() {
    let z = keystream(&s);
    let tail_len = buffer.len().strict_sub(offset);
    let mut pad = zero_block();
    pad[..tail_len].copy_from_slice(&buffer[offset..]);
    // Decrypt only valid bytes; rest stays zero for Update.
    let mut pt_pad = zero_block();
    for i in 0..tail_len {
      pt_pad[i] = pad[i] ^ z[i];
    }
    update(&mut s, &pt_pad);
    buffer[offset..].copy_from_slice(&pt_pad[..tail_len]);
  }

  finalize(&mut s, aad.len(), ct_len)
}

// ---------------------------------------------------------------------------
// Aead trait implementation
// ---------------------------------------------------------------------------

impl Aead for Aegis256 {
  const KEY_SIZE: usize = KEY_SIZE;
  const NONCE_SIZE: usize = NONCE_SIZE;
  const TAG_SIZE: usize = TAG_SIZE;

  type Key = Aegis256Key;
  type Nonce = Nonce256;
  type Tag = Aegis256Tag;

  fn new(key: &Self::Key) -> Self {
    Self { key: key.clone() }
  }

  fn tag_from_slice(bytes: &[u8]) -> Result<Self::Tag, AeadBufferError> {
    if bytes.len() != TAG_SIZE {
      return Err(AeadBufferError::new());
    }
    let mut tag = [0u8; TAG_SIZE];
    tag.copy_from_slice(bytes);
    Ok(Aegis256Tag::from_bytes(tag))
  }

  fn encrypt_in_place(&self, nonce: &Self::Nonce, aad: &[u8], buffer: &mut [u8]) -> Self::Tag {
    let key = self.key.as_bytes();
    let nonce = nonce.as_bytes();

    // NOTE: VAES-256 (`ni_wide`) is intentionally NOT dispatched here.
    // AEGIS-256's update is a serial chain of 6 AES rounds reading old state.
    // VAES-256 packs into 3 YMM registers but requires 3 cross-lane shuffles
    // (`vperm2i128`, 3-cycle latency) before each set of 3 VAESENC, adding
    // ~3 cycles to the critical path. On Zen4 with 2 AES ports: AES-NI steady-
    // state is ~5 cyc/block; VAES-256 is ~8 cyc/block. AES-NI wins for serial
    // update chains (unlike AES-GCM where blocks are independent).
    #[cfg(target_arch = "x86_64")]
    if has_hw_aes() {
      // SAFETY: has_hw_aes() confirmed AES-NI is available.
      // Fused path keeps state in XMM registers from init through finalize.
      let tag = unsafe { ni::encrypt_fused(key, nonce, aad, buffer) };
      return Aegis256Tag::from_bytes(tag);
    }

    #[cfg(target_arch = "aarch64")]
    if has_hw_aes() {
      let tag = unsafe { ce::encrypt_fused(key, nonce, aad, buffer) };
      return Aegis256Tag::from_bytes(tag);
    }

    #[cfg(all(target_arch = "powerpc64", target_endian = "little"))]
    if has_hw_aes() {
      let tag = unsafe { ppc::encrypt_fused(key, nonce, aad, buffer) };
      return Aegis256Tag::from_bytes(tag);
    }

    Aegis256Tag::from_bytes(encrypt_portable(key, nonce, aad, buffer))
  }

  fn decrypt_in_place(
    &self,
    nonce: &Self::Nonce,
    aad: &[u8],
    buffer: &mut [u8],
    tag: &Self::Tag,
  ) -> Result<(), VerificationError> {
    let key = self.key.as_bytes();
    let nonce = nonce.as_bytes();

    #[cfg(target_arch = "x86_64")]
    let computed = if has_hw_aes() {
      // SAFETY: has_hw_aes() confirmed AES-NI is available.
      // Fused path keeps state in XMM registers from init through finalize.
      unsafe { ni::decrypt_fused(key, nonce, aad, buffer) }
    } else {
      decrypt_portable(key, nonce, aad, buffer)
    };

    #[cfg(target_arch = "aarch64")]
    let computed = if has_hw_aes() {
      unsafe { ce::decrypt_fused(key, nonce, aad, buffer) }
    } else {
      decrypt_portable(key, nonce, aad, buffer)
    };

    #[cfg(all(target_arch = "powerpc64", target_endian = "little"))]
    let computed = if has_hw_aes() {
      unsafe { ppc::decrypt_fused(key, nonce, aad, buffer) }
    } else {
      decrypt_portable(key, nonce, aad, buffer)
    };

    #[cfg(not(any(
      target_arch = "x86_64",
      target_arch = "aarch64",
      all(target_arch = "powerpc64", target_endian = "little"),
    )))]
    let computed = decrypt_portable(key, nonce, aad, buffer);

    if !ct::constant_time_eq(&computed, tag.as_bytes()) {
      ct::zeroize(buffer);
      return Err(VerificationError::new());
    }

    Ok(())
  }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
  use super::*;

  fn hex(s: &str) -> Vec<u8> {
    (0..s.len())
      .step_by(2)
      .map(|i| u8::from_str_radix(&s[i..i.strict_add(2)], 16).unwrap())
      .collect()
  }

  fn hex_block(s: &str) -> [u8; 16] {
    let v = hex(s);
    let mut out = [0u8; 16];
    out.copy_from_slice(&v);
    out
  }

  // -- AESRound test vector (Appendix A.1) --

  #[test]
  fn aes_round_matches_spec_vector() {
    let input = hex_block("000102030405060708090a0b0c0d0e0f");
    let rk = hex_block("101112131415161718191a1b1c1d1e1f");
    let expected = hex_block("7a7b4e5638782546a8c0477a3b813f43");

    assert_eq!(aes_round(&input, &rk), expected);
  }

  // -- Update test vector (Appendix A.2) --

  #[test]
  fn update_matches_spec_vector() {
    let mut s: State = [
      hex_block("1fa1207ed76c86f2c4bb40e8b395b43e"),
      hex_block("b44c375e6c1e1978db64bcd12e9e332f"),
      hex_block("0dab84bfa9f0226432ff630f233d4e5b"),
      hex_block("d7ef65c9b93e8ee60c75161407b066e7"),
      hex_block("a760bb3da073fbd92bdc24734b1f56fb"),
      hex_block("a828a18d6a964497ac6e7e53c5f55c73"),
    ];
    let m = hex_block("b165617ed04ab738afb2612c6d18a1ec");

    update(&mut s, &m);

    assert_eq!(s[0], hex_block("e6bc643bae82dfa3d991b1b323839dcd"));
    assert_eq!(s[1], hex_block("648578232ba0f2f0a3677f617dc052c3"));
    assert_eq!(s[2], hex_block("ea788e0e572044a46059212dd007a789"));
    assert_eq!(s[3], hex_block("2f1498ae19b80da13fba698f088a8590"));
    assert_eq!(s[4], hex_block("a54c2ee95e8c2a2c3dae2ec743ae6b86"));
    assert_eq!(s[5], hex_block("a3240fceb68e32d5d114df1b5363ab67"));
  }

  // -- Spec test vectors (Appendix A.3) --

  fn spec_key() -> Aegis256Key {
    Aegis256Key::from_bytes(
      hex_block("10010000000000000000000000000000")
        .iter()
        .chain(hex_block("00000000000000000000000000000000").iter())
        .copied()
        .collect::<Vec<u8>>()
        .try_into()
        .unwrap(),
    )
  }

  fn spec_nonce() -> Nonce256 {
    let mut nonce_bytes = [0u8; 32];
    nonce_bytes[..16].copy_from_slice(&hex_block("10000200000000000000000000000000"));
    nonce_bytes[16..].copy_from_slice(&hex_block("00000000000000000000000000000000"));
    Nonce256::from_bytes(nonce_bytes)
  }

  fn verify_encrypt(msg: &[u8], aad: &[u8], expected_ct_hex: &str, expected_tag128_hex: &str) {
    let aead = Aegis256::new(&spec_key());
    let nonce = spec_nonce();
    let expected_ct = hex(expected_ct_hex);
    let expected_tag = hex(expected_tag128_hex);

    // Encrypt.
    let mut buf = msg.to_vec();
    let tag = aead.encrypt_in_place(&nonce, aad, &mut buf);
    assert_eq!(&buf, &expected_ct, "ciphertext mismatch");
    assert_eq!(tag.as_bytes(), expected_tag.as_slice(), "tag mismatch");

    // Decrypt round-trip.
    aead.decrypt_in_place(&nonce, aad, &mut buf, &tag).unwrap();
    assert_eq!(&buf, msg, "plaintext recovery mismatch");
  }

  /// Test vector 1: 16 bytes msg, no AAD.
  #[test]
  fn spec_vector_1() {
    verify_encrypt(
      &[0u8; 16],
      b"",
      "754fc3d8c973246dcc6d741412a4b236",
      "3fe91994768b332ed7f570a19ec5896e",
    );
  }

  /// Test vector 2: empty msg, no AAD (tag-only).
  #[test]
  fn spec_vector_2() {
    verify_encrypt(b"", b"", "", "e3def978a0f054afd1e761d7553afba3");
  }

  /// Test vector 3: 32 bytes msg + 8 bytes AAD.
  #[test]
  fn spec_vector_3() {
    verify_encrypt(
      &hex("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f"),
      &hex("0001020304050607"),
      "f373079ed84b2709faee373584585d60accd191db310ef5d8b11833df9dec711",
      "8d86f91ee606e9ff26a01b64ccbdd91d",
    );
  }

  /// Test vector 4: 13 bytes msg + 8 bytes AAD (partial block).
  #[test]
  fn spec_vector_4() {
    verify_encrypt(
      &hex("000102030405060708090a0b0c0d"),
      &hex("0001020304050607"),
      "f373079ed84b2709faee37358458",
      "c60b9c2d33ceb058f96e6dd03c215652",
    );
  }

  /// Test vector 5: 40 bytes msg + 42 bytes AAD (multiple blocks + partial).
  #[test]
  fn spec_vector_5() {
    verify_encrypt(
      &hex("101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f3031323334353637"),
      &hex("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20212223242526272829"),
      "57754a7d09963e7c787583a2e7b859bb24fa1e04d49fd550b2511a358e3bca252a9b1b8b30cc4a67",
      "ab8a7d53fd0e98d727accca94925e128",
    );
  }

  // -- Functional tests --

  #[test]
  fn round_trip_empty() {
    let key = Aegis256Key::from_bytes([0u8; 32]);
    let nonce = Nonce256::from_bytes([0u8; 32]);
    let aead = Aegis256::new(&key);

    let mut buf = [];
    let tag = aead.encrypt_in_place(&nonce, b"", &mut buf);
    aead.decrypt_in_place(&nonce, b"", &mut buf, &tag).unwrap();
  }

  #[test]
  fn round_trip_with_data() {
    let key = Aegis256Key::from_bytes([0x42; 32]);
    let nonce = Nonce256::from_bytes([0x13; 32]);
    let aead = Aegis256::new(&key);
    let plaintext = b"the quick brown fox jumps over the lazy dog";

    let mut buf = *plaintext;
    let tag = aead.encrypt_in_place(&nonce, b"header", &mut buf);
    assert_ne!(&buf[..], &plaintext[..]);

    aead.decrypt_in_place(&nonce, b"header", &mut buf, &tag).unwrap();
    assert_eq!(&buf[..], &plaintext[..]);
  }

  #[test]
  fn round_trip_with_aad_only() {
    let key = Aegis256Key::from_bytes([0xFF; 32]);
    let nonce = Nonce256::from_bytes([0xAA; 32]);
    let aead = Aegis256::new(&key);

    let mut buf = [];
    let tag = aead.encrypt_in_place(&nonce, b"associated data only", &mut buf);
    aead
      .decrypt_in_place(&nonce, b"associated data only", &mut buf, &tag)
      .unwrap();
  }

  #[test]
  fn tampered_ciphertext_fails() {
    let key = Aegis256Key::from_bytes([1; 32]);
    let nonce = Nonce256::from_bytes([2; 32]);
    let aead = Aegis256::new(&key);

    let mut buf = *b"secret";
    let tag = aead.encrypt_in_place(&nonce, b"", &mut buf);

    buf[0] ^= 1;
    let result = aead.decrypt_in_place(&nonce, b"", &mut buf, &tag);
    assert!(result.is_err());
    assert_eq!(&buf, &[0u8; 6]);
  }

  #[test]
  fn tampered_tag_fails() {
    let key = Aegis256Key::from_bytes([3; 32]);
    let nonce = Nonce256::from_bytes([4; 32]);
    let aead = Aegis256::new(&key);

    let mut buf = *b"data";
    let tag = aead.encrypt_in_place(&nonce, b"aad", &mut buf);

    let mut bad_tag_bytes = tag.to_bytes();
    bad_tag_bytes[15] ^= 1;
    let bad_tag = Aegis256Tag::from_bytes(bad_tag_bytes);

    let result = aead.decrypt_in_place(&nonce, b"aad", &mut buf, &bad_tag);
    assert!(result.is_err());
    assert_eq!(&buf, &[0u8; 4]);
  }

  #[test]
  fn wrong_aad_fails() {
    let key = Aegis256Key::from_bytes([5; 32]);
    let nonce = Nonce256::from_bytes([6; 32]);
    let aead = Aegis256::new(&key);

    let mut buf = *b"msg";
    let tag = aead.encrypt_in_place(&nonce, b"correct", &mut buf);

    let result = aead.decrypt_in_place(&nonce, b"wrong", &mut buf, &tag);
    assert!(result.is_err());
  }

  #[test]
  fn combined_encrypt_decrypt_round_trip() {
    let key = Aegis256Key::from_bytes([7; 32]);
    let nonce = Nonce256::from_bytes([8; 32]);
    let aead = Aegis256::new(&key);
    let pt = b"combined mode";

    let mut sealed = vec![0u8; pt.len().strict_add(TAG_SIZE)];
    aead.encrypt(&nonce, b"h", pt.as_slice(), &mut sealed).unwrap();

    let mut opened = vec![0u8; pt.len()];
    aead.decrypt(&nonce, b"h", &sealed, &mut opened).unwrap();
    assert_eq!(&opened, &pt[..]);
  }

  #[test]
  fn tag_from_slice_rejects_wrong_length() {
    assert!(Aegis256::tag_from_slice(&[0u8; 15]).is_err());
    assert!(Aegis256::tag_from_slice(&[0u8; 17]).is_err());
    assert!(Aegis256::tag_from_slice(&[0u8; 16]).is_ok());
  }

  #[test]
  fn multi_block_round_trip() {
    let key = Aegis256Key::from_bytes([0xAB; 32]);
    let nonce = Nonce256::from_bytes([0xCD; 32]);
    let aead = Aegis256::new(&key);

    // 100 bytes = 6 full blocks + 4-byte tail.
    let plaintext = [0x77u8; 100];
    let mut buf = plaintext;
    let tag = aead.encrypt_in_place(&nonce, b"multi-block aad that is longer than one rate block", &mut buf);
    aead
      .decrypt_in_place(
        &nonce,
        b"multi-block aad that is longer than one rate block",
        &mut buf,
        &tag,
      )
      .unwrap();
    assert_eq!(buf, plaintext);
  }

  #[test]
  fn exact_block_boundary() {
    let key = Aegis256Key::from_bytes([0x10; 32]);
    let nonce = Nonce256::from_bytes([0x20; 32]);
    let aead = Aegis256::new(&key);

    // Exactly 16 bytes = 1 full block, 0-byte tail.
    let plaintext = [0x55u8; 16];
    let mut buf = plaintext;
    let tag = aead.encrypt_in_place(&nonce, b"", &mut buf);
    aead.decrypt_in_place(&nonce, b"", &mut buf, &tag).unwrap();
    assert_eq!(buf, plaintext);

    // Exactly 32 bytes = 2 full blocks, 0-byte tail.
    let plaintext32 = [0x66u8; 32];
    let mut buf32 = plaintext32;
    let tag32 = aead.encrypt_in_place(&nonce, b"", &mut buf32);
    aead.decrypt_in_place(&nonce, b"", &mut buf32, &tag32).unwrap();
    assert_eq!(buf32, plaintext32);
  }
}
