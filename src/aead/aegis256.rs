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

#[cfg(not(target_arch = "s390x"))]
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

#[cfg(not(target_arch = "s390x"))]
#[inline(always)]
fn xor_block(a: &Block, b: &Block) -> Block {
  let mut out = [0u8; BLOCK_SIZE];
  for i in 0..BLOCK_SIZE {
    out[i] = a[i] ^ b[i];
  }
  out
}

#[cfg(not(target_arch = "s390x"))]
#[inline(always)]
fn and_block(a: &Block, b: &Block) -> Block {
  let mut out = [0u8; BLOCK_SIZE];
  for i in 0..BLOCK_SIZE {
    out[i] = a[i] & b[i];
  }
  out
}

#[cfg(not(target_arch = "s390x"))]
#[inline(always)]
fn zero_block() -> Block {
  [0u8; BLOCK_SIZE]
}

// ---------------------------------------------------------------------------
// Portable backend
// ---------------------------------------------------------------------------
//
// On s390x, the `zvec` module provides a T-table backend that unconditionally
// replaces the portable path. Gate this section to suppress dead-code warnings.

#[cfg(not(target_arch = "s390x"))]
type State = [Block; 6];

#[cfg(not(target_arch = "s390x"))]
#[inline(always)]
fn aes_round(block: &Block, round_key: &Block) -> Block {
  super::aes::aes_enc_round_portable(block, round_key)
}

/// AEGIS-256 Update function: absorb one 128-bit message block into the state.
///
/// Each step applies a single AES round to rotate the state pipeline.
/// The message block is XORed into S0 after the round.
#[cfg(not(target_arch = "s390x"))]
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
#[cfg(not(target_arch = "s390x"))]
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
#[cfg(not(target_arch = "s390x"))]
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
#[cfg(not(target_arch = "s390x"))]
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
#[cfg(not(target_arch = "s390x"))]
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

  #[target_feature(enable = "aes,avx")]
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
      k0_xor_n0,
      k1_xor_n1,
      c1,
      c0,
      _mm_xor_si128(k0, c0),
      _mm_xor_si128(k1, c1),
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
    let four_blocks = BLOCK_SIZE.strict_mul(4);
    let two_blocks = BLOCK_SIZE.strict_mul(2);
    while offset.strict_add(four_blocks) <= len {
      _mm_prefetch(ptr.add(offset.strict_add(256)).cast::<i8>(), _MM_HINT_T0);
      let z_a = keystream_regs(s1, s2, s3, s4, s5);
      let xi_a = _mm_loadu_si128(ptr.add(offset).cast());
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi_a);
      _mm_storeu_si128(ptr.add(offset).cast(), _mm_xor_si128(xi_a, z_a));
      let z_b = keystream_regs(s1, s2, s3, s4, s5);
      let xi_b = _mm_loadu_si128(ptr.add(offset.strict_add(BLOCK_SIZE)).cast());
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi_b);
      _mm_storeu_si128(ptr.add(offset.strict_add(BLOCK_SIZE)).cast(), _mm_xor_si128(xi_b, z_b));
      let z_c = keystream_regs(s1, s2, s3, s4, s5);
      let xi_c = _mm_loadu_si128(ptr.add(offset.strict_add(two_blocks)).cast());
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi_c);
      _mm_storeu_si128(ptr.add(offset.strict_add(two_blocks)).cast(), _mm_xor_si128(xi_c, z_c));
      let z_d = keystream_regs(s1, s2, s3, s4, s5);
      let xi_d = _mm_loadu_si128(ptr.add(offset.strict_add(two_blocks.strict_add(BLOCK_SIZE))).cast());
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi_d);
      _mm_storeu_si128(
        ptr.add(offset.strict_add(two_blocks.strict_add(BLOCK_SIZE))).cast(),
        _mm_xor_si128(xi_d, z_d),
      );
      offset = offset.strict_add(four_blocks);
    }
    if offset.strict_add(two_blocks) <= len {
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

  #[target_feature(enable = "aes,avx")]
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
      k0_xor_n0,
      k1_xor_n1,
      c1,
      c0,
      _mm_xor_si128(k0, c0),
      _mm_xor_si128(k1, c1),
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
    let four_blocks = BLOCK_SIZE.strict_mul(4);
    let two_blocks = BLOCK_SIZE.strict_mul(2);
    while offset.strict_add(four_blocks) <= len {
      _mm_prefetch(ptr.add(offset.strict_add(256)).cast::<i8>(), _MM_HINT_T0);
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
      let z_c = keystream_regs(s1, s2, s3, s4, s5);
      let ci_c = _mm_loadu_si128(ptr.add(offset.strict_add(two_blocks)).cast());
      let xi_c = _mm_xor_si128(ci_c, z_c);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi_c);
      _mm_storeu_si128(ptr.add(offset.strict_add(two_blocks)).cast(), xi_c);
      let z_d = keystream_regs(s1, s2, s3, s4, s5);
      let ci_d = _mm_loadu_si128(ptr.add(offset.strict_add(two_blocks.strict_add(BLOCK_SIZE))).cast());
      let xi_d = _mm_xor_si128(ci_d, z_d);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi_d);
      _mm_storeu_si128(
        ptr.add(offset.strict_add(two_blocks.strict_add(BLOCK_SIZE))).cast(),
        xi_d,
      );
      offset = offset.strict_add(four_blocks);
    }
    if offset.strict_add(two_blocks) <= len {
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
  #[inline(always)]
  #[allow(clippy::too_many_arguments)]
  unsafe fn update_regs(
    s0: &mut uint8x16_t,
    s1: &mut uint8x16_t,
    s2: &mut uint8x16_t,
    s3: &mut uint8x16_t,
    s4: &mut uint8x16_t,
    s5: &mut uint8x16_t,
    m: uint8x16_t,
    zero: uint8x16_t,
  ) {
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
    key: &[u8; KEY_SIZE],
    nonce: &[u8; NONCE_SIZE],
    aad: &[u8],
    buffer: &mut [u8],
  ) -> [u8; TAG_SIZE] {
    let (kh0, kh1) = super::split_halves(key);
    let (nh0, nh1) = super::split_halves(nonce);
    let k0 = load(kh0);
    let k1 = load(kh1);
    let n0 = load(nh0);
    let n1 = load(nh1);
    let c0 = load(&C0);
    let c1 = load(&C1);
    let k0_xor_n0 = veorq_u8(k0, n0);
    let k1_xor_n1 = veorq_u8(k1, n1);
    let zero = vdupq_n_u8(0);
    let (mut s0, mut s1, mut s2, mut s3, mut s4, mut s5) =
      (k0_xor_n0, k1_xor_n1, c1, c0, veorq_u8(k0, c0), veorq_u8(k1, c1));
    for _ in 0..4 {
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, k0, zero);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, k1, zero);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, k0_xor_n0, zero);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, k1_xor_n1, zero);
    }
    let mut offset = 0usize;
    while offset.strict_add(BLOCK_SIZE) <= aad.len() {
      update_regs(
        &mut s0,
        &mut s1,
        &mut s2,
        &mut s3,
        &mut s4,
        &mut s5,
        vld1q_u8(aad.as_ptr().add(offset)),
        zero,
      );
      offset = offset.strict_add(BLOCK_SIZE);
    }
    if offset < aad.len() {
      let mut pad = [0u8; BLOCK_SIZE];
      pad[..aad.len().strict_sub(offset)].copy_from_slice(&aad[offset..]);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, load(&pad), zero);
    }
    let msg_len = buffer.len();
    let ptr = buffer.as_mut_ptr();
    let len = buffer.len();
    offset = 0;
    let four_blocks = BLOCK_SIZE.strict_mul(4);
    let two_blocks = BLOCK_SIZE.strict_mul(2);
    while offset.strict_add(four_blocks) <= len {
      core::arch::asm!("prfm pldl1keep, [{ptr}]", ptr = in(reg) ptr.add(offset.strict_add(192)), options(nostack, preserves_flags));
      let z_a = keystream_regs(s1, s2, s3, s4, s5);
      let xi_a = vld1q_u8(ptr.add(offset));
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi_a, zero);
      vst1q_u8(ptr.add(offset), veorq_u8(xi_a, z_a));
      let z_b = keystream_regs(s1, s2, s3, s4, s5);
      let xi_b = vld1q_u8(ptr.add(offset.strict_add(BLOCK_SIZE)));
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi_b, zero);
      vst1q_u8(ptr.add(offset.strict_add(BLOCK_SIZE)), veorq_u8(xi_b, z_b));
      let z_c = keystream_regs(s1, s2, s3, s4, s5);
      let xi_c = vld1q_u8(ptr.add(offset.strict_add(two_blocks)));
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi_c, zero);
      vst1q_u8(ptr.add(offset.strict_add(two_blocks)), veorq_u8(xi_c, z_c));
      let z_d = keystream_regs(s1, s2, s3, s4, s5);
      let xi_d = vld1q_u8(ptr.add(offset.strict_add(two_blocks.strict_add(BLOCK_SIZE))));
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi_d, zero);
      vst1q_u8(
        ptr.add(offset.strict_add(two_blocks.strict_add(BLOCK_SIZE))),
        veorq_u8(xi_d, z_d),
      );
      offset = offset.strict_add(four_blocks);
    }
    if offset.strict_add(two_blocks) <= len {
      let z_a = keystream_regs(s1, s2, s3, s4, s5);
      let xi_a = vld1q_u8(ptr.add(offset));
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi_a, zero);
      vst1q_u8(ptr.add(offset), veorq_u8(xi_a, z_a));
      let z_b = keystream_regs(s1, s2, s3, s4, s5);
      let xi_b = vld1q_u8(ptr.add(offset.strict_add(BLOCK_SIZE)));
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi_b, zero);
      vst1q_u8(ptr.add(offset.strict_add(BLOCK_SIZE)), veorq_u8(xi_b, z_b));
      offset = offset.strict_add(two_blocks);
    }
    if offset.strict_add(BLOCK_SIZE) <= len {
      let z = keystream_regs(s1, s2, s3, s4, s5);
      let xi = vld1q_u8(ptr.add(offset));
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi, zero);
      vst1q_u8(ptr.add(offset), veorq_u8(xi, z));
      offset = offset.strict_add(BLOCK_SIZE);
    }
    if offset < len {
      let z = keystream_regs(s1, s2, s3, s4, s5);
      let tail_len = len.strict_sub(offset);
      let mut pad = [0u8; BLOCK_SIZE];
      pad[..tail_len].copy_from_slice(&buffer[offset..]);
      let xi = load(&pad);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi, zero);
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
    for _ in 0..7 {
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, t, zero);
    }
    let tag_vec = veorq_u8(veorq_u8(veorq_u8(s0, s1), veorq_u8(s2, s3)), veorq_u8(s4, s5));
    let mut tag = [0u8; TAG_SIZE];
    store(tag_vec, &mut tag);
    tag
  }

  #[target_feature(enable = "aes,neon")]
  pub(super) unsafe fn decrypt_fused(
    key: &[u8; KEY_SIZE],
    nonce: &[u8; NONCE_SIZE],
    aad: &[u8],
    buffer: &mut [u8],
  ) -> [u8; TAG_SIZE] {
    let (kh0, kh1) = super::split_halves(key);
    let (nh0, nh1) = super::split_halves(nonce);
    let k0 = load(kh0);
    let k1 = load(kh1);
    let n0 = load(nh0);
    let n1 = load(nh1);
    let c0 = load(&C0);
    let c1 = load(&C1);
    let k0_xor_n0 = veorq_u8(k0, n0);
    let k1_xor_n1 = veorq_u8(k1, n1);
    let zero = vdupq_n_u8(0);
    let (mut s0, mut s1, mut s2, mut s3, mut s4, mut s5) =
      (k0_xor_n0, k1_xor_n1, c1, c0, veorq_u8(k0, c0), veorq_u8(k1, c1));
    for _ in 0..4 {
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, k0, zero);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, k1, zero);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, k0_xor_n0, zero);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, k1_xor_n1, zero);
    }
    let mut offset = 0usize;
    while offset.strict_add(BLOCK_SIZE) <= aad.len() {
      update_regs(
        &mut s0,
        &mut s1,
        &mut s2,
        &mut s3,
        &mut s4,
        &mut s5,
        vld1q_u8(aad.as_ptr().add(offset)),
        zero,
      );
      offset = offset.strict_add(BLOCK_SIZE);
    }
    if offset < aad.len() {
      let mut pad = [0u8; BLOCK_SIZE];
      pad[..aad.len().strict_sub(offset)].copy_from_slice(&aad[offset..]);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, load(&pad), zero);
    }
    let ct_len = buffer.len();
    let ptr = buffer.as_mut_ptr();
    let len = buffer.len();
    offset = 0;
    let four_blocks = BLOCK_SIZE.strict_mul(4);
    let two_blocks = BLOCK_SIZE.strict_mul(2);
    while offset.strict_add(four_blocks) <= len {
      core::arch::asm!("prfm pldl1keep, [{ptr}]", ptr = in(reg) ptr.add(offset.strict_add(192)), options(nostack, preserves_flags));
      let z_a = keystream_regs(s1, s2, s3, s4, s5);
      let ci_a = vld1q_u8(ptr.add(offset));
      let xi_a = veorq_u8(ci_a, z_a);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi_a, zero);
      vst1q_u8(ptr.add(offset), xi_a);
      let z_b = keystream_regs(s1, s2, s3, s4, s5);
      let ci_b = vld1q_u8(ptr.add(offset.strict_add(BLOCK_SIZE)));
      let xi_b = veorq_u8(ci_b, z_b);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi_b, zero);
      vst1q_u8(ptr.add(offset.strict_add(BLOCK_SIZE)), xi_b);
      let z_c = keystream_regs(s1, s2, s3, s4, s5);
      let ci_c = vld1q_u8(ptr.add(offset.strict_add(two_blocks)));
      let xi_c = veorq_u8(ci_c, z_c);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi_c, zero);
      vst1q_u8(ptr.add(offset.strict_add(two_blocks)), xi_c);
      let z_d = keystream_regs(s1, s2, s3, s4, s5);
      let ci_d = vld1q_u8(ptr.add(offset.strict_add(two_blocks.strict_add(BLOCK_SIZE))));
      let xi_d = veorq_u8(ci_d, z_d);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi_d, zero);
      vst1q_u8(ptr.add(offset.strict_add(two_blocks.strict_add(BLOCK_SIZE))), xi_d);
      offset = offset.strict_add(four_blocks);
    }
    if offset.strict_add(two_blocks) <= len {
      let z_a = keystream_regs(s1, s2, s3, s4, s5);
      let ci_a = vld1q_u8(ptr.add(offset));
      let xi_a = veorq_u8(ci_a, z_a);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi_a, zero);
      vst1q_u8(ptr.add(offset), xi_a);
      let z_b = keystream_regs(s1, s2, s3, s4, s5);
      let ci_b = vld1q_u8(ptr.add(offset.strict_add(BLOCK_SIZE)));
      let xi_b = veorq_u8(ci_b, z_b);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi_b, zero);
      vst1q_u8(ptr.add(offset.strict_add(BLOCK_SIZE)), xi_b);
      offset = offset.strict_add(two_blocks);
    }
    if offset.strict_add(BLOCK_SIZE) <= len {
      let z = keystream_regs(s1, s2, s3, s4, s5);
      let ci = vld1q_u8(ptr.add(offset));
      let xi = veorq_u8(ci, z);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi, zero);
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
      for i in 0..tail_len {
        pt_pad[i] = pad[i] ^ z_bytes[i];
      }
      update_regs(
        &mut s0,
        &mut s1,
        &mut s2,
        &mut s3,
        &mut s4,
        &mut s5,
        load(&pt_pad),
        zero,
      );
      buffer[offset..].copy_from_slice(&pt_pad[..tail_len]);
    }
    let ad_bits = (aad.len() as u64).strict_mul(8);
    let ct_bits = (ct_len as u64).strict_mul(8);
    let mut len_bytes = [0u8; BLOCK_SIZE];
    len_bytes[..8].copy_from_slice(&ad_bits.to_le_bytes());
    len_bytes[8..].copy_from_slice(&ct_bits.to_le_bytes());
    let t = veorq_u8(s3, load(&len_bytes));
    for _ in 0..7 {
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, t, zero);
    }
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
    key: &[u8; KEY_SIZE],
    nonce: &[u8; NONCE_SIZE],
    aad: &[u8],
    buffer: &mut [u8],
  ) -> [u8; TAG_SIZE] {
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
    let four_blocks = BLOCK_SIZE.strict_mul(4);
    let two_blocks = BLOCK_SIZE.strict_mul(2);
    while offset.strict_add(four_blocks) <= len {
      // SAFETY: pointer arithmetic for dcbt prefetch; offset + 256 may exceed
      // the buffer but dcbt is a hint and never faults on POWER.
      asm!("dcbt 0, {ptr}", ptr = in(reg) buffer.as_ptr().add(offset.strict_add(256)), options(nostack));
      // block a
      let z_a = keystream_regs(s1, s2, s3, s4, s5);
      let mut tmp_a = [0u8; 16];
      tmp_a.copy_from_slice(&buffer[offset..offset.strict_add(BLOCK_SIZE)]);
      let xi_a = load_be(&tmp_a);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi_a);
      store_be(xor_vec(xi_a, z_a), &mut tmp_a);
      buffer[offset..offset.strict_add(BLOCK_SIZE)].copy_from_slice(&tmp_a);
      // block b
      let z_b = keystream_regs(s1, s2, s3, s4, s5);
      let off_b = offset.strict_add(BLOCK_SIZE);
      let mut tmp_b = [0u8; 16];
      tmp_b.copy_from_slice(&buffer[off_b..off_b.strict_add(BLOCK_SIZE)]);
      let xi_b = load_be(&tmp_b);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi_b);
      store_be(xor_vec(xi_b, z_b), &mut tmp_b);
      buffer[off_b..off_b.strict_add(BLOCK_SIZE)].copy_from_slice(&tmp_b);
      // block c
      let z_c = keystream_regs(s1, s2, s3, s4, s5);
      let off_c = offset.strict_add(two_blocks);
      let mut tmp_c = [0u8; 16];
      tmp_c.copy_from_slice(&buffer[off_c..off_c.strict_add(BLOCK_SIZE)]);
      let xi_c = load_be(&tmp_c);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi_c);
      store_be(xor_vec(xi_c, z_c), &mut tmp_c);
      buffer[off_c..off_c.strict_add(BLOCK_SIZE)].copy_from_slice(&tmp_c);
      // block d
      let z_d = keystream_regs(s1, s2, s3, s4, s5);
      let off_d = offset.strict_add(two_blocks.strict_add(BLOCK_SIZE));
      let mut tmp_d = [0u8; 16];
      tmp_d.copy_from_slice(&buffer[off_d..off_d.strict_add(BLOCK_SIZE)]);
      let xi_d = load_be(&tmp_d);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi_d);
      store_be(xor_vec(xi_d, z_d), &mut tmp_d);
      buffer[off_d..off_d.strict_add(BLOCK_SIZE)].copy_from_slice(&tmp_d);
      offset = offset.strict_add(four_blocks);
    }
    if offset.strict_add(two_blocks) <= len {
      // block a
      let z_a = keystream_regs(s1, s2, s3, s4, s5);
      let mut tmp_a = [0u8; 16];
      tmp_a.copy_from_slice(&buffer[offset..offset.strict_add(BLOCK_SIZE)]);
      let xi_a = load_be(&tmp_a);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi_a);
      store_be(xor_vec(xi_a, z_a), &mut tmp_a);
      buffer[offset..offset.strict_add(BLOCK_SIZE)].copy_from_slice(&tmp_a);
      // block b
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
    for _ in 0..7 {
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, t);
    }
    let tag_vec = xor_vec(xor_vec(xor_vec(s0, s1), xor_vec(s2, s3)), xor_vec(s4, s5));
    let mut tag = [0u8; TAG_SIZE];
    store_be(tag_vec, &mut tag);
    tag
  }

  #[target_feature(enable = "altivec,vsx,power8-vector,power8-crypto")]
  pub(super) unsafe fn decrypt_fused(
    key: &[u8; KEY_SIZE],
    nonce: &[u8; NONCE_SIZE],
    aad: &[u8],
    buffer: &mut [u8],
  ) -> [u8; TAG_SIZE] {
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
    let four_blocks = BLOCK_SIZE.strict_mul(4);
    let two_blocks = BLOCK_SIZE.strict_mul(2);
    while offset.strict_add(four_blocks) <= len {
      // SAFETY: pointer arithmetic for dcbt prefetch; offset + 256 may exceed
      // the buffer but dcbt is a hint and never faults on POWER.
      asm!("dcbt 0, {ptr}", ptr = in(reg) buffer.as_ptr().add(offset.strict_add(256)), options(nostack));
      // block a
      let z_a = keystream_regs(s1, s2, s3, s4, s5);
      let mut tmp_a = [0u8; 16];
      tmp_a.copy_from_slice(&buffer[offset..offset.strict_add(BLOCK_SIZE)]);
      let xi_a = xor_vec(load_be(&tmp_a), z_a);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi_a);
      store_be(xi_a, &mut tmp_a);
      buffer[offset..offset.strict_add(BLOCK_SIZE)].copy_from_slice(&tmp_a);
      // block b
      let z_b = keystream_regs(s1, s2, s3, s4, s5);
      let off_b = offset.strict_add(BLOCK_SIZE);
      let mut tmp_b = [0u8; 16];
      tmp_b.copy_from_slice(&buffer[off_b..off_b.strict_add(BLOCK_SIZE)]);
      let xi_b = xor_vec(load_be(&tmp_b), z_b);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi_b);
      store_be(xi_b, &mut tmp_b);
      buffer[off_b..off_b.strict_add(BLOCK_SIZE)].copy_from_slice(&tmp_b);
      // block c
      let z_c = keystream_regs(s1, s2, s3, s4, s5);
      let off_c = offset.strict_add(two_blocks);
      let mut tmp_c = [0u8; 16];
      tmp_c.copy_from_slice(&buffer[off_c..off_c.strict_add(BLOCK_SIZE)]);
      let xi_c = xor_vec(load_be(&tmp_c), z_c);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi_c);
      store_be(xi_c, &mut tmp_c);
      buffer[off_c..off_c.strict_add(BLOCK_SIZE)].copy_from_slice(&tmp_c);
      // block d
      let z_d = keystream_regs(s1, s2, s3, s4, s5);
      let off_d = offset.strict_add(two_blocks.strict_add(BLOCK_SIZE));
      let mut tmp_d = [0u8; 16];
      tmp_d.copy_from_slice(&buffer[off_d..off_d.strict_add(BLOCK_SIZE)]);
      let xi_d = xor_vec(load_be(&tmp_d), z_d);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi_d);
      store_be(xi_d, &mut tmp_d);
      buffer[off_d..off_d.strict_add(BLOCK_SIZE)].copy_from_slice(&tmp_d);
      offset = offset.strict_add(four_blocks);
    }
    if offset.strict_add(two_blocks) <= len {
      // block a
      let z_a = keystream_regs(s1, s2, s3, s4, s5);
      let mut tmp_a = [0u8; 16];
      tmp_a.copy_from_slice(&buffer[offset..offset.strict_add(BLOCK_SIZE)]);
      let xi_a = xor_vec(load_be(&tmp_a), z_a);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi_a);
      store_be(xi_a, &mut tmp_a);
      buffer[offset..offset.strict_add(BLOCK_SIZE)].copy_from_slice(&tmp_a);
      // block b
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
      for i in 0..tail_len {
        pt_pad[i] = pad[i] ^ z_bytes[i];
      }
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, load_be(&pt_pad));
      buffer[offset..].copy_from_slice(&pt_pad[..tail_len]);
    }
    let ad_bits = (aad.len() as u64).strict_mul(8);
    let ct_bits = (ct_len as u64).strict_mul(8);
    let mut len_bytes = [0u8; BLOCK_SIZE];
    len_bytes[..8].copy_from_slice(&ad_bits.to_le_bytes());
    len_bytes[8..].copy_from_slice(&ct_bits.to_le_bytes());
    let t = xor_vec(s3, load_be(&len_bytes));
    for _ in 0..7 {
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, t);
    }
    let tag_vec = xor_vec(xor_vec(xor_vec(s0, s1), xor_vec(s2, s3)), xor_vec(s4, s5));
    let mut tag = [0u8; TAG_SIZE];
    store_be(tag_vec, &mut tag);
    tag
  }
}

// ---------------------------------------------------------------------------
// s390x T-table backend
// ---------------------------------------------------------------------------
//
// s390x has NO single-round AES instruction (no equivalent of x86 AESENC,
// ARM AESE, or POWER vcipher). CPACF instructions (KM/KMA) only perform
// full-block AES encryption, which cannot be used for AEGIS's individual
// AES round function.
//
// This backend implements the AES round function using T-tables: 4 × 256-
// entry lookup tables that fuse SubBytes + MixColumns into a single 32-bit
// load per byte, matching the approach used by libaegis. Each AES round
// costs ~16 loads + 12 XOR + ShiftRows indexing.
//
// T-tables are NOT constant-time (cache-timing side channel). This is
// acceptable for AEGIS because: (1) libaegis — the de facto standard —
// uses T-tables by default; (2) AEGIS is designed for high-throughput
// bulk AEAD, not scenarios where cache-timing is the primary threat model;
// (3) the alternative (algebraic S-box) is ~100x slower.

#[cfg(target_arch = "s390x")]
mod zvec {
  use super::{BLOCK_SIZE, C0, C1, KEY_SIZE, NONCE_SIZE, TAG_SIZE};

  // ── T-table AES round ────────────────────────────────────────────────────
  //
  // T-tables fuse SubBytes + MixColumns into a single 32-bit lookup per
  // input byte. T0 is the base table; T1-T3 are byte rotations of T0.
  // ShiftRows is handled by indexing into different columns of the state.

  // ── AES S-box (FIPS 197) ──

  /// Full AES S-box lookup table (256 entries).
  #[rustfmt::skip]
  const SBOX: [u8; 256] = [
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
  ];

  // ── T-table generation ──

  /// xtime: multiply by 2 in GF(2^8) with AES polynomial.
  const fn xt(b: u8) -> u8 {
    let r = (b as u16) << 1;
    (r ^ (if r & 0x100 != 0 { 0x1B } else { 0 })) as u8
  }

  /// Generate T0: T0[b] = column [2*S(b), S(b), S(b), 3*S(b)] as big-endian u32.
  /// T1/T2/T3 are byte rotations of T0.
  const fn generate_t0() -> [u32; 256] {
    let mut t = [0u32; 256];
    let mut i = 0;
    while i < 256 {
      let s = SBOX[i];
      let s2 = xt(s);
      let s3 = s2 ^ s;
      t[i] = (s2 as u32) << 24 | (s as u32) << 16 | (s as u32) << 8 | s3 as u32;
      i += 1;
    }
    t
  }

  const fn generate_t1() -> [u32; 256] {
    let t0 = generate_t0();
    let mut t = [0u32; 256];
    let mut i = 0;
    while i < 256 {
      t[i] = t0[i].rotate_right(8);
      i += 1;
    }
    t
  }

  const fn generate_t2() -> [u32; 256] {
    let t0 = generate_t0();
    let mut t = [0u32; 256];
    let mut i = 0;
    while i < 256 {
      t[i] = t0[i].rotate_right(16);
      i += 1;
    }
    t
  }

  const fn generate_t3() -> [u32; 256] {
    let t0 = generate_t0();
    let mut t = [0u32; 256];
    let mut i = 0;
    while i < 256 {
      t[i] = t0[i].rotate_right(24);
      i += 1;
    }
    t
  }

  static T0: [u32; 256] = generate_t0();
  static T1: [u32; 256] = generate_t1();
  static T2: [u32; 256] = generate_t2();
  static T3: [u32; 256] = generate_t3();

  // ── Block helpers ──────────────────────────────────────────────────────

  type Block = [u8; BLOCK_SIZE];

  #[inline(always)]
  fn xor_block(a: &Block, b: &Block) -> Block {
    let mut out = [0u8; BLOCK_SIZE];
    let mut i = 0;
    while i < BLOCK_SIZE {
      out[i] = a[i] ^ b[i];
      i = i.strict_add(1);
    }
    out
  }

  #[inline(always)]
  fn and_block(a: &Block, b: &Block) -> Block {
    let mut out = [0u8; BLOCK_SIZE];
    let mut i = 0;
    while i < BLOCK_SIZE {
      out[i] = a[i] & b[i];
      i = i.strict_add(1);
    }
    out
  }

  // ── T-table AES round ──────────────────────────────────────────────────

  /// Single AES round via T-tables: SubBytes + ShiftRows + MixColumns + AddRoundKey.
  ///
  /// T-tables fuse SubBytes + MixColumns into a single 32-bit lookup per byte.
  /// ShiftRows is handled by indexing into the correct column positions.
  #[inline(always)]
  fn aes_round(block: &Block, round_key: &Block) -> Block {
    // Load state as 4 big-endian column words.
    let s0 = u32::from_be_bytes([block[0], block[1], block[2], block[3]]);
    let s1 = u32::from_be_bytes([block[4], block[5], block[6], block[7]]);
    let s2 = u32::from_be_bytes([block[8], block[9], block[10], block[11]]);
    let s3 = u32::from_be_bytes([block[12], block[13], block[14], block[15]]);

    // Load round key as 4 big-endian column words.
    let k0 = u32::from_be_bytes([round_key[0], round_key[1], round_key[2], round_key[3]]);
    let k1 = u32::from_be_bytes([round_key[4], round_key[5], round_key[6], round_key[7]]);
    let k2 = u32::from_be_bytes([round_key[8], round_key[9], round_key[10], round_key[11]]);
    let k3 = u32::from_be_bytes([round_key[12], round_key[13], round_key[14], round_key[15]]);

    // T-table lookups with ShiftRows baked into the column indexing.
    // Column j uses bytes from rows (0,j), (1,j+1), (2,j+2), (3,j+3) mod 4.
    // In big-endian column words: byte 0 = row 0 (bits 31:24), byte 3 = row 3 (bits 7:0).
    let c0 = T0[(s0 >> 24) as usize]
      ^ T1[((s1 >> 16) & 0xFF) as usize]
      ^ T2[((s2 >> 8) & 0xFF) as usize]
      ^ T3[(s3 & 0xFF) as usize]
      ^ k0;

    let c1 = T0[(s1 >> 24) as usize]
      ^ T1[((s2 >> 16) & 0xFF) as usize]
      ^ T2[((s3 >> 8) & 0xFF) as usize]
      ^ T3[(s0 & 0xFF) as usize]
      ^ k1;

    let c2 = T0[(s2 >> 24) as usize]
      ^ T1[((s3 >> 16) & 0xFF) as usize]
      ^ T2[((s0 >> 8) & 0xFF) as usize]
      ^ T3[(s1 & 0xFF) as usize]
      ^ k2;

    let c3 = T0[(s3 >> 24) as usize]
      ^ T1[((s0 >> 16) & 0xFF) as usize]
      ^ T2[((s1 >> 8) & 0xFF) as usize]
      ^ T3[(s2 & 0xFF) as usize]
      ^ k3;

    let mut out = [0u8; BLOCK_SIZE];
    out[0..4].copy_from_slice(&c0.to_be_bytes());
    out[4..8].copy_from_slice(&c1.to_be_bytes());
    out[8..12].copy_from_slice(&c2.to_be_bytes());
    out[12..16].copy_from_slice(&c3.to_be_bytes());
    out
  }

  // ── AEGIS-256 state operations ──────────────────────────────────────────

  type State = [Block; 6];

  #[inline(always)]
  fn update(s: &mut State, m: &Block) {
    let tmp = s[5];
    s[5] = aes_round(&s[4], &s[5]);
    s[4] = aes_round(&s[3], &s[4]);
    s[3] = aes_round(&s[2], &s[3]);
    s[2] = aes_round(&s[1], &s[2]);
    s[1] = aes_round(&s[0], &s[1]);
    s[0] = xor_block(&aes_round(&tmp, &s[0]), m);
  }

  #[inline(always)]
  fn keystream(s: &State) -> Block {
    xor_block(&xor_block(&s[1], &s[4]), &xor_block(&s[5], &and_block(&s[2], &s[3])))
  }

  // ── Fused encrypt/decrypt ───────────────────────────────────────────────

  pub(super) fn encrypt_fused(
    key: &[u8; KEY_SIZE],
    nonce: &[u8; NONCE_SIZE],
    aad: &[u8],
    buffer: &mut [u8],
  ) -> [u8; TAG_SIZE] {
    let (kh0, kh1) = super::split_halves(key);
    let (nh0, nh1) = super::split_halves(nonce);
    let k0_xor_n0 = xor_block(kh0, nh0);
    let k1_xor_n1 = xor_block(kh1, nh1);
    let mut s: State = [k0_xor_n0, k1_xor_n1, C1, C0, xor_block(kh0, &C0), xor_block(kh1, &C1)];

    for _ in 0..4 {
      update(&mut s, kh0);
      update(&mut s, kh1);
      update(&mut s, &k0_xor_n0);
      update(&mut s, &k1_xor_n1);
    }

    // ── aad ──
    let mut offset = 0usize;
    while offset.strict_add(BLOCK_SIZE) <= aad.len() {
      let mut tmp = [0u8; BLOCK_SIZE];
      tmp.copy_from_slice(&aad[offset..offset.strict_add(BLOCK_SIZE)]);
      update(&mut s, &tmp);
      offset = offset.strict_add(BLOCK_SIZE);
    }
    if offset < aad.len() {
      let mut pad = [0u8; BLOCK_SIZE];
      pad[..aad.len().strict_sub(offset)].copy_from_slice(&aad[offset..]);
      update(&mut s, &pad);
    }

    // ── encrypt ──
    let msg_len = buffer.len();
    let len = buffer.len();
    offset = 0;
    while offset.strict_add(BLOCK_SIZE) <= len {
      let z = keystream(&s);
      let mut xi = [0u8; BLOCK_SIZE];
      xi.copy_from_slice(&buffer[offset..offset.strict_add(BLOCK_SIZE)]);
      update(&mut s, &xi);
      buffer[offset..offset.strict_add(BLOCK_SIZE)].copy_from_slice(&xor_block(&xi, &z));
      offset = offset.strict_add(BLOCK_SIZE);
    }
    if offset < len {
      let z = keystream(&s);
      let tail_len = len.strict_sub(offset);
      let mut pad = [0u8; BLOCK_SIZE];
      pad[..tail_len].copy_from_slice(&buffer[offset..]);
      update(&mut s, &pad);
      let ct = xor_block(&pad, &z);
      buffer[offset..].copy_from_slice(&ct[..tail_len]);
    }

    // ── finalize ──
    let ad_bits = (aad.len() as u64).strict_mul(8);
    let msg_bits = (msg_len as u64).strict_mul(8);
    let mut len_bytes = [0u8; BLOCK_SIZE];
    len_bytes[..8].copy_from_slice(&ad_bits.to_le_bytes());
    len_bytes[8..].copy_from_slice(&msg_bits.to_le_bytes());
    let t = xor_block(&s[3], &len_bytes);
    for _ in 0..7 {
      update(&mut s, &t);
    }
    xor_block(
      &xor_block(&xor_block(&s[0], &s[1]), &xor_block(&s[2], &s[3])),
      &xor_block(&s[4], &s[5]),
    )
  }

  pub(super) fn decrypt_fused(
    key: &[u8; KEY_SIZE],
    nonce: &[u8; NONCE_SIZE],
    aad: &[u8],
    buffer: &mut [u8],
  ) -> [u8; TAG_SIZE] {
    let (kh0, kh1) = super::split_halves(key);
    let (nh0, nh1) = super::split_halves(nonce);
    let k0_xor_n0 = xor_block(kh0, nh0);
    let k1_xor_n1 = xor_block(kh1, nh1);
    let mut s: State = [k0_xor_n0, k1_xor_n1, C1, C0, xor_block(kh0, &C0), xor_block(kh1, &C1)];

    for _ in 0..4 {
      update(&mut s, kh0);
      update(&mut s, kh1);
      update(&mut s, &k0_xor_n0);
      update(&mut s, &k1_xor_n1);
    }

    // ── aad ──
    let mut offset = 0usize;
    while offset.strict_add(BLOCK_SIZE) <= aad.len() {
      let mut tmp = [0u8; BLOCK_SIZE];
      tmp.copy_from_slice(&aad[offset..offset.strict_add(BLOCK_SIZE)]);
      update(&mut s, &tmp);
      offset = offset.strict_add(BLOCK_SIZE);
    }
    if offset < aad.len() {
      let mut pad = [0u8; BLOCK_SIZE];
      pad[..aad.len().strict_sub(offset)].copy_from_slice(&aad[offset..]);
      update(&mut s, &pad);
    }

    // ── decrypt ──
    let ct_len = buffer.len();
    let len = buffer.len();
    offset = 0;
    while offset.strict_add(BLOCK_SIZE) <= len {
      let z = keystream(&s);
      let mut ci = [0u8; BLOCK_SIZE];
      ci.copy_from_slice(&buffer[offset..offset.strict_add(BLOCK_SIZE)]);
      let xi = xor_block(&ci, &z);
      update(&mut s, &xi);
      buffer[offset..offset.strict_add(BLOCK_SIZE)].copy_from_slice(&xi);
      offset = offset.strict_add(BLOCK_SIZE);
    }
    if offset < len {
      let z = keystream(&s);
      let tail_len = len.strict_sub(offset);
      let mut pad = [0u8; BLOCK_SIZE];
      pad[..tail_len].copy_from_slice(&buffer[offset..]);
      let mut pt_pad = [0u8; BLOCK_SIZE];
      for i in 0..tail_len {
        pt_pad[i] = pad[i] ^ z[i];
      }
      update(&mut s, &pt_pad);
      buffer[offset..].copy_from_slice(&pt_pad[..tail_len]);
    }

    // ── finalize ──
    let ad_bits = (aad.len() as u64).strict_mul(8);
    let ct_bits = (ct_len as u64).strict_mul(8);
    let mut len_bytes = [0u8; BLOCK_SIZE];
    len_bytes[..8].copy_from_slice(&ad_bits.to_le_bytes());
    len_bytes[8..].copy_from_slice(&ct_bits.to_le_bytes());
    let t = xor_block(&s[3], &len_bytes);
    for _ in 0..7 {
      update(&mut s, &t);
    }
    xor_block(
      &xor_block(&xor_block(&s[0], &s[1]), &xor_block(&s[2], &s[3])),
      &xor_block(&s[4], &s[5]),
    )
  }
}
// ---------------------------------------------------------------------------

/// Whether hardware AES acceleration is available.
///
/// Whether hardware AES acceleration is available (x86_64/aarch64/powerpc64).
///
/// s390x uses T-table software AES rounds unconditionally (no HW check needed).
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
    let caps = crate::platform::caps();
    caps.has(x86::AESNI) && caps.has(x86::AVX)
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

#[cfg(not(target_arch = "s390x"))]
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

#[cfg(not(target_arch = "s390x"))]
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
      // SAFETY: has_hw_aes() confirmed AES-NI + AVX are available.
      // VEX encoding gives 3-operand VAESENC, eliminating register copies.
      let tag = unsafe { ni::encrypt_fused(key, nonce, aad, buffer) };
      return Aegis256Tag::from_bytes(tag);
    }

    #[cfg(target_arch = "aarch64")]
    if has_hw_aes() {
      // SAFETY: has_hw_aes() confirmed AES-CE is available.
      let tag = unsafe { ce::encrypt_fused(key, nonce, aad, buffer) };
      return Aegis256Tag::from_bytes(tag);
    }

    #[cfg(all(target_arch = "powerpc64", target_endian = "little"))]
    if has_hw_aes() {
      // SAFETY: has_hw_aes() confirmed POWER8 crypto is available.
      let tag = unsafe { ppc::encrypt_fused(key, nonce, aad, buffer) };
      return Aegis256Tag::from_bytes(tag);
    }

    #[cfg(target_arch = "s390x")]
    {
      // T-table AES rounds — always available, no hardware feature check needed.
      #[allow(clippy::needless_return)]
      return Aegis256Tag::from_bytes(zvec::encrypt_fused(key, nonce, aad, buffer));
    }

    #[cfg(not(target_arch = "s390x"))]
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
      // SAFETY: has_hw_aes() confirmed AES-NI + AVX are available.
      // VEX encoding gives 3-operand VAESENC, eliminating register copies.
      unsafe { ni::decrypt_fused(key, nonce, aad, buffer) }
    } else {
      decrypt_portable(key, nonce, aad, buffer)
    };

    #[cfg(target_arch = "aarch64")]
    let computed = if has_hw_aes() {
      // SAFETY: has_hw_aes() confirmed AES-CE is available.
      unsafe { ce::decrypt_fused(key, nonce, aad, buffer) }
    } else {
      decrypt_portable(key, nonce, aad, buffer)
    };

    #[cfg(all(target_arch = "powerpc64", target_endian = "little"))]
    let computed = if has_hw_aes() {
      // SAFETY: has_hw_aes() confirmed POWER8 crypto is available.
      unsafe { ppc::decrypt_fused(key, nonce, aad, buffer) }
    } else {
      decrypt_portable(key, nonce, aad, buffer)
    };

    #[cfg(target_arch = "s390x")]
    let computed = zvec::decrypt_fused(key, nonce, aad, buffer);

    #[cfg(not(any(
      target_arch = "x86_64",
      target_arch = "aarch64",
      all(target_arch = "powerpc64", target_endian = "little"),
      target_arch = "s390x",
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

  #[cfg(not(target_arch = "s390x"))]
  #[inline(always)]
  fn portable_aes_round(block: &[u8; 16], round_key: &[u8; 16]) -> [u8; 16] {
    super::super::aes::aes_enc_round_portable(block, round_key)
  }

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

  #[cfg(not(target_arch = "s390x"))]
  #[test]
  fn aes_round_matches_spec_vector() {
    let input = hex_block("000102030405060708090a0b0c0d0e0f");
    let rk = hex_block("101112131415161718191a1b1c1d1e1f");
    let expected = hex_block("7a7b4e5638782546a8c0477a3b813f43");

    assert_eq!(portable_aes_round(&input, &rk), expected);
  }

  // -- Update test vector (Appendix A.2) --

  #[cfg(not(target_arch = "s390x"))]
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

  /// Round-trip at 4-block boundary sizes to exercise the 4x unrolled loop.
  #[test]
  fn four_block_boundaries() {
    let key = Aegis256Key::from_bytes([0xF4; 32]);
    let nonce = Nonce256::from_bytes([0xB0; 32]);
    let aead = Aegis256::new(&key);
    let aad = b"four-block-test";

    for &size in &[48, 64, 80, 96, 112, 128, 256, 1024, 4096] {
      let plaintext: Vec<u8> = (0..size).map(|i| (i & 0xFF) as u8).collect();
      let mut buf = plaintext.clone();
      let tag = aead.encrypt_in_place(&nonce, aad, &mut buf);
      assert_ne!(&buf, &plaintext, "size {size}: ciphertext must differ");
      aead.decrypt_in_place(&nonce, aad, &mut buf, &tag).unwrap();
      assert_eq!(&buf, &plaintext, "size {size}: round-trip failed");
    }
  }

  // -- vperm S-box table validation --
  //
  // Verifies the Hamburg tower-field lookup tables produce the correct AES
  // S-box output for all 256 input values. This runs on all platforms using
  // pure scalar simulation of the VPERM operations, catching transcription
  // errors in the constant tables without needing s390x hardware.

  /// Scalar simulation of vperm: table[index & 0x0F].
  fn vperm_scalar(table: &[u8; 16], index: u8) -> u8 {
    table[(index & 0x0F) as usize]
  }

  /// Scalar simulation of vperm with PSHUFB zeroing: returns 0 when bit 7 set.
  fn vperm_z_scalar(table: &[u8; 16], index: u8) -> u8 {
    if index & 0x80 != 0 {
      0
    } else {
      table[(index & 0x0F) as usize]
    }
  }

  // Inline copies of the Hamburg vperm tables for cross-platform testing.
  // These MUST match the constants in `mod zvec` exactly.
  const T_IPT_LO: [u8; 16] = [
    0x00, 0x70, 0x2A, 0x5A, 0x98, 0xE8, 0xB2, 0xC2, 0x08, 0x78, 0x22, 0x52, 0x90, 0xE0, 0xBA, 0xCA,
  ];
  const T_IPT_HI: [u8; 16] = [
    0x00, 0x4D, 0x7C, 0x31, 0x7D, 0x30, 0x01, 0x4C, 0x81, 0xCC, 0xFD, 0xB0, 0xFC, 0xB1, 0x80, 0xCD,
  ];
  const T_INV_LO: [u8; 16] = [
    0x80, 0x01, 0x08, 0x0D, 0x0F, 0x06, 0x05, 0x0E, 0x02, 0x0C, 0x0B, 0x0A, 0x09, 0x03, 0x07, 0x04,
  ];
  const T_INV_HI: [u8; 16] = [
    0x80, 0x07, 0x0B, 0x0F, 0x06, 0x0A, 0x04, 0x01, 0x09, 0x08, 0x05, 0x02, 0x0C, 0x0E, 0x0D, 0x03,
  ];
  // sbo (SubBytes Output) tables: pure S-box without MixColumns (for validation).
  // From OpenSSL Lk_sbo. These give: sbou[io] ^ sbot[jo] = AES_SBOX[input].
  const T_SBOU: [u8; 16] = [
    0x00, 0xC7, 0xBD, 0x6F, 0x17, 0x6D, 0xD2, 0xD0, 0x78, 0xA8, 0x02, 0xC5, 0x7A, 0xBF, 0xAA, 0x15,
  ];
  const T_SBOT: [u8; 16] = [
    0x00, 0x6A, 0xBB, 0x5F, 0xA5, 0x74, 0xE4, 0xCF, 0xFA, 0x35, 0x2B, 0x41, 0xD1, 0x90, 0x1E, 0x8E,
  ];

  /// Compute AES S-box for one byte using the vperm tower-field tables.
  ///
  /// This mirrors the `aes_round` Phase 2-4 logic (input transform,
  /// GF(2^4) inverse, output transform) but operates on a single byte
  /// without SIMD.
  fn vperm_sbox_scalar(input: u8) -> u8 {
    let lo_nib = input & 0x0F;
    let hi_nib = input >> 4;

    // Phase 2: Input transform (AES basis → tower field)
    let ipt_l = vperm_scalar(&T_IPT_LO, lo_nib);
    let ipt_h = vperm_scalar(&T_IPT_HI, hi_nib);
    let x = ipt_l ^ ipt_h;

    // Phase 3: Nibble extraction of transformed value
    let t_lo = x & 0x0F;
    let t_hi = x >> 4;

    // Phase 4: GF(2^4) inverse (5 vperm + 5 XOR)
    // vperm_z_scalar emulates PSHUFB zeroing for indices with bit 7 set.
    let ak = vperm_scalar(&T_INV_HI, t_lo);
    let j = t_hi ^ t_lo;
    let inv_i = vperm_scalar(&T_INV_LO, t_hi);
    let iak = inv_i ^ ak;
    let inv_j = vperm_scalar(&T_INV_LO, j);
    let jak = inv_j ^ ak;
    let inv_iak = vperm_z_scalar(&T_INV_LO, iak); // zeroing for 0x80 sentinel
    let io = inv_iak ^ j; // output high nibble
    let inv_jak = vperm_z_scalar(&T_INV_LO, jak); // zeroing for 0x80 sentinel
    let jo = inv_jak ^ t_hi; // output low nibble

    // Phase 5: Output transform (SubBytes only, no MixColumns)
    // sbo tables encode the combined inverse-isomorphism + AES affine
    // WITHOUT MixColumns — giving the pure S-box output.
    // io/jo can have bit 7 set from the 0x80 sentinel, so use vperm_z.
    let su = vperm_z_scalar(&T_SBOU, io);
    let st = vperm_z_scalar(&T_SBOT, jo);
    su ^ st
  }

  #[test]
  fn vperm_sbox_tables_match_aes_sbox() {
    // The canonical AES S-box (FIPS 197, Table 4).
    #[rustfmt::skip]
    const AES_SBOX: [u8; 256] = [
      0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
      0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
      0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
      0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
      0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
      0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
      0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
      0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
      0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
      0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
      0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
      0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
      0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
      0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
      0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
      0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
    ];

    let mut failures = 0u32;
    for input in 0u16..256 {
      let got = vperm_sbox_scalar(input as u8);
      // The Hamburg vperm S-box omits the AES affine constant 0x63.
      // vpaes_sbox(x) = AES_sbox(x) ^ 0x63 for all x.
      let expected = AES_SBOX[input as usize] ^ 0x63;
      if got != expected {
        if failures < 16 {
          eprintln!(
            "vperm S-box mismatch at input 0x{:02X}: got 0x{:02X}, expected 0x{:02X}",
            input, got, expected,
          );
        }
        failures = failures.strict_add(1);
      }
    }
    assert_eq!(failures, 0, "{failures} vperm S-box mismatches out of 256");
  }

  // -- Full vperm AES round validation --
  //
  // Simulates the complete vperm AES round (SubBytes + ShiftRows +
  // MixColumns + AddRoundKey) using scalar operations and verifies
  // it matches the portable aes_enc_round_portable() for multiple
  // test vectors.

  /// Full vperm AES round simulation (scalar): SubBytes → ShiftRows → MixColumns → AddRoundKey.
  #[cfg(not(target_arch = "s390x"))]
  fn vperm_aes_round_scalar(block: &[u8; 16], round_key: &[u8; 16]) -> [u8; 16] {
    const SR: [u8; 16] = [
      0x00, 0x05, 0x0A, 0x0F, 0x04, 0x09, 0x0E, 0x03, 0x08, 0x0D, 0x02, 0x07, 0x0C, 0x01, 0x06, 0x0B,
    ];

    // SubBytes via vperm tower field (includes affine constant compensation)
    let mut sb = [0u8; 16];
    for i in 0..16 {
      sb[i] = vperm_sbox_scalar(block[i]) ^ 0x63;
    }

    // ShiftRows
    let mut sr = [0u8; 16];
    for i in 0..16 {
      sr[i] = sb[SR[i] as usize];
    }

    // MixColumns via xtime decomposition
    fn xtime(b: u8) -> u8 {
      let r = (b as u16) << 1;
      (r ^ (if r & 0x100 != 0 { 0x1B } else { 0 })) as u8
    }
    let mut mc = [0u8; 16];
    for col in 0..4 {
      let c = col * 4;
      let (b0, b1, b2, b3) = (sr[c], sr[c + 1], sr[c + 2], sr[c + 3]);
      mc[c] = xtime(b0) ^ xtime(b1) ^ b1 ^ b2 ^ b3;
      mc[c + 1] = b0 ^ xtime(b1) ^ xtime(b2) ^ b2 ^ b3;
      mc[c + 2] = b0 ^ b1 ^ xtime(b2) ^ xtime(b3) ^ b3;
      mc[c + 3] = xtime(b0) ^ b0 ^ b1 ^ b2 ^ xtime(b3);
    }

    // AddRoundKey
    let mut result = [0u8; 16];
    for i in 0..16 {
      result[i] = mc[i] ^ round_key[i];
    }
    result
  }

  #[cfg(not(target_arch = "s390x"))]
  #[test]
  fn vperm_full_round_matches_portable() {
    let input = hex_block("000102030405060708090a0b0c0d0e0f");
    let rk = hex_block("101112131415161718191a1b1c1d1e1f");
    let expected = portable_aes_round(&input, &rk);

    let got = vperm_aes_round_scalar(&input, &rk);
    assert_eq!(got, expected, "vperm round mismatch for spec vector");

    // Test with all-zero
    let zero = [0u8; 16];
    assert_eq!(vperm_aes_round_scalar(&zero, &zero), portable_aes_round(&zero, &zero));

    // Test with all-0xFF
    let ff = [0xFFu8; 16];
    assert_eq!(vperm_aes_round_scalar(&ff, &ff), portable_aes_round(&ff, &ff));

    // Test with random-looking patterns
    let a = hex_block("deadbeefcafebabe0123456789abcdef");
    let b = hex_block("fedcba9876543210aabbccddeeff0011");
    assert_eq!(vperm_aes_round_scalar(&a, &b), portable_aes_round(&a, &b));

    // Exhaustive: test all single-byte patterns in position 0
    for val in 0u16..256 {
      let mut block = [0u8; 16];
      block[0] = val as u8;
      let key = [0u8; 16];
      let got = vperm_aes_round_scalar(&block, &key);
      let expected = portable_aes_round(&block, &key);
      assert_eq!(got, expected, "vperm round mismatch for block[0]=0x{:02X}", val,);
    }
  }
}
