use core::arch::x86_64::*;

use super::{BLOCK_SIZE, C0, C1, KEY_SIZE, NONCE_SIZE, TAG_SIZE};

#[inline]
/// # Safety
///
/// `bytes` must refer to a valid 16-byte block.
unsafe fn load(bytes: &[u8; BLOCK_SIZE]) -> __m128i {
  _mm_loadu_si128(bytes.as_ptr().cast())
}

#[inline]
/// # Safety
///
/// `out` must refer to a valid writable 16-byte block.
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

#[inline]
/// # Safety
///
/// Caller must ensure AES-NI is available and all register arguments come
/// from a valid AEGIS-256 state.
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

#[inline]
/// # Safety
///
/// The provided registers must come from a valid AEGIS-256 state on an
/// AES-NI capable CPU.
unsafe fn keystream_regs(s1: __m128i, s2: __m128i, s3: __m128i, s4: __m128i, s5: __m128i) -> __m128i {
  _mm_xor_si128(_mm_xor_si128(s1, s4), _mm_xor_si128(s5, _mm_and_si128(s2, s3)))
}

// ── Fused encrypt/decrypt ───────────────────────────────────────────────
//
// Single `#[target_feature]` entry points that keep state in XMM registers
// from init through finalize, eliminating ~15 cycles of stack spills that
// occur when init/aad/encrypt/finalize are separate function calls.

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
  //
  // Counted loop with raw pointer advancement: eliminates per-iteration
  // overflow-check branches (strict_add) from the hot path. The loop count
  // is pre-validated via integer division, and all intra-block offsets are
  // compile-time constants applied through ptr::add inside this unsafe block.
  let msg_len = buffer.len();
  let ptr = buffer.as_mut_ptr();
  let n_quads = msg_len / 64;
  let mut p = ptr;
  for _ in 0..n_quads {
    _mm_prefetch(p.add(256).cast::<i8>(), _MM_HINT_T0);
    let z_a = keystream_regs(s1, s2, s3, s4, s5);
    let xi_a = _mm_loadu_si128(p.cast());
    update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi_a);
    _mm_storeu_si128(p.cast(), _mm_xor_si128(xi_a, z_a));
    let z_b = keystream_regs(s1, s2, s3, s4, s5);
    let xi_b = _mm_loadu_si128(p.add(16).cast());
    update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi_b);
    _mm_storeu_si128(p.add(16).cast(), _mm_xor_si128(xi_b, z_b));
    let z_c = keystream_regs(s1, s2, s3, s4, s5);
    let xi_c = _mm_loadu_si128(p.add(32).cast());
    update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi_c);
    _mm_storeu_si128(p.add(32).cast(), _mm_xor_si128(xi_c, z_c));
    let z_d = keystream_regs(s1, s2, s3, s4, s5);
    let xi_d = _mm_loadu_si128(p.add(48).cast());
    update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi_d);
    _mm_storeu_si128(p.add(48).cast(), _mm_xor_si128(xi_d, z_d));
    p = p.add(64);
  }
  let mut remaining = msg_len.strict_sub(n_quads.strict_mul(64));
  if remaining >= 32 {
    let z_a = keystream_regs(s1, s2, s3, s4, s5);
    let xi_a = _mm_loadu_si128(p.cast());
    update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi_a);
    _mm_storeu_si128(p.cast(), _mm_xor_si128(xi_a, z_a));
    let z_b = keystream_regs(s1, s2, s3, s4, s5);
    let xi_b = _mm_loadu_si128(p.add(16).cast());
    update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi_b);
    _mm_storeu_si128(p.add(16).cast(), _mm_xor_si128(xi_b, z_b));
    p = p.add(32);
    remaining = remaining.strict_sub(32);
  }
  if remaining >= 16 {
    let z = keystream_regs(s1, s2, s3, s4, s5);
    let xi = _mm_loadu_si128(p.cast());
    update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi);
    _mm_storeu_si128(p.cast(), _mm_xor_si128(xi, z));
    remaining = remaining.strict_sub(16);
  }
  if remaining > 0 {
    let z = keystream_regs(s1, s2, s3, s4, s5);
    let tail_off = msg_len.strict_sub(remaining);
    let mut pad = [0u8; BLOCK_SIZE];
    pad[..remaining].copy_from_slice(&buffer[tail_off..]);
    let xi = load(&pad);
    update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi);
    let mut ct_bytes = [0u8; BLOCK_SIZE];
    store(_mm_xor_si128(xi, z), &mut ct_bytes);
    buffer[tail_off..].copy_from_slice(&ct_bytes[..remaining]);
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
/// # Safety
///
/// Caller must ensure the CPU supports `aes` and `avx`.
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
  //
  // Same counted-loop strategy as encrypt: raw pointer advancement with
  // compile-time-constant offsets, zero overflow checks in the hot path.
  let ct_len = buffer.len();
  let ptr = buffer.as_mut_ptr();
  let n_quads = ct_len / 64;
  let mut p = ptr;
  for _ in 0..n_quads {
    _mm_prefetch(p.add(256).cast::<i8>(), _MM_HINT_T0);
    let z_a = keystream_regs(s1, s2, s3, s4, s5);
    let ci_a = _mm_loadu_si128(p.cast());
    let xi_a = _mm_xor_si128(ci_a, z_a);
    update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi_a);
    _mm_storeu_si128(p.cast(), xi_a);
    let z_b = keystream_regs(s1, s2, s3, s4, s5);
    let ci_b = _mm_loadu_si128(p.add(16).cast());
    let xi_b = _mm_xor_si128(ci_b, z_b);
    update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi_b);
    _mm_storeu_si128(p.add(16).cast(), xi_b);
    let z_c = keystream_regs(s1, s2, s3, s4, s5);
    let ci_c = _mm_loadu_si128(p.add(32).cast());
    let xi_c = _mm_xor_si128(ci_c, z_c);
    update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi_c);
    _mm_storeu_si128(p.add(32).cast(), xi_c);
    let z_d = keystream_regs(s1, s2, s3, s4, s5);
    let ci_d = _mm_loadu_si128(p.add(48).cast());
    let xi_d = _mm_xor_si128(ci_d, z_d);
    update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi_d);
    _mm_storeu_si128(p.add(48).cast(), xi_d);
    p = p.add(64);
  }
  let mut remaining = ct_len.strict_sub(n_quads.strict_mul(64));
  if remaining >= 32 {
    let z_a = keystream_regs(s1, s2, s3, s4, s5);
    let ci_a = _mm_loadu_si128(p.cast());
    let xi_a = _mm_xor_si128(ci_a, z_a);
    update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi_a);
    _mm_storeu_si128(p.cast(), xi_a);
    let z_b = keystream_regs(s1, s2, s3, s4, s5);
    let ci_b = _mm_loadu_si128(p.add(16).cast());
    let xi_b = _mm_xor_si128(ci_b, z_b);
    update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi_b);
    _mm_storeu_si128(p.add(16).cast(), xi_b);
    p = p.add(32);
    remaining = remaining.strict_sub(32);
  }
  if remaining >= 16 {
    let z = keystream_regs(s1, s2, s3, s4, s5);
    let ci = _mm_loadu_si128(p.cast());
    let xi = _mm_xor_si128(ci, z);
    update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi);
    _mm_storeu_si128(p.cast(), xi);
    remaining = remaining.strict_sub(16);
  }
  if remaining > 0 {
    let z = keystream_regs(s1, s2, s3, s4, s5);
    let tail_off = ct_len.strict_sub(remaining);
    let mut pad = [0u8; BLOCK_SIZE];
    pad[..remaining].copy_from_slice(&buffer[tail_off..]);
    let mut z_bytes = [0u8; BLOCK_SIZE];
    store(z, &mut z_bytes);
    let mut pt_pad = [0u8; BLOCK_SIZE];
    for i in 0..remaining {
      pt_pad[i] = pad[i] ^ z_bytes[i];
    }
    update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, load(&pt_pad));
    buffer[tail_off..].copy_from_slice(&pt_pad[..remaining]);
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
