use core::arch::aarch64::*;

use super::{BLOCK_SIZE, C0, C1, KEY_SIZE, NONCE_SIZE, TAG_SIZE};

#[inline]
/// # Safety
///
/// `bytes` must refer to a valid 16-byte block.
unsafe fn load(bytes: &[u8; BLOCK_SIZE]) -> uint8x16_t {
  vld1q_u8(bytes.as_ptr())
}

#[inline]
/// # Safety
///
/// `out` must refer to a valid writable 16-byte block.
unsafe fn store(v: uint8x16_t, out: &mut [u8; BLOCK_SIZE]) {
  vst1q_u8(out.as_mut_ptr(), v);
}

// ── Register-based helpers ──────────────────────────────────────────────
//
// ARM AESE applies AddRoundKey before SubBytes, so both operand forms below
// compute the same AESENC-compatible round. Apple cores benchmark faster with
// the state as AESE's destructive data operand; non-Apple targets use the
// libaegis operand form, preserving old state registers for the next pipeline
// assignments on Neoverse-class cores.
//
// Neoverse V1/V2 has 2 crypto pipelines; register-based code gives the
// OOO engine maximum scheduling freedom across both pipes.

#[target_feature(enable = "aes,neon")]
#[inline]
#[allow(clippy::too_many_arguments)]
/// # Safety
///
/// Caller must ensure `aes` and `neon` support is available and all register
/// arguments come from a valid AEGIS-256 state.
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

  #[cfg(any(target_os = "macos", target_os = "ios", target_os = "tvos", target_os = "watchos"))]
  {
    *s5 = veorq_u8(vaesmcq_u8(vaeseq_u8(*s4, zero)), *s5);
    *s4 = veorq_u8(vaesmcq_u8(vaeseq_u8(*s3, zero)), *s4);
    *s3 = veorq_u8(vaesmcq_u8(vaeseq_u8(*s2, zero)), *s3);
    *s2 = veorq_u8(vaesmcq_u8(vaeseq_u8(*s1, zero)), *s2);
    *s1 = veorq_u8(vaesmcq_u8(vaeseq_u8(*s0, zero)), *s1);
    *s0 = veorq_u8(veorq_u8(vaesmcq_u8(vaeseq_u8(tmp, zero)), *s0), m);
  }

  #[cfg(not(any(target_os = "macos", target_os = "ios", target_os = "tvos", target_os = "watchos")))]
  {
    *s5 = veorq_u8(vaesmcq_u8(vaeseq_u8(zero, *s4)), *s5);
    *s4 = veorq_u8(vaesmcq_u8(vaeseq_u8(zero, *s3)), *s4);
    *s3 = veorq_u8(vaesmcq_u8(vaeseq_u8(zero, *s2)), *s3);
    *s2 = veorq_u8(vaesmcq_u8(vaeseq_u8(zero, *s1)), *s2);
    *s1 = veorq_u8(vaesmcq_u8(vaeseq_u8(zero, *s0)), *s1);
    *s0 = veorq_u8(veorq_u8(vaesmcq_u8(vaeseq_u8(zero, tmp)), *s0), m);
  }
}

#[inline]
/// # Safety
///
/// The provided registers must come from a valid AEGIS-256 state on an
/// `aes` + `neon` capable CPU.
unsafe fn keystream_regs(s1: uint8x16_t, s2: uint8x16_t, s3: uint8x16_t, s4: uint8x16_t, s5: uint8x16_t) -> uint8x16_t {
  veorq_u8(veorq_u8(s1, s4), veorq_u8(s5, vandq_u8(s2, s3)))
}

// ── Fused encrypt/decrypt ───────────────────────────────────────────────

#[target_feature(enable = "aes,neon")]
/// # Safety
///
/// Caller must ensure the CPU supports `aes` and `neon`.
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
/// # Safety
///
/// Caller must ensure the CPU supports `aes` and `neon`.
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
