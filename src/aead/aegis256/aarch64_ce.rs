use core::arch::aarch64::*;

use super::{BLOCK_SIZE, C0, C1, KEY_SIZE, NONCE_SIZE, TAG_SIZE};

#[inline]
/// # Safety
/// Caller must ensure `neon` support is available.
unsafe fn load(bytes: &[u8; BLOCK_SIZE]) -> uint8x16_t {
  // SAFETY: the caller guarantees NEON; `bytes` provides 16 initialized bytes for this unaligned load.
  unsafe { vld1q_u8(bytes.as_ptr()) }
}

#[inline]
/// # Safety
/// Caller must ensure `neon` support is available.
unsafe fn store(v: uint8x16_t, out: &mut [u8; BLOCK_SIZE]) {
  // SAFETY: the caller guarantees NEON; `out` provides exclusive access to 16 bytes for this unaligned store.
  unsafe { vst1q_u8(out.as_mut_ptr(), v) };
}

#[inline(always)]
fn prefetch_read_l1(ptr: *const u8) {
  // SAFETY: `prfm` is an AArch64 cache hint that does not dereference `ptr` or alter architectural state.
  unsafe {
    core::arch::asm!("prfm pldl1keep, [{ptr}]", ptr = in(reg) ptr, options(nostack, preserves_flags));
  }
}

// ── Register-based helpers ──────────────────────────────────────────────
//
// ARM AESE applies AddRoundKey before SubBytes, so both operand forms below
// compute the same AESENC-compatible round. Apple targets use the state as
// AESE's destructive data operand; non-Apple targets use the libaegis operand
// form, preserving old state registers for the next pipeline assignments.

#[target_feature(enable = "aes,neon")]
#[inline]
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
) {
  let zero = vdupq_n_u8(0);
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
/// Caller must ensure `neon` support is available.
unsafe fn keystream_regs(s1: uint8x16_t, s2: uint8x16_t, s3: uint8x16_t, s4: uint8x16_t, s5: uint8x16_t) -> uint8x16_t {
  // SAFETY: the caller guarantees NEON support for these register-only operations.
  unsafe { veorq_u8(veorq_u8(s1, s4), veorq_u8(s5, vandq_u8(s2, s3))) }
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
  // SAFETY: the entrypoint enables NEON; `kh0` is a complete initialized block.
  let k0 = unsafe { load(kh0) };
  // SAFETY: the entrypoint enables NEON; `kh1` is a complete initialized block.
  let k1 = unsafe { load(kh1) };
  // SAFETY: the entrypoint enables NEON; `nh0` is a complete initialized block.
  let n0 = unsafe { load(nh0) };
  // SAFETY: the entrypoint enables NEON; `nh1` is a complete initialized block.
  let n1 = unsafe { load(nh1) };
  // SAFETY: the entrypoint enables NEON; `C0` is a complete initialized block.
  let c0 = unsafe { load(&C0) };
  // SAFETY: the entrypoint enables NEON; `C1` is a complete initialized block.
  let c1 = unsafe { load(&C1) };
  let k0_xor_n0 = veorq_u8(k0, n0);
  let k1_xor_n1 = veorq_u8(k1, n1);
  let (mut s0, mut s1, mut s2, mut s3, mut s4, mut s5) =
    (k0_xor_n0, k1_xor_n1, c1, c0, veorq_u8(k0, c0), veorq_u8(k1, c1));
  for _ in 0..4 {
    // SAFETY: the entrypoint establishes AES and NEON for these valid state registers.
    unsafe {
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, k0);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, k1);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, k0_xor_n0);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, k1_xor_n1);
    }
  }
  let mut offset = 0usize;
  while offset.strict_add(BLOCK_SIZE) <= aad.len() {
    // SAFETY: the loop bound provides a complete readable block at `offset`; the entrypoint enables AES and NEON.
    unsafe {
      update_regs(
        &mut s0,
        &mut s1,
        &mut s2,
        &mut s3,
        &mut s4,
        &mut s5,
        vld1q_u8(aad.as_ptr().add(offset)),
      );
    }
    offset = offset.strict_add(BLOCK_SIZE);
  }
  if offset < aad.len() {
    let mut pad = [0u8; BLOCK_SIZE];
    pad[..aad.len().strict_sub(offset)].copy_from_slice(&aad[offset..]);
    // SAFETY: `pad` is a complete initialized block, and the entrypoint enables AES and NEON.
    unsafe {
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, load(&pad));
    }
  }
  let msg_len = buffer.len();
  let ptr = buffer.as_mut_ptr();
  let len = buffer.len();
  offset = 0;
  let four_blocks = BLOCK_SIZE.strict_mul(4);
  let two_blocks = BLOCK_SIZE.strict_mul(2);
  while offset.strict_add(four_blocks) <= len {
    prefetch_read_l1(ptr.wrapping_add(offset.strict_add(192)));
    // SAFETY: the loop bound places all four 16-byte lanes inside the exclusively borrowed buffer. The entrypoint
    // enables AES and NEON, and each plaintext lane is loaded before its ciphertext overwrites the same bytes.
    unsafe {
      let z_a = keystream_regs(s1, s2, s3, s4, s5);
      let xi_a = vld1q_u8(ptr.add(offset));
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi_a);
      vst1q_u8(ptr.add(offset), veorq_u8(xi_a, z_a));
      let z_b = keystream_regs(s1, s2, s3, s4, s5);
      let xi_b = vld1q_u8(ptr.add(offset.strict_add(BLOCK_SIZE)));
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi_b);
      vst1q_u8(ptr.add(offset.strict_add(BLOCK_SIZE)), veorq_u8(xi_b, z_b));
      let z_c = keystream_regs(s1, s2, s3, s4, s5);
      let xi_c = vld1q_u8(ptr.add(offset.strict_add(two_blocks)));
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi_c);
      vst1q_u8(ptr.add(offset.strict_add(two_blocks)), veorq_u8(xi_c, z_c));
      let z_d = keystream_regs(s1, s2, s3, s4, s5);
      let xi_d = vld1q_u8(ptr.add(offset.strict_add(two_blocks.strict_add(BLOCK_SIZE))));
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi_d);
      vst1q_u8(
        ptr.add(offset.strict_add(two_blocks.strict_add(BLOCK_SIZE))),
        veorq_u8(xi_d, z_d),
      );
    }
    offset = offset.strict_add(four_blocks);
  }
  if offset.strict_add(two_blocks) <= len {
    // SAFETY: the branch bound places both lanes inside the exclusively borrowed buffer; AES and NEON are enabled.
    unsafe {
      let z_a = keystream_regs(s1, s2, s3, s4, s5);
      let xi_a = vld1q_u8(ptr.add(offset));
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi_a);
      vst1q_u8(ptr.add(offset), veorq_u8(xi_a, z_a));
      let z_b = keystream_regs(s1, s2, s3, s4, s5);
      let xi_b = vld1q_u8(ptr.add(offset.strict_add(BLOCK_SIZE)));
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi_b);
      vst1q_u8(ptr.add(offset.strict_add(BLOCK_SIZE)), veorq_u8(xi_b, z_b));
    }
    offset = offset.strict_add(two_blocks);
  }
  if offset.strict_add(BLOCK_SIZE) <= len {
    // SAFETY: the branch bound provides one complete writable lane in the buffer; AES and NEON are enabled.
    unsafe {
      let z = keystream_regs(s1, s2, s3, s4, s5);
      let xi = vld1q_u8(ptr.add(offset));
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi);
      vst1q_u8(ptr.add(offset), veorq_u8(xi, z));
    }
    offset = offset.strict_add(BLOCK_SIZE);
  }
  if offset < len {
    // SAFETY: the entrypoint enables NEON for the valid state registers.
    let z = unsafe { keystream_regs(s1, s2, s3, s4, s5) };
    let tail_len = len.strict_sub(offset);
    let mut pad = [0u8; BLOCK_SIZE];
    pad[..tail_len].copy_from_slice(&buffer[offset..]);
    // SAFETY: `pad` is a complete initialized block, and the entrypoint enables NEON.
    let xi = unsafe { load(&pad) };
    // SAFETY: the entrypoint establishes AES and NEON for these valid state registers.
    unsafe {
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi);
    }
    let mut ct_bytes = [0u8; BLOCK_SIZE];
    // SAFETY: `ct_bytes` is a complete writable block, and the entrypoint enables NEON.
    unsafe { store(veorq_u8(xi, z), &mut ct_bytes) };
    buffer[offset..].copy_from_slice(&ct_bytes[..tail_len]);
  }
  let ad_bits = (aad.len() as u64).strict_mul(8);
  let msg_bits = (msg_len as u64).strict_mul(8);
  let mut len_bytes = [0u8; BLOCK_SIZE];
  len_bytes[..8].copy_from_slice(&ad_bits.to_le_bytes());
  len_bytes[8..].copy_from_slice(&msg_bits.to_le_bytes());
  // SAFETY: `len_bytes` is a complete initialized block, and the entrypoint enables NEON.
  let t = unsafe { veorq_u8(s3, load(&len_bytes)) };
  for _ in 0..7 {
    // SAFETY: the entrypoint establishes AES and NEON for these valid state registers.
    unsafe {
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, t);
    }
  }
  let tag_vec = veorq_u8(veorq_u8(veorq_u8(s0, s1), veorq_u8(s2, s3)), veorq_u8(s4, s5));
  let mut tag = [0u8; TAG_SIZE];
  // SAFETY: `tag` is a complete writable block, and the entrypoint enables NEON.
  unsafe { store(tag_vec, &mut tag) };
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
  // SAFETY: the entrypoint enables NEON; `kh0` is a complete initialized block.
  let k0 = unsafe { load(kh0) };
  // SAFETY: the entrypoint enables NEON; `kh1` is a complete initialized block.
  let k1 = unsafe { load(kh1) };
  // SAFETY: the entrypoint enables NEON; `nh0` is a complete initialized block.
  let n0 = unsafe { load(nh0) };
  // SAFETY: the entrypoint enables NEON; `nh1` is a complete initialized block.
  let n1 = unsafe { load(nh1) };
  // SAFETY: the entrypoint enables NEON; `C0` is a complete initialized block.
  let c0 = unsafe { load(&C0) };
  // SAFETY: the entrypoint enables NEON; `C1` is a complete initialized block.
  let c1 = unsafe { load(&C1) };
  let k0_xor_n0 = veorq_u8(k0, n0);
  let k1_xor_n1 = veorq_u8(k1, n1);
  let (mut s0, mut s1, mut s2, mut s3, mut s4, mut s5) =
    (k0_xor_n0, k1_xor_n1, c1, c0, veorq_u8(k0, c0), veorq_u8(k1, c1));
  for _ in 0..4 {
    // SAFETY: the entrypoint establishes AES and NEON for these valid state registers.
    unsafe {
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, k0);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, k1);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, k0_xor_n0);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, k1_xor_n1);
    }
  }
  let mut offset = 0usize;
  while offset.strict_add(BLOCK_SIZE) <= aad.len() {
    // SAFETY: the loop bound provides a complete readable block at `offset`; the entrypoint enables AES and NEON.
    unsafe {
      update_regs(
        &mut s0,
        &mut s1,
        &mut s2,
        &mut s3,
        &mut s4,
        &mut s5,
        vld1q_u8(aad.as_ptr().add(offset)),
      );
    }
    offset = offset.strict_add(BLOCK_SIZE);
  }
  if offset < aad.len() {
    let mut pad = [0u8; BLOCK_SIZE];
    pad[..aad.len().strict_sub(offset)].copy_from_slice(&aad[offset..]);
    // SAFETY: `pad` is a complete initialized block, and the entrypoint enables AES and NEON.
    unsafe {
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, load(&pad));
    }
  }
  let ct_len = buffer.len();
  let ptr = buffer.as_mut_ptr();
  let len = buffer.len();
  offset = 0;
  let four_blocks = BLOCK_SIZE.strict_mul(4);
  let two_blocks = BLOCK_SIZE.strict_mul(2);
  while offset.strict_add(four_blocks) <= len {
    prefetch_read_l1(ptr.wrapping_add(offset.strict_add(192)));
    // SAFETY: the loop bound places all four 16-byte lanes inside the exclusively borrowed buffer. The entrypoint
    // enables AES and NEON, and each ciphertext lane is loaded before its plaintext overwrites the same bytes.
    unsafe {
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
      let z_c = keystream_regs(s1, s2, s3, s4, s5);
      let ci_c = vld1q_u8(ptr.add(offset.strict_add(two_blocks)));
      let xi_c = veorq_u8(ci_c, z_c);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi_c);
      vst1q_u8(ptr.add(offset.strict_add(two_blocks)), xi_c);
      let z_d = keystream_regs(s1, s2, s3, s4, s5);
      let ci_d = vld1q_u8(ptr.add(offset.strict_add(two_blocks.strict_add(BLOCK_SIZE))));
      let xi_d = veorq_u8(ci_d, z_d);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi_d);
      vst1q_u8(ptr.add(offset.strict_add(two_blocks.strict_add(BLOCK_SIZE))), xi_d);
    }
    offset = offset.strict_add(four_blocks);
  }
  if offset.strict_add(two_blocks) <= len {
    // SAFETY: the branch bound places both lanes inside the exclusively borrowed buffer; AES and NEON are enabled.
    unsafe {
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
    }
    offset = offset.strict_add(two_blocks);
  }
  if offset.strict_add(BLOCK_SIZE) <= len {
    // SAFETY: the branch bound provides one complete writable lane in the buffer; AES and NEON are enabled.
    unsafe {
      let z = keystream_regs(s1, s2, s3, s4, s5);
      let ci = vld1q_u8(ptr.add(offset));
      let xi = veorq_u8(ci, z);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi);
      vst1q_u8(ptr.add(offset), xi);
    }
    offset = offset.strict_add(BLOCK_SIZE);
  }
  if offset < len {
    // SAFETY: the entrypoint enables NEON for the valid state registers.
    let z = unsafe { keystream_regs(s1, s2, s3, s4, s5) };
    let tail_len = len.strict_sub(offset);
    let mut pad = [0u8; BLOCK_SIZE];
    pad[..tail_len].copy_from_slice(&buffer[offset..]);
    let mut z_bytes = [0u8; BLOCK_SIZE];
    // SAFETY: `z_bytes` is a complete writable block, and the entrypoint enables NEON.
    unsafe { store(z, &mut z_bytes) };
    let mut pt_pad = [0u8; BLOCK_SIZE];
    for i in 0..tail_len {
      pt_pad[i] = pad[i] ^ z_bytes[i];
    }
    // SAFETY: `pt_pad` is a complete initialized block, and the entrypoint enables AES and NEON.
    unsafe {
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, load(&pt_pad));
    }
    buffer[offset..].copy_from_slice(&pt_pad[..tail_len]);
  }
  let ad_bits = (aad.len() as u64).strict_mul(8);
  let ct_bits = (ct_len as u64).strict_mul(8);
  let mut len_bytes = [0u8; BLOCK_SIZE];
  len_bytes[..8].copy_from_slice(&ad_bits.to_le_bytes());
  len_bytes[8..].copy_from_slice(&ct_bits.to_le_bytes());
  // SAFETY: `len_bytes` is a complete initialized block, and the entrypoint enables NEON.
  let t = unsafe { veorq_u8(s3, load(&len_bytes)) };
  for _ in 0..7 {
    // SAFETY: the entrypoint establishes AES and NEON for these valid state registers.
    unsafe {
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, t);
    }
  }
  let tag_vec = veorq_u8(veorq_u8(veorq_u8(s0, s1), veorq_u8(s2, s3)), veorq_u8(s4, s5));
  let mut tag = [0u8; TAG_SIZE];
  // SAFETY: `tag` is a complete writable block, and the entrypoint enables NEON.
  unsafe { store(tag_vec, &mut tag) };
  tag
}
