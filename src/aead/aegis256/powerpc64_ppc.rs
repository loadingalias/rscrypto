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
/// # Safety
///
/// Caller must ensure POWER8 vector crypto support is available.
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
/// # Safety
///
/// Caller must ensure POWER8 vector crypto support is available and all state
/// registers belong to a valid AEGIS-256 state.
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
/// # Safety
///
/// Caller must ensure POWER8 vector crypto support is available.
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
/// # Safety
///
/// Caller must ensure POWER8 vector crypto support is available.
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
