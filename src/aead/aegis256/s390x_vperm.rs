use core::{arch::asm, simd::i64x2};

use super::{BLOCK_SIZE, C0, C1, KEY_SIZE, NONCE_SIZE, TAG_SIZE};
use crate::aead::aes_round::{
  AES_AFFINE, MC_ROT1, MC_ROT2, NIBBLE_MASK, VPERM_INV_HI, VPERM_INV_LO, VPERM_IPT_HI, VPERM_IPT_LO, VPERM_SBOT,
  VPERM_SBOU, VPERM_SR, XTIME_REDUCE,
};

// ── Vector register helpers ──────────────────────────────────────────────

/// Load a 16-byte block into an i64x2 vector register (big-endian native).
#[inline(always)]
fn load_be(bytes: &[u8; 16]) -> i64x2 {
  // s390x is big-endian: vector byte 0 = bytes[0] (MSB).
  let hi = i64::from_ne_bytes([
    bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
  ]);
  let lo = i64::from_ne_bytes([
    bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
  ]);
  i64x2::from_array([hi, lo])
}

/// Store an i64x2 back to a 16-byte block (big-endian native).
#[inline(always)]
fn store_be(v: i64x2, out: &mut [u8; 16]) {
  let arr = v.to_array();
  let hi = arr[0].to_ne_bytes();
  let lo = arr[1].to_ne_bytes();
  out[0..8].copy_from_slice(&hi);
  out[8..16].copy_from_slice(&lo);
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

/// Broadcast a byte to all 16 positions of a vector.
#[inline(always)]
fn splat_byte(b: u8) -> i64x2 {
  let w = u64::from_ne_bytes([b; 8]) as i64;
  i64x2::from_array([w, w])
}

// ── z/Vector inline asm primitives ─────────────────────────────────────

/// VPERM: byte permute — `out[i] = (v2||v3)[control[i] & 0x1F]`.
///
/// For 16-entry table lookups: pass `table` as both v2 and v3 (indices
/// 0-15 select from v2; indices 16-31 would select from v3 but never
/// occur when control bytes are 0-15).
///
/// # Safety
/// Requires the s390x vector facility (z13+).
#[target_feature(enable = "vector")]
#[inline]
unsafe fn vperm(v2: i64x2, v3: i64x2, control: i64x2) -> i64x2 {
  // SAFETY: Caller guarantees the s390x vector facility is available.
  // VPERM operates on pure register values with no memory access.
  unsafe {
    let out: i64x2;
    asm!(
      "vperm {out}, {v2}, {v3}, {ctl}",
      out = lateout(vreg) out,
      v2 = in(vreg) v2,
      v3 = in(vreg) v3,
      ctl = in(vreg) control,
      options(nomem, nostack, pure),
    );
    out
  }
}

/// VNC: AND with complement — `out = a & ~b`.
///
/// # Safety
/// Requires the s390x vector facility.
#[target_feature(enable = "vector")]
#[inline]
unsafe fn vnc(a: i64x2, b: i64x2) -> i64x2 {
  // SAFETY: caller guarantees the s390x vector facility is available.
  unsafe {
    let out: i64x2;
    asm!(
      "vnc {out}, {a}, {b}",
      out = lateout(vreg) out,
      a = in(vreg) a,
      b = in(vreg) b,
      options(nomem, nostack, pure),
    );
    out
  }
}

/// VESRLB: shift each byte element right (logical) by `count` bits.
///
/// # Safety
/// Requires the s390x vector facility.
#[target_feature(enable = "vector")]
#[inline]
unsafe fn vesrlb_4(v: i64x2) -> i64x2 {
  // SAFETY: caller guarantees the s390x vector facility is available.
  unsafe {
    let out: i64x2;
    asm!(
      "vesrlb {out}, {v}, 4",
      out = lateout(vreg) out,
      v = in(vreg) v,
      options(nomem, nostack, pure),
    );
    out
  }
}

/// VESRAB: arithmetic shift each byte right by 7 — produces 0xFF for
/// bytes with bit 7 set, 0x00 otherwise.
///
/// # Safety
/// Requires the s390x vector facility.
#[target_feature(enable = "vector")]
#[inline]
unsafe fn vesrab_7(v: i64x2) -> i64x2 {
  // SAFETY: caller guarantees the s390x vector facility is available.
  unsafe {
    let out: i64x2;
    asm!(
      "vesrab {out}, {v}, 7",
      out = lateout(vreg) out,
      v = in(vreg) v,
      options(nomem, nostack, pure),
    );
    out
  }
}

/// VESLB: shift each byte element left (logical) by 1 bit.
///
/// # Safety
/// Requires the s390x vector facility.
#[target_feature(enable = "vector")]
#[inline]
unsafe fn veslb_1(v: i64x2) -> i64x2 {
  // SAFETY: caller guarantees the s390x vector facility is available.
  unsafe {
    let out: i64x2;
    asm!(
      "veslb {out}, {v}, 1",
      out = lateout(vreg) out,
      v = in(vreg) v,
      options(nomem, nostack, pure),
    );
    out
  }
}

// ── Hamburg vperm S-box + ShiftRows + MixColumns + AddRoundKey ─────────

/// VPERM table lookup with PSHUFB zeroing emulation.
///
/// Returns `table[idx & 0x0F]` for each byte, but zeros the result byte
/// when bit 7 of the index is set (the "infinity" sentinel from the
/// GF(2^4) inverse). Uses VPERM + VESRAB + VNC.
#[target_feature(enable = "vector")]
#[inline]
/// # Safety
///
/// Caller must ensure the s390x vector facility is available.
unsafe fn vperm_z(table: i64x2, indices: i64x2, mask_0f: i64x2) -> i64x2 {
  // SAFETY: caller guarantees the s390x vector facility is available for
  // this helper; all invoked primitives operate only on register values.
  unsafe {
    let idx4 = and_vec(indices, mask_0f);
    let result = vperm(table, table, idx4);
    let bit7 = vesrab_7(indices);
    vnc(result, bit7)
  }
}

/// Single AES round via Hamburg vperm: SubBytes + ShiftRows + MixColumns +
/// AddRoundKey. Constant-time — all register-to-register, no table lookups.
///
/// # Safety
/// Requires the s390x vector facility (z13+).
#[target_feature(enable = "vector")]
#[inline]
unsafe fn aes_round(state: i64x2, round_key: i64x2, tables: &VpermTables) -> i64x2 {
  // SAFETY: caller guarantees the s390x vector facility is available for
  // this helper; all invoked primitives operate only on register values.
  unsafe {
    // ── Phase 1: Nibble extraction ──
    let lo_nib = and_vec(state, tables.mask_0f);
    let hi_nib = vesrlb_4(state);

    // ── Phase 2: Input transform (AES basis → tower field) ──
    let ipt_l = vperm(tables.ipt_lo, tables.ipt_lo, lo_nib);
    let ipt_h = vperm(tables.ipt_hi, tables.ipt_hi, hi_nib);
    let x = xor_vec(ipt_l, ipt_h);

    // ── Phase 3: Re-extract nibbles of transformed value ──
    let t_lo = and_vec(x, tables.mask_0f);
    let t_hi = vesrlb_4(x);

    // ── Phase 4: GF(2^4) inverse (5 plain VPERM + 4 zeroing VPERM + XORs) ──
    let ak = vperm(tables.inv_hi, tables.inv_hi, t_lo);
    let j = xor_vec(t_hi, t_lo);
    let inv_i = vperm(tables.inv_lo, tables.inv_lo, t_hi);
    let iak = xor_vec(inv_i, ak);
    let inv_j = vperm(tables.inv_lo, tables.inv_lo, j);
    let jak = xor_vec(inv_j, ak);

    // Zeroing lookups: sentinel 0x80 in T_INV_LO maps to "infinity" in GF(2^4).
    // When bit 7 is set in the index, the result must be 0 (not table[0]).
    let inv_iak = vperm_z(tables.inv_lo, iak, tables.mask_0f);
    let io = xor_vec(inv_iak, j);
    let inv_jak = vperm_z(tables.inv_lo, jak, tables.mask_0f);
    let jo = xor_vec(inv_jak, t_hi);

    // ── Phase 5: Output transform (SubBytes output, no MixColumns) ──
    // io/jo can have bit 7 set from the sentinel propagation.
    let su = vperm_z(tables.sbou, io, tables.mask_0f);
    let st = vperm_z(tables.sbot, jo, tables.mask_0f);
    let sb = xor_vec(xor_vec(su, st), tables.affine_63);

    // ── ShiftRows ──
    let sr = vperm(sb, sb, tables.sr_perm);

    // ── MixColumns (xtime decomposition) ──
    // rot1[i] = sr[(i+1) mod 4 within each column]
    let rot1 = vperm(sr, sr, tables.mc_rot1);
    // pair = sr ^ rot1 = [b0^b1, b1^b2, b2^b3, b3^b0] per column
    let pair = xor_vec(sr, rot1);
    // xtime(pair): (pair<<1) ^ (((pair>>7)&1) * 0x1B)
    let xt_shifted = veslb_1(pair);
    let xt_mask = and_vec(vesrab_7(pair), tables.xtime_1b);
    let xt = xor_vec(xt_shifted, xt_mask);
    // rot2_pair[i] = pair[(i+2) mod 4 within each column]
    let rot2_pair = vperm(pair, pair, tables.mc_rot2);
    // col_sum = pair ^ rot2(pair) = [b0^b1^b2^b3, ...] in every position
    let col_sum = xor_vec(pair, rot2_pair);
    // result = sr ^ col_sum ^ xtime(pair)
    let mc = xor_vec(xor_vec(sr, col_sum), xt);

    // ── AddRoundKey ──
    xor_vec(mc, round_key)
  }
}

/// Preloaded Hamburg vperm constant vectors.
///
/// Loaded once at the start of encrypt/decrypt and passed to every
/// `aes_round` call to avoid redundant memory loads.
struct VpermTables {
  ipt_lo: i64x2,
  ipt_hi: i64x2,
  inv_lo: i64x2,
  inv_hi: i64x2,
  sbou: i64x2,
  sbot: i64x2,
  sr_perm: i64x2,
  mc_rot1: i64x2,
  mc_rot2: i64x2,
  mask_0f: i64x2,
  affine_63: i64x2,
  xtime_1b: i64x2,
}

impl VpermTables {
  #[inline(always)]
  fn load() -> Self {
    Self {
      ipt_lo: load_be(&VPERM_IPT_LO),
      ipt_hi: load_be(&VPERM_IPT_HI),
      inv_lo: load_be(&VPERM_INV_LO),
      inv_hi: load_be(&VPERM_INV_HI),
      sbou: load_be(&VPERM_SBOU),
      sbot: load_be(&VPERM_SBOT),
      sr_perm: load_be(&VPERM_SR),
      mc_rot1: load_be(&MC_ROT1),
      mc_rot2: load_be(&MC_ROT2),
      mask_0f: splat_byte(NIBBLE_MASK),
      affine_63: splat_byte(AES_AFFINE),
      xtime_1b: splat_byte(XTIME_REDUCE),
    }
  }
}

// ── AEGIS-256 state operations ──────────────────────────────────────────

#[target_feature(enable = "vector")]
#[allow(clippy::too_many_arguments)]
#[inline]
/// # Safety
///
/// Caller must ensure the s390x vector facility is available and all state
/// registers belong to a valid AEGIS-256 state.
unsafe fn update_regs(
  s0: &mut i64x2,
  s1: &mut i64x2,
  s2: &mut i64x2,
  s3: &mut i64x2,
  s4: &mut i64x2,
  s5: &mut i64x2,
  m: i64x2,
  tables: &VpermTables,
) {
  // SAFETY: caller guarantees the s390x vector facility is available for
  // this helper and all state registers are valid local values.
  unsafe {
    let tmp = *s5;
    *s5 = aes_round(*s4, *s5, tables);
    *s4 = aes_round(*s3, *s4, tables);
    *s3 = aes_round(*s2, *s3, tables);
    *s2 = aes_round(*s1, *s2, tables);
    *s1 = aes_round(*s0, *s1, tables);
    *s0 = xor_vec(aes_round(tmp, *s0, tables), m);
  }
}

#[inline(always)]
fn keystream_regs(s1: i64x2, s2: i64x2, s3: i64x2, s4: i64x2, s5: i64x2) -> i64x2 {
  xor_vec(xor_vec(s1, s4), xor_vec(s5, and_vec(s2, s3)))
}

// ── Fused encrypt/decrypt ─────────────────────────────────────────────

#[target_feature(enable = "vector")]
/// # Safety
///
/// Caller must ensure the s390x vector facility is available.
pub(super) unsafe fn encrypt_fused(
  key: &[u8; KEY_SIZE],
  nonce: &[u8; NONCE_SIZE],
  aad: &[u8],
  buffer: &mut [u8],
) -> [u8; TAG_SIZE] {
  // SAFETY: caller guarantees the s390x vector facility is available for
  // the full fused operation and all references are valid Rust buffers.
  unsafe {
    let tables = VpermTables::load();
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
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, k0, &tables);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, k1, &tables);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, k0_xor_n0, &tables);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, k1_xor_n1, &tables);
    }
    let mut offset = 0usize;
    while offset.strict_add(BLOCK_SIZE) <= aad.len() {
      let mut tmp = [0u8; 16];
      tmp.copy_from_slice(&aad[offset..offset.strict_add(BLOCK_SIZE)]);
      update_regs(
        &mut s0,
        &mut s1,
        &mut s2,
        &mut s3,
        &mut s4,
        &mut s5,
        load_be(&tmp),
        &tables,
      );
      offset = offset.strict_add(BLOCK_SIZE);
    }
    if offset < aad.len() {
      let mut pad = [0u8; BLOCK_SIZE];
      pad[..aad.len().strict_sub(offset)].copy_from_slice(&aad[offset..]);
      update_regs(
        &mut s0,
        &mut s1,
        &mut s2,
        &mut s3,
        &mut s4,
        &mut s5,
        load_be(&pad),
        &tables,
      );
    }
    let msg_len = buffer.len();
    let len = buffer.len();
    offset = 0;
    while offset.strict_add(BLOCK_SIZE) <= len {
      let z = keystream_regs(s1, s2, s3, s4, s5);
      let mut tmp = [0u8; 16];
      tmp.copy_from_slice(&buffer[offset..offset.strict_add(BLOCK_SIZE)]);
      let xi = load_be(&tmp);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi, &tables);
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
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi, &tables);
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
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, t, &tables);
    }
    let tag_vec = xor_vec(xor_vec(xor_vec(s0, s1), xor_vec(s2, s3)), xor_vec(s4, s5));
    let mut tag = [0u8; TAG_SIZE];
    store_be(tag_vec, &mut tag);
    tag
  }
}

#[target_feature(enable = "vector")]
/// # Safety
///
/// Caller must ensure the s390x vector facility is available.
pub(super) unsafe fn decrypt_fused(
  key: &[u8; KEY_SIZE],
  nonce: &[u8; NONCE_SIZE],
  aad: &[u8],
  buffer: &mut [u8],
) -> [u8; TAG_SIZE] {
  // SAFETY: caller guarantees the s390x vector facility is available for
  // the full fused operation and all references are valid Rust buffers.
  unsafe {
    let tables = VpermTables::load();
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
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, k0, &tables);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, k1, &tables);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, k0_xor_n0, &tables);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, k1_xor_n1, &tables);
    }
    let mut offset = 0usize;
    while offset.strict_add(BLOCK_SIZE) <= aad.len() {
      let mut tmp = [0u8; 16];
      tmp.copy_from_slice(&aad[offset..offset.strict_add(BLOCK_SIZE)]);
      update_regs(
        &mut s0,
        &mut s1,
        &mut s2,
        &mut s3,
        &mut s4,
        &mut s5,
        load_be(&tmp),
        &tables,
      );
      offset = offset.strict_add(BLOCK_SIZE);
    }
    if offset < aad.len() {
      let mut pad = [0u8; BLOCK_SIZE];
      pad[..aad.len().strict_sub(offset)].copy_from_slice(&aad[offset..]);
      update_regs(
        &mut s0,
        &mut s1,
        &mut s2,
        &mut s3,
        &mut s4,
        &mut s5,
        load_be(&pad),
        &tables,
      );
    }
    let ct_len = buffer.len();
    let len = buffer.len();
    offset = 0;
    while offset.strict_add(BLOCK_SIZE) <= len {
      let z = keystream_regs(s1, s2, s3, s4, s5);
      let mut tmp = [0u8; 16];
      tmp.copy_from_slice(&buffer[offset..offset.strict_add(BLOCK_SIZE)]);
      let xi = xor_vec(load_be(&tmp), z);
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, xi, &tables);
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
      update_regs(
        &mut s0,
        &mut s1,
        &mut s2,
        &mut s3,
        &mut s4,
        &mut s5,
        load_be(&pt_pad),
        &tables,
      );
      buffer[offset..].copy_from_slice(&pt_pad[..tail_len]);
    }
    let ad_bits = (aad.len() as u64).strict_mul(8);
    let ct_bits = (ct_len as u64).strict_mul(8);
    let mut len_bytes = [0u8; BLOCK_SIZE];
    len_bytes[..8].copy_from_slice(&ad_bits.to_le_bytes());
    len_bytes[8..].copy_from_slice(&ct_bits.to_le_bytes());
    let t = xor_vec(s3, load_be(&len_bytes));
    for _ in 0..7 {
      update_regs(&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, t, &tables);
    }
    let tag_vec = xor_vec(xor_vec(xor_vec(s0, s1), xor_vec(s2, s3)), xor_vec(s4, s5));
    let mut tag = [0u8; TAG_SIZE];
    store_be(tag_vec, &mut tag);
    tag
  }
}
