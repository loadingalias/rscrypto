use core::arch::asm;

use super::{BLOCK_SIZE, Block, C0, C1, KEY_SIZE, NONCE_SIZE, TAG_SIZE, and_block, split_halves, xor_block};
use crate::aead::aes_round::{
  AES_AFFINE, MC_ROT1, MC_ROT2, VPERM_INV_HI, VPERM_INV_LO, VPERM_IPT_HI, VPERM_IPT_LO, VPERM_SBOT, VPERM_SBOU,
  VPERM_SR, XTIME_REDUCE,
};

type State = [Block; 6];

/// Precomputed Hamburg vperm table block — packed contiguously for
/// offset-based vector loads in the asm block.
#[repr(C, align(16))]
struct VpermTables {
  ipt_lo: [u8; 16],  // offset 0
  ipt_hi: [u8; 16],  // offset 16
  inv_lo: [u8; 16],  // offset 32
  inv_hi: [u8; 16],  // offset 48
  sbou: [u8; 16],    // offset 64
  sbot: [u8; 16],    // offset 80
  sr_perm: [u8; 16], // offset 96
  mc_rot1: [u8; 16], // offset 112
  mc_rot2: [u8; 16], // offset 128
  affine: [u8; 16],  // offset 144
  xtime: [u8; 16],   // offset 160
}

impl VpermTables {
  #[inline(always)]
  fn load() -> Self {
    Self {
      ipt_lo: VPERM_IPT_LO,
      ipt_hi: VPERM_IPT_HI,
      inv_lo: VPERM_INV_LO,
      inv_hi: VPERM_INV_HI,
      sbou: VPERM_SBOU,
      sbot: VPERM_SBOT,
      sr_perm: VPERM_SR,
      mc_rot1: MC_ROT1,
      mc_rot2: MC_ROT2,
      affine: [AES_AFFINE; 16],
      xtime: [XTIME_REDUCE; 16],
    }
  }
}

/// Single AES round via Hamburg vperm on RISC-V V: SubBytes + ShiftRows +
/// MixColumns + AddRoundKey. Uses `vrgather.vv` for all S-box nibble lookups.
///
/// # Safety
/// Requires the RISC-V V extension.
#[target_feature(enable = "v")]
#[inline]
unsafe fn aes_round(block: &Block, round_key: &Block, tables: &VpermTables) -> Block {
  let mut out = [0u8; BLOCK_SIZE];

  // SAFETY: Caller guarantees RISC-V V extension is available.
  // The asm block loads the state, tables, and round key from memory,
  // performs all computation in vector registers, and stores the result.
  // All vrgather indices are masked to 0-15 before lookup — no secret-
  // dependent memory access.
  unsafe {
    asm!(
      // ── Setup ──────────────────────────────────────────────────────
      "vsetivli zero, 16, e8, m1, ta, ma",

      // Load tables from the VpermTables struct (contiguous, 16B each).
      "vle8.v v2, ({tbl})",          // IPT_LO  (offset 0)
      "addi {tmp}, {tbl}, 16",
      "vle8.v v3, ({tmp})",          // IPT_HI  (offset 16)
      "addi {tmp}, {tbl}, 32",
      "vle8.v v4, ({tmp})",          // INV_LO  (offset 32)
      "addi {tmp}, {tbl}, 48",
      "vle8.v v5, ({tmp})",          // INV_HI  (offset 48)
      "addi {tmp}, {tbl}, 64",
      "vle8.v v6, ({tmp})",          // SBOU    (offset 64)
      "addi {tmp}, {tbl}, 80",
      "vle8.v v7, ({tmp})",          // SBOT    (offset 80)
      "addi {tmp}, {tbl}, 96",
      "vle8.v v8, ({tmp})",          // SR_PERM (offset 96)
      "addi {tmp}, {tbl}, 112",
      "vle8.v v9, ({tmp})",          // MC_ROT1 (offset 112)
      "addi {tmp}, {tbl}, 128",
      "vle8.v v10, ({tmp})",         // MC_ROT2 (offset 128)
      "addi {tmp}, {tbl}, 144",
      "vle8.v v11, ({tmp})",         // 0x63    (offset 144)
      "addi {tmp}, {tbl}, 160",
      "vle8.v v12, ({tmp})",         // 0x1B    (offset 160)

      // Load state and round key.
      "vle8.v v0, ({state})",
      "vle8.v v1, ({rk})",

      // ── Phase 1: Nibble extraction ─────────────────────────────────
      "vand.vi v14, v0, 15",         // lo_nib = state & 0x0F
      "vsrl.vi v15, v0, 4",          // hi_nib = state >> 4

      // ── Phase 2: Input transform (AES → tower field) ──────────────
      "vrgather.vv v16, v2, v14",    // ipt_l = IPT_LO[lo_nib]
      "vrgather.vv v17, v3, v15",    // ipt_h = IPT_HI[hi_nib]
      "vxor.vv v14, v16, v17",       // x = ipt_l ^ ipt_h

      // ── Phase 3: Re-extract nibbles of transformed value ───────────
      "vand.vi v15, v14, 15",        // t_lo = x & 0x0F
      "vsrl.vi v16, v14, 4",         // t_hi = x >> 4

      // ── Phase 4: GF(2^4) inverse ──────────────────────────────────
      "vrgather.vv v17, v5, v15",    // ak = INV_HI[t_lo]
      "vxor.vv v18, v16, v15",       // j = t_hi ^ t_lo
      "vrgather.vv v19, v4, v16",    // inv_i = INV_LO[t_hi]
      "vxor.vv v20, v19, v17",       // iak = inv_i ^ ak
      "vrgather.vv v21, v4, v18",    // inv_j = INV_LO[j]
      "vxor.vv v22, v21, v17",       // jak = inv_j ^ ak

      // vperm_z(INV_LO, iak): zero where bit 7 set
      "vand.vi v23, v20, 15",        // iak & 0x0F
      "vrgather.vv v24, v4, v23",    // INV_LO[iak & 0x0F]
      "vsra.vi v25, v20, 7",         // 0xFF where bit 7 set
      "vxor.vi v26, v25, -1",        // ~mask
      "vand.vv v24, v24, v26",       // zero masked positions
      "vxor.vv v14, v24, v18",       // io = inv_iak ^ j

      // vperm_z(INV_LO, jak): zero where bit 7 set
      "vand.vi v23, v22, 15",
      "vrgather.vv v24, v4, v23",
      "vsra.vi v25, v22, 7",
      "vxor.vi v26, v25, -1",
      "vand.vv v24, v24, v26",
      "vxor.vv v15, v24, v16",       // jo = inv_jak ^ t_hi

      // ── Phase 5: Output transform (SubBytes) ──────────────────────
      // vperm_z(SBOU, io)
      "vand.vi v23, v14, 15",
      "vrgather.vv v24, v6, v23",
      "vsra.vi v25, v14, 7",
      "vxor.vi v26, v25, -1",
      "vand.vv v16, v24, v26",       // su

      // vperm_z(SBOT, jo)
      "vand.vi v23, v15, 15",
      "vrgather.vv v24, v7, v23",
      "vsra.vi v25, v15, 7",
      "vxor.vi v26, v25, -1",
      "vand.vv v17, v24, v26",       // st

      // sb = su ^ st ^ 0x63
      "vxor.vv v14, v16, v17",
      "vxor.vv v14, v14, v11",

      // ── ShiftRows ─────────────────────────────────────────────────
      "vrgather.vv v15, v14, v8",    // sr = sb permuted

      // ── MixColumns (xtime decomposition) ──────────────────────────
      "vrgather.vv v16, v15, v9",    // rot1 = column-rotate-by-1
      "vxor.vv v17, v15, v16",       // pair = sr ^ rot1
      "vsll.vi v18, v17, 1",         // pair << 1
      "vsra.vi v19, v17, 7",         // bit-7 mask for xtime reduction
      "vand.vv v19, v19, v12",       // mask & 0x1B
      "vxor.vv v18, v18, v19",       // xt = xtime(pair)
      "vrgather.vv v19, v17, v10",   // rot2_pair = column-rotate-by-2
      "vxor.vv v20, v17, v19",       // col_sum = pair ^ rot2_pair
      "vxor.vv v14, v15, v20",       // sr ^ col_sum
      "vxor.vv v14, v14, v18",       // mc = sr ^ col_sum ^ xt

      // ── AddRoundKey ───────────────────────────────────────────────
      "vxor.vv v0, v14, v1",

      // ── Store result ──────────────────────────────────────────────
      "vse8.v v0, ({out})",

      state = in(reg) block.as_ptr(),
      rk = in(reg) round_key.as_ptr(),
      tbl = in(reg) tables as *const VpermTables as *const u8,
      out = in(reg) out.as_mut_ptr(),
      tmp = out(reg) _,
      options(nostack),
    );
  }
  out
}

// ── AEGIS-256 state operations ──────────────────────────────────────────

#[target_feature(enable = "v")]
#[inline]
/// # Safety
///
/// Caller must ensure the CPU supports the RISC-V `v` extension.
unsafe fn update(s: &mut State, m: &Block, tables: &VpermTables) {
  // SAFETY: caller guarantees the RISC-V V extension is available and the
  // state blocks are valid local buffers for the Hamburg round function.
  unsafe {
    let tmp = s[5];
    s[5] = aes_round(&s[4], &s[5], tables);
    s[4] = aes_round(&s[3], &s[4], tables);
    s[3] = aes_round(&s[2], &s[3], tables);
    s[2] = aes_round(&s[1], &s[2], tables);
    s[1] = aes_round(&s[0], &s[1], tables);
    s[0] = xor_block(&aes_round(&tmp, &s[0], tables), m);
  }
}

#[inline(always)]
fn keystream(s: &State) -> Block {
  xor_block(&xor_block(&s[1], &s[4]), &xor_block(&s[5], &and_block(&s[2], &s[3])))
}

// ── Fused encrypt/decrypt ─────────────────────────────────────────────

#[target_feature(enable = "v")]
/// # Safety
///
/// Caller must ensure the CPU supports the RISC-V `v` extension.
pub(super) unsafe fn encrypt_fused(
  key: &[u8; KEY_SIZE],
  nonce: &[u8; NONCE_SIZE],
  aad: &[u8],
  buffer: &mut [u8],
) -> [u8; TAG_SIZE] {
  // SAFETY: caller guarantees the RISC-V V extension is available for the
  // lifetime of this fused operation and all slices are valid Rust references.
  unsafe {
    let tables = VpermTables::load();
    let (kh0, kh1) = split_halves(key);
    let (nh0, nh1) = split_halves(nonce);
    let k0_xor_n0 = xor_block(kh0, nh0);
    let k1_xor_n1 = xor_block(kh1, nh1);
    let mut s: State = [k0_xor_n0, k1_xor_n1, C1, C0, xor_block(kh0, &C0), xor_block(kh1, &C1)];

    for _ in 0..4 {
      update(&mut s, kh0, &tables);
      update(&mut s, kh1, &tables);
      update(&mut s, &k0_xor_n0, &tables);
      update(&mut s, &k1_xor_n1, &tables);
    }

    let mut offset = 0usize;
    while offset.strict_add(BLOCK_SIZE) <= aad.len() {
      let mut tmp = [0u8; BLOCK_SIZE];
      tmp.copy_from_slice(&aad[offset..offset.strict_add(BLOCK_SIZE)]);
      update(&mut s, &tmp, &tables);
      offset = offset.strict_add(BLOCK_SIZE);
    }
    if offset < aad.len() {
      let mut pad = [0u8; BLOCK_SIZE];
      pad[..aad.len().strict_sub(offset)].copy_from_slice(&aad[offset..]);
      update(&mut s, &pad, &tables);
    }

    let msg_len = buffer.len();
    let len = buffer.len();
    offset = 0;
    while offset.strict_add(BLOCK_SIZE) <= len {
      let z = keystream(&s);
      let mut xi = [0u8; BLOCK_SIZE];
      xi.copy_from_slice(&buffer[offset..offset.strict_add(BLOCK_SIZE)]);
      update(&mut s, &xi, &tables);
      buffer[offset..offset.strict_add(BLOCK_SIZE)].copy_from_slice(&xor_block(&xi, &z));
      offset = offset.strict_add(BLOCK_SIZE);
    }
    if offset < len {
      let z = keystream(&s);
      let tail_len = len.strict_sub(offset);
      let mut pad = [0u8; BLOCK_SIZE];
      pad[..tail_len].copy_from_slice(&buffer[offset..]);
      update(&mut s, &pad, &tables);
      let ct = xor_block(&pad, &z);
      buffer[offset..].copy_from_slice(&ct[..tail_len]);
    }

    let ad_bits = (aad.len() as u64).strict_mul(8);
    let msg_bits = (msg_len as u64).strict_mul(8);
    let mut len_bytes = [0u8; BLOCK_SIZE];
    len_bytes[..8].copy_from_slice(&ad_bits.to_le_bytes());
    len_bytes[8..].copy_from_slice(&msg_bits.to_le_bytes());
    let t = xor_block(&s[3], &len_bytes);
    for _ in 0..7 {
      update(&mut s, &t, &tables);
    }

    xor_block(
      &xor_block(&xor_block(&s[0], &s[1]), &xor_block(&s[2], &s[3])),
      &xor_block(&s[4], &s[5]),
    )
  }
}

#[target_feature(enable = "v")]
/// # Safety
///
/// Caller must ensure the CPU supports the RISC-V `v` extension.
pub(super) unsafe fn decrypt_fused(
  key: &[u8; KEY_SIZE],
  nonce: &[u8; NONCE_SIZE],
  aad: &[u8],
  buffer: &mut [u8],
) -> [u8; TAG_SIZE] {
  // SAFETY: caller guarantees the RISC-V V extension is available for the
  // lifetime of this fused operation and all slices are valid Rust references.
  unsafe {
    let tables = VpermTables::load();
    let (kh0, kh1) = split_halves(key);
    let (nh0, nh1) = split_halves(nonce);
    let k0_xor_n0 = xor_block(kh0, nh0);
    let k1_xor_n1 = xor_block(kh1, nh1);
    let mut s: State = [k0_xor_n0, k1_xor_n1, C1, C0, xor_block(kh0, &C0), xor_block(kh1, &C1)];

    for _ in 0..4 {
      update(&mut s, kh0, &tables);
      update(&mut s, kh1, &tables);
      update(&mut s, &k0_xor_n0, &tables);
      update(&mut s, &k1_xor_n1, &tables);
    }

    let mut offset = 0usize;
    while offset.strict_add(BLOCK_SIZE) <= aad.len() {
      let mut tmp = [0u8; BLOCK_SIZE];
      tmp.copy_from_slice(&aad[offset..offset.strict_add(BLOCK_SIZE)]);
      update(&mut s, &tmp, &tables);
      offset = offset.strict_add(BLOCK_SIZE);
    }
    if offset < aad.len() {
      let mut pad = [0u8; BLOCK_SIZE];
      pad[..aad.len().strict_sub(offset)].copy_from_slice(&aad[offset..]);
      update(&mut s, &pad, &tables);
    }

    let ct_len = buffer.len();
    let len = buffer.len();
    offset = 0;
    while offset.strict_add(BLOCK_SIZE) <= len {
      let z = keystream(&s);
      let mut ci = [0u8; BLOCK_SIZE];
      ci.copy_from_slice(&buffer[offset..offset.strict_add(BLOCK_SIZE)]);
      let xi = xor_block(&ci, &z);
      update(&mut s, &xi, &tables);
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
      update(&mut s, &pt_pad, &tables);
      buffer[offset..].copy_from_slice(&pt_pad[..tail_len]);
    }

    let ad_bits = (aad.len() as u64).strict_mul(8);
    let ct_bits = (ct_len as u64).strict_mul(8);
    let mut len_bytes = [0u8; BLOCK_SIZE];
    len_bytes[..8].copy_from_slice(&ad_bits.to_le_bytes());
    len_bytes[8..].copy_from_slice(&ct_bits.to_le_bytes());
    let t = xor_block(&s[3], &len_bytes);
    for _ in 0..7 {
      update(&mut s, &t, &tables);
    }

    xor_block(
      &xor_block(&xor_block(&s[0], &s[1]), &xor_block(&s[2], &s[3])),
      &xor_block(&s[4], &s[5]),
    )
  }
}
