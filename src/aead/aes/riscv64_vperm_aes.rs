use core::arch::asm;

use crate::aead::aes_round::{
  AES_AFFINE, MC_ROT1, MC_ROT2, VPERM_INV_HI, VPERM_INV_LO, VPERM_IPT_HI, VPERM_IPT_LO, VPERM_SBOT, VPERM_SBOU,
  VPERM_SR, XTIME_REDUCE,
};

/// Precomputed Hamburg vperm table block — contiguous for offset-based loads.
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

/// Extract round key bytes from the portable key schedule.
#[inline]
fn round_key_bytes(rk: &[u32; super::EXPANDED_KEY_WORDS], round: usize) -> [u8; 16] {
  let off = round.strict_mul(4);
  let mut bytes = [0u8; 16];
  bytes[0..4].copy_from_slice(&rk[off].to_be_bytes());
  bytes[4..8].copy_from_slice(&rk[off.strict_add(1)].to_be_bytes());
  bytes[8..12].copy_from_slice(&rk[off.strict_add(2)].to_be_bytes());
  bytes[12..16].copy_from_slice(&rk[off.strict_add(3)].to_be_bytes());
  bytes
}

macro_rules! vperm_inner_round_m1 {
  ($state:literal, $rk:literal) => {
    concat!(
      "vand.vi v19, ",
      $state,
      ", 15\n",
      "vsrl.vi v20, ",
      $state,
      ", 4\n",
      "vrgather.vv v21, v8, v19\n",
      "vrgather.vv v22, v9, v20\n",
      "vxor.vv v19, v21, v22\n",
      "vand.vi v20, v19, 15\n",
      "vsrl.vi v21, v19, 4\n",
      "vrgather.vv v22, v11, v20\n",
      "vxor.vv v23, v21, v20\n",
      "vrgather.vv v24, v10, v21\n",
      "vxor.vv v25, v24, v22\n",
      "vrgather.vv v24, v10, v23\n",
      "vxor.vv v26, v24, v22\n",
      "vand.vi v27, v25, 15\n",
      "vrgather.vv v28, v10, v27\n",
      "vsra.vi v29, v25, 7\n",
      "vxor.vi v29, v29, -1\n",
      "vand.vv v28, v28, v29\n",
      "vxor.vv v19, v28, v23\n",
      "vand.vi v27, v26, 15\n",
      "vrgather.vv v28, v10, v27\n",
      "vsra.vi v29, v26, 7\n",
      "vxor.vi v29, v29, -1\n",
      "vand.vv v28, v28, v29\n",
      "vxor.vv v20, v28, v21\n",
      "vand.vi v27, v19, 15\n",
      "vrgather.vv v28, v12, v27\n",
      "vsra.vi v29, v19, 7\n",
      "vxor.vi v29, v29, -1\n",
      "vand.vv v22, v28, v29\n",
      "vand.vi v27, v20, 15\n",
      "vrgather.vv v30, v13, v27\n",
      "vsra.vi v29, v20, 7\n",
      "vxor.vi v29, v29, -1\n",
      "vand.vv v23, v30, v29\n",
      "vxor.vv v19, v22, v23\n",
      "vxor.vv v19, v19, v17\n",
      "vrgather.vv v20, v19, v14\n",
      "vrgather.vv v21, v20, v15\n",
      "vxor.vv v22, v20, v21\n",
      "vsll.vi v23, v22, 1\n",
      "vsra.vi v24, v22, 7\n",
      "vand.vv v24, v24, v18\n",
      "vxor.vv v23, v23, v24\n",
      "vrgather.vv v24, v22, v16\n",
      "vxor.vv v25, v22, v24\n",
      "vxor.vv v26, v20, v25\n",
      "vxor.vv v26, v26, v23\n",
      "vxor.vv ",
      $state,
      ", v26, ",
      $rk
    )
  };
}

macro_rules! vperm_final_round_m1 {
  ($state:literal, $rk:literal) => {
    concat!(
      "vand.vi v19, ",
      $state,
      ", 15\n",
      "vsrl.vi v20, ",
      $state,
      ", 4\n",
      "vrgather.vv v21, v8, v19\n",
      "vrgather.vv v22, v9, v20\n",
      "vxor.vv v19, v21, v22\n",
      "vand.vi v20, v19, 15\n",
      "vsrl.vi v21, v19, 4\n",
      "vrgather.vv v22, v11, v20\n",
      "vxor.vv v23, v21, v20\n",
      "vrgather.vv v24, v10, v21\n",
      "vxor.vv v25, v24, v22\n",
      "vrgather.vv v24, v10, v23\n",
      "vxor.vv v26, v24, v22\n",
      "vand.vi v27, v25, 15\n",
      "vrgather.vv v28, v10, v27\n",
      "vsra.vi v29, v25, 7\n",
      "vxor.vi v29, v29, -1\n",
      "vand.vv v28, v28, v29\n",
      "vxor.vv v19, v28, v23\n",
      "vand.vi v27, v26, 15\n",
      "vrgather.vv v28, v10, v27\n",
      "vsra.vi v29, v26, 7\n",
      "vxor.vi v29, v29, -1\n",
      "vand.vv v28, v28, v29\n",
      "vxor.vv v20, v28, v21\n",
      "vand.vi v27, v19, 15\n",
      "vrgather.vv v28, v12, v27\n",
      "vsra.vi v29, v19, 7\n",
      "vxor.vi v29, v29, -1\n",
      "vand.vv v22, v28, v29\n",
      "vand.vi v27, v20, 15\n",
      "vrgather.vv v30, v13, v27\n",
      "vsra.vi v29, v20, 7\n",
      "vxor.vi v29, v29, -1\n",
      "vand.vv v23, v30, v29\n",
      "vxor.vv v19, v22, v23\n",
      "vxor.vv v19, v19, v17\n",
      "vrgather.vv v20, v19, v14\n",
      "vxor.vv ",
      $state,
      ", v20, ",
      $rk
    )
  };
}

/// Single inner AES round (SubBytes + ShiftRows + MixColumns + AddRoundKey).
/// Same asm as aegis256::rv_vperm::aes_round.
#[target_feature(enable = "v")]
#[inline]
/// # Safety
///
/// Caller must ensure the CPU supports the RISC-V `v` extension.
unsafe fn aes_inner_round(block: &[u8; 16], round_key: &[u8; 16], tables: &VpermTables) -> [u8; 16] {
  let mut out = [0u8; 16];
  // SAFETY: caller guarantees the RISC-V V extension is available and the
  // asm block only reads the provided state/table/key buffers and writes `out`.
  unsafe {
    asm!(
      "vsetivli zero, 16, e8, m1, ta, ma",
      // Load tables.
      "vle8.v v2, ({tbl})",
      "addi {tmp}, {tbl}, 16",
      "vle8.v v3, ({tmp})",
      "addi {tmp}, {tbl}, 32",
      "vle8.v v4, ({tmp})",
      "addi {tmp}, {tbl}, 48",
      "vle8.v v5, ({tmp})",
      "addi {tmp}, {tbl}, 64",
      "vle8.v v6, ({tmp})",
      "addi {tmp}, {tbl}, 80",
      "vle8.v v7, ({tmp})",
      "addi {tmp}, {tbl}, 96",
      "vle8.v v8, ({tmp})",
      "addi {tmp}, {tbl}, 112",
      "vle8.v v9, ({tmp})",
      "addi {tmp}, {tbl}, 128",
      "vle8.v v10, ({tmp})",
      "addi {tmp}, {tbl}, 144",
      "vle8.v v11, ({tmp})",
      "addi {tmp}, {tbl}, 160",
      "vle8.v v12, ({tmp})",
      "vle8.v v0, ({state})",
      "vle8.v v1, ({rk})",
      // Phase 1: Nibble extraction.
      "vand.vi v14, v0, 15",
      "vsrl.vi v15, v0, 4",
      // Phase 2: Input transform.
      "vrgather.vv v16, v2, v14",
      "vrgather.vv v17, v3, v15",
      "vxor.vv v14, v16, v17",
      // Phase 3: Re-extract nibbles.
      "vand.vi v15, v14, 15",
      "vsrl.vi v16, v14, 4",
      // Phase 4: GF(2^4) inverse.
      "vrgather.vv v17, v5, v15",
      "vxor.vv v18, v16, v15",
      "vrgather.vv v19, v4, v16",
      "vxor.vv v20, v19, v17",
      "vrgather.vv v21, v4, v18",
      "vxor.vv v22, v21, v17",
      "vand.vi v23, v20, 15",
      "vrgather.vv v24, v4, v23",
      "vsra.vi v25, v20, 7",
      "vxor.vi v26, v25, -1",
      "vand.vv v24, v24, v26",
      "vxor.vv v14, v24, v18",
      "vand.vi v23, v22, 15",
      "vrgather.vv v24, v4, v23",
      "vsra.vi v25, v22, 7",
      "vxor.vi v26, v25, -1",
      "vand.vv v24, v24, v26",
      "vxor.vv v15, v24, v16",
      // Phase 5: Output transform.
      "vand.vi v23, v14, 15",
      "vrgather.vv v24, v6, v23",
      "vsra.vi v25, v14, 7",
      "vxor.vi v26, v25, -1",
      "vand.vv v16, v24, v26",
      "vand.vi v23, v15, 15",
      "vrgather.vv v24, v7, v23",
      "vsra.vi v25, v15, 7",
      "vxor.vi v26, v25, -1",
      "vand.vv v17, v24, v26",
      "vxor.vv v14, v16, v17",
      "vxor.vv v14, v14, v11",
      // ShiftRows.
      "vrgather.vv v15, v14, v8",
      // MixColumns.
      "vrgather.vv v16, v15, v9",
      "vxor.vv v17, v15, v16",
      "vsll.vi v18, v17, 1",
      "vsra.vi v19, v17, 7",
      "vand.vv v19, v19, v12",
      "vxor.vv v18, v18, v19",
      "vrgather.vv v19, v17, v10",
      "vxor.vv v20, v17, v19",
      "vxor.vv v14, v15, v20",
      "vxor.vv v14, v14, v18",
      // AddRoundKey.
      "vxor.vv v0, v14, v1",
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

/// Final AES round (SubBytes + ShiftRows + AddRoundKey, no MixColumns).
#[target_feature(enable = "v")]
#[inline]
/// # Safety
///
/// Caller must ensure the CPU supports the RISC-V `v` extension.
unsafe fn aes_final_round(block: &[u8; 16], round_key: &[u8; 16], tables: &VpermTables) -> [u8; 16] {
  let mut out = [0u8; 16];
  // SAFETY: caller guarantees the RISC-V V extension is available and the
  // asm block only reads the provided state/table/key buffers and writes `out`.
  unsafe {
    asm!(
      "vsetivli zero, 16, e8, m1, ta, ma",
      // Load tables (only need SubBytes + ShiftRows, no MixColumns).
      "vle8.v v2, ({tbl})",
      "addi {tmp}, {tbl}, 16",
      "vle8.v v3, ({tmp})",
      "addi {tmp}, {tbl}, 32",
      "vle8.v v4, ({tmp})",
      "addi {tmp}, {tbl}, 48",
      "vle8.v v5, ({tmp})",
      "addi {tmp}, {tbl}, 64",
      "vle8.v v6, ({tmp})",
      "addi {tmp}, {tbl}, 80",
      "vle8.v v7, ({tmp})",
      "addi {tmp}, {tbl}, 96",
      "vle8.v v8, ({tmp})",
      "addi {tmp}, {tbl}, 144",
      "vle8.v v11, ({tmp})",
      "vle8.v v0, ({state})",
      "vle8.v v1, ({rk})",
      // SubBytes (same as inner round).
      "vand.vi v14, v0, 15",
      "vsrl.vi v15, v0, 4",
      "vrgather.vv v16, v2, v14",
      "vrgather.vv v17, v3, v15",
      "vxor.vv v14, v16, v17",
      "vand.vi v15, v14, 15",
      "vsrl.vi v16, v14, 4",
      "vrgather.vv v17, v5, v15",
      "vxor.vv v18, v16, v15",
      "vrgather.vv v19, v4, v16",
      "vxor.vv v20, v19, v17",
      "vrgather.vv v21, v4, v18",
      "vxor.vv v22, v21, v17",
      "vand.vi v23, v20, 15",
      "vrgather.vv v24, v4, v23",
      "vsra.vi v25, v20, 7",
      "vxor.vi v26, v25, -1",
      "vand.vv v24, v24, v26",
      "vxor.vv v14, v24, v18",
      "vand.vi v23, v22, 15",
      "vrgather.vv v24, v4, v23",
      "vsra.vi v25, v22, 7",
      "vxor.vi v26, v25, -1",
      "vand.vv v24, v24, v26",
      "vxor.vv v15, v24, v16",
      "vand.vi v23, v14, 15",
      "vrgather.vv v24, v6, v23",
      "vsra.vi v25, v14, 7",
      "vxor.vi v26, v25, -1",
      "vand.vv v16, v24, v26",
      "vand.vi v23, v15, 15",
      "vrgather.vv v24, v7, v23",
      "vsra.vi v25, v15, 7",
      "vxor.vi v26, v25, -1",
      "vand.vv v17, v24, v26",
      "vxor.vv v14, v16, v17",
      "vxor.vv v14, v14, v11",
      // ShiftRows (no MixColumns).
      "vrgather.vv v15, v14, v8",
      // AddRoundKey.
      "vxor.vv v0, v15, v1",
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

/// Four independent inner rounds with one shared table load.
///
/// This targets the AES-GCM-SIV fixed-cost path, where six ECB blocks are
/// derived up front. Sharing the vperm tables across 4 blocks removes most
/// of the per-call table-load overhead even before we attempt a wider LMUL
/// kernel.
#[target_feature(enable = "v")]
#[inline]
/// # Safety
///
/// Caller must ensure the CPU supports the RISC-V `v` extension.
unsafe fn aes_inner_round_4(blocks: &mut [[u8; 16]; 4], round_key: &[u8; 16], tables: &VpermTables) {
  // SAFETY: caller guarantees the RISC-V V extension is available and all
  // block/key/table references are valid for 16-byte vector loads/stores.
  unsafe {
    asm!(
      "vsetivli zero, 16, e8, m1, ta, ma",
      "vle8.v v8, ({tbl})",
      "addi {tmp}, {tbl}, 16",
      "vle8.v v9, ({tmp})",
      "addi {tmp}, {tbl}, 32",
      "vle8.v v10, ({tmp})",
      "addi {tmp}, {tbl}, 48",
      "vle8.v v11, ({tmp})",
      "addi {tmp}, {tbl}, 64",
      "vle8.v v12, ({tmp})",
      "addi {tmp}, {tbl}, 80",
      "vle8.v v13, ({tmp})",
      "addi {tmp}, {tbl}, 96",
      "vle8.v v14, ({tmp})",
      "addi {tmp}, {tbl}, 112",
      "vle8.v v15, ({tmp})",
      "addi {tmp}, {tbl}, 128",
      "vle8.v v16, ({tmp})",
      "addi {tmp}, {tbl}, 144",
      "vle8.v v17, ({tmp})",
      "addi {tmp}, {tbl}, 160",
      "vle8.v v18, ({tmp})",
      "vle8.v v0, ({b0})",
      "vle8.v v1, ({b1})",
      "vle8.v v2, ({b2})",
      "vle8.v v3, ({b3})",
      "vle8.v v4, ({rk})",
      vperm_inner_round_m1!("v0", "v4"),
      vperm_inner_round_m1!("v1", "v4"),
      vperm_inner_round_m1!("v2", "v4"),
      vperm_inner_round_m1!("v3", "v4"),
      "vse8.v v0, ({b0})",
      "vse8.v v1, ({b1})",
      "vse8.v v2, ({b2})",
      "vse8.v v3, ({b3})",
      b0 = in(reg) blocks[0].as_mut_ptr(),
      b1 = in(reg) blocks[1].as_mut_ptr(),
      b2 = in(reg) blocks[2].as_mut_ptr(),
      b3 = in(reg) blocks[3].as_mut_ptr(),
      rk = in(reg) round_key.as_ptr(),
      tbl = in(reg) tables as *const VpermTables as *const u8,
      tmp = out(reg) _,
      options(nostack),
    );
  }
}

/// Four independent final rounds with one shared table load.
#[target_feature(enable = "v")]
#[inline]
/// # Safety
///
/// Caller must ensure the CPU supports the RISC-V `v` extension.
unsafe fn aes_final_round_4(blocks: &mut [[u8; 16]; 4], round_key: &[u8; 16], tables: &VpermTables) {
  // SAFETY: caller guarantees the RISC-V V extension is available and all
  // block/key/table references are valid for 16-byte vector loads/stores.
  unsafe {
    asm!(
      "vsetivli zero, 16, e8, m1, ta, ma",
      "vle8.v v8, ({tbl})",
      "addi {tmp}, {tbl}, 16",
      "vle8.v v9, ({tmp})",
      "addi {tmp}, {tbl}, 32",
      "vle8.v v10, ({tmp})",
      "addi {tmp}, {tbl}, 48",
      "vle8.v v11, ({tmp})",
      "addi {tmp}, {tbl}, 64",
      "vle8.v v12, ({tmp})",
      "addi {tmp}, {tbl}, 80",
      "vle8.v v13, ({tmp})",
      "addi {tmp}, {tbl}, 96",
      "vle8.v v14, ({tmp})",
      "addi {tmp}, {tbl}, 144",
      "vle8.v v17, ({tmp})",
      "vle8.v v0, ({b0})",
      "vle8.v v1, ({b1})",
      "vle8.v v2, ({b2})",
      "vle8.v v3, ({b3})",
      "vle8.v v4, ({rk})",
      vperm_final_round_m1!("v0", "v4"),
      vperm_final_round_m1!("v1", "v4"),
      vperm_final_round_m1!("v2", "v4"),
      vperm_final_round_m1!("v3", "v4"),
      "vse8.v v0, ({b0})",
      "vse8.v v1, ({b1})",
      "vse8.v v2, ({b2})",
      "vse8.v v3, ({b3})",
      b0 = in(reg) blocks[0].as_mut_ptr(),
      b1 = in(reg) blocks[1].as_mut_ptr(),
      b2 = in(reg) blocks[2].as_mut_ptr(),
      b3 = in(reg) blocks[3].as_mut_ptr(),
      rk = in(reg) round_key.as_ptr(),
      tbl = in(reg) tables as *const VpermTables as *const u8,
      tmp = out(reg) _,
      options(nostack),
    );
  }
}

/// AES-256 full-block encryption (14 rounds) using Hamburg vperm.
///
/// # Safety
/// Requires the RISC-V V extension.
#[target_feature(enable = "v")]
pub(super) unsafe fn encrypt_block(rk: &[u32; super::EXPANDED_KEY_WORDS], block: &mut [u8; 16]) {
  // SAFETY: caller guarantees the RISC-V V extension is available for the
  // full AES-256 block operation and all references are valid Rust buffers.
  unsafe {
    let tables = VpermTables::load();

    // Initial AddRoundKey (round 0).
    let rk0 = round_key_bytes(rk, 0);
    for i in 0..16 {
      block[i] ^= rk0[i];
    }

    // Rounds 1-13: full AES round (SubBytes + ShiftRows + MixColumns + AddRoundKey).
    let mut round = 1usize;
    while round < super::ROUNDS {
      let rk_r = round_key_bytes(rk, round);
      *block = aes_inner_round(block, &rk_r, &tables);
      round = round.strict_add(1);
    }

    // Round 14 (final): SubBytes + ShiftRows + AddRoundKey (no MixColumns).
    let rk14 = round_key_bytes(rk, super::ROUNDS);
    *block = aes_final_round(block, &rk14, &tables);
  }
}

/// Encrypt 4 independent AES-256 blocks using the vperm backend.
///
/// The 4-block batch is aimed at GCM-SIV key derivation, which produces 6
/// unrelated ECB inputs up front. Processing 4 of them together amortizes
/// table loads across the hot fixed-cost path while keeping the existing
/// single-block kernel for small tails.
#[target_feature(enable = "v")]
/// # Safety
///
/// Caller must ensure the CPU supports the RISC-V `v` extension.
pub(super) unsafe fn encrypt_4blocks(rk: &[u32; super::EXPANDED_KEY_WORDS], blocks: &mut [[u8; 16]; 4]) {
  // SAFETY: caller guarantees the RISC-V V extension is available for the
  // duration of the batch and all block buffers are valid 16-byte arrays.
  unsafe {
    let tables = VpermTables::load();

    let rk0 = round_key_bytes(rk, 0);
    for block in blocks.iter_mut() {
      for i in 0..16 {
        block[i] ^= rk0[i];
      }
    }

    let mut round = 1usize;
    while round < super::ROUNDS {
      let rk_r = round_key_bytes(rk, round);
      aes_inner_round_4(blocks, &rk_r, &tables);
      round = round.strict_add(1);
    }

    let rk14 = round_key_bytes(rk, super::ROUNDS);
    aes_final_round_4(blocks, &rk14, &tables);
  }
}
