use core::arch::asm;

/// AES-256 round keys stored as 15 × 16-byte arrays for Zvkned.
///
/// Round keys are loaded from memory into vector registers per call
/// since RISC-V `vreg` is clobber-only (cannot be used as input/output).
#[repr(C, align(64))]
pub(super) struct RvRoundKeys {
  rk: [[u8; 16]; 15],
}

impl RvRoundKeys {
  /// Zeroize all round keys via volatile writes.
  pub(super) fn zeroize(&mut self) {
    // SAFETY: [[u8; 16]; 15] is layout-compatible with [u8; 240].
    let bytes = unsafe { core::slice::from_raw_parts_mut(self.rk.as_mut_ptr().cast::<u8>(), 15usize.strict_mul(16)) };
    crate::traits::ct::zeroize(bytes);
  }
}

/// Convert portable round keys (60 × big-endian u32) to Zvkned byte format.
pub(super) fn from_portable(rk: &[u32; 60]) -> RvRoundKeys {
  let mut keys = [[0u8; 16]; 15];
  let mut i = 0usize;
  while i < 15 {
    let base = i.strict_mul(4);
    // Zvkned uses element-group byte order matching the AES state:
    // 4 × u32 in little-endian element order within each 128-bit group.
    keys[i][0..4].copy_from_slice(&rk[base].to_be_bytes());
    keys[i][4..8].copy_from_slice(&rk[base.strict_add(1)].to_be_bytes());
    keys[i][8..12].copy_from_slice(&rk[base.strict_add(2)].to_be_bytes());
    keys[i][12..16].copy_from_slice(&rk[base.strict_add(3)].to_be_bytes());
    i = i.strict_add(1);
  }
  RvRoundKeys { rk: keys }
}

/// Encrypt a single 16-byte block using AES-256 with Zvkned.
///
/// Performs all 14 rounds in a single asm block, keeping state in v1
/// and loading round keys into v2 from memory. This avoids the vreg
/// input/output limitation since everything goes through memory.
///
/// # Safety
/// Caller must ensure Zvkned vector crypto extension is available.
#[target_feature(enable = "v", enable = "zvkned")]
pub(super) unsafe fn encrypt_block(keys: &RvRoundKeys, block: &mut [u8; 16]) {
  // SAFETY: target_feature gate guarantees Zvkned availability.
  unsafe {
    let block_ptr = block.as_mut_ptr();
    let rk = &keys.rk;

    asm!(
      // Set vl=4 elements of e32 (= 128 bits) in a single vector register.
      "vsetivli zero, 4, e32, m1, ta, ma",
      // Load plaintext into v1.
      "vle32.v v1, ({block})",
      // Initial AddRoundKey (XOR with K0).
      "vle32.v v2, ({rk0})",
      "vaesz.vs v1, v2",
      // Rounds 1–13: SubBytes + ShiftRows + MixColumns + AddRoundKey.
      "vle32.v v2, ({rk1})",
      "vaesem.vs v1, v2",
      "vle32.v v2, ({rk2})",
      "vaesem.vs v1, v2",
      "vle32.v v2, ({rk3})",
      "vaesem.vs v1, v2",
      "vle32.v v2, ({rk4})",
      "vaesem.vs v1, v2",
      "vle32.v v2, ({rk5})",
      "vaesem.vs v1, v2",
      "vle32.v v2, ({rk6})",
      "vaesem.vs v1, v2",
      "vle32.v v2, ({rk7})",
      "vaesem.vs v1, v2",
      "vle32.v v2, ({rk8})",
      "vaesem.vs v1, v2",
      "vle32.v v2, ({rk9})",
      "vaesem.vs v1, v2",
      "vle32.v v2, ({rk10})",
      "vaesem.vs v1, v2",
      "vle32.v v2, ({rk11})",
      "vaesem.vs v1, v2",
      "vle32.v v2, ({rk12})",
      "vaesem.vs v1, v2",
      "vle32.v v2, ({rk13})",
      "vaesem.vs v1, v2",
      // Round 14 (final): SubBytes + ShiftRows + AddRoundKey (no MixColumns).
      "vle32.v v2, ({rk14})",
      "vaesef.vs v1, v2",
      // Store ciphertext.
      "vse32.v v1, ({block})",
      block = in(reg) block_ptr,
      rk0 = in(reg) rk[0].as_ptr(),
      rk1 = in(reg) rk[1].as_ptr(),
      rk2 = in(reg) rk[2].as_ptr(),
      rk3 = in(reg) rk[3].as_ptr(),
      rk4 = in(reg) rk[4].as_ptr(),
      rk5 = in(reg) rk[5].as_ptr(),
      rk6 = in(reg) rk[6].as_ptr(),
      rk7 = in(reg) rk[7].as_ptr(),
      rk8 = in(reg) rk[8].as_ptr(),
      rk9 = in(reg) rk[9].as_ptr(),
      rk10 = in(reg) rk[10].as_ptr(),
      rk11 = in(reg) rk[11].as_ptr(),
      rk12 = in(reg) rk[12].as_ptr(),
      rk13 = in(reg) rk[13].as_ptr(),
      rk14 = in(reg) rk[14].as_ptr(),
      out("v1") _,
      out("v2") _,
      options(nostack),
    );
  }
}

/// Encrypt four independent 16-byte blocks using AES-256 with Zvkned.
///
/// CTR mode feeds AES independent counter blocks. Running four states in one
/// target-feature scope exposes the same instruction-level parallelism that
/// the fallback vperm/fixslice kernels already use for large buffers.
///
/// # Safety
/// Caller must ensure Zvkned vector crypto extension is available.
#[target_feature(enable = "v", enable = "zvkned")]
pub(super) unsafe fn encrypt_4blocks(keys: &RvRoundKeys, blocks: &mut [[u8; 16]; 4]) {
  // SAFETY: four-block Zvkned AES-256 because:
  // 1. The caller guarantees `v` + `zvkned` are available for this target-feature scope.
  // 2. `blocks` is exactly four initialized 16-byte AES states and each pointer stays in bounds.
  // 3. Round-key pointers come from `keys.rk`, which contains all 15 AES-256 round keys.
  unsafe {
    let b0 = blocks[0].as_mut_ptr();
    let b1 = blocks[1].as_mut_ptr();
    let b2 = blocks[2].as_mut_ptr();
    let b3 = blocks[3].as_mut_ptr();
    let rk = &keys.rk;

    asm!(
      "vsetivli zero, 4, e32, m1, ta, ma",
      "vle32.v v1, ({b0})",
      "vle32.v v2, ({b1})",
      "vle32.v v3, ({b2})",
      "vle32.v v4, ({b3})",
      "vle32.v v5, ({rk0})",
      "vaesz.vs v1, v5",
      "vaesz.vs v2, v5",
      "vaesz.vs v3, v5",
      "vaesz.vs v4, v5",
      "vle32.v v5, ({rk1})",
      "vaesem.vs v1, v5",
      "vaesem.vs v2, v5",
      "vaesem.vs v3, v5",
      "vaesem.vs v4, v5",
      "vle32.v v5, ({rk2})",
      "vaesem.vs v1, v5",
      "vaesem.vs v2, v5",
      "vaesem.vs v3, v5",
      "vaesem.vs v4, v5",
      "vle32.v v5, ({rk3})",
      "vaesem.vs v1, v5",
      "vaesem.vs v2, v5",
      "vaesem.vs v3, v5",
      "vaesem.vs v4, v5",
      "vle32.v v5, ({rk4})",
      "vaesem.vs v1, v5",
      "vaesem.vs v2, v5",
      "vaesem.vs v3, v5",
      "vaesem.vs v4, v5",
      "vle32.v v5, ({rk5})",
      "vaesem.vs v1, v5",
      "vaesem.vs v2, v5",
      "vaesem.vs v3, v5",
      "vaesem.vs v4, v5",
      "vle32.v v5, ({rk6})",
      "vaesem.vs v1, v5",
      "vaesem.vs v2, v5",
      "vaesem.vs v3, v5",
      "vaesem.vs v4, v5",
      "vle32.v v5, ({rk7})",
      "vaesem.vs v1, v5",
      "vaesem.vs v2, v5",
      "vaesem.vs v3, v5",
      "vaesem.vs v4, v5",
      "vle32.v v5, ({rk8})",
      "vaesem.vs v1, v5",
      "vaesem.vs v2, v5",
      "vaesem.vs v3, v5",
      "vaesem.vs v4, v5",
      "vle32.v v5, ({rk9})",
      "vaesem.vs v1, v5",
      "vaesem.vs v2, v5",
      "vaesem.vs v3, v5",
      "vaesem.vs v4, v5",
      "vle32.v v5, ({rk10})",
      "vaesem.vs v1, v5",
      "vaesem.vs v2, v5",
      "vaesem.vs v3, v5",
      "vaesem.vs v4, v5",
      "vle32.v v5, ({rk11})",
      "vaesem.vs v1, v5",
      "vaesem.vs v2, v5",
      "vaesem.vs v3, v5",
      "vaesem.vs v4, v5",
      "vle32.v v5, ({rk12})",
      "vaesem.vs v1, v5",
      "vaesem.vs v2, v5",
      "vaesem.vs v3, v5",
      "vaesem.vs v4, v5",
      "vle32.v v5, ({rk13})",
      "vaesem.vs v1, v5",
      "vaesem.vs v2, v5",
      "vaesem.vs v3, v5",
      "vaesem.vs v4, v5",
      "vle32.v v5, ({rk14})",
      "vaesef.vs v1, v5",
      "vaesef.vs v2, v5",
      "vaesef.vs v3, v5",
      "vaesef.vs v4, v5",
      "vse32.v v1, ({b0})",
      "vse32.v v2, ({b1})",
      "vse32.v v3, ({b2})",
      "vse32.v v4, ({b3})",
      b0 = in(reg) b0,
      b1 = in(reg) b1,
      b2 = in(reg) b2,
      b3 = in(reg) b3,
      rk0 = in(reg) rk[0].as_ptr(),
      rk1 = in(reg) rk[1].as_ptr(),
      rk2 = in(reg) rk[2].as_ptr(),
      rk3 = in(reg) rk[3].as_ptr(),
      rk4 = in(reg) rk[4].as_ptr(),
      rk5 = in(reg) rk[5].as_ptr(),
      rk6 = in(reg) rk[6].as_ptr(),
      rk7 = in(reg) rk[7].as_ptr(),
      rk8 = in(reg) rk[8].as_ptr(),
      rk9 = in(reg) rk[9].as_ptr(),
      rk10 = in(reg) rk[10].as_ptr(),
      rk11 = in(reg) rk[11].as_ptr(),
      rk12 = in(reg) rk[12].as_ptr(),
      rk13 = in(reg) rk[13].as_ptr(),
      rk14 = in(reg) rk[14].as_ptr(),
      out("v1") _,
      out("v2") _,
      out("v3") _,
      out("v4") _,
      out("v5") _,
      options(nostack),
    );
  }
}

// AES-128 (11 round keys, 10 rounds)

/// AES-128 round keys stored as 11 × 16-byte arrays for Zvkned.
#[repr(C, align(64))]
pub(super) struct Rv128RoundKeys {
  rk: [[u8; 16]; 11],
}

impl Rv128RoundKeys {
  /// Zeroize all round keys via volatile writes.
  pub(super) fn zeroize(&mut self) {
    // SAFETY: [[u8; 16]; 11] is layout-compatible with [u8; 176].
    let bytes = unsafe { core::slice::from_raw_parts_mut(self.rk.as_mut_ptr().cast::<u8>(), 11usize.strict_mul(16)) };
    crate::traits::ct::zeroize(bytes);
  }
}

/// Convert portable round keys (44 × big-endian u32) to Zvkned byte format.
pub(super) fn from_portable_128(rk: &[u32; 44]) -> Rv128RoundKeys {
  let mut keys = [[0u8; 16]; 11];
  let mut i = 0usize;
  while i < 11 {
    let base = i.strict_mul(4);
    keys[i][0..4].copy_from_slice(&rk[base].to_be_bytes());
    keys[i][4..8].copy_from_slice(&rk[base.strict_add(1)].to_be_bytes());
    keys[i][8..12].copy_from_slice(&rk[base.strict_add(2)].to_be_bytes());
    keys[i][12..16].copy_from_slice(&rk[base.strict_add(3)].to_be_bytes());
    i = i.strict_add(1);
  }
  Rv128RoundKeys { rk: keys }
}

/// Encrypt a single 16-byte block using AES-128 with Zvkned.
///
/// # Safety
/// Caller must ensure Zvkned vector crypto extension is available.
#[target_feature(enable = "v", enable = "zvkned")]
pub(super) unsafe fn encrypt_block_128(keys: &Rv128RoundKeys, block: &mut [u8; 16]) {
  // SAFETY: target_feature gate guarantees Zvkned availability.
  unsafe {
    let block_ptr = block.as_mut_ptr();
    let rk = &keys.rk;

    asm!(
      "vsetivli zero, 4, e32, m1, ta, ma",
      "vle32.v v1, ({block})",
      "vle32.v v2, ({rk0})",
      "vaesz.vs v1, v2",
      // Rounds 1–9.
      "vle32.v v2, ({rk1})",
      "vaesem.vs v1, v2",
      "vle32.v v2, ({rk2})",
      "vaesem.vs v1, v2",
      "vle32.v v2, ({rk3})",
      "vaesem.vs v1, v2",
      "vle32.v v2, ({rk4})",
      "vaesem.vs v1, v2",
      "vle32.v v2, ({rk5})",
      "vaesem.vs v1, v2",
      "vle32.v v2, ({rk6})",
      "vaesem.vs v1, v2",
      "vle32.v v2, ({rk7})",
      "vaesem.vs v1, v2",
      "vle32.v v2, ({rk8})",
      "vaesem.vs v1, v2",
      "vle32.v v2, ({rk9})",
      "vaesem.vs v1, v2",
      // Round 10 (final).
      "vle32.v v2, ({rk10})",
      "vaesef.vs v1, v2",
      "vse32.v v1, ({block})",
      block = in(reg) block_ptr,
      rk0 = in(reg) rk[0].as_ptr(),
      rk1 = in(reg) rk[1].as_ptr(),
      rk2 = in(reg) rk[2].as_ptr(),
      rk3 = in(reg) rk[3].as_ptr(),
      rk4 = in(reg) rk[4].as_ptr(),
      rk5 = in(reg) rk[5].as_ptr(),
      rk6 = in(reg) rk[6].as_ptr(),
      rk7 = in(reg) rk[7].as_ptr(),
      rk8 = in(reg) rk[8].as_ptr(),
      rk9 = in(reg) rk[9].as_ptr(),
      rk10 = in(reg) rk[10].as_ptr(),
      out("v1") _,
      out("v2") _,
      options(nostack),
    );
  }
}

/// Encrypt four independent 16-byte blocks using AES-128 with Zvkned.
///
/// # Safety
/// Caller must ensure Zvkned vector crypto extension is available.
#[target_feature(enable = "v", enable = "zvkned")]
pub(super) unsafe fn encrypt_4blocks_128(keys: &Rv128RoundKeys, blocks: &mut [[u8; 16]; 4]) {
  // SAFETY: four-block Zvkned AES-128 because:
  // 1. The caller guarantees `v` + `zvkned` are available for this target-feature scope.
  // 2. `blocks` is exactly four initialized 16-byte AES states and each pointer stays in bounds.
  // 3. Round-key pointers come from `keys.rk`, which contains all 11 AES-128 round keys.
  unsafe {
    let b0 = blocks[0].as_mut_ptr();
    let b1 = blocks[1].as_mut_ptr();
    let b2 = blocks[2].as_mut_ptr();
    let b3 = blocks[3].as_mut_ptr();
    let rk = &keys.rk;

    asm!(
      "vsetivli zero, 4, e32, m1, ta, ma",
      "vle32.v v1, ({b0})",
      "vle32.v v2, ({b1})",
      "vle32.v v3, ({b2})",
      "vle32.v v4, ({b3})",
      "vle32.v v5, ({rk0})",
      "vaesz.vs v1, v5",
      "vaesz.vs v2, v5",
      "vaesz.vs v3, v5",
      "vaesz.vs v4, v5",
      "vle32.v v5, ({rk1})",
      "vaesem.vs v1, v5",
      "vaesem.vs v2, v5",
      "vaesem.vs v3, v5",
      "vaesem.vs v4, v5",
      "vle32.v v5, ({rk2})",
      "vaesem.vs v1, v5",
      "vaesem.vs v2, v5",
      "vaesem.vs v3, v5",
      "vaesem.vs v4, v5",
      "vle32.v v5, ({rk3})",
      "vaesem.vs v1, v5",
      "vaesem.vs v2, v5",
      "vaesem.vs v3, v5",
      "vaesem.vs v4, v5",
      "vle32.v v5, ({rk4})",
      "vaesem.vs v1, v5",
      "vaesem.vs v2, v5",
      "vaesem.vs v3, v5",
      "vaesem.vs v4, v5",
      "vle32.v v5, ({rk5})",
      "vaesem.vs v1, v5",
      "vaesem.vs v2, v5",
      "vaesem.vs v3, v5",
      "vaesem.vs v4, v5",
      "vle32.v v5, ({rk6})",
      "vaesem.vs v1, v5",
      "vaesem.vs v2, v5",
      "vaesem.vs v3, v5",
      "vaesem.vs v4, v5",
      "vle32.v v5, ({rk7})",
      "vaesem.vs v1, v5",
      "vaesem.vs v2, v5",
      "vaesem.vs v3, v5",
      "vaesem.vs v4, v5",
      "vle32.v v5, ({rk8})",
      "vaesem.vs v1, v5",
      "vaesem.vs v2, v5",
      "vaesem.vs v3, v5",
      "vaesem.vs v4, v5",
      "vle32.v v5, ({rk9})",
      "vaesem.vs v1, v5",
      "vaesem.vs v2, v5",
      "vaesem.vs v3, v5",
      "vaesem.vs v4, v5",
      "vle32.v v5, ({rk10})",
      "vaesef.vs v1, v5",
      "vaesef.vs v2, v5",
      "vaesef.vs v3, v5",
      "vaesef.vs v4, v5",
      "vse32.v v1, ({b0})",
      "vse32.v v2, ({b1})",
      "vse32.v v3, ({b2})",
      "vse32.v v4, ({b3})",
      b0 = in(reg) b0,
      b1 = in(reg) b1,
      b2 = in(reg) b2,
      b3 = in(reg) b3,
      rk0 = in(reg) rk[0].as_ptr(),
      rk1 = in(reg) rk[1].as_ptr(),
      rk2 = in(reg) rk[2].as_ptr(),
      rk3 = in(reg) rk[3].as_ptr(),
      rk4 = in(reg) rk[4].as_ptr(),
      rk5 = in(reg) rk[5].as_ptr(),
      rk6 = in(reg) rk[6].as_ptr(),
      rk7 = in(reg) rk[7].as_ptr(),
      rk8 = in(reg) rk[8].as_ptr(),
      rk9 = in(reg) rk[9].as_ptr(),
      rk10 = in(reg) rk[10].as_ptr(),
      out("v1") _,
      out("v2") _,
      out("v3") _,
      out("v4") _,
      out("v5") _,
      options(nostack),
    );
  }
}
