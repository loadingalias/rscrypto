use core::arch::aarch64::*;

/// AES-256 round keys stored as 15 × 128-bit NEON vectors for AES-CE.
#[derive(Clone, Copy)]
#[repr(C, align(16))]
pub(in crate::aead) struct CeRoundKeys {
  rk: [uint8x16_t; 15],
}

impl CeRoundKeys {
  /// Zeroize all round keys via volatile writes.
  pub(super) fn zeroize(&mut self) {
    // SAFETY: `self.rk` is a valid, aligned, fully-initialized [uint8x16_t; 15].
    // Reinterpreting as a byte slice for volatile zeroization is sound.
    let bytes = unsafe { core::slice::from_raw_parts_mut(self.rk.as_mut_ptr().cast::<u8>(), 15usize.strict_mul(16)) };
    crate::traits::ct::zeroize(bytes);
  }
}

/// Core key-conversion logic — `#[target_feature]` + `#[inline(always)]` for
/// guaranteed inlining without register spills.
#[target_feature(enable = "neon")]
#[inline]
/// # Safety
///
/// Caller must ensure NEON is available and `rk` points to a valid portable
/// AES-256 key schedule.
pub(super) unsafe fn from_portable_core(rk: &[u32; 60]) -> CeRoundKeys {
  // SAFETY: caller guarantees NEON via target_feature chain.
  unsafe {
    let mut keys = [vdupq_n_u8(0); 15];
    let mut i = 0usize;
    while i < 15 {
      let base = i.strict_mul(4);
      let mut bytes = [0u8; 16];
      bytes[0..4].copy_from_slice(&rk[base].to_be_bytes());
      bytes[4..8].copy_from_slice(&rk[base.strict_add(1)].to_be_bytes());
      bytes[8..12].copy_from_slice(&rk[base.strict_add(2)].to_be_bytes());
      bytes[12..16].copy_from_slice(&rk[base.strict_add(3)].to_be_bytes());
      keys[i] = vld1q_u8(bytes.as_ptr());
      i = i.strict_add(1);
    }
    CeRoundKeys { rk: keys }
  }
}

/// Hardware-accelerated AES-256 key expansion using AESE for SubWord.
///
/// The AESE instruction applies SubBytes to all 16 bytes. By broadcasting
/// a 32-bit word to all 4 columns of the AES state, ShiftRows becomes a
/// no-op (all columns identical), so `AESE(broadcast(w), 0)` = `SubWord(w)`.
/// This replaces ~1560 GF(2^8) field operations per SubWord call with a
/// single AESE instruction (~1 cycle on Neoverse V1/V2).
#[target_feature(enable = "aes,neon")]
#[inline]
/// # Safety
///
/// Caller must ensure the CPU supports `aes` and `neon`.
pub(super) unsafe fn expand_key_hw(key: &[u8; 32]) -> CeRoundKeys {
  // SAFETY: caller guarantees AES-CE + NEON availability.
  unsafe {
    // Hardware SubWord: AESE on broadcast input → SubBytes (ShiftRows is
    // no-op because all 4 columns are identical).
    #[target_feature(enable = "aes,neon")]
    #[inline]
    /// # Safety
    ///
    /// Caller must ensure the CPU supports `aes` and `neon`.
    unsafe fn sub_word_hw(w: u32) -> u32 {
      let state = vreinterpretq_u8_u32(vdupq_n_u32(w));
      let zero = vdupq_n_u8(0);
      let result = vaeseq_u8(state, zero);
      vgetq_lane_u32(vreinterpretq_u32_u8(result), 0)
    }

    let mut rk = [0u32; super::EXPANDED_KEY_WORDS];

    // Load initial key as big-endian u32 words.
    let mut i = 0usize;
    while i < 8 {
      let base = i.strict_mul(4);
      rk[i] = u32::from_be_bytes([
        key[base],
        key[base.strict_add(1)],
        key[base.strict_add(2)],
        key[base.strict_add(3)],
      ]);
      i = i.strict_add(1);
    }

    // Expand key schedule using hardware SubWord.
    i = 8;
    while i < super::EXPANDED_KEY_WORDS {
      let mut temp = rk[i.strict_sub(1)];
      if i.strict_rem(8) == 0 {
        temp = sub_word_hw(super::rot_word(temp)) ^ super::RCON[i.strict_div(8).strict_sub(1)];
      } else if i.strict_rem(8) == 4 {
        temp = sub_word_hw(temp);
      }
      rk[i] = rk[i.strict_sub(8)] ^ temp;
      i = i.strict_add(1);
    }

    // Convert to NEON round key format.
    from_portable_core(&rk)
  }
}

/// Non-inline entry point for hardware key expansion (called from the
/// runtime-dispatch `aes256_expand_key`).
#[target_feature(enable = "aes,neon")]
/// # Safety
///
/// Caller must ensure the CPU supports `aes` and `neon`.
pub(super) unsafe fn expand_key(key: &[u8; 32]) -> CeRoundKeys {
  // SAFETY: target_feature gate guarantees AES-CE + NEON.
  unsafe { expand_key_hw(key) }
}

/// Core block-encrypt logic — `#[target_feature]` + `#[inline(always)]` for
/// guaranteed inlining without register spills.
#[target_feature(enable = "aes,neon")]
#[inline]
/// # Safety
///
/// Caller must ensure the CPU supports `aes` and `neon`, and `block` points
/// to a valid writable AES block.
pub(super) unsafe fn encrypt_block_core(keys: &CeRoundKeys, block: &mut [u8; 16]) {
  // SAFETY: caller guarantees AES-CE + NEON via target_feature chain.
  unsafe {
    let k = &keys.rk;
    let mut state = vld1q_u8(block.as_ptr());

    // Rounds 1–13: AESE absorbs the previous round's AddRoundKey,
    // then SubBytes + ShiftRows. AESMC applies MixColumns.
    state = vaesmcq_u8(vaeseq_u8(state, k[0]));
    state = vaesmcq_u8(vaeseq_u8(state, k[1]));
    state = vaesmcq_u8(vaeseq_u8(state, k[2]));
    state = vaesmcq_u8(vaeseq_u8(state, k[3]));
    state = vaesmcq_u8(vaeseq_u8(state, k[4]));
    state = vaesmcq_u8(vaeseq_u8(state, k[5]));
    state = vaesmcq_u8(vaeseq_u8(state, k[6]));
    state = vaesmcq_u8(vaeseq_u8(state, k[7]));
    state = vaesmcq_u8(vaeseq_u8(state, k[8]));
    state = vaesmcq_u8(vaeseq_u8(state, k[9]));
    state = vaesmcq_u8(vaeseq_u8(state, k[10]));
    state = vaesmcq_u8(vaeseq_u8(state, k[11]));
    state = vaesmcq_u8(vaeseq_u8(state, k[12]));

    // Round 14 (final): SubBytes + ShiftRows, then AddRoundKey (no MixColumns).
    state = vaeseq_u8(state, k[13]);
    state = veorq_u8(state, k[14]);

    vst1q_u8(block.as_mut_ptr(), state);
  }
}

/// Encrypt a single 16-byte block using AES-256 with AES-CE.
///
/// ARM's `AESE` instruction performs XOR(state, key) → SubBytes → ShiftRows.
/// Combined with `AESMC` (MixColumns), each middle round is a single
/// `AESMC(AESE(state, K[i]))` pair. The final round omits MixColumns.
///
/// # Safety
/// Caller must ensure the CPU supports AES-CE (`target_feature = "aes"`).
#[target_feature(enable = "aes,neon")]
pub(super) unsafe fn encrypt_block(keys: &CeRoundKeys, block: &mut [u8; 16]) {
  // SAFETY: target_feature gate guarantees AES-CE + NEON.
  unsafe { encrypt_block_core(keys, block) }
}
