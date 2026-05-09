use core::arch::aarch64::*;

#[cfg(all(feature = "aes-gcm", target_os = "macos"))]
#[path = "aarch64/asm.rs"]
mod asm;

/// AES-256 round keys stored as 15 × 128-bit NEON vectors for AES-CE.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
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
  // SAFETY: AES-256 key expansion with AES-CE because:
  // 1. This function's caller guarantees AES-CE + NEON availability.
  // 2. `key` is exactly 32 initialized bytes and both vector loads are in bounds.
  // 3. The generated schedule fills all 15 AES-256 round keys before returning.
  unsafe {
    let zero = vdupq_n_u8(0);
    let rot_mask_bytes = [13u8, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12];
    let splat_mask_bytes = [12u8, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15];
    let rot_mask = vld1q_u8(rot_mask_bytes.as_ptr());
    let splat_mask = vld1q_u8(splat_mask_bytes.as_ptr());
    let rcon = [0x01u32, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40];

    let mut keys = [vdupq_n_u8(0); 15];
    let mut left = vld1q_u8(key.as_ptr());
    let mut right = vld1q_u8(key.as_ptr().add(16));
    keys[0] = left;
    keys[1] = right;

    let mut round = 0usize;
    let mut out = 2usize;
    while round < 7 {
      let mut sub = vqtbl1q_u8(right, rot_mask);
      sub = vaeseq_u8(sub, zero);
      sub = veorq_u8(sub, vreinterpretq_u8_u32(vdupq_n_u32(rcon[round])));

      let mut shifted = vextq_u8(zero, left, 12);
      left = veorq_u8(left, shifted);
      shifted = vextq_u8(zero, shifted, 12);
      left = veorq_u8(left, shifted);
      shifted = vextq_u8(zero, shifted, 12);
      left = veorq_u8(left, shifted);
      left = veorq_u8(left, sub);
      keys[out] = left;
      out = out.strict_add(1);

      if round == 6 {
        break;
      }

      let mut sub = vqtbl1q_u8(left, splat_mask);
      sub = vaeseq_u8(sub, zero);

      shifted = vextq_u8(zero, right, 12);
      right = veorq_u8(right, shifted);
      shifted = vextq_u8(zero, shifted, 12);
      right = veorq_u8(right, shifted);
      shifted = vextq_u8(zero, shifted, 12);
      right = veorq_u8(right, shifted);
      right = veorq_u8(right, sub);
      keys[out] = right;
      out = out.strict_add(1);

      round = round.strict_add(1);
    }

    CeRoundKeys { rk: keys }
  }
}

/// Non-inline entry point for hardware key expansion (called from the
/// runtime-dispatch `aes256_expand_key`).
#[target_feature(enable = "aes,neon")]
/// # Safety
///
/// Caller must ensure the CPU supports `aes` and `neon`.
pub(super) unsafe fn expand_key(key: &[u8; 32]) -> CeRoundKeys {
  // SAFETY: AES-256 hardware key expansion dispatch because:
  // 1. This function has `#[target_feature(enable = "aes,neon")]`.
  // 2. `key` is exactly 32 initialized bytes.
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
  // SAFETY: AES-CE AES-256 single-block encryption because:
  // 1. This function has `#[target_feature(enable = "aes,neon")]`.
  // 2. The caller guarantees the current CPU supports AES-CE + NEON.
  // 3. `block` is exactly one writable initialized AES block.
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

#[target_feature(enable = "aes,neon")]
#[inline]
/// # Safety
///
/// Caller must ensure the CPU supports `aes` and `neon`.
pub(super) unsafe fn encrypt_4blocks_core(keys: &CeRoundKeys, blocks: &mut [[u8; 16]; 4]) {
  // SAFETY: four-block AES-256 ECB processing because:
  // 1. This function's caller guarantees AES-CE + NEON availability.
  // 2. `blocks` is a valid mutable reference to exactly four initialized AES blocks.
  // 3. All vector loads and stores use fixed in-bounds block references.
  unsafe {
    let k = &keys.rk;
    let mut s0 = vld1q_u8(blocks[0].as_ptr());
    let mut s1 = vld1q_u8(blocks[1].as_ptr());
    let mut s2 = vld1q_u8(blocks[2].as_ptr());
    let mut s3 = vld1q_u8(blocks[3].as_ptr());

    macro_rules! round4 {
      ($round:expr) => {{
        s0 = vaesmcq_u8(vaeseq_u8(s0, k[$round]));
        s1 = vaesmcq_u8(vaeseq_u8(s1, k[$round]));
        s2 = vaesmcq_u8(vaeseq_u8(s2, k[$round]));
        s3 = vaesmcq_u8(vaeseq_u8(s3, k[$round]));
      }};
    }

    round4!(0);
    round4!(1);
    round4!(2);
    round4!(3);
    round4!(4);
    round4!(5);
    round4!(6);
    round4!(7);
    round4!(8);
    round4!(9);
    round4!(10);
    round4!(11);
    round4!(12);

    s0 = veorq_u8(vaeseq_u8(s0, k[13]), k[14]);
    s1 = veorq_u8(vaeseq_u8(s1, k[13]), k[14]);
    s2 = veorq_u8(vaeseq_u8(s2, k[13]), k[14]);
    s3 = veorq_u8(vaeseq_u8(s3, k[13]), k[14]);

    vst1q_u8(blocks[0].as_mut_ptr(), s0);
    vst1q_u8(blocks[1].as_mut_ptr(), s1);
    vst1q_u8(blocks[2].as_mut_ptr(), s2);
    vst1q_u8(blocks[3].as_mut_ptr(), s3);
  }
}

#[target_feature(enable = "aes,neon")]
#[inline]
/// # Safety
///
/// Caller must ensure the CPU supports `aes` and `neon`.
pub(super) unsafe fn encrypt_6blocks_core(keys: &CeRoundKeys, blocks: &mut [[u8; 16]; 6]) {
  // SAFETY: six-block AES-256 ECB processing because:
  // 1. This function's caller guarantees AES-CE + NEON availability.
  // 2. `blocks` is a valid mutable reference to exactly six initialized AES blocks.
  // 3. All vector loads and stores use fixed in-bounds block references.
  unsafe {
    let k = &keys.rk;
    let mut s0 = vld1q_u8(blocks[0].as_ptr());
    let mut s1 = vld1q_u8(blocks[1].as_ptr());
    let mut s2 = vld1q_u8(blocks[2].as_ptr());
    let mut s3 = vld1q_u8(blocks[3].as_ptr());
    let mut s4 = vld1q_u8(blocks[4].as_ptr());
    let mut s5 = vld1q_u8(blocks[5].as_ptr());

    macro_rules! round6 {
      ($round:expr) => {{
        s0 = vaesmcq_u8(vaeseq_u8(s0, k[$round]));
        s1 = vaesmcq_u8(vaeseq_u8(s1, k[$round]));
        s2 = vaesmcq_u8(vaeseq_u8(s2, k[$round]));
        s3 = vaesmcq_u8(vaeseq_u8(s3, k[$round]));
        s4 = vaesmcq_u8(vaeseq_u8(s4, k[$round]));
        s5 = vaesmcq_u8(vaeseq_u8(s5, k[$round]));
      }};
    }

    round6!(0);
    round6!(1);
    round6!(2);
    round6!(3);
    round6!(4);
    round6!(5);
    round6!(6);
    round6!(7);
    round6!(8);
    round6!(9);
    round6!(10);
    round6!(11);
    round6!(12);

    s0 = veorq_u8(vaeseq_u8(s0, k[13]), k[14]);
    s1 = veorq_u8(vaeseq_u8(s1, k[13]), k[14]);
    s2 = veorq_u8(vaeseq_u8(s2, k[13]), k[14]);
    s3 = veorq_u8(vaeseq_u8(s3, k[13]), k[14]);
    s4 = veorq_u8(vaeseq_u8(s4, k[13]), k[14]);
    s5 = veorq_u8(vaeseq_u8(s5, k[13]), k[14]);

    vst1q_u8(blocks[0].as_mut_ptr(), s0);
    vst1q_u8(blocks[1].as_mut_ptr(), s1);
    vst1q_u8(blocks[2].as_mut_ptr(), s2);
    vst1q_u8(blocks[3].as_mut_ptr(), s3);
    vst1q_u8(blocks[4].as_mut_ptr(), s4);
    vst1q_u8(blocks[5].as_mut_ptr(), s5);
  }
}

#[cfg(feature = "aes-gcm")]
#[inline]
unsafe fn ghash_be_u128_from_vec(block: uint8x16_t) -> u128 {
  // SAFETY: GHASH lane conversion because:
  // 1. The caller is already inside an AArch64 NEON target scope.
  // 2. `block` is an initialized vector register.
  // 3. The byte reversal maps memory-order big-endian GHASH bytes to the little-endian lane
  //    representation produced by `u128::from_be_bytes`.
  unsafe {
    let rev64 = vrev64q_u8(block);
    let rev = vextq_u8(rev64, rev64, 8);
    let lanes = vreinterpretq_u64_u8(rev);
    (vgetq_lane_u64(lanes, 0) as u128) | ((vgetq_lane_u64(lanes, 1) as u128) << 64)
  }
}

#[cfg(feature = "aes-gcm")]
#[inline(always)]
unsafe fn ghash_u128_to_lanes(x: u128) -> uint64x2_t {
  // SAFETY: GHASH accumulator lane construction because:
  // 1. The caller is already inside an AArch64 NEON target scope.
  // 2. `vcreate_u64` initializes one 64-bit lane from an integer value.
  // 3. `vcombine_u64` builds a fully initialized two-lane vector.
  unsafe { vcombine_u64(vcreate_u64(x as u64), vcreate_u64((x >> 64) as u64)) }
}

#[cfg(feature = "aes-gcm")]
#[inline(always)]
unsafe fn ghash_mont_reduce_neon(lo: uint64x2_t, hi: uint64x2_t) -> uint64x2_t {
  // SAFETY: GHASH Montgomery reduction because:
  // 1. The caller is already inside an AArch64 NEON target scope.
  // 2. `lo` and `hi` are initialized carryless-product accumulator lanes.
  // 3. All shifts use constant lane widths accepted by the NEON intrinsics.
  unsafe {
    let zero = vdupq_n_u64(0);
    let left = veorq_u64(veorq_u64(vshlq_n_u64(lo, 63), vshlq_n_u64(lo, 62)), vshlq_n_u64(lo, 57));
    let lo_folded = veorq_u64(lo, vextq_u64(zero, left, 1));
    let right = veorq_u64(
      veorq_u64(lo_folded, vshrq_n_u64(lo_folded, 1)),
      veorq_u64(vshrq_n_u64(lo_folded, 2), vshrq_n_u64(lo_folded, 7)),
    );
    let left2 = veorq_u64(
      veorq_u64(vshlq_n_u64(lo_folded, 63), vshlq_n_u64(lo_folded, 62)),
      vshlq_n_u64(lo_folded, 57),
    );
    veorq_u64(veorq_u64(hi, right), vextq_u64(left2, zero, 1))
  }
}

#[cfg(feature = "aes-gcm")]
#[inline(always)]
unsafe fn ghash_finish_products(ll: uint64x2_t, hh: uint64x2_t, mm: uint64x2_t) -> u128 {
  // SAFETY: GHASH Karatsuba accumulator finalization because:
  // 1. The caller is already inside an AArch64 NEON target scope.
  // 2. `ll`, `hh`, and `mm` are initialized low/high/middle product accumulators.
  // 3. The final Montgomery reduction returns a two-lane value convertible to `u128`.
  unsafe {
    let mid = veorq_u64(veorq_u64(mm, ll), hh);
    let zero = vdupq_n_u64(0);
    let lo = veorq_u64(ll, vextq_u64(zero, mid, 1));
    let hi = veorq_u64(hh, vextq_u64(mid, zero, 1));
    let result = ghash_mont_reduce_neon(lo, hi);
    (vgetq_lane_u64(result, 0) as u128) | ((vgetq_lane_u64(result, 1) as u128) << 64)
  }
}

#[cfg(feature = "aes-gcm")]
#[inline(always)]
unsafe fn gcm_load_ciphertext_block(ptr: *const u8) -> uint8x16_t {
  // SAFETY: opaque 16-byte ciphertext load for GCM open because:
  // 1. The caller passes a pointer into a 128-byte chunk already bounds-checked by the enclosing
  //    decrypt helper.
  // 2. `ldr qN, [xN]` accepts arbitrary byte alignment on supported AArch64 targets.
  // 3. The asm output is used only as an initialized NEON vector for GHASH; it does not alias or
  //    mutate the input buffer.
  unsafe {
    let block: uint8x16_t;
    core::arch::asm!(
      "ldr {block:q}, [{ptr}]",
      block = lateout(vreg) block,
      ptr = in(reg) ptr,
      options(readonly, nostack, preserves_flags),
    );
    block
  }
}

#[cfg(feature = "aes-gcm")]
#[inline(always)]
unsafe fn ghash_load_power(power: *const u128) -> uint64x2_t {
  // SAFETY: GHASH H-power vector load because:
  // 1. The caller passes a pointer into a live H-power table.
  // 2. `vld1q_u64` accepts arbitrary alignment.
  // 3. The load reads exactly one initialized `u128` represented as two `u64` lanes.
  unsafe { vld1q_u64(power.cast::<u64>()) }
}

#[cfg(feature = "aes-gcm")]
macro_rules! ghash_fold_be_vec {
  ($ll:ident, $hh:ident, $mm:ident, $block:expr, $idx:expr, $h_powers_rev:expr, $acc_lanes:expr) => {{
    let rev64 = vrev64q_u8($block);
    let rev = vextq_u8(rev64, rev64, 8);
    let lanes = veorq_u64(vreinterpretq_u64_u8(rev), $acc_lanes);
    let h = ghash_load_power($h_powers_rev.as_ptr().add($idx));

    let b_poly = vreinterpretq_p64_u64(lanes);
    let h_poly = vreinterpretq_p64_u64(h);
    $ll = veorq_u64(
      $ll,
      vreinterpretq_u64_p128(vmull_p64(vgetq_lane_p64(b_poly, 0), vgetq_lane_p64(h_poly, 0))),
    );
    $hh = veorq_u64($hh, vreinterpretq_u64_p128(vmull_high_p64(b_poly, h_poly)));

    let b_mid = veorq_u64(lanes, vextq_u64(lanes, lanes, 1));
    let h_mid = veorq_u64(h, vextq_u64(h, h, 1));
    let b_mid_poly = vreinterpretq_p64_u64(b_mid);
    let h_mid_poly = vreinterpretq_p64_u64(h_mid);
    $mm = veorq_u64(
      $mm,
      vreinterpretq_u64_p128(vmull_p64(vgetq_lane_p64(b_mid_poly, 0), vgetq_lane_p64(h_mid_poly, 0))),
    );
  }};
}

#[cfg(feature = "aes-gcm")]
macro_rules! gcm_schedule_barrier {
  ($s0:ident, $s1:ident, $s2:ident, $s3:ident, $s4:ident, $s5:ident, $s6:ident, $s7:ident, $ll:ident, $hh:ident, $mm:ident) => {{}};
}

#[cfg(feature = "aes-gcm")]
#[inline(always)]
unsafe fn gcm_ctr32_base(iv_prefix: &[u8; 12]) -> uint8x16_t {
  let mut block = [0u8; 16];
  block[..12].copy_from_slice(iv_prefix);
  // SAFETY: GCM counter template load because:
  // 1. The caller is already inside an AArch64 NEON target scope.
  // 2. `block` is a fully initialized 16-byte counter template.
  // 3. `vld1q_u8` accepts arbitrary alignment for the stack array.
  unsafe { vld1q_u8(block.as_ptr()) }
}

#[cfg(feature = "aes-gcm")]
#[inline(always)]
unsafe fn gcm_ctr32_block(base: uint8x16_t, ctr: u32) -> uint8x16_t {
  // SAFETY: GCM counter lane update because:
  // 1. The caller is already inside an AArch64 NEON target scope.
  // 2. Lane 3 maps to bytes 12..16 on little-endian AArch64 targets.
  // 3. `swap_bytes` gives the required GCM big-endian counter byte order in memory.
  unsafe { vreinterpretq_u8_u32(vsetq_lane_u32(ctr.swap_bytes(), vreinterpretq_u32_u8(base), 3)) }
}

#[cfg(feature = "aes-gcm-siv")]
#[inline(always)]
unsafe fn gcmsiv_ctr32_base(iv_suffix: &[u8; 12]) -> uint8x16_t {
  let mut block = [0u8; 16];
  block[4..16].copy_from_slice(iv_suffix);
  // SAFETY: GCM-SIV counter template load because:
  // 1. The caller is already inside an AArch64 NEON target scope.
  // 2. `block` is a fully initialized 16-byte counter template.
  // 3. `vld1q_u8` accepts arbitrary alignment for the stack array.
  unsafe { vld1q_u8(block.as_ptr()) }
}

#[cfg(feature = "aes-gcm-siv")]
#[inline(always)]
unsafe fn gcmsiv_ctr32_block(base: uint8x16_t, ctr: u32) -> uint8x16_t {
  // SAFETY: GCM-SIV counter lane update because:
  // 1. The caller is inside a NEON target scope.
  // 2. AArch64 is little-endian for supported Rust targets here.
  // 3. Lane 0 maps to bytes 0..4, exactly GCM-SIV's little-endian counter field.
  unsafe { vreinterpretq_u8_u32(vsetq_lane_u32(ctr, vreinterpretq_u32_u8(base), 0)) }
}

#[cfg(feature = "aes-gcm-siv")]
#[target_feature(enable = "aes,neon")]
#[inline]
/// # Safety
///
/// Caller must ensure the CPU supports `aes` and `neon`.
pub(super) unsafe fn gcmsiv_derive_keys_core(keys: &CeRoundKeys, nonce: &[u8; 12]) -> ([u8; 16], [u8; 32]) {
  // SAFETY: six-block AES-256 GCM-SIV KDF because:
  // 1. This function's caller guarantees AES-CE + NEON availability.
  // 2. The nonce is exactly 96 bits; counter blocks are constructed as LE32 counter || nonce.
  // 3. Only the low 64 bits of each encrypted block are copied, matching RFC 8452 key derivation.
  unsafe {
    let k = &keys.rk;
    let base = gcmsiv_ctr32_base(nonce);
    let mut s0 = gcmsiv_ctr32_block(base, 0);
    let mut s1 = gcmsiv_ctr32_block(base, 1);
    let mut s2 = gcmsiv_ctr32_block(base, 2);
    let mut s3 = gcmsiv_ctr32_block(base, 3);
    let mut s4 = gcmsiv_ctr32_block(base, 4);
    let mut s5 = gcmsiv_ctr32_block(base, 5);

    macro_rules! round6 {
      ($round:expr) => {{
        s0 = vaesmcq_u8(vaeseq_u8(s0, k[$round]));
        s1 = vaesmcq_u8(vaeseq_u8(s1, k[$round]));
        s2 = vaesmcq_u8(vaeseq_u8(s2, k[$round]));
        s3 = vaesmcq_u8(vaeseq_u8(s3, k[$round]));
        s4 = vaesmcq_u8(vaeseq_u8(s4, k[$round]));
        s5 = vaesmcq_u8(vaeseq_u8(s5, k[$round]));
      }};
    }

    round6!(0);
    round6!(1);
    round6!(2);
    round6!(3);
    round6!(4);
    round6!(5);
    round6!(6);
    round6!(7);
    round6!(8);
    round6!(9);
    round6!(10);
    round6!(11);
    round6!(12);

    s0 = veorq_u8(vaeseq_u8(s0, k[13]), k[14]);
    s1 = veorq_u8(vaeseq_u8(s1, k[13]), k[14]);
    s2 = veorq_u8(vaeseq_u8(s2, k[13]), k[14]);
    s3 = veorq_u8(vaeseq_u8(s3, k[13]), k[14]);
    s4 = veorq_u8(vaeseq_u8(s4, k[13]), k[14]);
    s5 = veorq_u8(vaeseq_u8(s5, k[13]), k[14]);

    let mut auth_key = [0u8; 16];
    let mut enc_key = [0u8; 32];
    let low0 = vgetq_lane_u64(vreinterpretq_u64_u8(s0), 0).to_ne_bytes();
    let low1 = vgetq_lane_u64(vreinterpretq_u64_u8(s1), 0).to_ne_bytes();
    let low2 = vgetq_lane_u64(vreinterpretq_u64_u8(s2), 0).to_ne_bytes();
    let low3 = vgetq_lane_u64(vreinterpretq_u64_u8(s3), 0).to_ne_bytes();
    let low4 = vgetq_lane_u64(vreinterpretq_u64_u8(s4), 0).to_ne_bytes();
    let low5 = vgetq_lane_u64(vreinterpretq_u64_u8(s5), 0).to_ne_bytes();
    auth_key[0..8].copy_from_slice(&low0);
    auth_key[8..16].copy_from_slice(&low1);
    enc_key[0..8].copy_from_slice(&low2);
    enc_key[8..16].copy_from_slice(&low3);
    enc_key[16..24].copy_from_slice(&low4);
    enc_key[24..32].copy_from_slice(&low5);

    (auth_key, enc_key)
  }
}

#[cfg(feature = "aes-gcm-siv")]
#[target_feature(enable = "aes,neon")]
#[inline]
/// # Safety
///
/// Caller must ensure the CPU supports `aes` and `neon`.
pub(super) unsafe fn gcmsiv_derive_keys_128_core(keys: &Ce128RoundKeys, nonce: &[u8; 12]) -> ([u8; 16], [u8; 16]) {
  // SAFETY: four-block AES-128 GCM-SIV KDF because:
  // 1. This function's caller guarantees AES-CE + NEON availability.
  // 2. The nonce is exactly 96 bits; counter blocks are constructed as LE32 counter || nonce.
  // 3. Only the low 64 bits of each encrypted block are copied, matching RFC 8452 key derivation.
  unsafe {
    let k = &keys.rk;
    let base = gcmsiv_ctr32_base(nonce);
    let mut s0 = gcmsiv_ctr32_block(base, 0);
    let mut s1 = gcmsiv_ctr32_block(base, 1);
    let mut s2 = gcmsiv_ctr32_block(base, 2);
    let mut s3 = gcmsiv_ctr32_block(base, 3);

    macro_rules! round4 {
      ($round:expr) => {{
        s0 = vaesmcq_u8(vaeseq_u8(s0, k[$round]));
        s1 = vaesmcq_u8(vaeseq_u8(s1, k[$round]));
        s2 = vaesmcq_u8(vaeseq_u8(s2, k[$round]));
        s3 = vaesmcq_u8(vaeseq_u8(s3, k[$round]));
      }};
    }

    round4!(0);
    round4!(1);
    round4!(2);
    round4!(3);
    round4!(4);
    round4!(5);
    round4!(6);
    round4!(7);
    round4!(8);

    s0 = veorq_u8(vaeseq_u8(s0, k[9]), k[10]);
    s1 = veorq_u8(vaeseq_u8(s1, k[9]), k[10]);
    s2 = veorq_u8(vaeseq_u8(s2, k[9]), k[10]);
    s3 = veorq_u8(vaeseq_u8(s3, k[9]), k[10]);

    let mut auth_key = [0u8; 16];
    let mut enc_key = [0u8; 16];
    let low0 = vgetq_lane_u64(vreinterpretq_u64_u8(s0), 0).to_ne_bytes();
    let low1 = vgetq_lane_u64(vreinterpretq_u64_u8(s1), 0).to_ne_bytes();
    let low2 = vgetq_lane_u64(vreinterpretq_u64_u8(s2), 0).to_ne_bytes();
    let low3 = vgetq_lane_u64(vreinterpretq_u64_u8(s3), 0).to_ne_bytes();
    auth_key[0..8].copy_from_slice(&low0);
    auth_key[8..16].copy_from_slice(&low1);
    enc_key[0..8].copy_from_slice(&low2);
    enc_key[8..16].copy_from_slice(&low3);

    (auth_key, enc_key)
  }
}

/// Encrypt four AES-256 GCM counter blocks, XOR them into `data`, and return
/// the resulting ciphertext lanes as `u128::from_be_bytes` GHASH inputs.
///
/// # Safety
///
/// Caller must ensure the CPU supports `aes` and `neon`, `data` has at least
/// 64 writable bytes, and `iv_prefix` is the fixed 96-bit GCM nonce prefix.
#[cfg(feature = "aes-gcm")]
#[target_feature(enable = "aes,neon")]
#[inline]
pub(super) unsafe fn encrypt_ctr32_be_xor_4blocks_core(
  keys: &CeRoundKeys,
  iv_prefix: &[u8; 12],
  ctr: u32,
  data: &mut [u8],
) -> [u128; 4] {
  debug_assert!(data.len() >= 64);

  // SAFETY: four-block AES-GCM CTR processing because:
  // 1. This function's caller guarantees AES-CE + NEON availability.
  // 2. `data` has at least 64 writable bytes; all loads/stores use fixed 16-byte offsets within that
  //    range.
  // 3. Counter blocks are initialized before loading into vector registers.
  unsafe {
    let k = &keys.rk;
    let base = gcm_ctr32_base(iv_prefix);
    let mut s0 = gcm_ctr32_block(base, ctr);
    let mut s1 = gcm_ctr32_block(base, ctr.wrapping_add(1));
    let mut s2 = gcm_ctr32_block(base, ctr.wrapping_add(2));
    let mut s3 = gcm_ctr32_block(base, ctr.wrapping_add(3));

    macro_rules! round4 {
      ($round:expr) => {{
        s0 = vaesmcq_u8(vaeseq_u8(s0, k[$round]));
        s1 = vaesmcq_u8(vaeseq_u8(s1, k[$round]));
        s2 = vaesmcq_u8(vaeseq_u8(s2, k[$round]));
        s3 = vaesmcq_u8(vaeseq_u8(s3, k[$round]));
      }};
    }

    round4!(0);
    round4!(1);
    round4!(2);
    round4!(3);
    round4!(4);
    round4!(5);
    round4!(6);
    round4!(7);
    round4!(8);
    round4!(9);
    round4!(10);
    round4!(11);
    round4!(12);

    s0 = veorq_u8(vaeseq_u8(s0, k[13]), k[14]);
    s1 = veorq_u8(vaeseq_u8(s1, k[13]), k[14]);
    s2 = veorq_u8(vaeseq_u8(s2, k[13]), k[14]);
    s3 = veorq_u8(vaeseq_u8(s3, k[13]), k[14]);

    let p0 = vld1q_u8(data.as_ptr());
    let p1 = vld1q_u8(data.as_ptr().add(16));
    let p2 = vld1q_u8(data.as_ptr().add(32));
    let p3 = vld1q_u8(data.as_ptr().add(48));
    let c0 = veorq_u8(p0, s0);
    let c1 = veorq_u8(p1, s1);
    let c2 = veorq_u8(p2, s2);
    let c3 = veorq_u8(p3, s3);
    vst1q_u8(data.as_mut_ptr(), c0);
    vst1q_u8(data.as_mut_ptr().add(16), c1);
    vst1q_u8(data.as_mut_ptr().add(32), c2);
    vst1q_u8(data.as_mut_ptr().add(48), c3);

    [
      ghash_be_u128_from_vec(c0),
      ghash_be_u128_from_vec(c1),
      ghash_be_u128_from_vec(c2),
      ghash_be_u128_from_vec(c3),
    ]
  }
}

/// Encrypt eight AES-256 GCM counter blocks, XOR them into `data`, and return
/// the resulting ciphertext lanes in memory byte order.
///
/// # Safety
///
/// Caller must ensure the CPU supports `aes` and `neon`, `data` has at least
/// 128 writable bytes, and `iv_prefix` is the fixed 96-bit GCM nonce prefix.
#[cfg(feature = "aes-gcm")]
#[target_feature(enable = "aes,neon")]
#[inline]
pub(super) unsafe fn encrypt_ctr32_be_xor_8blocks_core(
  keys: &CeRoundKeys,
  iv_prefix: &[u8; 12],
  ctr: u32,
  data: &mut [u8],
) -> [uint8x16_t; 8] {
  debug_assert!(data.len() >= 128);

  // SAFETY: eight-block AES-GCM CTR processing because:
  // 1. This function's caller guarantees AES-CE + NEON availability.
  // 2. `data` has at least 128 writable bytes; all loads/stores use fixed 16-byte offsets within that
  //    range.
  // 3. Counter blocks are initialized before loading into vector registers.
  unsafe {
    let k = &keys.rk;
    let base = gcm_ctr32_base(iv_prefix);
    let mut s0 = gcm_ctr32_block(base, ctr);
    let mut s1 = gcm_ctr32_block(base, ctr.wrapping_add(1));
    let mut s2 = gcm_ctr32_block(base, ctr.wrapping_add(2));
    let mut s3 = gcm_ctr32_block(base, ctr.wrapping_add(3));
    let mut s4 = gcm_ctr32_block(base, ctr.wrapping_add(4));
    let mut s5 = gcm_ctr32_block(base, ctr.wrapping_add(5));
    let mut s6 = gcm_ctr32_block(base, ctr.wrapping_add(6));
    let mut s7 = gcm_ctr32_block(base, ctr.wrapping_add(7));

    macro_rules! round8 {
      ($round:expr) => {{
        s0 = vaesmcq_u8(vaeseq_u8(s0, k[$round]));
        s1 = vaesmcq_u8(vaeseq_u8(s1, k[$round]));
        s2 = vaesmcq_u8(vaeseq_u8(s2, k[$round]));
        s3 = vaesmcq_u8(vaeseq_u8(s3, k[$round]));
        s4 = vaesmcq_u8(vaeseq_u8(s4, k[$round]));
        s5 = vaesmcq_u8(vaeseq_u8(s5, k[$round]));
        s6 = vaesmcq_u8(vaeseq_u8(s6, k[$round]));
        s7 = vaesmcq_u8(vaeseq_u8(s7, k[$round]));
      }};
    }

    round8!(0);
    round8!(1);
    round8!(2);
    round8!(3);
    round8!(4);
    round8!(5);
    round8!(6);
    round8!(7);
    round8!(8);
    round8!(9);
    round8!(10);
    round8!(11);
    round8!(12);

    s0 = veorq_u8(vaeseq_u8(s0, k[13]), k[14]);
    s1 = veorq_u8(vaeseq_u8(s1, k[13]), k[14]);
    s2 = veorq_u8(vaeseq_u8(s2, k[13]), k[14]);
    s3 = veorq_u8(vaeseq_u8(s3, k[13]), k[14]);
    s4 = veorq_u8(vaeseq_u8(s4, k[13]), k[14]);
    s5 = veorq_u8(vaeseq_u8(s5, k[13]), k[14]);
    s6 = veorq_u8(vaeseq_u8(s6, k[13]), k[14]);
    s7 = veorq_u8(vaeseq_u8(s7, k[13]), k[14]);

    let p0 = vld1q_u8(data.as_ptr());
    let p1 = vld1q_u8(data.as_ptr().add(16));
    let p2 = vld1q_u8(data.as_ptr().add(32));
    let p3 = vld1q_u8(data.as_ptr().add(48));
    let p4 = vld1q_u8(data.as_ptr().add(64));
    let p5 = vld1q_u8(data.as_ptr().add(80));
    let p6 = vld1q_u8(data.as_ptr().add(96));
    let p7 = vld1q_u8(data.as_ptr().add(112));
    let c0 = veorq_u8(p0, s0);
    let c1 = veorq_u8(p1, s1);
    let c2 = veorq_u8(p2, s2);
    let c3 = veorq_u8(p3, s3);
    let c4 = veorq_u8(p4, s4);
    let c5 = veorq_u8(p5, s5);
    let c6 = veorq_u8(p6, s6);
    let c7 = veorq_u8(p7, s7);
    vst1q_u8(data.as_mut_ptr(), c0);
    vst1q_u8(data.as_mut_ptr().add(16), c1);
    vst1q_u8(data.as_mut_ptr().add(32), c2);
    vst1q_u8(data.as_mut_ptr().add(48), c3);
    vst1q_u8(data.as_mut_ptr().add(64), c4);
    vst1q_u8(data.as_mut_ptr().add(80), c5);
    vst1q_u8(data.as_mut_ptr().add(96), c6);
    vst1q_u8(data.as_mut_ptr().add(112), c7);

    [c0, c1, c2, c3, c4, c5, c6, c7]
  }
}

#[cfg(feature = "aes-gcm")]
#[target_feature(enable = "aes,neon")]
#[inline]
pub(super) unsafe fn encrypt_ctr32_be_xor_8blocks_ghash_prev_bytes_core(
  keys: &CeRoundKeys,
  iv_prefix: &[u8; 12],
  ctr: u32,
  data: &mut [u8],
  acc: u128,
  h_powers_rev: &[u128; 8],
  prev_ciphertext: &[uint8x16_t; 8],
) -> (u128, [uint8x16_t; 8]) {
  debug_assert!(data.len() >= 128);

  // SAFETY: eight-block AES-256-GCM encryption with previous ciphertext lanes because:
  // 1. The caller guarantees AES-CE + PMULL availability.
  // 2. `data` has at least 128 bytes and `prev_ciphertext` is an initialized eight-lane array.
  // 3. `prev_ciphertext` is already-produced ciphertext, so folding it while encrypting this chunk
  //    preserves GCM authentication order.
  unsafe {
    let k = &keys.rk;
    let base = gcm_ctr32_base(iv_prefix);
    let mut s0 = gcm_ctr32_block(base, ctr);
    let mut s1 = gcm_ctr32_block(base, ctr.wrapping_add(1));
    let mut s2 = gcm_ctr32_block(base, ctr.wrapping_add(2));
    let mut s3 = gcm_ctr32_block(base, ctr.wrapping_add(3));
    let mut s4 = gcm_ctr32_block(base, ctr.wrapping_add(4));
    let mut s5 = gcm_ctr32_block(base, ctr.wrapping_add(5));
    let mut s6 = gcm_ctr32_block(base, ctr.wrapping_add(6));
    let mut s7 = gcm_ctr32_block(base, ctr.wrapping_add(7));

    let mut ll = vdupq_n_u64(0);
    let mut hh = vdupq_n_u64(0);
    let mut mm = vdupq_n_u64(0);
    let acc_lanes = ghash_u128_to_lanes(acc);
    let zero_lanes = vdupq_n_u64(0);

    macro_rules! round8 {
      ($round:expr) => {{
        s0 = vaesmcq_u8(vaeseq_u8(s0, k[$round]));
        s1 = vaesmcq_u8(vaeseq_u8(s1, k[$round]));
        s2 = vaesmcq_u8(vaeseq_u8(s2, k[$round]));
        s3 = vaesmcq_u8(vaeseq_u8(s3, k[$round]));
        s4 = vaesmcq_u8(vaeseq_u8(s4, k[$round]));
        s5 = vaesmcq_u8(vaeseq_u8(s5, k[$round]));
        s6 = vaesmcq_u8(vaeseq_u8(s6, k[$round]));
        s7 = vaesmcq_u8(vaeseq_u8(s7, k[$round]));
      }};
    }

    round8!(0);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    ghash_fold_be_vec!(ll, hh, mm, prev_ciphertext[0], 0, h_powers_rev, acc_lanes);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    round8!(1);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    ghash_fold_be_vec!(ll, hh, mm, prev_ciphertext[1], 1, h_powers_rev, zero_lanes);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    round8!(2);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    ghash_fold_be_vec!(ll, hh, mm, prev_ciphertext[2], 2, h_powers_rev, zero_lanes);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    round8!(3);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    ghash_fold_be_vec!(ll, hh, mm, prev_ciphertext[3], 3, h_powers_rev, zero_lanes);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    round8!(4);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    ghash_fold_be_vec!(ll, hh, mm, prev_ciphertext[4], 4, h_powers_rev, zero_lanes);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    round8!(5);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    ghash_fold_be_vec!(ll, hh, mm, prev_ciphertext[5], 5, h_powers_rev, zero_lanes);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    round8!(6);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    ghash_fold_be_vec!(ll, hh, mm, prev_ciphertext[6], 6, h_powers_rev, zero_lanes);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    round8!(7);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    ghash_fold_be_vec!(ll, hh, mm, prev_ciphertext[7], 7, h_powers_rev, zero_lanes);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    round8!(8);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    round8!(9);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    let acc = ghash_finish_products(ll, hh, mm);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    round8!(10);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    round8!(11);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    round8!(12);

    s0 = veorq_u8(vaeseq_u8(s0, k[13]), k[14]);
    s1 = veorq_u8(vaeseq_u8(s1, k[13]), k[14]);
    s2 = veorq_u8(vaeseq_u8(s2, k[13]), k[14]);
    s3 = veorq_u8(vaeseq_u8(s3, k[13]), k[14]);
    s4 = veorq_u8(vaeseq_u8(s4, k[13]), k[14]);
    s5 = veorq_u8(vaeseq_u8(s5, k[13]), k[14]);
    s6 = veorq_u8(vaeseq_u8(s6, k[13]), k[14]);
    s7 = veorq_u8(vaeseq_u8(s7, k[13]), k[14]);

    let p0 = vld1q_u8(data.as_ptr());
    let p1 = vld1q_u8(data.as_ptr().add(16));
    let p2 = vld1q_u8(data.as_ptr().add(32));
    let p3 = vld1q_u8(data.as_ptr().add(48));
    let p4 = vld1q_u8(data.as_ptr().add(64));
    let p5 = vld1q_u8(data.as_ptr().add(80));
    let p6 = vld1q_u8(data.as_ptr().add(96));
    let p7 = vld1q_u8(data.as_ptr().add(112));
    let c0 = veorq_u8(p0, s0);
    let c1 = veorq_u8(p1, s1);
    let c2 = veorq_u8(p2, s2);
    let c3 = veorq_u8(p3, s3);
    let c4 = veorq_u8(p4, s4);
    let c5 = veorq_u8(p5, s5);
    let c6 = veorq_u8(p6, s6);
    let c7 = veorq_u8(p7, s7);
    vst1q_u8(data.as_mut_ptr(), c0);
    vst1q_u8(data.as_mut_ptr().add(16), c1);
    vst1q_u8(data.as_mut_ptr().add(32), c2);
    vst1q_u8(data.as_mut_ptr().add(48), c3);
    vst1q_u8(data.as_mut_ptr().add(64), c4);
    vst1q_u8(data.as_mut_ptr().add(80), c5);
    vst1q_u8(data.as_mut_ptr().add(96), c6);
    vst1q_u8(data.as_mut_ptr().add(112), c7);

    (acc, [c0, c1, c2, c3, c4, c5, c6, c7])
  }
}

#[cfg(feature = "aes-gcm")]
#[target_feature(enable = "aes,neon")]
#[inline]
pub(super) unsafe fn decrypt_ctr32_be_xor_8blocks_ghash_current_core(
  keys: &CeRoundKeys,
  iv_prefix: &[u8; 12],
  ctr: u32,
  data: &mut [u8],
  acc: u128,
  h_powers_rev: &[u128; 8],
) -> u128 {
  debug_assert!(data.len() >= 128);

  // SAFETY: eight-block AES-256-GCM decryption with current-ciphertext GHASH because:
  // 1. The caller guarantees AES-CE + PMULL availability.
  // 2. `data` has at least 128 bytes; all fixed vector loads/stores stay in bounds.
  // 3. GHASH loads happen before plaintext stores, so authentication covers the original ciphertext
  //    bytes.
  unsafe {
    let k = &keys.rk;
    let base = gcm_ctr32_base(iv_prefix);
    let mut s0 = gcm_ctr32_block(base, ctr);
    let mut s1 = gcm_ctr32_block(base, ctr.wrapping_add(1));
    let mut s2 = gcm_ctr32_block(base, ctr.wrapping_add(2));
    let mut s3 = gcm_ctr32_block(base, ctr.wrapping_add(3));
    let mut s4 = gcm_ctr32_block(base, ctr.wrapping_add(4));
    let mut s5 = gcm_ctr32_block(base, ctr.wrapping_add(5));
    let mut s6 = gcm_ctr32_block(base, ctr.wrapping_add(6));
    let mut s7 = gcm_ctr32_block(base, ctr.wrapping_add(7));

    let mut ll = vdupq_n_u64(0);
    let mut hh = vdupq_n_u64(0);
    let mut mm = vdupq_n_u64(0);
    let acc_lanes = ghash_u128_to_lanes(acc);
    let zero_lanes = vdupq_n_u64(0);
    let data_ptr = data.as_ptr();

    macro_rules! round8 {
      ($round:expr) => {{
        s0 = vaesmcq_u8(vaeseq_u8(s0, k[$round]));
        s1 = vaesmcq_u8(vaeseq_u8(s1, k[$round]));
        s2 = vaesmcq_u8(vaeseq_u8(s2, k[$round]));
        s3 = vaesmcq_u8(vaeseq_u8(s3, k[$round]));
        s4 = vaesmcq_u8(vaeseq_u8(s4, k[$round]));
        s5 = vaesmcq_u8(vaeseq_u8(s5, k[$round]));
        s6 = vaesmcq_u8(vaeseq_u8(s6, k[$round]));
        s7 = vaesmcq_u8(vaeseq_u8(s7, k[$round]));
      }};
    }

    round8!(0);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    ghash_fold_be_vec!(
      ll,
      hh,
      mm,
      gcm_load_ciphertext_block(data_ptr),
      0,
      h_powers_rev,
      acc_lanes
    );
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    round8!(1);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    ghash_fold_be_vec!(
      ll,
      hh,
      mm,
      gcm_load_ciphertext_block(data_ptr.add(16)),
      1,
      h_powers_rev,
      zero_lanes
    );
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    round8!(2);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    ghash_fold_be_vec!(
      ll,
      hh,
      mm,
      gcm_load_ciphertext_block(data_ptr.add(32)),
      2,
      h_powers_rev,
      zero_lanes
    );
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    round8!(3);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    ghash_fold_be_vec!(
      ll,
      hh,
      mm,
      gcm_load_ciphertext_block(data_ptr.add(48)),
      3,
      h_powers_rev,
      zero_lanes
    );
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    round8!(4);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    ghash_fold_be_vec!(
      ll,
      hh,
      mm,
      gcm_load_ciphertext_block(data_ptr.add(64)),
      4,
      h_powers_rev,
      zero_lanes
    );
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    round8!(5);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    ghash_fold_be_vec!(
      ll,
      hh,
      mm,
      gcm_load_ciphertext_block(data_ptr.add(80)),
      5,
      h_powers_rev,
      zero_lanes
    );
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    round8!(6);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    ghash_fold_be_vec!(
      ll,
      hh,
      mm,
      gcm_load_ciphertext_block(data_ptr.add(96)),
      6,
      h_powers_rev,
      zero_lanes
    );
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    round8!(7);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    ghash_fold_be_vec!(
      ll,
      hh,
      mm,
      gcm_load_ciphertext_block(data_ptr.add(112)),
      7,
      h_powers_rev,
      zero_lanes
    );
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    round8!(8);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    round8!(9);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    let acc = ghash_finish_products(ll, hh, mm);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    round8!(10);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    round8!(11);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    round8!(12);

    s0 = veorq_u8(vaeseq_u8(s0, k[13]), k[14]);
    s1 = veorq_u8(vaeseq_u8(s1, k[13]), k[14]);
    s2 = veorq_u8(vaeseq_u8(s2, k[13]), k[14]);
    s3 = veorq_u8(vaeseq_u8(s3, k[13]), k[14]);
    s4 = veorq_u8(vaeseq_u8(s4, k[13]), k[14]);
    s5 = veorq_u8(vaeseq_u8(s5, k[13]), k[14]);
    s6 = veorq_u8(vaeseq_u8(s6, k[13]), k[14]);
    s7 = veorq_u8(vaeseq_u8(s7, k[13]), k[14]);

    let p0 = vld1q_u8(data.as_ptr());
    let p1 = vld1q_u8(data.as_ptr().add(16));
    let p2 = vld1q_u8(data.as_ptr().add(32));
    let p3 = vld1q_u8(data.as_ptr().add(48));
    let p4 = vld1q_u8(data.as_ptr().add(64));
    let p5 = vld1q_u8(data.as_ptr().add(80));
    let p6 = vld1q_u8(data.as_ptr().add(96));
    let p7 = vld1q_u8(data.as_ptr().add(112));
    vst1q_u8(data.as_mut_ptr(), veorq_u8(p0, s0));
    vst1q_u8(data.as_mut_ptr().add(16), veorq_u8(p1, s1));
    vst1q_u8(data.as_mut_ptr().add(32), veorq_u8(p2, s2));
    vst1q_u8(data.as_mut_ptr().add(48), veorq_u8(p3, s3));
    vst1q_u8(data.as_mut_ptr().add(64), veorq_u8(p4, s4));
    vst1q_u8(data.as_mut_ptr().add(80), veorq_u8(p5, s5));
    vst1q_u8(data.as_mut_ptr().add(96), veorq_u8(p6, s6));
    vst1q_u8(data.as_mut_ptr().add(112), veorq_u8(p7, s7));

    acc
  }
}

#[cfg(feature = "aes-gcm")]
#[target_feature(enable = "aes,neon")]
#[inline(never)]
pub(super) unsafe fn encrypt_ctr32_be_xor_ghash_128b_chunks_core(
  keys: &CeRoundKeys,
  iv_prefix: &[u8; 12],
  mut ctr: u32,
  data: &mut [u8],
  mut acc: u128,
  tables: &super::Aarch64GcmTables<'_>,
) -> (u128, u32, usize) {
  debug_assert!(data.len() >= 128);

  // SAFETY: AES-256-GCM full-chunk loop because:
  // 1. The caller guarantees AES-CE + PMULL availability.
  // 2. The macOS assembly path receives the checked slice pointer/length and reports its exact
  //    processed byte count.
  // 3. The Rust fallback only creates 128-byte slices at offsets proven in bounds by the loop guards.
  // 4. Previous ciphertext is carried as initialized NEON lanes between Rust fallback iterations.
  // 5. The final pending lane aggregate folds the last encrypted 128-byte fallback chunk exactly
  //    once.
  unsafe {
    #[cfg(target_os = "macos")]
    {
      if crate::platform::caps().has(crate::platform::caps::aarch64::PMULL_EOR3_READY) {
        let mut state = asm::AesGcmAarch64State::new(acc, ctr);
        asm::rscrypto_aes256_gcm_seal_16x_eor3_aarch64_apple_darwin(
          keys.rk.as_ptr().cast::<u8>(),
          iv_prefix.as_ptr(),
          data.as_mut_ptr(),
          data.len(),
          tables.h_powers_rev_16.as_ptr(),
          tables.h_powers_rev_16_mid.as_ptr(),
          tables.h_powers_rev_16_pair.as_ptr(),
          &mut state,
        );
        if state.processed != 0 {
          return (state.acc(), state.ctr, state.processed);
        }
      }

      let mut state = asm::AesGcmAarch64State::new(acc, ctr);
      asm::rscrypto_aes256_gcm_seal_8x_aarch64_apple_darwin(
        keys.rk.as_ptr().cast::<u8>(),
        iv_prefix.as_ptr(),
        data.as_mut_ptr(),
        data.len(),
        tables.h_powers_rev_8.as_ptr(),
        &mut state,
      );
      if state.processed != 0 {
        return (state.acc(), state.ctr, state.processed);
      }
    }

    let data_ptr = data.as_mut_ptr();
    let mut offset = 0usize;

    let first = core::slice::from_raw_parts_mut(data_ptr, 128);
    let mut prev_blocks = encrypt_ctr32_be_xor_8blocks_core(keys, iv_prefix, ctr, first);
    ctr = ctr.wrapping_add(8);
    offset = offset.strict_add(128);

    while offset.strict_add(128) <= data.len() {
      let current = core::slice::from_raw_parts_mut(data_ptr.add(offset), 128);
      let (next_acc, next_blocks) = encrypt_ctr32_be_xor_8blocks_ghash_prev_bytes_core(
        keys,
        iv_prefix,
        ctr,
        current,
        acc,
        tables.h_powers_rev_8,
        &prev_blocks,
      );
      acc = next_acc;
      prev_blocks = next_blocks;
      ctr = ctr.wrapping_add(8);
      offset = offset.strict_add(128);
    }

    acc = super::super::polyval::aarch64_aggregate_8blocks_be_lanes_inline(acc, tables.h_powers_rev_8, &prev_blocks);

    (acc, ctr, offset)
  }
}

#[cfg(feature = "aes-gcm")]
#[target_feature(enable = "aes,neon")]
#[inline(never)]
pub(super) unsafe fn decrypt_ctr32_be_xor_ghash_128b_chunks_core(
  keys: &CeRoundKeys,
  iv_prefix: &[u8; 12],
  mut ctr: u32,
  data: &mut [u8],
  mut acc: u128,
  tables: &super::Aarch64GcmTables<'_>,
) -> (u128, u32, usize) {
  // SAFETY: AES-256-GCM full-chunk open loop because:
  // 1. The caller guarantees AES-CE + PMULL availability.
  // 2. The macOS assembly path receives the checked slice pointer/length and reports its exact
  //    processed byte count.
  // 3. The Rust fallback creates each 128-byte mutable chunk only after proving it is in bounds.
  // 4. Both paths fold ciphertext before plaintext stores.
  unsafe {
    #[cfg(target_os = "macos")]
    {
      if crate::platform::caps().has(crate::platform::caps::aarch64::PMULL_EOR3_READY) {
        let mut state = asm::AesGcmAarch64State::new(acc, ctr);
        asm::rscrypto_aes256_gcm_open_16x_eor3_aarch64_apple_darwin(
          keys.rk.as_ptr().cast::<u8>(),
          iv_prefix.as_ptr(),
          data.as_mut_ptr(),
          data.len(),
          tables.h_powers_rev_16.as_ptr(),
          tables.h_powers_rev_16_mid.as_ptr(),
          tables.h_powers_rev_16_pair.as_ptr(),
          &mut state,
        );
        if state.processed != 0 {
          return (state.acc(), state.ctr, state.processed);
        }
      }

      let mut state = asm::AesGcmAarch64State::new(acc, ctr);
      asm::rscrypto_aes256_gcm_open_8x_aarch64_apple_darwin(
        keys.rk.as_ptr().cast::<u8>(),
        iv_prefix.as_ptr(),
        data.as_mut_ptr(),
        data.len(),
        tables.h_powers_rev_8.as_ptr(),
        &mut state,
      );
      if state.processed != 0 {
        return (state.acc(), state.ctr, state.processed);
      }
    }

    let data_ptr = data.as_mut_ptr();
    let mut offset = 0usize;

    while offset.strict_add(128) <= data.len() {
      let current = core::slice::from_raw_parts_mut(data_ptr.add(offset), 128);
      acc = decrypt_ctr32_be_xor_8blocks_ghash_current_core(keys, iv_prefix, ctr, current, acc, tables.h_powers_rev_8);
      ctr = ctr.wrapping_add(8);
      offset = offset.strict_add(128);
    }

    (acc, ctr, offset)
  }
}

/// Encrypt four AES-256 GCM-SIV counter blocks and XOR them into `data`.
///
/// # Safety
///
/// Caller must ensure the CPU supports `aes` and `neon`, `data` has at least
/// 64 writable bytes, and `iv_suffix` is bytes 4..16 of the SIV counter block.
#[cfg(feature = "aes-gcm-siv")]
#[target_feature(enable = "aes,neon")]
#[inline]
pub(super) unsafe fn encrypt_ctr32_le_xor_4blocks_core(
  keys: &CeRoundKeys,
  iv_suffix: &[u8; 12],
  ctr: u32,
  data: &mut [u8],
) {
  debug_assert!(data.len() >= 64);

  // SAFETY: four-block AES-GCM-SIV CTR processing because:
  // 1. This function's caller guarantees AES-CE + NEON availability.
  // 2. `data` has at least 64 writable bytes; all loads/stores use fixed 16-byte offsets within that
  //    range.
  // 3. Counter blocks are initialized before loading into vector registers.
  unsafe {
    let k = &keys.rk;
    let base = gcmsiv_ctr32_base(iv_suffix);
    let mut s0 = gcmsiv_ctr32_block(base, ctr);
    let mut s1 = gcmsiv_ctr32_block(base, ctr.wrapping_add(1));
    let mut s2 = gcmsiv_ctr32_block(base, ctr.wrapping_add(2));
    let mut s3 = gcmsiv_ctr32_block(base, ctr.wrapping_add(3));

    macro_rules! round4 {
      ($round:expr) => {{
        s0 = vaesmcq_u8(vaeseq_u8(s0, k[$round]));
        s1 = vaesmcq_u8(vaeseq_u8(s1, k[$round]));
        s2 = vaesmcq_u8(vaeseq_u8(s2, k[$round]));
        s3 = vaesmcq_u8(vaeseq_u8(s3, k[$round]));
      }};
    }

    round4!(0);
    round4!(1);
    round4!(2);
    round4!(3);
    round4!(4);
    round4!(5);
    round4!(6);
    round4!(7);
    round4!(8);
    round4!(9);
    round4!(10);
    round4!(11);
    round4!(12);

    s0 = veorq_u8(vaeseq_u8(s0, k[13]), k[14]);
    s1 = veorq_u8(vaeseq_u8(s1, k[13]), k[14]);
    s2 = veorq_u8(vaeseq_u8(s2, k[13]), k[14]);
    s3 = veorq_u8(vaeseq_u8(s3, k[13]), k[14]);

    let p0 = vld1q_u8(data.as_ptr());
    let p1 = vld1q_u8(data.as_ptr().add(16));
    let p2 = vld1q_u8(data.as_ptr().add(32));
    let p3 = vld1q_u8(data.as_ptr().add(48));
    vst1q_u8(data.as_mut_ptr(), veorq_u8(p0, s0));
    vst1q_u8(data.as_mut_ptr().add(16), veorq_u8(p1, s1));
    vst1q_u8(data.as_mut_ptr().add(32), veorq_u8(p2, s2));
    vst1q_u8(data.as_mut_ptr().add(48), veorq_u8(p3, s3));
  }
}

/// Encrypt eight AES-256 GCM-SIV counter blocks and XOR them into `data`.
///
/// # Safety
///
/// Caller must ensure the CPU supports `aes` and `neon`, `data` has at least
/// 128 writable bytes, and `iv_suffix` is bytes 4..16 of the SIV counter block.
#[cfg(feature = "aes-gcm-siv")]
#[target_feature(enable = "aes,neon")]
#[inline]
pub(super) unsafe fn encrypt_ctr32_le_xor_8blocks_core(
  keys: &CeRoundKeys,
  iv_suffix: &[u8; 12],
  ctr: u32,
  data: &mut [u8],
) {
  debug_assert!(data.len() >= 128);

  // SAFETY: eight-block AES-GCM-SIV CTR processing because:
  // 1. This function's caller guarantees AES-CE + NEON availability.
  // 2. `data` has at least 128 writable bytes; all loads/stores use fixed 16-byte offsets within that
  //    range.
  // 3. Counter blocks are initialized before loading into vector registers.
  unsafe {
    let k = &keys.rk;
    let base = gcmsiv_ctr32_base(iv_suffix);
    let mut s0 = gcmsiv_ctr32_block(base, ctr);
    let mut s1 = gcmsiv_ctr32_block(base, ctr.wrapping_add(1));
    let mut s2 = gcmsiv_ctr32_block(base, ctr.wrapping_add(2));
    let mut s3 = gcmsiv_ctr32_block(base, ctr.wrapping_add(3));
    let mut s4 = gcmsiv_ctr32_block(base, ctr.wrapping_add(4));
    let mut s5 = gcmsiv_ctr32_block(base, ctr.wrapping_add(5));
    let mut s6 = gcmsiv_ctr32_block(base, ctr.wrapping_add(6));
    let mut s7 = gcmsiv_ctr32_block(base, ctr.wrapping_add(7));

    macro_rules! round8 {
      ($round:expr) => {{
        s0 = vaesmcq_u8(vaeseq_u8(s0, k[$round]));
        s1 = vaesmcq_u8(vaeseq_u8(s1, k[$round]));
        s2 = vaesmcq_u8(vaeseq_u8(s2, k[$round]));
        s3 = vaesmcq_u8(vaeseq_u8(s3, k[$round]));
        s4 = vaesmcq_u8(vaeseq_u8(s4, k[$round]));
        s5 = vaesmcq_u8(vaeseq_u8(s5, k[$round]));
        s6 = vaesmcq_u8(vaeseq_u8(s6, k[$round]));
        s7 = vaesmcq_u8(vaeseq_u8(s7, k[$round]));
      }};
    }

    round8!(0);
    round8!(1);
    round8!(2);
    round8!(3);
    round8!(4);
    round8!(5);
    round8!(6);
    round8!(7);
    round8!(8);
    round8!(9);
    round8!(10);
    round8!(11);
    round8!(12);

    s0 = veorq_u8(vaeseq_u8(s0, k[13]), k[14]);
    s1 = veorq_u8(vaeseq_u8(s1, k[13]), k[14]);
    s2 = veorq_u8(vaeseq_u8(s2, k[13]), k[14]);
    s3 = veorq_u8(vaeseq_u8(s3, k[13]), k[14]);
    s4 = veorq_u8(vaeseq_u8(s4, k[13]), k[14]);
    s5 = veorq_u8(vaeseq_u8(s5, k[13]), k[14]);
    s6 = veorq_u8(vaeseq_u8(s6, k[13]), k[14]);
    s7 = veorq_u8(vaeseq_u8(s7, k[13]), k[14]);

    let p0 = vld1q_u8(data.as_ptr());
    let p1 = vld1q_u8(data.as_ptr().add(16));
    let p2 = vld1q_u8(data.as_ptr().add(32));
    let p3 = vld1q_u8(data.as_ptr().add(48));
    let p4 = vld1q_u8(data.as_ptr().add(64));
    let p5 = vld1q_u8(data.as_ptr().add(80));
    let p6 = vld1q_u8(data.as_ptr().add(96));
    let p7 = vld1q_u8(data.as_ptr().add(112));
    vst1q_u8(data.as_mut_ptr(), veorq_u8(p0, s0));
    vst1q_u8(data.as_mut_ptr().add(16), veorq_u8(p1, s1));
    vst1q_u8(data.as_mut_ptr().add(32), veorq_u8(p2, s2));
    vst1q_u8(data.as_mut_ptr().add(48), veorq_u8(p3, s3));
    vst1q_u8(data.as_mut_ptr().add(64), veorq_u8(p4, s4));
    vst1q_u8(data.as_mut_ptr().add(80), veorq_u8(p5, s5));
    vst1q_u8(data.as_mut_ptr().add(96), veorq_u8(p6, s6));
    vst1q_u8(data.as_mut_ptr().add(112), veorq_u8(p7, s7));
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

// ---------------------------------------------------------------------------
// AES-128 (11 round keys, 10 rounds)
// ---------------------------------------------------------------------------

/// AES-128 round keys stored as 11 × 128-bit NEON vectors for AES-CE.
///
/// Visibility mirrors [`CeRoundKeys`]: `pub(in crate::aead)` so the fused
/// AES-128-GCM-SIV path in `aes128gcmsiv.rs` can hold these round keys
/// across the `#[target_feature(enable = "aes,neon")]` scope via
/// `aes::aarch64_expand_key_128_inline` / `aes::aarch64_encrypt_block_128_inline`.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub(in crate::aead) struct Ce128RoundKeys {
  rk: [uint8x16_t; 11],
}

impl Ce128RoundKeys {
  /// Zeroize all round keys via volatile writes.
  pub(super) fn zeroize(&mut self) {
    // SAFETY: `self.rk` is a valid, aligned, fully-initialized [uint8x16_t; 11].
    let bytes = unsafe { core::slice::from_raw_parts_mut(self.rk.as_mut_ptr().cast::<u8>(), 11usize.strict_mul(16)) };
    crate::traits::ct::zeroize(bytes);
  }
}

/// Hardware-accelerated AES-128 key expansion using AESE for SubWord.
#[target_feature(enable = "aes,neon")]
#[inline]
/// # Safety
///
/// Caller must ensure the CPU supports `aes` and `neon`.
unsafe fn expand_key_128_hw(key: &[u8; 16]) -> Ce128RoundKeys {
  // SAFETY: AES-128 key expansion with AES-CE because:
  // 1. This function's caller guarantees AES-CE + NEON availability.
  // 2. `key` is exactly 16 initialized bytes and the vector load is in bounds.
  // 3. The generated schedule fills all 11 AES-128 round keys before returning.
  unsafe {
    let zero = vdupq_n_u8(0);
    let rot_mask_bytes = [13u8, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12];
    let rot_mask = vld1q_u8(rot_mask_bytes.as_ptr());
    let rcon = [0x01u32, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36];

    let mut keys = [vdupq_n_u8(0); 11];
    let mut state = vld1q_u8(key.as_ptr());
    keys[0] = state;

    let mut round = 0usize;
    while round < 10 {
      let mut sub = vqtbl1q_u8(state, rot_mask);
      sub = vaeseq_u8(sub, zero);
      sub = veorq_u8(sub, vreinterpretq_u8_u32(vdupq_n_u32(rcon[round])));

      let mut shifted = vextq_u8(zero, state, 12);
      state = veorq_u8(state, shifted);
      shifted = vextq_u8(zero, shifted, 12);
      state = veorq_u8(state, shifted);
      shifted = vextq_u8(zero, shifted, 12);
      state = veorq_u8(state, shifted);
      state = veorq_u8(state, sub);

      keys[round.strict_add(1)] = state;
      round = round.strict_add(1);
    }

    Ce128RoundKeys { rk: keys }
  }
}

/// Non-inline entry point for AES-128 hardware key expansion.
#[target_feature(enable = "aes,neon")]
/// # Safety
///
/// Caller must ensure the CPU supports `aes` and `neon`.
pub(super) unsafe fn expand_key_128(key: &[u8; 16]) -> Ce128RoundKeys {
  // SAFETY: AES-128 hardware key expansion dispatch because:
  // 1. This function has `#[target_feature(enable = "aes,neon")]`.
  // 2. `key` is exactly 16 initialized bytes.
  unsafe { expand_key_128_hw(key) }
}

#[target_feature(enable = "aes,neon")]
#[inline]
/// # Safety
///
/// Caller must ensure the CPU supports `aes` and `neon`, and `block` points
/// to a valid writable AES block.
pub(super) unsafe fn encrypt_block_128_core(keys: &Ce128RoundKeys, block: &mut [u8; 16]) {
  // SAFETY: AES-CE AES-128 single-block encryption because:
  // 1. This function has `#[target_feature(enable = "aes,neon")]`.
  // 2. The caller guarantees the current CPU supports AES-CE + NEON.
  // 3. `block` is exactly one writable initialized AES block.
  unsafe {
    let k = &keys.rk;
    let mut state = vld1q_u8(block.as_ptr());

    // Rounds 1–9: AESE absorbs the previous round's AddRoundKey,
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

    // Round 10 (final): SubBytes + ShiftRows, then AddRoundKey (no MixColumns).
    state = vaeseq_u8(state, k[9]);
    state = veorq_u8(state, k[10]);

    vst1q_u8(block.as_mut_ptr(), state);
  }
}

#[target_feature(enable = "aes,neon")]
#[inline]
/// # Safety
///
/// Caller must ensure the CPU supports `aes` and `neon`.
pub(super) unsafe fn encrypt_4blocks_128_core(keys: &Ce128RoundKeys, blocks: &mut [[u8; 16]; 4]) {
  // SAFETY: four-block AES-128 ECB processing because:
  // 1. This function's caller guarantees AES-CE + NEON availability.
  // 2. `blocks` is a valid mutable reference to exactly four initialized AES blocks.
  // 3. All vector loads and stores use fixed in-bounds block references.
  unsafe {
    let k = &keys.rk;
    let mut s0 = vld1q_u8(blocks[0].as_ptr());
    let mut s1 = vld1q_u8(blocks[1].as_ptr());
    let mut s2 = vld1q_u8(blocks[2].as_ptr());
    let mut s3 = vld1q_u8(blocks[3].as_ptr());

    macro_rules! round4 {
      ($round:expr) => {{
        s0 = vaesmcq_u8(vaeseq_u8(s0, k[$round]));
        s1 = vaesmcq_u8(vaeseq_u8(s1, k[$round]));
        s2 = vaesmcq_u8(vaeseq_u8(s2, k[$round]));
        s3 = vaesmcq_u8(vaeseq_u8(s3, k[$round]));
      }};
    }

    round4!(0);
    round4!(1);
    round4!(2);
    round4!(3);
    round4!(4);
    round4!(5);
    round4!(6);
    round4!(7);
    round4!(8);

    s0 = veorq_u8(vaeseq_u8(s0, k[9]), k[10]);
    s1 = veorq_u8(vaeseq_u8(s1, k[9]), k[10]);
    s2 = veorq_u8(vaeseq_u8(s2, k[9]), k[10]);
    s3 = veorq_u8(vaeseq_u8(s3, k[9]), k[10]);

    vst1q_u8(blocks[0].as_mut_ptr(), s0);
    vst1q_u8(blocks[1].as_mut_ptr(), s1);
    vst1q_u8(blocks[2].as_mut_ptr(), s2);
    vst1q_u8(blocks[3].as_mut_ptr(), s3);
  }
}

/// Encrypt four AES-128 GCM counter blocks, XOR them into `data`, and return
/// the resulting ciphertext lanes as `u128::from_be_bytes` GHASH inputs.
///
/// # Safety
///
/// Caller must ensure the CPU supports `aes` and `neon`, `data` has at least
/// 64 writable bytes, and `iv_prefix` is the fixed 96-bit GCM nonce prefix.
#[cfg(feature = "aes-gcm")]
#[target_feature(enable = "aes,neon")]
#[inline]
pub(super) unsafe fn encrypt_ctr32_be_xor_4blocks_128_core(
  keys: &Ce128RoundKeys,
  iv_prefix: &[u8; 12],
  ctr: u32,
  data: &mut [u8],
) -> [u128; 4] {
  debug_assert!(data.len() >= 64);

  // SAFETY: four-block AES-GCM CTR processing because:
  // 1. This function's caller guarantees AES-CE + NEON availability.
  // 2. `data` has at least 64 writable bytes; all loads/stores use fixed 16-byte offsets within that
  //    range.
  // 3. Counter blocks are initialized before loading into vector registers.
  unsafe {
    let k = &keys.rk;
    let base = gcm_ctr32_base(iv_prefix);
    let mut s0 = gcm_ctr32_block(base, ctr);
    let mut s1 = gcm_ctr32_block(base, ctr.wrapping_add(1));
    let mut s2 = gcm_ctr32_block(base, ctr.wrapping_add(2));
    let mut s3 = gcm_ctr32_block(base, ctr.wrapping_add(3));

    macro_rules! round4 {
      ($round:expr) => {{
        s0 = vaesmcq_u8(vaeseq_u8(s0, k[$round]));
        s1 = vaesmcq_u8(vaeseq_u8(s1, k[$round]));
        s2 = vaesmcq_u8(vaeseq_u8(s2, k[$round]));
        s3 = vaesmcq_u8(vaeseq_u8(s3, k[$round]));
      }};
    }

    round4!(0);
    round4!(1);
    round4!(2);
    round4!(3);
    round4!(4);
    round4!(5);
    round4!(6);
    round4!(7);
    round4!(8);

    s0 = veorq_u8(vaeseq_u8(s0, k[9]), k[10]);
    s1 = veorq_u8(vaeseq_u8(s1, k[9]), k[10]);
    s2 = veorq_u8(vaeseq_u8(s2, k[9]), k[10]);
    s3 = veorq_u8(vaeseq_u8(s3, k[9]), k[10]);

    let p0 = vld1q_u8(data.as_ptr());
    let p1 = vld1q_u8(data.as_ptr().add(16));
    let p2 = vld1q_u8(data.as_ptr().add(32));
    let p3 = vld1q_u8(data.as_ptr().add(48));
    let c0 = veorq_u8(p0, s0);
    let c1 = veorq_u8(p1, s1);
    let c2 = veorq_u8(p2, s2);
    let c3 = veorq_u8(p3, s3);
    vst1q_u8(data.as_mut_ptr(), c0);
    vst1q_u8(data.as_mut_ptr().add(16), c1);
    vst1q_u8(data.as_mut_ptr().add(32), c2);
    vst1q_u8(data.as_mut_ptr().add(48), c3);

    [
      ghash_be_u128_from_vec(c0),
      ghash_be_u128_from_vec(c1),
      ghash_be_u128_from_vec(c2),
      ghash_be_u128_from_vec(c3),
    ]
  }
}

/// Encrypt eight AES-128 GCM counter blocks, XOR them into `data`, and return
/// the resulting ciphertext lanes in memory byte order.
///
/// # Safety
///
/// Caller must ensure the CPU supports `aes` and `neon`, `data` has at least
/// 128 writable bytes, and `iv_prefix` is the fixed 96-bit GCM nonce prefix.
#[cfg(feature = "aes-gcm")]
#[target_feature(enable = "aes,neon")]
#[inline]
pub(super) unsafe fn encrypt_ctr32_be_xor_8blocks_128_core(
  keys: &Ce128RoundKeys,
  iv_prefix: &[u8; 12],
  ctr: u32,
  data: &mut [u8],
) -> [uint8x16_t; 8] {
  debug_assert!(data.len() >= 128);

  // SAFETY: eight-block AES-GCM CTR processing because:
  // 1. This function's caller guarantees AES-CE + NEON availability.
  // 2. `data` has at least 128 writable bytes; all loads/stores use fixed 16-byte offsets within that
  //    range.
  // 3. Counter blocks are initialized before loading into vector registers.
  unsafe {
    let k = &keys.rk;
    let base = gcm_ctr32_base(iv_prefix);
    let mut s0 = gcm_ctr32_block(base, ctr);
    let mut s1 = gcm_ctr32_block(base, ctr.wrapping_add(1));
    let mut s2 = gcm_ctr32_block(base, ctr.wrapping_add(2));
    let mut s3 = gcm_ctr32_block(base, ctr.wrapping_add(3));
    let mut s4 = gcm_ctr32_block(base, ctr.wrapping_add(4));
    let mut s5 = gcm_ctr32_block(base, ctr.wrapping_add(5));
    let mut s6 = gcm_ctr32_block(base, ctr.wrapping_add(6));
    let mut s7 = gcm_ctr32_block(base, ctr.wrapping_add(7));

    macro_rules! round8 {
      ($round:expr) => {{
        s0 = vaesmcq_u8(vaeseq_u8(s0, k[$round]));
        s1 = vaesmcq_u8(vaeseq_u8(s1, k[$round]));
        s2 = vaesmcq_u8(vaeseq_u8(s2, k[$round]));
        s3 = vaesmcq_u8(vaeseq_u8(s3, k[$round]));
        s4 = vaesmcq_u8(vaeseq_u8(s4, k[$round]));
        s5 = vaesmcq_u8(vaeseq_u8(s5, k[$round]));
        s6 = vaesmcq_u8(vaeseq_u8(s6, k[$round]));
        s7 = vaesmcq_u8(vaeseq_u8(s7, k[$round]));
      }};
    }

    round8!(0);
    round8!(1);
    round8!(2);
    round8!(3);
    round8!(4);
    round8!(5);
    round8!(6);
    round8!(7);
    round8!(8);

    s0 = veorq_u8(vaeseq_u8(s0, k[9]), k[10]);
    s1 = veorq_u8(vaeseq_u8(s1, k[9]), k[10]);
    s2 = veorq_u8(vaeseq_u8(s2, k[9]), k[10]);
    s3 = veorq_u8(vaeseq_u8(s3, k[9]), k[10]);
    s4 = veorq_u8(vaeseq_u8(s4, k[9]), k[10]);
    s5 = veorq_u8(vaeseq_u8(s5, k[9]), k[10]);
    s6 = veorq_u8(vaeseq_u8(s6, k[9]), k[10]);
    s7 = veorq_u8(vaeseq_u8(s7, k[9]), k[10]);

    let p0 = vld1q_u8(data.as_ptr());
    let p1 = vld1q_u8(data.as_ptr().add(16));
    let p2 = vld1q_u8(data.as_ptr().add(32));
    let p3 = vld1q_u8(data.as_ptr().add(48));
    let p4 = vld1q_u8(data.as_ptr().add(64));
    let p5 = vld1q_u8(data.as_ptr().add(80));
    let p6 = vld1q_u8(data.as_ptr().add(96));
    let p7 = vld1q_u8(data.as_ptr().add(112));
    let c0 = veorq_u8(p0, s0);
    let c1 = veorq_u8(p1, s1);
    let c2 = veorq_u8(p2, s2);
    let c3 = veorq_u8(p3, s3);
    let c4 = veorq_u8(p4, s4);
    let c5 = veorq_u8(p5, s5);
    let c6 = veorq_u8(p6, s6);
    let c7 = veorq_u8(p7, s7);
    vst1q_u8(data.as_mut_ptr(), c0);
    vst1q_u8(data.as_mut_ptr().add(16), c1);
    vst1q_u8(data.as_mut_ptr().add(32), c2);
    vst1q_u8(data.as_mut_ptr().add(48), c3);
    vst1q_u8(data.as_mut_ptr().add(64), c4);
    vst1q_u8(data.as_mut_ptr().add(80), c5);
    vst1q_u8(data.as_mut_ptr().add(96), c6);
    vst1q_u8(data.as_mut_ptr().add(112), c7);

    [c0, c1, c2, c3, c4, c5, c6, c7]
  }
}

#[cfg(feature = "aes-gcm")]
#[target_feature(enable = "aes,neon")]
#[inline]
pub(super) unsafe fn encrypt_ctr32_be_xor_8blocks_ghash_prev_bytes_128_core(
  keys: &Ce128RoundKeys,
  iv_prefix: &[u8; 12],
  ctr: u32,
  data: &mut [u8],
  acc: u128,
  h_powers_rev: &[u128; 8],
  prev_ciphertext: &[uint8x16_t; 8],
) -> (u128, [uint8x16_t; 8]) {
  debug_assert!(data.len() >= 128);

  // SAFETY: eight-block AES-128-GCM encryption with previous ciphertext lanes because:
  // 1. The caller guarantees AES-CE + PMULL availability.
  // 2. `data` has at least 128 bytes and `prev_ciphertext` is an initialized eight-lane array.
  // 3. `prev_ciphertext` is already-produced ciphertext, so folding it while encrypting this chunk
  //    preserves GCM authentication order.
  unsafe {
    let k = &keys.rk;
    let base = gcm_ctr32_base(iv_prefix);
    let mut s0 = gcm_ctr32_block(base, ctr);
    let mut s1 = gcm_ctr32_block(base, ctr.wrapping_add(1));
    let mut s2 = gcm_ctr32_block(base, ctr.wrapping_add(2));
    let mut s3 = gcm_ctr32_block(base, ctr.wrapping_add(3));
    let mut s4 = gcm_ctr32_block(base, ctr.wrapping_add(4));
    let mut s5 = gcm_ctr32_block(base, ctr.wrapping_add(5));
    let mut s6 = gcm_ctr32_block(base, ctr.wrapping_add(6));
    let mut s7 = gcm_ctr32_block(base, ctr.wrapping_add(7));

    let mut ll = vdupq_n_u64(0);
    let mut hh = vdupq_n_u64(0);
    let mut mm = vdupq_n_u64(0);
    let acc_lanes = ghash_u128_to_lanes(acc);
    let zero_lanes = vdupq_n_u64(0);

    macro_rules! round8 {
      ($round:expr) => {{
        s0 = vaesmcq_u8(vaeseq_u8(s0, k[$round]));
        s1 = vaesmcq_u8(vaeseq_u8(s1, k[$round]));
        s2 = vaesmcq_u8(vaeseq_u8(s2, k[$round]));
        s3 = vaesmcq_u8(vaeseq_u8(s3, k[$round]));
        s4 = vaesmcq_u8(vaeseq_u8(s4, k[$round]));
        s5 = vaesmcq_u8(vaeseq_u8(s5, k[$round]));
        s6 = vaesmcq_u8(vaeseq_u8(s6, k[$round]));
        s7 = vaesmcq_u8(vaeseq_u8(s7, k[$round]));
      }};
    }

    round8!(0);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    ghash_fold_be_vec!(ll, hh, mm, prev_ciphertext[0], 0, h_powers_rev, acc_lanes);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    round8!(1);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    ghash_fold_be_vec!(ll, hh, mm, prev_ciphertext[1], 1, h_powers_rev, zero_lanes);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    round8!(2);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    ghash_fold_be_vec!(ll, hh, mm, prev_ciphertext[2], 2, h_powers_rev, zero_lanes);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    round8!(3);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    ghash_fold_be_vec!(ll, hh, mm, prev_ciphertext[3], 3, h_powers_rev, zero_lanes);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    round8!(4);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    ghash_fold_be_vec!(ll, hh, mm, prev_ciphertext[4], 4, h_powers_rev, zero_lanes);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    round8!(5);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    ghash_fold_be_vec!(ll, hh, mm, prev_ciphertext[5], 5, h_powers_rev, zero_lanes);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    round8!(6);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    ghash_fold_be_vec!(ll, hh, mm, prev_ciphertext[6], 6, h_powers_rev, zero_lanes);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    round8!(7);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    ghash_fold_be_vec!(ll, hh, mm, prev_ciphertext[7], 7, h_powers_rev, zero_lanes);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    round8!(8);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    let acc = ghash_finish_products(ll, hh, mm);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);

    s0 = veorq_u8(vaeseq_u8(s0, k[9]), k[10]);
    s1 = veorq_u8(vaeseq_u8(s1, k[9]), k[10]);
    s2 = veorq_u8(vaeseq_u8(s2, k[9]), k[10]);
    s3 = veorq_u8(vaeseq_u8(s3, k[9]), k[10]);
    s4 = veorq_u8(vaeseq_u8(s4, k[9]), k[10]);
    s5 = veorq_u8(vaeseq_u8(s5, k[9]), k[10]);
    s6 = veorq_u8(vaeseq_u8(s6, k[9]), k[10]);
    s7 = veorq_u8(vaeseq_u8(s7, k[9]), k[10]);

    let p0 = vld1q_u8(data.as_ptr());
    let p1 = vld1q_u8(data.as_ptr().add(16));
    let p2 = vld1q_u8(data.as_ptr().add(32));
    let p3 = vld1q_u8(data.as_ptr().add(48));
    let p4 = vld1q_u8(data.as_ptr().add(64));
    let p5 = vld1q_u8(data.as_ptr().add(80));
    let p6 = vld1q_u8(data.as_ptr().add(96));
    let p7 = vld1q_u8(data.as_ptr().add(112));
    let c0 = veorq_u8(p0, s0);
    let c1 = veorq_u8(p1, s1);
    let c2 = veorq_u8(p2, s2);
    let c3 = veorq_u8(p3, s3);
    let c4 = veorq_u8(p4, s4);
    let c5 = veorq_u8(p5, s5);
    let c6 = veorq_u8(p6, s6);
    let c7 = veorq_u8(p7, s7);
    vst1q_u8(data.as_mut_ptr(), c0);
    vst1q_u8(data.as_mut_ptr().add(16), c1);
    vst1q_u8(data.as_mut_ptr().add(32), c2);
    vst1q_u8(data.as_mut_ptr().add(48), c3);
    vst1q_u8(data.as_mut_ptr().add(64), c4);
    vst1q_u8(data.as_mut_ptr().add(80), c5);
    vst1q_u8(data.as_mut_ptr().add(96), c6);
    vst1q_u8(data.as_mut_ptr().add(112), c7);

    (acc, [c0, c1, c2, c3, c4, c5, c6, c7])
  }
}

#[cfg(feature = "aes-gcm")]
#[target_feature(enable = "aes,neon")]
#[inline]
pub(super) unsafe fn decrypt_ctr32_be_xor_8blocks_ghash_current_128_core(
  keys: &Ce128RoundKeys,
  iv_prefix: &[u8; 12],
  ctr: u32,
  data: &mut [u8],
  acc: u128,
  h_powers_rev: &[u128; 8],
) -> u128 {
  debug_assert!(data.len() >= 128);

  // SAFETY: eight-block AES-128-GCM decryption with current-ciphertext GHASH because:
  // 1. The caller guarantees AES-CE + PMULL availability.
  // 2. `data` has at least 128 bytes; all fixed vector loads/stores stay in bounds.
  // 3. GHASH loads happen before plaintext stores, so authentication covers the original ciphertext
  //    bytes.
  unsafe {
    let k = &keys.rk;
    let base = gcm_ctr32_base(iv_prefix);
    let mut s0 = gcm_ctr32_block(base, ctr);
    let mut s1 = gcm_ctr32_block(base, ctr.wrapping_add(1));
    let mut s2 = gcm_ctr32_block(base, ctr.wrapping_add(2));
    let mut s3 = gcm_ctr32_block(base, ctr.wrapping_add(3));
    let mut s4 = gcm_ctr32_block(base, ctr.wrapping_add(4));
    let mut s5 = gcm_ctr32_block(base, ctr.wrapping_add(5));
    let mut s6 = gcm_ctr32_block(base, ctr.wrapping_add(6));
    let mut s7 = gcm_ctr32_block(base, ctr.wrapping_add(7));

    let mut ll = vdupq_n_u64(0);
    let mut hh = vdupq_n_u64(0);
    let mut mm = vdupq_n_u64(0);
    let acc_lanes = ghash_u128_to_lanes(acc);
    let zero_lanes = vdupq_n_u64(0);
    let data_ptr = data.as_ptr();

    macro_rules! round8 {
      ($round:expr) => {{
        s0 = vaesmcq_u8(vaeseq_u8(s0, k[$round]));
        s1 = vaesmcq_u8(vaeseq_u8(s1, k[$round]));
        s2 = vaesmcq_u8(vaeseq_u8(s2, k[$round]));
        s3 = vaesmcq_u8(vaeseq_u8(s3, k[$round]));
        s4 = vaesmcq_u8(vaeseq_u8(s4, k[$round]));
        s5 = vaesmcq_u8(vaeseq_u8(s5, k[$round]));
        s6 = vaesmcq_u8(vaeseq_u8(s6, k[$round]));
        s7 = vaesmcq_u8(vaeseq_u8(s7, k[$round]));
      }};
    }

    round8!(0);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    ghash_fold_be_vec!(
      ll,
      hh,
      mm,
      gcm_load_ciphertext_block(data_ptr),
      0,
      h_powers_rev,
      acc_lanes
    );
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    round8!(1);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    ghash_fold_be_vec!(
      ll,
      hh,
      mm,
      gcm_load_ciphertext_block(data_ptr.add(16)),
      1,
      h_powers_rev,
      zero_lanes
    );
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    round8!(2);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    ghash_fold_be_vec!(
      ll,
      hh,
      mm,
      gcm_load_ciphertext_block(data_ptr.add(32)),
      2,
      h_powers_rev,
      zero_lanes
    );
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    round8!(3);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    ghash_fold_be_vec!(
      ll,
      hh,
      mm,
      gcm_load_ciphertext_block(data_ptr.add(48)),
      3,
      h_powers_rev,
      zero_lanes
    );
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    round8!(4);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    ghash_fold_be_vec!(
      ll,
      hh,
      mm,
      gcm_load_ciphertext_block(data_ptr.add(64)),
      4,
      h_powers_rev,
      zero_lanes
    );
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    round8!(5);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    ghash_fold_be_vec!(
      ll,
      hh,
      mm,
      gcm_load_ciphertext_block(data_ptr.add(80)),
      5,
      h_powers_rev,
      zero_lanes
    );
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    round8!(6);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    ghash_fold_be_vec!(
      ll,
      hh,
      mm,
      gcm_load_ciphertext_block(data_ptr.add(96)),
      6,
      h_powers_rev,
      zero_lanes
    );
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    round8!(7);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    ghash_fold_be_vec!(
      ll,
      hh,
      mm,
      gcm_load_ciphertext_block(data_ptr.add(112)),
      7,
      h_powers_rev,
      zero_lanes
    );
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    round8!(8);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);
    let acc = ghash_finish_products(ll, hh, mm);
    gcm_schedule_barrier!(s0, s1, s2, s3, s4, s5, s6, s7, ll, hh, mm);

    s0 = veorq_u8(vaeseq_u8(s0, k[9]), k[10]);
    s1 = veorq_u8(vaeseq_u8(s1, k[9]), k[10]);
    s2 = veorq_u8(vaeseq_u8(s2, k[9]), k[10]);
    s3 = veorq_u8(vaeseq_u8(s3, k[9]), k[10]);
    s4 = veorq_u8(vaeseq_u8(s4, k[9]), k[10]);
    s5 = veorq_u8(vaeseq_u8(s5, k[9]), k[10]);
    s6 = veorq_u8(vaeseq_u8(s6, k[9]), k[10]);
    s7 = veorq_u8(vaeseq_u8(s7, k[9]), k[10]);

    let p0 = vld1q_u8(data.as_ptr());
    let p1 = vld1q_u8(data.as_ptr().add(16));
    let p2 = vld1q_u8(data.as_ptr().add(32));
    let p3 = vld1q_u8(data.as_ptr().add(48));
    let p4 = vld1q_u8(data.as_ptr().add(64));
    let p5 = vld1q_u8(data.as_ptr().add(80));
    let p6 = vld1q_u8(data.as_ptr().add(96));
    let p7 = vld1q_u8(data.as_ptr().add(112));
    vst1q_u8(data.as_mut_ptr(), veorq_u8(p0, s0));
    vst1q_u8(data.as_mut_ptr().add(16), veorq_u8(p1, s1));
    vst1q_u8(data.as_mut_ptr().add(32), veorq_u8(p2, s2));
    vst1q_u8(data.as_mut_ptr().add(48), veorq_u8(p3, s3));
    vst1q_u8(data.as_mut_ptr().add(64), veorq_u8(p4, s4));
    vst1q_u8(data.as_mut_ptr().add(80), veorq_u8(p5, s5));
    vst1q_u8(data.as_mut_ptr().add(96), veorq_u8(p6, s6));
    vst1q_u8(data.as_mut_ptr().add(112), veorq_u8(p7, s7));

    acc
  }
}

#[cfg(feature = "aes-gcm")]
#[target_feature(enable = "aes,neon")]
#[inline(never)]
pub(super) unsafe fn encrypt_ctr32_be_xor_ghash_128b_chunks_128_core(
  keys: &Ce128RoundKeys,
  iv_prefix: &[u8; 12],
  mut ctr: u32,
  data: &mut [u8],
  mut acc: u128,
  tables: &super::Aarch64GcmTables<'_>,
) -> (u128, u32, usize) {
  debug_assert!(data.len() >= 128);

  // SAFETY: AES-128-GCM full-chunk loop because:
  // 1. The caller guarantees AES-CE + PMULL availability.
  // 2. The macOS assembly path receives the checked slice pointer/length and reports its exact
  //    processed byte count.
  // 3. The Rust fallback only creates 128-byte slices at offsets proven in bounds by the loop guards.
  // 4. Previous ciphertext is carried as initialized NEON lanes between Rust fallback iterations.
  // 5. The final pending lane aggregate folds the last encrypted 128-byte fallback chunk exactly
  //    once.
  unsafe {
    #[cfg(target_os = "macos")]
    {
      if crate::platform::caps().has(crate::platform::caps::aarch64::PMULL_EOR3_READY) {
        let mut state = asm::AesGcmAarch64State::new(acc, ctr);
        asm::rscrypto_aes128_gcm_seal_16x_eor3_aarch64_apple_darwin(
          keys.rk.as_ptr().cast::<u8>(),
          iv_prefix.as_ptr(),
          data.as_mut_ptr(),
          data.len(),
          tables.h_powers_rev_16.as_ptr(),
          tables.h_powers_rev_16_mid.as_ptr(),
          tables.h_powers_rev_16_pair.as_ptr(),
          &mut state,
        );
        if state.processed != 0 {
          return (state.acc(), state.ctr, state.processed);
        }
      }

      let mut state = asm::AesGcmAarch64State::new(acc, ctr);
      asm::rscrypto_aes128_gcm_seal_8x_aarch64_apple_darwin(
        keys.rk.as_ptr().cast::<u8>(),
        iv_prefix.as_ptr(),
        data.as_mut_ptr(),
        data.len(),
        tables.h_powers_rev_8.as_ptr(),
        &mut state,
      );
      if state.processed != 0 {
        return (state.acc(), state.ctr, state.processed);
      }
    }

    let data_ptr = data.as_mut_ptr();
    let mut offset = 0usize;

    let first = core::slice::from_raw_parts_mut(data_ptr, 128);
    let mut prev_blocks = encrypt_ctr32_be_xor_8blocks_128_core(keys, iv_prefix, ctr, first);
    ctr = ctr.wrapping_add(8);
    offset = offset.strict_add(128);

    while offset.strict_add(128) <= data.len() {
      let current = core::slice::from_raw_parts_mut(data_ptr.add(offset), 128);
      let (next_acc, next_blocks) = encrypt_ctr32_be_xor_8blocks_ghash_prev_bytes_128_core(
        keys,
        iv_prefix,
        ctr,
        current,
        acc,
        tables.h_powers_rev_8,
        &prev_blocks,
      );
      acc = next_acc;
      prev_blocks = next_blocks;
      ctr = ctr.wrapping_add(8);
      offset = offset.strict_add(128);
    }

    acc = super::super::polyval::aarch64_aggregate_8blocks_be_lanes_inline(acc, tables.h_powers_rev_8, &prev_blocks);

    (acc, ctr, offset)
  }
}

#[cfg(feature = "aes-gcm")]
#[target_feature(enable = "aes,neon")]
#[inline(never)]
pub(super) unsafe fn decrypt_ctr32_be_xor_ghash_128b_chunks_128_core(
  keys: &Ce128RoundKeys,
  iv_prefix: &[u8; 12],
  mut ctr: u32,
  data: &mut [u8],
  mut acc: u128,
  tables: &super::Aarch64GcmTables<'_>,
) -> (u128, u32, usize) {
  // SAFETY: AES-128-GCM full-chunk open loop because:
  // 1. The caller guarantees AES-CE + PMULL availability.
  // 2. The macOS assembly path receives the checked slice pointer/length and reports its exact
  //    processed byte count.
  // 3. The Rust fallback creates each 128-byte mutable chunk only after proving it is in bounds.
  // 4. Both paths fold ciphertext before plaintext stores.
  unsafe {
    #[cfg(target_os = "macos")]
    {
      if crate::platform::caps().has(crate::platform::caps::aarch64::PMULL_EOR3_READY) {
        let mut state = asm::AesGcmAarch64State::new(acc, ctr);
        asm::rscrypto_aes128_gcm_open_16x_eor3_aarch64_apple_darwin(
          keys.rk.as_ptr().cast::<u8>(),
          iv_prefix.as_ptr(),
          data.as_mut_ptr(),
          data.len(),
          tables.h_powers_rev_16.as_ptr(),
          tables.h_powers_rev_16_mid.as_ptr(),
          tables.h_powers_rev_16_pair.as_ptr(),
          &mut state,
        );
        if state.processed != 0 {
          return (state.acc(), state.ctr, state.processed);
        }
      }

      let mut state = asm::AesGcmAarch64State::new(acc, ctr);
      asm::rscrypto_aes128_gcm_open_8x_aarch64_apple_darwin(
        keys.rk.as_ptr().cast::<u8>(),
        iv_prefix.as_ptr(),
        data.as_mut_ptr(),
        data.len(),
        tables.h_powers_rev_8.as_ptr(),
        &mut state,
      );
      if state.processed != 0 {
        return (state.acc(), state.ctr, state.processed);
      }
    }

    let data_ptr = data.as_mut_ptr();
    let mut offset = 0usize;

    while offset.strict_add(128) <= data.len() {
      let current = core::slice::from_raw_parts_mut(data_ptr.add(offset), 128);
      acc =
        decrypt_ctr32_be_xor_8blocks_ghash_current_128_core(keys, iv_prefix, ctr, current, acc, tables.h_powers_rev_8);
      ctr = ctr.wrapping_add(8);
      offset = offset.strict_add(128);
    }

    (acc, ctr, offset)
  }
}

/// Encrypt four AES-128 GCM-SIV counter blocks and XOR them into `data`.
///
/// # Safety
///
/// Caller must ensure the CPU supports `aes` and `neon`, `data` has at least
/// 64 writable bytes, and `iv_suffix` is bytes 4..16 of the SIV counter block.
#[cfg(feature = "aes-gcm-siv")]
#[target_feature(enable = "aes,neon")]
#[inline]
pub(super) unsafe fn encrypt_ctr32_le_xor_4blocks_128_core(
  keys: &Ce128RoundKeys,
  iv_suffix: &[u8; 12],
  ctr: u32,
  data: &mut [u8],
) {
  debug_assert!(data.len() >= 64);

  // SAFETY: four-block AES-GCM-SIV CTR processing because:
  // 1. This function's caller guarantees AES-CE + NEON availability.
  // 2. `data` has at least 64 writable bytes; all loads/stores use fixed 16-byte offsets within that
  //    range.
  // 3. Counter blocks are initialized before loading into vector registers.
  unsafe {
    let k = &keys.rk;
    let base = gcmsiv_ctr32_base(iv_suffix);
    let mut s0 = gcmsiv_ctr32_block(base, ctr);
    let mut s1 = gcmsiv_ctr32_block(base, ctr.wrapping_add(1));
    let mut s2 = gcmsiv_ctr32_block(base, ctr.wrapping_add(2));
    let mut s3 = gcmsiv_ctr32_block(base, ctr.wrapping_add(3));

    macro_rules! round4 {
      ($round:expr) => {{
        s0 = vaesmcq_u8(vaeseq_u8(s0, k[$round]));
        s1 = vaesmcq_u8(vaeseq_u8(s1, k[$round]));
        s2 = vaesmcq_u8(vaeseq_u8(s2, k[$round]));
        s3 = vaesmcq_u8(vaeseq_u8(s3, k[$round]));
      }};
    }

    round4!(0);
    round4!(1);
    round4!(2);
    round4!(3);
    round4!(4);
    round4!(5);
    round4!(6);
    round4!(7);
    round4!(8);

    s0 = veorq_u8(vaeseq_u8(s0, k[9]), k[10]);
    s1 = veorq_u8(vaeseq_u8(s1, k[9]), k[10]);
    s2 = veorq_u8(vaeseq_u8(s2, k[9]), k[10]);
    s3 = veorq_u8(vaeseq_u8(s3, k[9]), k[10]);

    let p0 = vld1q_u8(data.as_ptr());
    let p1 = vld1q_u8(data.as_ptr().add(16));
    let p2 = vld1q_u8(data.as_ptr().add(32));
    let p3 = vld1q_u8(data.as_ptr().add(48));
    vst1q_u8(data.as_mut_ptr(), veorq_u8(p0, s0));
    vst1q_u8(data.as_mut_ptr().add(16), veorq_u8(p1, s1));
    vst1q_u8(data.as_mut_ptr().add(32), veorq_u8(p2, s2));
    vst1q_u8(data.as_mut_ptr().add(48), veorq_u8(p3, s3));
  }
}

/// Encrypt eight AES-128 GCM-SIV counter blocks and XOR them into `data`.
///
/// # Safety
///
/// Caller must ensure the CPU supports `aes` and `neon`, `data` has at least
/// 128 writable bytes, and `iv_suffix` is bytes 4..16 of the SIV counter block.
#[cfg(feature = "aes-gcm-siv")]
#[target_feature(enable = "aes,neon")]
#[inline]
pub(super) unsafe fn encrypt_ctr32_le_xor_8blocks_128_core(
  keys: &Ce128RoundKeys,
  iv_suffix: &[u8; 12],
  ctr: u32,
  data: &mut [u8],
) {
  debug_assert!(data.len() >= 128);

  // SAFETY: eight-block AES-GCM-SIV CTR processing because:
  // 1. This function's caller guarantees AES-CE + NEON availability.
  // 2. `data` has at least 128 writable bytes; all loads/stores use fixed 16-byte offsets within that
  //    range.
  // 3. Counter blocks are initialized before loading into vector registers.
  unsafe {
    let k = &keys.rk;
    let base = gcmsiv_ctr32_base(iv_suffix);
    let mut s0 = gcmsiv_ctr32_block(base, ctr);
    let mut s1 = gcmsiv_ctr32_block(base, ctr.wrapping_add(1));
    let mut s2 = gcmsiv_ctr32_block(base, ctr.wrapping_add(2));
    let mut s3 = gcmsiv_ctr32_block(base, ctr.wrapping_add(3));
    let mut s4 = gcmsiv_ctr32_block(base, ctr.wrapping_add(4));
    let mut s5 = gcmsiv_ctr32_block(base, ctr.wrapping_add(5));
    let mut s6 = gcmsiv_ctr32_block(base, ctr.wrapping_add(6));
    let mut s7 = gcmsiv_ctr32_block(base, ctr.wrapping_add(7));

    macro_rules! round8 {
      ($round:expr) => {{
        s0 = vaesmcq_u8(vaeseq_u8(s0, k[$round]));
        s1 = vaesmcq_u8(vaeseq_u8(s1, k[$round]));
        s2 = vaesmcq_u8(vaeseq_u8(s2, k[$round]));
        s3 = vaesmcq_u8(vaeseq_u8(s3, k[$round]));
        s4 = vaesmcq_u8(vaeseq_u8(s4, k[$round]));
        s5 = vaesmcq_u8(vaeseq_u8(s5, k[$round]));
        s6 = vaesmcq_u8(vaeseq_u8(s6, k[$round]));
        s7 = vaesmcq_u8(vaeseq_u8(s7, k[$round]));
      }};
    }

    round8!(0);
    round8!(1);
    round8!(2);
    round8!(3);
    round8!(4);
    round8!(5);
    round8!(6);
    round8!(7);
    round8!(8);

    s0 = veorq_u8(vaeseq_u8(s0, k[9]), k[10]);
    s1 = veorq_u8(vaeseq_u8(s1, k[9]), k[10]);
    s2 = veorq_u8(vaeseq_u8(s2, k[9]), k[10]);
    s3 = veorq_u8(vaeseq_u8(s3, k[9]), k[10]);
    s4 = veorq_u8(vaeseq_u8(s4, k[9]), k[10]);
    s5 = veorq_u8(vaeseq_u8(s5, k[9]), k[10]);
    s6 = veorq_u8(vaeseq_u8(s6, k[9]), k[10]);
    s7 = veorq_u8(vaeseq_u8(s7, k[9]), k[10]);

    let p0 = vld1q_u8(data.as_ptr());
    let p1 = vld1q_u8(data.as_ptr().add(16));
    let p2 = vld1q_u8(data.as_ptr().add(32));
    let p3 = vld1q_u8(data.as_ptr().add(48));
    let p4 = vld1q_u8(data.as_ptr().add(64));
    let p5 = vld1q_u8(data.as_ptr().add(80));
    let p6 = vld1q_u8(data.as_ptr().add(96));
    let p7 = vld1q_u8(data.as_ptr().add(112));
    vst1q_u8(data.as_mut_ptr(), veorq_u8(p0, s0));
    vst1q_u8(data.as_mut_ptr().add(16), veorq_u8(p1, s1));
    vst1q_u8(data.as_mut_ptr().add(32), veorq_u8(p2, s2));
    vst1q_u8(data.as_mut_ptr().add(48), veorq_u8(p3, s3));
    vst1q_u8(data.as_mut_ptr().add(64), veorq_u8(p4, s4));
    vst1q_u8(data.as_mut_ptr().add(80), veorq_u8(p5, s5));
    vst1q_u8(data.as_mut_ptr().add(96), veorq_u8(p6, s6));
    vst1q_u8(data.as_mut_ptr().add(112), veorq_u8(p7, s7));
  }
}

/// Encrypt a single 16-byte block using AES-128 with AES-CE.
///
/// # Safety
/// Caller must ensure the CPU supports AES-CE (`target_feature = "aes"`).
#[target_feature(enable = "aes,neon")]
pub(super) unsafe fn encrypt_block_128(keys: &Ce128RoundKeys, block: &mut [u8; 16]) {
  // SAFETY: target_feature gate guarantees AES-CE + NEON.
  unsafe { encrypt_block_128_core(keys, block) }
}
