use core::{arch::asm, simd::i64x2};

/// AES-256 round keys stored as 15 × 128-bit vectors for POWER8 vcipher.
///
/// POWER8 vcipher expects round keys in big-endian byte order, which
/// matches our portable key schedule (stored as big-endian u32 words).
#[repr(C, align(64))]
pub(in crate::aead) struct PpcRoundKeys {
  rk: [i64x2; 15],
}

impl PpcRoundKeys {
  /// Zeroize all round keys via volatile writes.
  pub(super) fn zeroize(&mut self) {
    // SAFETY: [i64x2; 15] is layout-compatible with [u8; 240].
    let bytes = unsafe { core::slice::from_raw_parts_mut(self.rk.as_mut_ptr().cast::<u8>(), 15usize.strict_mul(16)) };
    crate::traits::ct::zeroize(bytes);
  }
}

/// Load 16 bytes into `i64x2` in ISA byte order (big-endian AES state).
///
/// On ppc64le, `i64x2` element `[0]` maps to ISA doubleword 1 (bytes 8-15)
/// and element `[1]` maps to ISA doubleword 0 (bytes 0-7). So to place
/// memory bytes `[0..8)` into ISA bytes `[0..8)` (high doubleword), we put
/// them into element `[1]`. Memory bytes `[8..16)` go into element `[0]`.
///
/// Pure-Rust approach avoids VSX `lxvd2x` asm which needs VSR register
/// numbers incompatible with the `vreg` register class.
#[inline]
fn load_block_be(bytes: &[u8; 16]) -> i64x2 {
  let dw0 = i64::from_be_bytes([
    bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
  ]);
  let dw1 = i64::from_be_bytes([
    bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
  ]);
  #[cfg(target_endian = "little")]
  {
    // LE element[0] = ISA DW1, element[1] = ISA DW0.
    i64x2::from_array([dw1, dw0])
  }
  #[cfg(target_endian = "big")]
  {
    i64x2::from_array([dw0, dw1])
  }
}

/// Store an `i64x2` (in ISA byte order = big-endian AES state) to memory.
///
/// Inverse of `load_block_be`.
#[inline]
fn store_block_be(out: &mut [u8; 16], block: i64x2) {
  let elems = block.to_array();
  #[cfg(target_endian = "little")]
  let (hi, lo) = (elems[1].to_be_bytes(), elems[0].to_be_bytes());
  #[cfg(target_endian = "big")]
  let (hi, lo) = (elems[0].to_be_bytes(), elems[1].to_be_bytes());
  out[0..8].copy_from_slice(&hi);
  out[8..16].copy_from_slice(&lo);
}

/// Convert portable round keys (60 × big-endian u32) to POWER8 vector format.
///
/// Each group of 4 u32 words forms one 128-bit round key in canonical AES
/// byte order. On `powerpc64le`, POWER vector registers need the same
/// big-endian byte normalization that the compiler applies for
/// `vec_xl_be`/`vec_xst_be`.
pub(super) fn from_portable(rk: &[u32; 60]) -> PpcRoundKeys {
  let mut keys = [i64x2::from_array([0, 0]); 15];
  let mut i = 0usize;
  while i < 15 {
    let base = i.strict_mul(4);
    let mut bytes = [0u8; 16];
    bytes[0..4].copy_from_slice(&rk[base].to_be_bytes());
    bytes[4..8].copy_from_slice(&rk[base.strict_add(1)].to_be_bytes());
    bytes[8..12].copy_from_slice(&rk[base.strict_add(2)].to_be_bytes());
    bytes[12..16].copy_from_slice(&rk[base.strict_add(3)].to_be_bytes());
    keys[i] = load_block_be(&bytes);
    i = i.strict_add(1);
  }
  PpcRoundKeys { rk: keys }
}

/// Hardware SubWord using POWER8 `vsbox`.
///
/// Broadcasts the input word to all 4 columns, applies the byte-wise AES
/// S-box in parallel, then returns the first substituted word.
#[target_feature(enable = "altivec,vsx,power8-vector,power8-crypto")]
#[inline]
/// # Safety
///
/// Caller must ensure POWER8 vector crypto support is available.
unsafe fn sub_word_hw(w: u32) -> u32 {
  let word = w.to_be_bytes();
  let bytes = [
    word[0], word[1], word[2], word[3], word[0], word[1], word[2], word[3], word[0], word[1], word[2], word[3],
    word[0], word[1], word[2], word[3],
  ];
  let state = load_block_be(&bytes);
  // SAFETY: caller guarantees POWER8 crypto availability.
  unsafe {
    let out: i64x2;
    asm!(
      "vsbox {out}, {state}",
      out = lateout(vreg) out,
      state = in(vreg) state,
      options(nomem, nostack, pure),
    );
    let mut out_bytes = [0u8; 16];
    store_block_be(&mut out_bytes, out);
    u32::from_be_bytes([out_bytes[0], out_bytes[1], out_bytes[2], out_bytes[3]])
  }
}

/// Hardware-accelerated AES-256 key expansion using POWER8 `vsbox`.
#[target_feature(enable = "altivec,vsx,power8-vector,power8-crypto")]
#[inline]
/// # Safety
///
/// Caller must ensure POWER8 vector crypto support is available.
pub(super) unsafe fn expand_key_hw(key: &[u8; 32]) -> PpcRoundKeys {
  // SAFETY: caller guarantees POWER8 crypto availability.
  unsafe {
    let mut rk = [0u32; super::EXPANDED_KEY_WORDS];

    // Load the initial key as big-endian u32 words.
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

    let keys = from_portable(&rk);
    // SAFETY: [u32; 60] is layout-compatible with [u8; 240].
    crate::traits::ct::zeroize(core::slice::from_raw_parts_mut(
      rk.as_mut_ptr().cast::<u8>(),
      super::EXPANDED_KEY_WORDS.strict_mul(4),
    ));
    keys
  }
}

/// Non-inline entry point for hardware key expansion.
#[target_feature(enable = "altivec,vsx,power8-vector,power8-crypto")]
/// # Safety
///
/// Caller must ensure POWER8 vector crypto support is available.
pub(super) unsafe fn expand_key(key: &[u8; 32]) -> PpcRoundKeys {
  // SAFETY: target_feature gate guarantees POWER8 crypto.
  unsafe { expand_key_hw(key) }
}

/// Core block-encrypt logic — `#[target_feature]` + `#[inline(always)]` for
/// guaranteed inlining without register spills.
#[target_feature(enable = "altivec,vsx,power8-vector,power8-crypto")]
#[inline]
/// # Safety
///
/// Caller must ensure POWER8 vector crypto support is available and `block`
/// points to a valid writable AES block.
pub(super) unsafe fn encrypt_block_core(keys: &PpcRoundKeys, block: &mut [u8; 16]) {
  // SAFETY: caller guarantees POWER8 crypto via target_feature chain.
  unsafe {
    let k = &keys.rk;

    let mut state = load_block_be(block);

    // Rounds 1–13: vcipher (SubBytes + ShiftRows + MixColumns + AddRoundKey).
    macro_rules! vcipher_round {
      ($rk:expr) => {
        asm!(
          "vcipher {s}, {s}, {rk}",
          s = inlateout(vreg) state,
          rk = in(vreg) $rk,
          options(nomem, nostack),
        );
      };
    }

    // Initial XOR is folded into the first vcipher: state = vcipher(plaintext, K0)
    // means SubBytes(ShiftRows(MixColumns(plaintext XOR K0))). But vcipher actually
    // computes ShiftRows(SubBytes(MixColumns(state))) XOR rk — so we need the initial
    // AddRoundKey (K0) separately.
    //
    // Actually, vcipher does: ShiftRows → SubBytes → MixColumns → XOR(rk).
    // The initial step of AES is AddRoundKey(K0), then 13 middle rounds, then final.
    // We pre-XOR state with K0, then run 13 vcipher rounds with K1..K13, then
    // vcipherlast with K14.
    asm!(
      "vxor {s}, {s}, {rk}",
      s = inlateout(vreg) state,
      rk = in(vreg) k[0],
      options(nomem, nostack),
    );

    vcipher_round!(k[1]);
    vcipher_round!(k[2]);
    vcipher_round!(k[3]);
    vcipher_round!(k[4]);
    vcipher_round!(k[5]);
    vcipher_round!(k[6]);
    vcipher_round!(k[7]);
    vcipher_round!(k[8]);
    vcipher_round!(k[9]);
    vcipher_round!(k[10]);
    vcipher_round!(k[11]);
    vcipher_round!(k[12]);
    vcipher_round!(k[13]);

    // Round 14 (final): SubBytes + ShiftRows + AddRoundKey (no MixColumns).
    asm!(
      "vcipherlast {s}, {s}, {rk}",
      s = inlateout(vreg) state,
      rk = in(vreg) k[14],
      options(nomem, nostack),
    );

    store_block_be(block, state);
  }
}

/// Encrypt four independent AES-256 blocks with POWER8 vcipher.
///
/// POWER10 has four AES pipes; keeping four independent states live lets the
/// scheduler issue vcipher work across those pipes instead of serializing one
/// block through all rounds before starting the next counter block.
///
/// # Safety
///
/// Caller must ensure POWER8 crypto support is available and `blocks` points
/// to four valid writable AES blocks.
#[target_feature(enable = "altivec,vsx,power8-vector,power8-crypto")]
pub(super) unsafe fn encrypt_4blocks_core(keys: &PpcRoundKeys, blocks: &mut [[u8; 16]; 4]) {
  // SAFETY: four-block POWER8 AES because:
  // 1. The caller guarantees POWER8 crypto support before entering this target-feature scope.
  // 2. `blocks` is a live mutable reference to four initialized 16-byte AES blocks.
  // 3. Each block is loaded once, transformed in registers, and stored back to its original slot.
  unsafe {
    let k = &keys.rk;

    let mut s0 = load_block_be(&blocks[0]);
    let mut s1 = load_block_be(&blocks[1]);
    let mut s2 = load_block_be(&blocks[2]);
    let mut s3 = load_block_be(&blocks[3]);

    macro_rules! vxor_all {
      ($rk:expr) => {
        asm!(
          "vxor {s0}, {s0}, {rk}",
          "vxor {s1}, {s1}, {rk}",
          "vxor {s2}, {s2}, {rk}",
          "vxor {s3}, {s3}, {rk}",
          s0 = inlateout(vreg) s0,
          s1 = inlateout(vreg) s1,
          s2 = inlateout(vreg) s2,
          s3 = inlateout(vreg) s3,
          rk = in(vreg) $rk,
          options(nomem, nostack),
        );
      };
    }

    macro_rules! vcipher_all {
      ($rk:expr) => {
        asm!(
          "vcipher {s0}, {s0}, {rk}",
          "vcipher {s1}, {s1}, {rk}",
          "vcipher {s2}, {s2}, {rk}",
          "vcipher {s3}, {s3}, {rk}",
          s0 = inlateout(vreg) s0,
          s1 = inlateout(vreg) s1,
          s2 = inlateout(vreg) s2,
          s3 = inlateout(vreg) s3,
          rk = in(vreg) $rk,
          options(nomem, nostack),
        );
      };
    }

    vxor_all!(k[0]);
    vcipher_all!(k[1]);
    vcipher_all!(k[2]);
    vcipher_all!(k[3]);
    vcipher_all!(k[4]);
    vcipher_all!(k[5]);
    vcipher_all!(k[6]);
    vcipher_all!(k[7]);
    vcipher_all!(k[8]);
    vcipher_all!(k[9]);
    vcipher_all!(k[10]);
    vcipher_all!(k[11]);
    vcipher_all!(k[12]);
    vcipher_all!(k[13]);

    asm!(
      "vcipherlast {s0}, {s0}, {rk}",
      "vcipherlast {s1}, {s1}, {rk}",
      "vcipherlast {s2}, {s2}, {rk}",
      "vcipherlast {s3}, {s3}, {rk}",
      s0 = inlateout(vreg) s0,
      s1 = inlateout(vreg) s1,
      s2 = inlateout(vreg) s2,
      s3 = inlateout(vreg) s3,
      rk = in(vreg) k[14],
      options(nomem, nostack),
    );

    store_block_be(&mut blocks[0], s0);
    store_block_be(&mut blocks[1], s1);
    store_block_be(&mut blocks[2], s2);
    store_block_be(&mut blocks[3], s3);
  }
}

/// Encrypt a single 16-byte block using AES-256 with POWER8 vcipher.
///
/// vcipher performs one AES middle round (SubBytes + ShiftRows + MixColumns
/// + AddRoundKey). vcipherlast performs the final round (no MixColumns).
///
/// # Safety
/// Caller must ensure POWER8 crypto instructions are available.
#[target_feature(enable = "altivec,vsx,power8-vector,power8-crypto")]
pub(super) unsafe fn encrypt_block(keys: &PpcRoundKeys, block: &mut [u8; 16]) {
  // SAFETY: target_feature gate guarantees POWER8 crypto.
  unsafe { encrypt_block_core(keys, block) }
}

// AES-128 (11 round keys, 10 rounds)

/// AES-128 round keys stored as 11 × 128-bit vectors for POWER8 vcipher.
#[repr(C, align(64))]
pub(in crate::aead) struct Ppc128RoundKeys {
  rk: [i64x2; 11],
}

impl Ppc128RoundKeys {
  /// Zeroize all round keys via volatile writes.
  pub(super) fn zeroize(&mut self) {
    // SAFETY: [i64x2; 11] is layout-compatible with [u8; 176].
    let bytes = unsafe { core::slice::from_raw_parts_mut(self.rk.as_mut_ptr().cast::<u8>(), 11usize.strict_mul(16)) };
    crate::traits::ct::zeroize(bytes);
  }
}

/// Convert portable round keys (44 × big-endian u32) to POWER8 vector format.
pub(super) fn from_portable_128(rk: &[u32; 44]) -> Ppc128RoundKeys {
  let mut keys = [i64x2::from_array([0, 0]); 11];
  let mut i = 0usize;
  while i < 11 {
    let base = i.strict_mul(4);
    let mut bytes = [0u8; 16];
    bytes[0..4].copy_from_slice(&rk[base].to_be_bytes());
    bytes[4..8].copy_from_slice(&rk[base.strict_add(1)].to_be_bytes());
    bytes[8..12].copy_from_slice(&rk[base.strict_add(2)].to_be_bytes());
    bytes[12..16].copy_from_slice(&rk[base.strict_add(3)].to_be_bytes());
    keys[i] = load_block_be(&bytes);
    i = i.strict_add(1);
  }
  Ppc128RoundKeys { rk: keys }
}

/// Hardware-accelerated AES-128 key expansion using POWER8 `vsbox`.
#[target_feature(enable = "altivec,vsx,power8-vector,power8-crypto")]
#[inline]
/// # Safety
///
/// Caller must ensure POWER8 vector crypto support is available.
unsafe fn expand_key_128_hw(key: &[u8; 16]) -> Ppc128RoundKeys {
  // SAFETY: caller guarantees POWER8 crypto availability.
  unsafe {
    let mut rk = [0u32; super::EXPANDED_KEY_WORDS_128];

    let mut i = 0usize;
    while i < 4 {
      let base = i.strict_mul(4);
      rk[i] = u32::from_be_bytes([
        key[base],
        key[base.strict_add(1)],
        key[base.strict_add(2)],
        key[base.strict_add(3)],
      ]);
      i = i.strict_add(1);
    }

    i = 4;
    while i < super::EXPANDED_KEY_WORDS_128 {
      let mut temp = rk[i.strict_sub(1)];
      if i.strict_rem(4) == 0 {
        temp = sub_word_hw(super::rot_word(temp)) ^ super::RCON[i.strict_div(4).strict_sub(1)];
      }
      rk[i] = rk[i.strict_sub(4)] ^ temp;
      i = i.strict_add(1);
    }

    let keys = from_portable_128(&rk);
    // SAFETY: [u32; 44] is layout-compatible with [u8; 176].
    crate::traits::ct::zeroize(core::slice::from_raw_parts_mut(
      rk.as_mut_ptr().cast::<u8>(),
      super::EXPANDED_KEY_WORDS_128.strict_mul(4),
    ));
    keys
  }
}

/// Non-inline entry point for AES-128 hardware key expansion.
#[target_feature(enable = "altivec,vsx,power8-vector,power8-crypto")]
/// # Safety
///
/// Caller must ensure POWER8 vector crypto support is available.
pub(super) unsafe fn expand_key_128(key: &[u8; 16]) -> Ppc128RoundKeys {
  // SAFETY: target_feature gate guarantees POWER8 crypto.
  unsafe { expand_key_128_hw(key) }
}

#[target_feature(enable = "altivec,vsx,power8-vector,power8-crypto")]
#[inline]
/// # Safety
///
/// Caller must ensure POWER8 vector crypto support is available and `block`
/// points to a valid writable AES block.
pub(super) unsafe fn encrypt_block_128_core(keys: &Ppc128RoundKeys, block: &mut [u8; 16]) {
  // SAFETY: caller guarantees POWER8 crypto via target_feature chain.
  unsafe {
    let k = &keys.rk;

    let mut state = load_block_be(block);

    macro_rules! vcipher_round {
      ($rk:expr) => {
        asm!(
          "vcipher {s}, {s}, {rk}",
          s = inlateout(vreg) state,
          rk = in(vreg) $rk,
          options(nomem, nostack),
        );
      };
    }

    asm!(
      "vxor {s}, {s}, {rk}",
      s = inlateout(vreg) state,
      rk = in(vreg) k[0],
      options(nomem, nostack),
    );

    vcipher_round!(k[1]);
    vcipher_round!(k[2]);
    vcipher_round!(k[3]);
    vcipher_round!(k[4]);
    vcipher_round!(k[5]);
    vcipher_round!(k[6]);
    vcipher_round!(k[7]);
    vcipher_round!(k[8]);
    vcipher_round!(k[9]);

    asm!(
      "vcipherlast {s}, {s}, {rk}",
      s = inlateout(vreg) state,
      rk = in(vreg) k[10],
      options(nomem, nostack),
    );

    store_block_be(block, state);
  }
}

/// Encrypt four independent AES-128 blocks with POWER8 vcipher.
///
/// # Safety
///
/// Caller must ensure POWER8 crypto support is available and `blocks` points
/// to four valid writable AES blocks.
#[target_feature(enable = "altivec,vsx,power8-vector,power8-crypto")]
pub(super) unsafe fn encrypt_4blocks_128_core(keys: &Ppc128RoundKeys, blocks: &mut [[u8; 16]; 4]) {
  // SAFETY: four-block POWER8 AES because:
  // 1. The caller guarantees POWER8 crypto support before entering this target-feature scope.
  // 2. `blocks` is a live mutable reference to four initialized 16-byte AES blocks.
  // 3. Each block is loaded once, transformed in registers, and stored back to its original slot.
  unsafe {
    let k = &keys.rk;

    let mut s0 = load_block_be(&blocks[0]);
    let mut s1 = load_block_be(&blocks[1]);
    let mut s2 = load_block_be(&blocks[2]);
    let mut s3 = load_block_be(&blocks[3]);

    macro_rules! vxor_all {
      ($rk:expr) => {
        asm!(
          "vxor {s0}, {s0}, {rk}",
          "vxor {s1}, {s1}, {rk}",
          "vxor {s2}, {s2}, {rk}",
          "vxor {s3}, {s3}, {rk}",
          s0 = inlateout(vreg) s0,
          s1 = inlateout(vreg) s1,
          s2 = inlateout(vreg) s2,
          s3 = inlateout(vreg) s3,
          rk = in(vreg) $rk,
          options(nomem, nostack),
        );
      };
    }

    macro_rules! vcipher_all {
      ($rk:expr) => {
        asm!(
          "vcipher {s0}, {s0}, {rk}",
          "vcipher {s1}, {s1}, {rk}",
          "vcipher {s2}, {s2}, {rk}",
          "vcipher {s3}, {s3}, {rk}",
          s0 = inlateout(vreg) s0,
          s1 = inlateout(vreg) s1,
          s2 = inlateout(vreg) s2,
          s3 = inlateout(vreg) s3,
          rk = in(vreg) $rk,
          options(nomem, nostack),
        );
      };
    }

    vxor_all!(k[0]);
    vcipher_all!(k[1]);
    vcipher_all!(k[2]);
    vcipher_all!(k[3]);
    vcipher_all!(k[4]);
    vcipher_all!(k[5]);
    vcipher_all!(k[6]);
    vcipher_all!(k[7]);
    vcipher_all!(k[8]);
    vcipher_all!(k[9]);

    asm!(
      "vcipherlast {s0}, {s0}, {rk}",
      "vcipherlast {s1}, {s1}, {rk}",
      "vcipherlast {s2}, {s2}, {rk}",
      "vcipherlast {s3}, {s3}, {rk}",
      s0 = inlateout(vreg) s0,
      s1 = inlateout(vreg) s1,
      s2 = inlateout(vreg) s2,
      s3 = inlateout(vreg) s3,
      rk = in(vreg) k[10],
      options(nomem, nostack),
    );

    store_block_be(&mut blocks[0], s0);
    store_block_be(&mut blocks[1], s1);
    store_block_be(&mut blocks[2], s2);
    store_block_be(&mut blocks[3], s3);
  }
}

/// Encrypt a single 16-byte block using AES-128 with POWER8 vcipher.
///
/// # Safety
/// Caller must ensure POWER8 crypto instructions are available.
#[target_feature(enable = "altivec,vsx,power8-vector,power8-crypto")]
pub(super) unsafe fn encrypt_block_128(keys: &Ppc128RoundKeys, block: &mut [u8; 16]) {
  // SAFETY: target_feature gate guarantees POWER8 crypto.
  unsafe { encrypt_block_128_core(keys, block) }
}

// Eight-block AES-GCM: CTR keystream, data, and GHASH input stay in vector registers.

/// `vperm` control that reverses the 16 register bytes.
///
/// A little-endian vector load puts memory byte `15 - k` in register byte `k`; reversing gives
/// the AES state order (register byte `k` = memory byte `k`), and a second reversal undoes it
/// before the store. The values are the register bytes 15..8 (element 1, ISA doubleword 0) and
/// 7..0 (element 0, ISA doubleword 1).
#[cfg(feature = "aes-gcm")]
const BYTE_REVERSE: i64x2 = i64x2::from_array([0x0706_0504_0302_0100, 0x0f0e_0d0c_0b0a_0908]);

/// POLYVAL reduction constant `x^63 + x^62 + x^57` in ISA doubleword 1 (element 0).
#[cfg(feature = "aes-gcm")]
const POLYVAL_FOLD: i64x2 = i64x2::from_array([0xc200_0000_0000_0000_u64.cast_signed(), 0]);

/// `value` as a vector whose 128 register bits are its numeric bits (element 0 holds the low
/// doubleword on little-endian POWER).
#[cfg(feature = "aes-gcm")]
#[inline(always)]
fn vector_from_u128(value: u128) -> i64x2 {
  let bytes = value.to_le_bytes();
  let (low, high) = bytes.split_at(8);
  i64x2::from_array([
    i64::from_le_bytes(low.try_into().expect("eight low bytes")),
    i64::from_le_bytes(high.try_into().expect("eight high bytes")),
  ])
}

/// A little-endian vector load of 16 bytes: register byte `k` holds memory byte `15 - k`.
#[cfg(feature = "aes-gcm")]
#[inline(always)]
fn load_le(bytes: &[u8; 16]) -> i64x2 {
  let (low, high) = bytes.split_at(8);
  i64x2::from_array([
    i64::from_le_bytes(low.try_into().expect("eight low bytes")),
    i64::from_le_bytes(high.try_into().expect("eight high bytes")),
  ])
}

/// The inverse of [`load_le`].
#[cfg(feature = "aes-gcm")]
#[inline(always)]
fn store_le(out: &mut [u8; 16], value: i64x2) {
  let [low, high] = value.to_array();
  let (out_low, out_high) = out.split_at_mut(8);
  out_low.copy_from_slice(&low.to_le_bytes());
  out_high.copy_from_slice(&high.to_le_bytes());
}

#[cfg(feature = "aes-gcm")]
#[inline(always)]
fn u128_from_vector(value: i64x2) -> u128 {
  let [low, high] = value.to_array();
  u128::from(low.cast_unsigned()) | (u128::from(high.cast_unsigned()) << u64::BITS)
}

/// The GCM key powers `[H^8, ..., H]` (POLYVAL domain) in the three `vpmsumd` operand forms.
///
/// With a block `b = b1·x^64 + b0` held as ISA doublewords `[b1, b0]`, `vpmsumd` multiplies the
/// matching doublewords and adds the two products, so `low = [0, h0]` gives `b0·h0`,
/// `mid = [h0, h1]` gives `b1·h0 + b0·h1`, and `high = [h1, 0]` gives `b1·h1`.
#[cfg(feature = "aes-gcm")]
struct GhashPowers8 {
  low: [i64x2; 8],
  mid: [i64x2; 8],
  high: [i64x2; 8],
}

#[cfg(feature = "aes-gcm")]
impl GhashPowers8 {
  #[inline(always)]
  fn new(h_powers_rev_8: &[u128; 8]) -> Self {
    let split = |power: &u128| {
      let [h0, h1] = vector_from_u128(*power).to_array();
      (h0, h1)
    };
    Self {
      low: h_powers_rev_8.each_ref().map(|power| {
        let (h0, _) = split(power);
        i64x2::from_array([h0, 0])
      }),
      mid: h_powers_rev_8.each_ref().map(|power| {
        let (h0, h1) = split(power);
        i64x2::from_array([h1, h0])
      }),
      high: h_powers_rev_8.each_ref().map(|power| {
        let (_, h1) = split(power);
        i64x2::from_array([0, h1])
      }),
    }
  }
}

/// CTR-encrypt or decrypt `data`, a whole number of 128-byte chunks, from `initial_counter`, and
/// fold the ciphertext into the POLYVAL accumulator `acc`. Returns the accumulator.
///
/// Each chunk runs eight AES states through the rounds together, then multiplies its eight
/// ciphertext blocks by `H^8, ..., H` and reduces once. The counter word wraps modulo 2^32, as
/// GCM's `inc32` does.
///
/// # Safety
///
/// Caller must ensure POWER8 vector crypto support is available.
#[cfg(feature = "aes-gcm")]
#[target_feature(enable = "altivec,vsx,power8-vector,power8-crypto")]
unsafe fn gcm_ctr32_ghash_8x<const KEYS: usize, const OPEN: bool>(
  rk: &[i64x2; KEYS],
  initial_counter: &[u8; 16],
  data: &mut [u8],
  acc: u128,
  h_powers_rev_8: &[u128; 8],
) -> u128 {
  debug_assert_eq!(data.len() % 128, 0);
  let powers = GhashPowers8::new(h_powers_rev_8);
  let increments: [i64x2; 9] =
    core::array::from_fn(|index| i64x2::from_array([i64::try_from(index).expect("small counter increment"), 0]));
  // SAFETY: eight-block POWER AES-GCM because:
  // 1. The caller guarantees POWER8 crypto support before entering this target-feature scope.
  // 2. Every asm block reads and writes only its vector register operands.
  // 3. Data moves through safe 16-byte array loads and stores.
  unsafe {
    let mut counter = load_le(initial_counter);
    asm!(
      "vperm {c}, {c}, {c}, {rev}",
      c = inout(vreg) counter,
      rev = in(vreg) BYTE_REVERSE,
      options(nomem, nostack, pure),
    );
    let mut acc = vector_from_u128(acc);

    for chunk in data.as_chunks_mut::<128>().0 {
      let blocks = chunk.as_chunks_mut::<16>().0;
      let (mut s0, mut s1, mut s2, mut s3, mut s4, mut s5, mut s6, mut s7): (
        i64x2,
        i64x2,
        i64x2,
        i64x2,
        i64x2,
        i64x2,
        i64x2,
        i64x2,
      );
      asm!(
        "vadduwm {s1}, {c}, {i1}",
        "vadduwm {s2}, {c}, {i2}",
        "vadduwm {s3}, {c}, {i3}",
        "vadduwm {s4}, {c}, {i4}",
        "vadduwm {s5}, {c}, {i5}",
        "vadduwm {s6}, {c}, {i6}",
        "vadduwm {s7}, {c}, {i7}",
        s1 = out(vreg) s1,
        s2 = out(vreg) s2,
        s3 = out(vreg) s3,
        s4 = out(vreg) s4,
        s5 = out(vreg) s5,
        s6 = out(vreg) s6,
        s7 = out(vreg) s7,
        c = in(vreg) counter,
        i1 = in(vreg) increments[1],
        i2 = in(vreg) increments[2],
        i3 = in(vreg) increments[3],
        i4 = in(vreg) increments[4],
        i5 = in(vreg) increments[5],
        i6 = in(vreg) increments[6],
        i7 = in(vreg) increments[7],
        options(nomem, nostack, pure),
      );
      s0 = counter;
      asm!(
        "vadduwm {c}, {c}, {i}",
        c = inout(vreg) counter,
        i = in(vreg) increments[8],
        options(nomem, nostack, pure),
      );

      macro_rules! round_all {
        ($op:literal, $rk:expr) => {
          asm!(
            concat!($op, " {s0}, {s0}, {rk}"),
            concat!($op, " {s1}, {s1}, {rk}"),
            concat!($op, " {s2}, {s2}, {rk}"),
            concat!($op, " {s3}, {s3}, {rk}"),
            concat!($op, " {s4}, {s4}, {rk}"),
            concat!($op, " {s5}, {s5}, {rk}"),
            concat!($op, " {s6}, {s6}, {rk}"),
            concat!($op, " {s7}, {s7}, {rk}"),
            s0 = inlateout(vreg) s0,
            s1 = inlateout(vreg) s1,
            s2 = inlateout(vreg) s2,
            s3 = inlateout(vreg) s3,
            s4 = inlateout(vreg) s4,
            s5 = inlateout(vreg) s5,
            s6 = inlateout(vreg) s6,
            s7 = inlateout(vreg) s7,
            rk = in(vreg) $rk,
            options(nomem, nostack, pure),
          );
        };
      }

      round_all!("vxor", rk[0]);
      for &round_key in &rk[1..KEYS.strict_sub(1)] {
        round_all!("vcipher", round_key);
      }
      round_all!("vcipherlast", rk[KEYS.strict_sub(1)]);

      // XOR the keystream into the data. GHASH takes the ciphertext, which is the output when
      // sealing and the input when opening; in AES byte order it is the numeric POLYVAL block.
      macro_rules! xor_block {
        ($keystream:ident, $offset:literal) => {{
          let slot = &mut blocks[$offset];
          let input = load_le(slot);
          let (input_be, output_be, output): (i64x2, i64x2, i64x2);
          asm!(
            "vperm {x}, {x}, {x}, {rev}",
            "vxor {y}, {x}, {k}",
            "vperm {out}, {y}, {y}, {rev}",
            x = inout(vreg) input => input_be,
            y = out(vreg) output_be,
            out = lateout(vreg) output,
            k = in(vreg) $keystream,
            rev = in(vreg) BYTE_REVERSE,
            options(nomem, nostack, pure),
          );
          store_le(slot, output);
          if OPEN { input_be } else { output_be }
        }};
      }
      let c0 = xor_block!(s0, 0);
      let c1 = xor_block!(s1, 1);
      let c2 = xor_block!(s2, 2);
      let c3 = xor_block!(s3, 3);
      let c4 = xor_block!(s4, 4);
      let c5 = xor_block!(s5, 5);
      let c6 = xor_block!(s6, 6);
      let c7 = xor_block!(s7, 7);

      // POLYVAL: (acc + c0)·H^8 + c1·H^7 + ... + c7·H, one reduction per chunk.
      let (mut low, mut mid, mut high): (i64x2, i64x2, i64x2);
      // Each product is written before the later power operands are read, so the outputs need
      // registers of their own (`out`, not `lateout`).
      asm!(
        "vxor {b}, {b}, {a}",
        "vpmsumd {low}, {b}, {hl}",
        "vpmsumd {mid}, {b}, {hm}",
        "vpmsumd {high}, {b}, {hh}",
        b = inout(vreg) c0 => _,
        a = in(vreg) acc,
        hl = in(vreg) powers.low[0],
        hm = in(vreg) powers.mid[0],
        hh = in(vreg) powers.high[0],
        low = out(vreg) low,
        mid = out(vreg) mid,
        high = out(vreg) high,
        options(nomem, nostack, pure),
      );
      macro_rules! ghash_block {
        ($block:ident, $index:literal) => {
          asm!(
            "vpmsumd {t}, {b}, {hl}",
            "vxor {low}, {low}, {t}",
            "vpmsumd {t}, {b}, {hm}",
            "vxor {mid}, {mid}, {t}",
            "vpmsumd {t}, {b}, {hh}",
            "vxor {high}, {high}, {t}",
            b = in(vreg) $block,
            hl = in(vreg) powers.low[$index],
            hm = in(vreg) powers.mid[$index],
            hh = in(vreg) powers.high[$index],
            low = inout(vreg) low,
            mid = inout(vreg) mid,
            high = inout(vreg) high,
            t = out(vreg) _,
            options(nomem, nostack, pure),
          );
        };
      }
      ghash_block!(c1, 1);
      ghash_block!(c2, 2);
      ghash_block!(c3, 3);
      ghash_block!(c4, 4);
      ghash_block!(c5, 5);
      ghash_block!(c6, 6);
      ghash_block!(c7, 7);

      // low += mid·x^64 and high += mid/x^64; then two folds of x^-64 each: multiply the low
      // doubleword by the constant, swap the doublewords, and add.
      asm!(
        "vxor {z}, {z}, {z}",
        "vsldoi {t}, {mid}, {z}, 8",
        "vxor {low}, {low}, {t}",
        "vsldoi {t}, {z}, {mid}, 8",
        "vxor {high}, {high}, {t}",
        "vpmsumd {t}, {low}, {k}",
        "vsldoi {low}, {low}, {low}, 8",
        "vxor {low}, {low}, {t}",
        "vpmsumd {t}, {low}, {k}",
        "vsldoi {low}, {low}, {low}, 8",
        "vxor {low}, {low}, {t}",
        "vxor {low}, {low}, {high}",
        low = inout(vreg) low,
        mid = in(vreg) mid,
        high = inout(vreg) high,
        k = in(vreg) POLYVAL_FOLD,
        z = out(vreg) _,
        t = out(vreg) _,
        options(nomem, nostack, pure),
      );
      acc = low;
    }
    u128_from_vector(acc)
  }
}

/// Eight-block AES-256-GCM sealing over whole 128-byte chunks; see [`gcm_ctr32_ghash_8x`].
///
/// # Safety
///
/// Caller must ensure POWER8 vector crypto support is available.
#[cfg(feature = "aes-gcm")]
#[target_feature(enable = "altivec,vsx,power8-vector,power8-crypto")]
pub(super) unsafe fn gcm_seal_8x(
  keys: &PpcRoundKeys,
  initial_counter: &[u8; 16],
  data: &mut [u8],
  acc: u128,
  h_powers_rev_8: &[u128; 8],
) -> u128 {
  // SAFETY: the caller guarantees POWER8 crypto support.
  unsafe { gcm_ctr32_ghash_8x::<15, false>(&keys.rk, initial_counter, data, acc, h_powers_rev_8) }
}

/// Eight-block AES-256-GCM opening over whole 128-byte chunks; see [`gcm_ctr32_ghash_8x`].
///
/// # Safety
///
/// Caller must ensure POWER8 vector crypto support is available.
#[cfg(feature = "aes-gcm")]
#[target_feature(enable = "altivec,vsx,power8-vector,power8-crypto")]
pub(super) unsafe fn gcm_open_8x(
  keys: &PpcRoundKeys,
  initial_counter: &[u8; 16],
  data: &mut [u8],
  acc: u128,
  h_powers_rev_8: &[u128; 8],
) -> u128 {
  // SAFETY: the caller guarantees POWER8 crypto support.
  unsafe { gcm_ctr32_ghash_8x::<15, true>(&keys.rk, initial_counter, data, acc, h_powers_rev_8) }
}

/// Eight-block AES-128-GCM sealing over whole 128-byte chunks; see [`gcm_ctr32_ghash_8x`].
///
/// # Safety
///
/// Caller must ensure POWER8 vector crypto support is available.
#[cfg(feature = "aes-gcm")]
#[target_feature(enable = "altivec,vsx,power8-vector,power8-crypto")]
pub(super) unsafe fn gcm_seal_8x_128(
  keys: &Ppc128RoundKeys,
  initial_counter: &[u8; 16],
  data: &mut [u8],
  acc: u128,
  h_powers_rev_8: &[u128; 8],
) -> u128 {
  // SAFETY: the caller guarantees POWER8 crypto support.
  unsafe { gcm_ctr32_ghash_8x::<11, false>(&keys.rk, initial_counter, data, acc, h_powers_rev_8) }
}

/// Eight-block AES-128-GCM opening over whole 128-byte chunks; see [`gcm_ctr32_ghash_8x`].
///
/// # Safety
///
/// Caller must ensure POWER8 vector crypto support is available.
#[cfg(feature = "aes-gcm")]
#[target_feature(enable = "altivec,vsx,power8-vector,power8-crypto")]
pub(super) unsafe fn gcm_open_8x_128(
  keys: &Ppc128RoundKeys,
  initial_counter: &[u8; 16],
  data: &mut [u8],
  acc: u128,
  h_powers_rev_8: &[u128; 8],
) -> u128 {
  // SAFETY: the caller guarantees POWER8 crypto support.
  unsafe { gcm_ctr32_ghash_8x::<11, true>(&keys.rk, initial_counter, data, acc, h_powers_rev_8) }
}
