use core::{arch::asm, simd::i64x2};

/// AES-256 round keys stored as 15 × 128-bit vectors for POWER8 vcipher.
///
/// POWER8 vcipher expects round keys in big-endian byte order, which
/// matches our portable key schedule (stored as big-endian u32 words).
#[derive(Clone)]
#[repr(C, align(16))]
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

/// Load 16 bytes from `ptr` into `i64x2` in ISA byte order (big-endian AES state).
///
/// On ppc64le, `i64x2` element `[0]` maps to ISA doubleword 1 (bytes 8-15)
/// and element `[1]` maps to ISA doubleword 0 (bytes 0-7). So to place
/// memory bytes `[0..8)` into ISA bytes `[0..8)` (high doubleword), we put
/// them into element `[1]`. Memory bytes `[8..16)` go into element `[0]`.
///
/// Pure-Rust approach avoids VSX `lxvd2x` asm which needs VSR register
/// numbers incompatible with the `vreg` register class.
#[inline]
fn load_block_be(ptr: *const u8) -> i64x2 {
  // SAFETY: Caller guarantees ptr is valid for 16 bytes.
  let bytes: [u8; 16] = unsafe { core::ptr::read_unaligned(ptr.cast()) };
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
fn store_block_be(ptr: *mut u8, block: i64x2) {
  let elems = block.to_array();
  #[cfg(target_endian = "little")]
  let (hi, lo) = (elems[1].to_be_bytes(), elems[0].to_be_bytes());
  #[cfg(target_endian = "big")]
  let (hi, lo) = (elems[0].to_be_bytes(), elems[1].to_be_bytes());
  let mut bytes = [0u8; 16];
  bytes[0..8].copy_from_slice(&hi);
  bytes[8..16].copy_from_slice(&lo);
  // SAFETY: Caller guarantees ptr is valid for 16 bytes.
  unsafe { core::ptr::write_unaligned(ptr.cast(), bytes) };
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
    keys[i] = load_block_be(bytes.as_ptr());
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
  let state = load_block_be(bytes.as_ptr());
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
    store_block_be(out_bytes.as_mut_ptr(), out);
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

    let mut state = load_block_be(block.as_ptr());

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

    store_block_be(block.as_mut_ptr(), state);
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
