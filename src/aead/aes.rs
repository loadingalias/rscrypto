#![allow(clippy::indexing_slicing)]

//! Portable constant-time AES block cipher core with hardware dispatch.
//!
//! This module provides AES-128 and AES-256 key expansion and single-block
//! encryption for use by AES-based AEAD constructions (GCM, GCM-SIV). All
//! operations are constant-time: no table lookups indexed by secret data.
//!
//! On x86_64 with AES-NI, aarch64 with AES-CE, or RISC-V with scalar/vector
//! crypto AES, the hardware path is selected at key-expansion time. The
//! portable S-box uses algebraic inversion in GF(2^8) via the Fermat power
//! chain (x^254) and constant-time field arithmetic, avoiding any lookup
//! tables that could leak through cache timing.

/// AES block size in bytes.
pub(crate) const BLOCK_SIZE: usize = 16;

/// AES-256 key size in bytes.
pub(crate) const KEY_SIZE: usize = 32;

/// AES-128 key size in bytes.
pub(crate) const KEY_SIZE_128: usize = 16;

/// Number of rounds for AES-256.
const ROUNDS: usize = 14;

/// Number of rounds for AES-128.
pub(crate) const ROUNDS_128: usize = 10;

/// Number of 32-bit words in the AES-256 expanded key schedule.
const EXPANDED_KEY_WORDS: usize = 4 * (ROUNDS + 1); // 60

/// Number of 32-bit words in the AES-128 expanded key schedule.
pub(crate) const EXPANDED_KEY_WORDS_128: usize = 4 * (ROUNDS_128 + 1); // 44

// ---------------------------------------------------------------------------
// x86_64 AES-NI backend
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
#[path = "aes/aarch64_ce.rs"]
mod ce;
#[cfg(target_arch = "s390x")]
#[allow(unsafe_code)]
#[path = "aes/s390x_km.rs"]
mod km;
#[cfg(target_arch = "x86_64")]
#[path = "aes/x86_64_ni.rs"]
mod ni;
#[cfg(target_arch = "powerpc64")]
#[allow(unsafe_code)]
#[path = "aes/powerpc64_ppc.rs"]
mod ppc;
#[cfg(target_arch = "riscv64")]
#[allow(unsafe_code)]
#[path = "aes/riscv64_aes.rs"]
mod rv_aes;
#[cfg(any(target_arch = "riscv64", test))]
#[path = "aes/riscv64_fixslice_aes.rs"]
mod rv_fixslice_aes;
#[cfg(target_arch = "riscv64")]
#[allow(unsafe_code)]
#[path = "aes/riscv64_scalar_aes.rs"]
mod rv_scalar_aes;
#[cfg(target_arch = "riscv64")]
#[allow(unsafe_code)]
#[path = "aes/riscv64_vperm_aes.rs"]
mod rv_vperm_aes;
#[cfg(all(target_arch = "x86_64", target_os = "linux", feature = "aes-gcm"))]
#[path = "aes/x86_64/asm.rs"]
mod x86_64_asm;
#[cfg(all(target_arch = "aarch64", feature = "aes-gcm"))]
pub(crate) struct Aarch64GcmTables<'a> {
  pub(crate) h_polyval: u128,
  pub(crate) h_powers_rev: &'a [u128; 4],
  pub(crate) h_powers_rev_8: &'a [u128; 8],
  #[cfg(target_os = "macos")]
  pub(crate) h_powers_rev_16: &'a [u128; 16],
  #[cfg(target_os = "macos")]
  pub(crate) h_powers_rev_16_mid: &'a [u128; 16],
  #[cfg(target_os = "macos")]
  pub(crate) h_powers_rev_16_pair: &'a [u128; 24],
}

// ---------------------------------------------------------------------------
// Aes256EncKey: enum-dispatched key storage
// ---------------------------------------------------------------------------

/// AES-256 expanded round keys.
///
/// On x86_64 with AES-NI, stores round keys as `__m128i`; on aarch64 with
/// AES-CE, stores round keys as `uint8x16_t`; on s390x with MSA, stores the
/// raw 32-byte key for the KM instruction; on powerpc64 with POWER8 crypto,
/// stores round keys as 128-bit vectors. Otherwise stores 60 big-endian
/// u32 words for the portable path. Zeroized on drop.
#[derive(Clone)]
pub(crate) struct Aes256EncKey {
  inner: KeyInner,
}

#[derive(Clone)]
enum KeyInner {
  #[allow(dead_code)]
  PortableRoundKeys([u32; EXPANDED_KEY_WORDS]),
  #[cfg(target_arch = "x86_64")]
  X86AesNi(ni::NiRoundKeys),
  #[cfg(target_arch = "aarch64")]
  Aarch64Aes(ce::CeRoundKeys),
  #[cfg(target_arch = "s390x")]
  S390xMsa(km::KmKey),
  #[cfg(target_arch = "powerpc64")]
  Power8Crypto(ppc::PpcRoundKeys),
  #[cfg(target_arch = "riscv64")]
  Riscv64ScalarCrypto(rv_scalar_aes::RvScalarRoundKeys),
  #[cfg(target_arch = "riscv64")]
  Riscv64VectorCrypto(rv_aes::RvRoundKeys),
  /// Hamburg vperm via vrgather.vv -- uses portable key schedule, constant-time.
  #[cfg(target_arch = "riscv64")]
  #[allow(dead_code)] // V-only AES is kept for GCM-SIV and explicit diagnostic paths; GCM does not select it yet.
  Riscv64Vperm([u32; EXPANDED_KEY_WORDS]),
  /// Four-block table-free fixslice fallback for scalar RV64 without AES extensions.
  #[cfg(all(target_arch = "riscv64", feature = "alloc"))]
  Riscv64Fixslice(alloc::boxed::Box<rv_fixslice_aes::RvFixsliceRoundKeys>),
  /// No-alloc RV64 builds keep the larger fixslice key schedule inline.
  #[cfg(all(target_arch = "riscv64", not(feature = "alloc")))]
  Riscv64Fixslice(rv_fixslice_aes::RvFixsliceRoundKeys),
}

impl Drop for Aes256EncKey {
  fn drop(&mut self) {
    match &mut self.inner {
      KeyInner::PortableRoundKeys(rk) => {
        // SAFETY: [u32; 60] is layout-compatible with [u8; 240].
        crate::traits::ct::zeroize(unsafe {
          core::slice::from_raw_parts_mut(rk.as_mut_ptr().cast::<u8>(), EXPANDED_KEY_WORDS.strict_mul(4))
        });
      }
      #[cfg(target_arch = "x86_64")]
      KeyInner::X86AesNi(ni_rk) => {
        ni_rk.zeroize();
      }
      #[cfg(target_arch = "aarch64")]
      KeyInner::Aarch64Aes(ce_rk) => {
        ce_rk.zeroize();
      }
      #[cfg(target_arch = "s390x")]
      KeyInner::S390xMsa(km_key) => {
        km_key.zeroize();
      }
      #[cfg(target_arch = "powerpc64")]
      KeyInner::Power8Crypto(ppc_rk) => {
        ppc_rk.zeroize();
      }
      #[cfg(target_arch = "riscv64")]
      KeyInner::Riscv64ScalarCrypto(rv_rk) => {
        rv_rk.zeroize();
      }
      #[cfg(target_arch = "riscv64")]
      KeyInner::Riscv64VectorCrypto(rv_rk) => {
        rv_rk.zeroize();
      }
      #[cfg(target_arch = "riscv64")]
      KeyInner::Riscv64Vperm(rk) => {
        // SAFETY: the expanded key is a contiguous `[u32; 60]`, so viewing it as
        // a mutable byte slice for zeroization is valid for its exact size.
        crate::traits::ct::zeroize(unsafe {
          core::slice::from_raw_parts_mut(rk.as_mut_ptr().cast::<u8>(), EXPANDED_KEY_WORDS.strict_mul(4))
        });
      }
      #[cfg(target_arch = "riscv64")]
      KeyInner::Riscv64Fixslice(rk) => {
        rk.zeroize();
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Aes128EncKey: enum-dispatched key storage
// ---------------------------------------------------------------------------

/// AES-128 expanded round keys.
///
/// Per-backend representation mirrors [`Aes256EncKey`] but stores 11 round
/// keys (176 bytes) instead of 15. Zeroized on drop.
#[derive(Clone)]
pub(crate) struct Aes128EncKey {
  inner: Key128Inner,
}

#[derive(Clone)]
enum Key128Inner {
  #[allow(dead_code)]
  PortableRoundKeys([u32; EXPANDED_KEY_WORDS_128]),
  #[cfg(target_arch = "x86_64")]
  X86AesNi(ni::Ni128RoundKeys),
  #[cfg(target_arch = "aarch64")]
  Aarch64Aes(ce::Ce128RoundKeys),
  #[cfg(target_arch = "s390x")]
  S390xMsa(km::Km128Key),
  #[cfg(target_arch = "powerpc64")]
  Power8Crypto(ppc::Ppc128RoundKeys),
  #[cfg(target_arch = "riscv64")]
  Riscv64ScalarCrypto(rv_scalar_aes::RvScalar128RoundKeys),
  #[cfg(target_arch = "riscv64")]
  Riscv64VectorCrypto(rv_aes::Rv128RoundKeys),
  /// Hamburg vperm via vrgather.vv -- uses portable key schedule, constant-time.
  #[cfg(target_arch = "riscv64")]
  #[allow(dead_code)] // V-only AES is kept for GCM-SIV and explicit diagnostic paths; GCM does not select it yet.
  Riscv64Vperm([u32; EXPANDED_KEY_WORDS_128]),
  /// Four-block table-free fixslice fallback for scalar RV64 without AES extensions.
  #[cfg(all(target_arch = "riscv64", feature = "alloc"))]
  Riscv64Fixslice(alloc::boxed::Box<rv_fixslice_aes::RvFixslice128RoundKeys>),
  /// No-alloc RV64 builds keep the larger fixslice key schedule inline.
  #[cfg(all(target_arch = "riscv64", not(feature = "alloc")))]
  Riscv64Fixslice(rv_fixslice_aes::RvFixslice128RoundKeys),
}

impl Drop for Aes128EncKey {
  fn drop(&mut self) {
    match &mut self.inner {
      Key128Inner::PortableRoundKeys(rk) => {
        // SAFETY: [u32; 44] is layout-compatible with [u8; 176].
        crate::traits::ct::zeroize(unsafe {
          core::slice::from_raw_parts_mut(rk.as_mut_ptr().cast::<u8>(), EXPANDED_KEY_WORDS_128.strict_mul(4))
        });
      }
      #[cfg(target_arch = "x86_64")]
      Key128Inner::X86AesNi(ni_rk) => {
        ni_rk.zeroize();
      }
      #[cfg(target_arch = "aarch64")]
      Key128Inner::Aarch64Aes(ce_rk) => {
        ce_rk.zeroize();
      }
      #[cfg(target_arch = "s390x")]
      Key128Inner::S390xMsa(km_key) => {
        km_key.zeroize();
      }
      #[cfg(target_arch = "powerpc64")]
      Key128Inner::Power8Crypto(ppc_rk) => {
        ppc_rk.zeroize();
      }
      #[cfg(target_arch = "riscv64")]
      Key128Inner::Riscv64ScalarCrypto(rv_rk) => {
        rv_rk.zeroize();
      }
      #[cfg(target_arch = "riscv64")]
      Key128Inner::Riscv64VectorCrypto(rv_rk) => {
        rv_rk.zeroize();
      }
      #[cfg(target_arch = "riscv64")]
      Key128Inner::Riscv64Vperm(rk) => {
        // SAFETY: [u32; 44] is layout-compatible with [u8; 176].
        crate::traits::ct::zeroize(unsafe {
          core::slice::from_raw_parts_mut(rk.as_mut_ptr().cast::<u8>(), EXPANDED_KEY_WORDS_128.strict_mul(4))
        });
      }
      #[cfg(target_arch = "riscv64")]
      Key128Inner::Riscv64Fixslice(rk) => {
        rk.zeroize();
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Constant-time GF(2^8) arithmetic for the AES S-box
// ---------------------------------------------------------------------------

/// Multiply two elements in GF(2^8) mod the AES irreducible polynomial
/// p(x) = x^8 + x^4 + x^3 + x + 1 (0x11b).
///
/// Constant-time: fixed iteration count, no secret-dependent branches.
#[inline(always)]
const fn gf256_mul(a: u8, b: u8) -> u8 {
  // Schoolbook carryless multiply into a u16, then reduce.
  let a = a as u16;
  let b = b as u16;

  // Accumulate partial products (unrolled, constant-time).
  let mut prod: u16 = 0;
  prod ^= a.wrapping_mul(b & 1);
  prod ^= (a << 1).wrapping_mul((b >> 1) & 1);
  prod ^= (a << 2).wrapping_mul((b >> 2) & 1);
  prod ^= (a << 3).wrapping_mul((b >> 3) & 1);
  prod ^= (a << 4).wrapping_mul((b >> 4) & 1);
  prod ^= (a << 5).wrapping_mul((b >> 5) & 1);
  prod ^= (a << 6).wrapping_mul((b >> 6) & 1);
  prod ^= (a << 7).wrapping_mul((b >> 7) & 1);

  // Reduce modulo x^8 + x^4 + x^3 + x + 1 (0x11b).
  // Process bits 14 down to 8.
  prod ^= (prod >> 14).wrapping_mul(0x11b << 6);
  prod ^= (prod >> 13).wrapping_mul(0x11b << 5);
  prod ^= (prod >> 12).wrapping_mul(0x11b << 4);
  prod ^= (prod >> 11).wrapping_mul(0x11b << 3);
  prod ^= (prod >> 10).wrapping_mul(0x11b << 2);
  prod ^= (prod >> 9).wrapping_mul(0x11b << 1);
  prod ^= (prod >> 8).wrapping_mul(0x11b);

  prod as u8
}

/// Square in GF(2^8). Equivalent to `gf256_mul(x, x)` but slightly cheaper.
#[inline(always)]
const fn gf256_sq(x: u8) -> u8 {
  gf256_mul(x, x)
}

/// Compute x^(-1) in GF(2^8) via the Fermat power chain: x^254.
///
/// Returns 0 for input 0 (matching the AES S-box convention).
/// Constant-time: always executes the same operations regardless of input.
#[inline(always)]
const fn gf256_inv(x: u8) -> u8 {
  // Addition chain for 254 = 2+4+8+16+32+64+128:
  //   x^2, x^3, x^6, x^12, x^14, x^15, x^30, x^60, x^62, x^63,
  //   x^126, x^252, x^254
  let x2 = gf256_sq(x); // x^2
  let x3 = gf256_mul(x2, x); // x^3
  let x6 = gf256_sq(x3); // x^6
  let x12 = gf256_sq(x6); // x^12
  let x14 = gf256_mul(x12, x2); // x^14
  let x15 = gf256_mul(x14, x); // x^15
  let x30 = gf256_sq(x15); // x^30
  let x60 = gf256_sq(x30); // x^60
  let x62 = gf256_mul(x60, x2); // x^62
  let x63 = gf256_mul(x62, x); // x^63
  let x126 = gf256_sq(x63); // x^126
  let x252 = gf256_sq(x126); // x^252
  gf256_mul(x252, x2) // x^254
}

/// AES forward S-box: S(x) = affine(x^{-1}).
///
/// Computes the inverse in GF(2^8), then applies the AES affine transform.
/// Constant-time: no table lookups, fixed operations.
#[inline(always)]
const fn sbox(x: u8) -> u8 {
  let inv = gf256_inv(x);

  // Affine transform over GF(2):
  // s_i = b_i XOR b_{(i+4) mod 8} XOR b_{(i+5) mod 8}
  //       XOR b_{(i+6) mod 8} XOR b_{(i+7) mod 8} XOR c_i
  // where c = 0x63
  let r = inv ^ inv.rotate_left(1) ^ inv.rotate_left(2) ^ inv.rotate_left(3) ^ inv.rotate_left(4);
  r ^ 0x63
}

/// Apply SubBytes to a 32-bit word (four S-box applications).
#[inline(always)]
const fn sub_word(w: u32) -> u32 {
  let b0 = sbox((w >> 24) as u8) as u32;
  let b1 = sbox((w >> 16) as u8) as u32;
  let b2 = sbox((w >> 8) as u8) as u32;
  let b3 = sbox(w as u8) as u32;
  (b0 << 24) | (b1 << 16) | (b2 << 8) | b3
}

/// Rotate a 32-bit word left by 8 bits.
#[inline(always)]
const fn rot_word(w: u32) -> u32 {
  w.rotate_left(8)
}

// ---------------------------------------------------------------------------
// AES round constants
// ---------------------------------------------------------------------------

/// AES key schedule round constants (rcon).
/// Only the high byte is nonzero: rcon[i] = (rc[i], 0, 0, 0).
const RCON: [u32; 10] = [
  0x0100_0000,
  0x0200_0000,
  0x0400_0000,
  0x0800_0000,
  0x1000_0000,
  0x2000_0000,
  0x4000_0000,
  0x8000_0000,
  0x1b00_0000,
  0x3600_0000,
];

// ---------------------------------------------------------------------------
// Key expansion
// ---------------------------------------------------------------------------

/// Portable AES-256 key expansion into 60 big-endian u32 words.
#[inline]
fn aes256_expand_key_portable(key: &[u8; KEY_SIZE]) -> [u32; EXPANDED_KEY_WORDS] {
  let mut rk = [0u32; EXPANDED_KEY_WORDS];

  // Load the initial key as 8 big-endian words.
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

  // Expand.
  i = 8;
  while i < EXPANDED_KEY_WORDS {
    let mut temp = rk[i.strict_sub(1)];
    if i.strict_rem(8) == 0 {
      temp = sub_word(rot_word(temp)) ^ RCON[i.strict_div(8).strict_sub(1)];
    } else if i.strict_rem(8) == 4 {
      temp = sub_word(temp);
    }
    rk[i] = rk[i.strict_sub(8)] ^ temp;
    i = i.strict_add(1);
  }

  rk
}

#[cfg(target_arch = "riscv64")]
#[inline]
fn zeroize_expanded_key_words(rk: &mut [u32; EXPANDED_KEY_WORDS]) {
  // SAFETY: [u32; 60] is layout-compatible with [u8; 240].
  crate::traits::ct::zeroize(unsafe {
    core::slice::from_raw_parts_mut(rk.as_mut_ptr().cast::<u8>(), EXPANDED_KEY_WORDS.strict_mul(4))
  });
}

#[cfg(target_arch = "riscv64")]
#[inline]
fn zeroize_expanded_key_words_128(rk: &mut [u32; EXPANDED_KEY_WORDS_128]) {
  // SAFETY: [u32; 44] is layout-compatible with [u8; 176].
  crate::traits::ct::zeroize(unsafe {
    core::slice::from_raw_parts_mut(rk.as_mut_ptr().cast::<u8>(), EXPANDED_KEY_WORDS_128.strict_mul(4))
  });
}

/// Portable AES-128 key expansion into 44 big-endian u32 words.
#[inline]
pub(crate) fn aes128_expand_key_portable(key: &[u8; KEY_SIZE_128]) -> [u32; EXPANDED_KEY_WORDS_128] {
  let mut rk = [0u32; EXPANDED_KEY_WORDS_128];

  // Load the initial key as 4 big-endian words.
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

  // Expand. AES-128 schedule has period 4: only the i % 4 == 0 case
  // applies SubWord(RotWord(temp)) ^ RCON, in contrast to AES-256's
  // dual SubWord at i % 8 == 0 || i % 8 == 4.
  i = 4;
  while i < EXPANDED_KEY_WORDS_128 {
    let mut temp = rk[i.strict_sub(1)];
    if i.strict_rem(4) == 0 {
      temp = sub_word(rot_word(temp)) ^ RCON[i.strict_div(4).strict_sub(1)];
    }
    rk[i] = rk[i.strict_sub(4)] ^ temp;
    i = i.strict_add(1);
  }

  rk
}

#[cfg(all(target_arch = "riscv64", feature = "alloc"))]
#[inline]
fn riscv64_fixslice_key_inner(key: &[u8; KEY_SIZE]) -> KeyInner {
  KeyInner::Riscv64Fixslice(alloc::boxed::Box::new(rv_fixslice_aes::RvFixsliceRoundKeys::new(key)))
}

#[cfg(all(target_arch = "riscv64", not(feature = "alloc")))]
#[inline]
fn riscv64_fixslice_key_inner(key: &[u8; KEY_SIZE]) -> KeyInner {
  KeyInner::Riscv64Fixslice(rv_fixslice_aes::RvFixsliceRoundKeys::new(key))
}

/// Expand a 256-bit AES key into round keys.
///
/// On x86_64 with AES-NI or aarch64 with AES-CE, converts to hardware-native
/// round key format at expansion time. Otherwise uses the portable path.
#[inline]
pub(crate) fn aes256_expand_key(key: &[u8; KEY_SIZE]) -> Aes256EncKey {
  #[cfg(target_arch = "x86_64")]
  {
    if crate::platform::caps().has(crate::platform::caps::x86::AESNI) {
      return Aes256EncKey {
        // SAFETY: AES-NI availability verified via CPUID above.
        inner: KeyInner::X86AesNi(unsafe { ni::expand_key(key) }),
      };
    }
  }
  #[cfg(target_arch = "aarch64")]
  {
    if crate::platform::caps().has(crate::platform::caps::aarch64::AES) {
      return Aes256EncKey {
        // SAFETY: AES-CE availability verified via HWCAP above.
        // Uses AESE for SubWord — ~750x faster than the algebraic GF(2^8) S-box.
        inner: KeyInner::Aarch64Aes(unsafe { ce::expand_key(key) }),
      };
    }
  }
  #[cfg(target_arch = "s390x")]
  {
    if crate::platform::caps().has(crate::platform::caps::s390x::MSA) {
      return Aes256EncKey {
        inner: KeyInner::S390xMsa(km::KmKey::from_portable(aes256_expand_key_portable(key))),
      };
    }
  }
  #[cfg(target_arch = "powerpc64")]
  {
    if crate::platform::caps().has(crate::platform::caps::power::POWER8_CRYPTO) {
      return Aes256EncKey {
        // SAFETY: POWER8 crypto availability verified via HWCAP above.
        inner: KeyInner::Power8Crypto(unsafe { ppc::expand_key(key) }),
      };
    }
  }
  #[cfg(target_arch = "riscv64")]
  {
    if crate::platform::caps().has(crate::platform::caps::riscv::ZVKNED) {
      let mut portable_rk = aes256_expand_key_portable(key);
      let rv_keys = rv_aes::from_portable(&portable_rk);
      zeroize_expanded_key_words(&mut portable_rk);
      return Aes256EncKey {
        inner: KeyInner::Riscv64VectorCrypto(rv_keys),
      };
    }
    if crate::platform::caps().has(crate::platform::caps::riscv::ZKNE) {
      let mut portable_rk = aes256_expand_key_portable(key);
      let rv_keys = rv_scalar_aes::from_portable(&portable_rk);
      zeroize_expanded_key_words(&mut portable_rk);
      return Aes256EncKey {
        inner: KeyInner::Riscv64ScalarCrypto(rv_keys),
      };
    }
    Aes256EncKey {
      inner: riscv64_fixslice_key_inner(key),
    }
  }
  #[cfg(not(target_arch = "riscv64"))]
  Aes256EncKey {
    inner: KeyInner::PortableRoundKeys(aes256_expand_key_portable(key)),
  }
}

#[cfg(all(target_arch = "riscv64", feature = "alloc"))]
#[inline]
fn riscv64_fixslice_key_inner_128(key: &[u8; KEY_SIZE_128]) -> Key128Inner {
  Key128Inner::Riscv64Fixslice(alloc::boxed::Box::new(rv_fixslice_aes::RvFixslice128RoundKeys::new(
    key,
  )))
}

#[cfg(all(target_arch = "riscv64", not(feature = "alloc")))]
#[inline]
fn riscv64_fixslice_key_inner_128(key: &[u8; KEY_SIZE_128]) -> Key128Inner {
  Key128Inner::Riscv64Fixslice(rv_fixslice_aes::RvFixslice128RoundKeys::new(key))
}

/// Expand a 128-bit AES key into round keys.
///
/// Mirrors [`aes256_expand_key`]: each backend converts to its native
/// hardware round-key format at expansion time when its capability is
/// detected; otherwise the constant-time portable schedule is used.
#[inline]
pub(crate) fn aes128_expand_key(key: &[u8; KEY_SIZE_128]) -> Aes128EncKey {
  #[cfg(target_arch = "x86_64")]
  {
    if crate::platform::caps().has(crate::platform::caps::x86::AESNI) {
      return Aes128EncKey {
        // SAFETY: AES-NI availability verified via CPUID above.
        inner: Key128Inner::X86AesNi(unsafe { ni::expand_key_128(key) }),
      };
    }
  }
  #[cfg(target_arch = "aarch64")]
  {
    if crate::platform::caps().has(crate::platform::caps::aarch64::AES) {
      return Aes128EncKey {
        // SAFETY: AES-CE availability verified via HWCAP above.
        inner: Key128Inner::Aarch64Aes(unsafe { ce::expand_key_128(key) }),
      };
    }
  }
  #[cfg(target_arch = "s390x")]
  {
    if crate::platform::caps().has(crate::platform::caps::s390x::MSA) {
      return Aes128EncKey {
        inner: Key128Inner::S390xMsa(km::Km128Key::from_portable(aes128_expand_key_portable(key))),
      };
    }
  }
  #[cfg(target_arch = "powerpc64")]
  {
    if crate::platform::caps().has(crate::platform::caps::power::POWER8_CRYPTO) {
      return Aes128EncKey {
        // SAFETY: POWER8 crypto availability verified via HWCAP above.
        inner: Key128Inner::Power8Crypto(unsafe { ppc::expand_key_128(key) }),
      };
    }
  }
  #[cfg(target_arch = "riscv64")]
  {
    if crate::platform::caps().has(crate::platform::caps::riscv::ZVKNED) {
      let mut portable_rk = aes128_expand_key_portable(key);
      let rv_keys = rv_aes::from_portable_128(&portable_rk);
      zeroize_expanded_key_words_128(&mut portable_rk);
      return Aes128EncKey {
        inner: Key128Inner::Riscv64VectorCrypto(rv_keys),
      };
    }
    if crate::platform::caps().has(crate::platform::caps::riscv::ZKNE) {
      let mut portable_rk = aes128_expand_key_portable(key);
      let rv_keys = rv_scalar_aes::from_portable_128(&portable_rk);
      zeroize_expanded_key_words_128(&mut portable_rk);
      return Aes128EncKey {
        inner: Key128Inner::Riscv64ScalarCrypto(rv_keys),
      };
    }
    Aes128EncKey {
      inner: riscv64_fixslice_key_inner_128(key),
    }
  }
  #[cfg(not(target_arch = "riscv64"))]
  Aes128EncKey {
    inner: Key128Inner::PortableRoundKeys(aes128_expand_key_portable(key)),
  }
}

#[cfg(all(target_arch = "riscv64", feature = "aes-gcm-siv"))]
#[inline]
pub(crate) fn aes256_expand_key_riscv_vector(key: &[u8; KEY_SIZE]) -> Aes256EncKey {
  let mut portable_rk = aes256_expand_key_portable(key);
  let rv_keys = rv_aes::from_portable(&portable_rk);
  zeroize_expanded_key_words(&mut portable_rk);
  Aes256EncKey {
    inner: KeyInner::Riscv64VectorCrypto(rv_keys),
  }
}

#[cfg(all(target_arch = "riscv64", feature = "aes-gcm-siv"))]
#[inline]
pub(crate) fn aes256_expand_key_riscv_scalar(key: &[u8; KEY_SIZE]) -> Aes256EncKey {
  let mut portable_rk = aes256_expand_key_portable(key);
  let rv_keys = rv_scalar_aes::from_portable(&portable_rk);
  zeroize_expanded_key_words(&mut portable_rk);
  Aes256EncKey {
    inner: KeyInner::Riscv64ScalarCrypto(rv_keys),
  }
}

#[cfg(all(target_arch = "riscv64", feature = "aes-gcm-siv"))]
#[inline]
pub(crate) fn aes256_expand_key_riscv_vperm(key: &[u8; KEY_SIZE]) -> Aes256EncKey {
  Aes256EncKey {
    inner: KeyInner::Riscv64Vperm(aes256_expand_key_portable(key)),
  }
}

#[cfg(all(target_arch = "riscv64", feature = "aes-gcm-siv"))]
#[inline]
pub(crate) fn aes256_expand_key_riscv_ttable(key: &[u8; KEY_SIZE]) -> Aes256EncKey {
  // Legacy helper retained for callers that still mention the old backend
  // label. The implementation now uses the constant-time RV64 fixslice path.
  Aes256EncKey {
    inner: riscv64_fixslice_key_inner(key),
  }
}

#[cfg(all(target_arch = "riscv64", feature = "aes-gcm-siv"))]
#[inline]
pub(crate) fn aes128_expand_key_riscv_vector(key: &[u8; KEY_SIZE_128]) -> Aes128EncKey {
  let mut portable_rk = aes128_expand_key_portable(key);
  let rv_keys = rv_aes::from_portable_128(&portable_rk);
  zeroize_expanded_key_words_128(&mut portable_rk);
  Aes128EncKey {
    inner: Key128Inner::Riscv64VectorCrypto(rv_keys),
  }
}

#[cfg(all(target_arch = "riscv64", feature = "aes-gcm-siv"))]
#[inline]
pub(crate) fn aes128_expand_key_riscv_scalar(key: &[u8; KEY_SIZE_128]) -> Aes128EncKey {
  let mut portable_rk = aes128_expand_key_portable(key);
  let rv_keys = rv_scalar_aes::from_portable_128(&portable_rk);
  zeroize_expanded_key_words_128(&mut portable_rk);
  Aes128EncKey {
    inner: Key128Inner::Riscv64ScalarCrypto(rv_keys),
  }
}

#[cfg(all(target_arch = "riscv64", feature = "aes-gcm-siv"))]
#[inline]
pub(crate) fn aes128_expand_key_riscv_vperm(key: &[u8; KEY_SIZE_128]) -> Aes128EncKey {
  Aes128EncKey {
    inner: Key128Inner::Riscv64Vperm(aes128_expand_key_portable(key)),
  }
}

#[cfg(all(target_arch = "riscv64", feature = "aes-gcm-siv"))]
#[inline]
pub(crate) fn aes128_expand_key_riscv_ttable(key: &[u8; KEY_SIZE_128]) -> Aes128EncKey {
  // Legacy helper retained for parity with AES-256. The implementation now
  // uses the constant-time RV64 fixslice path.
  Aes128EncKey {
    inner: riscv64_fixslice_key_inner_128(key),
  }
}

// ---------------------------------------------------------------------------
// aarch64: inline helpers for fused paths (#[target_feature] + #[inline(always)])
// ---------------------------------------------------------------------------
//
// aarch64 intrinsics are `#[inline(always)]` with `#[target_feature]`, so
// callers must also have matching features (unlike x86). We use both
// `#[target_feature]` (for intrinsic compatibility) and `#[inline(always)]`
// (to guarantee inlining → no function boundary → no register spills).

/// Expand AES-256 key directly to AES-CE round keys.
///
/// This is the hot-path helper used from fused
/// `#[target_feature(enable = "aes,neon")]` scopes.
///
/// # Safety
/// Caller must ensure AES-CE is available.
#[cfg(all(target_arch = "aarch64", feature = "aes-gcm-siv"))]
#[target_feature(enable = "aes,neon")]
#[inline]
pub(super) unsafe fn aarch64_expand_key_inline(key: &[u8; KEY_SIZE]) -> ce::CeRoundKeys {
  // SAFETY: AES-CE key expansion because:
  // 1. This helper is only callable from an `aes,neon` target-feature scope.
  // 2. The caller guarantees the current CPU supports AES-CE.
  // 3. `key` is a fixed 32-byte AES-256 key.
  unsafe { ce::expand_key_hw(key) }
}

/// Encrypt a single AES-256 block with AES-CE.
///
/// # Safety
/// Caller must ensure AES-CE is available.
#[cfg(all(target_arch = "aarch64", feature = "aes-gcm-siv"))]
#[target_feature(enable = "aes,neon")]
#[inline]
pub(super) unsafe fn aarch64_encrypt_block_inline(keys: &ce::CeRoundKeys, block: &mut [u8; BLOCK_SIZE]) {
  // SAFETY: AES-CE block encryption because:
  // 1. This helper is only callable from an `aes,neon` target-feature scope.
  // 2. The caller guarantees the current CPU supports AES-CE.
  // 3. `block` is exactly one initialized 16-byte AES block.
  unsafe { ce::encrypt_block_core(keys, block) }
}

/// Encrypt multiple AES-256 blocks inside one AES-CE target scope.
///
/// # Safety
/// Caller must ensure AES-CE is available.
#[cfg(all(target_arch = "aarch64", any(feature = "aes-gcm", feature = "aes-gcm-siv")))]
#[target_feature(enable = "aes,neon")]
#[inline]
unsafe fn aarch64_encrypt_blocks_inline(keys: &ce::CeRoundKeys, blocks: &mut [[u8; BLOCK_SIZE]]) {
  if blocks.len() == 6 {
    // SAFETY: exact six-block AES-256 batch because:
    // 1. The caller guarantees AES-CE availability for this target-feature scope.
    // 2. `blocks.len() == 6`, so the slice is exactly one contiguous six-block array.
    // 3. `[[u8; 16]]` elements have no padding and remain valid under the fixed-size array view.
    let batch = unsafe { &mut *blocks.as_mut_ptr().cast::<[[u8; BLOCK_SIZE]; 6]>() };
    // SAFETY: AES-CE batch encryption because:
    // 1. This function's caller must guarantee AES-CE availability.
    // 2. `batch` is a valid mutable reference to six initialized AES blocks.
    unsafe { ce::encrypt_6blocks_core(keys, batch) };
    return;
  }

  let mut offset = 0usize;
  while offset.strict_add(4) <= blocks.len() {
    let batch_slice = &mut blocks[offset..offset.strict_add(4)];
    // SAFETY: exact four-block AES-256 batch because:
    // 1. `batch_slice` was sliced to length 4 immediately above.
    // 2. `[[u8; 16]]` elements are contiguous and have no padding.
    // 3. The mutable borrow is limited to this loop iteration.
    let batch = unsafe { &mut *batch_slice.as_mut_ptr().cast::<[[u8; BLOCK_SIZE]; 4]>() };
    // SAFETY: AES-CE batch encryption because:
    // 1. This function's caller must guarantee AES-CE availability.
    // 2. `batch` is a valid mutable reference to four initialized AES blocks.
    unsafe { ce::encrypt_4blocks_core(keys, batch) };
    offset = offset.strict_add(4);
  }

  while offset < blocks.len() {
    // SAFETY: AES-CE block encryption because:
    // 1. This function's caller must guarantee AES-CE availability.
    // 2. `blocks[offset]` is a valid mutable `[u8; 16]` inside the caller-owned slice.
    unsafe { ce::encrypt_block_core(keys, &mut blocks[offset]) };
    offset = offset.strict_add(1);
  }
}

/// Derive AES-256-GCM-SIV per-message keys directly with AES-CE.
///
/// # Safety
/// Caller must ensure AES-CE is available and `master_ek` is the AArch64 AES
/// backend variant.
#[cfg(all(target_arch = "aarch64", feature = "aes-gcm-siv"))]
#[target_feature(enable = "aes,neon")]
#[inline]
pub(super) unsafe fn aarch64_gcmsiv_derive_keys_inline(
  master_ek: &Aes256EncKey,
  nonce: &[u8; 12],
) -> ([u8; 16], [u8; 32]) {
  let KeyInner::Aarch64Aes(ce_rk) = &master_ek.inner else {
    unreachable!("AArch64 GCM-SIV KDF requires an AES-CE master key");
  };
  // SAFETY: direct AES-CE GCM-SIV KDF because:
  // 1. This function's caller must guarantee AES-CE availability.
  // 2. The matched key variant proves the master key schedule is AES-CE compatible.
  // 3. `nonce` is exactly the 96-bit GCM-SIV nonce.
  unsafe { ce::gcmsiv_derive_keys_core(ce_rk, nonce) }
}

/// XOR four AES-256 GCM-SIV CTR blocks inside one AES-CE target scope.
///
/// # Safety
/// Caller must ensure AES-CE is available and `data` has at least 64 bytes.
#[cfg(all(target_arch = "aarch64", feature = "aes-gcm-siv"))]
#[target_feature(enable = "aes,neon")]
#[inline]
pub(super) unsafe fn aarch64_ctr32_le_xor_4blocks_inline(
  keys: &ce::CeRoundKeys,
  iv_suffix: &[u8; 12],
  ctr: u32,
  data: &mut [u8],
) {
  // SAFETY: four-block AES-CE CTR XOR because:
  // 1. This helper is only callable from an `aes,neon` target-feature scope.
  // 2. The caller guarantees `data` has at least 64 writable bytes.
  // 3. `iv_suffix` is exactly the 96-bit GCM-SIV nonce suffix.
  unsafe { ce::encrypt_ctr32_le_xor_4blocks_core(keys, iv_suffix, ctr, data) }
}

/// XOR eight AES-256 GCM-SIV CTR blocks inside one AES-CE target scope.
///
/// # Safety
/// Caller must ensure AES-CE is available and `data` has at least 128 bytes.
#[cfg(all(target_arch = "aarch64", feature = "aes-gcm-siv"))]
#[target_feature(enable = "aes,neon")]
#[inline]
pub(super) unsafe fn aarch64_ctr32_le_xor_8blocks_inline(
  keys: &ce::CeRoundKeys,
  iv_suffix: &[u8; 12],
  ctr: u32,
  data: &mut [u8],
) {
  // SAFETY: eight-block AES-CE CTR XOR because:
  // 1. This helper is only callable from an `aes,neon` target-feature scope.
  // 2. The caller guarantees `data` has at least 128 writable bytes.
  // 3. `iv_suffix` is exactly the 96-bit GCM-SIV nonce suffix.
  unsafe { ce::encrypt_ctr32_le_xor_8blocks_core(keys, iv_suffix, ctr, data) }
}

/// Expand AES-128 key directly to AES-CE round keys.
///
/// AES-128 sibling of [`aarch64_expand_key_inline`]. Used by the fused
/// AES-128-GCM-SIV path on aarch64.
///
/// # Safety
/// Caller must ensure AES-CE is available.
#[cfg(all(target_arch = "aarch64", feature = "aes-gcm-siv"))]
#[target_feature(enable = "aes,neon")]
#[inline]
pub(super) unsafe fn aarch64_expand_key_128_inline(key: &[u8; KEY_SIZE_128]) -> ce::Ce128RoundKeys {
  // SAFETY: AES-CE AES-128 key expansion because:
  // 1. This helper is only callable from an `aes,neon` target-feature scope.
  // 2. The caller guarantees the current CPU supports AES-CE.
  // 3. `key` is a fixed 16-byte AES-128 key.
  unsafe { ce::expand_key_128(key) }
}

/// Encrypt a single AES-128 block with AES-CE.
///
/// # Safety
/// Caller must ensure AES-CE is available.
#[cfg(all(target_arch = "aarch64", feature = "aes-gcm-siv"))]
#[target_feature(enable = "aes,neon")]
#[inline]
pub(super) unsafe fn aarch64_encrypt_block_128_inline(keys: &ce::Ce128RoundKeys, block: &mut [u8; BLOCK_SIZE]) {
  // SAFETY: AES-CE AES-128 block encryption because:
  // 1. This helper is only callable from an `aes,neon` target-feature scope.
  // 2. The caller guarantees the current CPU supports AES-CE.
  // 3. `block` is exactly one initialized 16-byte AES block.
  unsafe { ce::encrypt_block_128_core(keys, block) }
}

/// Encrypt multiple AES-128 blocks inside one AES-CE target scope.
///
/// # Safety
/// Caller must ensure AES-CE is available.
#[cfg(all(target_arch = "aarch64", any(feature = "aes-gcm", feature = "aes-gcm-siv")))]
#[target_feature(enable = "aes,neon")]
#[inline]
unsafe fn aarch64_encrypt_blocks_128_inline(keys: &ce::Ce128RoundKeys, blocks: &mut [[u8; BLOCK_SIZE]]) {
  let mut offset = 0usize;
  while offset.strict_add(4) <= blocks.len() {
    let batch_slice = &mut blocks[offset..offset.strict_add(4)];
    // SAFETY: exact four-block AES-128 batch because:
    // 1. `batch_slice` was sliced to length 4 immediately above.
    // 2. `[[u8; 16]]` elements are contiguous and have no padding.
    // 3. The mutable borrow is limited to this loop iteration.
    let batch = unsafe { &mut *batch_slice.as_mut_ptr().cast::<[[u8; BLOCK_SIZE]; 4]>() };
    // SAFETY: AES-CE batch encryption because:
    // 1. This function's caller must guarantee AES-CE availability.
    // 2. `batch` is a valid mutable reference to four initialized AES blocks.
    unsafe { ce::encrypt_4blocks_128_core(keys, batch) };
    offset = offset.strict_add(4);
  }

  while offset < blocks.len() {
    // SAFETY: AES-CE block encryption because:
    // 1. This function's caller must guarantee AES-CE availability.
    // 2. `blocks[offset]` is a valid mutable `[u8; 16]` inside the caller-owned slice.
    unsafe { ce::encrypt_block_128_core(keys, &mut blocks[offset]) };
    offset = offset.strict_add(1);
  }
}

/// Derive AES-128-GCM-SIV per-message keys directly with AES-CE.
///
/// # Safety
/// Caller must ensure AES-CE is available and `master_ek` is the AArch64 AES
/// backend variant.
#[cfg(all(target_arch = "aarch64", feature = "aes-gcm-siv"))]
#[target_feature(enable = "aes,neon")]
#[inline]
pub(super) unsafe fn aarch64_gcmsiv_derive_keys_128_inline(
  master_ek: &Aes128EncKey,
  nonce: &[u8; 12],
) -> ([u8; 16], [u8; 16]) {
  let Key128Inner::Aarch64Aes(ce_rk) = &master_ek.inner else {
    unreachable!("AArch64 GCM-SIV KDF requires an AES-CE master key");
  };
  // SAFETY: direct AES-CE GCM-SIV KDF because:
  // 1. This function's caller must guarantee AES-CE availability.
  // 2. The matched key variant proves the master key schedule is AES-CE compatible.
  // 3. `nonce` is exactly the 96-bit GCM-SIV nonce.
  unsafe { ce::gcmsiv_derive_keys_128_core(ce_rk, nonce) }
}

/// XOR four AES-128 GCM-SIV CTR blocks inside one AES-CE target scope.
///
/// # Safety
/// Caller must ensure AES-CE is available and `data` has at least 64 bytes.
#[cfg(all(target_arch = "aarch64", feature = "aes-gcm-siv"))]
#[target_feature(enable = "aes,neon")]
#[inline]
pub(super) unsafe fn aarch64_ctr32_le_xor_4blocks_128_inline(
  keys: &ce::Ce128RoundKeys,
  iv_suffix: &[u8; 12],
  ctr: u32,
  data: &mut [u8],
) {
  // SAFETY: four-block AES-CE AES-128 CTR XOR because:
  // 1. This helper is only callable from an `aes,neon` target-feature scope.
  // 2. The caller guarantees `data` has at least 64 writable bytes.
  // 3. `iv_suffix` is exactly the 96-bit GCM-SIV nonce suffix.
  unsafe { ce::encrypt_ctr32_le_xor_4blocks_128_core(keys, iv_suffix, ctr, data) }
}

/// XOR eight AES-128 GCM-SIV CTR blocks inside one AES-CE target scope.
///
/// # Safety
/// Caller must ensure AES-CE is available and `data` has at least 128 bytes.
#[cfg(all(target_arch = "aarch64", feature = "aes-gcm-siv"))]
#[target_feature(enable = "aes,neon")]
#[inline]
pub(super) unsafe fn aarch64_ctr32_le_xor_8blocks_128_inline(
  keys: &ce::Ce128RoundKeys,
  iv_suffix: &[u8; 12],
  ctr: u32,
  data: &mut [u8],
) {
  // SAFETY: eight-block AES-CE AES-128 CTR XOR because:
  // 1. This helper is only callable from an `aes,neon` target-feature scope.
  // 2. The caller guarantees `data` has at least 128 writable bytes.
  // 3. `iv_suffix` is exactly the 96-bit GCM-SIV nonce suffix.
  unsafe { ce::encrypt_ctr32_le_xor_8blocks_128_core(keys, iv_suffix, ctr, data) }
}

// ---------------------------------------------------------------------------
// powerpc64: hot-path helpers for fused target-feature scopes
// ---------------------------------------------------------------------------

/// Expand AES-256 key directly to POWER round keys.
///
/// # Safety
/// Caller must ensure POWER8 crypto is available.
#[cfg(all(target_arch = "powerpc64", feature = "aes-gcm-siv"))]
#[target_feature(enable = "altivec,vsx,power8-vector,power8-crypto")]
#[inline]
pub(super) unsafe fn ppc_expand_key_inline(key: &[u8; KEY_SIZE]) -> ppc::PpcRoundKeys {
  // SAFETY: POWER8 AES-256 key expansion because:
  // 1. This helper is only callable from a POWER8 crypto target-feature scope.
  // 2. The caller guarantees the current CPU supports POWER8 crypto.
  // 3. `key` is a fixed 32-byte AES-256 key.
  unsafe { ppc::expand_key_hw(key) }
}

/// Encrypt a single AES-256 block with POWER crypto.
///
/// # Safety
/// Caller must ensure POWER8 crypto is available.
#[cfg(all(target_arch = "powerpc64", feature = "aes-gcm-siv"))]
#[target_feature(enable = "altivec,vsx,power8-vector,power8-crypto")]
#[inline]
pub(super) unsafe fn ppc_encrypt_block_inline(keys: &ppc::PpcRoundKeys, block: &mut [u8; BLOCK_SIZE]) {
  // SAFETY: POWER8 AES-256 block encryption because:
  // 1. This helper is only callable from a POWER8 crypto target-feature scope.
  // 2. The caller guarantees the current CPU supports POWER8 crypto.
  // 3. `block` is exactly one initialized 16-byte AES block.
  unsafe { ppc::encrypt_block_core(keys, block) }
}

/// Encrypt multiple AES-256 blocks inside one POWER crypto target scope.
///
/// # Safety
/// Caller must ensure POWER8 crypto is available.
#[cfg(all(target_arch = "powerpc64", any(feature = "aes-gcm", feature = "aes-gcm-siv")))]
#[target_feature(enable = "altivec,vsx,power8-vector,power8-crypto")]
#[inline]
unsafe fn ppc_encrypt_blocks_inline(keys: &ppc::PpcRoundKeys, blocks: &mut [[u8; BLOCK_SIZE]]) {
  let mut offset = 0usize;
  while offset.strict_add(4) <= blocks.len() {
    let batch_slice = &mut blocks[offset..offset.strict_add(4)];
    // SAFETY: exact four-block POWER AES batch because:
    // 1. `batch_slice` was sliced to length 4 immediately above.
    // 2. `[[u8; 16]]` elements are contiguous and have no padding.
    // 3. This function's caller guarantees POWER8 crypto availability.
    let batch = unsafe { &mut *batch_slice.as_mut_ptr().cast::<[[u8; BLOCK_SIZE]; 4]>() };
    // SAFETY: POWER8 four-block AES because:
    // 1. This function's caller must guarantee POWER8 crypto availability.
    // 2. `batch` is a valid mutable reference to four initialized AES blocks.
    unsafe { ppc::encrypt_4blocks_core(keys, batch) };
    offset = offset.strict_add(4);
  }

  while offset < blocks.len() {
    // SAFETY: POWER8 AES block encryption because:
    // 1. This function's caller must guarantee POWER8 crypto availability.
    // 2. `blocks[offset]` is a valid mutable `[u8; 16]` inside the caller-owned slice.
    unsafe { ppc::encrypt_block_core(keys, &mut blocks[offset]) };
    offset = offset.strict_add(1);
  }
}

/// Expand AES-128 key directly to POWER round keys.
///
/// AES-128 sibling of [`ppc_expand_key_inline`]. Used by the fused
/// AES-128-GCM-SIV path on powerpc64.
///
/// # Safety
/// Caller must ensure POWER8 crypto is available.
#[cfg(all(target_arch = "powerpc64", feature = "aes-gcm-siv"))]
#[target_feature(enable = "altivec,vsx,power8-vector,power8-crypto")]
#[inline]
pub(super) unsafe fn ppc_expand_key_128_inline(key: &[u8; KEY_SIZE_128]) -> ppc::Ppc128RoundKeys {
  // SAFETY: POWER8 AES-128 key expansion because:
  // 1. This helper is only callable from a POWER8 crypto target-feature scope.
  // 2. The caller guarantees the current CPU supports POWER8 crypto.
  // 3. `key` is a fixed 16-byte AES-128 key.
  unsafe { ppc::expand_key_128(key) }
}

/// Encrypt a single AES-128 block with POWER crypto.
///
/// # Safety
/// Caller must ensure POWER8 crypto is available.
#[cfg(all(target_arch = "powerpc64", feature = "aes-gcm-siv"))]
#[target_feature(enable = "altivec,vsx,power8-vector,power8-crypto")]
#[inline]
pub(super) unsafe fn ppc_encrypt_block_128_inline(keys: &ppc::Ppc128RoundKeys, block: &mut [u8; BLOCK_SIZE]) {
  // SAFETY: POWER8 AES-128 block encryption because:
  // 1. This helper is only callable from a POWER8 crypto target-feature scope.
  // 2. The caller guarantees the current CPU supports POWER8 crypto.
  // 3. `block` is exactly one initialized 16-byte AES block.
  unsafe { ppc::encrypt_block_128_core(keys, block) }
}

/// Encrypt multiple AES-128 blocks inside one POWER crypto target scope.
///
/// # Safety
/// Caller must ensure POWER8 crypto is available.
#[cfg(all(target_arch = "powerpc64", any(feature = "aes-gcm", feature = "aes-gcm-siv")))]
#[target_feature(enable = "altivec,vsx,power8-vector,power8-crypto")]
#[inline]
unsafe fn ppc_encrypt_blocks_128_inline(keys: &ppc::Ppc128RoundKeys, blocks: &mut [[u8; BLOCK_SIZE]]) {
  let mut offset = 0usize;
  while offset.strict_add(4) <= blocks.len() {
    let batch_slice = &mut blocks[offset..offset.strict_add(4)];
    // SAFETY: exact four-block POWER AES batch because:
    // 1. `batch_slice` was sliced to length 4 immediately above.
    // 2. `[[u8; 16]]` elements are contiguous and have no padding.
    // 3. This function's caller guarantees POWER8 crypto availability.
    let batch = unsafe { &mut *batch_slice.as_mut_ptr().cast::<[[u8; BLOCK_SIZE]; 4]>() };
    // SAFETY: POWER8 four-block AES because:
    // 1. This function's caller must guarantee POWER8 crypto availability.
    // 2. `batch` is a valid mutable reference to four initialized AES blocks.
    unsafe { ppc::encrypt_4blocks_128_core(keys, batch) };
    offset = offset.strict_add(4);
  }

  while offset < blocks.len() {
    // SAFETY: POWER8 AES block encryption because:
    // 1. This function's caller must guarantee POWER8 crypto availability.
    // 2. `blocks[offset]` is a valid mutable `[u8; 16]` inside the caller-owned slice.
    unsafe { ppc::encrypt_block_128_core(keys, &mut blocks[offset]) };
    offset = offset.strict_add(1);
  }
}

// ---------------------------------------------------------------------------
// s390x: inline helpers for fused paths (#[inline(always)])
// ---------------------------------------------------------------------------

/// Encrypt a single AES-256 block with s390x KM using a raw 32-byte key.
///
/// # Safety
/// Caller must ensure MSA is available.
#[cfg(all(target_arch = "s390x", feature = "aes-gcm-siv"))]
#[inline(always)]
pub(super) unsafe fn s390x_encrypt_block_raw_inline(raw_key: &[u8; KEY_SIZE], block: &mut [u8; BLOCK_SIZE]) {
  // SAFETY: s390x KM AES-256 block encryption because:
  // 1. The caller guarantees MSA/CPACF availability.
  // 2. `raw_key` is a fixed 32-byte AES-256 key.
  // 3. `block` is exactly one initialized 16-byte AES block.
  unsafe { km::encrypt_block_raw(raw_key, block) }
}

/// Encrypt multiple AES-256 blocks with KM (batch), guaranteed to inline.
///
/// # Safety
/// Caller must ensure MSA is available.
#[cfg(target_arch = "s390x")]
#[inline(always)]
pub(super) unsafe fn s390x_encrypt_blocks_inline(key: &km::KmKey, blocks: &mut [u8], count: usize) {
  // SAFETY: s390x KM AES-256 block batch because:
  // 1. The caller guarantees MSA/CPACF availability.
  // 2. `blocks` is valid for at least `count * 16` writable bytes.
  // 3. `key` is an initialized AES-256 KM key descriptor.
  unsafe { km::encrypt_blocks(key, blocks, count) }
}

/// Encrypt a single AES-128 block with s390x KM using a raw 16-byte key.
///
/// AES-128 sibling of [`s390x_encrypt_block_raw_inline`]. Used by the fused
/// AES-128-GCM-SIV path on s390x.
///
/// # Safety
/// Caller must ensure MSA is available.
#[cfg(all(target_arch = "s390x", feature = "aes-gcm-siv"))]
#[inline(always)]
pub(super) unsafe fn s390x_encrypt_block_raw_128_inline(raw_key: &[u8; KEY_SIZE_128], block: &mut [u8; BLOCK_SIZE]) {
  // SAFETY: s390x KM AES-128 block encryption because:
  // 1. The caller guarantees MSA/CPACF availability.
  // 2. `raw_key` is a fixed 16-byte AES-128 key.
  // 3. `block` is exactly one initialized 16-byte AES block.
  unsafe { km::encrypt_block_raw_128(raw_key, block) }
}

/// Encrypt multiple AES-128 blocks with KM (batch), guaranteed to inline.
///
/// Mirrors [`s390x_encrypt_blocks_inline`] but issues KM function code 18
/// (AES-128) instead of 20 (AES-256).
///
/// # Safety
/// Caller must ensure MSA is available.
#[cfg(all(target_arch = "s390x", any(feature = "aes-gcm", feature = "aes-gcm-siv")))]
#[inline(always)]
pub(super) unsafe fn s390x_encrypt_blocks_128_inline(key: &km::Km128Key, blocks: &mut [u8], count: usize) {
  // SAFETY: s390x KM AES-128 block batch because:
  // 1. The caller guarantees MSA/CPACF availability.
  // 2. `blocks` is valid for at least `count * 16` writable bytes.
  // 3. `key` is an initialized AES-128 KM key descriptor.
  unsafe { km::encrypt_blocks_128(key, blocks, count) }
}

// ---------------------------------------------------------------------------
// Block encryption
// ---------------------------------------------------------------------------

/// Encrypt a single 16-byte block with AES-256.
///
/// Dispatches to AES-NI (x86_64) or AES-CE (aarch64) when available,
/// otherwise uses the portable path.
#[inline]
pub(crate) fn aes256_encrypt_block(ek: &Aes256EncKey, block: &mut [u8; BLOCK_SIZE]) {
  match &ek.inner {
    KeyInner::PortableRoundKeys(rk) => aes256_encrypt_block_portable(rk, block),
    #[cfg(target_arch = "x86_64")]
    KeyInner::X86AesNi(ni_rk) => {
      // SAFETY: AesNi variant is only constructed after runtime detection confirms AES-NI.
      unsafe { ni::encrypt_block(ni_rk, block) }
    }
    #[cfg(target_arch = "aarch64")]
    KeyInner::Aarch64Aes(ce_rk) => {
      // SAFETY: Aarch64Ce variant is only constructed after runtime detection confirms AES-CE.
      unsafe { ce::encrypt_block(ce_rk, block) }
    }
    #[cfg(target_arch = "s390x")]
    KeyInner::S390xMsa(km_key) => {
      // SAFETY: S390xKm variant is only constructed after runtime detection confirms MSA/CPACF.
      unsafe { km::encrypt_block(km_key, block) }
    }
    #[cfg(target_arch = "powerpc64")]
    KeyInner::Power8Crypto(ppc_rk) => {
      // SAFETY: Power variant is only constructed after runtime detection confirms POWER8 crypto.
      unsafe { ppc::encrypt_block(ppc_rk, block) }
    }
    #[cfg(target_arch = "riscv64")]
    KeyInner::Riscv64ScalarCrypto(rv_rk) => {
      // SAFETY: RvScalar variant is only constructed after runtime detection confirms Zkne.
      unsafe { rv_scalar_aes::encrypt_block(rv_rk, block) }
    }
    #[cfg(target_arch = "riscv64")]
    KeyInner::Riscv64VectorCrypto(rv_rk) => {
      // SAFETY: RvAes variant is only constructed after runtime detection confirms Zvkned.
      unsafe { rv_aes::encrypt_block(rv_rk, block) }
    }
    #[cfg(target_arch = "riscv64")]
    KeyInner::Riscv64Vperm(rk) => {
      // SAFETY: RvVperm variant is only constructed after runtime detection confirms V extension.
      unsafe { rv_vperm_aes::encrypt_block(rk, block) }
    }
    #[cfg(target_arch = "riscv64")]
    KeyInner::Riscv64Fixslice(rk) => rv_fixslice_aes::encrypt_block(rk, block),
  }
}

#[cfg(any(target_arch = "riscv64", test))]
#[allow(dead_code)]
#[inline]
pub(super) fn aes_enc_round_4_fixslice(blocks: &mut [[u8; BLOCK_SIZE]; 4], round_keys: &[[u8; BLOCK_SIZE]; 4]) {
  rv_fixslice_aes::cipher_round_4(blocks, round_keys);
}

/// Encrypt a single 16-byte block with AES-128.
///
/// Mirrors [`aes256_encrypt_block`]: dispatches to AES-NI (x86_64), AES-CE
/// (aarch64), CPACF KM (s390x), POWER8 vcipher, RV64 scalar/vector crypto,
/// or the table-free fixslice fallback when the relevant capability is
/// absent.
#[inline]
pub(crate) fn aes128_encrypt_block(ek: &Aes128EncKey, block: &mut [u8; BLOCK_SIZE]) {
  match &ek.inner {
    Key128Inner::PortableRoundKeys(rk) => aes128_encrypt_block_portable(rk, block),
    #[cfg(target_arch = "x86_64")]
    Key128Inner::X86AesNi(ni_rk) => {
      // SAFETY: X86AesNi variant is only constructed after runtime detection confirms AES-NI.
      unsafe { ni::encrypt_block_128(ni_rk, block) }
    }
    #[cfg(target_arch = "aarch64")]
    Key128Inner::Aarch64Aes(ce_rk) => {
      // SAFETY: Aarch64Aes variant is only constructed after runtime detection confirms AES-CE.
      unsafe { ce::encrypt_block_128(ce_rk, block) }
    }
    #[cfg(target_arch = "s390x")]
    Key128Inner::S390xMsa(km_key) => {
      // SAFETY: S390xMsa variant is only constructed after runtime detection confirms MSA/CPACF.
      unsafe { km::encrypt_block_128(km_key, block) }
    }
    #[cfg(target_arch = "powerpc64")]
    Key128Inner::Power8Crypto(ppc_rk) => {
      // SAFETY: Power8Crypto variant is only constructed after runtime detection confirms POWER8 crypto.
      unsafe { ppc::encrypt_block_128(ppc_rk, block) }
    }
    #[cfg(target_arch = "riscv64")]
    Key128Inner::Riscv64ScalarCrypto(rv_rk) => {
      // SAFETY: Riscv64ScalarCrypto variant is only constructed after runtime detection confirms Zkne.
      unsafe { rv_scalar_aes::encrypt_block_128(rv_rk, block) }
    }
    #[cfg(target_arch = "riscv64")]
    Key128Inner::Riscv64VectorCrypto(rv_rk) => {
      // SAFETY: Riscv64VectorCrypto variant is only constructed after runtime detection confirms Zvkned.
      unsafe { rv_aes::encrypt_block_128(rv_rk, block) }
    }
    #[cfg(target_arch = "riscv64")]
    Key128Inner::Riscv64Vperm(rk) => {
      // SAFETY: Riscv64Vperm variant is only constructed after runtime detection confirms V extension.
      unsafe { rv_vperm_aes::encrypt_block_128(rk, block) }
    }
    #[cfg(target_arch = "riscv64")]
    Key128Inner::Riscv64Fixslice(rk) => rv_fixslice_aes::encrypt_block_128(rk, block),
  }
}

/// Encrypt multiple independent 16-byte blocks with AES-128 ECB.
///
/// Mirrors [`aes256_encrypt_blocks_ecb`]: routes to the s390x KM batch
/// instruction or the RV64 4-block kernels when available, otherwise calls
/// the per-block dispatcher. Used by `riscv64` from the AES-128 CTR paths
/// and by AES-128-GCM-SIV key derivation.
#[cfg_attr(
  not(any(
    target_arch = "aarch64",
    target_arch = "powerpc64",
    target_arch = "riscv64",
    target_arch = "s390x",
    feature = "aes-gcm-siv",
    test
  )),
  allow(dead_code)
)]
#[inline]
pub(crate) fn aes128_encrypt_blocks_ecb(ek: &Aes128EncKey, blocks: &mut [[u8; BLOCK_SIZE]]) {
  #[cfg(target_arch = "aarch64")]
  if let Key128Inner::Aarch64Aes(ce_rk) = &ek.inner {
    if !blocks.is_empty() {
      // SAFETY: AES-CE AES-128 batch encryption because:
      // 1. The `Aarch64Aes` key variant is only constructed after runtime detection confirms AES-CE.
      // 2. `blocks` is a mutable slice of initialized 16-byte AES blocks.
      unsafe { aarch64_encrypt_blocks_128_inline(ce_rk, blocks) };
    }
    return;
  }
  #[cfg(target_arch = "powerpc64")]
  if let Key128Inner::Power8Crypto(ppc_rk) = &ek.inner {
    if !blocks.is_empty() {
      // SAFETY: POWER8 AES-128 batch encryption because:
      // 1. The `Power8Crypto` key variant is only constructed after runtime detection confirms POWER8
      //    crypto.
      // 2. `blocks` is a mutable slice of initialized 16-byte AES blocks.
      unsafe { ppc_encrypt_blocks_128_inline(ppc_rk, blocks) };
    }
    return;
  }
  #[cfg(target_arch = "s390x")]
  if let Key128Inner::S390xMsa(km_key) = &ek.inner {
    let count = blocks.len();
    if count > 0 {
      // SAFETY: `[[u8; 16]]` is layout-compatible with a contiguous `[u8]` of `count*16`.
      // S390xMsa variant is only constructed after MSA is confirmed.
      let flat =
        unsafe { core::slice::from_raw_parts_mut(blocks.as_mut_ptr().cast::<u8>(), count.strict_mul(BLOCK_SIZE)) };
      // SAFETY: MSA verified by the S390xMsa variant constructor. `flat` is valid for `count*16` bytes.
      unsafe { s390x_encrypt_blocks_128_inline(km_key, flat, count) };
    }
    return;
  }
  #[cfg(target_arch = "riscv64")]
  if let Key128Inner::Riscv64VectorCrypto(rk) = &ek.inner {
    let mut offset = 0usize;
    while offset.strict_add(4) <= blocks.len() {
      let batch_slice = &mut blocks[offset..offset.strict_add(4)];
      debug_assert_eq!(batch_slice.len(), 4);
      // SAFETY: exact four-block RISC-V Zvkned AES-128 batch because:
      // 1. `batch_slice` is sliced to exactly four contiguous `[u8; 16]` elements.
      // 2. `Riscv64VectorCrypto` is only constructed after runtime detection confirms Zvkned.
      // 3. The mutable borrow is scoped to this loop iteration.
      let batch: &mut [[u8; BLOCK_SIZE]; 4] = unsafe { &mut *batch_slice.as_mut_ptr().cast::<[[u8; BLOCK_SIZE]; 4]>() };
      // SAFETY: `Riscv64VectorCrypto` proves Zvkned availability for this key.
      unsafe { rv_aes::encrypt_4blocks_128(rk, batch) };
      offset = offset.strict_add(4);
    }
    while offset < blocks.len() {
      // SAFETY: same Zvkned availability guarantee as the wide path above.
      unsafe { rv_aes::encrypt_block_128(rk, &mut blocks[offset]) };
      offset = offset.strict_add(1);
    }
    return;
  }
  #[cfg(target_arch = "riscv64")]
  if let Key128Inner::Riscv64ScalarCrypto(rk) = &ek.inner {
    let mut offset = 0usize;
    while offset.strict_add(4) <= blocks.len() {
      let batch_slice = &mut blocks[offset..offset.strict_add(4)];
      debug_assert_eq!(batch_slice.len(), 4);
      // SAFETY: exact four-block RISC-V Zkne AES-128 batch because:
      // 1. `batch_slice` is sliced to exactly four contiguous `[u8; 16]` elements.
      // 2. `Riscv64ScalarCrypto` is only constructed after runtime detection confirms Zkne.
      // 3. The mutable borrow is scoped to this loop iteration.
      let batch: &mut [[u8; BLOCK_SIZE]; 4] = unsafe { &mut *batch_slice.as_mut_ptr().cast::<[[u8; BLOCK_SIZE]; 4]>() };
      // SAFETY: `Riscv64ScalarCrypto` proves Zkne availability for this key.
      unsafe { rv_scalar_aes::encrypt_4blocks_128(rk, batch) };
      offset = offset.strict_add(4);
    }
    while offset < blocks.len() {
      // SAFETY: same Zkne availability guarantee as the wide path above.
      unsafe { rv_scalar_aes::encrypt_block_128(rk, &mut blocks[offset]) };
      offset = offset.strict_add(1);
    }
    return;
  }
  #[cfg(target_arch = "riscv64")]
  if let Key128Inner::Riscv64Vperm(rk) = &ek.inner {
    let mut offset = 0usize;
    while offset.strict_add(4) <= blocks.len() {
      let batch_slice = &mut blocks[offset..offset.strict_add(4)];
      debug_assert_eq!(batch_slice.len(), 4);
      // SAFETY: `batch_slice` is exactly 4 contiguous `[u8; 16]` elements.
      let batch: &mut [[u8; BLOCK_SIZE]; 4] = unsafe { &mut *batch_slice.as_mut_ptr().cast::<[[u8; BLOCK_SIZE]; 4]>() };
      // SAFETY: Riscv64Vperm variant is only constructed after runtime detection confirms V extension.
      unsafe { rv_vperm_aes::encrypt_4blocks_128(rk, batch) };
      offset = offset.strict_add(4);
    }
    while offset < blocks.len() {
      // SAFETY: same V-extension guarantee as the wide path above.
      unsafe { rv_vperm_aes::encrypt_block_128(rk, &mut blocks[offset]) };
      offset = offset.strict_add(1);
    }
    return;
  }
  #[cfg(target_arch = "riscv64")]
  if let Key128Inner::Riscv64Fixslice(rk) = &ek.inner {
    let mut offset = 0usize;
    while offset.strict_add(4) <= blocks.len() {
      let batch_slice = &mut blocks[offset..offset.strict_add(4)];
      debug_assert_eq!(batch_slice.len(), 4);
      // SAFETY: `batch_slice` is exactly 4 contiguous `[u8; 16]` elements.
      let batch: &mut [[u8; BLOCK_SIZE]; 4] = unsafe { &mut *batch_slice.as_mut_ptr().cast::<[[u8; BLOCK_SIZE]; 4]>() };
      rv_fixslice_aes::encrypt_4blocks_128(rk, batch);
      offset = offset.strict_add(4);
    }
    if offset < blocks.len() {
      let remaining = blocks.len().strict_sub(offset);
      let mut tail = [[0u8; BLOCK_SIZE]; 4];
      let mut i = 0usize;
      while i < remaining {
        tail[i] = blocks[offset.strict_add(i)];
        i = i.strict_add(1);
      }
      rv_fixslice_aes::encrypt_4blocks_128(rk, &mut tail);
      i = 0;
      while i < remaining {
        blocks[offset.strict_add(i)] = tail[i];
        i = i.strict_add(1);
      }
    }
    return;
  }
  for block in blocks {
    aes128_encrypt_block(ek, block);
  }
}

/// Portable AES-128 block encryption (10 rounds).
#[inline]
fn aes128_encrypt_block_portable(rk: &[u32; EXPANDED_KEY_WORDS_128], block: &mut [u8; BLOCK_SIZE]) {
  // Load state as four big-endian u32 columns.
  let mut s0 = u32::from_be_bytes([block[0], block[1], block[2], block[3]]);
  let mut s1 = u32::from_be_bytes([block[4], block[5], block[6], block[7]]);
  let mut s2 = u32::from_be_bytes([block[8], block[9], block[10], block[11]]);
  let mut s3 = u32::from_be_bytes([block[12], block[13], block[14], block[15]]);

  // Initial AddRoundKey.
  s0 ^= rk[0];
  s1 ^= rk[1];
  s2 ^= rk[2];
  s3 ^= rk[3];

  // Rounds 1..9: SubBytes, ShiftRows, MixColumns, AddRoundKey.
  let mut round = 1;
  while round < ROUNDS_128 {
    let (t0, t1, t2, t3) = aes_round(s0, s1, s2, s3);
    let rk_off = round.strict_mul(4);
    s0 = t0 ^ rk[rk_off];
    s1 = t1 ^ rk[rk_off.strict_add(1)];
    s2 = t2 ^ rk[rk_off.strict_add(2)];
    s3 = t3 ^ rk[rk_off.strict_add(3)];
    round = round.strict_add(1);
  }

  // Final round (no MixColumns).
  let (t0, t1, t2, t3) = aes_final_round(s0, s1, s2, s3);
  let rk_off = ROUNDS_128.strict_mul(4);
  s0 = t0 ^ rk[rk_off];
  s1 = t1 ^ rk[rk_off.strict_add(1)];
  s2 = t2 ^ rk[rk_off.strict_add(2)];
  s3 = t3 ^ rk[rk_off.strict_add(3)];

  // Store back.
  block[0..4].copy_from_slice(&s0.to_be_bytes());
  block[4..8].copy_from_slice(&s1.to_be_bytes());
  block[8..12].copy_from_slice(&s2.to_be_bytes());
  block[12..16].copy_from_slice(&s3.to_be_bytes());
}

/// Encrypt multiple independent 16-byte blocks with AES-256 ECB.
///
/// On s390x this issues a single KM instruction for all `blocks`,
/// avoiding per-block parameter-block setup overhead. On other platforms
/// falls back to per-block dispatch.
#[cfg_attr(
  not(any(
    target_arch = "aarch64",
    target_arch = "powerpc64",
    target_arch = "riscv64",
    target_arch = "s390x",
    feature = "aes-gcm-siv",
    test
  )),
  allow(dead_code)
)]
#[inline]
pub(crate) fn aes256_encrypt_blocks_ecb(ek: &Aes256EncKey, blocks: &mut [[u8; BLOCK_SIZE]]) {
  #[cfg(target_arch = "aarch64")]
  if let KeyInner::Aarch64Aes(ce_rk) = &ek.inner {
    if !blocks.is_empty() {
      // SAFETY: AES-CE AES-256 batch encryption because:
      // 1. The `Aarch64Aes` key variant is only constructed after runtime detection confirms AES-CE.
      // 2. `blocks` is a mutable slice of initialized 16-byte AES blocks.
      unsafe { aarch64_encrypt_blocks_inline(ce_rk, blocks) };
    }
    return;
  }
  #[cfg(target_arch = "powerpc64")]
  if let KeyInner::Power8Crypto(ppc_rk) = &ek.inner {
    if !blocks.is_empty() {
      // SAFETY: POWER8 AES-256 batch encryption because:
      // 1. The `Power8Crypto` key variant is only constructed after runtime detection confirms POWER8
      //    crypto.
      // 2. `blocks` is a mutable slice of initialized 16-byte AES blocks.
      unsafe { ppc_encrypt_blocks_inline(ppc_rk, blocks) };
    }
    return;
  }
  #[cfg(target_arch = "s390x")]
  if let KeyInner::S390xMsa(km_key) = &ek.inner {
    let count = blocks.len();
    if count > 0 {
      // SAFETY: `[[u8; 16]]` is layout-compatible with a contiguous `[u8]` of `count*16`.
      // S390xKm variant is only constructed after MSA is confirmed.
      let flat =
        unsafe { core::slice::from_raw_parts_mut(blocks.as_mut_ptr().cast::<u8>(), count.strict_mul(BLOCK_SIZE)) };
      // SAFETY: MSA verified by the S390xKm variant constructor. `flat` is valid for `count*16` bytes.
      unsafe { s390x_encrypt_blocks_inline(km_key, flat, count) };
    }
    return;
  }
  #[cfg(target_arch = "riscv64")]
  if let KeyInner::Riscv64VectorCrypto(rk) = &ek.inner {
    let mut offset = 0usize;
    while offset.strict_add(4) <= blocks.len() {
      let batch_slice = &mut blocks[offset..offset.strict_add(4)];
      debug_assert_eq!(batch_slice.len(), 4);
      // SAFETY: exact four-block RISC-V Zvkned AES-256 batch because:
      // 1. `batch_slice` is sliced to exactly four contiguous `[u8; 16]` elements.
      // 2. `Riscv64VectorCrypto` is only constructed after runtime detection confirms Zvkned.
      // 3. The mutable borrow is scoped to this loop iteration.
      let batch: &mut [[u8; BLOCK_SIZE]; 4] = unsafe { &mut *batch_slice.as_mut_ptr().cast::<[[u8; BLOCK_SIZE]; 4]>() };
      // SAFETY: `Riscv64VectorCrypto` proves Zvkned availability for this key.
      unsafe { rv_aes::encrypt_4blocks(rk, batch) };
      offset = offset.strict_add(4);
    }
    while offset < blocks.len() {
      // SAFETY: Same Zvkned availability guarantee as the wide path above.
      unsafe { rv_aes::encrypt_block(rk, &mut blocks[offset]) };
      offset = offset.strict_add(1);
    }
    return;
  }
  #[cfg(target_arch = "riscv64")]
  if let KeyInner::Riscv64ScalarCrypto(rk) = &ek.inner {
    let mut offset = 0usize;
    while offset.strict_add(4) <= blocks.len() {
      let batch_slice = &mut blocks[offset..offset.strict_add(4)];
      debug_assert_eq!(batch_slice.len(), 4);
      // SAFETY: exact four-block RISC-V Zkne AES-256 batch because:
      // 1. `batch_slice` is sliced to exactly four contiguous `[u8; 16]` elements.
      // 2. `Riscv64ScalarCrypto` is only constructed after runtime detection confirms Zkne.
      // 3. The mutable borrow is scoped to this loop iteration.
      let batch: &mut [[u8; BLOCK_SIZE]; 4] = unsafe { &mut *batch_slice.as_mut_ptr().cast::<[[u8; BLOCK_SIZE]; 4]>() };
      // SAFETY: `Riscv64ScalarCrypto` proves Zkne availability for this key.
      unsafe { rv_scalar_aes::encrypt_4blocks(rk, batch) };
      offset = offset.strict_add(4);
    }
    while offset < blocks.len() {
      // SAFETY: Same Zkne availability guarantee as the wide path above.
      unsafe { rv_scalar_aes::encrypt_block(rk, &mut blocks[offset]) };
      offset = offset.strict_add(1);
    }
    return;
  }
  #[cfg(target_arch = "riscv64")]
  if let KeyInner::Riscv64Vperm(rk) = &ek.inner {
    let mut offset = 0usize;
    while offset.strict_add(4) <= blocks.len() {
      let batch_slice = &mut blocks[offset..offset.strict_add(4)];
      debug_assert_eq!(batch_slice.len(), 4);
      // SAFETY: `batch_slice` is exactly 4 contiguous `[u8; 16]` elements, so
      // reborrowing it as `&mut [[u8; 16]; 4]` preserves layout and bounds.
      let batch: &mut [[u8; BLOCK_SIZE]; 4] = unsafe { &mut *batch_slice.as_mut_ptr().cast::<[[u8; BLOCK_SIZE]; 4]>() };
      // SAFETY: RvVperm variant is only constructed after runtime detection confirms V extension.
      unsafe { rv_vperm_aes::encrypt_4blocks(rk, batch) };
      offset = offset.strict_add(4);
    }
    while offset < blocks.len() {
      // SAFETY: Same runtime V-extension guarantee as above; tail stays on the
      // existing single-block kernel to avoid special-casing 1-3 blocks.
      unsafe { rv_vperm_aes::encrypt_block(rk, &mut blocks[offset]) };
      offset = offset.strict_add(1);
    }
    return;
  }
  #[cfg(target_arch = "riscv64")]
  if let KeyInner::Riscv64Fixslice(rk) = &ek.inner {
    let mut offset = 0usize;
    while offset.strict_add(4) <= blocks.len() {
      let batch_slice = &mut blocks[offset..offset.strict_add(4)];
      debug_assert_eq!(batch_slice.len(), 4);
      // SAFETY: `batch_slice` is exactly four contiguous `[u8; 16]` elements.
      let batch: &mut [[u8; BLOCK_SIZE]; 4] = unsafe { &mut *batch_slice.as_mut_ptr().cast::<[[u8; BLOCK_SIZE]; 4]>() };
      rv_fixslice_aes::encrypt_4blocks(rk, batch);
      offset = offset.strict_add(4);
    }
    if offset < blocks.len() {
      let remaining = blocks.len().strict_sub(offset);
      let mut tail = [[0u8; BLOCK_SIZE]; 4];
      let mut i = 0usize;
      while i < remaining {
        tail[i] = blocks[offset.strict_add(i)];
        i = i.strict_add(1);
      }
      rv_fixslice_aes::encrypt_4blocks(rk, &mut tail);
      i = 0;
      while i < remaining {
        blocks[offset.strict_add(i)] = tail[i];
        i = i.strict_add(1);
      }
    }
    return;
  }
  for block in blocks {
    aes256_encrypt_block(ek, block);
  }
}

/// Portable AES-256 block encryption.
#[inline]
fn aes256_encrypt_block_portable(rk: &[u32; EXPANDED_KEY_WORDS], block: &mut [u8; BLOCK_SIZE]) {
  // Load state as four big-endian u32 columns.
  let mut s0 = u32::from_be_bytes([block[0], block[1], block[2], block[3]]);
  let mut s1 = u32::from_be_bytes([block[4], block[5], block[6], block[7]]);
  let mut s2 = u32::from_be_bytes([block[8], block[9], block[10], block[11]]);
  let mut s3 = u32::from_be_bytes([block[12], block[13], block[14], block[15]]);

  // Initial AddRoundKey.
  s0 ^= rk[0];
  s1 ^= rk[1];
  s2 ^= rk[2];
  s3 ^= rk[3];

  // Rounds 1..13: SubBytes, ShiftRows, MixColumns, AddRoundKey.
  let mut round = 1;
  while round < ROUNDS {
    let (t0, t1, t2, t3) = aes_round(s0, s1, s2, s3);
    let rk_off = round.strict_mul(4);
    s0 = t0 ^ rk[rk_off];
    s1 = t1 ^ rk[rk_off.strict_add(1)];
    s2 = t2 ^ rk[rk_off.strict_add(2)];
    s3 = t3 ^ rk[rk_off.strict_add(3)];
    round = round.strict_add(1);
  }

  // Final round (no MixColumns).
  let (t0, t1, t2, t3) = aes_final_round(s0, s1, s2, s3);
  let rk_off = ROUNDS.strict_mul(4);
  s0 = t0 ^ rk[rk_off];
  s1 = t1 ^ rk[rk_off.strict_add(1)];
  s2 = t2 ^ rk[rk_off.strict_add(2)];
  s3 = t3 ^ rk[rk_off.strict_add(3)];

  // Store back.
  block[0..4].copy_from_slice(&s0.to_be_bytes());
  block[4..8].copy_from_slice(&s1.to_be_bytes());
  block[8..12].copy_from_slice(&s2.to_be_bytes());
  block[12..16].copy_from_slice(&s3.to_be_bytes());
}

/// Extract byte `row` from a big-endian column word.
#[inline(always)]
const fn col_byte(col: u32, row: usize) -> u8 {
  (col >> (24u32.strict_sub((row as u32).strict_mul(8)))) as u8
}

/// xtime: multiply by x in GF(2^8), i.e. x << 1 with conditional reduction.
#[inline(always)]
const fn xtime(x: u8) -> u8 {
  let hi = (x >> 7) & 1;
  (x << 1) ^ (hi.wrapping_mul(0x1b))
}

/// One AES round: SubBytes → ShiftRows → MixColumns.
///
/// Input/output: four column words in big-endian byte order.
/// AddRoundKey is done by the caller.
#[inline(always)]
const fn aes_round(s0: u32, s1: u32, s2: u32, s3: u32) -> (u32, u32, u32, u32) {
  // After SubBytes + ShiftRows, column j contains:
  //   row 0 from column j, row 1 from (j+1)%4, row 2 from (j+2)%4, row 3 from (j+3)%4
  let sr0 = [
    sbox(col_byte(s0, 0)),
    sbox(col_byte(s1, 1)),
    sbox(col_byte(s2, 2)),
    sbox(col_byte(s3, 3)),
  ];
  let sr1 = [
    sbox(col_byte(s1, 0)),
    sbox(col_byte(s2, 1)),
    sbox(col_byte(s3, 2)),
    sbox(col_byte(s0, 3)),
  ];
  let sr2 = [
    sbox(col_byte(s2, 0)),
    sbox(col_byte(s3, 1)),
    sbox(col_byte(s0, 2)),
    sbox(col_byte(s1, 3)),
  ];
  let sr3 = [
    sbox(col_byte(s3, 0)),
    sbox(col_byte(s0, 1)),
    sbox(col_byte(s1, 2)),
    sbox(col_byte(s2, 3)),
  ];

  (mix_column(sr0), mix_column(sr1), mix_column(sr2), mix_column(sr3))
}

/// Final AES round: SubBytes → ShiftRows (no MixColumns).
#[inline(always)]
const fn aes_final_round(s0: u32, s1: u32, s2: u32, s3: u32) -> (u32, u32, u32, u32) {
  let t0 = (sbox(col_byte(s0, 0)) as u32) << 24
    | (sbox(col_byte(s1, 1)) as u32) << 16
    | (sbox(col_byte(s2, 2)) as u32) << 8
    | sbox(col_byte(s3, 3)) as u32;
  let t1 = (sbox(col_byte(s1, 0)) as u32) << 24
    | (sbox(col_byte(s2, 1)) as u32) << 16
    | (sbox(col_byte(s3, 2)) as u32) << 8
    | sbox(col_byte(s0, 3)) as u32;
  let t2 = (sbox(col_byte(s2, 0)) as u32) << 24
    | (sbox(col_byte(s3, 1)) as u32) << 16
    | (sbox(col_byte(s0, 2)) as u32) << 8
    | sbox(col_byte(s1, 3)) as u32;
  let t3 = (sbox(col_byte(s3, 0)) as u32) << 24
    | (sbox(col_byte(s0, 1)) as u32) << 16
    | (sbox(col_byte(s1, 2)) as u32) << 8
    | sbox(col_byte(s2, 3)) as u32;

  (t0, t1, t2, t3)
}

/// MixColumns on a single column [b0, b1, b2, b3].
#[inline(always)]
const fn mix_column(col: [u8; 4]) -> u32 {
  let [b0, b1, b2, b3] = col;

  // 2*a XOR 3*b XOR c XOR d
  let r0 = xtime(b0) ^ xtime(b1) ^ b1 ^ b2 ^ b3;
  let r1 = b0 ^ xtime(b1) ^ xtime(b2) ^ b2 ^ b3;
  let r2 = b0 ^ b1 ^ xtime(b2) ^ xtime(b3) ^ b3;
  let r3 = xtime(b0) ^ b0 ^ b1 ^ b2 ^ xtime(b3);

  (r0 as u32) << 24 | (r1 as u32) << 16 | (r2 as u32) << 8 | r3 as u32
}

// ---------------------------------------------------------------------------
// AES-CTR for GCM-SIV
// ---------------------------------------------------------------------------

/// AES-256 CTR encryption/decryption for GCM-SIV.
///
/// The initial counter block is the tag with bit 31 set (MSB of byte 15).
/// The counter increments the first 32 bits (little-endian) of the block.
#[cfg(feature = "aes-gcm-siv")]
#[inline]
pub(crate) fn aes256_ctr32_encrypt(ek: &Aes256EncKey, initial_counter: &[u8; BLOCK_SIZE], data: &mut [u8]) {
  let mut counter_block = *initial_counter;
  // Maintain counter as u32 to avoid per-block LE decode/encode.
  let mut ctr = u32::from_le_bytes([counter_block[0], counter_block[1], counter_block[2], counter_block[3]]);
  let mut offset = 0usize;

  #[cfg(target_arch = "riscv64")]
  if matches!(
    &ek.inner,
    KeyInner::Riscv64VectorCrypto(_)
      | KeyInner::Riscv64ScalarCrypto(_)
      | KeyInner::Riscv64Vperm(_)
      | KeyInner::Riscv64Fixslice(_)
  ) {
    let iv_suffix: [u8; 12] = {
      let mut buf = [0u8; 12];
      buf.copy_from_slice(&initial_counter[4..16]);
      buf
    };

    while offset.strict_add(64) <= data.len() {
      let mut keystream = [[0u8; BLOCK_SIZE]; 4];
      let mut i = 0u32;
      while i < 4 {
        keystream[i as usize][0..4].copy_from_slice(&ctr.wrapping_add(i).to_le_bytes());
        keystream[i as usize][4..16].copy_from_slice(&iv_suffix);
        i = i.strict_add(1);
      }

      aes256_encrypt_blocks_ecb(ek, &mut keystream);

      let mut lane = 0usize;
      while lane < 4 {
        let block_offset = offset.strict_add(lane.strict_mul(BLOCK_SIZE));
        let ks = u128::from_ne_bytes(keystream[lane]);
        let mut d = [0u8; BLOCK_SIZE];
        d.copy_from_slice(&data[block_offset..block_offset.strict_add(BLOCK_SIZE)]);
        let xored = u128::from_ne_bytes(d) ^ ks;
        data[block_offset..block_offset.strict_add(BLOCK_SIZE)].copy_from_slice(&xored.to_ne_bytes());
        lane = lane.strict_add(1);
      }

      ctr = ctr.wrapping_add(4);
      offset = offset.strict_add(64);
    }

    counter_block[4..16].copy_from_slice(&iv_suffix);
  }

  while offset < data.len() {
    // Encode current counter into the block.
    counter_block[0..4].copy_from_slice(&ctr.to_le_bytes());

    let mut keystream = counter_block;
    aes256_encrypt_block(ek, &mut keystream);

    let remaining = data.len().strict_sub(offset);
    if remaining >= BLOCK_SIZE {
      // Full block: XOR as u128 for vectorization.
      let ks = u128::from_ne_bytes(keystream);
      let mut d = [0u8; BLOCK_SIZE];
      d.copy_from_slice(&data[offset..offset.strict_add(BLOCK_SIZE)]);
      let xored = u128::from_ne_bytes(d) ^ ks;
      data[offset..offset.strict_add(BLOCK_SIZE)].copy_from_slice(&xored.to_ne_bytes());
      offset = offset.strict_add(BLOCK_SIZE);
    } else {
      // Partial tail: byte-wise XOR.
      let mut i = 0usize;
      while i < remaining {
        data[offset.strict_add(i)] ^= keystream[i];
        i = i.strict_add(1);
      }
      offset = offset.strict_add(remaining);
    }

    ctr = ctr.wrapping_add(1);
  }
}

/// AES-128 CTR encryption/decryption for GCM-SIV.
///
/// Mirrors [`aes256_ctr32_encrypt`]: the counter occupies the first 4 bytes
/// of the block (little-endian) and increments as a 32-bit integer per
/// RFC 8452 §4.
#[cfg(feature = "aes-gcm-siv")]
#[inline]
pub(crate) fn aes128_ctr32_encrypt(ek: &Aes128EncKey, initial_counter: &[u8; BLOCK_SIZE], data: &mut [u8]) {
  let mut counter_block = *initial_counter;
  let mut ctr = u32::from_le_bytes([counter_block[0], counter_block[1], counter_block[2], counter_block[3]]);
  let mut offset = 0usize;

  #[cfg(target_arch = "riscv64")]
  if matches!(
    &ek.inner,
    Key128Inner::Riscv64VectorCrypto(_)
      | Key128Inner::Riscv64ScalarCrypto(_)
      | Key128Inner::Riscv64Vperm(_)
      | Key128Inner::Riscv64Fixslice(_)
  ) {
    let iv_suffix: [u8; 12] = {
      let mut buf = [0u8; 12];
      buf.copy_from_slice(&initial_counter[4..16]);
      buf
    };

    while offset.strict_add(64) <= data.len() {
      let mut keystream = [[0u8; BLOCK_SIZE]; 4];
      let mut i = 0u32;
      while i < 4 {
        keystream[i as usize][0..4].copy_from_slice(&ctr.wrapping_add(i).to_le_bytes());
        keystream[i as usize][4..16].copy_from_slice(&iv_suffix);
        i = i.strict_add(1);
      }

      aes128_encrypt_blocks_ecb(ek, &mut keystream);

      let mut lane = 0usize;
      while lane < 4 {
        let block_offset = offset.strict_add(lane.strict_mul(BLOCK_SIZE));
        let ks = u128::from_ne_bytes(keystream[lane]);
        let mut d = [0u8; BLOCK_SIZE];
        d.copy_from_slice(&data[block_offset..block_offset.strict_add(BLOCK_SIZE)]);
        let xored = u128::from_ne_bytes(d) ^ ks;
        data[block_offset..block_offset.strict_add(BLOCK_SIZE)].copy_from_slice(&xored.to_ne_bytes());
        lane = lane.strict_add(1);
      }

      ctr = ctr.wrapping_add(4);
      offset = offset.strict_add(64);
    }

    counter_block[4..16].copy_from_slice(&iv_suffix);
  }

  while offset < data.len() {
    counter_block[0..4].copy_from_slice(&ctr.to_le_bytes());

    let mut keystream = counter_block;
    aes128_encrypt_block(ek, &mut keystream);

    let remaining = data.len().strict_sub(offset);
    if remaining >= BLOCK_SIZE {
      let ks = u128::from_ne_bytes(keystream);
      let mut d = [0u8; BLOCK_SIZE];
      d.copy_from_slice(&data[offset..offset.strict_add(BLOCK_SIZE)]);
      let xored = u128::from_ne_bytes(d) ^ ks;
      data[offset..offset.strict_add(BLOCK_SIZE)].copy_from_slice(&xored.to_ne_bytes());
      offset = offset.strict_add(BLOCK_SIZE);
    } else {
      let mut i = 0usize;
      while i < remaining {
        data[offset.strict_add(i)] ^= keystream[i];
        i = i.strict_add(1);
      }
      offset = offset.strict_add(remaining);
    }

    ctr = ctr.wrapping_add(1);
  }
}

// ---------------------------------------------------------------------------
// AES-CTR for GCM (big-endian 32-bit counter in bytes 12..15)
// ---------------------------------------------------------------------------

#[cfg(all(
  feature = "aes-gcm",
  any(
    target_arch = "aarch64",
    target_arch = "powerpc64",
    target_arch = "riscv64",
    target_arch = "s390x"
  )
))]
#[inline]
fn aes256_ctr32_be_uses_block_batch(ek: &Aes256EncKey) -> bool {
  match &ek.inner {
    #[cfg(target_arch = "aarch64")]
    KeyInner::Aarch64Aes(_) => true,
    #[cfg(target_arch = "powerpc64")]
    KeyInner::Power8Crypto(_) => true,
    #[cfg(target_arch = "s390x")]
    KeyInner::S390xMsa(_) => true,
    #[cfg(target_arch = "riscv64")]
    KeyInner::Riscv64VectorCrypto(_)
    | KeyInner::Riscv64ScalarCrypto(_)
    | KeyInner::Riscv64Vperm(_)
    | KeyInner::Riscv64Fixslice(_) => true,
    _ => false,
  }
}

#[cfg(all(
  feature = "aes-gcm",
  any(
    target_arch = "aarch64",
    target_arch = "powerpc64",
    target_arch = "riscv64",
    target_arch = "s390x"
  )
))]
#[inline]
fn aes128_ctr32_be_uses_block_batch(ek: &Aes128EncKey) -> bool {
  match &ek.inner {
    #[cfg(target_arch = "aarch64")]
    Key128Inner::Aarch64Aes(_) => true,
    #[cfg(target_arch = "powerpc64")]
    Key128Inner::Power8Crypto(_) => true,
    #[cfg(target_arch = "s390x")]
    Key128Inner::S390xMsa(_) => true,
    #[cfg(target_arch = "riscv64")]
    Key128Inner::Riscv64VectorCrypto(_)
    | Key128Inner::Riscv64ScalarCrypto(_)
    | Key128Inner::Riscv64Vperm(_)
    | Key128Inner::Riscv64Fixslice(_) => true,
    _ => false,
  }
}

/// AES-256 CTR encryption/decryption for GCM.
///
/// The counter occupies the last 4 bytes (12..15) of the 16-byte counter
/// block and increments as a big-endian 32-bit integer. This matches the
/// `inc_32` function from NIST SP 800-38D § 6.2.
#[inline]
#[cfg(feature = "aes-gcm")]
pub(crate) fn aes256_ctr32_encrypt_be(ek: &Aes256EncKey, initial_counter: &[u8; BLOCK_SIZE], data: &mut [u8]) {
  let mut counter_block = *initial_counter;
  // Maintain the 32-bit counter separately to avoid per-block BE decode/encode.
  let mut ctr = u32::from_be_bytes([
    counter_block[12],
    counter_block[13],
    counter_block[14],
    counter_block[15],
  ]);
  let mut offset = 0usize;

  #[cfg(target_arch = "aarch64")]
  if let KeyInner::Aarch64Aes(ce_rk) = &ek.inner {
    let iv_prefix: [u8; 12] = {
      let mut buf = [0u8; 12];
      buf.copy_from_slice(&initial_counter[..12]);
      buf
    };

    while offset.strict_add(128) <= data.len() {
      let end = offset.strict_add(128);
      // SAFETY: direct aarch64 CTR batch because:
      // 1. `Aarch64Aes` keys are only constructed after runtime detection confirms AES-CE.
      // 2. The loop condition guarantees the helper receives at least 128 writable bytes.
      // 3. The counter prefix is copied from the caller-provided GCM counter block.
      let _ = unsafe { ce::encrypt_ctr32_be_xor_8blocks_core(ce_rk, &iv_prefix, ctr, &mut data[offset..end]) };
      ctr = ctr.wrapping_add(8);
      offset = end;
    }

    while offset.strict_add(64) <= data.len() {
      let end = offset.strict_add(64);
      // SAFETY: direct aarch64 CTR batch because:
      // 1. `Aarch64Aes` keys are only constructed after runtime detection confirms AES-CE.
      // 2. The loop condition guarantees the helper receives at least 64 writable bytes.
      // 3. The counter prefix is copied from the caller-provided GCM counter block.
      let _ = unsafe { ce::encrypt_ctr32_be_xor_4blocks_core(ce_rk, &iv_prefix, ctr, &mut data[offset..end]) };
      ctr = ctr.wrapping_add(4);
      offset = end;
    }
  }

  #[cfg(any(
    target_arch = "aarch64",
    target_arch = "powerpc64",
    target_arch = "riscv64",
    target_arch = "s390x"
  ))]
  if aes256_ctr32_be_uses_block_batch(ek) {
    let iv_prefix: [u8; 12] = {
      let mut buf = [0u8; 12];
      buf.copy_from_slice(&initial_counter[..12]);
      buf
    };

    while offset.strict_add(64) <= data.len() {
      let mut keystream = [[0u8; BLOCK_SIZE]; 4];
      let mut i = 0u32;
      while i < 4 {
        keystream[i as usize][..12].copy_from_slice(&iv_prefix);
        keystream[i as usize][12..16].copy_from_slice(&ctr.wrapping_add(i).to_be_bytes());
        i = i.strict_add(1);
      }

      aes256_encrypt_blocks_ecb(ek, &mut keystream);

      let mut lane = 0usize;
      while lane < 4 {
        let block_offset = offset.strict_add(lane.strict_mul(BLOCK_SIZE));
        let ks = u128::from_ne_bytes(keystream[lane]);
        let mut d = [0u8; BLOCK_SIZE];
        d.copy_from_slice(&data[block_offset..block_offset.strict_add(BLOCK_SIZE)]);
        let xored = u128::from_ne_bytes(d) ^ ks;
        data[block_offset..block_offset.strict_add(BLOCK_SIZE)].copy_from_slice(&xored.to_ne_bytes());
        lane = lane.strict_add(1);
      }

      ctr = ctr.wrapping_add(4);
      offset = offset.strict_add(64);
    }

    counter_block[..12].copy_from_slice(&iv_prefix);
  }

  while offset < data.len() {
    // Encode current counter into bytes 12..15 (big-endian).
    counter_block[12..16].copy_from_slice(&ctr.to_be_bytes());

    let mut keystream = counter_block;
    aes256_encrypt_block(ek, &mut keystream);

    let remaining = data.len().strict_sub(offset);
    if remaining >= BLOCK_SIZE {
      // Full block: XOR as u128 for vectorization.
      let ks = u128::from_ne_bytes(keystream);
      let mut d = [0u8; BLOCK_SIZE];
      d.copy_from_slice(&data[offset..offset.strict_add(BLOCK_SIZE)]);
      let xored = u128::from_ne_bytes(d) ^ ks;
      data[offset..offset.strict_add(BLOCK_SIZE)].copy_from_slice(&xored.to_ne_bytes());
      offset = offset.strict_add(BLOCK_SIZE);
    } else {
      // Partial tail: byte-wise XOR.
      let mut i = 0usize;
      while i < remaining {
        data[offset.strict_add(i)] ^= keystream[i];
        i = i.strict_add(1);
      }
      offset = offset.strict_add(remaining);
    }

    ctr = ctr.wrapping_add(1);
  }
}

// ---------------------------------------------------------------------------
// Wide AES-CTR for GCM (big-endian 32-bit counter, VAES-512)
// ---------------------------------------------------------------------------

/// Build four big-endian GCM counter blocks directly in a VAES register.
///
/// # Safety
/// Caller must ensure AVX-512F is available.
#[cfg(all(target_arch = "x86_64", feature = "aes-gcm"))]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn x86_gcm_ctr_blocks_be_4(iv_words: [u32; 3], ctr: u32) -> core::arch::x86_64::__m512i {
  use core::arch::x86_64::*;

  let c0 = ctr.swap_bytes();
  let c1 = ctr.wrapping_add(1).swap_bytes();
  let c2 = ctr.wrapping_add(2).swap_bytes();
  let c3 = ctr.wrapping_add(3).swap_bytes();
  let iv0 = iv_words[0] as i32;
  let iv1 = iv_words[1] as i32;
  let iv2 = iv_words[2] as i32;
  let b0 = _mm_set_epi32(c0 as i32, iv2, iv1, iv0);
  let b1 = _mm_set_epi32(c1 as i32, iv2, iv1, iv0);
  let b2 = _mm_set_epi32(c2 as i32, iv2, iv1, iv0);
  let b3 = _mm_set_epi32(c3 as i32, iv2, iv1, iv0);

  let z = _mm512_zextsi128_si512(b0);
  let z = _mm512_inserti32x4(z, b1, 1);
  let z = _mm512_inserti32x4(z, b2, 2);
  _mm512_inserti32x4(z, b3, 3)
}

#[cfg(all(target_arch = "x86_64", feature = "aes-gcm"))]
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
unsafe fn x86_gcm_ctr_blocks_be_16(
  iv_words: [u32; 3],
  ctr: u32,
) -> (
  core::arch::x86_64::__m512i,
  core::arch::x86_64::__m512i,
  core::arch::x86_64::__m512i,
  core::arch::x86_64::__m512i,
) {
  use core::arch::x86_64::*;

  let iv = _mm_set_epi32(0, iv_words[2] as i32, iv_words[1] as i32, iv_words[0] as i32);
  let template = _mm512_broadcast_i32x4(iv);
  let ctrs = _mm512_set1_epi32(ctr as i32);
  let bswap = _mm512_broadcast_i32x4(_mm_set_epi8(12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3));

  let offsets0 = _mm512_set_epi32(3, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0);
  let offsets1 = _mm512_set_epi32(7, 0, 0, 0, 6, 0, 0, 0, 5, 0, 0, 0, 4, 0, 0, 0);
  let offsets2 = _mm512_set_epi32(11, 0, 0, 0, 10, 0, 0, 0, 9, 0, 0, 0, 8, 0, 0, 0);
  let offsets3 = _mm512_set_epi32(15, 0, 0, 0, 14, 0, 0, 0, 13, 0, 0, 0, 12, 0, 0, 0);

  macro_rules! make {
    ($offsets:expr) => {{
      let be = _mm512_shuffle_epi8(_mm512_add_epi32(ctrs, $offsets), bswap);
      _mm512_mask_mov_epi32(template, 0x8888, be)
    }};
  }

  (make!(offsets0), make!(offsets1), make!(offsets2), make!(offsets3))
}

#[cfg(all(target_arch = "x86_64", feature = "aes-gcm", test))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn x86_gcm_ctr_blocks_be_2(iv_words: [u32; 3], ctr: u32) -> core::arch::x86_64::__m256i {
  use core::arch::x86_64::*;

  let p0 = iv_words[0] as i32;
  let p1 = iv_words[1] as i32;
  let p2 = iv_words[2] as i32;
  let b0 = _mm_set_epi32(ctr.to_be() as i32, p2, p1, p0);
  let b1 = _mm_set_epi32(ctr.wrapping_add(1).to_be() as i32, p2, p1, p0);

  let z = _mm256_castsi128_si256(b0);
  _mm256_inserti128_si256(z, b1, 1)
}

#[cfg(all(target_arch = "x86_64", feature = "aes-gcm"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn x86_gcm_ctr_blocks_be_8_y256(
  iv_words: [u32; 3],
  ctr: u32,
) -> (
  core::arch::x86_64::__m256i,
  core::arch::x86_64::__m256i,
  core::arch::x86_64::__m256i,
  core::arch::x86_64::__m256i,
) {
  use core::arch::x86_64::*;

  let iv = _mm_set_epi32(0, iv_words[2] as i32, iv_words[1] as i32, iv_words[0] as i32);
  let template = _mm256_broadcastsi128_si256(iv);
  let ctrs = _mm256_set1_epi32(ctr as i32);
  let bswap = _mm256_broadcastsi128_si256(_mm_set_epi8(12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3));

  let offsets0 = _mm256_set_epi32(1, 0, 0, 0, 0, 0, 0, 0);
  let offsets1 = _mm256_set_epi32(3, 0, 0, 0, 2, 0, 0, 0);
  let offsets2 = _mm256_set_epi32(5, 0, 0, 0, 4, 0, 0, 0);
  let offsets3 = _mm256_set_epi32(7, 0, 0, 0, 6, 0, 0, 0);

  macro_rules! make {
    ($offsets:expr) => {{
      let be = _mm256_shuffle_epi8(_mm256_add_epi32(ctrs, $offsets), bswap);
      _mm256_blend_epi32::<0x88>(template, be)
    }};
  }

  (make!(offsets0), make!(offsets1), make!(offsets2), make!(offsets3))
}

/// Build one big-endian GCM counter block directly in an XMM register.
///
/// # Safety
/// Caller must ensure SSE2 is available.
#[cfg(all(target_arch = "x86_64", feature = "aes-gcm"))]
#[target_feature(enable = "sse2")]
#[inline]
unsafe fn x86_gcm_ctr_block_be(iv_words: [u32; 3], ctr: u32) -> core::arch::x86_64::__m128i {
  use core::arch::x86_64::*;

  _mm_set_epi32(
    ctr.swap_bytes() as i32,
    iv_words[2] as i32,
    iv_words[1] as i32,
    iv_words[0] as i32,
  )
}

/// Build four little-endian GCM-SIV counter blocks directly in a VAES register.
///
/// # Safety
/// Caller must ensure AVX-512F is available.
#[cfg(all(target_arch = "x86_64", feature = "aes-gcm-siv"))]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn x86_gcmsiv_ctr_blocks_le_4(suffix_words: [u32; 3], ctr: u32) -> core::arch::x86_64::__m512i {
  use core::arch::x86_64::*;

  let s0 = suffix_words[0] as i32;
  let s1 = suffix_words[1] as i32;
  let s2 = suffix_words[2] as i32;
  let b0 = _mm_set_epi32(s2, s1, s0, ctr as i32);
  let b1 = _mm_set_epi32(s2, s1, s0, ctr.wrapping_add(1) as i32);
  let b2 = _mm_set_epi32(s2, s1, s0, ctr.wrapping_add(2) as i32);
  let b3 = _mm_set_epi32(s2, s1, s0, ctr.wrapping_add(3) as i32);

  let z = _mm512_zextsi128_si512(b0);
  let z = _mm512_inserti32x4(z, b1, 1);
  let z = _mm512_inserti32x4(z, b2, 2);
  _mm512_inserti32x4(z, b3, 3)
}

#[cfg(all(any(target_arch = "x86_64", target_arch = "aarch64"), feature = "aes-gcm"))]
#[inline]
fn ghash_ciphertext_fallback(mut acc: u128, h_polyval: u128, data: &[u8]) -> u128 {
  let (blocks, remainder) = data.as_chunks::<16>();
  for block in blocks {
    acc ^= u128::from_be_bytes(*block);
    acc = super::polyval::clmul128_reduce(acc, h_polyval);
  }

  if !remainder.is_empty() {
    let mut block = [0u8; 16];
    block[..remainder.len()].copy_from_slice(remainder);
    acc ^= u128::from_be_bytes(block);
    acc = super::polyval::clmul128_reduce(acc, h_polyval);
  }

  acc
}

/// AES-256 CTR encryption fused with PCLMUL GHASH accumulation for GCM sealing.
///
/// Classic AES-NI/PCLMUL path for x86_64 hosts without VAES/VPCLMUL. This stays
/// separate from the VAES path so older x86 cores still get a fused CTR/GHASH
/// kernel instead of scalar CTR plus a second GHASH pass.
///
/// # Safety
/// Caller must ensure AES-NI, PCLMULQDQ, SSE2, and SSSE3 are available.
#[cfg(all(target_arch = "x86_64", feature = "aes-gcm"))]
#[target_feature(enable = "aes,pclmulqdq,sse2,ssse3")]
pub(crate) unsafe fn aes256_ctr32_encrypt_be_aesni_pclmul_ghash(
  ek: &Aes256EncKey,
  initial_counter: &[u8; BLOCK_SIZE],
  data: &mut [u8],
  mut acc: u128,
  h_polyval: u128,
  h_powers_rev: &[u128; 4],
) -> u128 {
  use core::arch::x86_64::*;

  // SAFETY: fused x86 AES-NI/PCLMUL sealing because:
  // 1. This function's caller guarantees AES-NI, PCLMULQDQ, SSE2, and SSSE3.
  // 2. `data` is a valid mutable byte slice; pointer arithmetic is bounded by checked loop
  //    conditions.
  // 3. GHASH folds ciphertext registers after encryption, matching GCM authentication semantics.
  unsafe {
    let ni_rk = match &ek.inner {
      KeyInner::X86AesNi(rk) => rk,
      _ => {
        aes256_ctr32_encrypt_be(ek, initial_counter, data);
        return ghash_ciphertext_fallback(acc, h_polyval, data);
      }
    };

    let iv_prefix: [u8; 12] = {
      let mut buf = [0u8; 12];
      buf.copy_from_slice(&initial_counter[..12]);
      buf
    };
    let iv_words = [
      u32::from_le_bytes([iv_prefix[0], iv_prefix[1], iv_prefix[2], iv_prefix[3]]),
      u32::from_le_bytes([iv_prefix[4], iv_prefix[5], iv_prefix[6], iv_prefix[7]]),
      u32::from_le_bytes([iv_prefix[8], iv_prefix[9], iv_prefix[10], iv_prefix[11]]),
    ];
    let mut ctr = u32::from_be_bytes([
      initial_counter[12],
      initial_counter[13],
      initial_counter[14],
      initial_counter[15],
    ]);
    let mut offset = 0usize;

    while offset.strict_add(64) <= data.len() {
      let ctr0 = x86_gcm_ctr_block_be(iv_words, ctr);
      let ctr1 = x86_gcm_ctr_block_be(iv_words, ctr.wrapping_add(1));
      let ctr2 = x86_gcm_ctr_block_be(iv_words, ctr.wrapping_add(2));
      let ctr3 = x86_gcm_ctr_block_be(iv_words, ctr.wrapping_add(3));
      let (ks0, ks1, ks2, ks3) = ni::encrypt_4blocks_aesni(ni_rk, ctr0, ctr1, ctr2, ctr3);

      let ptr = data.as_mut_ptr().add(offset);
      let p0 = _mm_loadu_si128(ptr.cast());
      let p1 = _mm_loadu_si128(ptr.add(16).cast());
      let p2 = _mm_loadu_si128(ptr.add(32).cast());
      let p3 = _mm_loadu_si128(ptr.add(48).cast());
      let c0 = _mm_xor_si128(p0, ks0);
      let c1 = _mm_xor_si128(p1, ks1);
      let c2 = _mm_xor_si128(p2, ks2);
      let c3 = _mm_xor_si128(p3, ks3);
      _mm_storeu_si128(ptr.cast(), c0);
      _mm_storeu_si128(ptr.add(16).cast(), c1);
      _mm_storeu_si128(ptr.add(32).cast(), c2);
      _mm_storeu_si128(ptr.add(48).cast(), c3);
      acc = super::polyval::x86_pclmul_aggregate_4blocks_be_xmm_inline(acc, h_powers_rev, c0, c1, c2, c3);

      ctr = ctr.wrapping_add(4);
      offset = offset.strict_add(64);
    }

    while offset < data.len() {
      let mut counter_block = [0u8; BLOCK_SIZE];
      counter_block[..12].copy_from_slice(&iv_prefix);
      counter_block[12..16].copy_from_slice(&ctr.to_be_bytes());

      let mut keystream = counter_block;
      ni::encrypt_block(ni_rk, &mut keystream);

      let remaining = data.len().strict_sub(offset);
      if remaining >= BLOCK_SIZE {
        let ks = u128::from_ne_bytes(keystream);
        let mut d = [0u8; BLOCK_SIZE];
        d.copy_from_slice(&data[offset..offset.strict_add(BLOCK_SIZE)]);
        let xored = u128::from_ne_bytes(d) ^ ks;
        let ciphertext = xored.to_ne_bytes();
        data[offset..offset.strict_add(BLOCK_SIZE)].copy_from_slice(&ciphertext);
        acc ^= u128::from_be_bytes(ciphertext);
        acc = super::polyval::x86_clmul128_reduce_inline(acc, h_polyval);
        offset = offset.strict_add(BLOCK_SIZE);
      } else {
        let mut block = [0u8; BLOCK_SIZE];
        let mut i = 0usize;
        while i < remaining {
          data[offset.strict_add(i)] ^= keystream[i];
          block[i] = data[offset.strict_add(i)];
          i = i.strict_add(1);
        }
        acc ^= u128::from_be_bytes(block);
        acc = super::polyval::x86_clmul128_reduce_inline(acc, h_polyval);
        offset = offset.strict_add(remaining);
      }
      ctr = ctr.wrapping_add(1);
    }

    acc
  }
}

/// AES-256 CTR decryption fused with PCLMUL GHASH accumulation for GCM open.
///
/// # Safety
/// Caller must ensure AES-NI, PCLMULQDQ, SSE2, and SSSE3 are available.
#[cfg(all(target_arch = "x86_64", feature = "aes-gcm"))]
#[target_feature(enable = "aes,pclmulqdq,sse2,ssse3")]
pub(crate) unsafe fn aes256_ctr32_decrypt_be_aesni_pclmul_ghash(
  ek: &Aes256EncKey,
  initial_counter: &[u8; BLOCK_SIZE],
  data: &mut [u8],
  mut acc: u128,
  h_polyval: u128,
  h_powers_rev: &[u128; 4],
) -> u128 {
  use core::arch::x86_64::*;

  // SAFETY: fused x86 AES-NI/PCLMUL opening because:
  // 1. This function's caller guarantees AES-NI, PCLMULQDQ, SSE2, and SSSE3.
  // 2. Ciphertext registers are folded into GHASH before plaintext is stored back.
  // 3. All pointer arithmetic stays inside checked chunk bounds.
  unsafe {
    let ni_rk = match &ek.inner {
      KeyInner::X86AesNi(rk) => rk,
      _ => {
        acc = ghash_ciphertext_fallback(acc, h_polyval, data);
        aes256_ctr32_encrypt_be(ek, initial_counter, data);
        return acc;
      }
    };

    let iv_prefix: [u8; 12] = {
      let mut buf = [0u8; 12];
      buf.copy_from_slice(&initial_counter[..12]);
      buf
    };
    let iv_words = [
      u32::from_le_bytes([iv_prefix[0], iv_prefix[1], iv_prefix[2], iv_prefix[3]]),
      u32::from_le_bytes([iv_prefix[4], iv_prefix[5], iv_prefix[6], iv_prefix[7]]),
      u32::from_le_bytes([iv_prefix[8], iv_prefix[9], iv_prefix[10], iv_prefix[11]]),
    ];
    let mut ctr = u32::from_be_bytes([
      initial_counter[12],
      initial_counter[13],
      initial_counter[14],
      initial_counter[15],
    ]);
    let mut offset = 0usize;

    while offset.strict_add(64) <= data.len() {
      let ctr0 = x86_gcm_ctr_block_be(iv_words, ctr);
      let ctr1 = x86_gcm_ctr_block_be(iv_words, ctr.wrapping_add(1));
      let ctr2 = x86_gcm_ctr_block_be(iv_words, ctr.wrapping_add(2));
      let ctr3 = x86_gcm_ctr_block_be(iv_words, ctr.wrapping_add(3));
      let (ks0, ks1, ks2, ks3) = ni::encrypt_4blocks_aesni(ni_rk, ctr0, ctr1, ctr2, ctr3);

      let ptr = data.as_mut_ptr().add(offset);
      let c0 = _mm_loadu_si128(ptr.cast());
      let c1 = _mm_loadu_si128(ptr.add(16).cast());
      let c2 = _mm_loadu_si128(ptr.add(32).cast());
      let c3 = _mm_loadu_si128(ptr.add(48).cast());
      acc = super::polyval::x86_pclmul_aggregate_4blocks_be_xmm_inline(acc, h_powers_rev, c0, c1, c2, c3);
      _mm_storeu_si128(ptr.cast(), _mm_xor_si128(c0, ks0));
      _mm_storeu_si128(ptr.add(16).cast(), _mm_xor_si128(c1, ks1));
      _mm_storeu_si128(ptr.add(32).cast(), _mm_xor_si128(c2, ks2));
      _mm_storeu_si128(ptr.add(48).cast(), _mm_xor_si128(c3, ks3));

      ctr = ctr.wrapping_add(4);
      offset = offset.strict_add(64);
    }

    while offset < data.len() {
      let mut counter_block = [0u8; BLOCK_SIZE];
      counter_block[..12].copy_from_slice(&iv_prefix);
      counter_block[12..16].copy_from_slice(&ctr.to_be_bytes());

      let mut keystream = counter_block;
      ni::encrypt_block(ni_rk, &mut keystream);

      let remaining = data.len().strict_sub(offset);
      if remaining >= BLOCK_SIZE {
        let mut ciphertext = [0u8; BLOCK_SIZE];
        ciphertext.copy_from_slice(&data[offset..offset.strict_add(BLOCK_SIZE)]);
        acc ^= u128::from_be_bytes(ciphertext);
        acc = super::polyval::x86_clmul128_reduce_inline(acc, h_polyval);

        let plaintext = u128::from_ne_bytes(ciphertext) ^ u128::from_ne_bytes(keystream);
        data[offset..offset.strict_add(BLOCK_SIZE)].copy_from_slice(&plaintext.to_ne_bytes());
        offset = offset.strict_add(BLOCK_SIZE);
      } else {
        let mut block = [0u8; BLOCK_SIZE];
        block[..remaining].copy_from_slice(&data[offset..offset.strict_add(remaining)]);
        acc ^= u128::from_be_bytes(block);
        acc = super::polyval::x86_clmul128_reduce_inline(acc, h_polyval);

        let mut i = 0usize;
        while i < remaining {
          data[offset.strict_add(i)] ^= keystream[i];
          i = i.strict_add(1);
        }
        offset = offset.strict_add(remaining);
      }
      ctr = ctr.wrapping_add(1);
    }

    acc
  }
}

/// AES-256 CTR encryption fused with GHASH accumulation for GCM sealing.
///
/// Encrypts `data` in place and returns the GHASH accumulator after the
/// ciphertext has been folded in. The incoming `acc` is normally the GHASH
/// state after AAD processing.
///
/// # Safety
/// Caller must ensure AVX-512F + AVX-512VL + AVX-512BW + AVX-512DQ +
/// VAES + VPCLMULQDQ + PCLMULQDQ + AES + SSE2.
#[cfg(all(target_arch = "x86_64", feature = "aes-gcm"))]
#[target_feature(enable = "aes,sse2,avx512f,avx512vl,avx512bw,avx512dq,vaes,vpclmulqdq,pclmulqdq")]
pub(crate) unsafe fn aes256_ctr32_encrypt_be_wide_ghash(
  ek: &Aes256EncKey,
  initial_counter: &[u8; BLOCK_SIZE],
  data: &mut [u8],
  mut acc: u128,
  h_polyval: u128,
  h_powers_rev: &[u128; 4],
  h_powers_rev_16: &[u128; 16],
) -> u128 {
  use core::arch::x86_64::*;

  // SAFETY: fused x86 AES-GCM sealing because:
  // 1. This function's caller guarantees all required x86 target features.
  // 2. `data` is a valid mutable byte slice; all pointer arithmetic stays inside checked chunk
  //    bounds.
  unsafe {
    let ni_rk = match &ek.inner {
      KeyInner::X86AesNi(rk) => rk,
      _ => {
        aes256_ctr32_encrypt_be(ek, initial_counter, data);
        return ghash_ciphertext_fallback(acc, h_polyval, data);
      }
    };

    let iv_prefix: [u8; 12] = {
      let mut buf = [0u8; 12];
      buf.copy_from_slice(&initial_counter[..12]);
      buf
    };
    let iv_words = [
      u32::from_le_bytes([iv_prefix[0], iv_prefix[1], iv_prefix[2], iv_prefix[3]]),
      u32::from_le_bytes([iv_prefix[4], iv_prefix[5], iv_prefix[6], iv_prefix[7]]),
      u32::from_le_bytes([iv_prefix[8], iv_prefix[9], iv_prefix[10], iv_prefix[11]]),
    ];
    let mut ctr = u32::from_be_bytes([
      initial_counter[12],
      initial_counter[13],
      initial_counter[14],
      initial_counter[15],
    ]);
    let mut offset = 0usize;

    #[cfg(target_os = "linux")]
    if data.len() >= 256 {
      let mut state = x86_64_asm::AesGcmX86State::new(acc, ctr);
      // SAFETY: external x86-64 VAES-512 AES-256-GCM seal kernel because:
      // 1. This target-feature function is only entered after VAES, VPCLMULQDQ, AVX-512, AES-NI, and SSE2
      //    were selected by runtime/backend dispatch.
      // 2. `ni_rk.as_ptr()` addresses 15 initialized 128-bit AES-256 round keys.
      // 3. `initial_counter` points to the full 16-byte GCM counter block, and `data` is valid for
      //    `data.len()` mutable bytes.
      // 4. `h_powers_rev_16` contains exactly the [H^16..H] powers required by the 16-block fold.
      // 5. The kernel only processes complete 256-byte chunks and reports the processed byte count so the
      //    Rust fallback below handles every remaining full/partial tail.
      x86_64_asm::rscrypto_aes256_gcm_seal_16x_vaes512_x86_64_linux(
        ni_rk.as_ptr(),
        initial_counter.as_ptr(),
        data.as_mut_ptr(),
        data.len(),
        h_powers_rev_16.as_ptr(),
        &mut state,
      );
      acc = state.acc();
      ctr = state.ctr;
      offset = state.processed;
    }

    while offset.strict_add(256) <= data.len() {
      let (ctr0, ctr1, ctr2, ctr3) = x86_gcm_ctr_blocks_be_16(iv_words, ctr);
      let (ks0, ks1, ks2, ks3) = ni::encrypt_16blocks(ni_rk, ctr0, ctr1, ctr2, ctr3);

      let p0 = _mm512_loadu_si512(data.as_ptr().add(offset).cast());
      let c0 = _mm512_xor_si512(p0, ks0);
      _mm512_storeu_si512(data.as_mut_ptr().add(offset).cast(), c0);

      let p1 = _mm512_loadu_si512(data.as_ptr().add(offset.strict_add(64)).cast());
      let c1 = _mm512_xor_si512(p1, ks1);
      _mm512_storeu_si512(data.as_mut_ptr().add(offset.strict_add(64)).cast(), c1);

      let p2 = _mm512_loadu_si512(data.as_ptr().add(offset.strict_add(128)).cast());
      let c2 = _mm512_xor_si512(p2, ks2);
      _mm512_storeu_si512(data.as_mut_ptr().add(offset.strict_add(128)).cast(), c2);

      let p3 = _mm512_loadu_si512(data.as_ptr().add(offset.strict_add(192)).cast());
      let c3 = _mm512_xor_si512(p3, ks3);
      _mm512_storeu_si512(data.as_mut_ptr().add(offset.strict_add(192)).cast(), c3);
      acc = super::polyval::x86_aggregate_16blocks_be_lanes_inline(acc, h_powers_rev_16, c0, c1, c2, c3);

      ctr = ctr.wrapping_add(16);
      offset = offset.strict_add(256);
    }

    while offset.strict_add(64) <= data.len() {
      let ctr_vec = x86_gcm_ctr_blocks_be_4(iv_words, ctr);
      let keystream = ni::encrypt_4blocks(ni_rk, ctr_vec);
      let plaintext = _mm512_loadu_si512(data.as_ptr().add(offset).cast());
      let ciphertext = _mm512_xor_si512(plaintext, keystream);
      _mm512_storeu_si512(data.as_mut_ptr().add(offset).cast(), ciphertext);
      acc = super::polyval::x86_aggregate_4blocks_be_lanes_inline(acc, h_powers_rev, ciphertext);

      ctr = ctr.wrapping_add(4);
      offset = offset.strict_add(64);
    }

    while offset < data.len() {
      let mut counter_block = [0u8; 16];
      counter_block[..12].copy_from_slice(&iv_prefix);
      counter_block[12..16].copy_from_slice(&ctr.to_be_bytes());

      let mut keystream = counter_block;
      ni::encrypt_block(ni_rk, &mut keystream);

      let remaining = data.len().strict_sub(offset);
      if remaining >= BLOCK_SIZE {
        let ks = u128::from_ne_bytes(keystream);
        let mut d = [0u8; BLOCK_SIZE];
        d.copy_from_slice(&data[offset..offset.strict_add(BLOCK_SIZE)]);
        let xored = u128::from_ne_bytes(d) ^ ks;
        let ciphertext = xored.to_ne_bytes();
        data[offset..offset.strict_add(BLOCK_SIZE)].copy_from_slice(&ciphertext);
        acc ^= u128::from_be_bytes(ciphertext);
        acc = super::polyval::x86_clmul128_reduce_inline(acc, h_polyval);
        offset = offset.strict_add(BLOCK_SIZE);
      } else {
        let mut block = [0u8; BLOCK_SIZE];
        let mut i = 0usize;
        while i < remaining {
          data[offset.strict_add(i)] ^= keystream[i];
          block[i] = data[offset.strict_add(i)];
          i = i.strict_add(1);
        }
        acc ^= u128::from_be_bytes(block);
        acc = super::polyval::x86_clmul128_reduce_inline(acc, h_polyval);
        offset = offset.strict_add(remaining);
      }
      ctr = ctr.wrapping_add(1);
    }

    acc
  }
}

/// AES-256 CTR decryption fused with GHASH accumulation for GCM open.
///
/// Folds ciphertext into GHASH, decrypts the same chunk in place, and returns
/// the accumulator. Authentication is finalized by the caller.
///
/// # Safety
/// Caller must ensure AVX-512F + AVX-512VL + AVX-512BW + AVX-512DQ +
/// VAES + VPCLMULQDQ + PCLMULQDQ + AES + SSE2.
#[cfg(all(target_arch = "x86_64", feature = "aes-gcm"))]
#[target_feature(enable = "aes,sse2,avx512f,avx512vl,avx512bw,avx512dq,vaes,vpclmulqdq,pclmulqdq")]
pub(crate) unsafe fn aes256_ctr32_decrypt_be_wide_ghash(
  ek: &Aes256EncKey,
  initial_counter: &[u8; BLOCK_SIZE],
  data: &mut [u8],
  mut acc: u128,
  h_polyval: u128,
  h_powers_rev: &[u128; 4],
  h_powers_rev_16: &[u128; 16],
) -> u128 {
  use core::arch::x86_64::*;

  // SAFETY: fused x86 AES-GCM opening because:
  // 1. This function's caller guarantees all required x86 target features.
  // 2. `data` is a valid mutable ciphertext slice. Each vector is folded into GHASH before the
  //    corresponding plaintext is stored back.
  // 3. All pointer arithmetic stays inside checked chunk bounds.
  unsafe {
    let ni_rk = match &ek.inner {
      KeyInner::X86AesNi(rk) => rk,
      _ => {
        acc = ghash_ciphertext_fallback(acc, h_polyval, data);
        aes256_ctr32_encrypt_be(ek, initial_counter, data);
        return acc;
      }
    };

    let iv_prefix: [u8; 12] = {
      let mut buf = [0u8; 12];
      buf.copy_from_slice(&initial_counter[..12]);
      buf
    };
    let iv_words = [
      u32::from_le_bytes([iv_prefix[0], iv_prefix[1], iv_prefix[2], iv_prefix[3]]),
      u32::from_le_bytes([iv_prefix[4], iv_prefix[5], iv_prefix[6], iv_prefix[7]]),
      u32::from_le_bytes([iv_prefix[8], iv_prefix[9], iv_prefix[10], iv_prefix[11]]),
    ];
    let mut ctr = u32::from_be_bytes([
      initial_counter[12],
      initial_counter[13],
      initial_counter[14],
      initial_counter[15],
    ]);
    let mut offset = 0usize;

    #[cfg(target_os = "linux")]
    if data.len() >= 256 {
      let mut state = x86_64_asm::AesGcmX86State::new(acc, ctr);
      // SAFETY: external x86-64 VAES-512 AES-256-GCM open kernel because:
      // 1. This target-feature function is only entered after VAES, VPCLMULQDQ, AVX-512, AES-NI, and SSE2
      //    were selected by runtime/backend dispatch.
      // 2. `ni_rk.as_ptr()` addresses 15 initialized 128-bit AES-256 round keys.
      // 3. `initial_counter` points to the full 16-byte GCM counter block, and `data` is valid for
      //    `data.len()` mutable bytes of ciphertext.
      // 4. The kernel folds ciphertext into GHASH before storing plaintext.
      // 5. The kernel only processes complete 256-byte chunks and reports the processed byte count so the
      //    Rust fallback below handles every remaining full/partial tail.
      x86_64_asm::rscrypto_aes256_gcm_open_16x_vaes512_x86_64_linux(
        ni_rk.as_ptr(),
        initial_counter.as_ptr(),
        data.as_mut_ptr(),
        data.len(),
        h_powers_rev_16.as_ptr(),
        &mut state,
      );
      acc = state.acc();
      ctr = state.ctr;
      offset = state.processed;
    }

    while offset.strict_add(256) <= data.len() {
      let (ctr0, ctr1, ctr2, ctr3) = x86_gcm_ctr_blocks_be_16(iv_words, ctr);
      let (ks0, ks1, ks2, ks3) = ni::encrypt_16blocks(ni_rk, ctr0, ctr1, ctr2, ctr3);

      let c0 = _mm512_loadu_si512(data.as_ptr().add(offset).cast());
      let c1 = _mm512_loadu_si512(data.as_ptr().add(offset.strict_add(64)).cast());
      let c2 = _mm512_loadu_si512(data.as_ptr().add(offset.strict_add(128)).cast());
      let c3 = _mm512_loadu_si512(data.as_ptr().add(offset.strict_add(192)).cast());
      acc = super::polyval::x86_aggregate_16blocks_be_lanes_inline(acc, h_powers_rev_16, c0, c1, c2, c3);
      _mm512_storeu_si512(data.as_mut_ptr().add(offset).cast(), _mm512_xor_si512(c0, ks0));
      _mm512_storeu_si512(
        data.as_mut_ptr().add(offset.strict_add(64)).cast(),
        _mm512_xor_si512(c1, ks1),
      );
      _mm512_storeu_si512(
        data.as_mut_ptr().add(offset.strict_add(128)).cast(),
        _mm512_xor_si512(c2, ks2),
      );
      _mm512_storeu_si512(
        data.as_mut_ptr().add(offset.strict_add(192)).cast(),
        _mm512_xor_si512(c3, ks3),
      );

      ctr = ctr.wrapping_add(16);
      offset = offset.strict_add(256);
    }

    while offset.strict_add(64) <= data.len() {
      let ctr_vec = x86_gcm_ctr_blocks_be_4(iv_words, ctr);
      let keystream = ni::encrypt_4blocks(ni_rk, ctr_vec);
      let ciphertext = _mm512_loadu_si512(data.as_ptr().add(offset).cast());
      acc = super::polyval::x86_aggregate_4blocks_be_lanes_inline(acc, h_powers_rev, ciphertext);
      _mm512_storeu_si512(
        data.as_mut_ptr().add(offset).cast(),
        _mm512_xor_si512(ciphertext, keystream),
      );

      ctr = ctr.wrapping_add(4);
      offset = offset.strict_add(64);
    }

    while offset < data.len() {
      let mut counter_block = [0u8; 16];
      counter_block[..12].copy_from_slice(&iv_prefix);
      counter_block[12..16].copy_from_slice(&ctr.to_be_bytes());

      let mut keystream = counter_block;
      ni::encrypt_block(ni_rk, &mut keystream);

      let remaining = data.len().strict_sub(offset);
      if remaining >= BLOCK_SIZE {
        let mut ciphertext = [0u8; BLOCK_SIZE];
        ciphertext.copy_from_slice(&data[offset..offset.strict_add(BLOCK_SIZE)]);
        acc ^= u128::from_be_bytes(ciphertext);
        acc = super::polyval::x86_clmul128_reduce_inline(acc, h_polyval);

        let plaintext = u128::from_ne_bytes(ciphertext) ^ u128::from_ne_bytes(keystream);
        data[offset..offset.strict_add(BLOCK_SIZE)].copy_from_slice(&plaintext.to_ne_bytes());
        offset = offset.strict_add(BLOCK_SIZE);
      } else {
        let mut block = [0u8; BLOCK_SIZE];
        block[..remaining].copy_from_slice(&data[offset..offset.strict_add(remaining)]);
        acc ^= u128::from_be_bytes(block);
        acc = super::polyval::x86_clmul128_reduce_inline(acc, h_polyval);

        let mut i = 0usize;
        while i < remaining {
          data[offset.strict_add(i)] ^= keystream[i];
          i = i.strict_add(1);
        }
        offset = offset.strict_add(remaining);
      }
      ctr = ctr.wrapping_add(1);
    }

    acc
  }
}

/// AES-256 CTR encryption fused with 256-bit VAES/VPCLMUL GHASH accumulation.
///
/// This avoids ZMM data-path pressure on AMD while preserving the fused
/// counter/AES/XOR/GHASH structure used by the 512-bit path.
///
/// # Safety
/// Caller must ensure AVX2 + AVX-512F + AVX-512VL + VAES + VPCLMULQDQ +
/// PCLMULQDQ + AES + SSE2 + SSSE3.
#[cfg(all(target_arch = "x86_64", feature = "aes-gcm"))]
#[target_feature(enable = "aes,sse2,ssse3,avx2,avx512f,avx512vl,vaes,vpclmulqdq,pclmulqdq")]
pub(crate) unsafe fn aes256_ctr32_encrypt_be_y256_ghash(
  ek: &Aes256EncKey,
  initial_counter: &[u8; BLOCK_SIZE],
  data: &mut [u8],
  mut acc: u128,
  h_polyval: u128,
  h_powers_rev: &[u128; 4],
  h_powers_rev_8: &[u128; 8],
) -> u128 {
  use core::arch::x86_64::*;

  // SAFETY: fused x86 VAES-256 AES-GCM sealing because:
  // 1. This function's caller guarantees all required x86 target features.
  // 2. `data` is a valid mutable byte slice; all pointer arithmetic stays inside checked chunk
  //    bounds.
  // 3. GHASH folds ciphertext registers after encryption, matching GCM authentication semantics.
  unsafe {
    let ni_rk = match &ek.inner {
      KeyInner::X86AesNi(rk) => rk,
      _ => {
        aes256_ctr32_encrypt_be(ek, initial_counter, data);
        return ghash_ciphertext_fallback(acc, h_polyval, data);
      }
    };

    let iv_prefix: [u8; 12] = {
      let mut buf = [0u8; 12];
      buf.copy_from_slice(&initial_counter[..12]);
      buf
    };
    let iv_words = [
      u32::from_le_bytes([iv_prefix[0], iv_prefix[1], iv_prefix[2], iv_prefix[3]]),
      u32::from_le_bytes([iv_prefix[4], iv_prefix[5], iv_prefix[6], iv_prefix[7]]),
      u32::from_le_bytes([iv_prefix[8], iv_prefix[9], iv_prefix[10], iv_prefix[11]]),
    ];
    let mut ctr = u32::from_be_bytes([
      initial_counter[12],
      initial_counter[13],
      initial_counter[14],
      initial_counter[15],
    ]);
    let mut offset = 0usize;

    while offset.strict_add(128) <= data.len() {
      let (ctr0, ctr1, ctr2, ctr3) = x86_gcm_ctr_blocks_be_8_y256(iv_words, ctr);
      let (ks0, ks1, ks2, ks3) = ni::encrypt_8blocks_y256(ni_rk, ctr0, ctr1, ctr2, ctr3);

      let p0 = _mm256_loadu_si256(data.as_ptr().add(offset).cast());
      let c0 = _mm256_xor_si256(p0, ks0);
      _mm256_storeu_si256(data.as_mut_ptr().add(offset).cast(), c0);

      let p1 = _mm256_loadu_si256(data.as_ptr().add(offset.strict_add(32)).cast());
      let c1 = _mm256_xor_si256(p1, ks1);
      _mm256_storeu_si256(data.as_mut_ptr().add(offset.strict_add(32)).cast(), c1);

      let p2 = _mm256_loadu_si256(data.as_ptr().add(offset.strict_add(64)).cast());
      let c2 = _mm256_xor_si256(p2, ks2);
      _mm256_storeu_si256(data.as_mut_ptr().add(offset.strict_add(64)).cast(), c2);

      let p3 = _mm256_loadu_si256(data.as_ptr().add(offset.strict_add(96)).cast());
      let c3 = _mm256_xor_si256(p3, ks3);
      _mm256_storeu_si256(data.as_mut_ptr().add(offset.strict_add(96)).cast(), c3);
      acc = super::polyval::x86_aggregate_8blocks_be_lanes_256_inline(acc, h_powers_rev_8, c0, c1, c2, c3);

      ctr = ctr.wrapping_add(8);
      offset = offset.strict_add(128);
    }

    while offset.strict_add(64) <= data.len() {
      let ctr0 = x86_gcm_ctr_block_be(iv_words, ctr);
      let ctr1 = x86_gcm_ctr_block_be(iv_words, ctr.wrapping_add(1));
      let ctr2 = x86_gcm_ctr_block_be(iv_words, ctr.wrapping_add(2));
      let ctr3 = x86_gcm_ctr_block_be(iv_words, ctr.wrapping_add(3));
      let (ks0, ks1, ks2, ks3) = ni::encrypt_4blocks_aesni(ni_rk, ctr0, ctr1, ctr2, ctr3);

      let ptr = data.as_mut_ptr().add(offset);
      let p0 = _mm_loadu_si128(ptr.cast());
      let p1 = _mm_loadu_si128(ptr.add(16).cast());
      let p2 = _mm_loadu_si128(ptr.add(32).cast());
      let p3 = _mm_loadu_si128(ptr.add(48).cast());
      let c0 = _mm_xor_si128(p0, ks0);
      let c1 = _mm_xor_si128(p1, ks1);
      let c2 = _mm_xor_si128(p2, ks2);
      let c3 = _mm_xor_si128(p3, ks3);
      _mm_storeu_si128(ptr.cast(), c0);
      _mm_storeu_si128(ptr.add(16).cast(), c1);
      _mm_storeu_si128(ptr.add(32).cast(), c2);
      _mm_storeu_si128(ptr.add(48).cast(), c3);
      acc = super::polyval::x86_pclmul_aggregate_4blocks_be_xmm_inline(acc, h_powers_rev, c0, c1, c2, c3);

      ctr = ctr.wrapping_add(4);
      offset = offset.strict_add(64);
    }

    while offset < data.len() {
      let mut counter_block = [0u8; 16];
      counter_block[..12].copy_from_slice(&iv_prefix);
      counter_block[12..16].copy_from_slice(&ctr.to_be_bytes());

      let mut keystream = counter_block;
      ni::encrypt_block(ni_rk, &mut keystream);

      let remaining = data.len().strict_sub(offset);
      if remaining >= BLOCK_SIZE {
        let ks = u128::from_ne_bytes(keystream);
        let mut d = [0u8; BLOCK_SIZE];
        d.copy_from_slice(&data[offset..offset.strict_add(BLOCK_SIZE)]);
        let xored = u128::from_ne_bytes(d) ^ ks;
        let ciphertext = xored.to_ne_bytes();
        data[offset..offset.strict_add(BLOCK_SIZE)].copy_from_slice(&ciphertext);
        acc ^= u128::from_be_bytes(ciphertext);
        acc = super::polyval::x86_clmul128_reduce_inline(acc, h_polyval);
        offset = offset.strict_add(BLOCK_SIZE);
      } else {
        let mut block = [0u8; BLOCK_SIZE];
        let mut i = 0usize;
        while i < remaining {
          data[offset.strict_add(i)] ^= keystream[i];
          block[i] = data[offset.strict_add(i)];
          i = i.strict_add(1);
        }
        acc ^= u128::from_be_bytes(block);
        acc = super::polyval::x86_clmul128_reduce_inline(acc, h_polyval);
        offset = offset.strict_add(remaining);
      }
      ctr = ctr.wrapping_add(1);
    }

    acc
  }
}

/// AES-256 CTR decryption fused with 256-bit VAES/VPCLMUL GHASH accumulation.
///
/// # Safety
/// Caller must ensure AVX2 + AVX-512F + AVX-512VL + VAES + VPCLMULQDQ +
/// PCLMULQDQ + AES + SSE2 + SSSE3.
#[cfg(all(target_arch = "x86_64", feature = "aes-gcm"))]
#[target_feature(enable = "aes,sse2,ssse3,avx2,avx512f,avx512vl,vaes,vpclmulqdq,pclmulqdq")]
pub(crate) unsafe fn aes256_ctr32_decrypt_be_y256_ghash(
  ek: &Aes256EncKey,
  initial_counter: &[u8; BLOCK_SIZE],
  data: &mut [u8],
  mut acc: u128,
  h_polyval: u128,
  h_powers_rev: &[u128; 4],
  h_powers_rev_8: &[u128; 8],
) -> u128 {
  use core::arch::x86_64::*;

  // SAFETY: fused x86 VAES-256 AES-GCM opening because:
  // 1. This function's caller guarantees all required x86 target features.
  // 2. Ciphertext registers are folded into GHASH before plaintext is stored back.
  // 3. All pointer arithmetic stays inside checked chunk bounds.
  unsafe {
    let ni_rk = match &ek.inner {
      KeyInner::X86AesNi(rk) => rk,
      _ => {
        acc = ghash_ciphertext_fallback(acc, h_polyval, data);
        aes256_ctr32_encrypt_be(ek, initial_counter, data);
        return acc;
      }
    };

    let iv_prefix: [u8; 12] = {
      let mut buf = [0u8; 12];
      buf.copy_from_slice(&initial_counter[..12]);
      buf
    };
    let iv_words = [
      u32::from_le_bytes([iv_prefix[0], iv_prefix[1], iv_prefix[2], iv_prefix[3]]),
      u32::from_le_bytes([iv_prefix[4], iv_prefix[5], iv_prefix[6], iv_prefix[7]]),
      u32::from_le_bytes([iv_prefix[8], iv_prefix[9], iv_prefix[10], iv_prefix[11]]),
    ];
    let mut ctr = u32::from_be_bytes([
      initial_counter[12],
      initial_counter[13],
      initial_counter[14],
      initial_counter[15],
    ]);
    let mut offset = 0usize;

    while offset.strict_add(128) <= data.len() {
      let (ctr0, ctr1, ctr2, ctr3) = x86_gcm_ctr_blocks_be_8_y256(iv_words, ctr);
      let (ks0, ks1, ks2, ks3) = ni::encrypt_8blocks_y256(ni_rk, ctr0, ctr1, ctr2, ctr3);

      let c0 = _mm256_loadu_si256(data.as_ptr().add(offset).cast());
      let c1 = _mm256_loadu_si256(data.as_ptr().add(offset.strict_add(32)).cast());
      let c2 = _mm256_loadu_si256(data.as_ptr().add(offset.strict_add(64)).cast());
      let c3 = _mm256_loadu_si256(data.as_ptr().add(offset.strict_add(96)).cast());
      acc = super::polyval::x86_aggregate_8blocks_be_lanes_256_inline(acc, h_powers_rev_8, c0, c1, c2, c3);

      _mm256_storeu_si256(data.as_mut_ptr().add(offset).cast(), _mm256_xor_si256(c0, ks0));
      _mm256_storeu_si256(
        data.as_mut_ptr().add(offset.strict_add(32)).cast(),
        _mm256_xor_si256(c1, ks1),
      );
      _mm256_storeu_si256(
        data.as_mut_ptr().add(offset.strict_add(64)).cast(),
        _mm256_xor_si256(c2, ks2),
      );
      _mm256_storeu_si256(
        data.as_mut_ptr().add(offset.strict_add(96)).cast(),
        _mm256_xor_si256(c3, ks3),
      );

      ctr = ctr.wrapping_add(8);
      offset = offset.strict_add(128);
    }

    while offset.strict_add(64) <= data.len() {
      let ctr0 = x86_gcm_ctr_block_be(iv_words, ctr);
      let ctr1 = x86_gcm_ctr_block_be(iv_words, ctr.wrapping_add(1));
      let ctr2 = x86_gcm_ctr_block_be(iv_words, ctr.wrapping_add(2));
      let ctr3 = x86_gcm_ctr_block_be(iv_words, ctr.wrapping_add(3));
      let (ks0, ks1, ks2, ks3) = ni::encrypt_4blocks_aesni(ni_rk, ctr0, ctr1, ctr2, ctr3);

      let ptr = data.as_mut_ptr().add(offset);
      let c0 = _mm_loadu_si128(ptr.cast());
      let c1 = _mm_loadu_si128(ptr.add(16).cast());
      let c2 = _mm_loadu_si128(ptr.add(32).cast());
      let c3 = _mm_loadu_si128(ptr.add(48).cast());
      acc = super::polyval::x86_pclmul_aggregate_4blocks_be_xmm_inline(acc, h_powers_rev, c0, c1, c2, c3);
      _mm_storeu_si128(ptr.cast(), _mm_xor_si128(c0, ks0));
      _mm_storeu_si128(ptr.add(16).cast(), _mm_xor_si128(c1, ks1));
      _mm_storeu_si128(ptr.add(32).cast(), _mm_xor_si128(c2, ks2));
      _mm_storeu_si128(ptr.add(48).cast(), _mm_xor_si128(c3, ks3));

      ctr = ctr.wrapping_add(4);
      offset = offset.strict_add(64);
    }

    while offset < data.len() {
      let mut counter_block = [0u8; 16];
      counter_block[..12].copy_from_slice(&iv_prefix);
      counter_block[12..16].copy_from_slice(&ctr.to_be_bytes());

      let mut keystream = counter_block;
      ni::encrypt_block(ni_rk, &mut keystream);

      let remaining = data.len().strict_sub(offset);
      if remaining >= BLOCK_SIZE {
        let mut ciphertext = [0u8; BLOCK_SIZE];
        ciphertext.copy_from_slice(&data[offset..offset.strict_add(BLOCK_SIZE)]);
        acc ^= u128::from_be_bytes(ciphertext);
        acc = super::polyval::x86_clmul128_reduce_inline(acc, h_polyval);

        let plaintext = u128::from_ne_bytes(ciphertext) ^ u128::from_ne_bytes(keystream);
        data[offset..offset.strict_add(BLOCK_SIZE)].copy_from_slice(&plaintext.to_ne_bytes());
        offset = offset.strict_add(BLOCK_SIZE);
      } else {
        let mut block = [0u8; BLOCK_SIZE];
        block[..remaining].copy_from_slice(&data[offset..offset.strict_add(remaining)]);
        acc ^= u128::from_be_bytes(block);
        acc = super::polyval::x86_clmul128_reduce_inline(acc, h_polyval);

        let mut i = 0usize;
        while i < remaining {
          data[offset.strict_add(i)] ^= keystream[i];
          i = i.strict_add(1);
        }
        offset = offset.strict_add(remaining);
      }
      ctr = ctr.wrapping_add(1);
    }

    acc
  }
}

/// AES-256 CTR encryption fused with PMULL GHASH accumulation for GCM sealing.
///
/// Encrypts `data` in place and returns the GHASH accumulator after the
/// ciphertext has been folded in. The incoming `acc` is normally the GHASH
/// state after AAD processing.
///
/// # Safety
/// Caller must ensure AES-CE and PMULL are available.
#[cfg(all(target_arch = "aarch64", feature = "aes-gcm"))]
#[target_feature(enable = "aes,neon")]
pub(crate) unsafe fn aes256_ctr32_encrypt_be_aarch64_ghash(
  ek: &Aes256EncKey,
  initial_counter: &[u8; BLOCK_SIZE],
  data: &mut [u8],
  mut acc: u128,
  tables: &Aarch64GcmTables<'_>,
) -> u128 {
  // SAFETY: fused aarch64 AES-GCM sealing because:
  // 1. This function's caller guarantees AES-CE and PMULL availability.
  // 2. `data` is a valid mutable byte slice; all chunk processing uses checked slice ranges and the
  //    tail path handles partial blocks.
  unsafe {
    let ce_rk = match &ek.inner {
      KeyInner::Aarch64Aes(rk) => rk,
      _ => {
        aes256_ctr32_encrypt_be(ek, initial_counter, data);
        return ghash_ciphertext_fallback(acc, tables.h_polyval, data);
      }
    };

    let iv_prefix: [u8; 12] = {
      let mut buf = [0u8; 12];
      buf.copy_from_slice(&initial_counter[..12]);
      buf
    };
    let mut ctr = u32::from_be_bytes([
      initial_counter[12],
      initial_counter[13],
      initial_counter[14],
      initial_counter[15],
    ]);
    let mut offset = 0usize;

    if offset.strict_add(128) <= data.len() {
      let state = ce::encrypt_ctr32_be_xor_ghash_128b_chunks_core(ce_rk, &iv_prefix, ctr, data, acc, tables);
      acc = state.0;
      ctr = state.1;
      offset = state.2;
    }

    while offset.strict_add(64) <= data.len() {
      let end = offset.strict_add(64);
      let blocks = ce::encrypt_ctr32_be_xor_4blocks_core(ce_rk, &iv_prefix, ctr, &mut data[offset..end]);
      acc = super::polyval::aarch64_aggregate_4blocks_inline(acc, tables.h_powers_rev, &blocks);
      ctr = ctr.wrapping_add(4);
      offset = end;
    }

    while offset < data.len() {
      let mut counter_block = [0u8; BLOCK_SIZE];
      counter_block[..12].copy_from_slice(&iv_prefix);
      counter_block[12..16].copy_from_slice(&ctr.to_be_bytes());

      let mut keystream = counter_block;
      ce::encrypt_block_core(ce_rk, &mut keystream);

      let remaining = data.len().strict_sub(offset);
      if remaining >= BLOCK_SIZE {
        let ks = u128::from_ne_bytes(keystream);
        let mut d = [0u8; BLOCK_SIZE];
        d.copy_from_slice(&data[offset..offset.strict_add(BLOCK_SIZE)]);
        let xored = u128::from_ne_bytes(d) ^ ks;
        let ciphertext = xored.to_ne_bytes();
        data[offset..offset.strict_add(BLOCK_SIZE)].copy_from_slice(&ciphertext);
        acc ^= u128::from_be_bytes(ciphertext);
        acc = super::polyval::aarch64_clmul128_reduce_inline(acc, tables.h_polyval);
        offset = offset.strict_add(BLOCK_SIZE);
      } else {
        let mut block = [0u8; BLOCK_SIZE];
        let mut i = 0usize;
        while i < remaining {
          data[offset.strict_add(i)] ^= keystream[i];
          block[i] = data[offset.strict_add(i)];
          i = i.strict_add(1);
        }
        acc ^= u128::from_be_bytes(block);
        acc = super::polyval::aarch64_clmul128_reduce_inline(acc, tables.h_polyval);
        offset = offset.strict_add(remaining);
      }
      ctr = ctr.wrapping_add(1);
    }

    acc
  }
}

/// AES-256 CTR decryption fused with PMULL GHASH accumulation for GCM open.
///
/// Folds ciphertext into GHASH, decrypts the same chunk in place, and returns
/// the accumulator. Authentication is still finalized and checked by the
/// caller, which zeroizes the buffer on failure.
///
/// # Safety
/// Caller must ensure AES-CE and PMULL are available.
#[cfg(all(target_arch = "aarch64", feature = "aes-gcm"))]
#[target_feature(enable = "aes,neon")]
pub(crate) unsafe fn aes256_ctr32_decrypt_be_aarch64_ghash(
  ek: &Aes256EncKey,
  initial_counter: &[u8; BLOCK_SIZE],
  data: &mut [u8],
  mut acc: u128,
  tables: &Aarch64GcmTables<'_>,
) -> u128 {
  // SAFETY: fused aarch64 AES-GCM opening because:
  // 1. This function's caller guarantees AES-CE and PMULL availability.
  // 2. `data` is a valid mutable ciphertext slice. Ciphertext blocks are copied before in-place
  //    decryption, so GHASH always authenticates the original bytes.
  // 3. All chunk processing uses checked slice bounds and the tail path handles partial blocks.
  unsafe {
    let ce_rk = match &ek.inner {
      KeyInner::Aarch64Aes(rk) => rk,
      _ => {
        acc = ghash_ciphertext_fallback(acc, tables.h_polyval, data);
        aes256_ctr32_encrypt_be(ek, initial_counter, data);
        return acc;
      }
    };

    let iv_prefix: [u8; 12] = {
      let mut buf = [0u8; 12];
      buf.copy_from_slice(&initial_counter[..12]);
      buf
    };
    let mut ctr = u32::from_be_bytes([
      initial_counter[12],
      initial_counter[13],
      initial_counter[14],
      initial_counter[15],
    ]);
    let state = ce::decrypt_ctr32_be_xor_ghash_128b_chunks_core(ce_rk, &iv_prefix, ctr, data, acc, tables);
    acc = state.0;
    ctr = state.1;
    let mut offset = state.2;

    while offset.strict_add(64) <= data.len() {
      let end = offset.strict_add(64);
      let chunk = &data[offset..end];
      let (ciphertext_blocks, _) = chunk.as_chunks::<16>();
      let blocks = [
        u128::from_be_bytes(ciphertext_blocks[0]),
        u128::from_be_bytes(ciphertext_blocks[1]),
        u128::from_be_bytes(ciphertext_blocks[2]),
        u128::from_be_bytes(ciphertext_blocks[3]),
      ];
      let _ = ce::encrypt_ctr32_be_xor_4blocks_core(ce_rk, &iv_prefix, ctr, &mut data[offset..end]);
      acc = super::polyval::aarch64_aggregate_4blocks_inline(acc, tables.h_powers_rev, &blocks);
      ctr = ctr.wrapping_add(4);
      offset = end;
    }

    while offset < data.len() {
      let mut counter_block = [0u8; BLOCK_SIZE];
      counter_block[..12].copy_from_slice(&iv_prefix);
      counter_block[12..16].copy_from_slice(&ctr.to_be_bytes());

      let mut keystream = counter_block;
      ce::encrypt_block_core(ce_rk, &mut keystream);

      let remaining = data.len().strict_sub(offset);
      if remaining >= BLOCK_SIZE {
        let mut ciphertext = [0u8; BLOCK_SIZE];
        ciphertext.copy_from_slice(&data[offset..offset.strict_add(BLOCK_SIZE)]);
        acc ^= u128::from_be_bytes(ciphertext);
        acc = super::polyval::aarch64_clmul128_reduce_inline(acc, tables.h_polyval);

        let ks = u128::from_ne_bytes(keystream);
        let plaintext = u128::from_ne_bytes(ciphertext) ^ ks;
        data[offset..offset.strict_add(BLOCK_SIZE)].copy_from_slice(&plaintext.to_ne_bytes());
        offset = offset.strict_add(BLOCK_SIZE);
      } else {
        let mut block = [0u8; BLOCK_SIZE];
        block[..remaining].copy_from_slice(&data[offset..offset.strict_add(remaining)]);
        acc ^= u128::from_be_bytes(block);
        acc = super::polyval::aarch64_clmul128_reduce_inline(acc, tables.h_polyval);

        let mut i = 0usize;
        while i < remaining {
          data[offset.strict_add(i)] ^= keystream[i];
          i = i.strict_add(1);
        }
        offset = offset.strict_add(remaining);
      }
      ctr = ctr.wrapping_add(1);
    }

    acc
  }
}

// ---------------------------------------------------------------------------
// AES-128 CTR for GCM (big-endian 32-bit counter in bytes 12..15)
// ---------------------------------------------------------------------------

/// AES-128 CTR encryption/decryption for GCM.
///
/// Mirrors [`aes256_ctr32_encrypt_be`]: the counter occupies bytes 12..15
/// of the 16-byte counter block and increments as a big-endian 32-bit
/// integer per NIST SP 800-38D `inc_32`.
#[inline]
#[cfg(feature = "aes-gcm")]
pub(crate) fn aes128_ctr32_encrypt_be(ek: &Aes128EncKey, initial_counter: &[u8; BLOCK_SIZE], data: &mut [u8]) {
  let mut counter_block = *initial_counter;
  let mut ctr = u32::from_be_bytes([
    counter_block[12],
    counter_block[13],
    counter_block[14],
    counter_block[15],
  ]);
  let mut offset = 0usize;

  #[cfg(target_arch = "aarch64")]
  if let Key128Inner::Aarch64Aes(ce_rk) = &ek.inner {
    let iv_prefix: [u8; 12] = {
      let mut buf = [0u8; 12];
      buf.copy_from_slice(&initial_counter[..12]);
      buf
    };

    while offset.strict_add(128) <= data.len() {
      let end = offset.strict_add(128);
      // SAFETY: direct aarch64 CTR batch because:
      // 1. `Aarch64Aes` keys are only constructed after runtime detection confirms AES-CE.
      // 2. The loop condition guarantees the helper receives at least 128 writable bytes.
      // 3. The counter prefix is copied from the caller-provided GCM counter block.
      let _ = unsafe { ce::encrypt_ctr32_be_xor_8blocks_128_core(ce_rk, &iv_prefix, ctr, &mut data[offset..end]) };
      ctr = ctr.wrapping_add(8);
      offset = end;
    }

    while offset.strict_add(64) <= data.len() {
      let end = offset.strict_add(64);
      // SAFETY: direct aarch64 CTR batch because:
      // 1. `Aarch64Aes` keys are only constructed after runtime detection confirms AES-CE.
      // 2. The loop condition guarantees the helper receives at least 64 writable bytes.
      // 3. The counter prefix is copied from the caller-provided GCM counter block.
      let _ = unsafe { ce::encrypt_ctr32_be_xor_4blocks_128_core(ce_rk, &iv_prefix, ctr, &mut data[offset..end]) };
      ctr = ctr.wrapping_add(4);
      offset = end;
    }
  }

  #[cfg(any(
    target_arch = "aarch64",
    target_arch = "powerpc64",
    target_arch = "riscv64",
    target_arch = "s390x"
  ))]
  if aes128_ctr32_be_uses_block_batch(ek) {
    let iv_prefix: [u8; 12] = {
      let mut buf = [0u8; 12];
      buf.copy_from_slice(&initial_counter[..12]);
      buf
    };

    while offset.strict_add(64) <= data.len() {
      let mut keystream = [[0u8; BLOCK_SIZE]; 4];
      let mut i = 0u32;
      while i < 4 {
        keystream[i as usize][..12].copy_from_slice(&iv_prefix);
        keystream[i as usize][12..16].copy_from_slice(&ctr.wrapping_add(i).to_be_bytes());
        i = i.strict_add(1);
      }

      aes128_encrypt_blocks_ecb(ek, &mut keystream);

      let mut lane = 0usize;
      while lane < 4 {
        let block_offset = offset.strict_add(lane.strict_mul(BLOCK_SIZE));
        let ks = u128::from_ne_bytes(keystream[lane]);
        let mut d = [0u8; BLOCK_SIZE];
        d.copy_from_slice(&data[block_offset..block_offset.strict_add(BLOCK_SIZE)]);
        let xored = u128::from_ne_bytes(d) ^ ks;
        data[block_offset..block_offset.strict_add(BLOCK_SIZE)].copy_from_slice(&xored.to_ne_bytes());
        lane = lane.strict_add(1);
      }

      ctr = ctr.wrapping_add(4);
      offset = offset.strict_add(64);
    }

    counter_block[..12].copy_from_slice(&iv_prefix);
  }

  while offset < data.len() {
    counter_block[12..16].copy_from_slice(&ctr.to_be_bytes());

    let mut keystream = counter_block;
    aes128_encrypt_block(ek, &mut keystream);

    let remaining = data.len().strict_sub(offset);
    if remaining >= BLOCK_SIZE {
      let ks = u128::from_ne_bytes(keystream);
      let mut d = [0u8; BLOCK_SIZE];
      d.copy_from_slice(&data[offset..offset.strict_add(BLOCK_SIZE)]);
      let xored = u128::from_ne_bytes(d) ^ ks;
      data[offset..offset.strict_add(BLOCK_SIZE)].copy_from_slice(&xored.to_ne_bytes());
      offset = offset.strict_add(BLOCK_SIZE);
    } else {
      let mut i = 0usize;
      while i < remaining {
        data[offset.strict_add(i)] ^= keystream[i];
        i = i.strict_add(1);
      }
      offset = offset.strict_add(remaining);
    }

    ctr = ctr.wrapping_add(1);
  }
}

/// AES-128 CTR encryption fused with PMULL GHASH accumulation for GCM sealing.
///
/// Encrypts `data` in place and returns the GHASH accumulator after the
/// ciphertext has been folded in. The incoming `acc` is normally the GHASH
/// state after AAD processing.
///
/// # Safety
/// Caller must ensure AES-CE and PMULL are available.
#[cfg(all(target_arch = "aarch64", feature = "aes-gcm"))]
#[target_feature(enable = "aes,neon")]
pub(crate) unsafe fn aes128_ctr32_encrypt_be_aarch64_ghash(
  ek: &Aes128EncKey,
  initial_counter: &[u8; BLOCK_SIZE],
  data: &mut [u8],
  mut acc: u128,
  tables: &Aarch64GcmTables<'_>,
) -> u128 {
  // SAFETY: fused aarch64 AES-GCM sealing because:
  // 1. This function's caller guarantees AES-CE and PMULL availability.
  // 2. `data` is a valid mutable byte slice; all chunk processing uses checked slice ranges and the
  //    tail path handles partial blocks.
  unsafe {
    let ce_rk = match &ek.inner {
      Key128Inner::Aarch64Aes(rk) => rk,
      _ => {
        aes128_ctr32_encrypt_be(ek, initial_counter, data);
        return ghash_ciphertext_fallback(acc, tables.h_polyval, data);
      }
    };

    let iv_prefix: [u8; 12] = {
      let mut buf = [0u8; 12];
      buf.copy_from_slice(&initial_counter[..12]);
      buf
    };
    let mut ctr = u32::from_be_bytes([
      initial_counter[12],
      initial_counter[13],
      initial_counter[14],
      initial_counter[15],
    ]);
    let mut offset = 0usize;

    // Short messages are faster if GHASH consumes ciphertext registers immediately. The lagged
    // pipeline below wins once there is enough work to hide GHASH behind the next AES chunk.
    if data.len() <= 256 {
      while offset.strict_add(128) <= data.len() {
        let end = offset.strict_add(128);
        let blocks = ce::encrypt_ctr32_be_xor_8blocks_128_core(ce_rk, &iv_prefix, ctr, &mut data[offset..end]);
        acc = super::polyval::aarch64_aggregate_8blocks_be_lanes_inline(acc, tables.h_powers_rev_8, &blocks);
        ctr = ctr.wrapping_add(8);
        offset = end;
      }
    } else if offset.strict_add(128) <= data.len() {
      let state = ce::encrypt_ctr32_be_xor_ghash_128b_chunks_128_core(ce_rk, &iv_prefix, ctr, data, acc, tables);
      acc = state.0;
      ctr = state.1;
      offset = state.2;
    }

    while offset.strict_add(64) <= data.len() {
      let end = offset.strict_add(64);
      let blocks = ce::encrypt_ctr32_be_xor_4blocks_128_core(ce_rk, &iv_prefix, ctr, &mut data[offset..end]);
      acc = super::polyval::aarch64_aggregate_4blocks_inline(acc, tables.h_powers_rev, &blocks);
      ctr = ctr.wrapping_add(4);
      offset = end;
    }

    while offset < data.len() {
      let mut counter_block = [0u8; BLOCK_SIZE];
      counter_block[..12].copy_from_slice(&iv_prefix);
      counter_block[12..16].copy_from_slice(&ctr.to_be_bytes());

      let mut keystream = counter_block;
      ce::encrypt_block_128_core(ce_rk, &mut keystream);

      let remaining = data.len().strict_sub(offset);
      if remaining >= BLOCK_SIZE {
        let ks = u128::from_ne_bytes(keystream);
        let mut d = [0u8; BLOCK_SIZE];
        d.copy_from_slice(&data[offset..offset.strict_add(BLOCK_SIZE)]);
        let xored = u128::from_ne_bytes(d) ^ ks;
        let ciphertext = xored.to_ne_bytes();
        data[offset..offset.strict_add(BLOCK_SIZE)].copy_from_slice(&ciphertext);
        acc ^= u128::from_be_bytes(ciphertext);
        acc = super::polyval::aarch64_clmul128_reduce_inline(acc, tables.h_polyval);
        offset = offset.strict_add(BLOCK_SIZE);
      } else {
        let mut block = [0u8; BLOCK_SIZE];
        let mut i = 0usize;
        while i < remaining {
          data[offset.strict_add(i)] ^= keystream[i];
          block[i] = data[offset.strict_add(i)];
          i = i.strict_add(1);
        }
        acc ^= u128::from_be_bytes(block);
        acc = super::polyval::aarch64_clmul128_reduce_inline(acc, tables.h_polyval);
        offset = offset.strict_add(remaining);
      }
      ctr = ctr.wrapping_add(1);
    }

    acc
  }
}

/// AES-128 CTR decryption fused with PMULL GHASH accumulation for GCM open.
///
/// Folds ciphertext into GHASH, decrypts the same chunk in place, and returns
/// the accumulator. Authentication is still finalized and checked by the
/// caller, which zeroizes the buffer on failure.
///
/// # Safety
/// Caller must ensure AES-CE and PMULL are available.
#[cfg(all(target_arch = "aarch64", feature = "aes-gcm"))]
#[target_feature(enable = "aes,neon")]
pub(crate) unsafe fn aes128_ctr32_decrypt_be_aarch64_ghash(
  ek: &Aes128EncKey,
  initial_counter: &[u8; BLOCK_SIZE],
  data: &mut [u8],
  mut acc: u128,
  tables: &Aarch64GcmTables<'_>,
) -> u128 {
  // SAFETY: fused aarch64 AES-GCM opening because:
  // 1. This function's caller guarantees AES-CE and PMULL availability.
  // 2. `data` is a valid mutable ciphertext slice. Ciphertext blocks are copied before in-place
  //    decryption, so GHASH always authenticates the original bytes.
  // 3. All chunk processing uses checked slice bounds and the tail path handles partial blocks.
  unsafe {
    let ce_rk = match &ek.inner {
      Key128Inner::Aarch64Aes(rk) => rk,
      _ => {
        acc = ghash_ciphertext_fallback(acc, tables.h_polyval, data);
        aes128_ctr32_encrypt_be(ek, initial_counter, data);
        return acc;
      }
    };

    let iv_prefix: [u8; 12] = {
      let mut buf = [0u8; 12];
      buf.copy_from_slice(&initial_counter[..12]);
      buf
    };
    let mut ctr = u32::from_be_bytes([
      initial_counter[12],
      initial_counter[13],
      initial_counter[14],
      initial_counter[15],
    ]);
    let state = ce::decrypt_ctr32_be_xor_ghash_128b_chunks_128_core(ce_rk, &iv_prefix, ctr, data, acc, tables);
    acc = state.0;
    ctr = state.1;
    let mut offset = state.2;

    while offset.strict_add(64) <= data.len() {
      let end = offset.strict_add(64);
      let chunk = &data[offset..end];
      let (ciphertext_blocks, _) = chunk.as_chunks::<16>();
      let blocks = [
        u128::from_be_bytes(ciphertext_blocks[0]),
        u128::from_be_bytes(ciphertext_blocks[1]),
        u128::from_be_bytes(ciphertext_blocks[2]),
        u128::from_be_bytes(ciphertext_blocks[3]),
      ];
      let _ = ce::encrypt_ctr32_be_xor_4blocks_128_core(ce_rk, &iv_prefix, ctr, &mut data[offset..end]);
      acc = super::polyval::aarch64_aggregate_4blocks_inline(acc, tables.h_powers_rev, &blocks);
      ctr = ctr.wrapping_add(4);
      offset = end;
    }

    while offset < data.len() {
      let mut counter_block = [0u8; BLOCK_SIZE];
      counter_block[..12].copy_from_slice(&iv_prefix);
      counter_block[12..16].copy_from_slice(&ctr.to_be_bytes());

      let mut keystream = counter_block;
      ce::encrypt_block_128_core(ce_rk, &mut keystream);

      let remaining = data.len().strict_sub(offset);
      if remaining >= BLOCK_SIZE {
        let mut ciphertext = [0u8; BLOCK_SIZE];
        ciphertext.copy_from_slice(&data[offset..offset.strict_add(BLOCK_SIZE)]);
        acc ^= u128::from_be_bytes(ciphertext);
        acc = super::polyval::aarch64_clmul128_reduce_inline(acc, tables.h_polyval);

        let ks = u128::from_ne_bytes(keystream);
        let plaintext = u128::from_ne_bytes(ciphertext) ^ ks;
        data[offset..offset.strict_add(BLOCK_SIZE)].copy_from_slice(&plaintext.to_ne_bytes());
        offset = offset.strict_add(BLOCK_SIZE);
      } else {
        let mut block = [0u8; BLOCK_SIZE];
        block[..remaining].copy_from_slice(&data[offset..offset.strict_add(remaining)]);
        acc ^= u128::from_be_bytes(block);
        acc = super::polyval::aarch64_clmul128_reduce_inline(acc, tables.h_polyval);

        let mut i = 0usize;
        while i < remaining {
          data[offset.strict_add(i)] ^= keystream[i];
          i = i.strict_add(1);
        }
        offset = offset.strict_add(remaining);
      }
      ctr = ctr.wrapping_add(1);
    }

    acc
  }
}

/// AES-128 CTR encryption fused with PCLMUL GHASH accumulation for GCM sealing.
///
/// # Safety
/// Caller must ensure AES-NI, PCLMULQDQ, SSE2, and SSSE3 are available.
#[cfg(all(target_arch = "x86_64", feature = "aes-gcm"))]
#[target_feature(enable = "aes,pclmulqdq,sse2,ssse3")]
pub(crate) unsafe fn aes128_ctr32_encrypt_be_aesni_pclmul_ghash(
  ek: &Aes128EncKey,
  initial_counter: &[u8; BLOCK_SIZE],
  data: &mut [u8],
  mut acc: u128,
  h_polyval: u128,
  h_powers_rev: &[u128; 4],
) -> u128 {
  use core::arch::x86_64::*;

  // SAFETY: fused x86 AES-NI/PCLMUL sealing because:
  // 1. This function's caller guarantees AES-NI, PCLMULQDQ, SSE2, and SSSE3.
  // 2. `data` is a valid mutable byte slice; pointer arithmetic is bounded by checked loop
  //    conditions.
  // 3. GHASH folds ciphertext registers after encryption, matching GCM authentication semantics.
  unsafe {
    let ni_rk = match &ek.inner {
      Key128Inner::X86AesNi(rk) => rk,
      _ => {
        aes128_ctr32_encrypt_be(ek, initial_counter, data);
        return ghash_ciphertext_fallback(acc, h_polyval, data);
      }
    };

    let iv_prefix: [u8; 12] = {
      let mut buf = [0u8; 12];
      buf.copy_from_slice(&initial_counter[..12]);
      buf
    };
    let iv_words = [
      u32::from_le_bytes([iv_prefix[0], iv_prefix[1], iv_prefix[2], iv_prefix[3]]),
      u32::from_le_bytes([iv_prefix[4], iv_prefix[5], iv_prefix[6], iv_prefix[7]]),
      u32::from_le_bytes([iv_prefix[8], iv_prefix[9], iv_prefix[10], iv_prefix[11]]),
    ];
    let mut ctr = u32::from_be_bytes([
      initial_counter[12],
      initial_counter[13],
      initial_counter[14],
      initial_counter[15],
    ]);
    let mut offset = 0usize;

    while offset.strict_add(64) <= data.len() {
      let ctr0 = x86_gcm_ctr_block_be(iv_words, ctr);
      let ctr1 = x86_gcm_ctr_block_be(iv_words, ctr.wrapping_add(1));
      let ctr2 = x86_gcm_ctr_block_be(iv_words, ctr.wrapping_add(2));
      let ctr3 = x86_gcm_ctr_block_be(iv_words, ctr.wrapping_add(3));
      let (ks0, ks1, ks2, ks3) = ni::encrypt_4blocks_128_aesni(ni_rk, ctr0, ctr1, ctr2, ctr3);

      let ptr = data.as_mut_ptr().add(offset);
      let p0 = _mm_loadu_si128(ptr.cast());
      let p1 = _mm_loadu_si128(ptr.add(16).cast());
      let p2 = _mm_loadu_si128(ptr.add(32).cast());
      let p3 = _mm_loadu_si128(ptr.add(48).cast());
      let c0 = _mm_xor_si128(p0, ks0);
      let c1 = _mm_xor_si128(p1, ks1);
      let c2 = _mm_xor_si128(p2, ks2);
      let c3 = _mm_xor_si128(p3, ks3);
      _mm_storeu_si128(ptr.cast(), c0);
      _mm_storeu_si128(ptr.add(16).cast(), c1);
      _mm_storeu_si128(ptr.add(32).cast(), c2);
      _mm_storeu_si128(ptr.add(48).cast(), c3);
      acc = super::polyval::x86_pclmul_aggregate_4blocks_be_xmm_inline(acc, h_powers_rev, c0, c1, c2, c3);

      ctr = ctr.wrapping_add(4);
      offset = offset.strict_add(64);
    }

    while offset < data.len() {
      let mut counter_block = [0u8; BLOCK_SIZE];
      counter_block[..12].copy_from_slice(&iv_prefix);
      counter_block[12..16].copy_from_slice(&ctr.to_be_bytes());

      let mut keystream = counter_block;
      ni::encrypt_block_128(ni_rk, &mut keystream);

      let remaining = data.len().strict_sub(offset);
      if remaining >= BLOCK_SIZE {
        let ks = u128::from_ne_bytes(keystream);
        let mut d = [0u8; BLOCK_SIZE];
        d.copy_from_slice(&data[offset..offset.strict_add(BLOCK_SIZE)]);
        let xored = u128::from_ne_bytes(d) ^ ks;
        let ciphertext = xored.to_ne_bytes();
        data[offset..offset.strict_add(BLOCK_SIZE)].copy_from_slice(&ciphertext);
        acc ^= u128::from_be_bytes(ciphertext);
        acc = super::polyval::x86_clmul128_reduce_inline(acc, h_polyval);
        offset = offset.strict_add(BLOCK_SIZE);
      } else {
        let mut block = [0u8; BLOCK_SIZE];
        let mut i = 0usize;
        while i < remaining {
          data[offset.strict_add(i)] ^= keystream[i];
          block[i] = data[offset.strict_add(i)];
          i = i.strict_add(1);
        }
        acc ^= u128::from_be_bytes(block);
        acc = super::polyval::x86_clmul128_reduce_inline(acc, h_polyval);
        offset = offset.strict_add(remaining);
      }
      ctr = ctr.wrapping_add(1);
    }

    acc
  }
}

/// AES-128 CTR decryption fused with PCLMUL GHASH accumulation for GCM open.
///
/// # Safety
/// Caller must ensure AES-NI, PCLMULQDQ, SSE2, and SSSE3 are available.
#[cfg(all(target_arch = "x86_64", feature = "aes-gcm"))]
#[target_feature(enable = "aes,pclmulqdq,sse2,ssse3")]
pub(crate) unsafe fn aes128_ctr32_decrypt_be_aesni_pclmul_ghash(
  ek: &Aes128EncKey,
  initial_counter: &[u8; BLOCK_SIZE],
  data: &mut [u8],
  mut acc: u128,
  h_polyval: u128,
  h_powers_rev: &[u128; 4],
) -> u128 {
  use core::arch::x86_64::*;

  // SAFETY: fused x86 AES-NI/PCLMUL opening because:
  // 1. This function's caller guarantees AES-NI, PCLMULQDQ, SSE2, and SSSE3.
  // 2. Ciphertext registers are folded into GHASH before plaintext is stored back.
  // 3. All pointer arithmetic stays inside checked chunk bounds.
  unsafe {
    let ni_rk = match &ek.inner {
      Key128Inner::X86AesNi(rk) => rk,
      _ => {
        acc = ghash_ciphertext_fallback(acc, h_polyval, data);
        aes128_ctr32_encrypt_be(ek, initial_counter, data);
        return acc;
      }
    };

    let iv_prefix: [u8; 12] = {
      let mut buf = [0u8; 12];
      buf.copy_from_slice(&initial_counter[..12]);
      buf
    };
    let iv_words = [
      u32::from_le_bytes([iv_prefix[0], iv_prefix[1], iv_prefix[2], iv_prefix[3]]),
      u32::from_le_bytes([iv_prefix[4], iv_prefix[5], iv_prefix[6], iv_prefix[7]]),
      u32::from_le_bytes([iv_prefix[8], iv_prefix[9], iv_prefix[10], iv_prefix[11]]),
    ];
    let mut ctr = u32::from_be_bytes([
      initial_counter[12],
      initial_counter[13],
      initial_counter[14],
      initial_counter[15],
    ]);
    let mut offset = 0usize;

    while offset.strict_add(64) <= data.len() {
      let ctr0 = x86_gcm_ctr_block_be(iv_words, ctr);
      let ctr1 = x86_gcm_ctr_block_be(iv_words, ctr.wrapping_add(1));
      let ctr2 = x86_gcm_ctr_block_be(iv_words, ctr.wrapping_add(2));
      let ctr3 = x86_gcm_ctr_block_be(iv_words, ctr.wrapping_add(3));
      let (ks0, ks1, ks2, ks3) = ni::encrypt_4blocks_128_aesni(ni_rk, ctr0, ctr1, ctr2, ctr3);

      let ptr = data.as_mut_ptr().add(offset);
      let c0 = _mm_loadu_si128(ptr.cast());
      let c1 = _mm_loadu_si128(ptr.add(16).cast());
      let c2 = _mm_loadu_si128(ptr.add(32).cast());
      let c3 = _mm_loadu_si128(ptr.add(48).cast());
      acc = super::polyval::x86_pclmul_aggregate_4blocks_be_xmm_inline(acc, h_powers_rev, c0, c1, c2, c3);
      _mm_storeu_si128(ptr.cast(), _mm_xor_si128(c0, ks0));
      _mm_storeu_si128(ptr.add(16).cast(), _mm_xor_si128(c1, ks1));
      _mm_storeu_si128(ptr.add(32).cast(), _mm_xor_si128(c2, ks2));
      _mm_storeu_si128(ptr.add(48).cast(), _mm_xor_si128(c3, ks3));

      ctr = ctr.wrapping_add(4);
      offset = offset.strict_add(64);
    }

    while offset < data.len() {
      let mut counter_block = [0u8; BLOCK_SIZE];
      counter_block[..12].copy_from_slice(&iv_prefix);
      counter_block[12..16].copy_from_slice(&ctr.to_be_bytes());

      let mut keystream = counter_block;
      ni::encrypt_block_128(ni_rk, &mut keystream);

      let remaining = data.len().strict_sub(offset);
      if remaining >= BLOCK_SIZE {
        let mut ciphertext = [0u8; BLOCK_SIZE];
        ciphertext.copy_from_slice(&data[offset..offset.strict_add(BLOCK_SIZE)]);
        acc ^= u128::from_be_bytes(ciphertext);
        acc = super::polyval::x86_clmul128_reduce_inline(acc, h_polyval);

        let plaintext = u128::from_ne_bytes(ciphertext) ^ u128::from_ne_bytes(keystream);
        data[offset..offset.strict_add(BLOCK_SIZE)].copy_from_slice(&plaintext.to_ne_bytes());
        offset = offset.strict_add(BLOCK_SIZE);
      } else {
        let mut block = [0u8; BLOCK_SIZE];
        block[..remaining].copy_from_slice(&data[offset..offset.strict_add(remaining)]);
        acc ^= u128::from_be_bytes(block);
        acc = super::polyval::x86_clmul128_reduce_inline(acc, h_polyval);

        let mut i = 0usize;
        while i < remaining {
          data[offset.strict_add(i)] ^= keystream[i];
          i = i.strict_add(1);
        }
        offset = offset.strict_add(remaining);
      }
      ctr = ctr.wrapping_add(1);
    }

    acc
  }
}

/// AES-128 CTR encryption fused with GHASH accumulation for GCM sealing.
///
/// Encrypts `data` in place and returns the GHASH accumulator after the
/// ciphertext has been folded in. The incoming `acc` is normally the GHASH
/// state after AAD processing.
///
/// # Safety
/// Caller must ensure AVX-512F + AVX-512VL + AVX-512BW + AVX-512DQ +
/// VAES + VPCLMULQDQ + PCLMULQDQ + AES + SSE2.
#[cfg(all(target_arch = "x86_64", feature = "aes-gcm"))]
#[target_feature(enable = "aes,sse2,avx512f,avx512vl,avx512bw,avx512dq,vaes,vpclmulqdq,pclmulqdq")]
pub(crate) unsafe fn aes128_ctr32_encrypt_be_wide_ghash(
  ek: &Aes128EncKey,
  initial_counter: &[u8; BLOCK_SIZE],
  data: &mut [u8],
  mut acc: u128,
  h_polyval: u128,
  h_powers_rev: &[u128; 4],
  h_powers_rev_16: &[u128; 16],
) -> u128 {
  use core::arch::x86_64::*;

  // SAFETY: fused x86 AES-GCM sealing because:
  // 1. This function's caller guarantees all required x86 target features.
  // 2. `data` is a valid mutable byte slice; all pointer arithmetic stays inside checked chunk
  //    bounds.
  unsafe {
    let ni_rk = match &ek.inner {
      Key128Inner::X86AesNi(rk) => rk,
      _ => {
        aes128_ctr32_encrypt_be(ek, initial_counter, data);
        return ghash_ciphertext_fallback(acc, h_polyval, data);
      }
    };

    let iv_prefix: [u8; 12] = {
      let mut buf = [0u8; 12];
      buf.copy_from_slice(&initial_counter[..12]);
      buf
    };
    let iv_words = [
      u32::from_le_bytes([iv_prefix[0], iv_prefix[1], iv_prefix[2], iv_prefix[3]]),
      u32::from_le_bytes([iv_prefix[4], iv_prefix[5], iv_prefix[6], iv_prefix[7]]),
      u32::from_le_bytes([iv_prefix[8], iv_prefix[9], iv_prefix[10], iv_prefix[11]]),
    ];
    let mut ctr = u32::from_be_bytes([
      initial_counter[12],
      initial_counter[13],
      initial_counter[14],
      initial_counter[15],
    ]);
    let mut offset = 0usize;

    #[cfg(target_os = "linux")]
    if data.len() >= 256 {
      let mut state = x86_64_asm::AesGcmX86State::new(acc, ctr);
      // SAFETY: external x86-64 VAES-512 AES-128-GCM seal kernel because:
      // 1. This target-feature function is only entered after VAES, VPCLMULQDQ, AVX-512, AES-NI, and SSE2
      //    were selected by runtime/backend dispatch.
      // 2. `ni_rk.as_ptr()` addresses 11 initialized 128-bit AES-128 round keys.
      // 3. `initial_counter` points to the full 16-byte GCM counter block, and `data` is valid for
      //    `data.len()` mutable bytes.
      // 4. `h_powers_rev_16` contains exactly the [H^16..H] powers required by the 16-block fold.
      // 5. The kernel only processes complete 256-byte chunks and reports the processed byte count so the
      //    Rust fallback below handles every remaining full/partial tail.
      x86_64_asm::rscrypto_aes128_gcm_seal_16x_vaes512_x86_64_linux(
        ni_rk.as_ptr(),
        initial_counter.as_ptr(),
        data.as_mut_ptr(),
        data.len(),
        h_powers_rev_16.as_ptr(),
        &mut state,
      );
      acc = state.acc();
      ctr = state.ctr;
      offset = state.processed;
    }

    while offset.strict_add(256) <= data.len() {
      let (ctr0, ctr1, ctr2, ctr3) = x86_gcm_ctr_blocks_be_16(iv_words, ctr);
      let (ks0, ks1, ks2, ks3) = ni::encrypt_16blocks_128(ni_rk, ctr0, ctr1, ctr2, ctr3);

      let p0 = _mm512_loadu_si512(data.as_ptr().add(offset).cast());
      let c0 = _mm512_xor_si512(p0, ks0);
      _mm512_storeu_si512(data.as_mut_ptr().add(offset).cast(), c0);

      let p1 = _mm512_loadu_si512(data.as_ptr().add(offset.strict_add(64)).cast());
      let c1 = _mm512_xor_si512(p1, ks1);
      _mm512_storeu_si512(data.as_mut_ptr().add(offset.strict_add(64)).cast(), c1);

      let p2 = _mm512_loadu_si512(data.as_ptr().add(offset.strict_add(128)).cast());
      let c2 = _mm512_xor_si512(p2, ks2);
      _mm512_storeu_si512(data.as_mut_ptr().add(offset.strict_add(128)).cast(), c2);

      let p3 = _mm512_loadu_si512(data.as_ptr().add(offset.strict_add(192)).cast());
      let c3 = _mm512_xor_si512(p3, ks3);
      _mm512_storeu_si512(data.as_mut_ptr().add(offset.strict_add(192)).cast(), c3);
      acc = super::polyval::x86_aggregate_16blocks_be_lanes_inline(acc, h_powers_rev_16, c0, c1, c2, c3);

      ctr = ctr.wrapping_add(16);
      offset = offset.strict_add(256);
    }

    while offset.strict_add(64) <= data.len() {
      let ctr_vec = x86_gcm_ctr_blocks_be_4(iv_words, ctr);
      let keystream = ni::encrypt_4blocks_128(ni_rk, ctr_vec);
      let plaintext = _mm512_loadu_si512(data.as_ptr().add(offset).cast());
      let ciphertext = _mm512_xor_si512(plaintext, keystream);
      _mm512_storeu_si512(data.as_mut_ptr().add(offset).cast(), ciphertext);
      acc = super::polyval::x86_aggregate_4blocks_be_lanes_inline(acc, h_powers_rev, ciphertext);

      ctr = ctr.wrapping_add(4);
      offset = offset.strict_add(64);
    }

    while offset < data.len() {
      let mut counter_block = [0u8; 16];
      counter_block[..12].copy_from_slice(&iv_prefix);
      counter_block[12..16].copy_from_slice(&ctr.to_be_bytes());

      let mut keystream = counter_block;
      ni::encrypt_block_128(ni_rk, &mut keystream);

      let remaining = data.len().strict_sub(offset);
      if remaining >= BLOCK_SIZE {
        let ks = u128::from_ne_bytes(keystream);
        let mut d = [0u8; BLOCK_SIZE];
        d.copy_from_slice(&data[offset..offset.strict_add(BLOCK_SIZE)]);
        let xored = u128::from_ne_bytes(d) ^ ks;
        let ciphertext = xored.to_ne_bytes();
        data[offset..offset.strict_add(BLOCK_SIZE)].copy_from_slice(&ciphertext);
        acc ^= u128::from_be_bytes(ciphertext);
        acc = super::polyval::x86_clmul128_reduce_inline(acc, h_polyval);
        offset = offset.strict_add(BLOCK_SIZE);
      } else {
        let mut block = [0u8; BLOCK_SIZE];
        let mut i = 0usize;
        while i < remaining {
          data[offset.strict_add(i)] ^= keystream[i];
          block[i] = data[offset.strict_add(i)];
          i = i.strict_add(1);
        }
        acc ^= u128::from_be_bytes(block);
        acc = super::polyval::x86_clmul128_reduce_inline(acc, h_polyval);
        offset = offset.strict_add(remaining);
      }
      ctr = ctr.wrapping_add(1);
    }

    acc
  }
}

/// AES-128 CTR decryption fused with GHASH accumulation for GCM open.
///
/// Folds ciphertext into GHASH, decrypts the same chunk in place, and returns
/// the accumulator. Authentication is finalized by the caller.
///
/// # Safety
/// Caller must ensure AVX-512F + AVX-512VL + AVX-512BW + AVX-512DQ +
/// VAES + VPCLMULQDQ + PCLMULQDQ + AES + SSE2.
#[cfg(all(target_arch = "x86_64", feature = "aes-gcm"))]
#[target_feature(enable = "aes,sse2,avx512f,avx512vl,avx512bw,avx512dq,vaes,vpclmulqdq,pclmulqdq")]
pub(crate) unsafe fn aes128_ctr32_decrypt_be_wide_ghash(
  ek: &Aes128EncKey,
  initial_counter: &[u8; BLOCK_SIZE],
  data: &mut [u8],
  mut acc: u128,
  h_polyval: u128,
  h_powers_rev: &[u128; 4],
  h_powers_rev_16: &[u128; 16],
) -> u128 {
  use core::arch::x86_64::*;

  // SAFETY: fused x86 AES-GCM opening because:
  // 1. This function's caller guarantees all required x86 target features.
  // 2. `data` is a valid mutable ciphertext slice. Each vector is folded into GHASH before the
  //    corresponding plaintext is stored back.
  // 3. All pointer arithmetic stays inside checked chunk bounds.
  unsafe {
    let ni_rk = match &ek.inner {
      Key128Inner::X86AesNi(rk) => rk,
      _ => {
        acc = ghash_ciphertext_fallback(acc, h_polyval, data);
        aes128_ctr32_encrypt_be(ek, initial_counter, data);
        return acc;
      }
    };

    let iv_prefix: [u8; 12] = {
      let mut buf = [0u8; 12];
      buf.copy_from_slice(&initial_counter[..12]);
      buf
    };
    let iv_words = [
      u32::from_le_bytes([iv_prefix[0], iv_prefix[1], iv_prefix[2], iv_prefix[3]]),
      u32::from_le_bytes([iv_prefix[4], iv_prefix[5], iv_prefix[6], iv_prefix[7]]),
      u32::from_le_bytes([iv_prefix[8], iv_prefix[9], iv_prefix[10], iv_prefix[11]]),
    ];
    let mut ctr = u32::from_be_bytes([
      initial_counter[12],
      initial_counter[13],
      initial_counter[14],
      initial_counter[15],
    ]);
    let mut offset = 0usize;

    #[cfg(target_os = "linux")]
    if data.len() >= 256 {
      let mut state = x86_64_asm::AesGcmX86State::new(acc, ctr);
      // SAFETY: external x86-64 VAES-512 AES-128-GCM open kernel because:
      // 1. This target-feature function is only entered after VAES, VPCLMULQDQ, AVX-512, AES-NI, and SSE2
      //    were selected by runtime/backend dispatch.
      // 2. `ni_rk.as_ptr()` addresses 11 initialized 128-bit AES-128 round keys.
      // 3. `initial_counter` points to the full 16-byte GCM counter block, and `data` is valid for
      //    `data.len()` mutable bytes of ciphertext.
      // 4. The kernel folds ciphertext into GHASH before storing plaintext.
      // 5. The kernel only processes complete 256-byte chunks and reports the processed byte count so the
      //    Rust fallback below handles every remaining full/partial tail.
      x86_64_asm::rscrypto_aes128_gcm_open_16x_vaes512_x86_64_linux(
        ni_rk.as_ptr(),
        initial_counter.as_ptr(),
        data.as_mut_ptr(),
        data.len(),
        h_powers_rev_16.as_ptr(),
        &mut state,
      );
      acc = state.acc();
      ctr = state.ctr;
      offset = state.processed;
    }

    while offset.strict_add(256) <= data.len() {
      let (ctr0, ctr1, ctr2, ctr3) = x86_gcm_ctr_blocks_be_16(iv_words, ctr);
      let (ks0, ks1, ks2, ks3) = ni::encrypt_16blocks_128(ni_rk, ctr0, ctr1, ctr2, ctr3);

      let c0 = _mm512_loadu_si512(data.as_ptr().add(offset).cast());
      let c1 = _mm512_loadu_si512(data.as_ptr().add(offset.strict_add(64)).cast());
      let c2 = _mm512_loadu_si512(data.as_ptr().add(offset.strict_add(128)).cast());
      let c3 = _mm512_loadu_si512(data.as_ptr().add(offset.strict_add(192)).cast());
      acc = super::polyval::x86_aggregate_16blocks_be_lanes_inline(acc, h_powers_rev_16, c0, c1, c2, c3);
      _mm512_storeu_si512(data.as_mut_ptr().add(offset).cast(), _mm512_xor_si512(c0, ks0));
      _mm512_storeu_si512(
        data.as_mut_ptr().add(offset.strict_add(64)).cast(),
        _mm512_xor_si512(c1, ks1),
      );
      _mm512_storeu_si512(
        data.as_mut_ptr().add(offset.strict_add(128)).cast(),
        _mm512_xor_si512(c2, ks2),
      );
      _mm512_storeu_si512(
        data.as_mut_ptr().add(offset.strict_add(192)).cast(),
        _mm512_xor_si512(c3, ks3),
      );

      ctr = ctr.wrapping_add(16);
      offset = offset.strict_add(256);
    }

    while offset.strict_add(64) <= data.len() {
      let ctr_vec = x86_gcm_ctr_blocks_be_4(iv_words, ctr);
      let keystream = ni::encrypt_4blocks_128(ni_rk, ctr_vec);
      let ciphertext = _mm512_loadu_si512(data.as_ptr().add(offset).cast());
      acc = super::polyval::x86_aggregate_4blocks_be_lanes_inline(acc, h_powers_rev, ciphertext);
      _mm512_storeu_si512(
        data.as_mut_ptr().add(offset).cast(),
        _mm512_xor_si512(ciphertext, keystream),
      );

      ctr = ctr.wrapping_add(4);
      offset = offset.strict_add(64);
    }

    while offset < data.len() {
      let mut counter_block = [0u8; 16];
      counter_block[..12].copy_from_slice(&iv_prefix);
      counter_block[12..16].copy_from_slice(&ctr.to_be_bytes());

      let mut keystream = counter_block;
      ni::encrypt_block_128(ni_rk, &mut keystream);

      let remaining = data.len().strict_sub(offset);
      if remaining >= BLOCK_SIZE {
        let mut ciphertext = [0u8; BLOCK_SIZE];
        ciphertext.copy_from_slice(&data[offset..offset.strict_add(BLOCK_SIZE)]);
        acc ^= u128::from_be_bytes(ciphertext);
        acc = super::polyval::x86_clmul128_reduce_inline(acc, h_polyval);

        let plaintext = u128::from_ne_bytes(ciphertext) ^ u128::from_ne_bytes(keystream);
        data[offset..offset.strict_add(BLOCK_SIZE)].copy_from_slice(&plaintext.to_ne_bytes());
        offset = offset.strict_add(BLOCK_SIZE);
      } else {
        let mut block = [0u8; BLOCK_SIZE];
        block[..remaining].copy_from_slice(&data[offset..offset.strict_add(remaining)]);
        acc ^= u128::from_be_bytes(block);
        acc = super::polyval::x86_clmul128_reduce_inline(acc, h_polyval);

        let mut i = 0usize;
        while i < remaining {
          data[offset.strict_add(i)] ^= keystream[i];
          i = i.strict_add(1);
        }
        offset = offset.strict_add(remaining);
      }
      ctr = ctr.wrapping_add(1);
    }

    acc
  }
}

/// AES-128 CTR encryption fused with 256-bit VAES/VPCLMUL GHASH accumulation.
///
/// # Safety
/// Caller must ensure AVX2 + AVX-512F + AVX-512VL + VAES + VPCLMULQDQ +
/// PCLMULQDQ + AES + SSE2 + SSSE3.
#[cfg(all(target_arch = "x86_64", feature = "aes-gcm"))]
#[target_feature(enable = "aes,sse2,ssse3,avx2,avx512f,avx512vl,vaes,vpclmulqdq,pclmulqdq")]
pub(crate) unsafe fn aes128_ctr32_encrypt_be_y256_ghash(
  ek: &Aes128EncKey,
  initial_counter: &[u8; BLOCK_SIZE],
  data: &mut [u8],
  mut acc: u128,
  h_polyval: u128,
  h_powers_rev: &[u128; 4],
  h_powers_rev_8: &[u128; 8],
) -> u128 {
  use core::arch::x86_64::*;

  // SAFETY: fused x86 VAES-128 AES-GCM sealing because:
  // 1. This function's caller guarantees all required x86 target features.
  // 2. `data` is a valid mutable byte slice; all pointer arithmetic stays inside checked chunk
  //    bounds.
  // 3. GHASH folds ciphertext registers after encryption, matching GCM authentication semantics.
  unsafe {
    let ni_rk = match &ek.inner {
      Key128Inner::X86AesNi(rk) => rk,
      _ => {
        aes128_ctr32_encrypt_be(ek, initial_counter, data);
        return ghash_ciphertext_fallback(acc, h_polyval, data);
      }
    };

    let iv_prefix: [u8; 12] = {
      let mut buf = [0u8; 12];
      buf.copy_from_slice(&initial_counter[..12]);
      buf
    };
    let iv_words = [
      u32::from_le_bytes([iv_prefix[0], iv_prefix[1], iv_prefix[2], iv_prefix[3]]),
      u32::from_le_bytes([iv_prefix[4], iv_prefix[5], iv_prefix[6], iv_prefix[7]]),
      u32::from_le_bytes([iv_prefix[8], iv_prefix[9], iv_prefix[10], iv_prefix[11]]),
    ];
    let mut ctr = u32::from_be_bytes([
      initial_counter[12],
      initial_counter[13],
      initial_counter[14],
      initial_counter[15],
    ]);
    let mut offset = 0usize;

    while offset.strict_add(128) <= data.len() {
      let (ctr0, ctr1, ctr2, ctr3) = x86_gcm_ctr_blocks_be_8_y256(iv_words, ctr);
      let (ks0, ks1, ks2, ks3) = ni::encrypt_8blocks_128_y256(ni_rk, ctr0, ctr1, ctr2, ctr3);

      let p0 = _mm256_loadu_si256(data.as_ptr().add(offset).cast());
      let c0 = _mm256_xor_si256(p0, ks0);
      _mm256_storeu_si256(data.as_mut_ptr().add(offset).cast(), c0);

      let p1 = _mm256_loadu_si256(data.as_ptr().add(offset.strict_add(32)).cast());
      let c1 = _mm256_xor_si256(p1, ks1);
      _mm256_storeu_si256(data.as_mut_ptr().add(offset.strict_add(32)).cast(), c1);

      let p2 = _mm256_loadu_si256(data.as_ptr().add(offset.strict_add(64)).cast());
      let c2 = _mm256_xor_si256(p2, ks2);
      _mm256_storeu_si256(data.as_mut_ptr().add(offset.strict_add(64)).cast(), c2);

      let p3 = _mm256_loadu_si256(data.as_ptr().add(offset.strict_add(96)).cast());
      let c3 = _mm256_xor_si256(p3, ks3);
      _mm256_storeu_si256(data.as_mut_ptr().add(offset.strict_add(96)).cast(), c3);
      acc = super::polyval::x86_aggregate_8blocks_be_lanes_256_inline(acc, h_powers_rev_8, c0, c1, c2, c3);

      ctr = ctr.wrapping_add(8);
      offset = offset.strict_add(128);
    }

    while offset.strict_add(64) <= data.len() {
      let ctr0 = x86_gcm_ctr_block_be(iv_words, ctr);
      let ctr1 = x86_gcm_ctr_block_be(iv_words, ctr.wrapping_add(1));
      let ctr2 = x86_gcm_ctr_block_be(iv_words, ctr.wrapping_add(2));
      let ctr3 = x86_gcm_ctr_block_be(iv_words, ctr.wrapping_add(3));
      let (ks0, ks1, ks2, ks3) = ni::encrypt_4blocks_128_aesni(ni_rk, ctr0, ctr1, ctr2, ctr3);

      let ptr = data.as_mut_ptr().add(offset);
      let p0 = _mm_loadu_si128(ptr.cast());
      let p1 = _mm_loadu_si128(ptr.add(16).cast());
      let p2 = _mm_loadu_si128(ptr.add(32).cast());
      let p3 = _mm_loadu_si128(ptr.add(48).cast());
      let c0 = _mm_xor_si128(p0, ks0);
      let c1 = _mm_xor_si128(p1, ks1);
      let c2 = _mm_xor_si128(p2, ks2);
      let c3 = _mm_xor_si128(p3, ks3);
      _mm_storeu_si128(ptr.cast(), c0);
      _mm_storeu_si128(ptr.add(16).cast(), c1);
      _mm_storeu_si128(ptr.add(32).cast(), c2);
      _mm_storeu_si128(ptr.add(48).cast(), c3);
      acc = super::polyval::x86_pclmul_aggregate_4blocks_be_xmm_inline(acc, h_powers_rev, c0, c1, c2, c3);

      ctr = ctr.wrapping_add(4);
      offset = offset.strict_add(64);
    }

    while offset < data.len() {
      let mut counter_block = [0u8; 16];
      counter_block[..12].copy_from_slice(&iv_prefix);
      counter_block[12..16].copy_from_slice(&ctr.to_be_bytes());

      let mut keystream = counter_block;
      ni::encrypt_block_128(ni_rk, &mut keystream);

      let remaining = data.len().strict_sub(offset);
      if remaining >= BLOCK_SIZE {
        let ks = u128::from_ne_bytes(keystream);
        let mut d = [0u8; BLOCK_SIZE];
        d.copy_from_slice(&data[offset..offset.strict_add(BLOCK_SIZE)]);
        let xored = u128::from_ne_bytes(d) ^ ks;
        let ciphertext = xored.to_ne_bytes();
        data[offset..offset.strict_add(BLOCK_SIZE)].copy_from_slice(&ciphertext);
        acc ^= u128::from_be_bytes(ciphertext);
        acc = super::polyval::x86_clmul128_reduce_inline(acc, h_polyval);
        offset = offset.strict_add(BLOCK_SIZE);
      } else {
        let mut block = [0u8; BLOCK_SIZE];
        let mut i = 0usize;
        while i < remaining {
          data[offset.strict_add(i)] ^= keystream[i];
          block[i] = data[offset.strict_add(i)];
          i = i.strict_add(1);
        }
        acc ^= u128::from_be_bytes(block);
        acc = super::polyval::x86_clmul128_reduce_inline(acc, h_polyval);
        offset = offset.strict_add(remaining);
      }
      ctr = ctr.wrapping_add(1);
    }

    acc
  }
}

/// AES-128 CTR decryption fused with 256-bit VAES/VPCLMUL GHASH accumulation.
///
/// # Safety
/// Caller must ensure AVX2 + AVX-512F + AVX-512VL + VAES + VPCLMULQDQ +
/// PCLMULQDQ + AES + SSE2 + SSSE3.
#[cfg(all(target_arch = "x86_64", feature = "aes-gcm"))]
#[target_feature(enable = "aes,sse2,ssse3,avx2,avx512f,avx512vl,vaes,vpclmulqdq,pclmulqdq")]
pub(crate) unsafe fn aes128_ctr32_decrypt_be_y256_ghash(
  ek: &Aes128EncKey,
  initial_counter: &[u8; BLOCK_SIZE],
  data: &mut [u8],
  mut acc: u128,
  h_polyval: u128,
  h_powers_rev: &[u128; 4],
  h_powers_rev_8: &[u128; 8],
) -> u128 {
  use core::arch::x86_64::*;

  // SAFETY: fused x86 VAES-128 AES-GCM opening because:
  // 1. This function's caller guarantees all required x86 target features.
  // 2. Ciphertext registers are folded into GHASH before plaintext is stored back.
  // 3. All pointer arithmetic stays inside checked chunk bounds.
  unsafe {
    let ni_rk = match &ek.inner {
      Key128Inner::X86AesNi(rk) => rk,
      _ => {
        acc = ghash_ciphertext_fallback(acc, h_polyval, data);
        aes128_ctr32_encrypt_be(ek, initial_counter, data);
        return acc;
      }
    };

    let iv_prefix: [u8; 12] = {
      let mut buf = [0u8; 12];
      buf.copy_from_slice(&initial_counter[..12]);
      buf
    };
    let iv_words = [
      u32::from_le_bytes([iv_prefix[0], iv_prefix[1], iv_prefix[2], iv_prefix[3]]),
      u32::from_le_bytes([iv_prefix[4], iv_prefix[5], iv_prefix[6], iv_prefix[7]]),
      u32::from_le_bytes([iv_prefix[8], iv_prefix[9], iv_prefix[10], iv_prefix[11]]),
    ];
    let mut ctr = u32::from_be_bytes([
      initial_counter[12],
      initial_counter[13],
      initial_counter[14],
      initial_counter[15],
    ]);
    let mut offset = 0usize;

    while offset.strict_add(128) <= data.len() {
      let (ctr0, ctr1, ctr2, ctr3) = x86_gcm_ctr_blocks_be_8_y256(iv_words, ctr);
      let (ks0, ks1, ks2, ks3) = ni::encrypt_8blocks_128_y256(ni_rk, ctr0, ctr1, ctr2, ctr3);

      let c0 = _mm256_loadu_si256(data.as_ptr().add(offset).cast());
      let c1 = _mm256_loadu_si256(data.as_ptr().add(offset.strict_add(32)).cast());
      let c2 = _mm256_loadu_si256(data.as_ptr().add(offset.strict_add(64)).cast());
      let c3 = _mm256_loadu_si256(data.as_ptr().add(offset.strict_add(96)).cast());
      acc = super::polyval::x86_aggregate_8blocks_be_lanes_256_inline(acc, h_powers_rev_8, c0, c1, c2, c3);

      _mm256_storeu_si256(data.as_mut_ptr().add(offset).cast(), _mm256_xor_si256(c0, ks0));
      _mm256_storeu_si256(
        data.as_mut_ptr().add(offset.strict_add(32)).cast(),
        _mm256_xor_si256(c1, ks1),
      );
      _mm256_storeu_si256(
        data.as_mut_ptr().add(offset.strict_add(64)).cast(),
        _mm256_xor_si256(c2, ks2),
      );
      _mm256_storeu_si256(
        data.as_mut_ptr().add(offset.strict_add(96)).cast(),
        _mm256_xor_si256(c3, ks3),
      );

      ctr = ctr.wrapping_add(8);
      offset = offset.strict_add(128);
    }

    while offset.strict_add(64) <= data.len() {
      let ctr0 = x86_gcm_ctr_block_be(iv_words, ctr);
      let ctr1 = x86_gcm_ctr_block_be(iv_words, ctr.wrapping_add(1));
      let ctr2 = x86_gcm_ctr_block_be(iv_words, ctr.wrapping_add(2));
      let ctr3 = x86_gcm_ctr_block_be(iv_words, ctr.wrapping_add(3));
      let (ks0, ks1, ks2, ks3) = ni::encrypt_4blocks_128_aesni(ni_rk, ctr0, ctr1, ctr2, ctr3);

      let ptr = data.as_mut_ptr().add(offset);
      let c0 = _mm_loadu_si128(ptr.cast());
      let c1 = _mm_loadu_si128(ptr.add(16).cast());
      let c2 = _mm_loadu_si128(ptr.add(32).cast());
      let c3 = _mm_loadu_si128(ptr.add(48).cast());
      acc = super::polyval::x86_pclmul_aggregate_4blocks_be_xmm_inline(acc, h_powers_rev, c0, c1, c2, c3);
      _mm_storeu_si128(ptr.cast(), _mm_xor_si128(c0, ks0));
      _mm_storeu_si128(ptr.add(16).cast(), _mm_xor_si128(c1, ks1));
      _mm_storeu_si128(ptr.add(32).cast(), _mm_xor_si128(c2, ks2));
      _mm_storeu_si128(ptr.add(48).cast(), _mm_xor_si128(c3, ks3));

      ctr = ctr.wrapping_add(4);
      offset = offset.strict_add(64);
    }

    while offset < data.len() {
      let mut counter_block = [0u8; 16];
      counter_block[..12].copy_from_slice(&iv_prefix);
      counter_block[12..16].copy_from_slice(&ctr.to_be_bytes());

      let mut keystream = counter_block;
      ni::encrypt_block_128(ni_rk, &mut keystream);

      let remaining = data.len().strict_sub(offset);
      if remaining >= BLOCK_SIZE {
        let mut ciphertext = [0u8; BLOCK_SIZE];
        ciphertext.copy_from_slice(&data[offset..offset.strict_add(BLOCK_SIZE)]);
        acc ^= u128::from_be_bytes(ciphertext);
        acc = super::polyval::x86_clmul128_reduce_inline(acc, h_polyval);

        let plaintext = u128::from_ne_bytes(ciphertext) ^ u128::from_ne_bytes(keystream);
        data[offset..offset.strict_add(BLOCK_SIZE)].copy_from_slice(&plaintext.to_ne_bytes());
        offset = offset.strict_add(BLOCK_SIZE);
      } else {
        let mut block = [0u8; BLOCK_SIZE];
        block[..remaining].copy_from_slice(&data[offset..offset.strict_add(remaining)]);
        acc ^= u128::from_be_bytes(block);
        acc = super::polyval::x86_clmul128_reduce_inline(acc, h_polyval);

        let mut i = 0usize;
        while i < remaining {
          data[offset.strict_add(i)] ^= keystream[i];
          i = i.strict_add(1);
        }
        offset = offset.strict_add(remaining);
      }
      ctr = ctr.wrapping_add(1);
    }

    acc
  }
}

/// AES-256 CTR encryption using VAES-512 for the bulk, AES-NI for the tail.
///
/// GCM-SIV variant: counter occupies bytes 0..3 (little-endian).
#[cfg(all(target_arch = "x86_64", feature = "aes-gcm-siv"))]
#[target_feature(enable = "aes,sse2,avx512f,avx512vl,vaes")]
pub(crate) unsafe fn aes256_ctr32_encrypt_wide(ek: &Aes256EncKey, initial_counter: &[u8; BLOCK_SIZE], data: &mut [u8]) {
  use core::arch::x86_64::*;

  // SAFETY: AesNi variant is guaranteed by caller; target_feature gate
  // ensures VAES + AVX-512 instructions are available.
  unsafe {
    let ni_rk = match &ek.inner {
      KeyInner::X86AesNi(rk) => rk,
      _ => {
        aes256_ctr32_encrypt(ek, initial_counter, data);
        return;
      }
    };

    let iv_suffix: [u8; 12] = {
      let mut buf = [0u8; 12];
      buf.copy_from_slice(&initial_counter[4..16]);
      buf
    };
    let suffix_words = [
      u32::from_le_bytes([iv_suffix[0], iv_suffix[1], iv_suffix[2], iv_suffix[3]]),
      u32::from_le_bytes([iv_suffix[4], iv_suffix[5], iv_suffix[6], iv_suffix[7]]),
      u32::from_le_bytes([iv_suffix[8], iv_suffix[9], iv_suffix[10], iv_suffix[11]]),
    ];
    let mut ctr = u32::from_le_bytes([
      initial_counter[0],
      initial_counter[1],
      initial_counter[2],
      initial_counter[3],
    ]);
    let mut offset = 0usize;

    while offset.strict_add(256) <= data.len() {
      let ctr0 = x86_gcmsiv_ctr_blocks_le_4(suffix_words, ctr);
      let ctr1 = x86_gcmsiv_ctr_blocks_le_4(suffix_words, ctr.wrapping_add(4));
      let ctr2 = x86_gcmsiv_ctr_blocks_le_4(suffix_words, ctr.wrapping_add(8));
      let ctr3 = x86_gcmsiv_ctr_blocks_le_4(suffix_words, ctr.wrapping_add(12));
      let (ks0, ks1, ks2, ks3) = ni::encrypt_16blocks(ni_rk, ctr0, ctr1, ctr2, ctr3);

      let p0 = _mm512_loadu_si512(data.as_ptr().add(offset).cast());
      _mm512_storeu_si512(data.as_mut_ptr().add(offset).cast(), _mm512_xor_si512(p0, ks0));

      let p1 = _mm512_loadu_si512(data.as_ptr().add(offset.strict_add(64)).cast());
      _mm512_storeu_si512(
        data.as_mut_ptr().add(offset.strict_add(64)).cast(),
        _mm512_xor_si512(p1, ks1),
      );

      let p2 = _mm512_loadu_si512(data.as_ptr().add(offset.strict_add(128)).cast());
      _mm512_storeu_si512(
        data.as_mut_ptr().add(offset.strict_add(128)).cast(),
        _mm512_xor_si512(p2, ks2),
      );

      let p3 = _mm512_loadu_si512(data.as_ptr().add(offset.strict_add(192)).cast());
      _mm512_storeu_si512(
        data.as_mut_ptr().add(offset.strict_add(192)).cast(),
        _mm512_xor_si512(p3, ks3),
      );

      ctr = ctr.wrapping_add(16);
      offset = offset.strict_add(256);
    }

    while offset.strict_add(64) <= data.len() {
      let ctr_vec = x86_gcmsiv_ctr_blocks_le_4(suffix_words, ctr);
      let keystream = ni::encrypt_4blocks(ni_rk, ctr_vec);
      let plaintext = _mm512_loadu_si512(data.as_ptr().add(offset).cast());
      let ciphertext = _mm512_xor_si512(plaintext, keystream);
      _mm512_storeu_si512(data.as_mut_ptr().add(offset).cast(), ciphertext);

      ctr = ctr.wrapping_add(4);
      offset = offset.strict_add(64);
    }

    // Tail: 0-3 remaining blocks via single-block AES-NI.
    while offset < data.len() {
      let mut counter_block = [0u8; 16];
      counter_block[0..4].copy_from_slice(&ctr.to_le_bytes());
      counter_block[4..16].copy_from_slice(&iv_suffix);

      let mut keystream = counter_block;
      ni::encrypt_block(ni_rk, &mut keystream);

      let remaining = data.len().strict_sub(offset);
      if remaining >= BLOCK_SIZE {
        let ks = u128::from_ne_bytes(keystream);
        let mut d = [0u8; BLOCK_SIZE];
        d.copy_from_slice(&data[offset..offset.strict_add(BLOCK_SIZE)]);
        let xored = u128::from_ne_bytes(d) ^ ks;
        data[offset..offset.strict_add(BLOCK_SIZE)].copy_from_slice(&xored.to_ne_bytes());
        offset = offset.strict_add(BLOCK_SIZE);
      } else {
        let mut i = 0usize;
        while i < remaining {
          data[offset.strict_add(i)] ^= keystream[i];
          i = i.strict_add(1);
        }
        offset = offset.strict_add(remaining);
      }
      ctr = ctr.wrapping_add(1);
    }
  }
}

/// AES-128 CTR encryption using VAES-512 for the bulk, AES-NI for the tail.
///
/// Mirrors [`aes256_ctr32_encrypt_wide`] but runs the AES-128 round
/// schedule (10 rounds). GCM-SIV variant: counter occupies bytes 0..3
/// (little-endian) per RFC 8452 §4.
///
/// # Safety
/// Caller must ensure AVX-512F + AVX-512VL + VAES + AES + SSE2.
#[cfg(all(target_arch = "x86_64", feature = "aes-gcm-siv"))]
#[target_feature(enable = "aes,sse2,avx512f,avx512vl,vaes")]
pub(crate) unsafe fn aes128_ctr32_encrypt_wide(ek: &Aes128EncKey, initial_counter: &[u8; BLOCK_SIZE], data: &mut [u8]) {
  use core::arch::x86_64::*;

  // SAFETY: X86AesNi variant guaranteed by caller; target_feature gate
  // ensures VAES + AVX-512 instructions are available.
  unsafe {
    let ni_rk = match &ek.inner {
      Key128Inner::X86AesNi(rk) => rk,
      _ => {
        aes128_ctr32_encrypt(ek, initial_counter, data);
        return;
      }
    };

    let iv_suffix: [u8; 12] = {
      let mut buf = [0u8; 12];
      buf.copy_from_slice(&initial_counter[4..16]);
      buf
    };
    let suffix_words = [
      u32::from_le_bytes([iv_suffix[0], iv_suffix[1], iv_suffix[2], iv_suffix[3]]),
      u32::from_le_bytes([iv_suffix[4], iv_suffix[5], iv_suffix[6], iv_suffix[7]]),
      u32::from_le_bytes([iv_suffix[8], iv_suffix[9], iv_suffix[10], iv_suffix[11]]),
    ];
    let mut ctr = u32::from_le_bytes([
      initial_counter[0],
      initial_counter[1],
      initial_counter[2],
      initial_counter[3],
    ]);
    let mut offset = 0usize;

    while offset.strict_add(256) <= data.len() {
      let ctr0 = x86_gcmsiv_ctr_blocks_le_4(suffix_words, ctr);
      let ctr1 = x86_gcmsiv_ctr_blocks_le_4(suffix_words, ctr.wrapping_add(4));
      let ctr2 = x86_gcmsiv_ctr_blocks_le_4(suffix_words, ctr.wrapping_add(8));
      let ctr3 = x86_gcmsiv_ctr_blocks_le_4(suffix_words, ctr.wrapping_add(12));
      let (ks0, ks1, ks2, ks3) = ni::encrypt_16blocks_128(ni_rk, ctr0, ctr1, ctr2, ctr3);

      let p0 = _mm512_loadu_si512(data.as_ptr().add(offset).cast());
      _mm512_storeu_si512(data.as_mut_ptr().add(offset).cast(), _mm512_xor_si512(p0, ks0));

      let p1 = _mm512_loadu_si512(data.as_ptr().add(offset.strict_add(64)).cast());
      _mm512_storeu_si512(
        data.as_mut_ptr().add(offset.strict_add(64)).cast(),
        _mm512_xor_si512(p1, ks1),
      );

      let p2 = _mm512_loadu_si512(data.as_ptr().add(offset.strict_add(128)).cast());
      _mm512_storeu_si512(
        data.as_mut_ptr().add(offset.strict_add(128)).cast(),
        _mm512_xor_si512(p2, ks2),
      );

      let p3 = _mm512_loadu_si512(data.as_ptr().add(offset.strict_add(192)).cast());
      _mm512_storeu_si512(
        data.as_mut_ptr().add(offset.strict_add(192)).cast(),
        _mm512_xor_si512(p3, ks3),
      );

      ctr = ctr.wrapping_add(16);
      offset = offset.strict_add(256);
    }

    while offset.strict_add(64) <= data.len() {
      let ctr_vec = x86_gcmsiv_ctr_blocks_le_4(suffix_words, ctr);
      let keystream = ni::encrypt_4blocks_128(ni_rk, ctr_vec);
      let plaintext = _mm512_loadu_si512(data.as_ptr().add(offset).cast());
      let ciphertext = _mm512_xor_si512(plaintext, keystream);
      _mm512_storeu_si512(data.as_mut_ptr().add(offset).cast(), ciphertext);

      ctr = ctr.wrapping_add(4);
      offset = offset.strict_add(64);
    }

    while offset < data.len() {
      let mut counter_block = [0u8; 16];
      counter_block[0..4].copy_from_slice(&ctr.to_le_bytes());
      counter_block[4..16].copy_from_slice(&iv_suffix);

      let mut keystream = counter_block;
      ni::encrypt_block_128(ni_rk, &mut keystream);

      let remaining = data.len().strict_sub(offset);
      if remaining >= BLOCK_SIZE {
        let ks = u128::from_ne_bytes(keystream);
        let mut d = [0u8; BLOCK_SIZE];
        d.copy_from_slice(&data[offset..offset.strict_add(BLOCK_SIZE)]);
        let xored = u128::from_ne_bytes(d) ^ ks;
        data[offset..offset.strict_add(BLOCK_SIZE)].copy_from_slice(&xored.to_ne_bytes());
        offset = offset.strict_add(BLOCK_SIZE);
      } else {
        let mut i = 0usize;
        while i < remaining {
          data[offset.strict_add(i)] ^= keystream[i];
          i = i.strict_add(1);
        }
        offset = offset.strict_add(remaining);
      }
      ctr = ctr.wrapping_add(1);
    }
  }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
  use super::*;

  /// Verify the S-box against the canonical AES S-box table.
  #[test]
  fn sbox_matches_canonical() {
    #[rustfmt::skip]
    const CANONICAL: [u8; 256] = [
      0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
      0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
      0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
      0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
      0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
      0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
      0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
      0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
      0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
      0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
      0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
      0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
      0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
      0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
      0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
      0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16,
    ];

    for (i, &expected) in CANONICAL.iter().enumerate() {
      assert_eq!(
        sbox(i as u8),
        expected,
        "S-box mismatch at index {i:#04x}: got {:#04x}, expected {expected:#04x}",
        sbox(i as u8),
      );
    }
  }

  /// NIST FIPS 197 Appendix C.1: AES-128 encryption test vector.
  ///
  /// Key:  000102030405060708090a0b0c0d0e0f
  /// PT:   00112233445566778899aabbccddeeff
  /// CT:   69c4e0d86a7b0430d8cdb78070b4c55a
  #[test]
  fn aes128_encrypt_nist_appendix_c1() {
    let key: [u8; 16] = [
      0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
    ];
    let plaintext: [u8; 16] = [
      0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff,
    ];
    let expected: [u8; 16] = [
      0x69, 0xc4, 0xe0, 0xd8, 0x6a, 0x7b, 0x04, 0x30, 0xd8, 0xcd, 0xb7, 0x80, 0x70, 0xb4, 0xc5, 0x5a,
    ];

    let ek = aes128_expand_key(&key);
    let mut block = plaintext;
    aes128_encrypt_block(&ek, &mut block);
    assert_eq!(block, expected);
  }

  /// AES-128 with all-zero key and plaintext.
  #[test]
  fn aes128_encrypt_zero_key_zero_plaintext() {
    let key = [0u8; 16];
    let plaintext = [0u8; 16];
    // Known answer for AES-128(zero_key, zero_block).
    let expected: [u8; 16] = [
      0x66, 0xe9, 0x4b, 0xd4, 0xef, 0x8a, 0x2c, 0x3b, 0x88, 0x4c, 0xfa, 0x59, 0xca, 0x34, 0x2b, 0x2e,
    ];

    let ek = aes128_expand_key(&key);
    let mut block = plaintext;
    aes128_encrypt_block(&ek, &mut block);
    assert_eq!(block, expected);
  }

  /// NIST FIPS 197 Appendix C.3: AES-256 encryption test vector.
  #[test]
  fn aes256_encrypt_nist_appendix_c3() {
    let key: [u8; 32] = [
      0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11, 0x12,
      0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
    ];
    let plaintext: [u8; 16] = [
      0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff,
    ];
    let expected: [u8; 16] = [
      0x8e, 0xa2, 0xb7, 0xca, 0x51, 0x67, 0x45, 0xbf, 0xea, 0xfc, 0x49, 0x90, 0x4b, 0x49, 0x60, 0x89,
    ];

    let ek = aes256_expand_key(&key);
    let mut block = plaintext;
    aes256_encrypt_block(&ek, &mut block);
    assert_eq!(block, expected);
  }

  /// Additional AES-256 test: all-zero key and all-zero plaintext.
  #[test]
  fn aes256_encrypt_zero_key_zero_plaintext() {
    let key = [0u8; 32];
    let plaintext = [0u8; 16];
    // Known answer for AES-256(zero_key, zero_block).
    let expected: [u8; 16] = [
      0xdc, 0x95, 0xc0, 0x78, 0xa2, 0x40, 0x89, 0x89, 0xad, 0x48, 0xa2, 0x14, 0x92, 0x84, 0x20, 0x87,
    ];

    let ek = aes256_expand_key(&key);
    let mut block = plaintext;
    aes256_encrypt_block(&ek, &mut block);
    assert_eq!(block, expected);
  }

  /// GF(2^8) multiplication spot checks.
  #[test]
  fn gf256_mul_spot_checks() {
    // 0 * anything = 0
    assert_eq!(gf256_mul(0x00, 0x53), 0x00);
    // 1 * x = x
    assert_eq!(gf256_mul(0x01, 0x53), 0x53);
    // 0x57 * 0x83 = 0xc1 (from FIPS 197 §4.2.1)
    assert_eq!(gf256_mul(0x57, 0x83), 0xc1);
  }

  /// Exhaustive GF(2^8) inverse: x * x^{-1} = 1 for all nonzero x.
  #[test]
  fn gf256_inv_exhaustive() {
    assert_eq!(gf256_inv(0), 0, "inv(0) must be 0 by AES convention");
    for x in 1u16..=255 {
      let x = x as u8;
      let inv = gf256_inv(x);
      assert_eq!(gf256_mul(x, inv), 1, "x={x:#04x}, inv={inv:#04x}: x * inv != 1");
    }
  }

  /// AES-256 CTR mode: round-trip (encrypt then decrypt = identity).
  #[cfg(feature = "aes-gcm-siv")]
  #[test]
  fn aes256_ctr32_round_trip() {
    let key = [0x42u8; 32];
    let ek = aes256_expand_key(&key);
    let iv = [0x01u8; 16];

    let plaintext = b"Hello, AES-CTR mode test with multiple blocks of data!!";
    let mut buf = plaintext.to_vec();

    // Encrypt.
    aes256_ctr32_encrypt(&ek, &iv, &mut buf);
    assert_ne!(&buf[..], &plaintext[..]);

    // Decrypt (CTR is symmetric).
    aes256_ctr32_encrypt(&ek, &iv, &mut buf);
    assert_eq!(&buf[..], &plaintext[..]);
  }

  /// AES-256 CTR: known-answer test using the GCM-SIV test vector.
  /// Validates actual keystream output, not just round-trip.
  #[cfg(feature = "aes-gcm-siv")]
  #[test]
  fn aes256_ctr32_known_answer() {
    // From RFC 8452 Appendix C.2 test case 2 (AES-256):
    // enc_key derived from key=01..00, nonce=03..00
    // The ciphertext "1de22967237a8132" is AES-CTR(enc_key, counter_block, plaintext=0200000000000000)
    // We verify the encrypt output is deterministic and matches the expected ciphertext bytes.
    let key = [0x42u8; 32];
    let ek = aes256_expand_key(&key);
    let iv = [0u8; 16];

    // Encrypt 32 zero bytes (2 full blocks).
    let mut buf = [0u8; 32];
    aes256_ctr32_encrypt(&ek, &iv, &mut buf);

    // First 16 bytes = AES(key, iv) — the keystream for block 0.
    let mut expected_block0 = iv;
    aes256_encrypt_block(&ek, &mut expected_block0);
    assert_eq!(&buf[..16], &expected_block0, "CTR block 0 keystream mismatch");

    // Second 16 bytes = AES(key, iv+1).
    let mut iv_plus_1 = iv;
    iv_plus_1[0] = 1; // counter increments LE32
    let mut expected_block1 = iv_plus_1;
    aes256_encrypt_block(&ek, &mut expected_block1);
    assert_eq!(&buf[16..32], &expected_block1, "CTR block 1 keystream mismatch");
  }

  #[cfg(feature = "aes-gcm")]
  #[test]
  fn aes128_ctr32_be_known_answer() {
    let key = [0x12u8; 16];
    let ek = aes128_expand_key(&key);
    let mut iv = [0u8; 16];
    iv[..12].copy_from_slice(b"GCM nonce IV");
    iv[12..16].copy_from_slice(&3u32.to_be_bytes());

    let mut buf = [0u8; 80];
    aes128_ctr32_encrypt_be(&ek, &iv, &mut buf);

    for block_idx in 0..5usize {
      let mut expected = iv;
      let ctr = 3u32.wrapping_add(block_idx as u32);
      expected[12..16].copy_from_slice(&ctr.to_be_bytes());
      aes128_encrypt_block(&ek, &mut expected);
      let start = block_idx.strict_mul(BLOCK_SIZE);
      let end = start.strict_add(BLOCK_SIZE);
      assert_eq!(
        &buf[start..end],
        &expected,
        "AES-128 CTR-BE block {block_idx} keystream mismatch"
      );
    }
  }

  #[cfg(feature = "aes-gcm")]
  #[test]
  fn aes128_ctr32_be_counter_wraps() {
    let key = [0x4du8; KEY_SIZE_128];
    let ek = aes128_expand_key(&key);
    let mut iv = [0u8; 16];
    iv[..12].copy_from_slice(b"wrap nonce12");
    let start_ctr = u32::MAX - 2;
    iv[12..16].copy_from_slice(&start_ctr.to_be_bytes());

    let mut buf = [0u8; 80];
    aes128_ctr32_encrypt_be(&ek, &iv, &mut buf);

    for block_idx in 0..5usize {
      let mut expected = iv;
      let ctr = start_ctr.wrapping_add(block_idx as u32);
      expected[12..16].copy_from_slice(&ctr.to_be_bytes());
      aes128_encrypt_block(&ek, &mut expected);
      let start = block_idx.strict_mul(BLOCK_SIZE);
      let end = start.strict_add(BLOCK_SIZE);
      assert_eq!(
        &buf[start..end],
        &expected,
        "AES-128 CTR-BE wrap block {block_idx} keystream mismatch"
      );
    }
  }

  #[cfg(feature = "aes-gcm")]
  #[test]
  fn aes256_ctr32_be_known_answer() {
    let key = [0x24u8; 32];
    let ek = aes256_expand_key(&key);
    let mut iv = [0u8; 16];
    iv[..12].copy_from_slice(b"GCM nonce IV");
    iv[12..16].copy_from_slice(&7u32.to_be_bytes());

    let mut buf = [0u8; 80];
    aes256_ctr32_encrypt_be(&ek, &iv, &mut buf);

    for block_idx in 0..5usize {
      let mut expected = iv;
      let ctr = 7u32.wrapping_add(block_idx as u32);
      expected[12..16].copy_from_slice(&ctr.to_be_bytes());
      aes256_encrypt_block(&ek, &mut expected);
      let start = block_idx.strict_mul(BLOCK_SIZE);
      let end = start.strict_add(BLOCK_SIZE);
      assert_eq!(
        &buf[start..end],
        &expected,
        "CTR-BE block {block_idx} keystream mismatch"
      );
    }
  }

  #[cfg(feature = "aes-gcm")]
  #[test]
  fn aes256_ctr32_be_counter_wraps() {
    let key = [0x7eu8; 32];
    let ek = aes256_expand_key(&key);
    let mut iv = [0u8; 16];
    iv[..12].copy_from_slice(b"wrap nonce12");
    let start_ctr = u32::MAX - 2;
    iv[12..16].copy_from_slice(&start_ctr.to_be_bytes());

    let mut buf = [0u8; 80];
    aes256_ctr32_encrypt_be(&ek, &iv, &mut buf);

    for block_idx in 0..5usize {
      let mut expected = iv;
      let ctr = start_ctr.wrapping_add(block_idx as u32);
      expected[12..16].copy_from_slice(&ctr.to_be_bytes());
      aes256_encrypt_block(&ek, &mut expected);
      let start = block_idx.strict_mul(BLOCK_SIZE);
      let end = start.strict_add(BLOCK_SIZE);
      assert_eq!(
        &buf[start..end],
        &expected,
        "AES-256 CTR-BE wrap block {block_idx} keystream mismatch"
      );
    }
  }

  #[cfg(all(target_arch = "aarch64", feature = "aes-gcm"))]
  fn aarch64_gcm_caps_available() -> bool {
    crate::platform::caps().has(crate::platform::caps::aarch64::AES_PMULL)
  }

  #[cfg(all(target_arch = "aarch64", feature = "aes-gcm"))]
  struct Aarch64GcmTestPowers {
    h_polyval: u128,
    h_powers_rev: [u128; 4],
    h_powers_rev_8: [u128; 8],
    #[cfg(target_os = "macos")]
    h_powers_rev_16: [u128; 16],
    #[cfg(target_os = "macos")]
    h_powers_rev_16_mid: [u128; 16],
    #[cfg(target_os = "macos")]
    h_powers_rev_16_pair: [u128; 24],
  }

  #[cfg(all(target_arch = "aarch64", feature = "aes-gcm"))]
  impl Aarch64GcmTestPowers {
    fn tables(&self) -> Aarch64GcmTables<'_> {
      Aarch64GcmTables {
        h_polyval: self.h_polyval,
        h_powers_rev: &self.h_powers_rev,
        h_powers_rev_8: &self.h_powers_rev_8,
        #[cfg(target_os = "macos")]
        h_powers_rev_16: &self.h_powers_rev_16,
        #[cfg(target_os = "macos")]
        h_powers_rev_16_mid: &self.h_powers_rev_16_mid,
        #[cfg(target_os = "macos")]
        h_powers_rev_16_pair: &self.h_powers_rev_16_pair,
      }
    }
  }

  #[cfg(all(target_arch = "aarch64", feature = "aes-gcm"))]
  fn aarch64_gcm_test_powers() -> Aarch64GcmTestPowers {
    let h_polyval = 0x1287_3d5b_fedc_ba09_7654_3210_f0e1_d2c3u128;
    let powers = crate::aead::polyval::precompute_powers_8(h_polyval);
    let h_powers_rev = [powers[3], powers[2], powers[1], powers[0]];
    let h_powers_rev_8 = [
      powers[7], powers[6], powers[5], powers[4], powers[3], powers[2], powers[1], powers[0],
    ];
    #[cfg(target_os = "macos")]
    let h_powers_rev_16 = {
      let powers = crate::aead::polyval::precompute_powers_16(h_polyval);
      [
        powers[15], powers[14], powers[13], powers[12], powers[11], powers[10], powers[9], powers[8], powers[7],
        powers[6], powers[5], powers[4], powers[3], powers[2], powers[1], powers[0],
      ]
    };
    #[cfg(target_os = "macos")]
    let h_powers_rev_16_mid = crate::aead::polyval::precompute_powers_16_mid(&h_powers_rev_16);
    #[cfg(target_os = "macos")]
    let h_powers_rev_16_pair = crate::aead::polyval::precompute_powers_16_pair(&h_powers_rev_16);
    Aarch64GcmTestPowers {
      h_polyval,
      h_powers_rev,
      h_powers_rev_8,
      #[cfg(target_os = "macos")]
      h_powers_rev_16,
      #[cfg(target_os = "macos")]
      h_powers_rev_16_mid,
      #[cfg(target_os = "macos")]
      h_powers_rev_16_pair,
    }
  }

  #[cfg(all(target_arch = "aarch64", feature = "aes-gcm"))]
  fn aarch64_gcm_wrap_counter_block() -> [u8; 16] {
    let mut counter = [0u8; 16];
    counter[..12].copy_from_slice(b"ctr wrap iv!");
    counter[12..16].copy_from_slice(&(u32::MAX - 7).to_be_bytes());
    counter
  }

  #[cfg(all(target_arch = "aarch64", feature = "aes-gcm"))]
  fn fill_aarch64_gcm_test_plaintext<const N: usize>(out: &mut [u8; N]) {
    let mut i = 0usize;
    while i < N {
      out[i] = (i as u8).wrapping_mul(0x3d).wrapping_add(0x47) ^ ((i >> 3) as u8).wrapping_mul(0x91);
      i = i.strict_add(1);
    }
  }

  #[cfg(all(target_arch = "aarch64", feature = "aes-gcm"))]
  #[test]
  fn aarch64_aes128_gcm_encrypt_matches_portable_across_counter_wrap() {
    if !aarch64_gcm_caps_available() {
      return;
    }

    let key = [0xA1u8; KEY_SIZE_128];
    let ek = aes128_expand_key(&key);
    let portable_ek = Aes128EncKey {
      inner: Key128Inner::PortableRoundKeys(aes128_expand_key_portable(&key)),
    };
    let counter = aarch64_gcm_wrap_counter_block();
    let powers = aarch64_gcm_test_powers();
    let tables = powers.tables();
    let seed_acc = 0xfeed_face_cafe_babe_1020_3040_5060_7080u128;

    let mut plaintext = [0u8; 384];
    fill_aarch64_gcm_test_plaintext(&mut plaintext);
    let mut expected = plaintext;
    aes128_ctr32_encrypt_be(&portable_ek, &counter, &mut expected);
    let expected_acc = ghash_ciphertext_fallback(seed_acc, powers.h_polyval, &expected);

    let mut actual = plaintext;
    // SAFETY: runtime caps above confirmed AES-CE + PMULL before calling the AArch64 target-feature
    // helper. Inputs are fixed-size initialized test buffers.
    let actual_acc = unsafe { aes128_ctr32_encrypt_be_aarch64_ghash(&ek, &counter, &mut actual, seed_acc, &tables) };

    assert_eq!(
      actual, expected,
      "AES-128 AArch64 seal ciphertext must match portable CTR across wrap"
    );
    assert_eq!(
      actual_acc, expected_acc,
      "AES-128 AArch64 seal GHASH accumulator must match portable fold across wrap"
    );
  }

  #[cfg(all(target_arch = "aarch64", feature = "aes-gcm"))]
  #[test]
  fn aarch64_aes128_gcm_decrypt_matches_portable_across_counter_wrap() {
    if !aarch64_gcm_caps_available() {
      return;
    }

    let key = [0xB2u8; KEY_SIZE_128];
    let ek = aes128_expand_key(&key);
    let portable_ek = Aes128EncKey {
      inner: Key128Inner::PortableRoundKeys(aes128_expand_key_portable(&key)),
    };
    let counter = aarch64_gcm_wrap_counter_block();
    let powers = aarch64_gcm_test_powers();
    let tables = powers.tables();
    let seed_acc = 0x9ace_0246_8bdf_1357_1122_3344_5566_7788u128;

    let mut plaintext = [0u8; 384];
    fill_aarch64_gcm_test_plaintext(&mut plaintext);
    let mut ciphertext = plaintext;
    aes128_ctr32_encrypt_be(&portable_ek, &counter, &mut ciphertext);
    let expected_acc = ghash_ciphertext_fallback(seed_acc, powers.h_polyval, &ciphertext);

    let mut actual = ciphertext;
    // SAFETY: runtime caps above confirmed AES-CE + PMULL before calling the AArch64 target-feature
    // helper. Inputs are fixed-size initialized test buffers.
    let actual_acc = unsafe { aes128_ctr32_decrypt_be_aarch64_ghash(&ek, &counter, &mut actual, seed_acc, &tables) };

    assert_eq!(
      actual, plaintext,
      "AES-128 AArch64 open plaintext must match portable CTR across wrap"
    );
    assert_eq!(
      actual_acc, expected_acc,
      "AES-128 AArch64 open GHASH accumulator must match portable fold across wrap"
    );
  }

  #[cfg(all(target_arch = "aarch64", feature = "aes-gcm"))]
  #[test]
  fn aarch64_aes256_gcm_encrypt_matches_portable_across_counter_wrap() {
    if !aarch64_gcm_caps_available() {
      return;
    }

    let key = [0xC3u8; KEY_SIZE];
    let ek = aes256_expand_key(&key);
    let portable_ek = Aes256EncKey {
      inner: KeyInner::PortableRoundKeys(aes256_expand_key_portable(&key)),
    };
    let counter = aarch64_gcm_wrap_counter_block();
    let powers = aarch64_gcm_test_powers();
    let tables = powers.tables();
    let seed_acc = 0x0123_4567_89ab_cdef_fedc_ba98_7654_3210u128;

    let mut plaintext = [0u8; 384];
    fill_aarch64_gcm_test_plaintext(&mut plaintext);
    let mut expected = plaintext;
    aes256_ctr32_encrypt_be(&portable_ek, &counter, &mut expected);
    let expected_acc = ghash_ciphertext_fallback(seed_acc, powers.h_polyval, &expected);

    let mut actual = plaintext;
    // SAFETY: runtime caps above confirmed AES-CE + PMULL before calling the AArch64 target-feature
    // helper. Inputs are fixed-size initialized test buffers.
    let actual_acc = unsafe { aes256_ctr32_encrypt_be_aarch64_ghash(&ek, &counter, &mut actual, seed_acc, &tables) };

    assert_eq!(
      actual, expected,
      "AES-256 AArch64 seal ciphertext must match portable CTR across wrap"
    );
    assert_eq!(
      actual_acc, expected_acc,
      "AES-256 AArch64 seal GHASH accumulator must match portable fold across wrap"
    );
  }

  #[cfg(all(target_arch = "aarch64", feature = "aes-gcm"))]
  #[test]
  fn aarch64_aes256_gcm_decrypt_matches_portable_across_counter_wrap() {
    if !aarch64_gcm_caps_available() {
      return;
    }

    let key = [0xD4u8; KEY_SIZE];
    let ek = aes256_expand_key(&key);
    let portable_ek = Aes256EncKey {
      inner: KeyInner::PortableRoundKeys(aes256_expand_key_portable(&key)),
    };
    let counter = aarch64_gcm_wrap_counter_block();
    let powers = aarch64_gcm_test_powers();
    let tables = powers.tables();
    let seed_acc = 0xaa55_aa55_55aa_55aa_cc33_cc33_33cc_33ccu128;

    let mut plaintext = [0u8; 384];
    fill_aarch64_gcm_test_plaintext(&mut plaintext);
    let mut ciphertext = plaintext;
    aes256_ctr32_encrypt_be(&portable_ek, &counter, &mut ciphertext);
    let expected_acc = ghash_ciphertext_fallback(seed_acc, powers.h_polyval, &ciphertext);

    let mut actual = ciphertext;
    // SAFETY: runtime caps above confirmed AES-CE + PMULL before calling the AArch64 target-feature
    // helper. Inputs are fixed-size initialized test buffers.
    let actual_acc = unsafe { aes256_ctr32_decrypt_be_aarch64_ghash(&ek, &counter, &mut actual, seed_acc, &tables) };

    assert_eq!(
      actual, plaintext,
      "AES-256 AArch64 open plaintext must match portable CTR across wrap"
    );
    assert_eq!(
      actual_acc, expected_acc,
      "AES-256 AArch64 open GHASH accumulator must match portable fold across wrap"
    );
  }

  #[cfg(all(target_arch = "x86_64", feature = "aes-gcm"))]
  fn x86_gcm_iv_words(iv_prefix: &[u8; 12]) -> [u32; 3] {
    [
      u32::from_le_bytes([iv_prefix[0], iv_prefix[1], iv_prefix[2], iv_prefix[3]]),
      u32::from_le_bytes([iv_prefix[4], iv_prefix[5], iv_prefix[6], iv_prefix[7]]),
      u32::from_le_bytes([iv_prefix[8], iv_prefix[9], iv_prefix[10], iv_prefix[11]]),
    ]
  }

  #[cfg(all(target_arch = "x86_64", feature = "aes-gcm"))]
  fn fill_expected_gcm_counter_blocks<const N: usize>(iv_prefix: &[u8; 12], ctr: u32, expected: &mut [u8; N]) {
    debug_assert_eq!(N.strict_rem(BLOCK_SIZE), 0);
    let blocks = N.strict_div(BLOCK_SIZE);
    let mut block_idx = 0usize;
    while block_idx < blocks {
      let start = block_idx.strict_mul(BLOCK_SIZE);
      expected[start..start.strict_add(12)].copy_from_slice(iv_prefix);
      expected[start.strict_add(12)..start.strict_add(16)]
        .copy_from_slice(&ctr.wrapping_add(block_idx as u32).to_be_bytes());
      block_idx = block_idx.strict_add(1);
    }
  }

  #[cfg(all(target_arch = "x86_64", feature = "aes-gcm"))]
  #[target_feature(enable = "sse2")]
  /// # Safety
  ///
  /// Caller must ensure SSE2 is available before calling this target-feature helper.
  unsafe fn x86_gcm_ctr_block_be_test_bytes(iv_words: [u32; 3], ctr: u32) -> [u8; 16] {
    use core::arch::x86_64::*;

    let mut out = [0u8; 16];
    // SAFETY: test-only XMM counter block store because:
    // 1. The caller verified SSE2 before invoking this target-feature helper.
    // 2. `out` is exactly 16 writable bytes, matching one `__m128i` store.
    // 3. The vector under test is produced directly from the counter constructor.
    unsafe { _mm_storeu_si128(out.as_mut_ptr().cast(), x86_gcm_ctr_block_be(iv_words, ctr)) };
    out
  }

  #[cfg(all(target_arch = "x86_64", feature = "aes-gcm"))]
  #[target_feature(enable = "avx2")]
  /// # Safety
  ///
  /// Caller must ensure AVX2 is available before calling this target-feature helper.
  unsafe fn x86_gcm_ctr_blocks_be_2_test_bytes(iv_words: [u32; 3], ctr: u32) -> [u8; 32] {
    use core::arch::x86_64::*;

    let mut out = [0u8; 32];
    // SAFETY: test-only YMM counter block store because:
    // 1. The caller verified AVX2 before invoking this target-feature helper.
    // 2. `out` is exactly 32 writable bytes, matching one `__m256i` store.
    // 3. The vector under test is produced directly from the two-block counter constructor.
    unsafe { _mm256_storeu_si256(out.as_mut_ptr().cast(), x86_gcm_ctr_blocks_be_2(iv_words, ctr)) };
    out
  }

  #[cfg(all(target_arch = "x86_64", feature = "aes-gcm"))]
  #[target_feature(enable = "avx512f")]
  /// # Safety
  ///
  /// Caller must ensure AVX-512F is available before calling this target-feature helper.
  unsafe fn x86_gcm_ctr_blocks_be_4_test_bytes(iv_words: [u32; 3], ctr: u32) -> [u8; 64] {
    use core::arch::x86_64::*;

    let mut out = [0u8; 64];
    // SAFETY: test-only ZMM counter block store because:
    // 1. The caller verified AVX-512F before invoking this target-feature helper.
    // 2. `out` is exactly 64 writable bytes, matching one `__m512i` store.
    // 3. The vector under test is produced directly from the four-block counter constructor.
    unsafe { _mm512_storeu_si512(out.as_mut_ptr().cast(), x86_gcm_ctr_blocks_be_4(iv_words, ctr)) };
    out
  }

  #[cfg(all(target_arch = "x86_64", feature = "aes-gcm"))]
  #[target_feature(enable = "avx512f,avx512bw")]
  /// # Safety
  ///
  /// Caller must ensure AVX-512F and AVX-512BW are available before calling this target-feature
  /// helper.
  unsafe fn x86_gcm_ctr_blocks_be_16_test_bytes(iv_words: [u32; 3], ctr: u32) -> [u8; 256] {
    use core::arch::x86_64::*;

    let mut out = [0u8; 256];
    // SAFETY: test-only ZMM counter block stores because:
    // 1. The caller verified AVX-512F + AVX-512BW before invoking this target-feature helper.
    // 2. `out` is exactly 256 writable bytes, matching four contiguous `__m512i` stores.
    // 3. The vectors under test are produced directly from the sixteen-block counter constructor.
    unsafe {
      let (c0, c1, c2, c3) = x86_gcm_ctr_blocks_be_16(iv_words, ctr);
      _mm512_storeu_si512(out.as_mut_ptr().cast(), c0);
      _mm512_storeu_si512(out.as_mut_ptr().add(64).cast(), c1);
      _mm512_storeu_si512(out.as_mut_ptr().add(128).cast(), c2);
      _mm512_storeu_si512(out.as_mut_ptr().add(192).cast(), c3);
    }
    out
  }

  #[cfg(all(target_arch = "x86_64", feature = "aes-gcm"))]
  #[target_feature(enable = "avx2")]
  /// # Safety
  ///
  /// Caller must ensure AVX2 is available before calling this target-feature helper.
  unsafe fn x86_gcm_ctr_blocks_be_8_y256_test_bytes(iv_words: [u32; 3], ctr: u32) -> [u8; 128] {
    use core::arch::x86_64::*;

    let mut out = [0u8; 128];
    // SAFETY: test-only YMM counter block stores because:
    // 1. The caller verified AVX2 before invoking this target-feature helper.
    // 2. `out` is exactly 128 writable bytes, matching four contiguous `__m256i` stores.
    // 3. The vectors under test are produced directly from the eight-block counter constructor.
    unsafe {
      let (c0, c1, c2, c3) = x86_gcm_ctr_blocks_be_8_y256(iv_words, ctr);
      _mm256_storeu_si256(out.as_mut_ptr().cast(), c0);
      _mm256_storeu_si256(out.as_mut_ptr().add(32).cast(), c1);
      _mm256_storeu_si256(out.as_mut_ptr().add(64).cast(), c2);
      _mm256_storeu_si256(out.as_mut_ptr().add(96).cast(), c3);
    }
    out
  }

  #[cfg(all(target_arch = "x86_64", feature = "aes-gcm"))]
  #[test]
  fn x86_gcm_ctr_block_be_preserves_prefix_and_encodes_counter() {
    if !crate::platform::caps().has(crate::platform::caps::x86::SSE2) {
      return;
    }

    let iv_prefix = *b"ctr wrap iv!";
    let iv_words = x86_gcm_iv_words(&iv_prefix);

    for ctr in [0x0102_0304, u32::MAX] {
      let mut expected = [0u8; 16];
      fill_expected_gcm_counter_blocks(&iv_prefix, ctr, &mut expected);
      // SAFETY: runtime caps above confirmed SSE2 before calling the target-feature helper.
      let actual = unsafe { x86_gcm_ctr_block_be_test_bytes(iv_words, ctr) };
      assert_eq!(
        actual.as_slice(),
        expected.as_slice(),
        "x86 XMM GCM counter block mismatch"
      );
    }
  }

  #[cfg(all(target_arch = "x86_64", feature = "aes-gcm"))]
  #[test]
  fn x86_gcm_ctr_blocks_be_2_preserves_prefix_and_wraps_counter() {
    if !crate::platform::caps().has(crate::platform::caps::x86::AVX2) {
      return;
    }

    let iv_prefix = *b"ctr wrap iv!";
    let iv_words = x86_gcm_iv_words(&iv_prefix);

    for ctr in [0x0102_0304, u32::MAX] {
      let mut expected = [0u8; 32];
      fill_expected_gcm_counter_blocks(&iv_prefix, ctr, &mut expected);
      // SAFETY: runtime caps above confirmed AVX2 before calling the target-feature helper.
      let actual = unsafe { x86_gcm_ctr_blocks_be_2_test_bytes(iv_words, ctr) };
      assert_eq!(
        actual.as_slice(),
        expected.as_slice(),
        "x86 YMM GCM counter block mismatch"
      );
    }
  }

  #[cfg(all(target_arch = "x86_64", feature = "aes-gcm"))]
  #[test]
  fn x86_gcm_ctr_blocks_be_4_preserves_prefix_and_wraps_counter() {
    if !crate::platform::caps().has(crate::platform::caps::x86::AVX512F) {
      return;
    }

    let iv_prefix = *b"ctr wrap iv!";
    let iv_words = x86_gcm_iv_words(&iv_prefix);

    for ctr in [0x0102_0304, u32::MAX - 1] {
      let mut expected = [0u8; 64];
      fill_expected_gcm_counter_blocks(&iv_prefix, ctr, &mut expected);
      // SAFETY: runtime caps above confirmed AVX-512F before calling the target-feature helper.
      let actual = unsafe { x86_gcm_ctr_blocks_be_4_test_bytes(iv_words, ctr) };
      assert_eq!(
        actual.as_slice(),
        expected.as_slice(),
        "x86 ZMM GCM counter block mismatch"
      );
    }
  }

  #[cfg(all(target_arch = "x86_64", feature = "aes-gcm"))]
  #[test]
  fn x86_gcm_ctr_blocks_be_16_preserves_prefix_and_wraps_counter() {
    let required = crate::platform::caps::x86::AVX512F | crate::platform::caps::x86::AVX512BW;
    if !crate::platform::caps().has(required) {
      return;
    }

    let iv_prefix = *b"ctr wrap iv!";
    let iv_words = x86_gcm_iv_words(&iv_prefix);

    for ctr in [0x0102_0304, u32::MAX - 7] {
      let mut expected = [0u8; 256];
      fill_expected_gcm_counter_blocks(&iv_prefix, ctr, &mut expected);
      // SAFETY: runtime caps above confirmed AVX-512F + AVX-512BW before calling the
      // target-feature helper.
      let actual = unsafe { x86_gcm_ctr_blocks_be_16_test_bytes(iv_words, ctr) };
      assert_eq!(
        actual.as_slice(),
        expected.as_slice(),
        "x86 vectorized ZMM GCM counter block mismatch"
      );
    }
  }

  #[cfg(all(target_arch = "x86_64", feature = "aes-gcm"))]
  #[test]
  fn x86_gcm_ctr_blocks_be_8_y256_preserves_prefix_and_wraps_counter() {
    if !crate::platform::caps().has(crate::platform::caps::x86::AVX2) {
      return;
    }

    let iv_prefix = *b"ctr wrap iv!";
    let iv_words = x86_gcm_iv_words(&iv_prefix);

    for ctr in [0x0102_0304, u32::MAX - 3] {
      let mut expected = [0u8; 128];
      fill_expected_gcm_counter_blocks(&iv_prefix, ctr, &mut expected);
      // SAFETY: runtime caps above confirmed AVX2 before calling the target-feature helper.
      let actual = unsafe { x86_gcm_ctr_blocks_be_8_y256_test_bytes(iv_words, ctr) };
      assert_eq!(
        actual.as_slice(),
        expected.as_slice(),
        "x86 vectorized YMM GCM counter block mismatch"
      );
    }
  }

  #[cfg(all(target_arch = "x86_64", feature = "aes-gcm"))]
  fn x86_y256_gcm_caps_available() -> bool {
    let required = crate::platform::caps::x86::VAES_READY
      | crate::platform::caps::x86::VPCLMUL_READY
      | crate::platform::caps::x86::AVX2
      | crate::platform::caps::x86::AESNI;
    crate::platform::caps().has(required)
  }

  #[cfg(all(target_arch = "x86_64", feature = "aes-gcm"))]
  fn x86_z512_gcm_caps_available() -> bool {
    let required = crate::platform::caps::x86::VAES_READY
      | crate::platform::caps::x86::VPCLMUL_READY
      | crate::platform::caps::x86::AESNI;
    crate::platform::caps().has(required)
  }

  #[cfg(all(target_arch = "x86_64", feature = "aes-gcm"))]
  fn x86_gcm_test_powers() -> (u128, [u128; 4], [u128; 8]) {
    let h_polyval = 0x1287_3d5b_fedc_ba09_7654_3210_f0e1_d2c3u128;
    let powers = crate::aead::polyval::precompute_powers_8(h_polyval);
    (
      h_polyval,
      [powers[3], powers[2], powers[1], powers[0]],
      [
        powers[7], powers[6], powers[5], powers[4], powers[3], powers[2], powers[1], powers[0],
      ],
    )
  }

  #[cfg(all(target_arch = "x86_64", feature = "aes-gcm"))]
  fn x86_gcm_test_powers_16() -> (u128, [u128; 4], [u128; 16]) {
    let h_polyval = 0x1287_3d5b_fedc_ba09_7654_3210_f0e1_d2c3u128;
    let powers = crate::aead::polyval::precompute_powers_16(h_polyval);
    (
      h_polyval,
      [powers[3], powers[2], powers[1], powers[0]],
      [
        powers[15], powers[14], powers[13], powers[12], powers[11], powers[10], powers[9], powers[8], powers[7],
        powers[6], powers[5], powers[4], powers[3], powers[2], powers[1], powers[0],
      ],
    )
  }

  #[cfg(all(target_arch = "x86_64", feature = "aes-gcm"))]
  fn x86_gcm_wrap_counter_block() -> [u8; 16] {
    let mut counter = [0u8; 16];
    counter[..12].copy_from_slice(b"ctr wrap iv!");
    counter[12..16].copy_from_slice(&(u32::MAX - 3).to_be_bytes());
    counter
  }

  #[cfg(all(target_arch = "x86_64", feature = "aes-gcm"))]
  fn fill_x86_gcm_test_plaintext<const N: usize>(out: &mut [u8; N]) {
    let mut i = 0usize;
    while i < N {
      out[i] = (i as u8).wrapping_mul(0x3d).wrapping_add(0x47) ^ ((i >> 3) as u8).wrapping_mul(0x91);
      i = i.strict_add(1);
    }
  }

  #[cfg(all(target_arch = "x86_64", feature = "aes-gcm"))]
  #[test]
  fn x86_aes128_gcm_y256_encrypt_matches_scalar_across_counter_wrap() {
    if !x86_y256_gcm_caps_available() {
      return;
    }

    let ek = aes128_expand_key(&[0xA1u8; KEY_SIZE_128]);
    let counter = x86_gcm_wrap_counter_block();
    let (h_polyval, h_powers_rev, h_powers_rev_8) = x86_gcm_test_powers();
    let seed_acc = 0xfeed_face_cafe_babe_1020_3040_5060_7080u128;

    let mut plaintext = [0u8; 128];
    fill_x86_gcm_test_plaintext(&mut plaintext);
    let mut expected = plaintext;
    aes128_ctr32_encrypt_be(&ek, &counter, &mut expected);
    let expected_acc = ghash_ciphertext_fallback(seed_acc, h_polyval, &expected);

    let mut actual = plaintext;
    // SAFETY: runtime caps above confirmed VAES + VPCLMULQDQ + AVX2 + AES-NI before calling
    // the y256 target-feature helper. Inputs are fixed-size initialized test buffers.
    let actual_acc = unsafe {
      aes128_ctr32_encrypt_be_y256_ghash(
        &ek,
        &counter,
        &mut actual,
        seed_acc,
        h_polyval,
        &h_powers_rev,
        &h_powers_rev_8,
      )
    };

    assert_eq!(
      actual, expected,
      "AES-128 y256 seal ciphertext must match scalar CTR across wrap"
    );
    assert_eq!(
      actual_acc, expected_acc,
      "AES-128 y256 seal GHASH accumulator must match scalar fold across wrap"
    );
  }

  #[cfg(all(target_arch = "x86_64", feature = "aes-gcm"))]
  #[test]
  fn x86_aes128_gcm_y256_decrypt_matches_scalar_across_counter_wrap() {
    if !x86_y256_gcm_caps_available() {
      return;
    }

    let ek = aes128_expand_key(&[0xB2u8; KEY_SIZE_128]);
    let counter = x86_gcm_wrap_counter_block();
    let (h_polyval, h_powers_rev, h_powers_rev_8) = x86_gcm_test_powers();
    let seed_acc = 0x9ace_0246_8bdf_1357_1122_3344_5566_7788u128;

    let mut plaintext = [0u8; 128];
    fill_x86_gcm_test_plaintext(&mut plaintext);
    let mut ciphertext = plaintext;
    aes128_ctr32_encrypt_be(&ek, &counter, &mut ciphertext);
    let expected_acc = ghash_ciphertext_fallback(seed_acc, h_polyval, &ciphertext);

    let mut actual = ciphertext;
    // SAFETY: runtime caps above confirmed VAES + VPCLMULQDQ + AVX2 + AES-NI before calling
    // the y256 target-feature helper. Inputs are fixed-size initialized test buffers.
    let actual_acc = unsafe {
      aes128_ctr32_decrypt_be_y256_ghash(
        &ek,
        &counter,
        &mut actual,
        seed_acc,
        h_polyval,
        &h_powers_rev,
        &h_powers_rev_8,
      )
    };

    assert_eq!(
      actual, plaintext,
      "AES-128 y256 open plaintext must match scalar CTR across wrap"
    );
    assert_eq!(
      actual_acc, expected_acc,
      "AES-128 y256 open GHASH accumulator must match scalar fold across wrap"
    );
  }

  #[cfg(all(target_arch = "x86_64", feature = "aes-gcm"))]
  #[test]
  fn x86_aes256_gcm_y256_encrypt_matches_scalar_across_counter_wrap() {
    if !x86_y256_gcm_caps_available() {
      return;
    }

    let ek = aes256_expand_key(&[0xC3u8; KEY_SIZE]);
    let counter = x86_gcm_wrap_counter_block();
    let (h_polyval, h_powers_rev, h_powers_rev_8) = x86_gcm_test_powers();
    let seed_acc = 0x0123_4567_89ab_cdef_fedc_ba98_7654_3210u128;

    let mut plaintext = [0u8; 128];
    fill_x86_gcm_test_plaintext(&mut plaintext);
    let mut expected = plaintext;
    aes256_ctr32_encrypt_be(&ek, &counter, &mut expected);
    let expected_acc = ghash_ciphertext_fallback(seed_acc, h_polyval, &expected);

    let mut actual = plaintext;
    // SAFETY: runtime caps above confirmed VAES + VPCLMULQDQ + AVX2 + AES-NI before calling
    // the y256 target-feature helper. Inputs are fixed-size initialized test buffers.
    let actual_acc = unsafe {
      aes256_ctr32_encrypt_be_y256_ghash(
        &ek,
        &counter,
        &mut actual,
        seed_acc,
        h_polyval,
        &h_powers_rev,
        &h_powers_rev_8,
      )
    };

    assert_eq!(
      actual, expected,
      "AES-256 y256 seal ciphertext must match scalar CTR across wrap"
    );
    assert_eq!(
      actual_acc, expected_acc,
      "AES-256 y256 seal GHASH accumulator must match scalar fold across wrap"
    );
  }

  #[cfg(all(target_arch = "x86_64", feature = "aes-gcm"))]
  #[test]
  fn x86_aes256_gcm_y256_decrypt_matches_scalar_across_counter_wrap() {
    if !x86_y256_gcm_caps_available() {
      return;
    }

    let ek = aes256_expand_key(&[0xD4u8; KEY_SIZE]);
    let counter = x86_gcm_wrap_counter_block();
    let (h_polyval, h_powers_rev, h_powers_rev_8) = x86_gcm_test_powers();
    let seed_acc = 0xaa55_aa55_55aa_55aa_cc33_cc33_33cc_33ccu128;

    let mut plaintext = [0u8; 128];
    fill_x86_gcm_test_plaintext(&mut plaintext);
    let mut ciphertext = plaintext;
    aes256_ctr32_encrypt_be(&ek, &counter, &mut ciphertext);
    let expected_acc = ghash_ciphertext_fallback(seed_acc, h_polyval, &ciphertext);

    let mut actual = ciphertext;
    // SAFETY: runtime caps above confirmed VAES + VPCLMULQDQ + AVX2 + AES-NI before calling
    // the y256 target-feature helper. Inputs are fixed-size initialized test buffers.
    let actual_acc = unsafe {
      aes256_ctr32_decrypt_be_y256_ghash(
        &ek,
        &counter,
        &mut actual,
        seed_acc,
        h_polyval,
        &h_powers_rev,
        &h_powers_rev_8,
      )
    };

    assert_eq!(
      actual, plaintext,
      "AES-256 y256 open plaintext must match scalar CTR across wrap"
    );
    assert_eq!(
      actual_acc, expected_acc,
      "AES-256 y256 open GHASH accumulator must match scalar fold across wrap"
    );
  }

  #[cfg(all(
    target_arch = "x86_64",
    feature = "aes-gcm",
    any(target_os = "linux", target_os = "macos", target_os = "windows")
  ))]
  #[test]
  fn x86_aes128_gcm_y256_large_asm_tail_matches_scalar() {
    if !x86_y256_gcm_caps_available() {
      return;
    }

    const LEN: usize = 1057;

    let ek = aes128_expand_key(&[0xE5u8; KEY_SIZE_128]);
    let counter = x86_gcm_wrap_counter_block();
    let (h_polyval, h_powers_rev, h_powers_rev_8) = x86_gcm_test_powers();
    let seed_acc = 0x3141_5926_5358_9793_2384_6264_3383_2795u128;

    let mut plaintext = [0u8; LEN];
    fill_x86_gcm_test_plaintext(&mut plaintext);

    let mut expected = plaintext;
    aes128_ctr32_encrypt_be(&ek, &counter, &mut expected);
    let expected_acc = ghash_ciphertext_fallback(seed_acc, h_polyval, &expected);

    let mut actual = plaintext;
    // SAFETY: large x86 AES-128-GCM seal test because:
    // 1. Runtime caps above confirmed VAES + VPCLMULQDQ + AVX2 + AES-NI.
    // 2. The 1057-byte input forces the x86 ASM bulk path to process full blocks.
    // 3. The final byte is handled by the shared Rust tail after the ASM path returns.
    let actual_acc = unsafe {
      aes128_ctr32_encrypt_be_y256_ghash(
        &ek,
        &counter,
        &mut actual,
        seed_acc,
        h_polyval,
        &h_powers_rev,
        &h_powers_rev_8,
      )
    };

    assert_eq!(actual, expected, "AES-128 large y256/ASM seal ciphertext mismatch");
    assert_eq!(
      actual_acc, expected_acc,
      "AES-128 large y256/ASM seal GHASH accumulator mismatch"
    );

    let mut opened = actual;
    // SAFETY: large x86 AES-128-GCM open test because:
    // 1. Runtime caps above confirmed VAES + VPCLMULQDQ + AVX2 + AES-NI.
    // 2. The 1057-byte input forces the x86 ASM bulk path to process full blocks.
    // 3. The helper GHASHes ciphertext before decrypting and leaves the final byte to the Rust tail.
    let open_acc = unsafe {
      aes128_ctr32_decrypt_be_y256_ghash(
        &ek,
        &counter,
        &mut opened,
        seed_acc,
        h_polyval,
        &h_powers_rev,
        &h_powers_rev_8,
      )
    };

    assert_eq!(opened, plaintext, "AES-128 large y256/ASM open plaintext mismatch");
    assert_eq!(
      open_acc, expected_acc,
      "AES-128 large y256/ASM open GHASH accumulator mismatch"
    );
  }

  #[cfg(all(
    target_arch = "x86_64",
    feature = "aes-gcm",
    any(target_os = "linux", target_os = "macos", target_os = "windows")
  ))]
  #[test]
  fn x86_aes256_gcm_y256_large_asm_tail_matches_scalar() {
    if !x86_y256_gcm_caps_available() {
      return;
    }

    const LEN: usize = 1057;

    let ek = aes256_expand_key(&[0xF6u8; KEY_SIZE]);
    let counter = x86_gcm_wrap_counter_block();
    let (h_polyval, h_powers_rev, h_powers_rev_8) = x86_gcm_test_powers();
    let seed_acc = 0x2718_2818_2845_9045_2353_6028_7471_3526u128;

    let mut plaintext = [0u8; LEN];
    fill_x86_gcm_test_plaintext(&mut plaintext);

    let mut expected = plaintext;
    aes256_ctr32_encrypt_be(&ek, &counter, &mut expected);
    let expected_acc = ghash_ciphertext_fallback(seed_acc, h_polyval, &expected);

    let mut actual = plaintext;
    // SAFETY: large x86 AES-256-GCM seal test because:
    // 1. Runtime caps above confirmed VAES + VPCLMULQDQ + AVX2 + AES-NI.
    // 2. The 1057-byte input forces the x86 ASM bulk path to process full blocks.
    // 3. The final byte is handled by the shared Rust tail after the ASM path returns.
    let actual_acc = unsafe {
      aes256_ctr32_encrypt_be_y256_ghash(
        &ek,
        &counter,
        &mut actual,
        seed_acc,
        h_polyval,
        &h_powers_rev,
        &h_powers_rev_8,
      )
    };

    assert_eq!(actual, expected, "AES-256 large y256/ASM seal ciphertext mismatch");
    assert_eq!(
      actual_acc, expected_acc,
      "AES-256 large y256/ASM seal GHASH accumulator mismatch"
    );

    let mut opened = actual;
    // SAFETY: large x86 AES-256-GCM open test because:
    // 1. Runtime caps above confirmed VAES + VPCLMULQDQ + AVX2 + AES-NI.
    // 2. The 1057-byte input forces the x86 ASM bulk path to process full blocks.
    // 3. The helper GHASHes ciphertext before decrypting and leaves the final byte to the Rust tail.
    let open_acc = unsafe {
      aes256_ctr32_decrypt_be_y256_ghash(
        &ek,
        &counter,
        &mut opened,
        seed_acc,
        h_polyval,
        &h_powers_rev,
        &h_powers_rev_8,
      )
    };

    assert_eq!(opened, plaintext, "AES-256 large y256/ASM open plaintext mismatch");
    assert_eq!(
      open_acc, expected_acc,
      "AES-256 large y256/ASM open GHASH accumulator mismatch"
    );
  }

  #[cfg(all(
    target_arch = "x86_64",
    feature = "aes-gcm",
    any(target_os = "linux", target_os = "macos")
  ))]
  #[test]
  fn x86_aes128_gcm_avx512_large_asm_tail_matches_scalar() {
    if !x86_z512_gcm_caps_available() {
      return;
    }

    const LEN: usize = 1057;

    let ek = aes128_expand_key(&[0x17u8; KEY_SIZE_128]);
    let counter = x86_gcm_wrap_counter_block();
    let (h_polyval, h_powers_rev, h_powers_rev_16) = x86_gcm_test_powers_16();
    let seed_acc = 0x1123_5813_2134_5589_1442_3337_6109_8715u128;

    let mut plaintext = [0u8; LEN];
    fill_x86_gcm_test_plaintext(&mut plaintext);

    let mut expected = plaintext;
    aes128_ctr32_encrypt_be(&ek, &counter, &mut expected);
    let expected_acc = ghash_ciphertext_fallback(seed_acc, h_polyval, &expected);

    let mut actual = plaintext;
    // SAFETY: large x86 AES-128-GCM AVX-512 seal test because:
    // 1. Runtime caps above confirmed VAES + VPCLMULQDQ + AVX-512 state + AES-NI.
    // 2. The 1057-byte input forces the Unix AVX-512 ASM bulk path to process full blocks.
    // 3. The final byte is handled by the shared Rust tail after the ASM path returns.
    let actual_acc = unsafe {
      aes128_ctr32_encrypt_be_wide_ghash(
        &ek,
        &counter,
        &mut actual,
        seed_acc,
        h_polyval,
        &h_powers_rev,
        &h_powers_rev_16,
      )
    };

    assert_eq!(actual, expected, "AES-128 large AVX-512 ASM seal ciphertext mismatch");
    assert_eq!(
      actual_acc, expected_acc,
      "AES-128 large AVX-512 ASM seal GHASH accumulator mismatch"
    );

    let mut opened = actual;
    // SAFETY: large x86 AES-128-GCM AVX-512 open test because:
    // 1. Runtime caps above confirmed VAES + VPCLMULQDQ + AVX-512 state + AES-NI.
    // 2. The 1057-byte input forces the Unix AVX-512 ASM bulk path to process full blocks.
    // 3. The helper GHASHes ciphertext before decrypting and leaves the final byte to the Rust tail.
    let open_acc = unsafe {
      aes128_ctr32_decrypt_be_wide_ghash(
        &ek,
        &counter,
        &mut opened,
        seed_acc,
        h_polyval,
        &h_powers_rev,
        &h_powers_rev_16,
      )
    };

    assert_eq!(opened, plaintext, "AES-128 large AVX-512 ASM open plaintext mismatch");
    assert_eq!(
      open_acc, expected_acc,
      "AES-128 large AVX-512 ASM open GHASH accumulator mismatch"
    );
  }

  #[cfg(all(
    target_arch = "x86_64",
    feature = "aes-gcm",
    any(target_os = "linux", target_os = "macos")
  ))]
  #[test]
  fn x86_aes256_gcm_avx512_large_asm_tail_matches_scalar() {
    if !x86_z512_gcm_caps_available() {
      return;
    }

    const LEN: usize = 1057;

    let ek = aes256_expand_key(&[0x29u8; KEY_SIZE]);
    let counter = x86_gcm_wrap_counter_block();
    let (h_polyval, h_powers_rev, h_powers_rev_16) = x86_gcm_test_powers_16();
    let seed_acc = 0x2357_1113_1719_2329_3137_4143_4753_5961u128;

    let mut plaintext = [0u8; LEN];
    fill_x86_gcm_test_plaintext(&mut plaintext);

    let mut expected = plaintext;
    aes256_ctr32_encrypt_be(&ek, &counter, &mut expected);
    let expected_acc = ghash_ciphertext_fallback(seed_acc, h_polyval, &expected);

    let mut actual = plaintext;
    // SAFETY: large x86 AES-256-GCM AVX-512 seal test because:
    // 1. Runtime caps above confirmed VAES + VPCLMULQDQ + AVX-512 state + AES-NI.
    // 2. The 1057-byte input forces the Unix AVX-512 ASM bulk path to process full blocks.
    // 3. The final byte is handled by the shared Rust tail after the ASM path returns.
    let actual_acc = unsafe {
      aes256_ctr32_encrypt_be_wide_ghash(
        &ek,
        &counter,
        &mut actual,
        seed_acc,
        h_polyval,
        &h_powers_rev,
        &h_powers_rev_16,
      )
    };

    assert_eq!(actual, expected, "AES-256 large AVX-512 ASM seal ciphertext mismatch");
    assert_eq!(
      actual_acc, expected_acc,
      "AES-256 large AVX-512 ASM seal GHASH accumulator mismatch"
    );

    let mut opened = actual;
    // SAFETY: large x86 AES-256-GCM AVX-512 open test because:
    // 1. Runtime caps above confirmed VAES + VPCLMULQDQ + AVX-512 state + AES-NI.
    // 2. The 1057-byte input forces the Unix AVX-512 ASM bulk path to process full blocks.
    // 3. The helper GHASHes ciphertext before decrypting and leaves the final byte to the Rust tail.
    let open_acc = unsafe {
      aes256_ctr32_decrypt_be_wide_ghash(
        &ek,
        &counter,
        &mut opened,
        seed_acc,
        h_polyval,
        &h_powers_rev,
        &h_powers_rev_16,
      )
    };

    assert_eq!(opened, plaintext, "AES-256 large AVX-512 ASM open plaintext mismatch");
    assert_eq!(
      open_acc, expected_acc,
      "AES-256 large AVX-512 ASM open GHASH accumulator mismatch"
    );
  }

  #[test]
  fn aes128_encrypt_blocks_ecb_matches_scalar_dispatch() {
    let key = [0xC3u8; KEY_SIZE_128];
    let ek = aes128_expand_key(&key);
    let mut blocks = [[0u8; BLOCK_SIZE]; 6];

    for (i, block) in blocks.iter_mut().enumerate() {
      for (j, byte) in block.iter_mut().enumerate() {
        *byte = (i as u8).wrapping_mul(13) ^ (j as u8).wrapping_mul(31) ^ 0xA3;
      }
    }

    let mut expected = blocks;
    for block in &mut expected {
      aes128_encrypt_block(&ek, block);
    }

    aes128_encrypt_blocks_ecb(&ek, &mut blocks);
    assert_eq!(blocks, expected);
  }

  #[test]
  fn aes256_encrypt_blocks_ecb_matches_scalar_dispatch() {
    let key = [0xA5u8; 32];
    let ek = aes256_expand_key(&key);
    let mut blocks = [[0u8; BLOCK_SIZE]; 6];

    for (i, block) in blocks.iter_mut().enumerate() {
      for (j, byte) in block.iter_mut().enumerate() {
        *byte = (i as u8).wrapping_mul(17) ^ (j as u8).wrapping_mul(29) ^ 0x5C;
      }
    }

    let mut expected = blocks;
    for block in &mut expected {
      aes256_encrypt_block(&ek, block);
    }

    aes256_encrypt_blocks_ecb(&ek, &mut blocks);
    assert_eq!(blocks, expected);
  }

  /// FIPS 197 Appendix C.1 against the table-free fixslice AES-128 path.
  #[test]
  fn riscv64_fixslice_matches_nist_aes128_vector() {
    let key: [u8; 16] = [
      0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
    ];
    let plaintext: [u8; 16] = [
      0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff,
    ];
    let expected: [u8; 16] = [
      0x69, 0xc4, 0xe0, 0xd8, 0x6a, 0x7b, 0x04, 0x30, 0xd8, 0xcd, 0xb7, 0x80, 0x70, 0xb4, 0xc5, 0x5a,
    ];

    let rk = rv_fixslice_aes::RvFixslice128RoundKeys::new(&key);
    let mut block = plaintext;
    rv_fixslice_aes::encrypt_block_128(&rk, &mut block);
    assert_eq!(block, expected);
  }

  #[test]
  fn riscv64_fixslice_128_4blocks_matches_portable() {
    let key = [0xC4u8; KEY_SIZE_128];
    let portable = aes128_expand_key_portable(&key);
    let fixslice = rv_fixslice_aes::RvFixslice128RoundKeys::new(&key);
    let mut blocks = [[0u8; BLOCK_SIZE]; 4];

    for (i, block) in blocks.iter_mut().enumerate() {
      for (j, byte) in block.iter_mut().enumerate() {
        *byte = (i as u8).wrapping_mul(0x47) ^ (j as u8).wrapping_mul(0x6d) ^ 0x9c;
      }
    }

    let mut expected = blocks;
    for block in &mut expected {
      aes128_encrypt_block_portable(&portable, block);
    }

    rv_fixslice_aes::encrypt_4blocks_128(&fixslice, &mut blocks);
    assert_eq!(blocks, expected);
  }

  #[test]
  fn riscv64_fixslice_matches_nist_aes256_vector() {
    let key: [u8; 32] = [
      0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11, 0x12,
      0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
    ];
    let plaintext: [u8; 16] = [
      0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff,
    ];
    let expected: [u8; 16] = [
      0x8e, 0xa2, 0xb7, 0xca, 0x51, 0x67, 0x45, 0xbf, 0xea, 0xfc, 0x49, 0x90, 0x4b, 0x49, 0x60, 0x89,
    ];

    let rk = rv_fixslice_aes::RvFixsliceRoundKeys::new(&key);
    let mut block = plaintext;
    rv_fixslice_aes::encrypt_block(&rk, &mut block);
    assert_eq!(block, expected);
  }

  #[test]
  fn riscv64_fixslice_4blocks_matches_portable() {
    let key = [0x3cu8; KEY_SIZE];
    let portable = aes256_expand_key_portable(&key);
    let fixslice = rv_fixslice_aes::RvFixsliceRoundKeys::new(&key);
    let mut blocks = [[0u8; BLOCK_SIZE]; 4];

    for (i, block) in blocks.iter_mut().enumerate() {
      for (j, byte) in block.iter_mut().enumerate() {
        *byte = (i as u8).wrapping_mul(0x31) ^ (j as u8).wrapping_mul(0x57) ^ 0xa6;
      }
    }

    let mut expected = blocks;
    for block in &mut expected {
      aes256_encrypt_block_portable(&portable, block);
    }

    rv_fixslice_aes::encrypt_4blocks(&fixslice, &mut blocks);
    assert_eq!(blocks, expected);
  }
}
