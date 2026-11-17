//! AES-128 inverse cipher (FIPS 197 section 5.3) for single-block decryption.
//!
//! The only caller is the single-pass QUIC-LB connection-ID decoder. The decryption schedule is
//! selected at expansion time with the same capability rules as [`super::aes128_expand_key`]. The
//! portable inverse cipher is the semantic authority. On RV64 and IBM Z, the bitsliced inverse in
//! the fixslice backend takes that role, as it does for encryption.

#[cfg(target_arch = "aarch64")]
use super::ce;
#[cfg(any(target_arch = "riscv64", target_arch = "s390x"))]
use super::fixslice64;
#[cfg(all(target_arch = "s390x", not(feature = "portable-only")))]
use super::km;
#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
use super::ni;
#[cfg(all(target_arch = "powerpc64", not(feature = "portable-only")))]
use super::ppc;
use super::{BLOCK_SIZE, KEY_SIZE_128};
#[cfg(not(any(target_arch = "riscv64", target_arch = "s390x")))]
use super::{EXPANDED_KEY_WORDS_128, ROUNDS_128, col_byte, gf256_inv, mix_column, xtime};
#[cfg(all(target_arch = "riscv64", not(feature = "portable-only")))]
use super::{rv_aes, rv_scalar_aes};

/// AES-128 decryption round keys in the form that the selected backend consumes. Cleared on drop.
pub(crate) struct Aes128DecKey {
  inner: Dec128Inner,
}

#[cfg_attr(
  all(
    not(feature = "portable-only"),
    any(target_arch = "s390x", all(target_arch = "riscv64", not(feature = "alloc")))
  ),
  expect(
    clippy::large_enum_variant,
    reason = "IBM Z and no-alloc RISC-V keep the fallback schedule inline"
  )
)]
enum Dec128Inner {
  /// The encryption schedule, used in reverse order.
  #[cfg(not(any(target_arch = "riscv64", target_arch = "s390x")))]
  PortableRoundKeys([u32; EXPANDED_KEY_WORDS_128]),
  /// The equivalent inverse cipher schedule for `aesdec`.
  #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
  X86AesNi(ni::Ni128RoundKeys),
  /// The equivalent inverse cipher schedule for `aesd`.
  #[cfg(target_arch = "aarch64")]
  Aarch64Aes(ce::Ce128RoundKeys),
  /// The raw key. KM decrypts with the decipher modifier.
  #[cfg(all(target_arch = "s390x", not(feature = "portable-only")))]
  S390xMsa(km::Km128Key),
  /// The encryption schedule, used in reverse order by `vncipher`.
  #[cfg(all(target_arch = "powerpc64", not(feature = "portable-only")))]
  Power8Crypto(ppc::Ppc128RoundKeys),
  /// The equivalent inverse cipher schedule for `aes64dsm`.
  #[cfg(all(target_arch = "riscv64", not(feature = "portable-only")))]
  ScalarCrypto(rv_scalar_aes::RvScalar128RoundKeys),
  /// The encryption schedule, used in reverse order by `vaesdm`.
  #[cfg(all(target_arch = "riscv64", not(feature = "portable-only")))]
  VectorCrypto(rv_aes::Rv128RoundKeys),
  /// The plain bitsliced schedule, boxed like the alloc-enabled RISC-V encryption fallback.
  #[cfg(all(target_arch = "riscv64", feature = "alloc"))]
  Fixslice(alloc::boxed::Box<fixslice64::Fixslice128InvRoundKeys>),
  /// IBM Z and no-alloc RISC-V builds keep the bitsliced schedule inline.
  #[cfg(any(target_arch = "s390x", all(target_arch = "riscv64", not(feature = "alloc"))))]
  Fixslice(fixslice64::Fixslice128InvRoundKeys),
}

impl Drop for Aes128DecKey {
  fn drop(&mut self) {
    match &mut self.inner {
      #[cfg(not(any(target_arch = "riscv64", target_arch = "s390x")))]
      Dec128Inner::PortableRoundKeys(rk) => crate::traits::ct::zeroize_words(rk),
      // The hardware schedules clear themselves on drop.
      #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
      Dec128Inner::X86AesNi(_) => {}
      #[cfg(target_arch = "aarch64")]
      Dec128Inner::Aarch64Aes(_) => {}
      #[cfg(all(target_arch = "s390x", not(feature = "portable-only")))]
      Dec128Inner::S390xMsa(key) => key.zeroize(),
      #[cfg(all(target_arch = "powerpc64", not(feature = "portable-only")))]
      Dec128Inner::Power8Crypto(rk) => rk.zeroize(),
      #[cfg(all(target_arch = "riscv64", not(feature = "portable-only")))]
      Dec128Inner::ScalarCrypto(rk) => rk.zeroize(),
      #[cfg(all(target_arch = "riscv64", not(feature = "portable-only")))]
      Dec128Inner::VectorCrypto(rk) => rk.zeroize(),
      #[cfg(any(target_arch = "riscv64", target_arch = "s390x"))]
      Dec128Inner::Fixslice(rk) => rk.zeroize(),
    }
  }
}

/// Expand an AES-128 key into the decryption schedule of the best available backend.
#[inline]
pub(crate) fn aes128_expand_decrypt_key(key: &[u8; KEY_SIZE_128]) -> Aes128DecKey {
  #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
  {
    if crate::platform::caps().has(crate::platform::caps::x86::AESNI) {
      return Aes128DecKey {
        // SAFETY: AES-NI availability verified via CPUID above.
        inner: Dec128Inner::X86AesNi(unsafe { ni::expand_decrypt_key_128(key) }),
      };
    }
  }
  #[cfg(target_arch = "aarch64")]
  {
    if crate::platform::caps().has(crate::platform::caps::aarch64::AES) {
      return Aes128DecKey {
        // SAFETY: AES-CE availability verified via HWCAP above.
        inner: Dec128Inner::Aarch64Aes(unsafe { ce::expand_decrypt_key_128(key) }),
      };
    }
  }
  #[cfg(all(target_arch = "s390x", not(feature = "portable-only")))]
  {
    if crate::platform::caps().has(crate::platform::caps::s390x::MSA) {
      return Aes128DecKey {
        inner: Dec128Inner::S390xMsa(km::Km128Key::new(key)),
      };
    }
  }
  #[cfg(all(target_arch = "powerpc64", not(feature = "portable-only")))]
  {
    if crate::platform::caps().has(crate::platform::caps::power::POWER8_CRYPTO) {
      return Aes128DecKey {
        // SAFETY: POWER8 crypto availability verified via HWCAP above.
        inner: Dec128Inner::Power8Crypto(unsafe { ppc::expand_key_128(key) }),
      };
    }
  }
  #[cfg(all(target_arch = "riscv64", not(feature = "portable-only")))]
  {
    if crate::platform::caps().has(crate::platform::caps::riscv::ZVKNED) {
      let mut portable_rk = super::aes128_expand_key_portable(key);
      let rv_keys = rv_aes::from_portable_128(&portable_rk);
      crate::traits::ct::zeroize_words(&mut portable_rk);
      return Aes128DecKey {
        inner: Dec128Inner::VectorCrypto(rv_keys),
      };
    }
    if crate::platform::caps().has(crate::platform::caps::riscv::ZKND) {
      let mut portable_rk = super::aes128_expand_key_portable(key);
      // SAFETY: Zknd availability verified via the capability check above.
      let rv_keys = unsafe { rv_scalar_aes::decrypt_keys_from_portable_128(&portable_rk) };
      crate::traits::ct::zeroize_words(&mut portable_rk);
      return Aes128DecKey {
        inner: Dec128Inner::ScalarCrypto(rv_keys),
      };
    }
  }
  let inner = cfg_select! {
    all(target_arch = "riscv64", feature = "alloc") => {{
      // Expand into the allocation, as `aes128_expand_key` does.
      let mut round_keys = alloc::boxed::Box::new(fixslice64::Fixslice128InvRoundKeys::ZERO);
      round_keys.expand(key);
      Dec128Inner::Fixslice(round_keys)
    }}
    any(target_arch = "riscv64", target_arch = "s390x") => {{
      let mut round_keys = fixslice64::Fixslice128InvRoundKeys::ZERO;
      round_keys.expand(key);
      Dec128Inner::Fixslice(round_keys)
    }}
    _ => Dec128Inner::PortableRoundKeys(super::aes128_expand_key_portable(key)),
  };
  Aes128DecKey { inner }
}

/// Decrypt one 16-byte block with AES-128.
#[inline]
pub(crate) fn aes128_decrypt_block(dk: &Aes128DecKey, block: &mut [u8; BLOCK_SIZE]) {
  match &dk.inner {
    #[cfg(not(any(target_arch = "riscv64", target_arch = "s390x")))]
    Dec128Inner::PortableRoundKeys(rk) => aes128_decrypt_block_portable(rk, block),
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    Dec128Inner::X86AesNi(rk) => {
      // SAFETY: this variant is only constructed after runtime detection confirms AES-NI.
      unsafe { ni::decrypt_block_128(rk, block) }
    }
    #[cfg(target_arch = "aarch64")]
    Dec128Inner::Aarch64Aes(rk) => {
      // SAFETY: this variant is only constructed after runtime detection confirms AES-CE.
      unsafe { ce::decrypt_block_128(rk, block) }
    }
    #[cfg(all(target_arch = "s390x", not(feature = "portable-only")))]
    Dec128Inner::S390xMsa(key) => {
      // SAFETY: this variant is only constructed after runtime detection confirms MSA/CPACF.
      unsafe { km::decrypt_block_128(key, block) }
    }
    #[cfg(all(target_arch = "powerpc64", not(feature = "portable-only")))]
    Dec128Inner::Power8Crypto(rk) => {
      // SAFETY: this variant is only constructed after runtime detection confirms POWER8 crypto.
      unsafe { ppc::decrypt_block_128(rk, block) }
    }
    #[cfg(all(target_arch = "riscv64", not(feature = "portable-only")))]
    Dec128Inner::ScalarCrypto(rk) => {
      // SAFETY: this variant is only constructed after runtime detection confirms Zknd.
      unsafe { rv_scalar_aes::decrypt_block_128(rk, block) }
    }
    #[cfg(all(target_arch = "riscv64", not(feature = "portable-only")))]
    Dec128Inner::VectorCrypto(rk) => {
      // SAFETY: this variant is only constructed after runtime detection confirms Zvkned.
      unsafe { rv_aes::decrypt_block_128(rk, block) }
    }
    #[cfg(any(target_arch = "riscv64", target_arch = "s390x"))]
    Dec128Inner::Fixslice(rk) => fixslice64::decrypt_block_128(rk, block),
  }
}

/// AES inverse S-box: `S^-1(y) = inv(L(y) ^ 0x05)`.
///
/// `L(y) = rotl(y, 1) ^ rotl(y, 3) ^ rotl(y, 6)` is the inverse of the AES affine matrix, and
/// `L(0x63) = 0x05` (FIPS 197 equation 5.6). The source has no table lookup and a fixed schedule.
#[inline(always)]
#[cfg(not(any(target_arch = "riscv64", target_arch = "s390x")))]
const fn inv_sbox(y: u8) -> u8 {
  gf256_inv(y.rotate_left(1) ^ y.rotate_left(3) ^ y.rotate_left(6) ^ 0x05)
}

/// InvShiftRows then InvSubBytes on four big-endian column words.
///
/// After InvShiftRows, row `r` of column `j` comes from column `(j - r) mod 4`.
#[inline(always)]
#[cfg(not(any(target_arch = "riscv64", target_arch = "s390x")))]
const fn inv_shift_sub(s0: u32, s1: u32, s2: u32, s3: u32) -> [[u8; 4]; 4] {
  [
    [
      inv_sbox(col_byte(s0, 0)),
      inv_sbox(col_byte(s3, 1)),
      inv_sbox(col_byte(s2, 2)),
      inv_sbox(col_byte(s1, 3)),
    ],
    [
      inv_sbox(col_byte(s1, 0)),
      inv_sbox(col_byte(s0, 1)),
      inv_sbox(col_byte(s3, 2)),
      inv_sbox(col_byte(s2, 3)),
    ],
    [
      inv_sbox(col_byte(s2, 0)),
      inv_sbox(col_byte(s1, 1)),
      inv_sbox(col_byte(s0, 2)),
      inv_sbox(col_byte(s3, 3)),
    ],
    [
      inv_sbox(col_byte(s3, 0)),
      inv_sbox(col_byte(s2, 1)),
      inv_sbox(col_byte(s1, 2)),
      inv_sbox(col_byte(s0, 3)),
    ],
  ]
}

/// InvMixColumns on one big-endian column word.
///
/// InvMixColumns multiplies by `0Bx^3 + 0Dx^2 + 09x + 0E = c(x) * (04x^2 + 05)` modulo `x^4 + 1`
/// (FIPS 197 equation 5.10). Daemen and Rijmen give the factorization in "The Design of
/// Rijndael", section 4.1.3. So the column is first multiplied by `04x^2 + 05`, then by MixColumns.
#[inline(always)]
#[cfg(not(any(target_arch = "riscv64", target_arch = "s390x")))]
const fn inv_mix_column(column: u32) -> u32 {
  let [b0, b1, b2, b3] = column.to_be_bytes();
  let u = xtime(xtime(b0 ^ b2));
  let v = xtime(xtime(b1 ^ b3));
  mix_column([b0 ^ u, b1 ^ v, b2 ^ u, b3 ^ v])
}

/// Portable AES-128 block decryption with the inverse cipher (FIPS 197 section 5.3).
#[inline]
#[cfg(not(any(target_arch = "riscv64", target_arch = "s390x")))]
fn aes128_decrypt_block_portable(rk: &[u32; EXPANDED_KEY_WORDS_128], block: &mut [u8; BLOCK_SIZE]) {
  let last = ROUNDS_128.strict_mul(4);
  let mut s0 = u32::from_be_bytes([block[0], block[1], block[2], block[3]]) ^ rk[last];
  let mut s1 = u32::from_be_bytes([block[4], block[5], block[6], block[7]]) ^ rk[last.strict_add(1)];
  let mut s2 = u32::from_be_bytes([block[8], block[9], block[10], block[11]]) ^ rk[last.strict_add(2)];
  let mut s3 = u32::from_be_bytes([block[12], block[13], block[14], block[15]]) ^ rk[last.strict_add(3)];

  // Rounds 9 down to 1: InvShiftRows, InvSubBytes, AddRoundKey, InvMixColumns.
  let mut round = ROUNDS_128;
  while round > 1 {
    round = round.strict_sub(1);
    let [t0, t1, t2, t3] = inv_shift_sub(s0, s1, s2, s3);
    let off = round.strict_mul(4);
    s0 = inv_mix_column(u32::from_be_bytes(t0) ^ rk[off]);
    s1 = inv_mix_column(u32::from_be_bytes(t1) ^ rk[off.strict_add(1)]);
    s2 = inv_mix_column(u32::from_be_bytes(t2) ^ rk[off.strict_add(2)]);
    s3 = inv_mix_column(u32::from_be_bytes(t3) ^ rk[off.strict_add(3)]);
  }

  // Final round: no InvMixColumns.
  let [t0, t1, t2, t3] = inv_shift_sub(s0, s1, s2, s3);
  block[0..4].copy_from_slice(&(u32::from_be_bytes(t0) ^ rk[0]).to_be_bytes());
  block[4..8].copy_from_slice(&(u32::from_be_bytes(t1) ^ rk[1]).to_be_bytes());
  block[8..12].copy_from_slice(&(u32::from_be_bytes(t2) ^ rk[2]).to_be_bytes());
  block[12..16].copy_from_slice(&(u32::from_be_bytes(t3) ^ rk[3]).to_be_bytes());
}

#[cfg(test)]
mod tests {
  use aes::cipher::{Array, BlockCipherDecrypt as _, KeyInit as _};

  use super::*;

  const FIPS_KEY: [u8; 16] = [
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
  ];
  const FIPS_PLAINTEXT: [u8; 16] = [
    0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff,
  ];
  const FIPS_CIPHERTEXT: [u8; 16] = [
    0x69, 0xc4, 0xe0, 0xd8, 0x6a, 0x7b, 0x04, 0x30, 0xd8, 0xcd, 0xb7, 0x80, 0x70, 0xb4, 0xc5, 0x5a,
  ];

  /// A deterministic stream of distinct keys and blocks.
  fn generated(state: &mut u64) -> [u8; 16] {
    let mut out = [0u8; 16];
    for byte in &mut out {
      *state ^= *state << 13;
      *state ^= *state >> 7;
      *state ^= *state << 17;
      *byte = state.to_le_bytes()[0];
    }
    out
  }

  fn oracle(key: &[u8; 16], block: &[u8; 16]) -> [u8; 16] {
    let mut out = Array::from(*block);
    aes::Aes128::new(&Array::from(*key)).decrypt_block(&mut out);
    out.into()
  }

  /// One compiled decryption backend, its required capability, and its entry points.
  struct Backend {
    name: &'static str,
    available: fn() -> bool,
    decrypt: fn(&[u8; 16], &mut [u8; 16]),
  }

  fn backends() -> alloc::vec::Vec<Backend> {
    let mut backends = alloc::vec::Vec::new();
    backends.push(Backend {
      name: "dispatched",
      available: || true,
      decrypt: |key, block| aes128_decrypt_block(&aes128_expand_decrypt_key(key), block),
    });
    #[cfg(not(any(target_arch = "riscv64", target_arch = "s390x")))]
    backends.push(Backend {
      name: "portable",
      available: || true,
      decrypt: |key, block| aes128_decrypt_block_portable(&super::super::aes128_expand_key_portable(key), block),
    });
    backends.push(Backend {
      name: "fixslice",
      available: || true,
      decrypt: |key, block| {
        let mut round_keys = super::super::fixslice64::Fixslice128InvRoundKeys::ZERO;
        round_keys.expand(key);
        super::super::fixslice64::decrypt_block_128(&round_keys, block)
      },
    });
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    backends.push(Backend {
      name: "aes-ni",
      available: || crate::platform::caps().has(crate::platform::caps::x86::AESNI),
      // SAFETY: the test calls this only when `available` reports AES-NI.
      decrypt: |key, block| unsafe { ni::decrypt_block_128(&ni::expand_decrypt_key_128(key), block) },
    });
    #[cfg(target_arch = "aarch64")]
    backends.push(Backend {
      name: "aes-ce",
      available: || crate::platform::caps().has(crate::platform::caps::aarch64::AES),
      // SAFETY: the test calls this only when `available` reports AES-CE.
      decrypt: |key, block| unsafe { ce::decrypt_block_128(&ce::expand_decrypt_key_128(key), block) },
    });
    #[cfg(all(target_arch = "s390x", not(feature = "portable-only")))]
    backends.push(Backend {
      name: "km",
      available: || crate::platform::caps().has(crate::platform::caps::s390x::MSA),
      // SAFETY: the test calls this only when `available` reports MSA.
      decrypt: |key, block| unsafe { km::decrypt_block_128(&km::Km128Key::new(key), block) },
    });
    #[cfg(all(target_arch = "powerpc64", not(feature = "portable-only")))]
    backends.push(Backend {
      name: "vncipher",
      available: || crate::platform::caps().has(crate::platform::caps::power::POWER8_CRYPTO),
      // SAFETY: the test calls this only when `available` reports POWER8 crypto.
      decrypt: |key, block| unsafe { ppc::decrypt_block_128(&ppc::expand_key_128(key), block) },
    });
    #[cfg(all(target_arch = "riscv64", not(feature = "portable-only")))]
    backends.push(Backend {
      name: "zvkned",
      available: || crate::platform::caps().has(crate::platform::caps::riscv::ZVKNED),
      // SAFETY: the test calls this only when `available` reports Zvkned.
      decrypt: |key, block| unsafe {
        rv_aes::decrypt_block_128(
          &rv_aes::from_portable_128(&super::super::aes128_expand_key_portable(key)),
          block,
        )
      },
    });
    #[cfg(all(target_arch = "riscv64", not(feature = "portable-only")))]
    backends.push(Backend {
      name: "zknd",
      available: || crate::platform::caps().has(crate::platform::caps::riscv::ZKND),
      // SAFETY: the test calls this only when `available` reports Zknd.
      decrypt: |key, block| unsafe {
        let keys = rv_scalar_aes::decrypt_keys_from_portable_128(&super::super::aes128_expand_key_portable(key));
        rv_scalar_aes::decrypt_block_128(&keys, block)
      },
    });
    backends
  }

  /// FIPS 197 Appendix C.1 decrypts on every backend that this host can run.
  #[test]
  fn every_backend_decrypts_fips_197_c1() {
    for backend in backends().iter().filter(|backend| (backend.available)()) {
      let mut block = FIPS_CIPHERTEXT;
      (backend.decrypt)(&FIPS_KEY, &mut block);
      assert_eq!(block, FIPS_PLAINTEXT, "{} backend", backend.name);
    }
  }

  /// Every backend that this host can run matches an independent AES-128 on generated inputs.
  #[test]
  fn every_backend_matches_independent_aes_on_generated_inputs() {
    let mut state = 0x7175_6963_2d6c_6221u64;
    let runnable: alloc::vec::Vec<_> = backends().into_iter().filter(|backend| (backend.available)()).collect();
    for _ in 0..256 {
      let key = generated(&mut state);
      let ciphertext = generated(&mut state);
      let expected = oracle(&key, &ciphertext);
      for backend in &runnable {
        let mut block = ciphertext;
        (backend.decrypt)(&key, &mut block);
        assert_eq!(block, expected, "{} backend", backend.name);
      }
    }
  }

  /// The inverse S-box of FIPS 197 Figure 14.
  #[rustfmt::skip]
  const FIPS_INV_SBOX: [u8; 256] = [
    0x52,0x09,0x6a,0xd5,0x30,0x36,0xa5,0x38,0xbf,0x40,0xa3,0x9e,0x81,0xf3,0xd7,0xfb,
    0x7c,0xe3,0x39,0x82,0x9b,0x2f,0xff,0x87,0x34,0x8e,0x43,0x44,0xc4,0xde,0xe9,0xcb,
    0x54,0x7b,0x94,0x32,0xa6,0xc2,0x23,0x3d,0xee,0x4c,0x95,0x0b,0x42,0xfa,0xc3,0x4e,
    0x08,0x2e,0xa1,0x66,0x28,0xd9,0x24,0xb2,0x76,0x5b,0xa2,0x49,0x6d,0x8b,0xd1,0x25,
    0x72,0xf8,0xf6,0x64,0x86,0x68,0x98,0x16,0xd4,0xa4,0x5c,0xcc,0x5d,0x65,0xb6,0x92,
    0x6c,0x70,0x48,0x50,0xfd,0xed,0xb9,0xda,0x5e,0x15,0x46,0x57,0xa7,0x8d,0x9d,0x84,
    0x90,0xd8,0xab,0x00,0x8c,0xbc,0xd3,0x0a,0xf7,0xe4,0x58,0x05,0xb8,0xb3,0x45,0x06,
    0xd0,0x2c,0x1e,0x8f,0xca,0x3f,0x0f,0x02,0xc1,0xaf,0xbd,0x03,0x01,0x13,0x8a,0x6b,
    0x3a,0x91,0x11,0x41,0x4f,0x67,0xdc,0xea,0x97,0xf2,0xcf,0xce,0xf0,0xb4,0xe6,0x73,
    0x96,0xac,0x74,0x22,0xe7,0xad,0x35,0x85,0xe2,0xf9,0x37,0xe8,0x1c,0x75,0xdf,0x6e,
    0x47,0xf1,0x1a,0x71,0x1d,0x29,0xc5,0x89,0x6f,0xb7,0x62,0x0e,0xaa,0x18,0xbe,0x1b,
    0xfc,0x56,0x3e,0x4b,0xc6,0xd2,0x79,0x20,0x9a,0xdb,0xc0,0xfe,0x78,0xcd,0x5a,0xf4,
    0x1f,0xdd,0xa8,0x33,0x88,0x07,0xc7,0x31,0xb1,0x12,0x10,0x59,0x27,0x80,0xec,0x5f,
    0x60,0x51,0x7f,0xa9,0x19,0xb5,0x4a,0x0d,0x2d,0xe5,0x7a,0x9f,0x93,0xc9,0x9c,0xef,
    0xa0,0xe0,0x3b,0x4d,0xae,0x2a,0xf5,0xb0,0xc8,0xeb,0xbb,0x3c,0x83,0x53,0x99,0x61,
    0x17,0x2b,0x04,0x7e,0xba,0x77,0xd6,0x26,0xe1,0x69,0x14,0x63,0x55,0x21,0x0c,0x7d,
  ];

  /// The bitsliced InvSubBytes matches FIPS 197 Figure 14 on all 256 inputs.
  #[test]
  fn bitsliced_inv_sub_bytes_matches_fips_197_exhaustively() {
    use crate::aead::aes_fixslice_round::{State, bitslice, inv_bitslice};

    for chunk in 0u8..4 {
      let mut blocks = [[0u8; 16]; 4];
      for (lane, block) in (0u8..).zip(&mut blocks) {
        for (position, byte) in (0u8..).zip(block) {
          *byte = chunk
            .strict_mul(64)
            .strict_add(lane.strict_mul(16))
            .strict_add(position);
        }
      }
      let mut state = State::default();
      bitslice(&mut state, &blocks[0], &blocks[1], &blocks[2], &blocks[3]);
      super::super::fixslice64::inv_sub_bytes(&mut state);
      for (input, output) in blocks.iter().flatten().zip(inv_bitslice(&state).iter().flatten()) {
        assert_eq!(*output, FIPS_INV_SBOX[usize::from(*input)], "input {input:#04x}");
      }
    }
  }

  /// The portable inverse S-box matches FIPS 197 Figure 14 on all 256 inputs.
  #[cfg(not(any(target_arch = "riscv64", target_arch = "s390x")))]
  #[test]
  fn portable_inv_sbox_matches_fips_197_exhaustively() {
    for (input, &expected) in (0u8..=u8::MAX).zip(&FIPS_INV_SBOX) {
      assert_eq!(inv_sbox(input), expected, "input {input:#04x}");
    }
  }
}
