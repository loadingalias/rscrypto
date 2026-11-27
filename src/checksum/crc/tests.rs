//! Engine tests over arbitrary CRC specifications.
//!
//! Every specification runs through the same portable and accelerated code, so random widths,
//! polynomials, and bit orders test the engines beyond the shipped CRCs. The oracle is the bitwise
//! [`reference`], which itself matches the published check values.

use alloc::{boxed::Box, vec::Vec};

use super::{Crc, SLICES, Spec, Word, combine_powers, mask, wide_table};

/// Bitwise reference CRC: the register after `data`, from `register`, both in the specification's
/// form.
///
/// This follows the definition one bit at a time and shares no table or constant with the fast
/// paths, so it is their test oracle.
#[must_use]
fn reference(spec: &Spec, register: u64, data: &[u8]) -> u64 {
  let width_mask = mask(spec.width);
  let mut reg = register & width_mask;
  let mut i = 0usize;
  if spec.reflected {
    let poly = spec.poly.reverse_bits() >> (64u32.strict_sub(spec.width));
    while i < data.len() {
      reg ^= data[i] as u64;
      let mut bit = 0u32;
      while bit < 8 {
        reg = if reg & 1 == 1 { (reg >> 1) ^ poly } else { reg >> 1 };
        bit = bit.strict_add(1);
      }
      i = i.strict_add(1);
    }
  } else {
    let top = 1u64 << (spec.width.strict_sub(1));
    while i < data.len() {
      reg ^= (data[i] as u64) << (spec.width.strict_sub(8));
      let mut bit = 0u32;
      while bit < 8 {
        reg = if reg & top != 0 {
          ((reg << 1) ^ spec.poly) & width_mask
        } else {
          (reg << 1) & width_mask
        };
        bit = bit.strict_add(1);
      }
      i = i.strict_add(1);
    }
  }
  reg
}

impl<W: Word> Crc<W> {
  /// The word-domain register for a register value in the specification's form.
  fn register_to_word(&self, register: u64) -> W {
    let shift = if self.spec.reflected {
      0
    } else {
      W::BITS.strict_sub(self.spec.width)
    };
    W::narrow((register & mask(self.spec.width)) << shift)
  }

  /// The register value in the specification's form for a word-domain register.
  fn word_to_register(&self, word: W) -> u64 {
    self.spec.register(self.word_to_domain(word))
  }
}

/// The shipped CRCs with their published check values over `"123456789"`.
const CATALOG: [(Spec, u64); 7] = [
  (Spec::new(16, 0x1021, true, 0xFFFF, 0xFFFF), 0x906E),
  (Spec::new(16, 0x8005, true, 0, 0), 0xBB3D),
  (Spec::new(24, 0x86_4CFB, false, 0xB7_04CE, 0), 0x21_CF02),
  (Spec::new(32, 0x04C1_1DB7, true, 0xFFFF_FFFF, 0xFFFF_FFFF), 0xCBF4_3926),
  (Spec::new(32, 0x1EDC_6F41, true, 0xFFFF_FFFF, 0xFFFF_FFFF), 0xE306_9283),
  (
    Spec::new(64, 0x42F0_E1EB_A9EA_3693, true, !0, !0),
    0x995D_C9BB_DF19_39FA,
  ),
  (
    Spec::new(64, 0xAD93_D235_94C9_3659, true, !0, !0),
    0xAE8B_1486_0A79_9888,
  ),
];

/// Further CRC RevEng catalog entries with an MSB-first or unusual width.
const MORE: [(Spec, u64); 6] = [
  // CRC-8/SMBUS
  (Spec::new(8, 0x07, false, 0, 0), 0xF4),
  // CRC-16/XMODEM
  (Spec::new(16, 0x1021, false, 0, 0), 0x31C3),
  // CRC-32/BZIP2
  (Spec::new(32, 0x04C1_1DB7, false, 0xFFFF_FFFF, 0xFFFF_FFFF), 0xFC89_1918),
  // CRC-40/GSM
  (Spec::new(40, 0x00_0482_0009, false, 0, 0xFF_FFFF_FFFF), 0xD4_164F_C646),
  // CRC-64/ECMA-182
  (Spec::new(64, 0x42F0_E1EB_A9EA_3693, false, 0, 0), 0x6C40_DF5F_0B49_7347),
  // CRC-12/UMTS is refout without refin, so CRC-12/DECT stands in: MSB-first, width 12.
  (Spec::new(12, 0x80F, false, 0, 0), 0xF5B),
];

fn check(spec: &Spec, data: &[u8]) -> u64 {
  reference(spec, spec.init, data) ^ spec.xorout
}

#[test]
fn reference_matches_published_check_values() {
  for (spec, expected) in CATALOG.iter().chain(&MORE) {
    assert_eq!(check(spec, b"123456789"), *expected, "{spec:?}");
  }
}

/// A small deterministic generator, so failures reproduce.
struct Rng(u64);

impl Rng {
  fn next(&mut self) -> u64 {
    self.0 ^= self.0 << 13;
    self.0 ^= self.0 >> 7;
    self.0 ^= self.0 << 17;
    self.0
  }

  fn bytes(&mut self, len: usize) -> Vec<u8> {
    (0..len).map(|_| self.next().to_le_bytes()[0]).collect()
  }
}

fn random_spec(rng: &mut Rng, width: u32) -> Spec {
  let mask = super::mask(width);
  Spec::new(
    width,
    (rng.next() & mask) | 1,
    rng.next() & 1 == 1,
    rng.next() & mask,
    rng.next() & mask,
  )
}

/// Runtime copy of `crc_data!` for any specification.
fn data<W: Word>(spec: Spec) -> Box<Crc<W>> {
  let wide = wide_table::<SLICES>(&spec, W::BITS);
  let mut table = Box::new([[W::narrow(0); 256]; SLICES]);
  for (row, wide_row) in table.iter_mut().zip(&wide) {
    for (entry, &value) in row.iter_mut().zip(wide_row) {
      *entry = W::narrow(value);
    }
  }
  Box::new(Crc::new(spec, *table, combine_powers(&spec)))
}

fn specs() -> Vec<Spec> {
  let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
  let mut specs: Vec<Spec> = CATALOG.iter().chain(&MORE).map(|(spec, _)| *spec).collect();
  for width in [8, 9, 15, 16, 17, 23, 24, 31, 32, 33, 40, 47, 48, 55, 63, 64] {
    for _ in 0..3 {
      specs.push(random_spec(&mut rng, width));
    }
  }
  specs
}

/// Lengths around every block, lane, and tail boundary of the engines.
fn lengths() -> Vec<usize> {
  let mut lengths: Vec<usize> = (0..=300).collect();
  for base in [511usize, 1023, 1024, 2048, 4095, 4096] {
    lengths.extend(base.strict_sub(17)..base.strict_add(18));
  }
  lengths
}

fn portable_matches<W: Word>(spec: Spec, rng: &mut Rng) {
  let crc = data::<W>(spec);
  let input = rng.bytes(4200);
  for len in lengths() {
    for offset in [0usize, 1, 7] {
      let part = &input[offset..offset.strict_add(len)];
      let start = rng.next() & super::mask(spec.width);
      let got = crc.word_to_register(super::portable::update(&crc, crc.register_to_word(start), part));
      assert_eq!(got, reference(&spec, start, part), "{spec:?} len={len} offset={offset}");
    }
  }
}

#[test]
fn portable_matches_reference() {
  let mut rng = Rng(1);
  for spec in specs() {
    match spec.width {
      ..=16 => portable_matches::<u16>(spec, &mut rng),
      17..=32 => portable_matches::<u32>(spec, &mut rng),
      _ => portable_matches::<u64>(spec, &mut rng),
    }
  }
}

/// The `combine` multipliers that this CPU runs, as `(name, reflected, multiplier)`.
type Shift<'a> = Box<dyn Fn(u64, usize, &Crc<u64>) -> u64 + 'a>;

fn shifts() -> Vec<(&'static str, Shift<'static>)> {
  // Each architecture's block extends the list, so no binding needs `mut` where none exists.
  let shifts: Vec<(&'static str, Shift<'static>)> = Vec::new();
  #[cfg(target_arch = "aarch64")]
  let shifts = {
    let mut shifts = shifts;
    if crate::platform::caps().has(crate::platform::caps::aarch64::PMULL_READY) {
      shifts.push((
        "pmull",
        Box::new(|value, len, crc: &Crc<u64>| {
          // SAFETY: PMULL support is checked above.
          unsafe {
            if crc.spec.reflected {
              super::aarch64::shift::<true>(value, len, &crc.powers, &crc.keys)
            } else {
              super::aarch64::shift::<false>(value, len, &crc.powers, &crc.keys)
            }
          }
        }),
      ));
    }
    shifts
  };
  #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
  let shifts = {
    let mut shifts = shifts;
    if crate::platform::caps().has(crate::platform::caps::x86::PCLMUL_READY) {
      shifts.push((
        "pclmul",
        Box::new(|value, len, crc: &Crc<u64>| {
          // SAFETY: PCLMULQDQ support is checked above.
          unsafe {
            if crc.spec.reflected {
              super::x86_64::shift::<true>(value, len, &crc.powers, &crc.keys)
            } else {
              super::x86_64::shift::<false>(value, len, &crc.powers, &crc.keys)
            }
          }
        }),
      ));
    }
    shifts
  };
  #[cfg(all(target_arch = "powerpc64", not(feature = "portable-only")))]
  let shifts = {
    let mut shifts = shifts;
    if crate::platform::caps().has(crate::platform::caps::power::VPMSUM_READY) {
      shifts.push((
        "vpmsum",
        Box::new(|value, len, crc: &Crc<u64>| {
          // SAFETY: POWER8 vector crypto support is checked above.
          unsafe {
            if crc.spec.reflected {
              super::power::shift::<true>(value, len, &crc.powers, &crc.keys)
            } else {
              super::power::shift::<false>(value, len, &crc.powers, &crc.keys)
            }
          }
        }),
      ));
    }
    shifts
  };
  #[cfg(all(target_arch = "s390x", not(feature = "portable-only")))]
  let shifts = {
    let mut shifts = shifts;
    if crate::platform::caps().has(crate::platform::caps::s390x::VECTOR) {
      shifts.push((
        "vgfm",
        Box::new(|value, len, crc: &Crc<u64>| {
          // SAFETY: vector facility support is checked above.
          unsafe {
            if crc.spec.reflected {
              super::s390x::shift::<true>(value, len, &crc.powers, &crc.keys)
            } else {
              super::s390x::shift::<false>(value, len, &crc.powers, &crc.keys)
            }
          }
        }),
      ));
    }
    shifts
  };
  #[cfg(all(target_arch = "riscv64", not(feature = "portable-only")))]
  let shifts = {
    let mut shifts = shifts;
    use crate::platform::caps::riscv;
    let caps = crate::platform::caps();
    if caps.has(riscv::ZBC) {
      shifts.push((
        "zbc",
        Box::new(|value, len, crc: &Crc<u64>| {
          // SAFETY: Zbc support is checked above.
          unsafe {
            if crc.spec.reflected {
              super::riscv64::shift_zbc::<true>(value, len, &crc.powers, &crc.keys)
            } else {
              super::riscv64::shift_zbc::<false>(value, len, &crc.powers, &crc.keys)
            }
          }
        }),
      ));
    }
    if caps.has(riscv::V.union(riscv::ZVBC)) {
      shifts.push((
        "zvbc",
        Box::new(|value, len, crc: &Crc<u64>| {
          // SAFETY: V and Zvbc support is checked above.
          unsafe {
            if crc.spec.reflected {
              super::riscv64::shift_zvbc::<true>(value, len, &crc.powers, &crc.keys)
            } else {
              super::riscv64::shift_zvbc::<false>(value, len, &crc.powers, &crc.keys)
            }
          }
        }),
      ));
    }
    shifts
  };
  shifts
}

#[test]
fn combine_matches_reference() {
  let mut rng = Rng(2);
  let input = rng.bytes(3000);
  let accelerated = shifts();
  for spec in specs() {
    let crc = data::<u64>(spec);
    for (split, total) in [(0, 0), (0, 5), (5, 5), (3, 100), (1000, 3000), (17, 2049), (0, 2047)] {
      let (a, b) = input[..total].split_at(split);
      let expected = check(&spec, &input[..total]);
      let portable = |value, len| super::shift_by(value, len, &crc.powers, |v, p| crc.multiply_portable(v, p));
      let combined = crc.combine(check(&spec, a), check(&spec, b), b.len(), portable);
      assert_eq!(combined, expected, "portable: {spec:?} split={split} total={total}");
      for (name, shift) in &accelerated {
        let shift = |value, len| shift(value, len, &crc);
        let combined = crc.combine(check(&spec, a), check(&spec, b), b.len(), shift);
        assert_eq!(combined, expected, "{name}: {spec:?} split={split} total={total}");
      }
    }
  }
}

/// Long lengths use the high power entries; direct exponentiation of `x` checks them.
#[test]
fn combine_long_lengths_match_direct_exponentiation() {
  let accelerated = shifts();
  for spec in specs() {
    let crc = data::<u64>(spec);
    let g = spec.poly64();
    for len in [1usize << 20, (1 << 33) + 5, usize::MAX >> 4] {
      let value = spec.domain(0x5A5A_5A5A_5A5A_5A5A & super::mask(spec.width));
      // Normal form: aligned value times x^(8 len) modulo G'.
      let normal = if spec.reflected { value.reverse_bits() } else { value };
      let product = super::mulmod(normal, super::xpow((len as u64).strict_mul(8), g), g);
      let expected = if spec.reflected {
        product.reverse_bits()
      } else {
        product
      };
      let portable = super::shift_by(value, len, &crc.powers, |v, p| crc.multiply_portable(v, p));
      assert_eq!(portable, expected, "portable: {spec:?} len={len}");
      for (name, shift) in &accelerated {
        assert_eq!(shift(value, len, &crc), expected, "{name}: {spec:?} len={len}");
      }
    }
  }
}

engine! {
  if {
    /// A folding engine under test: `(reflected, state, data, keys)` to the new state.
    type Engine<'a> = dyn Fn(bool, u64, &[u8], &super::Keys) -> u64 + 'a;

    /// Checks an accelerated engine over every specification, length, alignment, and state.
    fn engine_matches(name: &str, engine: &Engine<'_>) {
      let mut rng = Rng(3);
      let input = rng.bytes(4200);
      for spec in specs() {
        let keys = super::Keys::new(&spec);
        for len in lengths().into_iter().filter(|&len| len >= 16) {
          for offset in [0usize, 1, 7, 8] {
            let part = &input[offset..offset.strict_add(len)];
            let start = rng.next() & super::mask(spec.width);
            let got = spec.register(engine(spec.reflected, spec.domain(start), part, &keys));
            assert_eq!(
              got,
              reference(&spec, start, part),
              "{name}: {spec:?} len={len} offset={offset}"
            );
          }
        }
      }
    }
  }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn aarch64_engines_match_reference() {
  use crate::platform::caps::aarch64::{PMULL_EOR3_READY, PMULL_READY};
  let caps = crate::platform::caps();
  if caps.has(PMULL_READY) {
    engine_matches("pmull", &|reflected, state, data, keys| {
      // SAFETY: PMULL support is checked above, and every length is at least 16.
      unsafe {
        if reflected {
          super::aarch64::pmull::<true>(state, data, keys)
        } else {
          super::aarch64::pmull::<false>(state, data, keys)
        }
      }
    });
  }
  if caps.has(PMULL_EOR3_READY) {
    engine_matches("pmull-eor3", &|reflected, state, data, keys| {
      // SAFETY: PMULL and EOR3 support is checked above, and every length is at least 16.
      unsafe {
        if reflected {
          super::aarch64::pmull_eor3::<true>(state, data, keys)
        } else {
          super::aarch64::pmull_eor3::<false>(state, data, keys)
        }
      }
    });
  }
}

#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
#[test]
fn x86_64_engines_match_reference() {
  use crate::platform::caps::x86::{PCLMUL_READY, VPCLMUL_READY};
  let caps = crate::platform::caps();
  if caps.has(PCLMUL_READY) {
    engine_matches("pclmul", &|reflected, state, data, keys| {
      // SAFETY: PCLMULQDQ support is checked above, and every length is at least 16.
      unsafe {
        if reflected {
          super::x86_64::pclmul::<true>(state, data, keys)
        } else {
          super::x86_64::pclmul::<false>(state, data, keys)
        }
      }
    });
  }
  if caps.has(VPCLMUL_READY) {
    engine_matches("vpclmul", &|reflected, state, data, keys| {
      // SAFETY: VPCLMULQDQ support is checked above, and every length is at least 16.
      unsafe {
        if reflected {
          super::x86_64::vpclmul::<true>(state, data, keys)
        } else {
          super::x86_64::vpclmul::<false>(state, data, keys)
        }
      }
    });
  }
}

#[cfg(all(target_arch = "powerpc64", not(feature = "portable-only")))]
#[test]
fn power_engine_matches_reference() {
  if crate::platform::caps().has(crate::platform::caps::power::VPMSUM_READY) {
    engine_matches("vpmsum", &|reflected, state, data, keys| {
      // SAFETY: POWER8 vector crypto support is checked above, and every length is at least 16.
      unsafe {
        if reflected {
          super::power::vpmsum::<true>(state, data, keys)
        } else {
          super::power::vpmsum::<false>(state, data, keys)
        }
      }
    });
  }
}

#[cfg(all(target_arch = "s390x", not(feature = "portable-only")))]
#[test]
fn s390x_engine_matches_reference() {
  if crate::platform::caps().has(crate::platform::caps::s390x::VECTOR) {
    engine_matches("vgfm", &|reflected, state, data, keys| {
      // SAFETY: vector facility support is checked above, and every length is at least 16.
      unsafe {
        if reflected {
          super::s390x::vgfm::<true>(state, data, keys)
        } else {
          super::s390x::vgfm::<false>(state, data, keys)
        }
      }
    });
  }
}

#[cfg(all(target_arch = "riscv64", not(feature = "portable-only")))]
#[test]
fn riscv64_engines_match_reference() {
  use crate::platform::caps::riscv;
  let caps = crate::platform::caps();
  if caps.has(riscv::ZBC) {
    engine_matches("zbc", &|reflected, state, data, keys| {
      // SAFETY: Zbc support is checked above, and every length is at least 16.
      unsafe {
        if reflected {
          super::riscv64::zbc::<true>(state, data, keys)
        } else {
          super::riscv64::zbc::<false>(state, data, keys)
        }
      }
    });
  }
  if caps.has(riscv::V.union(riscv::ZVBC)) {
    engine_matches("zvbc", &|reflected, state, data, keys| {
      // SAFETY: V and Zvbc support is checked above, and every length is at least 16.
      unsafe {
        if reflected {
          super::riscv64::zvbc::<true>(state, data, keys)
        } else {
          super::riscv64::zvbc::<false>(state, data, keys)
        }
      }
    });
  }
}

engine! {
  if {
    /// The derived CRC-64/XZ keys equal the constants that TiKV's `crc64fast` publishes.
    #[test]
    fn crc64_xz_keys_match_published_constants() {
      let keys = super::Keys::new(&CATALOG[5].0);
      assert_eq!(keys.poly, 0x92d8_af2b_af0e_1e85);
      assert_eq!(keys.mu, 0x9c3e_466c_1729_63d5);
      // 128-byte fold: (K_1087, K_1023) as (lo, hi).
      assert_eq!(keys.fold[7], [0x8757_d71d_4fcc_1000, 0xd7d8_6b2a_f73d_e740]);
      // 16-byte fold: (K_191, K_127).
      assert_eq!(keys.fold[0], [0xe05d_d497_ca39_3ae4, 0xdabe_95af_c787_5f40]);
    }
  }
}

/// Every tier that this CPU runs, for every shipped CRC, against the reference.
///
/// This covers the CRC-32 instruction kernels and their fusion with folding, whose rounds start
/// at 1408 bytes and double.
fn tiers_match<A: super::Algo>(name: &str) {
  use super::dispatch::{TIERS, Tier, available, kernel};
  let crc = A::CRC;
  let mut rng = Rng(4);
  let input = rng.bytes(90_000);
  let mut lengths: Vec<usize> = (0..=300).collect();
  for base in [1408usize, 2816, 4224, 5632, 11264, 22528, 45056, 90_000 - 40] {
    lengths.extend(base.strict_sub(17)..base.strict_add(18).min(90_000));
  }
  for tier in TIERS
    .iter()
    .copied()
    .chain([Tier::Portable])
    .filter(|&tier| available(tier, A::HW))
  {
    let update = kernel::<A>(tier);
    for &len in &lengths {
      let part = &input[1..len.strict_add(1)];
      let start = rng.next() & super::mask(crc.spec.width);
      let got = crc.word_to_register(update(crc.register_to_word(start), part));
      assert_eq!(got, reference(&crc.spec, start, part), "{name} {tier:?} len={len}");
    }
  }
}

#[test]
fn every_tier_of_every_crc_matches_reference() {
  #[cfg(feature = "crc16")]
  {
    tiers_match::<crate::checksum::Crc16Ccitt>("crc16-ccitt");
    tiers_match::<crate::checksum::Crc16Ibm>("crc16-ibm");
  }
  #[cfg(feature = "crc24")]
  tiers_match::<crate::checksum::Crc24OpenPgp>("crc24-openpgp");
  #[cfg(feature = "crc32")]
  {
    tiers_match::<crate::checksum::Crc32>("crc32");
    tiers_match::<crate::checksum::Crc32C>("crc32c");
  }
  #[cfg(feature = "crc64")]
  {
    tiers_match::<crate::checksum::Crc64>("crc64-xz");
    tiers_match::<crate::checksum::Crc64Nvme>("crc64-nvme");
  }
}
