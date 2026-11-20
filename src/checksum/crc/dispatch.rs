//! Kernel tiers, force controls, and run-time selection for every CRC.
//!
//! Each CRC selects one kernel per process. The kernel handles every input length itself, so
//! selection depends only on the CPU and the force request.

use super::{Algo, Hw, portable, reference};

/// A CRC kernel: the word-domain register after `data`.
pub(in crate::checksum) type Kernel<W> = fn(W, &[u8]) -> W;

/// Requested CRC kernel tier.
///
/// The request applies to one CRC type, from its environment variable. The selector clamps it to
/// the detected CPU: a tier that the CPU or the CRC cannot run becomes [`CrcForce::Auto`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum CrcForce {
  /// Select the fastest tier that the CPU supports.
  #[default]
  Auto,
  /// The bitwise reference implementation, the oracle for every other tier.
  Reference,
  /// The portable slice-by-16 tables.
  Portable,
  /// The CRC-32 instructions only: SSE4.2 for CRC-32C on x86-64, the CRC extension on AArch64.
  Hwcrc,
  /// x86-64 PCLMULQDQ folding, fused with the SSE4.2 instruction for CRC-32C.
  Pclmul,
  /// x86-64 VPCLMULQDQ folding with AVX-512, fused with the SSE4.2 instruction for CRC-32C.
  Vpclmul,
  /// AArch64 PMULL folding, fused with the CRC extension for CRC-32 and CRC-32C.
  Pmull,
  /// AArch64 PMULL folding with EOR3 (SHA3), fused with the CRC extension for CRC-32 and CRC-32C.
  PmullEor3,
  /// POWER VPMSUMD folding.
  Vpmsum,
  /// IBM Z VGFM folding.
  Vgfm,
  /// RISC-V Zbc scalar carryless-multiply folding.
  Zbc,
  /// RISC-V Zvbc vector carryless-multiply folding.
  Zvbc,
}

impl CrcForce {
  /// Return the stable selector name used by debug and configuration surfaces.
  #[must_use]
  pub const fn as_str(self) -> &'static str {
    match self {
      Self::Auto => "auto",
      Self::Reference => "reference",
      Self::Portable => "portable",
      Self::Hwcrc => "hwcrc",
      Self::Pclmul => "pclmul",
      Self::Vpclmul => "vpclmul",
      Self::Pmull => "pmull",
      Self::PmullEor3 => "pmull-eor3",
      Self::Vpmsum => "vpmsum",
      Self::Vgfm => "vgfm",
      Self::Zbc => "zbc",
      Self::Zvbc => "zvbc",
    }
  }
}

/// Resolved CRC force configuration.
///
/// `requested_force` is the request from the environment. `effective_force` is the same request
/// clamped to the CPU and the CRC.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CrcConfig {
  /// Requested force mode.
  pub requested_force: CrcForce,
  /// Force mode clamped to the detected CPU capabilities.
  pub effective_force: CrcForce,
}

/// Parses a force request. An unknown value selects [`CrcForce::Auto`].
#[cfg(feature = "std")]
fn parse(value: &str) -> CrcForce {
  const NAMES: [(&str, CrcForce); 18] = [
    ("auto", CrcForce::Auto),
    ("reference", CrcForce::Reference),
    ("bitwise", CrcForce::Reference),
    ("portable", CrcForce::Portable),
    ("scalar", CrcForce::Portable),
    ("table", CrcForce::Portable),
    ("hwcrc", CrcForce::Hwcrc),
    ("crc", CrcForce::Hwcrc),
    ("pclmul", CrcForce::Pclmul),
    ("vpclmul", CrcForce::Vpclmul),
    ("pmull", CrcForce::Pmull),
    ("pmull-eor3", CrcForce::PmullEor3),
    ("eor3", CrcForce::PmullEor3),
    ("vpmsum", CrcForce::Vpmsum),
    ("vgfm", CrcForce::Vgfm),
    ("zbc", CrcForce::Zbc),
    ("zvbc", CrcForce::Zvbc),
    ("clmul", BASE_CLMUL),
  ];
  let value = value.trim();
  NAMES
    .iter()
    .find(|(name, _)| value.eq_ignore_ascii_case(name))
    .map_or(CrcForce::Auto, |&(_, force)| force)
}

/// The tier that `clmul` names on this architecture.
#[cfg(feature = "std")]
const BASE_CLMUL: CrcForce = if cfg!(target_arch = "x86_64") {
  CrcForce::Pclmul
} else if cfg!(target_arch = "aarch64") {
  CrcForce::Pmull
} else if cfg!(target_arch = "powerpc64") {
  CrcForce::Vpmsum
} else if cfg!(target_arch = "s390x") {
  CrcForce::Vgfm
} else if cfg!(target_arch = "riscv64") {
  CrcForce::Zbc
} else {
  CrcForce::Auto
};

/// Accelerated tiers of this architecture, fastest first.
const TIERS: &[CrcForce] = &[
  #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
  CrcForce::Vpclmul,
  #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
  CrcForce::Pclmul,
  #[cfg(target_arch = "aarch64")]
  CrcForce::PmullEor3,
  #[cfg(target_arch = "aarch64")]
  CrcForce::Pmull,
  #[cfg(any(all(target_arch = "x86_64", target_feature = "sse2"), target_arch = "aarch64"))]
  CrcForce::Hwcrc,
  #[cfg(all(target_arch = "powerpc64", not(feature = "portable-only")))]
  CrcForce::Vpmsum,
  #[cfg(all(target_arch = "s390x", not(feature = "portable-only")))]
  CrcForce::Vgfm,
  #[cfg(all(target_arch = "riscv64", not(feature = "portable-only")))]
  CrcForce::Zvbc,
  #[cfg(all(target_arch = "riscv64", not(feature = "portable-only")))]
  CrcForce::Zbc,
];

/// Returns `true` when this CPU can run `tier` for a CRC with instruction support `hw`.
pub(in crate::checksum) fn available(tier: CrcForce, hw: Hw) -> bool {
  let _ = hw;
  match tier {
    CrcForce::Reference | CrcForce::Portable => true,
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    CrcForce::Pclmul => crate::platform::caps().has(crate::platform::caps::x86::PCLMUL_READY),
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    CrcForce::Vpclmul => crate::platform::caps().has(crate::platform::caps::x86::VPCLMUL_READY),
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    CrcForce::Hwcrc => hw == Hw::Crc32c && crate::platform::caps().has(crate::platform::caps::x86::CRC32C_READY),
    #[cfg(target_arch = "aarch64")]
    CrcForce::Pmull => crate::platform::caps().has(crate::platform::caps::aarch64::PMULL_READY),
    #[cfg(target_arch = "aarch64")]
    CrcForce::PmullEor3 => crate::platform::caps().has(crate::platform::caps::aarch64::PMULL_EOR3_READY),
    #[cfg(target_arch = "aarch64")]
    CrcForce::Hwcrc => hw != Hw::None && crate::platform::caps().has(crate::platform::caps::aarch64::CRC_READY),
    #[cfg(all(target_arch = "powerpc64", not(feature = "portable-only")))]
    CrcForce::Vpmsum => crate::platform::caps().has(crate::platform::caps::power::VPMSUM_READY),
    #[cfg(all(target_arch = "s390x", not(feature = "portable-only")))]
    CrcForce::Vgfm => crate::platform::caps().has(crate::platform::caps::s390x::VECTOR),
    #[cfg(all(target_arch = "riscv64", not(feature = "portable-only")))]
    CrcForce::Zbc => crate::platform::caps().has(crate::platform::caps::riscv::ZBC),
    #[cfg(all(target_arch = "riscv64", not(feature = "portable-only")))]
    CrcForce::Zvbc => {
      crate::platform::caps().has(crate::platform::caps::riscv::V.union(crate::platform::caps::riscv::ZVBC))
    }
    _ => false,
  }
}

/// Returns `true` when the CRCs without a CRC-32 instruction run an accelerated kernel.
#[cfg(feature = "diag")]
#[must_use]
pub fn is_hardware_accelerated() -> bool {
  tier(CrcForce::Auto, Hw::None) != CrcForce::Portable
}

/// The configuration of a CRC whose force variable is `env`.
pub(in crate::checksum) fn config(env: &str, hw: Hw) -> CrcConfig {
  #[cfg(feature = "std")]
  let requested_force = std::env::var(env).map_or(CrcForce::Auto, |value| parse(&value));
  #[cfg(not(feature = "std"))]
  let requested_force = {
    let _ = env;
    CrcForce::Auto
  };
  let effective_force = if requested_force != CrcForce::Auto && available(requested_force, hw) {
    requested_force
  } else {
    CrcForce::Auto
  };
  CrcConfig {
    requested_force,
    effective_force,
  }
}

/// The tier that runs for an effective force.
pub(in crate::checksum) fn tier(effective: CrcForce, hw: Hw) -> CrcForce {
  if effective != CrcForce::Auto {
    return effective;
  }
  TIERS
    .iter()
    .copied()
    .find(|&tier| available(tier, hw))
    .unwrap_or(CrcForce::Portable)
}

/// The kernel of `tier` for CRC `A`. The caller has checked that the CPU runs `tier`.
pub(in crate::checksum) fn kernel<A: Algo>(tier: CrcForce) -> Kernel<A::W> {
  match tier {
    CrcForce::Reference => reference_kernel::<A>,
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    CrcForce::Pclmul | CrcForce::Vpclmul | CrcForce::Hwcrc => super::x86_64::kernel::<A>(tier),
    #[cfg(target_arch = "aarch64")]
    CrcForce::Pmull | CrcForce::PmullEor3 | CrcForce::Hwcrc => super::aarch64::kernel::<A>(tier),
    #[cfg(all(target_arch = "powerpc64", not(feature = "portable-only")))]
    CrcForce::Vpmsum => super::power::vpmsum_kernel::<A>,
    #[cfg(all(target_arch = "s390x", not(feature = "portable-only")))]
    CrcForce::Vgfm => super::s390x::vgfm_kernel::<A>,
    #[cfg(all(target_arch = "riscv64", not(feature = "portable-only")))]
    CrcForce::Zbc => super::riscv64::zbc_kernel::<A>,
    #[cfg(all(target_arch = "riscv64", not(feature = "portable-only")))]
    CrcForce::Zvbc => super::riscv64::zvbc_kernel::<A>,
    _ => portable_kernel::<A>,
  }
}

/// A `combine` multiplier: an engine-domain value times `x^(8 * len)`.
pub(in crate::checksum) type Shift = fn(u64, usize) -> u64;

/// The `combine` multiplier of `tier` for CRC `A`. The caller has checked that the CPU runs `tier`.
pub(in crate::checksum) fn shift<A: Algo>(tier: CrcForce) -> Shift {
  match tier {
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    CrcForce::Pclmul | CrcForce::Vpclmul => super::x86_64::shift_kernel::<A>,
    #[cfg(target_arch = "aarch64")]
    CrcForce::Pmull | CrcForce::PmullEor3 => super::aarch64::shift_kernel::<A>,
    #[cfg(all(target_arch = "powerpc64", not(feature = "portable-only")))]
    CrcForce::Vpmsum => super::power::shift_kernel::<A>,
    #[cfg(all(target_arch = "s390x", not(feature = "portable-only")))]
    CrcForce::Vgfm => super::s390x::shift_kernel::<A>,
    #[cfg(all(target_arch = "riscv64", not(feature = "portable-only")))]
    CrcForce::Zbc => super::riscv64::shift_zbc_kernel::<A>,
    #[cfg(all(target_arch = "riscv64", not(feature = "portable-only")))]
    CrcForce::Zvbc => super::riscv64::shift_zvbc_kernel::<A>,
    _ => shift_portable::<A>,
  }
}

/// Portable `combine` multiplier.
fn shift_portable<A: Algo>(value: u64, len: usize) -> u64 {
  let crc = A::CRC;
  super::shift_by(value, len, &crc.powers, |value, power| {
    crc.multiply_portable(value, power)
  })
}

/// The stable kernel name of `tier`.
pub(in crate::checksum) const fn tier_name(tier: CrcForce) -> &'static str {
  match tier {
    CrcForce::Reference => "reference/bitwise",
    CrcForce::Auto | CrcForce::Portable => "portable/slice16",
    CrcForce::Hwcrc if cfg!(target_arch = "x86_64") => "x86_64/hwcrc",
    CrcForce::Hwcrc => "aarch64/hwcrc",
    CrcForce::Pclmul => "x86_64/pclmul",
    CrcForce::Vpclmul => "x86_64/vpclmul",
    CrcForce::Pmull => "aarch64/pmull",
    CrcForce::PmullEor3 => "aarch64/pmull-eor3",
    CrcForce::Vpmsum => "power/vpmsum",
    CrcForce::Vgfm => "s390x/vgfm",
    CrcForce::Zbc => "riscv64/zbc",
    CrcForce::Zvbc => "riscv64/zvbc",
  }
}

/// The name of the code that `tier` runs for an input of `len` bytes.
pub(in crate::checksum) fn kernel_name(tier: CrcForce, hw: Hw, len: usize) -> &'static str {
  match tier {
    CrcForce::Auto | CrcForce::Reference | CrcForce::Portable | CrcForce::Hwcrc => tier_name(tier),
    _ if len < 16 && hw != Hw::None && available(CrcForce::Hwcrc, hw) => tier_name(CrcForce::Hwcrc),
    _ if len < 16 => tier_name(CrcForce::Portable),
    _ => tier_name(tier),
  }
}

/// Portable tier.
pub(in crate::checksum) fn portable_kernel<A: Algo>(register: A::W, data: &[u8]) -> A::W {
  portable::update(A::CRC, register, data)
}

/// Reference tier.
fn reference_kernel<A: Algo>(register: A::W, data: &[u8]) -> A::W {
  let crc = A::CRC;
  crc.register_to_word(reference(&crc.spec, crc.word_to_register(register), data))
}

engine! {
  if {
    /// Runs a 64-bit-domain folding engine for CRC `A`, with the portable kernel below 16 bytes.
    ///
    /// # Safety
    ///
    /// The CPU must support the instructions of `reflected` and `msb`.
    #[inline(always)]
    pub(in crate::checksum) unsafe fn fold_with<A: Algo>(
      register: A::W,
      data: &[u8],
      reflected: unsafe fn(u64, &[u8], &super::Keys) -> u64,
      msb: unsafe fn(u64, &[u8], &super::Keys) -> u64,
    ) -> A::W {
      if data.len() < 16 {
        return portable::update(A::CRC, register, data);
      }
      let crc = A::CRC;
      let state = crc.word_to_domain(register);
      // SAFETY: the caller guarantees the instructions, and `data` holds at least 16 bytes.
      let state = unsafe {
        if crc.spec.reflected {
          reflected(state, data, &crc.keys)
        } else {
          msb(state, data, &crc.keys)
        }
      };
      crc.domain_to_word(state)
    }
  }
}
