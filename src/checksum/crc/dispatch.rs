//! CRC kernel selection from the process capabilities.

use super::{Algo, Hw, portable};

/// A CRC kernel: the word-domain register after `data`.
pub(in crate::checksum) type Kernel<W> = fn(W, &[u8]) -> W;

/// Kernels compiled for this target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::checksum) enum Tier {
  /// The portable slice-by-16 tables.
  Portable,
  /// The CRC-32 instructions only: SSE4.2 for CRC-32C on x86-64, the CRC extension on AArch64.
  #[cfg(any(all(target_arch = "x86_64", target_feature = "sse2"), target_arch = "aarch64"))]
  Hwcrc,
  /// x86-64 PCLMULQDQ folding, fused with the SSE4.2 instruction for CRC-32C.
  #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
  Pclmul,
  /// x86-64 VPCLMULQDQ folding with AVX-512, fused with the SSE4.2 instruction for CRC-32C.
  #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
  Vpclmul,
  /// AArch64 PMULL folding, fused with the CRC extension for CRC-32 and CRC-32C.
  #[cfg(target_arch = "aarch64")]
  Pmull,
  /// AArch64 PMULL folding with EOR3 (SHA3), fused with the CRC extension for CRC-32 and CRC-32C.
  #[cfg(target_arch = "aarch64")]
  PmullEor3,
  /// POWER VPMSUMD folding.
  #[cfg(all(target_arch = "powerpc64", not(feature = "portable-only")))]
  Vpmsum,
  /// IBM Z VGFM folding.
  #[cfg(all(target_arch = "s390x", not(feature = "portable-only")))]
  Vgfm,
  /// RISC-V Zbc scalar carryless-multiply folding.
  #[cfg(all(target_arch = "riscv64", not(feature = "portable-only")))]
  Zbc,
  /// RISC-V Zvbc vector carryless-multiply folding.
  #[cfg(all(target_arch = "riscv64", not(feature = "portable-only")))]
  Zvbc,
}

/// Accelerated tiers of this architecture, fastest first.
pub(super) const TIERS: &[Tier] = &[
  #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
  Tier::Vpclmul,
  #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
  Tier::Pclmul,
  #[cfg(target_arch = "aarch64")]
  Tier::PmullEor3,
  #[cfg(target_arch = "aarch64")]
  Tier::Pmull,
  #[cfg(any(all(target_arch = "x86_64", target_feature = "sse2"), target_arch = "aarch64"))]
  Tier::Hwcrc,
  #[cfg(all(target_arch = "powerpc64", not(feature = "portable-only")))]
  Tier::Vpmsum,
  #[cfg(all(target_arch = "s390x", not(feature = "portable-only")))]
  Tier::Vgfm,
  #[cfg(all(target_arch = "riscv64", not(feature = "portable-only")))]
  Tier::Zvbc,
  #[cfg(all(target_arch = "riscv64", not(feature = "portable-only")))]
  Tier::Zbc,
];

/// Returns `true` when this CPU can run `tier` for a CRC with instruction support `hw`.
pub(in crate::checksum) fn available(tier: Tier, hw: Hw) -> bool {
  let _ = hw;
  match tier {
    Tier::Portable => true,
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    Tier::Pclmul => crate::platform::caps().has(crate::platform::caps::x86::PCLMUL_READY),
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    Tier::Vpclmul => crate::platform::caps().has(crate::platform::caps::x86::VPCLMUL_READY),
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    Tier::Hwcrc => hw == Hw::Crc32c && crate::platform::caps().has(crate::platform::caps::x86::CRC32C_READY),
    #[cfg(target_arch = "aarch64")]
    Tier::Pmull => crate::platform::caps().has(crate::platform::caps::aarch64::PMULL_READY),
    #[cfg(target_arch = "aarch64")]
    Tier::PmullEor3 => crate::platform::caps().has(crate::platform::caps::aarch64::PMULL_EOR3_READY),
    #[cfg(target_arch = "aarch64")]
    Tier::Hwcrc => hw != Hw::None && crate::platform::caps().has(crate::platform::caps::aarch64::CRC_READY),
    #[cfg(all(target_arch = "powerpc64", not(feature = "portable-only")))]
    Tier::Vpmsum => crate::platform::caps().has(crate::platform::caps::power::VPMSUM_READY),
    #[cfg(all(target_arch = "s390x", not(feature = "portable-only")))]
    Tier::Vgfm => crate::platform::caps().has(crate::platform::caps::s390x::VECTOR),
    #[cfg(all(target_arch = "riscv64", not(feature = "portable-only")))]
    Tier::Zbc => crate::platform::caps().has(crate::platform::caps::riscv::ZBC),
    #[cfg(all(target_arch = "riscv64", not(feature = "portable-only")))]
    Tier::Zvbc => {
      crate::platform::caps().has(crate::platform::caps::riscv::V.union(crate::platform::caps::riscv::ZVBC))
    }
  }
}

/// Returns `true` when the CRCs without a CRC-32 instruction run an accelerated kernel.
#[must_use]
pub fn is_hardware_accelerated() -> bool {
  tier(Hw::None) != Tier::Portable
}

/// Select the fastest supported tier for this CRC.
pub(in crate::checksum) fn tier(hw: Hw) -> Tier {
  TIERS
    .iter()
    .copied()
    .find(|&tier| available(tier, hw))
    .unwrap_or(Tier::Portable)
}

/// The kernel of `tier` for CRC `A`. The caller has checked that the CPU runs `tier`.
pub(in crate::checksum) fn kernel<A: Algo>(tier: Tier) -> Kernel<A::W> {
  match tier {
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    Tier::Pclmul | Tier::Vpclmul | Tier::Hwcrc => super::x86_64::kernel::<A>(tier),
    #[cfg(target_arch = "aarch64")]
    Tier::Pmull | Tier::PmullEor3 | Tier::Hwcrc => super::aarch64::kernel::<A>(tier),
    #[cfg(all(target_arch = "powerpc64", not(feature = "portable-only")))]
    Tier::Vpmsum => super::power::vpmsum_kernel::<A>,
    #[cfg(all(target_arch = "s390x", not(feature = "portable-only")))]
    Tier::Vgfm => super::s390x::vgfm_kernel::<A>,
    #[cfg(all(target_arch = "riscv64", not(feature = "portable-only")))]
    Tier::Zbc => super::riscv64::zbc_kernel::<A>,
    #[cfg(all(target_arch = "riscv64", not(feature = "portable-only")))]
    Tier::Zvbc => super::riscv64::zvbc_kernel::<A>,
    Tier::Portable => portable_kernel::<A>,
  }
}

/// A `combine` multiplier: an engine-domain value times `x^(8 * len)`.
pub(in crate::checksum) type Shift = fn(u64, usize) -> u64;

/// The `combine` multiplier of `tier` for CRC `A`. The caller has checked that the CPU runs `tier`.
pub(in crate::checksum) fn shift<A: Algo>(tier: Tier) -> Shift {
  match tier {
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    Tier::Pclmul | Tier::Vpclmul => super::x86_64::shift_kernel::<A>,
    #[cfg(target_arch = "aarch64")]
    Tier::Pmull | Tier::PmullEor3 => super::aarch64::shift_kernel::<A>,
    #[cfg(all(target_arch = "powerpc64", not(feature = "portable-only")))]
    Tier::Vpmsum => super::power::shift_kernel::<A>,
    #[cfg(all(target_arch = "s390x", not(feature = "portable-only")))]
    Tier::Vgfm => super::s390x::shift_kernel::<A>,
    #[cfg(all(target_arch = "riscv64", not(feature = "portable-only")))]
    Tier::Zbc => super::riscv64::shift_zbc_kernel::<A>,
    #[cfg(all(target_arch = "riscv64", not(feature = "portable-only")))]
    Tier::Zvbc => super::riscv64::shift_zvbc_kernel::<A>,
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
pub(in crate::checksum) const fn tier_name(tier: Tier) -> &'static str {
  match tier {
    Tier::Portable => "portable/slice16",
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    Tier::Hwcrc => "x86_64/hwcrc",
    #[cfg(target_arch = "aarch64")]
    Tier::Hwcrc => "aarch64/hwcrc",
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    Tier::Pclmul => "x86_64/pclmul",
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    Tier::Vpclmul => "x86_64/vpclmul",
    #[cfg(target_arch = "aarch64")]
    Tier::Pmull => "aarch64/pmull",
    #[cfg(target_arch = "aarch64")]
    Tier::PmullEor3 => "aarch64/pmull-eor3",
    #[cfg(all(target_arch = "powerpc64", not(feature = "portable-only")))]
    Tier::Vpmsum => "power/vpmsum",
    #[cfg(all(target_arch = "s390x", not(feature = "portable-only")))]
    Tier::Vgfm => "s390x/vgfm",
    #[cfg(all(target_arch = "riscv64", not(feature = "portable-only")))]
    Tier::Zbc => "riscv64/zbc",
    #[cfg(all(target_arch = "riscv64", not(feature = "portable-only")))]
    Tier::Zvbc => "riscv64/zvbc",
  }
}

/// Portable tier.
pub(in crate::checksum) fn portable_kernel<A: Algo>(register: A::W, data: &[u8]) -> A::W {
  portable::update(A::CRC, register, data)
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
      use super::Word;

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
          // The MSB-first engine stores the register at the high end of its 64-bit word.
          msb(state, data, &crc.keys) >> (64u32.strict_sub(A::W::BITS))
        }
      };
      A::W::narrow(state)
    }
  }
}
