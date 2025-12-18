//! CPU capability detection and representation.
//!
//! This module provides a unified capability model for all supported architectures.
//! It answers the question: "What instructions can I legally run on this machine?"
//!
//! # Design
//!
//! `CpuCaps` is a compact bitset representing available CPU features. Each bit
//! corresponds to a specific ISA extension or instruction set. The bits are
//! architecture-specific but the API is uniform.
//!
//! # Usage
//!
//! ```ignore
//! use platform::caps::{CpuCaps, X86_64_VPCLMUL, X86_64_PCLMUL};
//!
//! let caps = platform::get().0;
//! if caps.has(X86_64_VPCLMUL) {
//!     // Use AVX-512 VPCLMULQDQ path
//! } else if caps.has(X86_64_PCLMUL) {
//!     // Use PCLMULQDQ path
//! }
//! ```

/// 256-bit feature bitset.
///
/// This provides enough room for all ISA features we care about across all
/// architectures. Each architecture uses a different region of the bitset.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Bits256(pub [u64; 4]);

impl Bits256 {
  /// Empty bitset (no features).
  pub const NONE: Self = Self([0; 4]);

  /// Check if all bits in `other` are set in `self`.
  #[inline]
  #[must_use]
  pub const fn contains(self, other: Self) -> bool {
    (self.0[0] & other.0[0]) == other.0[0]
      && (self.0[1] & other.0[1]) == other.0[1]
      && (self.0[2] & other.0[2]) == other.0[2]
      && (self.0[3] & other.0[3]) == other.0[3]
  }

  /// Union of two bitsets.
  #[inline]
  #[must_use]
  pub const fn union(self, other: Self) -> Self {
    Self([
      self.0[0] | other.0[0],
      self.0[1] | other.0[1],
      self.0[2] | other.0[2],
      self.0[3] | other.0[3],
    ])
  }

  /// Intersection of two bitsets.
  #[inline]
  #[must_use]
  pub const fn intersection(self, other: Self) -> Self {
    Self([
      self.0[0] & other.0[0],
      self.0[1] & other.0[1],
      self.0[2] & other.0[2],
      self.0[3] & other.0[3],
    ])
  }

  /// Create a bitset with a single bit set.
  #[inline]
  #[must_use]
  pub const fn from_bit(bit: u16) -> Self {
    let word = (bit / 64) as usize;
    let bit_in_word = bit % 64;
    let mut bits = [0u64; 4];
    bits[word] = 1u64 << bit_in_word;
    Self(bits)
  }

  /// Check if a specific bit is set.
  #[inline]
  #[must_use]
  pub const fn has_bit(self, bit: u16) -> bool {
    let word = (bit / 64) as usize;
    let bit_in_word = bit % 64;
    (self.0[word] & (1u64 << bit_in_word)) != 0
  }

  /// Check if the bitset is empty.
  #[inline]
  #[must_use]
  pub const fn is_empty(self) -> bool {
    self.0[0] == 0 && self.0[1] == 0 && self.0[2] == 0 && self.0[3] == 0
  }
}

impl core::ops::BitOr for Bits256 {
  type Output = Self;

  #[inline]
  fn bitor(self, rhs: Self) -> Self::Output {
    self.union(rhs)
  }
}

impl core::ops::BitAnd for Bits256 {
  type Output = Self;

  #[inline]
  fn bitand(self, rhs: Self) -> Self::Output {
    self.intersection(rhs)
  }
}

impl core::ops::BitOrAssign for Bits256 {
  #[inline]
  fn bitor_assign(&mut self, rhs: Self) {
    *self = self.union(rhs);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Architecture identification
// ─────────────────────────────────────────────────────────────────────────────

/// Target architecture enumeration.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Arch {
  X86_64,
  X86,
  Aarch64,
  Arm,
  Riscv64,
  Riscv32,
  Powerpc64,
  LoongArch64,
  Wasm32,
  Wasm64,
  #[default]
  Other,
}

impl Arch {
  /// Get the architecture for the current compilation target.
  #[inline]
  #[must_use]
  pub const fn current() -> Self {
    #[cfg(target_arch = "x86_64")]
    {
      Self::X86_64
    }
    #[cfg(target_arch = "x86")]
    {
      Self::X86
    }
    #[cfg(target_arch = "aarch64")]
    {
      Self::Aarch64
    }
    #[cfg(target_arch = "arm")]
    {
      Self::Arm
    }
    #[cfg(target_arch = "riscv64")]
    {
      Self::Riscv64
    }
    #[cfg(target_arch = "riscv32")]
    {
      Self::Riscv32
    }
    #[cfg(target_arch = "powerpc64")]
    {
      Self::Powerpc64
    }
    #[cfg(target_arch = "loongarch64")]
    {
      Self::LoongArch64
    }
    #[cfg(target_arch = "wasm32")]
    {
      Self::Wasm32
    }
    #[cfg(target_arch = "wasm64")]
    {
      Self::Wasm64
    }
    #[cfg(not(any(
      target_arch = "x86_64",
      target_arch = "x86",
      target_arch = "aarch64",
      target_arch = "arm",
      target_arch = "riscv64",
      target_arch = "riscv32",
      target_arch = "powerpc64",
      target_arch = "loongarch64",
      target_arch = "wasm32",
      target_arch = "wasm64"
    )))]
    {
      Self::Other
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// CPU Capabilities
// ─────────────────────────────────────────────────────────────────────────────

/// CPU capabilities: what instructions can run on this machine.
///
/// This is the core type for capability-based dispatch. It combines:
/// - Architecture identification
/// - Feature bitset (ISA extensions)
///
/// # Thread Safety
///
/// `CpuCaps` is `Copy`, `Send`, and `Sync`. It can be freely shared across threads.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct CpuCaps {
  /// Target architecture.
  pub arch: Arch,
  /// Feature bits.
  pub bits: Bits256,
}

impl CpuCaps {
  /// No capabilities (portable baseline).
  pub const NONE: Self = Self {
    arch: Arch::Other,
    bits: Bits256::NONE,
  };

  /// Create capabilities for the current architecture with given bits.
  #[inline]
  #[must_use]
  pub const fn new(bits: Bits256) -> Self {
    Self {
      arch: Arch::current(),
      bits,
    }
  }

  /// Check if `self` has all the capabilities required by `required`.
  #[inline]
  #[must_use]
  pub const fn has(self, required: Bits256) -> bool {
    self.bits.contains(required)
  }

  /// Check if a specific feature bit is set.
  #[inline]
  #[must_use]
  pub const fn has_feature(self, bit: u16) -> bool {
    self.bits.has_bit(bit)
  }

  /// Union of capabilities.
  #[inline]
  #[must_use]
  pub const fn union(self, other: Self) -> Self {
    Self {
      arch: self.arch,
      bits: self.bits.union(other.bits),
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Feature bit definitions
// ─────────────────────────────────────────────────────────────────────────────
//
// Layout:
// - Bits 0-63:   x86/x86_64 features
// - Bits 64-127: aarch64/arm features
// - Bits 128-191: RISC-V features
// - Bits 192-255: Other architectures (wasm, powerpc, loongarch, etc.)

/// x86/x86_64 feature bits (bits 0-63).
pub mod x86 {
  use super::Bits256;

  // Basic SIMD
  pub const SSE2: Bits256 = Bits256::from_bit(0);
  pub const SSE3: Bits256 = Bits256::from_bit(1);
  pub const SSSE3: Bits256 = Bits256::from_bit(2);
  pub const SSE41: Bits256 = Bits256::from_bit(3);
  pub const SSE42: Bits256 = Bits256::from_bit(4);

  // AVX family
  pub const AVX: Bits256 = Bits256::from_bit(5);
  pub const AVX2: Bits256 = Bits256::from_bit(6);
  pub const FMA: Bits256 = Bits256::from_bit(7);

  // Crypto instructions
  pub const AESNI: Bits256 = Bits256::from_bit(8);
  pub const PCLMULQDQ: Bits256 = Bits256::from_bit(9);
  pub const SHA: Bits256 = Bits256::from_bit(10);

  // AVX-512 foundation
  pub const AVX512F: Bits256 = Bits256::from_bit(11);
  pub const AVX512VL: Bits256 = Bits256::from_bit(12);
  pub const AVX512BW: Bits256 = Bits256::from_bit(13);
  pub const AVX512DQ: Bits256 = Bits256::from_bit(14);
  pub const AVX512CD: Bits256 = Bits256::from_bit(15);

  // AVX-512 crypto/advanced
  pub const VPCLMULQDQ: Bits256 = Bits256::from_bit(16);
  pub const VAES: Bits256 = Bits256::from_bit(17);
  pub const GFNI: Bits256 = Bits256::from_bit(18);
  pub const AVX512IFMA: Bits256 = Bits256::from_bit(19);

  // AVX-512 additional
  pub const AVX512VBMI: Bits256 = Bits256::from_bit(20);
  pub const AVX512VBMI2: Bits256 = Bits256::from_bit(21);
  pub const AVX512VNNI: Bits256 = Bits256::from_bit(22);
  pub const AVX512BITALG: Bits256 = Bits256::from_bit(23);
  pub const AVX512VPOPCNTDQ: Bits256 = Bits256::from_bit(24);

  // Bit manipulation
  pub const BMI1: Bits256 = Bits256::from_bit(25);
  pub const BMI2: Bits256 = Bits256::from_bit(26);
  pub const POPCNT: Bits256 = Bits256::from_bit(27);
  pub const LZCNT: Bits256 = Bits256::from_bit(28);
  pub const ADX: Bits256 = Bits256::from_bit(29);

  // SHA extensions
  pub const SHA512: Bits256 = Bits256::from_bit(30);

  // Additional features
  pub const SSE4A: Bits256 = Bits256::from_bit(31); // AMD only
  pub const F16C: Bits256 = Bits256::from_bit(32);

  // AVX-512 additional subfeatures
  pub const AVX512VP2INTERSECT: Bits256 = Bits256::from_bit(33);
  pub const AVX512FP16: Bits256 = Bits256::from_bit(34);
  pub const AVX512BF16: Bits256 = Bits256::from_bit(35);

  // Combined capability masks for common feature sets
  /// PCLMULQDQ-ready: PCLMULQDQ + SSSE3
  pub const PCLMUL_READY: Bits256 = Bits256([PCLMULQDQ.0[0] | SSSE3.0[0], 0, 0, 0]);

  /// VPCLMULQDQ-ready: VPCLMULQDQ + AVX512F + AVX512VL + AVX512BW + PCLMULQDQ
  pub const VPCLMUL_READY: Bits256 = Bits256([
    VPCLMULQDQ.0[0] | AVX512F.0[0] | AVX512VL.0[0] | AVX512BW.0[0] | PCLMULQDQ.0[0],
    0,
    0,
    0,
  ]);
}

/// aarch64/arm feature bits (bits 64-127).
pub mod aarch64 {
  use super::Bits256;

  // Basic SIMD (NEON is baseline on AArch64)
  pub const NEON: Bits256 = Bits256::from_bit(64);

  // Crypto instructions
  pub const AES: Bits256 = Bits256::from_bit(65);
  pub const PMULL: Bits256 = Bits256::from_bit(66); // Often bundled with AES
  pub const SHA2: Bits256 = Bits256::from_bit(67);
  pub const SHA3: Bits256 = Bits256::from_bit(68); // Includes EOR3
  pub const SM3: Bits256 = Bits256::from_bit(69);
  pub const SM4: Bits256 = Bits256::from_bit(70);

  // CRC extension
  pub const CRC: Bits256 = Bits256::from_bit(71);

  // Additional SIMD
  pub const DOTPROD: Bits256 = Bits256::from_bit(72);
  pub const I8MM: Bits256 = Bits256::from_bit(73);
  pub const BF16: Bits256 = Bits256::from_bit(74);
  pub const FP16: Bits256 = Bits256::from_bit(75);

  // SVE family
  pub const SVE: Bits256 = Bits256::from_bit(76);
  pub const SVE2: Bits256 = Bits256::from_bit(77);
  pub const SVE2_AES: Bits256 = Bits256::from_bit(78);
  pub const SVE2_SHA3: Bits256 = Bits256::from_bit(79);
  pub const SVE2_SM4: Bits256 = Bits256::from_bit(80);
  pub const SVE2_BITPERM: Bits256 = Bits256::from_bit(81);

  // Combined capability masks
  /// PMULL-ready: AES (includes PMULL on most implementations)
  pub const PMULL_READY: Bits256 = Bits256([0, AES.0[1], 0, 0]);

  /// PMULL+EOR3-ready: AES + SHA3 (SHA3 provides EOR3)
  pub const PMULL_EOR3_READY: Bits256 = Bits256([0, AES.0[1] | SHA3.0[1], 0, 0]);

  /// CRC32C-ready: CRC extension
  pub const CRC_READY: Bits256 = Bits256([0, CRC.0[1], 0, 0]);
}

/// RISC-V feature bits (bits 128-191).
pub mod riscv {
  use super::Bits256;

  // Vector extension
  pub const V: Bits256 = Bits256::from_bit(128);

  // Bit manipulation
  pub const ZBB: Bits256 = Bits256::from_bit(129);
  pub const ZBS: Bits256 = Bits256::from_bit(130);
  pub const ZBA: Bits256 = Bits256::from_bit(131);
  pub const ZBC: Bits256 = Bits256::from_bit(132); // Carryless multiply

  // Scalar crypto
  pub const ZBKB: Bits256 = Bits256::from_bit(133);
  pub const ZBKC: Bits256 = Bits256::from_bit(134);
  pub const ZBKX: Bits256 = Bits256::from_bit(135);
  pub const ZKND: Bits256 = Bits256::from_bit(136); // AES decrypt
  pub const ZKNE: Bits256 = Bits256::from_bit(137); // AES encrypt
  pub const ZKNH: Bits256 = Bits256::from_bit(138); // SHA2
  pub const ZKSED: Bits256 = Bits256::from_bit(139); // SM4
  pub const ZKSH: Bits256 = Bits256::from_bit(140); // SM3

  // Vector crypto
  pub const ZVBB: Bits256 = Bits256::from_bit(141);
  pub const ZVBC: Bits256 = Bits256::from_bit(142);
  pub const ZVKB: Bits256 = Bits256::from_bit(143);
  pub const ZVKG: Bits256 = Bits256::from_bit(144);
  pub const ZVKNED: Bits256 = Bits256::from_bit(145);
  pub const ZVKNHA: Bits256 = Bits256::from_bit(146);
  pub const ZVKNHB: Bits256 = Bits256::from_bit(147);
  pub const ZVKSED: Bits256 = Bits256::from_bit(148);
  pub const ZVKSH: Bits256 = Bits256::from_bit(149);
}

/// WebAssembly feature bits (bits 192-207).
pub mod wasm {
  use super::Bits256;

  pub const SIMD128: Bits256 = Bits256::from_bit(192);
  pub const RELAXED_SIMD: Bits256 = Bits256::from_bit(193);
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_bits256_basic() {
    let empty = Bits256::NONE;
    assert!(empty.is_empty());

    let bit0 = Bits256::from_bit(0);
    assert!(!bit0.is_empty());
    assert!(bit0.has_bit(0));
    assert!(!bit0.has_bit(1));

    let bit64 = Bits256::from_bit(64);
    assert!(bit64.has_bit(64));
    assert!(!bit64.has_bit(0));
  }

  #[test]
  fn test_bits256_union_intersection() {
    let a = Bits256::from_bit(0);
    let b = Bits256::from_bit(1);
    let ab = a.union(b);

    assert!(ab.has_bit(0));
    assert!(ab.has_bit(1));
    assert!(!ab.has_bit(2));

    assert!(ab.contains(a));
    assert!(ab.contains(b));
    assert!(!a.contains(ab));
  }

  #[test]
  fn test_cpu_caps_has() {
    let caps = CpuCaps::new(x86::SSE42.union(x86::PCLMULQDQ));

    assert!(caps.has(x86::SSE42));
    assert!(caps.has(x86::PCLMULQDQ));
    assert!(!caps.has(x86::AVX512F));

    // Check combined requirement
    let pclmul_ready = x86::PCLMULQDQ.union(x86::SSSE3);
    assert!(!caps.has(pclmul_ready)); // Missing SSSE3
  }

  #[test]
  fn test_arch_current() {
    let arch = Arch::current();
    #[cfg(target_arch = "x86_64")]
    assert_eq!(arch, Arch::X86_64);
    #[cfg(target_arch = "aarch64")]
    assert_eq!(arch, Arch::Aarch64);
  }

  #[test]
  fn test_x86_feature_bits() {
    // Verify feature bits don't overlap
    assert_ne!(x86::SSE42, x86::PCLMULQDQ);
    assert_ne!(x86::AVX512F, x86::VPCLMULQDQ);

    // Verify combined masks
    let vpclmul_ready = x86::VPCLMUL_READY;
    assert!(vpclmul_ready.contains(x86::VPCLMULQDQ));
    assert!(vpclmul_ready.contains(x86::AVX512F));
    assert!(vpclmul_ready.contains(x86::AVX512VL));
    assert!(vpclmul_ready.contains(x86::AVX512BW));
    assert!(vpclmul_ready.contains(x86::PCLMULQDQ));
  }

  #[test]
  fn test_aarch64_feature_bits() {
    // Verify aarch64 bits are in the second word (bits 64-127)
    assert!(aarch64::NEON.0[1] != 0);
    assert!(aarch64::NEON.0[0] == 0);

    // Verify combined masks
    let pmull_eor3 = aarch64::PMULL_EOR3_READY;
    assert!(pmull_eor3.contains(aarch64::AES));
    assert!(pmull_eor3.contains(aarch64::SHA3));
  }
}
