//! CPU capability detection and representation.
//!
//! This module provides a unified capability model for all supported architectures.
//! It answers the question: "What instructions can I legally run on this machine?"
//!
//! # Design
//!
//! [`Caps`] is a 256-bit bitset representing available CPU features. Each bit
//! corresponds to a specific ISA extension. The bits are architecture-specific
//! but the API is uniform across all targets.
//!
//! # Bit Layout
//!
//! - Bits 0-63: x86/x86_64 features
//! - Bits 64-127: aarch64/arm features
//! - Bits 128-191: RISC-V features
//! - Bits 192-255: WebAssembly and other architectures
//!
//! # Usage
//!
//! ```ignore
//! use platform::{caps, Caps};
//! use platform::caps::x86;
//!
//! let c = caps();
//! if c.has(x86::VPCLMUL_READY) {
//!     // Use AVX-512 VPCLMULQDQ path
//! } else if c.has(x86::PCLMUL_READY) {
//!     // Use PCLMULQDQ path
//! }
//! ```

// alloc is only needed for tests (feature_names iteration with Vec)
#[cfg(test)]
extern crate alloc;

// ─────────────────────────────────────────────────────────────────────────────
// Core Capability Type
// ─────────────────────────────────────────────────────────────────────────────

/// CPU capabilities: a 256-bit feature bitset.
///
/// This is the core type for capability-based dispatch. Use [`has()`](Caps::has)
/// to check if required features are available.
///
/// # Thread Safety
///
/// `Caps` is `Copy`, `Send`, and `Sync`. It can be freely shared across threads.
///
/// # Example
///
/// ```ignore
/// use platform::caps::{Caps, x86};
///
/// let caps = platform::caps();
/// if caps.has(x86::VPCLMUL_READY) {
///     // Use VPCLMULQDQ kernel
/// }
/// ```
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Caps(pub(crate) [u64; 4]);

impl Caps {
  /// Empty capability set (no features).
  pub const NONE: Self = Self([0; 4]);

  /// Create a capability set from raw words.
  ///
  /// This is primarily useful for testing and fuzzing.
  /// Normal usage should prefer the predefined constants.
  ///
  /// # Availability
  ///
  /// This function is only available when the `testing` feature is enabled
  /// or in test builds. Enable it in Cargo.toml:
  /// ```toml
  /// [dependencies]
  /// platform = { version = "...", features = ["testing"] }
  /// ```
  #[cfg(any(test, feature = "testing"))]
  #[inline]
  #[must_use]
  pub const fn from_raw(words: [u64; 4]) -> Self {
    Self(words)
  }

  /// Access the raw underlying words.
  ///
  /// Returns the four 64-bit words that make up the capability bitset.
  ///
  /// # Availability
  ///
  /// This function is only available when the `testing` feature is enabled
  /// or in test builds.
  #[cfg(any(test, feature = "testing"))]
  #[inline]
  #[must_use]
  pub const fn as_raw(&self) -> &[u64; 4] {
    &self.0
  }

  /// Check if all features in `required` are present.
  ///
  /// This is the core dispatch check, marked `#[inline(always)]` for zero overhead.
  ///
  /// # Example
  ///
  /// ```ignore
  /// if caps.has(x86::VPCLMUL_READY) {
  ///     // VPCLMULQDQ + AVX512F + AVX512VL + AVX512BW are all available
  /// }
  /// ```
  #[inline(always)]
  #[must_use]
  pub const fn has(self, required: Self) -> bool {
    (self.0[0] & required.0[0]) == required.0[0]
      && (self.0[1] & required.0[1]) == required.0[1]
      && (self.0[2] & required.0[2]) == required.0[2]
      && (self.0[3] & required.0[3]) == required.0[3]
  }

  /// Union of two capability sets.
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

  /// Intersection of two capability sets.
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

  /// Check if the capability set is empty.
  #[inline]
  #[must_use]
  pub const fn is_empty(self) -> bool {
    self.0[0] == 0 && self.0[1] == 0 && self.0[2] == 0 && self.0[3] == 0
  }

  /// Count the number of features present.
  #[inline]
  #[must_use]
  pub const fn count(self) -> u32 {
    self.0[0].count_ones() + self.0[1].count_ones() + self.0[2].count_ones() + self.0[3].count_ones()
  }

  /// Create a capability set with a single bit set.
  ///
  /// Bit must be 0-255 (enforced by type system via u8).
  #[inline]
  #[must_use]
  pub const fn bit(bit: u8) -> Self {
    // u8 is always < 256, so no assert needed
    let word = (bit / 64) as usize;
    let bit_in_word = bit % 64;
    // Use match instead of indexing to satisfy const evaluation
    let mut bits = [0u64; 4];
    match word {
      0 => bits[0] = 1u64 << bit_in_word,
      1 => bits[1] = 1u64 << bit_in_word,
      2 => bits[2] = 1u64 << bit_in_word,
      _ => bits[3] = 1u64 << bit_in_word,
    }
    Self(bits)
  }

  /// Check if a specific bit is set.
  #[inline]
  #[must_use]
  pub const fn has_bit(self, bit: u8) -> bool {
    let word = (bit / 64) as usize;
    let bit_in_word = bit % 64;
    let bits_word = match word {
      0 => self.0[0],
      1 => self.0[1],
      2 => self.0[2],
      _ => self.0[3],
    };
    (bits_word & (1u64 << bit_in_word)) != 0
  }
}

impl core::ops::BitOr for Caps {
  type Output = Self;

  #[inline]
  fn bitor(self, rhs: Self) -> Self::Output {
    self.union(rhs)
  }
}

impl core::ops::BitAnd for Caps {
  type Output = Self;

  #[inline]
  fn bitand(self, rhs: Self) -> Self::Output {
    self.intersection(rhs)
  }
}

impl core::ops::BitOrAssign for Caps {
  #[inline]
  fn bitor_assign(&mut self, rhs: Self) {
    *self = self.union(rhs);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Architecture Identification
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
  S390x,
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
    #[cfg(target_arch = "s390x")]
    {
      Self::S390x
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
      target_arch = "s390x",
      target_arch = "loongarch64",
      target_arch = "wasm32",
      target_arch = "wasm64"
    )))]
    {
      Self::Other
    }
  }

  /// Returns the human-readable name for this architecture.
  #[inline]
  #[must_use]
  pub const fn name(self) -> &'static str {
    match self {
      Self::X86_64 => "x86_64",
      Self::X86 => "x86",
      Self::Aarch64 => "aarch64",
      Self::Arm => "arm",
      Self::Riscv64 => "riscv64",
      Self::Riscv32 => "riscv32",
      Self::Powerpc64 => "powerpc64",
      Self::S390x => "s390x",
      Self::LoongArch64 => "loongarch64",
      Self::Wasm32 => "wasm32",
      Self::Wasm64 => "wasm64",
      Self::Other => "other",
    }
  }
}

impl core::fmt::Display for Arch {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.write_str(self.name())
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// x86/x86_64 Features (bits 0-63)
// ─────────────────────────────────────────────────────────────────────────────

/// x86/x86_64 CPU features.
///
/// Includes SSE, AVX, AVX-512, crypto extensions, and bit manipulation.
pub mod x86 {
  use super::Caps;

  // ─── SSE Family ───
  pub const SSE2: Caps = Caps::bit(0);
  pub const SSE3: Caps = Caps::bit(1);
  pub const SSSE3: Caps = Caps::bit(2);
  pub const SSE41: Caps = Caps::bit(3);
  pub const SSE42: Caps = Caps::bit(4);
  pub const SSE4A: Caps = Caps::bit(5); // AMD only

  // ─── AVX Family ───
  pub const AVX: Caps = Caps::bit(6);
  pub const AVX2: Caps = Caps::bit(7);
  pub const FMA: Caps = Caps::bit(8);
  pub const F16C: Caps = Caps::bit(9);

  // ─── Crypto Extensions ───
  pub const AESNI: Caps = Caps::bit(10);
  pub const PCLMULQDQ: Caps = Caps::bit(11);
  pub const SHA: Caps = Caps::bit(12);
  pub const SHA512: Caps = Caps::bit(13);

  // ─── AVX-512 Foundation ───
  pub const AVX512F: Caps = Caps::bit(14);
  pub const AVX512VL: Caps = Caps::bit(15);
  pub const AVX512BW: Caps = Caps::bit(16);
  pub const AVX512DQ: Caps = Caps::bit(17);
  pub const AVX512CD: Caps = Caps::bit(18);

  // ─── AVX-512 Crypto/Advanced ───
  pub const VPCLMULQDQ: Caps = Caps::bit(19);
  pub const VAES: Caps = Caps::bit(20);
  pub const GFNI: Caps = Caps::bit(21);

  // ─── AVX-512 Extended ───
  pub const AVX512IFMA: Caps = Caps::bit(22);
  pub const AVX512VBMI: Caps = Caps::bit(23);
  pub const AVX512VBMI2: Caps = Caps::bit(24);
  pub const AVX512VNNI: Caps = Caps::bit(25);
  pub const AVX512BITALG: Caps = Caps::bit(26);
  pub const AVX512VPOPCNTDQ: Caps = Caps::bit(27);
  pub const AVX512VP2INTERSECT: Caps = Caps::bit(28);
  pub const AVX512FP16: Caps = Caps::bit(29);
  pub const AVX512BF16: Caps = Caps::bit(30);

  // ─── Bit Manipulation ───
  pub const BMI1: Caps = Caps::bit(31);
  pub const BMI2: Caps = Caps::bit(32);
  pub const POPCNT: Caps = Caps::bit(33);
  pub const LZCNT: Caps = Caps::bit(34);
  pub const ADX: Caps = Caps::bit(35);

  // ─── AVX10 (unified AVX-512 replacement) ───
  pub const AVX10_1: Caps = Caps::bit(36);
  pub const AVX10_2: Caps = Caps::bit(37);

  // ─── AMX (Advanced Matrix Extensions) ───
  pub const AMX_TILE: Caps = Caps::bit(38);
  pub const AMX_BF16: Caps = Caps::bit(39);
  pub const AMX_INT8: Caps = Caps::bit(40);
  pub const AMX_FP16: Caps = Caps::bit(41);
  pub const AMX_COMPLEX: Caps = Caps::bit(42);

  // ─── Miscellaneous ───
  pub const MOVDIRI: Caps = Caps::bit(43);
  pub const MOVDIR64B: Caps = Caps::bit(44);
  pub const SERIALIZE: Caps = Caps::bit(45);
  pub const RDRAND: Caps = Caps::bit(46);
  pub const RDSEED: Caps = Caps::bit(47);

  // ─── APX (Advanced Performance Extensions) ───
  pub const APX: Caps = Caps::bit(48);

  // ─── Combined Capability Masks ───
  // These represent common feature combinations for dispatch decisions.

  /// PCLMULQDQ-ready: PCLMULQDQ + SSSE3 (for aligned loads)
  pub const PCLMUL_READY: Caps = Caps([PCLMULQDQ.0[0] | SSSE3.0[0], 0, 0, 0]);

  /// VPCLMULQDQ-ready: VPCLMULQDQ + AVX512F + AVX512VL + AVX512BW (+ base PCLMULQDQ).
  ///
  /// Note: CRC folding kernels may still rely on the 128-bit `pclmulqdq`
  /// instruction for tail reduction even when using VPCLMUL for the main loop.
  pub const VPCLMUL_READY: Caps = Caps([
    VPCLMULQDQ.0[0] | PCLMULQDQ.0[0] | AVX512F.0[0] | AVX512VL.0[0] | AVX512BW.0[0],
    0,
    0,
    0,
  ]);

  /// VAES-ready: VAES + AVX512F + AVX512VL
  pub const VAES_READY: Caps = Caps([VAES.0[0] | AVX512F.0[0] | AVX512VL.0[0], 0, 0, 0]);

  /// AVX2-ready: AVX2 (common baseline for modern SIMD)
  pub const AVX2_READY: Caps = Caps([AVX2.0[0], 0, 0, 0]);

  /// AVX-512 foundation ready
  pub const AVX512_READY: Caps = Caps([AVX512F.0[0] | AVX512VL.0[0] | AVX512BW.0[0] | AVX512DQ.0[0], 0, 0, 0]);

  /// AMX-ready: AMX_TILE + AMX_BF16 + AMX_INT8
  pub const AMX_READY: Caps = Caps([AMX_TILE.0[0] | AMX_BF16.0[0] | AMX_INT8.0[0], 0, 0, 0]);
}

// ─────────────────────────────────────────────────────────────────────────────
// aarch64/arm Features (bits 64-127)
// ─────────────────────────────────────────────────────────────────────────────

/// aarch64/arm CPU features.
///
/// Includes NEON, crypto extensions, SVE, and SVE2.
pub mod aarch64 {
  use super::Caps;

  // ─── Basic SIMD ───
  pub const NEON: Caps = Caps::bit(64); // Baseline on AArch64

  // ─── Crypto Extensions ───
  pub const AES: Caps = Caps::bit(65);
  pub const PMULL: Caps = Caps::bit(66); // Often bundled with AES
  pub const SHA2: Caps = Caps::bit(67);
  pub const SHA3: Caps = Caps::bit(68); // Includes EOR3
  pub const SHA512: Caps = Caps::bit(69);
  pub const SM3: Caps = Caps::bit(70);
  pub const SM4: Caps = Caps::bit(71);

  // ─── CRC Extension ───
  pub const CRC: Caps = Caps::bit(72);

  // ─── Additional SIMD ───
  pub const DOTPROD: Caps = Caps::bit(73);
  pub const I8MM: Caps = Caps::bit(74);
  pub const BF16: Caps = Caps::bit(75);
  pub const FP16: Caps = Caps::bit(76);
  pub const FRINTTS: Caps = Caps::bit(77);

  // ─── SVE Family ───
  pub const SVE: Caps = Caps::bit(78);
  pub const SVE2: Caps = Caps::bit(79);
  pub const SVE2_AES: Caps = Caps::bit(80);
  pub const SVE2_SHA3: Caps = Caps::bit(81);
  pub const SVE2_SM4: Caps = Caps::bit(82);
  pub const SVE2_BITPERM: Caps = Caps::bit(83);
  pub const SVE2_PMULL: Caps = Caps::bit(97); // SVE2 PMULL
  pub const SVE2_I8MM: Caps = Caps::bit(98); // SVE2 Int8 matmul
  pub const SVE2_F32MM: Caps = Caps::bit(99); // SVE2 FP32 matmul
  pub const SVE2_F64MM: Caps = Caps::bit(100); // SVE2 FP64 matmul
  pub const SVE2_BF16: Caps = Caps::bit(101); // SVE2 BFloat16
  pub const SVE2_EBF16: Caps = Caps::bit(102); // SVE2 Extended BFloat16

  // ─── Atomics (ARMv8.1+) ───
  pub const LSE: Caps = Caps::bit(84); // Large System Extensions
  pub const LSE2: Caps = Caps::bit(85); // ARMv8.4 atomics

  // ─── Memory Operations ───
  pub const MOPS: Caps = Caps::bit(86); // FEAT_MOPS memcpy acceleration

  // ─── Scalable Matrix Extension ───
  pub const SME: Caps = Caps::bit(87);
  pub const SME2: Caps = Caps::bit(88);

  // ─── SME2p1 and extended SME features ───
  pub const SME2P1: Caps = Caps::bit(89); // SME version 2.1
  pub const SME_I16I64: Caps = Caps::bit(90); // SME Int16xInt64
  pub const SME_F64F64: Caps = Caps::bit(91); // SME Float64xFloat64
  pub const SME_B16B16: Caps = Caps::bit(92); // SME BFloat16xBFloat16 (Apple M5)
  pub const SME_F16F16: Caps = Caps::bit(93); // SME Float16xFloat16 (Apple M5)
  pub const SME_I8I32: Caps = Caps::bit(103); // SME Int8xInt32
  pub const SME_F16F32: Caps = Caps::bit(104); // SME Float16xFloat32
  pub const SME_B16F32: Caps = Caps::bit(105); // SME BFloat16xFloat32
  pub const SME_F32F32: Caps = Caps::bit(106); // SME Float32xFloat32
  pub const SME_FA64: Caps = Caps::bit(107); // SME Full A64
  pub const SME_I16I32: Caps = Caps::bit(108); // SME Int16xInt32
  pub const SME_BI32I32: Caps = Caps::bit(109); // SME BrainInt32xInt32
  pub const EBF16: Caps = Caps::bit(110); // Extended BFloat16 (NEON)

  // ─── SVE2.1 features ───
  pub const SVE2P1: Caps = Caps::bit(94); // SVE version 2.1
  pub const SVE_B16B16: Caps = Caps::bit(95); // SVE BFloat16xBFloat16

  // ─── Hardware RNG ───
  pub const RNG: Caps = Caps::bit(96); // RNDR/RNDRRS

  // ─── Combined Capability Masks ───

  /// AES + PMULL combined (ARM bundles these together)
  pub const AES_PMULL: Caps = Caps([0, AES.0[1] | PMULL.0[1], 0, 0]);

  /// SM3 + SM4 combined
  pub const SM3_SM4: Caps = Caps([0, SM3.0[1] | SM4.0[1], 0, 0]);

  /// PMULL-ready: AES (includes PMULL on most implementations)
  pub const PMULL_READY: Caps = Caps([0, AES.0[1], 0, 0]);

  /// PMULL+EOR3-ready: AES + SHA3 (SHA3 provides EOR3 for faster GHASH)
  pub const PMULL_EOR3_READY: Caps = Caps([0, AES.0[1] | SHA3.0[1], 0, 0]);

  /// CRC32C-ready: CRC extension
  pub const CRC_READY: Caps = Caps([0, CRC.0[1], 0, 0]);

  /// SVE2 crypto ready: SVE2 + SVE2_AES
  pub const SVE2_CRYPTO_READY: Caps = Caps([0, SVE2.0[1] | SVE2_AES.0[1], 0, 0]);
}

// ─────────────────────────────────────────────────────────────────────────────
// RISC-V Features (bits 128-191)
// ─────────────────────────────────────────────────────────────────────────────

/// RISC-V CPU features.
///
/// Includes vector extensions and scalar/vector crypto.
pub mod riscv {
  use super::Caps;

  // ─── Vector Extension ───
  pub const V: Caps = Caps::bit(128);

  // ─── Bit Manipulation ───
  pub const ZBB: Caps = Caps::bit(129);
  pub const ZBS: Caps = Caps::bit(130);
  pub const ZBA: Caps = Caps::bit(131);
  pub const ZBC: Caps = Caps::bit(132); // Carryless multiply

  // ─── Scalar Crypto ───
  pub const ZBKB: Caps = Caps::bit(133);
  pub const ZBKC: Caps = Caps::bit(134);
  pub const ZBKX: Caps = Caps::bit(135);
  pub const ZKND: Caps = Caps::bit(136); // AES decrypt
  pub const ZKNE: Caps = Caps::bit(137); // AES encrypt
  pub const ZKNH: Caps = Caps::bit(138); // SHA2
  pub const ZKSED: Caps = Caps::bit(139); // SM4
  pub const ZKSH: Caps = Caps::bit(140); // SM3

  // ─── Vector Crypto ───
  pub const ZVBB: Caps = Caps::bit(141);
  pub const ZVBC: Caps = Caps::bit(142);
  pub const ZVKB: Caps = Caps::bit(143);
  pub const ZVKG: Caps = Caps::bit(144);
  pub const ZVKNED: Caps = Caps::bit(145);
  pub const ZVKNHA: Caps = Caps::bit(146);
  pub const ZVKNHB: Caps = Caps::bit(147);
  pub const ZVKSED: Caps = Caps::bit(148);
  pub const ZVKSH: Caps = Caps::bit(149);
}

// ─────────────────────────────────────────────────────────────────────────────
// WebAssembly Features (bits 192-207)
// ─────────────────────────────────────────────────────────────────────────────

/// WebAssembly CPU features.
pub mod wasm {
  use super::Caps;

  pub const SIMD128: Caps = Caps::bit(192);
  pub const RELAXED_SIMD: Caps = Caps::bit(193);
}

// ─────────────────────────────────────────────────────────────────────────────
// LoongArch Features (bits 208-223)
// ─────────────────────────────────────────────────────────────────────────────

/// LoongArch CPU features.
///
/// Includes LSX (128-bit SIMD), LASX (256-bit SIMD), and various extensions.
pub mod loongarch {
  use super::Caps;

  // ─── SIMD Extensions ───
  /// LSX: 128-bit SIMD (baseline on 3A5000+, required for loongarch64-linux-gnu/musl)
  pub const LSX: Caps = Caps::bit(208);
  /// LASX: 256-bit Advanced SIMD (requires LSX)
  pub const LASX: Caps = Caps::bit(209);

  // ─── Floating-Point ───
  /// Floating-point support (baseline)
  pub const F: Caps = Caps::bit(210);
  /// Double-precision floating-point (baseline for LP64D ABI)
  pub const D: Caps = Caps::bit(211);
  /// FRECIPE: Floating-point reciprocal estimate instructions
  pub const FRECIPE: Caps = Caps::bit(212);

  // ─── Atomics ───
  /// LAM_BH: Atomic byte/halfword operations
  pub const LAM_BH: Caps = Caps::bit(213);
  /// LAMCAS: Atomic compare-and-swap
  pub const LAMCAS: Caps = Caps::bit(214);

  // ─── Memory ───
  /// UAL: Unaligned memory access support
  pub const UAL: Caps = Caps::bit(215);
  /// LD_SEQ_SA: Load-load sequencing (self-assert)
  pub const LD_SEQ_SA: Caps = Caps::bit(216);
  /// SCQ: 16-byte conditional store
  pub const SCQ: Caps = Caps::bit(217);

  // ─── Virtualization & Binary Translation ───
  /// LVZ: Virtualization extension
  pub const LVZ: Caps = Caps::bit(218);
  /// LBT: Binary translation extension
  pub const LBT: Caps = Caps::bit(219);

  // ─── Combined Capability Masks ───

  /// LSX-ready: 128-bit SIMD available
  pub const LSX_READY: Caps = Caps([0, 0, 0, LSX.0[3]]);

  /// LASX-ready: 256-bit SIMD available (implies LSX)
  pub const LASX_READY: Caps = Caps([0, 0, 0, LSX.0[3] | LASX.0[3]]);
}

// ─────────────────────────────────────────────────────────────────────────────
// s390x (IBM Z) Features (bits 224-239)
// ─────────────────────────────────────────────────────────────────────────────

/// s390x (IBM Z) CPU features.
///
/// Includes vector facilities introduced in z13 and enhanced in subsequent generations,
/// plus CPACF (CP Assist for Cryptographic Functions) extensions.
pub mod s390x {
  use super::Caps;

  // ─── Vector Facilities ───
  /// VECTOR: 128-bit SIMD (z13+, Facility 129)
  pub const VECTOR: Caps = Caps::bit(224);
  /// VECTOR_ENH1: Vector Enhancements 1 (z14+, Facility 135)
  pub const VECTOR_ENH1: Caps = Caps::bit(225);
  /// VECTOR_ENH2: Vector Enhancements 2 (z15+, Facility 148)
  pub const VECTOR_ENH2: Caps = Caps::bit(226);
  /// VECTOR_ENH3: Vector Enhancements 3 (z16+)
  pub const VECTOR_ENH3: Caps = Caps::bit(227);
  /// VECTOR_PD: Vector Packed Decimal (z14+, Facility 134)
  pub const VECTOR_PD: Caps = Caps::bit(228);
  /// NNP_ASSIST: Neural Network Processing Assist (z16+, Facility 165)
  pub const NNP_ASSIST: Caps = Caps::bit(229);

  // ─── Miscellaneous Extensions ───
  /// MISC_EXT2: Miscellaneous Extensions 2
  pub const MISC_EXT2: Caps = Caps::bit(230);
  /// MISC_EXT3: Miscellaneous Extensions 3
  pub const MISC_EXT3: Caps = Caps::bit(231);

  // ─── Crypto (CPACF - Message Security Assist) ───
  /// MSA: Message Security Assist (base crypto)
  pub const MSA: Caps = Caps::bit(232);
  /// MSA4: Message Security Assist Extension 4 (z196+)
  pub const MSA4: Caps = Caps::bit(233);
  /// MSA5: Message Security Assist Extension 5 (zEC12+)
  pub const MSA5: Caps = Caps::bit(234);
  /// MSA8: Message Security Assist Extension 8 (z14+)
  pub const MSA8: Caps = Caps::bit(235);
  /// MSA9: Message Security Assist Extension 9 (z15+)
  pub const MSA9: Caps = Caps::bit(236);

  // ─── Other Facilities ───
  /// DEFLATE: DEFLATE Conversion (z15+, Facility 151)
  pub const DEFLATE: Caps = Caps::bit(237);
  /// ENHANCED_SORT: Enhanced Sort (z15+, Facility 150)
  pub const ENHANCED_SORT: Caps = Caps::bit(238);

  // ─── Combined Capability Masks ───

  /// z13-ready: Base vector facility
  pub const Z13_READY: Caps = Caps([0, 0, 0, VECTOR.0[3]]);

  /// z14-ready: Vector + VE1 + Vector Packed Decimal
  pub const Z14_READY: Caps = Caps([0, 0, 0, VECTOR.0[3] | VECTOR_ENH1.0[3] | VECTOR_PD.0[3]]);

  /// z15-ready: Full z15 vector stack
  pub const Z15_READY: Caps = Caps([0, 0, 0, VECTOR.0[3] | VECTOR_ENH1.0[3] | VECTOR_ENH2.0[3]]);

  /// z16-ready: Full z16 with NNP assist
  pub const Z16_READY: Caps = Caps([
    0,
    0,
    0,
    VECTOR.0[3] | VECTOR_ENH1.0[3] | VECTOR_ENH2.0[3] | VECTOR_ENH3.0[3] | NNP_ASSIST.0[3],
  ]);
}

// ─────────────────────────────────────────────────────────────────────────────
// PowerPC64 Features (bits 240-255)
// ─────────────────────────────────────────────────────────────────────────────

/// PowerPC64 CPU features.
///
/// Includes AltiVec/VMX and VSX vector extensions, plus crypto accelerators.
pub mod powerpc64 {
  use super::Caps;

  // ─── Vector Extensions ───
  /// ALTIVEC: AltiVec/VMX 128-bit SIMD (G4+)
  pub const ALTIVEC: Caps = Caps::bit(240);
  /// VSX: Vector-Scalar Extension (POWER7+)
  pub const VSX: Caps = Caps::bit(241);
  /// POWER8_VECTOR: POWER8 vector instructions
  pub const POWER8_VECTOR: Caps = Caps::bit(242);
  /// POWER8_CRYPTO: POWER8 crypto (AES, SHA)
  pub const POWER8_CRYPTO: Caps = Caps::bit(243);
  /// POWER9_VECTOR: POWER9 vector instructions
  pub const POWER9_VECTOR: Caps = Caps::bit(244);
  /// POWER10_VECTOR: POWER10 vector instructions
  pub const POWER10_VECTOR: Caps = Caps::bit(245);

  // ─── Atomics ───
  /// QUADWORD_ATOMICS: 16-byte atomic operations (POWER8+)
  pub const QUADWORD_ATOMICS: Caps = Caps::bit(246);
  /// PARTWORD_ATOMICS: Byte/halfword atomic operations
  pub const PARTWORD_ATOMICS: Caps = Caps::bit(247);

  // ─── Combined Capability Masks ───

  /// POWER7-ready: VSX available
  pub const POWER7_READY: Caps = Caps([0, 0, 0, ALTIVEC.0[3] | VSX.0[3]]);

  /// POWER8-ready: Full POWER8 vector + crypto
  pub const POWER8_READY: Caps = Caps([
    0,
    0,
    0,
    ALTIVEC.0[3] | VSX.0[3] | POWER8_VECTOR.0[3] | POWER8_CRYPTO.0[3],
  ]);

  /// POWER9-ready: Full POWER9 vector stack
  pub const POWER9_READY: Caps = Caps([
    0,
    0,
    0,
    ALTIVEC.0[3] | VSX.0[3] | POWER8_VECTOR.0[3] | POWER9_VECTOR.0[3],
  ]);

  /// POWER10-ready: Full POWER10 (4x crypto engines)
  pub const POWER10_READY: Caps = Caps([
    0,
    0,
    0,
    ALTIVEC.0[3] | VSX.0[3] | POWER8_VECTOR.0[3] | POWER9_VECTOR.0[3] | POWER10_VECTOR.0[3],
  ]);

  /// VPMSUM-ready: POWER8 vector crypto (`vpmsumd`) + baseline vector support.
  ///
  /// This is the minimum feature set required for carry-less multiply based
  /// CRC folding on POWER8+.
  pub const VPMSUM_READY: Caps = Caps([
    0,
    0,
    0,
    ALTIVEC.0[3] | VSX.0[3] | POWER8_VECTOR.0[3] | POWER8_CRYPTO.0[3],
  ]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Feature Name Lookup (for diagnostics)
// ─────────────────────────────────────────────────────────────────────────────

/// Feature name entry: (bit_index, name).
type FeatureEntry = (u8, &'static str);

/// x86/x86_64 feature names.
const X86_FEATURES: &[FeatureEntry] = &[
  (0, "sse2"),
  (1, "sse3"),
  (2, "ssse3"),
  (3, "sse4.1"),
  (4, "sse4.2"),
  (5, "sse4a"),
  (6, "avx"),
  (7, "avx2"),
  (8, "fma"),
  (9, "f16c"),
  (10, "aes"),
  (11, "pclmulqdq"),
  (12, "sha"),
  (13, "sha512"),
  (14, "avx512f"),
  (15, "avx512vl"),
  (16, "avx512bw"),
  (17, "avx512dq"),
  (18, "avx512cd"),
  (19, "vpclmulqdq"),
  (20, "vaes"),
  (21, "gfni"),
  (22, "avx512ifma"),
  (23, "avx512vbmi"),
  (24, "avx512vbmi2"),
  (25, "avx512vnni"),
  (26, "avx512bitalg"),
  (27, "avx512vpopcntdq"),
  (28, "avx512vp2intersect"),
  (29, "avx512fp16"),
  (30, "avx512bf16"),
  (31, "bmi1"),
  (32, "bmi2"),
  (33, "popcnt"),
  (34, "lzcnt"),
  (35, "adx"),
  (36, "avx10.1"),
  (37, "avx10.2"),
  (38, "amx-tile"),
  (39, "amx-bf16"),
  (40, "amx-int8"),
  (41, "amx-fp16"),
  (42, "amx-complex"),
  (43, "movdiri"),
  (44, "movdir64b"),
  (45, "serialize"),
  (46, "rdrand"),
  (47, "rdseed"),
  (48, "apx"),
];

/// aarch64 feature names.
const AARCH64_FEATURES: &[FeatureEntry] = &[
  (64, "neon"),
  (65, "aes"),
  (66, "pmull"),
  (67, "sha2"),
  (68, "sha3"),
  (69, "sha512"),
  (70, "sm3"),
  (71, "sm4"),
  (72, "crc"),
  (73, "dotprod"),
  (74, "i8mm"),
  (75, "bf16"),
  (76, "fp16"),
  (77, "frintts"),
  (78, "sve"),
  (79, "sve2"),
  (80, "sve2-aes"),
  (81, "sve2-sha3"),
  (82, "sve2-sm4"),
  (83, "sve2-bitperm"),
  (84, "lse"),
  (85, "lse2"),
  (86, "mops"),
  (87, "sme"),
  (88, "sme2"),
  (89, "sme2p1"),
  (90, "sme-i16i64"),
  (91, "sme-f64f64"),
  (92, "sme-b16b16"),
  (93, "sme-f16f16"),
  (94, "sve2p1"),
  (95, "sve-b16b16"),
  (96, "rng"),
  (97, "sve2-pmull"),
  (98, "sve2-i8mm"),
  (99, "sve2-f32mm"),
  (100, "sve2-f64mm"),
  (101, "sve2-bf16"),
  (102, "sve2-ebf16"),
  (103, "sme-i8i32"),
  (104, "sme-f16f32"),
  (105, "sme-b16f32"),
  (106, "sme-f32f32"),
  (107, "sme-fa64"),
  (108, "sme-i16i32"),
  (109, "sme-bi32i32"),
  (110, "ebf16"),
];

/// RISC-V feature names.
const RISCV_FEATURES: &[FeatureEntry] = &[
  (128, "v"),
  (129, "zbb"),
  (130, "zbs"),
  (131, "zba"),
  (132, "zbc"),
  (133, "zbkb"),
  (134, "zbkc"),
  (135, "zbkx"),
  (136, "zknd"),
  (137, "zkne"),
  (138, "zknh"),
  (139, "zksed"),
  (140, "zksh"),
  (141, "zvbb"),
  (142, "zvbc"),
  (143, "zvkb"),
  (144, "zvkg"),
  (145, "zvkned"),
  (146, "zvknha"),
  (147, "zvknhb"),
  (148, "zvksed"),
  (149, "zvksh"),
];

/// WebAssembly feature names.
const WASM_FEATURES: &[FeatureEntry] = &[(192, "simd128"), (193, "relaxed-simd")];

/// LoongArch feature names.
const LOONGARCH_FEATURES: &[FeatureEntry] = &[
  (208, "lsx"),
  (209, "lasx"),
  (210, "f"),
  (211, "d"),
  (212, "frecipe"),
  (213, "lam-bh"),
  (214, "lamcas"),
  (215, "ual"),
  (216, "ld-seq-sa"),
  (217, "scq"),
  (218, "lvz"),
  (219, "lbt"),
];

/// s390x (IBM Z) feature names.
const S390X_FEATURES: &[FeatureEntry] = &[
  (224, "vector"),
  (225, "vector-enhancements-1"),
  (226, "vector-enhancements-2"),
  (227, "vector-enhancements-3"),
  (228, "vector-packed-decimal"),
  (229, "nnp-assist"),
  (230, "miscellaneous-extensions-2"),
  (231, "miscellaneous-extensions-3"),
  (232, "message-security-assist"),
  (233, "message-security-assist-extension4"),
  (234, "message-security-assist-extension5"),
  (235, "message-security-assist-extension8"),
  (236, "message-security-assist-extension9"),
  (237, "deflate-conversion"),
  (238, "enhanced-sort"),
];

/// PowerPC64 feature names.
const POWERPC64_FEATURES: &[FeatureEntry] = &[
  (240, "altivec"),
  (241, "vsx"),
  (242, "power8-vector"),
  (243, "power8-crypto"),
  (244, "power9-vector"),
  (245, "power10-vector"),
  (246, "quadword-atomics"),
  (247, "partword-atomics"),
];

impl Caps {
  /// Returns an iterator over the names of all set feature bits.
  pub fn feature_names(self) -> impl Iterator<Item = &'static str> {
    X86_FEATURES
      .iter()
      .chain(AARCH64_FEATURES.iter())
      .chain(RISCV_FEATURES.iter())
      .chain(WASM_FEATURES.iter())
      .chain(LOONGARCH_FEATURES.iter())
      .chain(S390X_FEATURES.iter())
      .chain(POWERPC64_FEATURES.iter())
      .filter_map(move |(bit, name)| if self.has_bit(*bit) { Some(*name) } else { None })
  }
}

impl core::fmt::Debug for Caps {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    let arch = Arch::current();
    write!(f, "Caps({}", arch)?;

    let mut iter = self.feature_names().peekable();
    if iter.peek().is_none() {
      write!(f, ", none)")
    } else {
      write!(f, ", [")?;
      let mut first = true;
      for name in iter {
        if !first {
          write!(f, ", ")?;
        }
        first = false;
        write!(f, "{name}")?;
      }
      write!(f, "])")
    }
  }
}

impl core::fmt::Display for Caps {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    core::fmt::Debug::fmt(self, f)
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_caps_basic() {
    let empty = Caps::NONE;
    assert!(empty.is_empty());
    assert_eq!(empty.count(), 0);

    let bit0 = Caps::bit(0);
    assert!(!bit0.is_empty());
    assert_eq!(bit0.count(), 1);
    assert!(bit0.has_bit(0));
    assert!(!bit0.has_bit(1));
  }

  #[test]
  fn test_caps_union_intersection() {
    let a = Caps::bit(0);
    let b = Caps::bit(1);
    let ab = a.union(b);

    assert!(ab.has_bit(0));
    assert!(ab.has_bit(1));
    assert!(!ab.has_bit(2));
    assert_eq!(ab.count(), 2);

    assert!(ab.has(a));
    assert!(ab.has(b));
    assert!(!a.has(ab));
  }

  #[test]
  fn test_caps_all_words() {
    // Test bits in each word
    let w0 = Caps::bit(0);
    let w1 = Caps::bit(64);
    let w2 = Caps::bit(128);
    let w3 = Caps::bit(192);

    assert_eq!(w0.0[0], 1);
    assert_eq!(w1.0[1], 1);
    assert_eq!(w2.0[2], 1);
    assert_eq!(w3.0[3], 1);

    let all = w0 | w1 | w2 | w3;
    assert!(all.has(w0));
    assert!(all.has(w1));
    assert!(all.has(w2));
    assert!(all.has(w3));
    assert_eq!(all.count(), 4);
  }

  #[test]
  fn test_x86_combined_masks() {
    // Verify VPCLMUL_READY contains expected features
    let vpclmul = x86::VPCLMUL_READY;
    assert!(vpclmul.has(x86::VPCLMULQDQ));
    assert!(vpclmul.has(x86::PCLMULQDQ));
    assert!(vpclmul.has(x86::AVX512F));
    assert!(vpclmul.has(x86::AVX512VL));
    assert!(vpclmul.has(x86::AVX512BW));
  }

  #[test]
  fn test_aarch64_combined_masks() {
    let pmull_eor3 = aarch64::PMULL_EOR3_READY;
    assert!(pmull_eor3.has(aarch64::AES));
    assert!(pmull_eor3.has(aarch64::SHA3));
  }

  #[test]
  fn test_feature_names() {
    let caps = x86::SSE42 | x86::PCLMULQDQ;
    let names: alloc::vec::Vec<_> = caps.feature_names().collect();
    assert!(names.contains(&"sse4.2"));
    assert!(names.contains(&"pclmulqdq"));
    assert!(!names.contains(&"avx512f"));
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
  fn test_operators() {
    let a = Caps::bit(0);
    let b = Caps::bit(1);

    assert_eq!(a | b, a.union(b));
    assert_eq!((a | b) & a, a);

    let mut c = a;
    c |= b;
    assert_eq!(c, a | b);
  }

  #[test]
  fn test_x86_new_features() {
    // Test new x86 feature constants are properly defined
    assert!(!x86::SHA512.is_empty());
    assert!(!x86::AVX10_1.is_empty());
    assert!(!x86::AVX10_2.is_empty());
    assert!(!x86::AVX512FP16.is_empty());
    assert!(!x86::MOVDIRI.is_empty());
    assert!(!x86::MOVDIR64B.is_empty());
    assert!(!x86::SERIALIZE.is_empty());
    assert!(!x86::AMX_FP16.is_empty());
    assert!(!x86::AMX_COMPLEX.is_empty());

    // Verify they have distinct bit positions
    let all =
      x86::SHA512 | x86::AVX10_1 | x86::AVX10_2 | x86::AVX512FP16 | x86::MOVDIRI | x86::MOVDIR64B | x86::SERIALIZE;
    assert_eq!(all.count(), 7);
  }

  #[test]
  fn test_aarch64_new_features() {
    // Test new aarch64 feature constants are properly defined
    assert!(!aarch64::SHA512.is_empty());
    assert!(!aarch64::LSE2.is_empty());
    assert!(!aarch64::MOPS.is_empty());
    assert!(!aarch64::SME.is_empty());
    assert!(!aarch64::SME2.is_empty());
    assert!(!aarch64::RNG.is_empty());

    // Verify they have distinct bit positions
    let all = aarch64::SHA512 | aarch64::LSE2 | aarch64::MOPS | aarch64::SME | aarch64::SME2 | aarch64::RNG;
    assert_eq!(all.count(), 6);
  }

  #[test]
  fn test_riscv_features() {
    use super::riscv;

    // Test RISC-V feature constants are properly defined
    assert!(!riscv::V.is_empty());
    assert!(!riscv::ZBB.is_empty());
    assert!(!riscv::ZBC.is_empty());
    assert!(!riscv::ZKNE.is_empty());
    assert!(!riscv::ZKND.is_empty());
    assert!(!riscv::ZVBB.is_empty());
    assert!(!riscv::ZVKNED.is_empty());

    // Verify RISC-V features are in the correct word (bits 128-191 = word 2)
    assert!(riscv::V.0[2] != 0);
    assert_eq!(riscv::V.0[0], 0);
    assert_eq!(riscv::V.0[1], 0);
    assert_eq!(riscv::V.0[3], 0);
  }

  #[test]
  fn test_loongarch_features() {
    use super::loongarch;

    // Test LoongArch feature constants are properly defined
    assert!(!loongarch::LSX.is_empty());
    assert!(!loongarch::LASX.is_empty());
    assert!(!loongarch::F.is_empty());
    assert!(!loongarch::D.is_empty());
    assert!(!loongarch::FRECIPE.is_empty());
    assert!(!loongarch::LAM_BH.is_empty());
    assert!(!loongarch::LAMCAS.is_empty());
    assert!(!loongarch::UAL.is_empty());
    assert!(!loongarch::LVZ.is_empty());
    assert!(!loongarch::LBT.is_empty());

    // Verify LoongArch features are in the correct word (bits 208-223 = word 3)
    assert!(loongarch::LSX.0[3] != 0);
    assert_eq!(loongarch::LSX.0[0], 0);
    assert_eq!(loongarch::LSX.0[1], 0);
    assert_eq!(loongarch::LSX.0[2], 0);

    // Test combined masks
    assert!(loongarch::LSX_READY.has(loongarch::LSX));
    assert!(loongarch::LASX_READY.has(loongarch::LSX));
    assert!(loongarch::LASX_READY.has(loongarch::LASX));

    // Verify distinct bit positions
    let all = loongarch::LSX
      | loongarch::LASX
      | loongarch::F
      | loongarch::D
      | loongarch::FRECIPE
      | loongarch::LAM_BH
      | loongarch::LAMCAS
      | loongarch::UAL
      | loongarch::LD_SEQ_SA
      | loongarch::SCQ
      | loongarch::LVZ
      | loongarch::LBT;
    assert_eq!(all.count(), 12);
  }

  #[test]
  fn test_s390x_features() {
    use super::s390x;

    // Test s390x feature constants are properly defined
    assert!(!s390x::VECTOR.is_empty());
    assert!(!s390x::VECTOR_ENH1.is_empty());
    assert!(!s390x::VECTOR_ENH2.is_empty());
    assert!(!s390x::VECTOR_ENH3.is_empty());
    assert!(!s390x::VECTOR_PD.is_empty());
    assert!(!s390x::NNP_ASSIST.is_empty());
    assert!(!s390x::MSA.is_empty());
    assert!(!s390x::MSA8.is_empty());
    assert!(!s390x::DEFLATE.is_empty());

    // Verify s390x features are in the correct word (bits 224-239 = word 3)
    assert!(s390x::VECTOR.0[3] != 0);
    assert_eq!(s390x::VECTOR.0[0], 0);
    assert_eq!(s390x::VECTOR.0[1], 0);
    assert_eq!(s390x::VECTOR.0[2], 0);

    // Test combined masks
    assert!(s390x::Z13_READY.has(s390x::VECTOR));
    assert!(s390x::Z14_READY.has(s390x::VECTOR));
    assert!(s390x::Z14_READY.has(s390x::VECTOR_ENH1));
    assert!(s390x::Z15_READY.has(s390x::VECTOR_ENH2));
    assert!(s390x::Z16_READY.has(s390x::VECTOR_ENH3));
    assert!(s390x::Z16_READY.has(s390x::NNP_ASSIST));

    // Verify distinct bit positions
    let all = s390x::VECTOR
      | s390x::VECTOR_ENH1
      | s390x::VECTOR_ENH2
      | s390x::VECTOR_ENH3
      | s390x::VECTOR_PD
      | s390x::NNP_ASSIST
      | s390x::MISC_EXT2
      | s390x::MISC_EXT3
      | s390x::MSA
      | s390x::MSA4
      | s390x::MSA5
      | s390x::MSA8
      | s390x::MSA9
      | s390x::DEFLATE
      | s390x::ENHANCED_SORT;
    assert_eq!(all.count(), 15);
  }

  #[test]
  fn test_powerpc64_features() {
    use super::powerpc64;

    // Test powerpc64 feature constants are properly defined
    assert!(!powerpc64::ALTIVEC.is_empty());
    assert!(!powerpc64::VSX.is_empty());
    assert!(!powerpc64::POWER8_VECTOR.is_empty());
    assert!(!powerpc64::POWER8_CRYPTO.is_empty());
    assert!(!powerpc64::POWER9_VECTOR.is_empty());
    assert!(!powerpc64::POWER10_VECTOR.is_empty());
    assert!(!powerpc64::QUADWORD_ATOMICS.is_empty());
    assert!(!powerpc64::PARTWORD_ATOMICS.is_empty());

    // Verify powerpc64 features are in the correct word (bits 240-255 = word 3)
    assert!(powerpc64::ALTIVEC.0[3] != 0);
    assert_eq!(powerpc64::ALTIVEC.0[0], 0);
    assert_eq!(powerpc64::ALTIVEC.0[1], 0);
    assert_eq!(powerpc64::ALTIVEC.0[2], 0);

    // Test combined masks
    assert!(powerpc64::POWER7_READY.has(powerpc64::ALTIVEC));
    assert!(powerpc64::POWER7_READY.has(powerpc64::VSX));
    assert!(powerpc64::POWER8_READY.has(powerpc64::POWER8_VECTOR));
    assert!(powerpc64::POWER8_READY.has(powerpc64::POWER8_CRYPTO));
    assert!(powerpc64::POWER9_READY.has(powerpc64::POWER9_VECTOR));
    assert!(powerpc64::POWER10_READY.has(powerpc64::POWER10_VECTOR));

    // Verify distinct bit positions
    let all = powerpc64::ALTIVEC
      | powerpc64::VSX
      | powerpc64::POWER8_VECTOR
      | powerpc64::POWER8_CRYPTO
      | powerpc64::POWER9_VECTOR
      | powerpc64::POWER10_VECTOR
      | powerpc64::QUADWORD_ATOMICS
      | powerpc64::PARTWORD_ATOMICS;
    assert_eq!(all.count(), 8);
  }

  #[test]
  fn test_debug_impl_without_alloc() {
    // Test that Debug impl works correctly (it no longer uses alloc::vec::Vec)
    let caps = x86::SSE42 | x86::PCLMULQDQ;
    let debug_str = alloc::format!("{:?}", caps);

    // Should contain the arch and feature names
    assert!(debug_str.contains("Caps("));
    assert!(debug_str.contains("sse4.2"));
    assert!(debug_str.contains("pclmulqdq"));
  }

  #[test]
  fn test_debug_impl_empty() {
    let caps = Caps::NONE;
    let debug_str = alloc::format!("{:?}", caps);
    assert!(debug_str.contains("none"));
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Word Boundary Tests (bit 63→64, 127→128, 191→192)
  // ─────────────────────────────────────────────────────────────────────────────

  #[test]
  fn test_word_boundary_63_64() {
    let bit63 = Caps::bit(63);
    let bit64 = Caps::bit(64);

    // Verify they're in different words
    assert!(bit63.0[0] != 0 && bit63.0[1] == 0);
    assert!(bit64.0[0] == 0 && bit64.0[1] != 0);

    // Verify they don't interfere
    let both = bit63 | bit64;
    assert!(both.has(bit63));
    assert!(both.has(bit64));
    assert_eq!(both.count(), 2);

    // Intersection should be empty
    let intersection = bit63 & bit64;
    assert!(intersection.is_empty());
  }

  #[test]
  fn test_word_boundary_127_128() {
    let bit127 = Caps::bit(127);
    let bit128 = Caps::bit(128);

    // Verify they're in different words
    assert!(bit127.0[1] != 0 && bit127.0[2] == 0);
    assert!(bit128.0[1] == 0 && bit128.0[2] != 0);

    // Verify they don't interfere
    let both = bit127 | bit128;
    assert!(both.has(bit127));
    assert!(both.has(bit128));
    assert_eq!(both.count(), 2);
  }

  #[test]
  fn test_word_boundary_191_192() {
    let bit191 = Caps::bit(191);
    let bit192 = Caps::bit(192);

    // Verify they're in different words
    assert!(bit191.0[2] != 0 && bit191.0[3] == 0);
    assert!(bit192.0[2] == 0 && bit192.0[3] != 0);

    // Verify they don't interfere
    let both = bit191 | bit192;
    assert!(both.has(bit191));
    assert!(both.has(bit192));
    assert_eq!(both.count(), 2);
  }

  #[test]
  fn test_all_word_boundaries() {
    // Test that bits right at each boundary work correctly
    let boundaries: &[u8] = &[0, 63, 64, 127, 128, 191, 192, 255];
    let mut combined = Caps::NONE;

    for &bit in boundaries {
      let single = Caps::bit(bit);
      assert_eq!(single.count(), 1, "Caps::bit({bit}) should set exactly 1 bit");
      assert!(single.has_bit(bit), "Caps::bit({bit}) should have bit {bit} set");
      combined |= single;
    }

    assert_eq!(combined.count(), 8);
    for &bit in boundaries {
      assert!(combined.has_bit(bit), "Combined should have bit {bit}");
    }
  }

  #[test]
  fn test_bit_positions_no_overlap() {
    // Verify each bit position only sets one bit
    for i in 0u8..=255 {
      let caps = Caps::bit(i);
      assert_eq!(caps.count(), 1, "Caps::bit({i}) should set exactly 1 bit");
      assert!(caps.has_bit(i), "Caps::bit({i}) should have bit {i} set");

      // Verify no other bits are set (sample to avoid O(n^2))
      for j in [0u8, 63, 64, 127, 128, 191, 192, 255] {
        if i != j {
          assert!(!caps.has_bit(j), "Caps::bit({i}) should not have bit {j} set");
        }
      }
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Property-Based Tests (proptest)
// Note: proptest uses filesystem for failure persistence, which Miri doesn't support.
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(all(test, not(miri)))]
mod proptests {
  use proptest::prelude::*;

  use super::*;

  /// Strategy to generate random Caps values
  fn arb_caps() -> impl Strategy<Value = Caps> {
    prop::array::uniform4(any::<u64>()).prop_map(Caps::from_raw)
  }

  proptest! {
    /// Union is commutative: a | b == b | a
    #[test]
    fn caps_union_commutative(a in arb_caps(), b in arb_caps()) {
      prop_assert_eq!(a | b, b | a);
    }

    /// Union is associative: (a | b) | c == a | (b | c)
    #[test]
    fn caps_union_associative(a in arb_caps(), b in arb_caps(), c in arb_caps()) {
      prop_assert_eq!((a | b) | c, a | (b | c));
    }

    /// Intersection is commutative: a & b == b & a
    #[test]
    fn caps_intersection_commutative(a in arb_caps(), b in arb_caps()) {
      prop_assert_eq!(a & b, b & a);
    }

    /// Intersection is associative: (a & b) & c == a & (b & c)
    #[test]
    fn caps_intersection_associative(a in arb_caps(), b in arb_caps(), c in arb_caps()) {
      prop_assert_eq!((a & b) & c, a & (b & c));
    }

    /// Union identity: a | NONE == a
    #[test]
    fn caps_union_identity(a in arb_caps()) {
      prop_assert_eq!(a | Caps::NONE, a);
    }

    /// Intersection absorbing: a & NONE == NONE
    #[test]
    fn caps_intersection_absorbing(a in arb_caps()) {
      prop_assert_eq!(a & Caps::NONE, Caps::NONE);
    }

    /// Self-containment: caps.has(caps) is always true
    #[test]
    fn caps_self_containment(caps in arb_caps()) {
      prop_assert!(caps.has(caps));
    }

    /// After union, both operands are subsets of the result
    #[test]
    fn caps_union_superset(a in arb_caps(), b in arb_caps()) {
      let union = a | b;
      prop_assert!(union.has(a), "union should contain a");
      prop_assert!(union.has(b), "union should contain b");
    }

    /// After intersection, result is subset of both operands
    #[test]
    fn caps_intersection_subset(a in arb_caps(), b in arb_caps()) {
      let intersection = a & b;
      prop_assert!(a.has(intersection), "a should contain intersection");
      prop_assert!(b.has(intersection), "b should contain intersection");
    }

    /// Count is consistent: union count >= max of individual counts
    #[test]
    fn caps_union_count(a in arb_caps(), b in arb_caps()) {
      let union = a | b;
      prop_assert!(union.count() >= a.count().max(b.count()));
    }

    /// Count is consistent: intersection count <= min of individual counts
    #[test]
    fn caps_intersection_count(a in arb_caps(), b in arb_caps()) {
      let intersection = a & b;
      prop_assert!(intersection.count() <= a.count().min(b.count()));
    }

    /// Distributive law: a & (b | c) == (a & b) | (a & c)
    #[test]
    fn caps_distributive(a in arb_caps(), b in arb_caps(), c in arb_caps()) {
      prop_assert_eq!(a & (b | c), (a & b) | (a & c));
    }

    /// De Morgan's law approximation: complement of union
    /// Since we don't have complement, test a related property:
    /// If x is subset of a and subset of b, then x is subset of a & b
    #[test]
    fn caps_intersection_transitivity(a in arb_caps(), b in arb_caps(), x in arb_caps()) {
      if a.has(x) && b.has(x) {
        prop_assert!((a & b).has(x));
      }
    }

    /// Idempotence: a | a == a and a & a == a
    #[test]
    fn caps_idempotent(a in arb_caps()) {
      prop_assert_eq!(a | a, a);
      prop_assert_eq!(a & a, a);
    }

    /// Count accuracy: count equals sum of popcount of words
    #[test]
    fn caps_count_accuracy(caps in arb_caps()) {
      let expected: u32 = caps.0.iter().map(|w| w.count_ones()).sum();
      prop_assert_eq!(caps.count(), expected);
    }

    /// is_empty consistency: is_empty iff count == 0
    #[test]
    fn caps_is_empty_consistency(caps in arb_caps()) {
      prop_assert_eq!(caps.is_empty(), caps.count() == 0);
    }

    /// Bit setting: Caps::bit(n) sets exactly one bit at position n
    #[test]
    fn caps_bit_sets_exactly_one(n in 0u8..=255) {
      let caps = Caps::bit(n);
      prop_assert_eq!(caps.count(), 1);
      prop_assert!(caps.has_bit(n));
    }

    /// has_bit correctness: if has_bit(n), then has(Caps::bit(n))
    #[test]
    fn caps_has_bit_implies_has(caps in arb_caps(), n in 0u8..=255) {
      if caps.has_bit(n) {
        prop_assert!(caps.has(Caps::bit(n)));
      }
    }

    /// has correctness: if has(other), then all bits in other are in self
    #[test]
    fn caps_has_correctness(caps in arb_caps(), other in arb_caps()) {
      if caps.has(other) {
        // Every bit set in other should be set in caps
        for i in 0u8..=255 {
          if other.has_bit(i) {
            prop_assert!(caps.has_bit(i), "caps should have bit {i} if it has 'other'");
          }
        }
      }
    }
  }
}
