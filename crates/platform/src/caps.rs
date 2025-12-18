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

extern crate alloc;

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
  ///
  /// This is the core check used by dispatch selection, so it's marked
  /// `#[inline(always)]` to ensure zero overhead.
  #[inline(always)]
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
  ///
  /// # Panics
  ///
  /// Panics if `bit >= 256`.
  #[inline]
  #[must_use]
  #[allow(clippy::indexing_slicing)] // bounds enforced by assert above
  pub const fn from_bit(bit: u16) -> Self {
    assert!(bit < 256, "bit index must be < 256");
    let word = (bit / 64) as usize;
    let bit_in_word = bit % 64;
    let mut bits = [0u64; 4];
    bits[word] = 1u64 << bit_in_word;
    Self(bits)
  }

  /// Check if a specific bit is set.
  ///
  /// # Panics
  ///
  /// Panics if `bit >= 256`.
  #[inline]
  #[must_use]
  #[allow(clippy::indexing_slicing)] // bounds enforced by assert above
  pub const fn has_bit(self, bit: u16) -> bool {
    assert!(bit < 256, "bit index must be < 256");
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
  ///
  /// This is the core check used by dispatch selection.
  #[inline(always)]
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

  // AVX10 (unified AVX-512 replacement)
  pub const AVX10_1: Bits256 = Bits256::from_bit(36);
  pub const AVX10_2: Bits256 = Bits256::from_bit(37);

  // AMX (Advanced Matrix Extensions)
  pub const AMX_TILE: Bits256 = Bits256::from_bit(38);
  pub const AMX_BF16: Bits256 = Bits256::from_bit(39);
  pub const AMX_INT8: Bits256 = Bits256::from_bit(40);

  // Direct store instructions
  pub const MOVDIRI: Bits256 = Bits256::from_bit(41);
  pub const MOVDIR64B: Bits256 = Bits256::from_bit(42);

  // Instruction serialization
  pub const SERIALIZE: Bits256 = Bits256::from_bit(43);

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

  // Extended crypto and hash
  pub const SHA512: Bits256 = Bits256::from_bit(82);
  pub const RNG: Bits256 = Bits256::from_bit(83); // Hardware RNG (RNDR/RNDRRS)
  pub const SVE2_SHA512: Bits256 = Bits256::from_bit(84);

  // Atomics (ARMv8.1+)
  pub const LSE: Bits256 = Bits256::from_bit(85); // Large System Extensions (atomics)
  pub const LSE2: Bits256 = Bits256::from_bit(86); // ARMv8.4 atomics

  // Memory operations
  pub const MOPS: Bits256 = Bits256::from_bit(87); // FEAT_MOPS memcpy acceleration

  // Scalable Matrix Extension
  pub const SME: Bits256 = Bits256::from_bit(88);
  pub const SME2: Bits256 = Bits256::from_bit(89);

  // ─── Combined Feature Sets (for detection tables) ───
  // These are used by the feature detection macros to set multiple bits
  // from a single target_feature (e.g., "aes" enables both AES and PMULL).

  /// AES + PMULL combined (ARM bundles these together)
  pub const AES_PMULL: Bits256 = Bits256([0, AES.0[1] | PMULL.0[1], 0, 0]);

  /// SM3 + SM4 combined (both enabled by "sm4" target_feature)
  pub const SM3_SM4: Bits256 = Bits256([0, SM3.0[1] | SM4.0[1], 0, 0]);

  // ─── Combined Capability Masks (for dispatch checks) ───

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

// ─────────────────────────────────────────────────────────────────────────────
// Feature Name Lookup
// ─────────────────────────────────────────────────────────────────────────────

/// Feature name entry: (bit_index, name).
pub type FeatureEntry = (u16, &'static str);

/// x86/x86_64 feature names (bits 0-63).
pub const X86_FEATURES: &[FeatureEntry] = &[
  (0, "sse2"),
  (1, "sse3"),
  (2, "ssse3"),
  (3, "sse4.1"),
  (4, "sse4.2"),
  (5, "avx"),
  (6, "avx2"),
  (7, "fma"),
  (8, "aes"),
  (9, "pclmulqdq"),
  (10, "sha"),
  (11, "avx512f"),
  (12, "avx512vl"),
  (13, "avx512bw"),
  (14, "avx512dq"),
  (15, "avx512cd"),
  (16, "vpclmulqdq"),
  (17, "vaes"),
  (18, "gfni"),
  (19, "avx512ifma"),
  (20, "avx512vbmi"),
  (21, "avx512vbmi2"),
  (22, "avx512vnni"),
  (23, "avx512bitalg"),
  (24, "avx512vpopcntdq"),
  (25, "bmi1"),
  (26, "bmi2"),
  (27, "popcnt"),
  (28, "lzcnt"),
  (29, "adx"),
  (30, "sha512"),
  (31, "sse4a"),
  (32, "f16c"),
  (33, "avx512vp2intersect"),
  (34, "avx512fp16"),
  (35, "avx512bf16"),
  (36, "avx10.1"),
  (37, "avx10.2"),
  (38, "amx-tile"),
  (39, "amx-bf16"),
  (40, "amx-int8"),
  (41, "movdiri"),
  (42, "movdir64b"),
  (43, "serialize"),
];

/// aarch64 feature names (bits 64-127).
pub const AARCH64_FEATURES: &[FeatureEntry] = &[
  (64, "neon"),
  (65, "aes"),
  (66, "pmull"),
  (67, "sha2"),
  (68, "sha3"),
  (69, "sm3"),
  (70, "sm4"),
  (71, "crc"),
  (72, "dotprod"),
  (73, "i8mm"),
  (74, "bf16"),
  (75, "fp16"),
  (76, "sve"),
  (77, "sve2"),
  (78, "sve2-aes"),
  (79, "sve2-sha3"),
  (80, "sve2-sm4"),
  (81, "sve2-bitperm"),
  (82, "sha512"),
  (83, "rng"),
  (84, "sve2-sha512"),
  (85, "lse"),
  (86, "lse2"),
  (87, "mops"),
  (88, "sme"),
  (89, "sme2"),
];

/// RISC-V feature names (bits 128-191).
pub const RISCV_FEATURES: &[FeatureEntry] = &[
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

/// WebAssembly feature names (bits 192-255).
pub const WASM_FEATURES: &[FeatureEntry] = &[(192, "simd128"), (193, "relaxed-simd")];

impl Bits256 {
  /// Returns an iterator over the names of all set feature bits.
  ///
  /// The returned names match the Rust target feature strings where applicable.
  ///
  /// # Example
  ///
  /// ```ignore
  /// let bits = x86::SSE42.union(x86::PCLMULQDQ);
  /// let names: Vec<_> = bits.feature_names().collect();
  /// assert!(names.contains(&"sse4.2"));
  /// assert!(names.contains(&"pclmulqdq"));
  /// ```
  pub fn feature_names(self) -> impl Iterator<Item = &'static str> {
    // Chain all feature tables and filter by set bits
    X86_FEATURES
      .iter()
      .chain(AARCH64_FEATURES.iter())
      .chain(RISCV_FEATURES.iter())
      .chain(WASM_FEATURES.iter())
      .filter_map(move |(bit, name)| if self.has_bit(*bit) { Some(*name) } else { None })
  }

  /// Returns the number of feature bits set.
  #[inline]
  #[must_use]
  pub const fn count_ones(self) -> u32 {
    self.0[0].count_ones() + self.0[1].count_ones() + self.0[2].count_ones() + self.0[3].count_ones()
  }
}

impl Arch {
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

// Custom Debug for CpuCaps that shows feature names
impl core::fmt::Display for CpuCaps {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    write!(f, "CpuCaps({}", self.arch)?;

    let names: alloc::vec::Vec<_> = self.bits.feature_names().collect();
    if names.is_empty() {
      write!(f, ", no features")?;
    } else {
      write!(f, ", [")?;
      for (i, name) in names.iter().enumerate() {
        if i > 0 {
          write!(f, ", ")?;
        }
        write!(f, "{name}")?;
      }
      write!(f, "]")?;
    }

    write!(f, ")")
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  // ─────────────────────────────────────────────────────────────────────────────
  // Bits256 Basic Operations
  // ─────────────────────────────────────────────────────────────────────────────

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

  // ─────────────────────────────────────────────────────────────────────────────
  // Bits256 Algebraic Properties
  // ─────────────────────────────────────────────────────────────────────────────

  #[test]
  fn test_bits256_union_commutativity() {
    let a = Bits256::from_bit(5);
    let b = Bits256::from_bit(10);
    assert_eq!(a.union(b), b.union(a));
  }

  #[test]
  fn test_bits256_union_associativity() {
    let a = Bits256::from_bit(0);
    let b = Bits256::from_bit(64);
    let c = Bits256::from_bit(128);
    assert_eq!(a.union(b).union(c), a.union(b.union(c)));
  }

  #[test]
  fn test_bits256_union_idempotence() {
    let a = Bits256::from_bit(42);
    assert_eq!(a.union(a), a);
  }

  #[test]
  fn test_bits256_union_identity() {
    let a = Bits256::from_bit(100);
    assert_eq!(a.union(Bits256::NONE), a);
    assert_eq!(Bits256::NONE.union(a), a);
  }

  #[test]
  fn test_bits256_intersection_commutativity() {
    let a = x86::SSE42.union(x86::AVX);
    let b = x86::AVX.union(x86::AVX2);
    assert_eq!(a.intersection(b), b.intersection(a));
  }

  #[test]
  fn test_bits256_intersection_associativity() {
    let a = x86::SSE42.union(x86::AVX).union(x86::AVX2);
    let b = x86::AVX.union(x86::AVX2).union(x86::FMA);
    let c = x86::AVX2.union(x86::FMA).union(x86::AVX512F);
    assert_eq!(a.intersection(b).intersection(c), a.intersection(b.intersection(c)));
  }

  #[test]
  fn test_bits256_intersection_idempotence() {
    let a = x86::VPCLMUL_READY;
    assert_eq!(a.intersection(a), a);
  }

  #[test]
  fn test_bits256_contains_reflexive() {
    let a = x86::VPCLMUL_READY;
    assert!(a.contains(a));
  }

  #[test]
  fn test_bits256_contains_transitive() {
    let a = x86::SSE2;
    let b = x86::SSE2.union(x86::SSE3);
    let c = x86::SSE2.union(x86::SSE3).union(x86::SSSE3);
    // If c contains b and b contains a, then c contains a
    assert!(c.contains(b));
    assert!(b.contains(a));
    assert!(c.contains(a));
  }

  #[test]
  fn test_bits256_none_contains_only_none() {
    assert!(Bits256::NONE.contains(Bits256::NONE));
    assert!(!Bits256::NONE.contains(Bits256::from_bit(0)));
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Bits256 All Words Coverage
  // ─────────────────────────────────────────────────────────────────────────────

  #[test]
  fn test_bits256_all_words() {
    // Test bits in each of the 4 words
    let word0 = Bits256::from_bit(0); // First word (x86)
    let word1 = Bits256::from_bit(64); // Second word (aarch64)
    let word2 = Bits256::from_bit(128); // Third word (riscv)
    let word3 = Bits256::from_bit(192); // Fourth word (wasm/other)

    assert_eq!(word0.0[0], 1);
    assert_eq!(word0.0[1], 0);
    assert_eq!(word0.0[2], 0);
    assert_eq!(word0.0[3], 0);

    assert_eq!(word1.0[0], 0);
    assert_eq!(word1.0[1], 1);
    assert_eq!(word1.0[2], 0);
    assert_eq!(word1.0[3], 0);

    assert_eq!(word2.0[0], 0);
    assert_eq!(word2.0[1], 0);
    assert_eq!(word2.0[2], 1);
    assert_eq!(word2.0[3], 0);

    assert_eq!(word3.0[0], 0);
    assert_eq!(word3.0[1], 0);
    assert_eq!(word3.0[2], 0);
    assert_eq!(word3.0[3], 1);

    // Union across all words
    let all = word0.union(word1).union(word2).union(word3);
    assert!(all.has_bit(0));
    assert!(all.has_bit(64));
    assert!(all.has_bit(128));
    assert!(all.has_bit(192));

    // Contains checks across words
    assert!(all.contains(word0));
    assert!(all.contains(word1));
    assert!(all.contains(word2));
    assert!(all.contains(word3));
  }

  #[test]
  fn test_bits256_boundary_bits() {
    // Test bits at word boundaries
    for bit in [0, 63, 64, 127, 128, 191, 192, 255] {
      let b = Bits256::from_bit(bit);
      assert!(b.has_bit(bit), "bit {bit} should be set");
      assert!(
        !b.has_bit(if bit > 0 { bit - 1 } else { 1 }),
        "adjacent bit should not be set"
      );
    }
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Bits256 count_ones
  // ─────────────────────────────────────────────────────────────────────────────

  #[test]
  fn test_bits256_count_ones() {
    assert_eq!(Bits256::NONE.count_ones(), 0);
    assert_eq!(Bits256::from_bit(0).count_ones(), 1);
    assert_eq!(Bits256::from_bit(0).union(Bits256::from_bit(1)).count_ones(), 2);
    assert_eq!(x86::VPCLMUL_READY.count_ones(), 5); // VPCLMULQDQ + AVX512F + AVX512VL + AVX512BW + PCLMULQDQ
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Feature Name Lookup
  // ─────────────────────────────────────────────────────────────────────────────

  #[test]
  fn test_feature_names_x86() {
    let bits = x86::SSE42.union(x86::PCLMULQDQ);
    let names: alloc::vec::Vec<_> = bits.feature_names().collect();
    assert!(names.contains(&"sse4.2"));
    assert!(names.contains(&"pclmulqdq"));
    assert!(!names.contains(&"avx512f"));
  }

  #[test]
  fn test_feature_names_aarch64() {
    let bits = aarch64::NEON.union(aarch64::AES).union(aarch64::SHA3);
    let names: alloc::vec::Vec<_> = bits.feature_names().collect();
    assert!(names.contains(&"neon"));
    assert!(names.contains(&"aes"));
    assert!(names.contains(&"sha3"));
  }

  #[test]
  fn test_feature_names_empty() {
    let names: alloc::vec::Vec<_> = Bits256::NONE.feature_names().collect();
    assert!(names.is_empty());
  }

  #[test]
  fn test_feature_names_cross_arch() {
    // Features from different architectures (unusual but valid)
    let bits = x86::AVX2.union(aarch64::NEON).union(wasm::SIMD128);
    let names: alloc::vec::Vec<_> = bits.feature_names().collect();
    assert!(names.contains(&"avx2"));
    assert!(names.contains(&"neon"));
    assert!(names.contains(&"simd128"));
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // CpuCaps
  // ─────────────────────────────────────────────────────────────────────────────

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
  fn test_cpu_caps_has_feature() {
    let caps = CpuCaps::new(x86::SSE42.union(x86::AVX));
    assert!(caps.has_feature(4)); // SSE42 is bit 4
    assert!(caps.has_feature(5)); // AVX is bit 5
    assert!(!caps.has_feature(6)); // AVX2 is bit 6 (not set)
  }

  #[test]
  fn test_cpu_caps_union() {
    let caps1 = CpuCaps::new(x86::SSE42);
    let caps2 = CpuCaps::new(x86::AVX);
    let combined = caps1.union(caps2);
    assert!(combined.has(x86::SSE42));
    assert!(combined.has(x86::AVX));
  }

  #[test]
  fn test_cpu_caps_none() {
    let caps = CpuCaps::NONE;
    assert!(caps.bits.is_empty());
    assert_eq!(caps.arch, Arch::Other);
    assert!(!caps.has(x86::SSE2));
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Display Formatting
  // ─────────────────────────────────────────────────────────────────────────────

  #[test]
  fn test_cpu_caps_display() {
    let caps = CpuCaps::new(x86::SSE42.union(x86::PCLMULQDQ));
    let s = alloc::format!("{caps}");
    assert!(s.contains("sse4.2"));
    assert!(s.contains("pclmulqdq"));
  }

  #[test]
  fn test_cpu_caps_display_no_features() {
    let caps = CpuCaps::NONE;
    let s = alloc::format!("{caps}");
    assert!(s.contains("no features"));
  }

  #[test]
  fn test_arch_display() {
    assert_eq!(alloc::format!("{}", Arch::X86_64), "x86_64");
    assert_eq!(alloc::format!("{}", Arch::Aarch64), "aarch64");
    assert_eq!(alloc::format!("{}", Arch::Other), "other");
  }

  #[test]
  fn test_arch_name() {
    assert_eq!(Arch::X86_64.name(), "x86_64");
    assert_eq!(Arch::Aarch64.name(), "aarch64");
    assert_eq!(Arch::Riscv64.name(), "riscv64");
    assert_eq!(Arch::Wasm32.name(), "wasm32");
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Arch
  // ─────────────────────────────────────────────────────────────────────────────

  #[test]
  fn test_arch_current() {
    let arch = Arch::current();
    #[cfg(target_arch = "x86_64")]
    assert_eq!(arch, Arch::X86_64);
    #[cfg(target_arch = "aarch64")]
    assert_eq!(arch, Arch::Aarch64);
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Feature Bit Definitions
  // ─────────────────────────────────────────────────────────────────────────────

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

  #[test]
  fn test_riscv_feature_bits() {
    // Verify riscv bits are in the third word (bits 128-191)
    assert!(riscv::V.0[2] != 0);
    assert!(riscv::V.0[0] == 0);
    assert!(riscv::V.0[1] == 0);
  }

  #[test]
  fn test_wasm_feature_bits() {
    // Verify wasm bits are in the fourth word (bits 192-255)
    assert!(wasm::SIMD128.0[3] != 0);
    assert!(wasm::SIMD128.0[0] == 0);
    assert!(wasm::SIMD128.0[1] == 0);
    assert!(wasm::SIMD128.0[2] == 0);
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Operator Overloads
  // ─────────────────────────────────────────────────────────────────────────────

  #[test]
  fn test_bits256_bitor_operator() {
    let a = Bits256::from_bit(0);
    let b = Bits256::from_bit(1);
    let ab = a | b;
    assert_eq!(ab, a.union(b));
  }

  #[test]
  fn test_bits256_bitand_operator() {
    let a = x86::SSE42.union(x86::AVX);
    let b = x86::AVX.union(x86::AVX2);
    let intersection = a & b;
    assert_eq!(intersection, a.intersection(b));
    assert!(intersection.contains(x86::AVX));
    assert!(!intersection.contains(x86::SSE42));
    assert!(!intersection.contains(x86::AVX2));
  }

  #[test]
  fn test_bits256_bitor_assign() {
    let mut a = Bits256::from_bit(0);
    a |= Bits256::from_bit(1);
    assert!(a.has_bit(0));
    assert!(a.has_bit(1));
  }
}
