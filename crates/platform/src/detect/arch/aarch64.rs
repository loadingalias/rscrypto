// aarch64 Detection
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
fn detect_aarch64() -> Detected {
  // Start with compile-time detected features (includes NEON baseline)
  #[cfg(feature = "std")]
  let caps = caps_static() | runtime_aarch64();
  #[cfg(not(feature = "std"))]
  let caps = caps_static();

  let tune = select_aarch64_tune(caps);

  Detected {
    caps,
    tune,
    arch: Arch::Aarch64,
  }
}

/// Batch extraction of aarch64 features from /proc/self/auxv.
///
/// Reads AT_HWCAP and AT_HWCAP2 once from the ELF auxiliary vector.
/// This is faster than calling is_aarch64_feature_detected! 20+ times.
/// Pure Rust - no libc dependency.
///
/// Works on Linux and Android (both use procfs with ELF auxv format).
#[cfg(all(
  target_arch = "aarch64",
  feature = "std",
  any(target_os = "linux", target_os = "android")
))]
fn hwcap_batch_aarch64() -> Caps {
  use std::{fs::File, io::Read};

  use crate::caps::aarch64;

  // ELF auxiliary vector entry types
  const AT_HWCAP: u64 = 16;
  const AT_HWCAP2: u64 = 26;

  // HWCAP bit positions (from linux/arch/arm64/include/uapi/asm/hwcap.h)
  const HWCAP_AES: u64 = 1 << 3;
  const HWCAP_PMULL: u64 = 1 << 4;
  const HWCAP_SHA2: u64 = 1 << 6;
  const HWCAP_CRC32: u64 = 1 << 7;
  const HWCAP_ATOMICS: u64 = 1 << 8; // LSE
  const HWCAP_FPHP: u64 = 1 << 9; // FP16
  const HWCAP_ASIMDHP: u64 = 1 << 10;
  const HWCAP_SHA3: u64 = 1 << 17;
  const HWCAP_SM3: u64 = 1 << 18;
  const HWCAP_SM4: u64 = 1 << 19;
  const HWCAP_ASIMDDP: u64 = 1 << 20; // DOTPROD
  const HWCAP_SHA512: u64 = 1 << 21;
  const HWCAP_SVE: u64 = 1 << 22;

  // HWCAP2 bit positions
  const HWCAP2_SVE2: u64 = 1 << 1;
  const HWCAP2_SVEAES: u64 = 1 << 2;
  const HWCAP2_SVEPMULL: u64 = 1 << 3;
  const HWCAP2_SVEBITPERM: u64 = 1 << 4;
  const HWCAP2_SVESHA3: u64 = 1 << 5;
  const HWCAP2_SVESM4: u64 = 1 << 6;
  const HWCAP2_FRINT: u64 = 1 << 8; // FRINTTS
  const HWCAP2_SVEI8MM: u64 = 1 << 9;
  const HWCAP2_SVEF32MM: u64 = 1 << 10;
  const HWCAP2_SVEF64MM: u64 = 1 << 11;
  const HWCAP2_SVEBF16: u64 = 1 << 12;
  const HWCAP2_I8MM: u64 = 1 << 13;
  const HWCAP2_BF16: u64 = 1 << 14;
  const HWCAP2_RNG: u64 = 1 << 16;
  const HWCAP2_SME: u64 = 1 << 23;
  const HWCAP2_SME_I16I64: u64 = 1 << 24;
  const HWCAP2_SME_F64F64: u64 = 1 << 25;
  const HWCAP2_SME_I8I32: u64 = 1 << 26;
  const HWCAP2_SME_F16F32: u64 = 1 << 27;
  const HWCAP2_SME_B16F32: u64 = 1 << 28;
  const HWCAP2_SME_F32F32: u64 = 1 << 29;
  const HWCAP2_SME_FA64: u64 = 1 << 30;
  const HWCAP2_EBF16: u64 = 1 << 32;
  const HWCAP2_SVE_EBF16: u64 = 1 << 33;
  const HWCAP2_SVE2P1: u64 = 1 << 36;
  const HWCAP2_SME2: u64 = 1 << 37;
  const HWCAP2_SME2P1: u64 = 1 << 38;
  const HWCAP2_SME_I16I32: u64 = 1 << 39;
  const HWCAP2_SME_BI32I32: u64 = 1 << 40;
  const HWCAP2_SME_B16B16: u64 = 1 << 41;
  const HWCAP2_SME_F16F16: u64 = 1 << 42;
  const HWCAP2_MOPS: u64 = 1 << 43;
  const HWCAP2_SVE_B16B16: u64 = 1 << 45;
  const HWCAP2_LSE128: u64 = 1 << 47;

  // Read /proc/self/auxv - format is pairs of (type: u64, value: u64)
  let (hwcap, hwcap2) = (|| -> Option<(u64, u64)> {
    let mut file = File::open("/proc/self/auxv").ok()?;
    let mut buf = [0u8; 4096]; // Auxiliary vector is small
    let n = file.read(&mut buf).ok()?;

    let mut hwcap = 0u64;
    let mut hwcap2 = 0u64;

    // Parse as array of (u64, u64) pairs
    let entries = buf.get(..n)?;
    for chunk in entries.chunks_exact(16) {
      let a_type = u64::from_ne_bytes(chunk.get(0..8)?.try_into().ok()?);
      let a_val = u64::from_ne_bytes(chunk.get(8..16)?.try_into().ok()?);

      if a_type == AT_HWCAP {
        hwcap = a_val;
      } else if a_type == AT_HWCAP2 {
        hwcap2 = a_val;
      } else if a_type == 0 {
        // AT_NULL terminates the vector
        break;
      }
    }
    Some((hwcap, hwcap2))
  })()
  .unwrap_or((0, 0));

  let mut caps = Caps::NONE;

  // ─── HWCAP features ───
  if hwcap & HWCAP_AES != 0 {
    caps |= aarch64::AES;
  }
  if hwcap & HWCAP_PMULL != 0 {
    caps |= aarch64::PMULL;
  }
  if hwcap & HWCAP_SHA2 != 0 {
    caps |= aarch64::SHA2;
  }
  if hwcap & HWCAP_CRC32 != 0 {
    caps |= aarch64::CRC;
  }
  if hwcap & HWCAP_ATOMICS != 0 {
    caps |= aarch64::LSE;
  }
  if hwcap & (HWCAP_FPHP | HWCAP_ASIMDHP) != 0 {
    caps |= aarch64::FP16;
  }
  if hwcap & HWCAP_SHA3 != 0 {
    caps |= aarch64::SHA3;
  }
  if hwcap & HWCAP_SM3 != 0 {
    caps |= aarch64::SM3;
  }
  if hwcap & HWCAP_SM4 != 0 {
    caps |= aarch64::SM4;
  }
  if hwcap & HWCAP_ASIMDDP != 0 {
    caps |= aarch64::DOTPROD;
  }
  if hwcap & HWCAP_SHA512 != 0 {
    caps |= aarch64::SHA512;
  }
  if hwcap & HWCAP_SVE != 0 {
    caps |= aarch64::SVE;
  }

  // ─── HWCAP2 features ───
  if hwcap2 & HWCAP2_SVE2 != 0 {
    caps |= aarch64::SVE2;
  }
  if hwcap2 & HWCAP2_SVEAES != 0 {
    caps |= aarch64::SVE2_AES;
  }
  if hwcap2 & HWCAP2_SVEPMULL != 0 {
    caps |= aarch64::SVE2_PMULL;
  }
  if hwcap2 & HWCAP2_SVEBITPERM != 0 {
    caps |= aarch64::SVE2_BITPERM;
  }
  if hwcap2 & HWCAP2_SVESHA3 != 0 {
    caps |= aarch64::SVE2_SHA3;
  }
  if hwcap2 & HWCAP2_SVESM4 != 0 {
    caps |= aarch64::SVE2_SM4;
  }
  if hwcap2 & HWCAP2_FRINT != 0 {
    caps |= aarch64::FRINTTS;
  }
  if hwcap2 & HWCAP2_SVEI8MM != 0 {
    caps |= aarch64::SVE2_I8MM;
  }
  if hwcap2 & HWCAP2_SVEF32MM != 0 {
    caps |= aarch64::SVE2_F32MM;
  }
  if hwcap2 & HWCAP2_SVEF64MM != 0 {
    caps |= aarch64::SVE2_F64MM;
  }
  if hwcap2 & HWCAP2_SVEBF16 != 0 {
    caps |= aarch64::SVE2_BF16;
  }
  if hwcap2 & HWCAP2_I8MM != 0 {
    caps |= aarch64::I8MM;
  }
  if hwcap2 & HWCAP2_BF16 != 0 {
    caps |= aarch64::BF16;
  }
  if hwcap2 & HWCAP2_RNG != 0 {
    caps |= aarch64::RNG;
  }
  if hwcap2 & HWCAP2_SME != 0 {
    caps |= aarch64::SME;
  }
  if hwcap2 & HWCAP2_SME_I16I64 != 0 {
    caps |= aarch64::SME_I16I64;
  }
  if hwcap2 & HWCAP2_SME_F64F64 != 0 {
    caps |= aarch64::SME_F64F64;
  }
  if hwcap2 & HWCAP2_SME_I8I32 != 0 {
    caps |= aarch64::SME_I8I32;
  }
  if hwcap2 & HWCAP2_SME_F16F32 != 0 {
    caps |= aarch64::SME_F16F32;
  }
  if hwcap2 & HWCAP2_SME_B16F32 != 0 {
    caps |= aarch64::SME_B16F32;
  }
  if hwcap2 & HWCAP2_SME_F32F32 != 0 {
    caps |= aarch64::SME_F32F32;
  }
  if hwcap2 & HWCAP2_SME_FA64 != 0 {
    caps |= aarch64::SME_FA64;
  }
  if hwcap2 & HWCAP2_EBF16 != 0 {
    caps |= aarch64::EBF16;
  }
  if hwcap2 & HWCAP2_SVE_EBF16 != 0 {
    caps |= aarch64::SVE2_EBF16;
  }
  if hwcap2 & HWCAP2_SVE2P1 != 0 {
    caps |= aarch64::SVE2P1;
  }
  if hwcap2 & HWCAP2_SME2 != 0 {
    caps |= aarch64::SME2;
  }
  if hwcap2 & HWCAP2_SME2P1 != 0 {
    caps |= aarch64::SME2P1;
  }
  if hwcap2 & HWCAP2_SME_I16I32 != 0 {
    caps |= aarch64::SME_I16I32;
  }
  if hwcap2 & HWCAP2_SME_BI32I32 != 0 {
    caps |= aarch64::SME_BI32I32;
  }
  if hwcap2 & HWCAP2_SME_B16B16 != 0 {
    caps |= aarch64::SME_B16B16;
  }
  if hwcap2 & HWCAP2_SME_F16F16 != 0 {
    caps |= aarch64::SME_F16F16;
  }
  if hwcap2 & HWCAP2_MOPS != 0 {
    caps |= aarch64::MOPS;
  }
  if hwcap2 & HWCAP2_SVE_B16B16 != 0 {
    caps |= aarch64::SVE_B16B16;
  }
  if hwcap2 & HWCAP2_LSE128 != 0 {
    caps |= aarch64::LSE2;
  }

  caps
}

/// Runtime aarch64 detection for Linux/Android (batch HWCAP from /proc/self/auxv).
#[cfg(all(
  target_arch = "aarch64",
  feature = "std",
  any(target_os = "linux", target_os = "android")
))]
fn runtime_aarch64() -> Caps {
  hwcap_batch_aarch64()
}

/// Runtime aarch64 detection for other platforms (fallback to macro calls).
#[cfg(all(
  target_arch = "aarch64",
  feature = "std",
  not(any(target_os = "linux", target_os = "android"))
))]
fn runtime_aarch64() -> Caps {
  use crate::caps::aarch64;

  let mut caps = Caps::NONE;

  // ─── Crypto Extensions ───
  if std::arch::is_aarch64_feature_detected!("aes") {
    caps |= aarch64::AES;
  }
  if std::arch::is_aarch64_feature_detected!("pmull") {
    caps |= aarch64::PMULL;
  }
  if std::arch::is_aarch64_feature_detected!("sha2") {
    caps |= aarch64::SHA2;
  }
  if std::arch::is_aarch64_feature_detected!("sha3") {
    caps |= aarch64::SHA3 | aarch64::SHA512;
  }
  if std::arch::is_aarch64_feature_detected!("sm4") {
    caps |= aarch64::SM3 | aarch64::SM4;
  }

  // ─── CRC Extension ───
  if std::arch::is_aarch64_feature_detected!("crc") {
    caps |= aarch64::CRC;
  }

  // ─── Additional SIMD ───
  if std::arch::is_aarch64_feature_detected!("dotprod") {
    caps |= aarch64::DOTPROD;
  }
  if std::arch::is_aarch64_feature_detected!("fp16") {
    caps |= aarch64::FP16;
  }
  if std::arch::is_aarch64_feature_detected!("i8mm") {
    caps |= aarch64::I8MM;
  }
  if std::arch::is_aarch64_feature_detected!("bf16") {
    caps |= aarch64::BF16;
  }
  if std::arch::is_aarch64_feature_detected!("frintts") {
    caps |= aarch64::FRINTTS;
  }

  // ─── SVE Family ───
  if std::arch::is_aarch64_feature_detected!("sve") {
    caps |= aarch64::SVE;
  }
  if std::arch::is_aarch64_feature_detected!("sve2") {
    caps |= aarch64::SVE2;
  }

  // ─── Atomics ───
  if std::arch::is_aarch64_feature_detected!("lse") {
    caps |= aarch64::LSE;
  }
  if std::arch::is_aarch64_feature_detected!("lse2") {
    caps |= aarch64::LSE2;
  }

  // ─── Memory Operations ───
  // MOPS is detected on Linux via HWCAP2 in hwcap_batch_aarch64()
  // On other platforms, compile-time detection via target_feature is used
  #[cfg(all(target_feature = "mops", not(any(target_os = "linux", target_os = "android"))))]
  {
    caps |= aarch64::MOPS;
  }

  // ─── Hardware RNG ───
  if std::arch::is_aarch64_feature_detected!("rand") {
    caps |= aarch64::RNG;
  }

  // ─── SME Detection ───
  // On macOS and other Apple platforms, use sysctl for comprehensive SME detection.
  // std::arch::is_aarch64_feature_detected doesn't currently detect SME reliably
  // on macOS, so we use platform-specific detection.
  #[cfg(any(target_os = "macos", target_os = "ios", target_os = "tvos", target_os = "watchos"))]
  {
    caps |= detect_apple_sme_features();
  }

  // For Linux (non-Apple), fall back to is_aarch64_feature_detected for SME.
  // Note: SME detection via std::arch is only stable on Linux; Windows ARM64 requires
  // unstable `stdarch_aarch64_feature_detection` feature which we avoid.
  #[cfg(target_os = "linux")]
  {
    if std::arch::is_aarch64_feature_detected!("sme") {
      caps |= aarch64::SME;
    }
    if std::arch::is_aarch64_feature_detected!("sme2") {
      caps |= aarch64::SME2;
    }
  }

  caps
}

// ─────────────────────────────────────────────────────────────────────────────
// Apple Silicon Detection (macOS/iOS)
// ─────────────────────────────────────────────────────────────────────────────

/// Apple Silicon chip generation.
#[cfg(all(
  target_arch = "aarch64",
  feature = "std",
  any(target_os = "macos", target_os = "ios", target_os = "tvos", target_os = "watchos")
))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AppleSiliconGen {
  /// M1, M1 Pro, M1 Max, M1 Ultra (Firestorm/Icestorm)
  M1,
  /// M2, M2 Pro, M2 Max, M2 Ultra (Blizzard/Avalanche)
  M2,
  /// M3, M3 Pro, M3 Max (Ibiza/Lobos/Palma)
  M3,
  /// M4, M4 Pro, M4 Max (Donan/Brava) - has SME
  M4,
  /// M5, M5 Pro, M5 Max (Hidra/Sotra) - has SME2p1
  /// Released October 2025. Adds SME2p1, SMEB16B16, SMEF16F16 per LLVM.
  // TODO(M5): Remove allow(dead_code) once CPUFAMILY_ARM_HIDRA/SOTRA values are added.
  #[allow(dead_code)]
  M5,
}

/// Detect Apple Silicon generation via sysctlbyname("hw.cpufamily").
///
/// Uses direct extern "C" linkage to libSystem (always linked on Apple platforms)
/// to avoid adding libc as a dependency.
///
/// Returns `None` for unknown/future chips or A-series (pre-M1) processors.
#[cfg(all(
  target_arch = "aarch64",
  feature = "std",
  any(target_os = "macos", target_os = "ios", target_os = "tvos", target_os = "watchos")
))]
fn detect_apple_silicon_gen() -> Option<AppleSiliconGen> {
  // CPUFAMILY constants from Apple's machine.h (via Zig's darwin.zig / Homebrew / LLVM)
  // These identify the CPU microarchitecture, not the marketing name.
  //
  // Reference: https://github.com/Homebrew/brew/blob/master/Library/Homebrew/extend/os/mac/hardware/cpu.rb
  // Reference: https://github.com/ziglang/zig/blob/master/lib/std/c/darwin.zig
  const CPUFAMILY_ARM_FIRESTORM_ICESTORM: u32 = 0x1b58_8bb3; // M1 family
  const CPUFAMILY_ARM_BLIZZARD_AVALANCHE: u32 = 0xda33_d83d; // M2 family
  const CPUFAMILY_ARM_EVEREST_SAWTOOTH: u32 = 0x8765_edea; // A16/M2 variant
  const CPUFAMILY_ARM_COLL: u32 = 0x2876_f5b5; // A17 Pro
  const CPUFAMILY_ARM_IBIZA: u32 = 0xfa33_415e; // M3
  const CPUFAMILY_ARM_LOBOS: u32 = 0x5f4d_ea93; // M3 Pro
  const CPUFAMILY_ARM_PALMA: u32 = 0x7201_5832; // M3 Max
  const CPUFAMILY_ARM_DONAN: u32 = 0x6f51_29ac; // M4
  const CPUFAMILY_ARM_BRAVA: u32 = 0x17d5_b93a; // M4 Pro/Max
  const CPUFAMILY_ARM_TAHITI: u32 = 0x75d4_acb9; // A18
  const CPUFAMILY_ARM_TUPAI: u32 = 0x2045_26d0; // A18 Pro

  // M5 family (released October 2025)
  // Codenames: Hidra (M5 - H17G), Sotra (M5 Pro/Max)
  // Features: SME2p1, SMEB16B16, SMEF16F16 per LLVM commit f85494f6afeb
  // TODO(M5): Add CPUFAMILY_ARM_HIDRA and CPUFAMILY_ARM_SOTRA hex values
  //           when publicly documented in Xcode SDK / Homebrew / Zig darwin.zig
  // Reference: Xcode 26.1b3 added CPUFAMILY_ARM_HIDRA (H17G)
  // Until then, M5 detection falls back to SME2 feature detection (see select_aarch64_tune)

  // Direct extern "C" linkage to libSystem's sysctlbyname
  // (libSystem is always linked on Apple platforms)
  // SAFETY: This extern block declares a C function from libSystem.
  // The function signature matches Apple's sysctlbyname(3).
  #[allow(unsafe_code)]
  unsafe extern "C" {
    fn sysctlbyname(
      name: *const u8,
      oldp: *mut core::ffi::c_void,
      oldlenp: *mut usize,
      newp: *const core::ffi::c_void,
      newlen: usize,
    ) -> i32;
  }

  let mut cpufamily: u32 = 0;
  let mut size = core::mem::size_of::<u32>();

  // SAFETY: sysctlbyname is safe to call with valid pointers.
  // "hw.cpufamily" is a valid null-terminated string.
  // The output buffer is properly sized for u32.
  #[allow(unsafe_code)]
  let ret = unsafe {
    sysctlbyname(
      c"hw.cpufamily".as_ptr().cast(),
      core::ptr::addr_of_mut!(cpufamily).cast(),
      core::ptr::addr_of_mut!(size),
      core::ptr::null(),
      0,
    )
  };

  if ret != 0 {
    return None;
  }

  match cpufamily {
    CPUFAMILY_ARM_FIRESTORM_ICESTORM => Some(AppleSiliconGen::M1),
    CPUFAMILY_ARM_BLIZZARD_AVALANCHE | CPUFAMILY_ARM_EVEREST_SAWTOOTH => Some(AppleSiliconGen::M2),
    CPUFAMILY_ARM_IBIZA | CPUFAMILY_ARM_LOBOS | CPUFAMILY_ARM_PALMA => Some(AppleSiliconGen::M3),
    CPUFAMILY_ARM_DONAN | CPUFAMILY_ARM_BRAVA => Some(AppleSiliconGen::M4),
    // A-series chips (A16, A17, A18) - treat as M-series equivalent for tuning
    CPUFAMILY_ARM_COLL => Some(AppleSiliconGen::M2), // A17 Pro ≈ M2 architecture
    CPUFAMILY_ARM_TAHITI | CPUFAMILY_ARM_TUPAI => Some(AppleSiliconGen::M4), // A18 ≈ M4 architecture
    _ => None,                                       // Unknown future chip - will fall back to feature-based detection
  }
}

// ─────────────────────────────────────────────────────────────────────────────
/// Detect SME features on Apple platforms via sysctlbyname.
///
/// Apple exposes SME and related features through hw.optional.arm.FEAT_* sysctl keys.
/// This provides more reliable detection than is_aarch64_feature_detected on macOS.
///
/// Returns a Caps bitset with detected SME features.
#[cfg(all(
  target_arch = "aarch64",
  feature = "std",
  any(target_os = "macos", target_os = "ios", target_os = "tvos", target_os = "watchos")
))]
fn detect_apple_sme_features() -> Caps {
  use crate::caps::aarch64;

  // Helper to read a u32 sysctl value (returns 0 on error or false, 1 on true)
  fn sysctl_u32(name: &[u8]) -> u32 {
    // Direct extern "C" linkage to libSystem's sysctlbyname
    // SAFETY: This extern block declares a C function from libSystem.
    #[allow(unsafe_code)]
    unsafe extern "C" {
      fn sysctlbyname(
        name: *const u8,
        oldp: *mut core::ffi::c_void,
        oldlenp: *mut usize,
        newp: *const core::ffi::c_void,
        newlen: usize,
      ) -> i32;
    }

    let mut value: u32 = 0;
    let mut size = core::mem::size_of::<u32>();

    // SAFETY: sysctlbyname is safe to call with valid pointers.
    // name is a valid null-terminated C string.
    // The output buffer is properly sized for u32.
    #[allow(unsafe_code)]
    let ret = unsafe {
      sysctlbyname(
        name.as_ptr(),
        core::ptr::addr_of_mut!(value).cast(),
        core::ptr::addr_of_mut!(size),
        core::ptr::null(),
        0,
      )
    };

    if ret == 0 { value } else { 0 }
  }

  let mut caps = Caps::NONE;

  // ─── SME Base and Versions ───
  if sysctl_u32(c"hw.optional.arm.FEAT_SME".to_bytes_with_nul()) != 0 {
    caps |= aarch64::SME;
  }
  if sysctl_u32(c"hw.optional.arm.FEAT_SME2".to_bytes_with_nul()) != 0 {
    caps |= aarch64::SME2;
  }
  if sysctl_u32(c"hw.optional.arm.FEAT_SME2p1".to_bytes_with_nul()) != 0 {
    caps |= aarch64::SME2P1;
  }

  // ─── SME Extended Features ───
  if sysctl_u32(c"hw.optional.arm.FEAT_SME_I16I64".to_bytes_with_nul()) != 0 {
    caps |= aarch64::SME_I16I64;
  }
  if sysctl_u32(c"hw.optional.arm.FEAT_SME_F64F64".to_bytes_with_nul()) != 0 {
    caps |= aarch64::SME_F64F64;
  }
  if sysctl_u32(c"hw.optional.arm.FEAT_SME_B16B16".to_bytes_with_nul()) != 0 {
    caps |= aarch64::SME_B16B16;
  }
  if sysctl_u32(c"hw.optional.arm.FEAT_SME_F16F16".to_bytes_with_nul()) != 0 {
    caps |= aarch64::SME_F16F16;
  }

  // ─── Fallback: Infer SME from chip generation if sysctl unavailable ───
  // This handles cases where the OS doesn't expose SME sysctl keys yet.
  // M4 has SME, M5 has SME2p1 + additional features.
  if caps.is_empty()
    && let Some(chip_gen) = detect_apple_silicon_gen()
  {
    match chip_gen {
      AppleSiliconGen::M4 => {
        caps |= aarch64::SME;
      }
      AppleSiliconGen::M5 => {
        // M5 has SME2p1, SMEB16B16, SMEF16F16 per LLVM
        caps |= aarch64::SME | aarch64::SME2 | aarch64::SME2P1 | aarch64::SME_B16B16 | aarch64::SME_F16F16;
      }
      _ => {}
    }
  }

  caps
}

/// Detect SME tile size (SVL - Streaming Vector Length) on Apple platforms.
///
/// Returns the maximum SVL in bytes, or 0 if SME is not supported or detection failed.
///
/// On Apple Silicon:
/// - M4: SME with 128-bit tiles (SVL = 16 bytes)
/// - M5: SME2p1 with 128-bit tiles (SVL = 16 bytes)
///
/// Note: Apple's implementation uses fixed 128-bit SVL, unlike server ARM chips
/// which may support 128-512 bit configurable SVL.
#[cfg(all(
  target_arch = "aarch64",
  feature = "std",
  any(target_os = "macos", target_os = "ios", target_os = "tvos", target_os = "watchos")
))]
#[allow(dead_code)] // Will be used when SME kernels are implemented
fn detect_apple_sme_tile_size() -> u16 {
  // Helper to read a u32 sysctl value
  fn sysctl_u32(name: &[u8]) -> u32 {
    #[allow(unsafe_code)]
    unsafe extern "C" {
      fn sysctlbyname(
        name: *const u8,
        oldp: *mut core::ffi::c_void,
        oldlenp: *mut usize,
        newp: *const core::ffi::c_void,
        newlen: usize,
      ) -> i32;
    }

    let mut value: u32 = 0;
    let mut size = core::mem::size_of::<u32>();

    #[allow(unsafe_code)]
    // SAFETY: `sysctlbyname` expects `name` to be a valid NUL-terminated C string (caller provides this),
    // `oldp`/`oldlenp` point to writable locals, and `newp` is null with `newlen = 0` (no write).
    let ret = unsafe {
      sysctlbyname(
        name.as_ptr(),
        core::ptr::addr_of_mut!(value).cast(),
        core::ptr::addr_of_mut!(size),
        core::ptr::null(),
        0,
      )
    };

    if ret == 0 { value } else { 0 }
  }

  // Try to read the SVL from sysctl
  let svl_bytes = sysctl_u32(c"hw.optional.arm.sme_max_svl_b".to_bytes_with_nul());
  if svl_bytes > 0 {
    return svl_bytes as u16;
  }

  // Fallback: Use chip generation to infer SVL
  // Apple Silicon uses fixed 128-bit (16 byte) SVL for SME
  if let Some(AppleSiliconGen::M4 | AppleSiliconGen::M5) = detect_apple_silicon_gen() {
    return 16; // 128 bits = 16 bytes
  }

  0 // SME not supported or unknown
}
// SVE Vector Length Detection (Linux aarch64)
// ─────────────────────────────────────────────────────────────────────────────

/// Detect SVE vector length in bits via prctl(PR_SVE_GET_VL).
///
/// Uses raw syscall to avoid libc dependency. Returns 0 if SVE is not supported.
#[cfg(all(target_arch = "aarch64", target_os = "linux", feature = "std"))]
fn detect_sve_vlen() -> u16 {
  // prctl syscall number on aarch64-linux
  const SYS_PRCTL: u64 = 167;
  const PR_SVE_GET_VL: u64 = 51;
  const PR_SVE_VL_LEN_MASK: u64 = 0xFFFF;

  let result: i64;

  // SAFETY: prctl(PR_SVE_GET_VL) is always safe to call.
  // Returns the vector length in bytes on success, or -EINVAL if SVE unsupported.
  #[allow(unsafe_code)]
  unsafe {
    core::arch::asm!(
      "svc #0",
      in("x8") SYS_PRCTL,
      in("x0") PR_SVE_GET_VL,
      in("x1") 0u64,
      in("x2") 0u64,
      in("x3") 0u64,
      in("x4") 0u64,
      lateout("x0") result,
      options(nostack)
    );
  }

  if result < 0 {
    return 0; // SVE not supported
  }

  // Result is VL in bytes; convert to bits
  let vl_bytes = (result as u64) & PR_SVE_VL_LEN_MASK;
  // Saturate to u16::MAX (65535 bits = 8KB, well above SVE's 2048-bit max)
  (vl_bytes.saturating_mul(8)) as u16
}

/// Fallback SVE vector length detection for non-Linux platforms.
#[cfg(all(target_arch = "aarch64", not(all(target_os = "linux", feature = "std"))))]
fn detect_sve_vlen() -> u16 {
  // On non-Linux platforms, we can't easily detect SVE VL.
  // Return 0 to indicate unknown; tuning will use hardcoded defaults.
  0
}

// ─────────────────────────────────────────────────────────────────────────────
// SME Vector Length Detection (Linux aarch64)
// ─────────────────────────────────────────────────────────────────────────────

/// Detect SME streaming vector length in bits via prctl(PR_SME_GET_VL).
///
/// Uses raw syscall to avoid libc dependency. Returns 0 if SME is not supported.
/// The SME vector length determines the tile size (SVL × SVL bits).
#[cfg(all(target_arch = "aarch64", target_os = "linux", feature = "std"))]
#[allow(dead_code)] // Reserved for future Linux SME support (Grace Hopper, future Gravitons)
fn detect_sme_vlen() -> u16 {
  // prctl syscall number on aarch64-linux
  const SYS_PRCTL: u64 = 167;
  const PR_SME_GET_VL: u64 = 63;
  const PR_SME_VL_LEN_MASK: u64 = 0xFFFF;

  let result: i64;

  // SAFETY: prctl(PR_SME_GET_VL) is always safe to call.
  // Returns the streaming vector length in bytes on success, or -EINVAL if SME unsupported.
  #[allow(unsafe_code)]
  unsafe {
    core::arch::asm!(
      "svc #0",
      in("x8") SYS_PRCTL,
      in("x0") PR_SME_GET_VL,
      in("x1") 0u64,
      in("x2") 0u64,
      in("x3") 0u64,
      in("x4") 0u64,
      lateout("x0") result,
      options(nostack)
    );
  }

  if result < 0 {
    return 0; // SME not supported
  }

  // Result is VL in bytes; convert to bits
  let vl_bytes = (result as u64) & PR_SME_VL_LEN_MASK;
  // Saturate to u16::MAX (65535 bits = 8KB, well above SME's typical 256-512 bit max)
  (vl_bytes.saturating_mul(8)) as u16
}

/// Fallback SME vector length detection for non-Linux platforms.
#[cfg(all(target_arch = "aarch64", not(all(target_os = "linux", feature = "std"))))]
#[allow(dead_code)] // May be unused on some aarch64 bare-metal configs
fn detect_sme_vlen() -> u16 {
  // On non-Linux platforms, we can't easily detect SME VL.
  // Return 0 to indicate unknown; tuning will use hardcoded defaults.
  0
}

// ─────────────────────────────────────────────────────────────────────────────
// MIDR_EL1 Detection (Linux aarch64)
// ─────────────────────────────────────────────────────────────────────────────

/// Read MIDR_EL1 (Main ID Register) to identify the CPU part number.
///
/// Linux userspace must not execute privileged `mrs midr_el1` reads.
/// We only use kernel-exposed sysfs data and return `None` when unavailable.
///
/// Returns the full MIDR value, where bits [15:4] contain the part number.
/// Common part numbers:
/// - 0xd0c: Neoverse N1 (Ampere Altra, Graviton 2)
/// - 0xd40: Neoverse V1 (Graviton 3)
/// - 0xd49: Neoverse N2
/// - 0xd4f: Neoverse V2 (NVIDIA Grace, Graviton 4)
/// - 0xd8e: Neoverse N3
#[cfg(all(target_arch = "aarch64", target_os = "linux", feature = "std"))]
fn read_midr_el1() -> Option<u64> {
  use std::fs;

  // Try to read from /sys/devices/system/cpu/cpu0/regs/identification/midr_el1
  if let Ok(contents) = fs::read_to_string("/sys/devices/system/cpu/cpu0/regs/identification/midr_el1")
    && let Ok(midr) = u64::from_str_radix(contents.trim().trim_start_matches("0x"), 16)
  {
    return Some(midr);
  }

  None
}

#[cfg(all(target_arch = "aarch64", target_os = "linux", feature = "std"))]
const MIDR_IMPL_ARM: u8 = 0x41;
#[cfg(all(target_arch = "aarch64", target_os = "linux", feature = "std"))]
const MIDR_IMPL_NVIDIA: u8 = 0x4E;
#[cfg(all(target_arch = "aarch64", target_os = "linux", feature = "std"))]
const MIDR_IMPL_AMPERE: u8 = 0xC0;

#[cfg(all(target_arch = "aarch64", target_os = "linux", feature = "std"))]
const MIDR_PART_NEOVERSE_N1: u16 = 0xD0C;
#[cfg(all(target_arch = "aarch64", target_os = "linux", feature = "std"))]
const MIDR_PART_NEOVERSE_V1: u16 = 0xD40;
#[cfg(all(target_arch = "aarch64", target_os = "linux", feature = "std"))]
const MIDR_PART_NEOVERSE_N2: u16 = 0xD49;
#[cfg(all(target_arch = "aarch64", target_os = "linux", feature = "std"))]
const MIDR_PART_NEOVERSE_V2: u16 = 0xD4F;
#[cfg(all(target_arch = "aarch64", target_os = "linux", feature = "std"))]
const MIDR_PART_NEOVERSE_N3: u16 = 0xD8E;

#[cfg(all(target_arch = "aarch64", target_os = "linux", feature = "std"))]
#[derive(Clone, Copy, Debug)]
struct MidrInfo {
  implementer: u8,
  part: u16,
}

#[cfg(all(target_arch = "aarch64", target_os = "linux", feature = "std"))]
#[inline]
#[must_use]
const fn decode_midr(midr: u64) -> MidrInfo {
  MidrInfo {
    implementer: ((midr >> 24) & 0xFF) as u8,
    part: ((midr >> 4) & 0xFFF) as u16,
  }
}

#[cfg(all(target_arch = "aarch64", target_os = "linux", feature = "std"))]
#[inline]
#[must_use]
fn tune_from_midr_sve2(midr: u64) -> Option<Tune> {
  let info = decode_midr(midr);
  match (info.implementer, info.part) {
    (MIDR_IMPL_NVIDIA, _) => Some(Tune::NVIDIA_GRACE),
    (MIDR_IMPL_AMPERE, _) => Some(Tune::AMPERE_ALTRA),
    (MIDR_IMPL_ARM, MIDR_PART_NEOVERSE_N3) => Some(Tune::NEOVERSE_N3),
    (MIDR_IMPL_ARM, MIDR_PART_NEOVERSE_N2) => Some(Tune::NEOVERSE_N2),
    (MIDR_IMPL_ARM, MIDR_PART_NEOVERSE_V2) => Some(Tune::GRAVITON4),
    (MIDR_IMPL_ARM, MIDR_PART_NEOVERSE_V1) => Some(Tune::GRAVITON3),
    _ => None,
  }
}

#[cfg(all(target_arch = "aarch64", target_os = "linux", feature = "std"))]
#[inline]
#[must_use]
fn tune_from_midr_sve(midr: u64) -> Option<Tune> {
  let info = decode_midr(midr);
  match (info.implementer, info.part) {
    (MIDR_IMPL_NVIDIA, _) => Some(Tune::NVIDIA_GRACE),
    (MIDR_IMPL_AMPERE, _) => Some(Tune::AMPERE_ALTRA),
    (MIDR_IMPL_ARM, MIDR_PART_NEOVERSE_N2) => Some(Tune::NEOVERSE_N2),
    (MIDR_IMPL_ARM, MIDR_PART_NEOVERSE_V1) => Some(Tune::GRAVITON3),
    _ => None,
  }
}

#[cfg(all(target_arch = "aarch64", target_os = "linux", feature = "std"))]
#[inline]
#[must_use]
fn tune_from_midr_pmull_eor3(midr: u64) -> Option<Tune> {
  let info = decode_midr(midr);
  match (info.implementer, info.part) {
    (MIDR_IMPL_AMPERE, _) => Some(Tune::AMPERE_ALTRA),
    (MIDR_IMPL_ARM, MIDR_PART_NEOVERSE_N1) => Some(Tune::GRAVITON2),
    _ => None,
  }
}

#[cfg(target_arch = "aarch64")]
fn select_aarch64_tune(caps: Caps) -> Tune {
  use crate::caps::aarch64;

  // Apple Silicon - use cpufamily detection for precise chip identification
  #[cfg(all(
    feature = "std",
    any(target_os = "macos", target_os = "ios", target_os = "tvos", target_os = "watchos")
  ))]
  {
    if caps.has(aarch64::PMULL_EOR3_READY) {
      match detect_apple_silicon_gen() {
        Some(AppleSiliconGen::M5) => {
          let mut tune = Tune::APPLE_M5;
          // Runtime SME vector length detection
          let sme_vl = detect_sme_vlen();
          if sme_vl > 0 {
            tune.sme_tile = sme_vl;
          }
          return tune;
        }
        Some(AppleSiliconGen::M4) => {
          let mut tune = Tune::APPLE_M4;
          // Runtime SME vector length detection
          let sme_vl = detect_sme_vlen();
          if sme_vl > 0 {
            tune.sme_tile = sme_vl;
          }
          return tune;
        }
        Some(AppleSiliconGen::M1 | AppleSiliconGen::M2 | AppleSiliconGen::M3) => return Tune::APPLE_M1_M3,
        None => {
          // Unknown chip but has PMULL+EOR3 - assume M-series compatible
          // Use SME2 to detect M5+, SME for M4, otherwise M1-M3
          if caps.has(aarch64::SME2) {
            let mut tune = Tune::APPLE_M5;
            let sme_vl = detect_sme_vlen();
            if sme_vl > 0 {
              tune.sme_tile = sme_vl;
            }
            return tune;
          }
          if caps.has(aarch64::SME) {
            let mut tune = Tune::APPLE_M4;
            let sme_vl = detect_sme_vlen();
            if sme_vl > 0 {
              tune.sme_tile = sme_vl;
            }
            return tune;
          }
          return Tune::APPLE_M1_M3;
        }
      }
    }
  }

  // Apple Silicon fallback for no_std or when detection unavailable
  #[cfg(all(
    not(feature = "std"),
    any(target_os = "macos", target_os = "ios", target_os = "tvos", target_os = "watchos")
  ))]
  {
    if caps.has(aarch64::PMULL_EOR3_READY) {
      // Without std, use SME2/SME presence to differentiate generations
      if caps.has(aarch64::SME2) {
        let mut tune = Tune::APPLE_M5;
        let sme_vl = detect_sme_vlen();
        if sme_vl > 0 {
          tune.sme_tile = sme_vl;
        }
        return tune;
      }
      if caps.has(aarch64::SME) {
        let mut tune = Tune::APPLE_M4;
        let sme_vl = detect_sme_vlen();
        if sme_vl > 0 {
          tune.sme_tile = sme_vl;
        }
        return tune;
      }
      return Tune::APPLE_M1_M3;
    }
  }

  // SVE2 with runtime VL detection (Graviton 4/5, Neoverse V2/V3/N3, NVIDIA Grace)
  if caps.has(aarch64::SVE2) {
    let vlen = detect_sve_vlen();

    // Check for V3-specific features (SME2P1 indicates newer generation)
    if caps.has(aarch64::SME2P1) {
      return Tune::GRAVITON5; // or NEOVERSE_V3
    }

    // Try MIDR classification first: this is more precise than SVE VL-only
    // heuristics and distinguishes implementer-specific cores.
    #[cfg(all(feature = "std", target_os = "linux"))]
    {
      if let Some(midr) = read_midr_el1()
        && let Some(tune) = tune_from_midr_sve2(midr)
      {
        return tune;
      }
    }

    // Graviton 4 / Neoverse V2 use 128-bit SVE
    // Neoverse V1 used 256-bit SVE (with SVE1)
    if vlen > 0 && vlen <= 128 {
      return Tune::GRAVITON4; // 128-bit SVE2
    }
    if vlen > 128 {
      return Tune::NEOVERSE_V3;
    }
    // MIDR unavailable and SVE VL unknown: degrade to unknown family.
    return Tune::DEFAULT;
  }

  // SVE (Graviton 3, Neoverse V1) with runtime VL detection
  if caps.has(aarch64::SVE) {
    let vlen = detect_sve_vlen();
    #[cfg(all(feature = "std", target_os = "linux"))]
    {
      if let Some(midr) = read_midr_el1()
        && let Some(tune) = tune_from_midr_sve(midr)
      {
        return tune;
      }
    }
    // Return a tune with the detected SVE vector length
    if vlen >= 256 {
      return Tune::GRAVITON3; // 256-bit SVE
    }
    // Narrower SVE (128-bit) - could be Neoverse N2
    if vlen > 0 && vlen < 256 {
      return Tune::NEOVERSE_N2;
    }
    // MIDR unavailable and SVE VL unknown: degrade to unknown family.
    return Tune::DEFAULT;
  }

  // PMULL + EOR3 (non-Apple, non-SVE) - Graviton2 or Ampere Altra
  if caps.has(aarch64::PMULL_EOR3_READY) {
    // Use MIDR implementer + part to separate Arm N1 (Graviton2) from Ampere.
    #[cfg(all(feature = "std", target_os = "linux"))]
    {
      if let Some(midr) = read_midr_el1()
        && let Some(tune) = tune_from_midr_pmull_eor3(midr)
      {
        return tune;
      }
    }
    // MIDR unavailable: keep PMULL-capable generic tune, avoid misclassifying
    // to a specific microarchitecture family.
    return Tune::AARCH64_PMULL;
  }

  // PMULL only
  if caps.has(aarch64::PMULL_READY) {
    return Tune::AARCH64_PMULL;
  }

  Tune::PORTABLE
}

