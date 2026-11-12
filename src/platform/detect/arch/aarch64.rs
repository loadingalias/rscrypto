// aarch64 Detection

#[cfg(target_arch = "aarch64")]
fn detect_aarch64() -> Detected {
  // Start with compile-time detected features (includes NEON baseline)
  #[cfg(feature = "std")]
  let caps = caps_static() | runtime_aarch64();
  #[cfg(not(feature = "std"))]
  let caps = caps_static();

  Detected {
    caps,
    arch: Arch::Aarch64,
  }
}

/// Batch extraction of aarch64 features from `AT_HWCAP` and `AT_HWCAP2`.
#[cfg(all(
  target_arch = "aarch64",
  feature = "std",
  any(target_os = "linux", target_os = "android")
))]
fn hwcap_batch_aarch64() -> Caps {
  use crate::platform::caps::aarch64;

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

  let (hwcap, hwcap2) = linux_hwcaps();

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

/// Runtime aarch64 detection for Linux/Android (batch HWCAP from `getauxval`).
#[cfg(all(
  target_arch = "aarch64",
  feature = "std",
  any(target_os = "linux", target_os = "android")
))]
fn runtime_aarch64() -> Caps {
  hwcap_batch_aarch64()
}

/// Runtime aarch64 detection for Apple platforms from `hw.optional` sysctl keys.
///
/// The keys and their combinations are those of `std::arch::is_aarch64_feature_detected!` on
/// Darwin: its `aes` and `pmull` each need both FEAT_AES and FEAT_PMULL, `sha2` needs FEAT_SHA1,
/// FEAT_SHA256, and AdvSIMD, and `sha3` needs FEAT_SHA512, FEAT_SHA3, and AdvSIMD. Darwin reports
/// no SVE, SM4, or RNDR. Apple-specific SME keys follow.
#[cfg(all(
  target_arch = "aarch64",
  feature = "std",
  any(target_os = "macos", target_os = "ios", target_os = "tvos", target_os = "watchos")
))]
fn runtime_aarch64() -> Caps {
  use crate::platform::caps::aarch64;

  let flag = apple_sysctl_flag;
  let asimd = flag(c"hw.optional.AdvSIMD") || flag(c"hw.optional.arm.AdvSIMD");
  let mut caps = Caps::NONE;

  // ─── Crypto Extensions ───
  if flag(c"hw.optional.arm.FEAT_AES") && flag(c"hw.optional.arm.FEAT_PMULL") {
    caps |= aarch64::AES | aarch64::PMULL;
  }
  if asimd && flag(c"hw.optional.arm.FEAT_SHA1") && flag(c"hw.optional.arm.FEAT_SHA256") {
    caps |= aarch64::SHA2;
  }
  if asimd && flag(c"hw.optional.arm.FEAT_SHA512") && flag(c"hw.optional.arm.FEAT_SHA3") {
    caps |= aarch64::SHA3 | aarch64::SHA512;
  }

  // ─── CRC Extension ───
  if flag(c"hw.optional.armv8_crc32") || flag(c"hw.optional.arm.FEAT_CRC32") {
    caps |= aarch64::CRC;
  }

  // ─── Additional SIMD ───
  for (key, cap) in [
    (c"hw.optional.arm.FEAT_DotProd", aarch64::DOTPROD),
    (c"hw.optional.arm.FEAT_FP16", aarch64::FP16),
    (c"hw.optional.arm.FEAT_I8MM", aarch64::I8MM),
    (c"hw.optional.arm.FEAT_BF16", aarch64::BF16),
    (c"hw.optional.arm.FEAT_FRINTTS", aarch64::FRINTTS),
    // ─── Atomics ───
    (c"hw.optional.arm.FEAT_LSE", aarch64::LSE),
    (c"hw.optional.arm.FEAT_LSE2", aarch64::LSE2),
  ] {
    if flag(key) {
      caps |= cap;
    }
  }

  // ─── Memory Operations ───
  // Darwin exposes no MOPS key; compile-time detection carries it.
  #[cfg(target_feature = "mops")]
  {
    caps |= aarch64::MOPS;
  }

  // ─── SME ───
  caps |= detect_apple_sme_features();

  caps
}

/// Runtime aarch64 detection for other platforms (Windows, the BSDs) through the standard library.
#[cfg(all(
  target_arch = "aarch64",
  feature = "std",
  not(any(
    target_os = "linux",
    target_os = "android",
    target_os = "macos",
    target_os = "ios",
    target_os = "tvos",
    target_os = "watchos"
  ))
))]
fn runtime_aarch64() -> Caps {
  use crate::platform::caps::aarch64;

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
  // Compile-time detection carries MOPS on these platforms.
  #[cfg(target_feature = "mops")]
  {
    caps |= aarch64::MOPS;
  }

  // ─── Hardware RNG ───
  if std::arch::is_aarch64_feature_detected!("rand") {
    caps |= aarch64::RNG;
  }

  caps
}

/// Whether the Apple `sysctl` key `name` reads as a nonzero 32-bit value. A missing key is false.
#[cfg(all(
  target_arch = "aarch64",
  feature = "std",
  any(target_os = "macos", target_os = "ios", target_os = "tvos", target_os = "watchos")
))]
fn apple_sysctl_flag(name: &core::ffi::CStr) -> bool {
  // libSystem, always linked on Apple platforms, exports `sysctlbyname(3)` with this C signature.
  unsafe extern "C" {
    fn sysctlbyname(
      name: *const core::ffi::c_char,
      oldp: *mut core::ffi::c_void,
      oldlenp: *mut usize,
      newp: *const core::ffi::c_void,
      newlen: usize,
    ) -> core::ffi::c_int;
  }

  let mut value: u32 = 0;
  let mut size = core::mem::size_of::<u32>();
  // SAFETY: `sysctlbyname` call because:
  // 1. `name` is a NUL-terminated `CStr` that outlives the call.
  // 2. `oldp` and `oldlenp` point to a live, initialized `u32` and its exact byte size; the kernel
  //    writes at most `*oldlenp` bytes and updates the length.
  // 3. A null `newp` with zero `newlen` requests a read only.
  let status = unsafe {
    sysctlbyname(
      name.as_ptr(),
      core::ptr::addr_of_mut!(value).cast(),
      core::ptr::addr_of_mut!(size),
      core::ptr::null(),
      0,
    )
  };
  status == 0 && value != 0
}

/// The FEAT_DIT cache: zero until probed, then `DIT_KNOWN`, plus `DIT_PRESENT` when the CPU has it.
#[cfg(all(target_arch = "aarch64", feature = "std", not(miri)))]
static DIT_STATE: core::sync::atomic::AtomicU8 = core::sync::atomic::AtomicU8::new(0);
#[cfg(all(target_arch = "aarch64", feature = "std", not(miri)))]
const DIT_KNOWN: u8 = 1;
#[cfg(all(target_arch = "aarch64", feature = "std", not(miri)))]
const DIT_PRESENT: u8 = 2;

/// Whether the CPU implements FEAT_DIT (data-independent timing), probed once per process.
///
/// The `DataIndependentTiming` guard reads this. It ignores capability overrides and
/// `portable-only`: DIT is a timing mitigation, not a backend choice. Racing first callers each
/// run the same idempotent probe, so a relaxed byte is the whole cache. Every probed state is
/// nonzero, so an inlined check is one load, a zero test, and a bit test.
#[cfg(all(target_arch = "aarch64", feature = "std", not(miri)))]
#[inline]
pub(crate) fn dit_supported() -> bool {
  use core::sync::atomic::Ordering;

  // A compile-time DIT target (every Apple AArch64 target) needs no runtime check at all.
  if cfg!(target_feature = "dit") {
    return true;
  }
  let state = DIT_STATE.load(Ordering::Relaxed);
  if state == 0 {
    return probe_and_cache_dit();
  }
  state & DIT_PRESENT != 0
}

#[cfg(all(target_arch = "aarch64", feature = "std", not(miri)))]
#[cold]
#[inline(never)]
fn probe_and_cache_dit() -> bool {
  use core::sync::atomic::Ordering;

  let present = probe_dit();
  DIT_STATE.store(DIT_KNOWN | if present { DIT_PRESENT } else { 0 }, Ordering::Relaxed);
  present
}

/// FEAT_DIT from the same sources as `std::arch::is_aarch64_feature_detected!("dit")`.
#[cfg(all(target_arch = "aarch64", feature = "std", not(miri)))]
#[cold]
fn probe_dit() -> bool {
  if cfg!(target_feature = "dit") {
    return true;
  }

  #[cfg(any(target_os = "linux", target_os = "android"))]
  {
    // linux/arch/arm64/include/uapi/asm/hwcap.h
    const HWCAP_DIT: u64 = 1 << 24;
    linux_hwcaps().0 & HWCAP_DIT != 0
  }

  #[cfg(any(target_os = "macos", target_os = "ios", target_os = "tvos", target_os = "watchos"))]
  {
    apple_sysctl_flag(c"hw.optional.arm.FEAT_DIT")
  }

  #[cfg(not(any(
    target_os = "linux",
    target_os = "android",
    target_os = "macos",
    target_os = "ios",
    target_os = "tvos",
    target_os = "watchos"
  )))]
  {
    std::arch::is_aarch64_feature_detected!("dit")
  }
}

// Apple Silicon Detection (macOS/iOS)

/// Apple Silicon chip generation.
#[cfg(all(
  not(miri),
  target_arch = "aarch64",
  feature = "std",
  any(
    test,
    feature = "crc16",
    feature = "crc24",
    feature = "crc32",
    feature = "crc64",
    feature = "adler32"
  ),
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
  M5,
}

/// Microarchitecture family used for aarch64 kernel-table selection.
#[cfg(all(
  not(miri),
  target_arch = "aarch64",
  feature = "std",
  any(
    feature = "crc16",
    feature = "crc24",
    feature = "crc32",
    feature = "crc64",
    feature = "adler32"
  )
))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Aarch64TuneFamily {
  #[cfg(any(target_os = "macos", target_os = "ios", target_os = "tvos", target_os = "watchos"))]
  AppleM1M3,
  #[cfg(any(target_os = "macos", target_os = "ios", target_os = "tvos", target_os = "watchos"))]
  AppleM4M5,
  #[cfg(any(target_os = "linux", target_os = "android"))]
  Graviton2,
  #[cfg(any(target_os = "linux", target_os = "android"))]
  Graviton3,
  #[cfg(any(target_os = "linux", target_os = "android"))]
  Graviton4,
}

/// Detect Apple Silicon generation via sysctlbyname("hw.cpufamily").
///
/// Uses direct extern "C" linkage to libSystem (always linked on Apple platforms)
/// to avoid adding libc as a dependency.
///
/// Returns `None` for unknown/future chips or A-series (pre-M1) processors.
#[cfg(all(
  not(miri),
  target_arch = "aarch64",
  feature = "std",
  any(
    test,
    feature = "crc16",
    feature = "crc24",
    feature = "crc32",
    feature = "crc64",
    feature = "adler32"
  ),
  any(target_os = "macos", target_os = "ios", target_os = "tvos", target_os = "watchos")
))]
fn detect_apple_silicon_gen() -> Option<AppleSiliconGen> {
  // CPUFAMILY constants from Apple's mach/machine.h
  // These identify the CPU microarchitecture, not the marketing name.
  //
  // Reference: Xcode SDK <mach/machine.h>
  // Reference: https://github.com/pytorch/cpuinfo/blob/main/src/arm/mach/init.c
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
  // Reference: Xcode SDK mach/machine.h, pytorch/cpuinfo, p-x9/swift-cpu-info
  const CPUFAMILY_ARM_HIDRA: u32 = 0x1d5a_87e8; // M5
  const CPUFAMILY_ARM_SOTRA: u32 = 0xf76c_5b1a; // M5 Pro/Max
  // A19 family (same generation as M5)
  const CPUFAMILY_ARM_TILOS: u32 = 0x01d7_a72b; // A19
  const CPUFAMILY_ARM_THERA: u32 = 0xab34_5f09; // A19 Pro

  // Direct extern "C" linkage to libSystem's sysctlbyname
  // (libSystem is always linked on Apple platforms)
  // The function signature matches Apple's sysctlbyname(3).
  unsafe extern "C" {
    fn sysctlbyname(
      name: *const core::ffi::c_char,
      oldp: *mut core::ffi::c_void,
      oldlenp: *mut usize,
      newp: *const core::ffi::c_void,
      newlen: usize,
    ) -> core::ffi::c_int;
  }

  let mut cpufamily: u32 = 0;
  let mut size = core::mem::size_of::<u32>();

  // SAFETY: sysctlbyname is safe to call with valid pointers.
  // "hw.cpufamily" is a valid null-terminated string.
  // The output buffer is properly sized for u32.
  let ret = unsafe {
    sysctlbyname(
      c"hw.cpufamily".as_ptr(),
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
    CPUFAMILY_ARM_HIDRA | CPUFAMILY_ARM_SOTRA => Some(AppleSiliconGen::M5),
    // A-series chips - treat as M-series equivalent for tuning
    CPUFAMILY_ARM_COLL => Some(AppleSiliconGen::M2), // A17 Pro ≈ M2 architecture
    CPUFAMILY_ARM_TAHITI | CPUFAMILY_ARM_TUPAI => Some(AppleSiliconGen::M4), // A18 ≈ M4 architecture
    CPUFAMILY_ARM_TILOS | CPUFAMILY_ARM_THERA => Some(AppleSiliconGen::M5), // A19 ≈ M5 architecture
    _ => None,                                       // Unknown future chip - will fall back to feature-based detection
  }
}

/// Detect the aarch64 microarchitecture family used by runtime table selection.
#[cfg(all(
  not(miri),
  target_arch = "aarch64",
  feature = "std",
  any(
    feature = "crc16",
    feature = "crc24",
    feature = "crc32",
    feature = "crc64",
    feature = "adler32"
  )
))]
#[must_use]
pub(crate) fn detect_aarch64_tune_family() -> Option<Aarch64TuneFamily> {
  #[cfg(all(
    feature = "std",
    any(target_os = "macos", target_os = "ios", target_os = "tvos", target_os = "watchos")
  ))]
  if let Some(chip_gen) = detect_apple_silicon_gen() {
    return Some(match chip_gen {
      AppleSiliconGen::M1 | AppleSiliconGen::M2 | AppleSiliconGen::M3 => Aarch64TuneFamily::AppleM1M3,
      AppleSiliconGen::M4 | AppleSiliconGen::M5 => Aarch64TuneFamily::AppleM4M5,
    });
  }

  #[cfg(all(feature = "std", any(target_os = "linux", target_os = "android")))]
  if let Some(family) = detect_linux_aarch64_tune_family() {
    return Some(family);
  }

  None
}

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
  use crate::platform::caps::aarch64;

  let mut caps = Caps::NONE;

  // ─── SME Base and Versions ───
  if apple_sysctl_flag(c"hw.optional.arm.FEAT_SME") {
    caps |= aarch64::SME;
  }
  if apple_sysctl_flag(c"hw.optional.arm.FEAT_SME2") {
    caps |= aarch64::SME2;
  }
  if apple_sysctl_flag(c"hw.optional.arm.FEAT_SME2p1") {
    caps |= aarch64::SME2P1;
  }

  // ─── SME Extended Features ───
  if apple_sysctl_flag(c"hw.optional.arm.FEAT_SME_I16I64") {
    caps |= aarch64::SME_I16I64;
  }
  if apple_sysctl_flag(c"hw.optional.arm.FEAT_SME_F64F64") {
    caps |= aarch64::SME_F64F64;
  }
  if apple_sysctl_flag(c"hw.optional.arm.FEAT_SME_B16B16") {
    caps |= aarch64::SME_B16B16;
  }
  if apple_sysctl_flag(c"hw.optional.arm.FEAT_SME_F16F16") {
    caps |= aarch64::SME_F16F16;
  }

  caps
}

// MIDR_EL1 Detection (Linux aarch64)

#[cfg(all(
  not(miri),
  target_arch = "aarch64",
  feature = "std",
  any(target_os = "linux", target_os = "android"),
  any(
    feature = "crc16",
    feature = "crc24",
    feature = "crc32",
    feature = "crc64",
    feature = "adler32"
  )
))]
const MIDR_IMPLEMENTER_SHIFT: u32 = 24;
#[cfg(all(
  not(miri),
  target_arch = "aarch64",
  feature = "std",
  any(target_os = "linux", target_os = "android"),
  any(
    feature = "crc16",
    feature = "crc24",
    feature = "crc32",
    feature = "crc64",
    feature = "adler32"
  )
))]
const MIDR_PARTNUM_SHIFT: u32 = 4;

#[cfg(all(
  not(miri),
  target_arch = "aarch64",
  feature = "std",
  any(target_os = "linux", target_os = "android"),
  any(
    feature = "crc16",
    feature = "crc24",
    feature = "crc32",
    feature = "crc64",
    feature = "adler32"
  )
))]
const ARM_CPU_IMP_ARM: u32 = 0x41;
#[cfg(all(
  not(miri),
  target_arch = "aarch64",
  feature = "std",
  any(target_os = "linux", target_os = "android"),
  any(
    feature = "crc16",
    feature = "crc24",
    feature = "crc32",
    feature = "crc64",
    feature = "adler32"
  )
))]
const ARM_CPU_PART_NEOVERSE_N1: u32 = 0xD0C;
#[cfg(all(
  not(miri),
  target_arch = "aarch64",
  feature = "std",
  any(target_os = "linux", target_os = "android"),
  any(
    feature = "crc16",
    feature = "crc24",
    feature = "crc32",
    feature = "crc64",
    feature = "adler32"
  )
))]
const ARM_CPU_PART_NEOVERSE_V1: u32 = 0xD40;
#[cfg(all(
  not(miri),
  target_arch = "aarch64",
  feature = "std",
  any(target_os = "linux", target_os = "android"),
  any(
    feature = "crc16",
    feature = "crc24",
    feature = "crc32",
    feature = "crc64",
    feature = "adler32"
  )
))]
const ARM_CPU_PART_NEOVERSE_V2: u32 = 0xD4F;

#[cfg(all(
  not(miri),
  target_arch = "aarch64",
  feature = "std",
  any(target_os = "linux", target_os = "android"),
  any(
    feature = "crc16",
    feature = "crc24",
    feature = "crc32",
    feature = "crc64",
    feature = "adler32"
  )
))]
#[inline]
fn midr_implementer(midr: u32) -> u32 {
  midr >> MIDR_IMPLEMENTER_SHIFT
}

#[cfg(all(
  not(miri),
  target_arch = "aarch64",
  feature = "std",
  any(target_os = "linux", target_os = "android"),
  any(
    feature = "crc16",
    feature = "crc24",
    feature = "crc32",
    feature = "crc64",
    feature = "adler32"
  )
))]
#[inline]
fn midr_partnum(midr: u32) -> u32 {
  (midr >> MIDR_PARTNUM_SHIFT) & 0x0fff
}

#[cfg(all(
  not(miri),
  target_arch = "aarch64",
  feature = "std",
  any(target_os = "linux", target_os = "android"),
  any(
    feature = "crc16",
    feature = "crc24",
    feature = "crc32",
    feature = "crc64",
    feature = "adler32"
  )
))]
fn parse_u32_auto_radix(value: &str) -> Option<u32> {
  let value = value.trim();
  if let Some(hex) = value.strip_prefix("0x").or_else(|| value.strip_prefix("0X")) {
    return u32::from_str_radix(hex, 16).ok();
  }

  value
    .parse::<u32>()
    .ok()
    .or_else(|| u32::from_str_radix(value, 16).ok())
}

#[cfg(all(
  not(miri),
  target_arch = "aarch64",
  feature = "std",
  any(target_os = "linux", target_os = "android"),
  any(
    feature = "crc16",
    feature = "crc24",
    feature = "crc32",
    feature = "crc64",
    feature = "adler32"
  )
))]
fn read_linux_midr_sysfs() -> Option<u32> {
  let paths = [
    "/sys/devices/system/cpu/cpu0/regs/identification/midr_el1",
    "/sys/devices/system/cpu/cpu0/identification/midr_el1",
  ];

  for path in paths {
    if let Ok(text) = std::fs::read_to_string(path)
      && let Some(midr) = parse_u32_auto_radix(&text)
    {
      return Some(midr);
    }
  }

  None
}

#[cfg(all(
  not(miri),
  target_arch = "aarch64",
  feature = "std",
  any(target_os = "linux", target_os = "android"),
  any(
    feature = "crc16",
    feature = "crc24",
    feature = "crc32",
    feature = "crc64",
    feature = "adler32"
  )
))]
fn read_linux_midr_cpuinfo() -> Option<u32> {
  let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").ok()?;
  let mut implementer = None;
  let mut part = None;

  for line in cpuinfo.lines() {
    let Some((key, value)) = line.split_once(':') else {
      continue;
    };
    let key = key.trim();
    let value = value.trim();

    if key.eq_ignore_ascii_case("CPU implementer") {
      implementer = parse_u32_auto_radix(value);
    } else if key.eq_ignore_ascii_case("CPU part") {
      part = parse_u32_auto_radix(value);
    }

    if implementer.is_some() && part.is_some() {
      break;
    }
  }

  Some((implementer? << MIDR_IMPLEMENTER_SHIFT) | (0xF << 16) | (part? << MIDR_PARTNUM_SHIFT))
}

#[cfg(all(
  not(miri),
  target_arch = "aarch64",
  feature = "std",
  any(target_os = "linux", target_os = "android"),
  any(
    feature = "crc16",
    feature = "crc24",
    feature = "crc32",
    feature = "crc64",
    feature = "adler32"
  )
))]
#[inline]
fn map_linux_midr_to_tune_family(midr: u32) -> Option<Aarch64TuneFamily> {
  if midr_implementer(midr) != ARM_CPU_IMP_ARM {
    return None;
  }

  match midr_partnum(midr) {
    ARM_CPU_PART_NEOVERSE_N1 => Some(Aarch64TuneFamily::Graviton2),
    ARM_CPU_PART_NEOVERSE_V1 => Some(Aarch64TuneFamily::Graviton3),
    ARM_CPU_PART_NEOVERSE_V2 => Some(Aarch64TuneFamily::Graviton4),
    _ => None,
  }
}

#[cfg(all(
  not(miri),
  target_arch = "aarch64",
  feature = "std",
  any(target_os = "linux", target_os = "android"),
  any(
    feature = "crc16",
    feature = "crc24",
    feature = "crc32",
    feature = "crc64",
    feature = "adler32"
  )
))]
fn detect_linux_aarch64_tune_family() -> Option<Aarch64TuneFamily> {
  let midr = read_linux_midr_sysfs().or_else(read_linux_midr_cpuinfo)?;
  map_linux_midr_to_tune_family(midr)
}
