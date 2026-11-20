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
  any(feature = "std", not(miri)),
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
#[cfg(all(target_arch = "aarch64", not(miri)))]
static DIT_STATE: core::sync::atomic::AtomicU8 = core::sync::atomic::AtomicU8::new(0);
#[cfg(all(target_arch = "aarch64", not(miri)))]
const DIT_KNOWN: u8 = 1;
#[cfg(all(target_arch = "aarch64", not(miri)))]
const DIT_PRESENT: u8 = 2;

/// Whether the CPU implements FEAT_DIT (data-independent timing), probed once per process.
///
/// The `DataIndependentTiming` guard reads this. It ignores capability overrides and
/// `portable-only`: DIT is a timing mitigation, not a backend choice. Racing first callers each
/// run the same idempotent probe, so a relaxed byte is the whole cache. Every probed state is
/// nonzero, so an inlined check is one load, a zero test, and a bit test.
#[cfg(all(target_arch = "aarch64", not(miri)))]
#[inline]
pub(crate) fn dit_supported() -> bool {
  use core::sync::atomic::Ordering;

  // A compile-time DIT target, such as `aarch64-apple-darwin`, needs no runtime check.
  if cfg!(target_feature = "dit") {
    return true;
  }
  let state = DIT_STATE.load(Ordering::Relaxed);
  if state == 0 {
    return probe_and_cache_dit();
  }
  state & DIT_PRESENT != 0
}

#[cfg(all(target_arch = "aarch64", not(miri)))]
#[cold]
#[inline(never)]
fn probe_and_cache_dit() -> bool {
  use core::sync::atomic::Ordering;

  let present = probe_dit();
  DIT_STATE.store(DIT_KNOWN | if present { DIT_PRESENT } else { 0 }, Ordering::Relaxed);
  present
}

/// FEAT_DIT from the same sources as `std::arch::is_aarch64_feature_detected!("dit")`.
///
/// The Linux and Apple probes call the C library, not `std`, so `no_std` builds run them too.
#[cfg(all(target_arch = "aarch64", not(miri)))]
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
    // Without `std`, these targets have no safe probe: a read of an ID register or of DIT can trap.
    #[cfg(feature = "std")]
    {
      std::arch::is_aarch64_feature_detected!("dit")
    }
    #[cfg(not(feature = "std"))]
    {
      false
    }
  }
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

