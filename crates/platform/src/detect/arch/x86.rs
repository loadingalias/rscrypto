// x86_64 Detection
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
fn detect_x86_64() -> Detected {
  use crate::caps::x86;

  // Start with compile-time detected features (includes SSE2 baseline)
  let caps_static = caps_static();

  // Runtime detection extracts features + vendor/family/model in batch
  #[cfg(feature = "std")]
  let (runtime_caps, is_amd, family, model) = {
    let batch = cpuid_batch_x86_64();
    (batch.caps, batch.is_amd, batch.family, batch.model)
  };
  #[cfg(not(feature = "std"))]
  let (is_amd, family, model) = (false, 0u32, 0u32);

  #[cfg(feature = "std")]
  let mut caps = caps_static.union(runtime_caps);
  #[cfg(not(feature = "std"))]
  let caps = caps_static;

  // ─────────────────────────────────────────────────────────────────────────────
  // Hybrid Intel AVX-512 Safety: Clear AVX-512 caps on hybrid CPUs
  // ─────────────────────────────────────────────────────────────────────────────
  // On hybrid Intel CPUs (Alder Lake, Raptor Lake, etc.), the P-cores have
  // AVX-512 but E-cores don't. If a thread migrates to an E-core while
  // executing AVX-512 code, it will SIGILL. The only safe approach is to
  // disable AVX-512 entirely unless the user explicitly overrides.
  #[cfg(feature = "std")]
  {
    if is_intel_hybrid(is_amd, family, model) && !hybrid_avx512_override() {
      // Clear all AVX-512 related capabilities to prevent kernel selection
      // from choosing AVX-512/VPCLMUL paths that could SIGILL on E-cores.
      caps = caps
        .difference(x86::AVX512F)
        .difference(x86::AVX512DQ)
        .difference(x86::AVX512IFMA)
        .difference(x86::AVX512CD)
        .difference(x86::AVX512BW)
        .difference(x86::AVX512VL)
        .difference(x86::AVX512VBMI)
        .difference(x86::AVX512VBMI2)
        .difference(x86::AVX512VNNI)
        .difference(x86::AVX512BITALG)
        .difference(x86::AVX512VPOPCNTDQ)
        .difference(x86::AVX512BF16)
        .difference(x86::AVX512FP16)
        .difference(x86::VPCLMULQDQ)
        .difference(x86::VAES)
        .difference(x86::GFNI)
        .difference(x86::AVX10_1)
        .difference(x86::AVX10_2);
    }
  }

  let tune = select_x86_tune(caps, is_amd, family, model);

  Detected {
    caps,
    tune,
    arch: Arch::X86_64,
  }
}

#[cfg(target_arch = "x86")]
fn detect_x86() -> Detected {
  use crate::caps::x86;

  // Start with compile-time detected features
  let mut caps = caps_static();

  #[cfg(feature = "std")]
  {
    // SSE2 is not guaranteed on 32-bit x86, detect at runtime
    if std::arch::is_x86_feature_detected!("sse2") {
      caps |= x86::SSE2;
    }
    caps |= runtime_x86_32();
  }

  Detected {
    caps,
    tune: Tune::DEFAULT,
    arch: Arch::X86,
  }
}

/// Batch CPUID result containing all extracted information.
///
/// This struct consolidates all CPUID-derived data to avoid redundant calls.
/// A single call to `cpuid_batch_x86_64()` extracts:
/// - Feature capabilities (Caps)
/// - Vendor identification (Intel/AMD/Unknown)
/// - CPU family and model for microarchitecture selection
#[cfg(all(target_arch = "x86_64", feature = "std"))]
struct CpuidBatch {
  caps: Caps,
  is_amd: bool,
  family: u32,
  model: u32,
}

/// Batch CPUID extraction - extracts all features and CPU info in minimal CPUID calls.
///
/// Makes 5-7 CPUID calls (depending on CPU capabilities):
/// - Leaf 0: vendor string
/// - Leaf 1: processor info + basic features
/// - Leaf 7.0: extended features
/// - Leaf 7.1: more extended features
/// - Leaf 0x24: AVX10 detection (if max leaf >= 0x24)
/// - Leaf 0x29: APX detection (if max leaf >= 0x29)
/// - Leaf 0x80000001: AMD-specific features
///
/// **Critical**: This function properly gates AVX/AVX-512 features by checking
/// OSXSAVE and XGETBV(XCR0) to ensure the OS will save/restore extended registers.
/// Without this check, using AVX/AVX-512 instructions could cause SIGILL.
///
/// # Safety
/// Uses XGETBV, which requires `unsafe` and is only called when OSXSAVE is set.
#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[allow(unsafe_code)]
fn cpuid_batch_x86_64() -> CpuidBatch {
  use core::arch::x86_64::{__cpuid, __cpuid_count, _xgetbv};

  use crate::caps::x86;

  // XCR0 bit masks for OS support verification
  // Bits 1-2: XMM (SSE) + YMM (AVX) state - must be set for AVX
  const XCR0_AVX_MASK: u64 = 0x6;
  // Bits 5-7: opmask + ZMM_Hi256 + Hi16_ZMM state - must be set for AVX-512
  const XCR0_AVX512_MASK: u64 = 0xE0;

  let mut caps = Caps::NONE;

  // CPUID leaf 0: vendor string
  let cpuid0 = __cpuid(0);
  // "AuthenticAMD" has ebx = 0x68747541 ("Auth")
  let is_amd = cpuid0.ebx == 0x6874_7541;

  // CPUID leaf 1: processor info and feature bits
  let cpuid1 = __cpuid(1);

  // Extract extended family (bits 27:20) + base family (bits 11:8)
  let base_family = (cpuid1.eax >> 8) & 0xF;
  let ext_family = (cpuid1.eax >> 20) & 0xFF;
  let family = base_family + ext_family;

  // Extract model (bits 7:4 + extended model bits 19:16 for family 6/15)
  let base_model = (cpuid1.eax >> 4) & 0xF;
  let ext_model = (cpuid1.eax >> 16) & 0xF;
  let model = if base_family == 6 || base_family == 15 {
    base_model + (ext_model << 4)
  } else {
    base_model
  };

  // ─────────────────────────────────────────────────────────────────────────────
  // OS Support Detection via OSXSAVE + XGETBV
  // ─────────────────────────────────────────────────────────────────────────────
  // CRITICAL: CPUID reports what the CPU supports, not what the OS allows.
  // We must check OSXSAVE (indicates OS uses XSAVE) and read XCR0 to verify
  // the OS will actually save/restore AVX/AVX-512 registers. Without this,
  // using AVX instructions on an OS that doesn't save YMM/ZMM state causes SIGILL.

  // OSXSAVE (bit 27): OS has set CR4.OSXSAVE and supports XSAVE/XGETBV
  let osxsave = cpuid1.ecx & (1 << 27) != 0;

  // Read XCR0 if OSXSAVE is enabled, otherwise assume no extended state support
  let xcr0 = if osxsave {
    // SAFETY: XGETBV is safe when OSXSAVE is set (checked above)
    unsafe { _xgetbv(0) }
  } else {
    0
  };

  // Determine OS support for AVX and AVX-512 register state
  let os_avx = (xcr0 & XCR0_AVX_MASK) == XCR0_AVX_MASK;
  let os_avx512 = os_avx && (xcr0 & XCR0_AVX512_MASK) == XCR0_AVX512_MASK;

  // ─────────────────────────────────────────────────────────────────────────────
  // ECX features (leaf 1) - SSE/basic features (no OS gating needed)
  // ─────────────────────────────────────────────────────────────────────────────
  if cpuid1.ecx & (1 << 0) != 0 {
    caps |= x86::SSE3;
  }
  if cpuid1.ecx & (1 << 9) != 0 {
    caps |= x86::SSSE3;
  }
  if cpuid1.ecx & (1 << 19) != 0 {
    caps |= x86::SSE41;
  }
  if cpuid1.ecx & (1 << 20) != 0 {
    caps |= x86::SSE42;
  }
  if cpuid1.ecx & (1 << 23) != 0 {
    caps |= x86::POPCNT;
  }
  if cpuid1.ecx & (1 << 25) != 0 {
    caps |= x86::AESNI;
  }
  if cpuid1.ecx & (1 << 1) != 0 {
    caps |= x86::PCLMULQDQ;
  }
  if cpuid1.ecx & (1 << 30) != 0 {
    caps |= x86::RDRAND;
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // AVX-class features (require OS AVX support via XCR0)
  // ─────────────────────────────────────────────────────────────────────────────
  if os_avx {
    if cpuid1.ecx & (1 << 28) != 0 {
      caps |= x86::AVX;
    }
    if cpuid1.ecx & (1 << 12) != 0 {
      caps |= x86::FMA;
    }
    if cpuid1.ecx & (1 << 29) != 0 {
      caps |= x86::F16C;
    }
  }

  // Extended feature flags (leaf 7, subleaf 0)
  let cpuid7 = __cpuid_count(7, 0);

  // ─────────────────────────────────────────────────────────────────────────────
  // EBX features (leaf 7) - non-AVX features (no OS gating needed)
  // ─────────────────────────────────────────────────────────────────────────────
  if cpuid7.ebx & (1 << 3) != 0 {
    caps |= x86::BMI1;
  }
  if cpuid7.ebx & (1 << 8) != 0 {
    caps |= x86::BMI2;
  }
  if cpuid7.ebx & (1 << 19) != 0 {
    caps |= x86::ADX;
  }
  if cpuid7.ebx & (1 << 29) != 0 {
    caps |= x86::SHA;
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // AVX2 (requires OS AVX support for YMM registers)
  // ─────────────────────────────────────────────────────────────────────────────
  if os_avx && cpuid7.ebx & (1 << 5) != 0 {
    caps |= x86::AVX2;
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // AVX-512 features (require OS AVX-512 support for ZMM/opmask registers)
  // ─────────────────────────────────────────────────────────────────────────────
  if os_avx512 {
    if cpuid7.ebx & (1 << 16) != 0 {
      caps |= x86::AVX512F;
    }
    if cpuid7.ebx & (1 << 17) != 0 {
      caps |= x86::AVX512DQ;
    }
    if cpuid7.ebx & (1 << 21) != 0 {
      caps |= x86::AVX512IFMA;
    }
    if cpuid7.ebx & (1 << 28) != 0 {
      caps |= x86::AVX512CD;
    }
    if cpuid7.ebx & (1 << 30) != 0 {
      caps |= x86::AVX512BW;
    }
    if cpuid7.ebx & (1 << 31) != 0 {
      caps |= x86::AVX512VL;
    }

    // ECX AVX-512 features (leaf 7)
    if cpuid7.ecx & (1 << 1) != 0 {
      caps |= x86::AVX512VBMI;
    }
    if cpuid7.ecx & (1 << 6) != 0 {
      caps |= x86::AVX512VBMI2;
    }
    if cpuid7.ecx & (1 << 11) != 0 {
      caps |= x86::AVX512VNNI;
    }
    if cpuid7.ecx & (1 << 12) != 0 {
      caps |= x86::AVX512BITALG;
    }
    if cpuid7.ecx & (1 << 14) != 0 {
      caps |= x86::AVX512VPOPCNTDQ;
    }

    // Vector extensions that use 512-bit registers (gate with AVX-512 OS support)
    if cpuid7.ecx & (1 << 8) != 0 {
      caps |= x86::GFNI;
    }
    if cpuid7.ecx & (1 << 9) != 0 {
      caps |= x86::VAES;
    }
    if cpuid7.ecx & (1 << 10) != 0 {
      caps |= x86::VPCLMULQDQ;
    }
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // EDX features (leaf 7) - non-AVX features
  // ─────────────────────────────────────────────────────────────────────────────
  if cpuid7.edx & (1 << 18) != 0 {
    caps |= x86::RDSEED;
  }
  if cpuid7.edx & (1 << 24) != 0 {
    caps |= x86::AMX_TILE;
  }
  if cpuid7.edx & (1 << 22) != 0 {
    caps |= x86::AMX_BF16;
  }
  if cpuid7.edx & (1 << 25) != 0 {
    caps |= x86::AMX_INT8;
  }

  // Extended feature flags (leaf 7, subleaf 1)
  let cpuid7_1 = __cpuid_count(7, 1);

  // ─────────────────────────────────────────────────────────────────────────────
  // EAX features (leaf 7, subleaf 1)
  // ─────────────────────────────────────────────────────────────────────────────
  // SHA512 doesn't require AVX-512 (uses XMM registers)
  if cpuid7_1.eax & (1 << 0) != 0 {
    caps |= x86::SHA512;
  }

  // AVX-512 extensions (require OS AVX-512 support)
  if os_avx512 {
    if cpuid7_1.eax & (1 << 4) != 0 {
      caps |= x86::AVX512BF16;
    }
    if cpuid7_1.eax & (1 << 5) != 0 {
      caps |= x86::AVX512FP16;
    }
  }

  // AMX extensions (Granite Rapids and newer) - separate state component
  // Note: AMX has its own XCR0 bits (17-18), but for now we don't gate these
  // as they're not used for crypto kernels
  if cpuid7_1.eax & (1 << 21) != 0 {
    caps |= x86::AMX_FP16;
  }
  if cpuid7_1.eax & (1 << 8) != 0 {
    caps |= x86::AMX_COMPLEX;
  }

  // AVX10 detection via CPUID leaf 0x24 (requires OS AVX-512 support)
  // AVX10 is Intel's unified vector ISA that subsumes AVX-512
  if os_avx512 && cpuid0.eax >= 0x24 {
    let cpuid24 = __cpuid_count(0x24, 0);
    let avx10_version = cpuid24.ebx & 0xFF;
    if avx10_version >= 1 {
      caps |= x86::AVX10_1;
    }
    if avx10_version >= 2 {
      caps |= x86::AVX10_2;
    }
  }

  // APX detection via CPUID leaf 0x29
  // APX doubles GPRs from 16 to 32 (R16-R31) on Granite Rapids+
  if cpuid0.eax >= 0x29 {
    let cpuid29 = __cpuid_count(0x29, 0);
    // APX_NCI_NDD_NF is bit 0 of EBX
    if cpuid29.ebx & 1 != 0 {
      caps |= x86::APX;
    }
  }

  // Extended CPUID (leaf 0x80000001) for AMD-specific features
  let cpuid_ext = __cpuid(0x8000_0001);
  if cpuid_ext.ecx & (1 << 5) != 0 {
    caps |= x86::LZCNT;
  }
  if cpuid_ext.ecx & (1 << 6) != 0 {
    caps |= x86::SSE4A;
  }

  CpuidBatch {
    caps,
    is_amd,
    family,
    model,
  }
}

/// Runtime x86 (32-bit) feature detection using CPUID.
///
/// # Safety
/// Uses CPUID instruction which requires unsafe, but is always safe to call on x86.
#[cfg(all(target_arch = "x86", feature = "std"))]
#[allow(unsafe_code)]
fn runtime_x86_32() -> Caps {
  use core::arch::x86::{__cpuid, __cpuid_count};

  use crate::caps::x86;

  let mut caps = Caps::NONE;

  // CPUID leaf 1: processor info and feature bits
  // SAFETY: CPUID is always safe on x86
  let cpuid1 = unsafe { __cpuid(1) };

  // ECX features (leaf 1)
  if cpuid1.ecx & (1 << 0) != 0 {
    caps |= x86::SSE3;
  }
  if cpuid1.ecx & (1 << 9) != 0 {
    caps |= x86::SSSE3;
  }
  if cpuid1.ecx & (1 << 19) != 0 {
    caps |= x86::SSE41;
  }
  if cpuid1.ecx & (1 << 20) != 0 {
    caps |= x86::SSE42;
  }
  if cpuid1.ecx & (1 << 1) != 0 {
    caps |= x86::PCLMULQDQ;
  }
  if cpuid1.ecx & (1 << 25) != 0 {
    caps |= x86::AESNI;
  }

  caps
}

/// Check if user has explicitly enabled AVX-512 on hybrid Intel CPUs.
///
/// On Alder Lake and newer hybrid Intel CPUs, AVX-512 is disabled by default
/// because E-cores don't support it. Power users who have disabled E-cores
/// in BIOS or are using early unfused chips can set this environment variable
/// to force AVX-512 usage.
///
/// # Environment Variable
///
/// `RSCRYPTO_FORCE_AVX512=1` enables AVX-512 on hybrid Intel CPUs.
#[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "std"))]
fn hybrid_avx512_override() -> bool {
  // Check environment variable for explicit opt-in
  std::env::var("RSCRYPTO_FORCE_AVX512")
    .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
    .unwrap_or(false)
}

/// Detect Intel hybrid CPU (Alder Lake family and newer).
///
/// Returns true if this is an Intel hybrid CPU (P+E cores) where AVX-512
/// is problematic. These CPUs have family 6, model 0x97 (ADL-S), 0x9A (ADL-P),
/// 0xB7 (RPL-S), 0xBA (RPL-P), etc.
#[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "std"))]
fn is_intel_hybrid(is_amd: bool, family: u32, model: u32) -> bool {
  if is_amd {
    return false;
  }

  // Intel uses extended family + base family for family >= 15
  // Family 6 is used for all modern Intel client/server CPUs
  if family != 6 {
    return false;
  }

  // Hybrid CPU models (Alder Lake, Raptor Lake, Meteor Lake, etc.)
  // These have E-cores that don't support AVX-512
  matches!(
    model,
    0x97  // Alder Lake-S (desktop)
    | 0x9A  // Alder Lake-P/H/U (mobile)
    | 0x9C  // Alder Lake-N (low power)
    | 0xB7  // Raptor Lake-S (desktop)
    | 0xBA  // Raptor Lake-P/H (mobile)
    | 0xBF  // Raptor Lake-S refresh
    | 0xAA  // Meteor Lake-H
    | 0xAC  // Meteor Lake-U
    | 0xBD  // Lunar Lake
    | 0xC5  // Arrow Lake-S
    | 0xC6 // Arrow Lake-H
  )
}

/// Detect Intel Ice Lake family models.
///
/// We use model-based identification here because feature-only identification
/// cannot reliably separate Ice Lake AVX-512 parts from newer Intel AVX-512
/// generations in all environments.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
fn is_intel_icl_model(family: u32, model: u32) -> bool {
  // Intel uses family 6 for modern server/client generations.
  if family != 6 {
    return false;
  }

  // Ice Lake known models:
  // - 0x6A, 0x6C: Ice Lake-SP / Ice Lake-D server parts
  // - 0x7D, 0x7E: Ice Lake client derivatives
  matches!(model, 0x6A | 0x6C | 0x7D | 0x7E)
}

/// Detect Intel Sapphire Rapids / Emerald Rapids family models.
///
/// Model-based detection protects tune kind stability when hypervisors mask
/// BF16/AMX capability bits.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
fn is_intel_spr_model(family: u32, model: u32) -> bool {
  if family != 6 {
    return false;
  }

  // Sapphire Rapids (0x8F), Emerald Rapids (0xCF)
  matches!(model, 0x8F | 0xCF)
}

/// Detect Intel Granite Rapids family models.
///
/// Model-based detection protects tune kind stability when AMX FP16/COMPLEX
/// bits are masked by virtualization.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
fn is_intel_gnr_model(family: u32, model: u32) -> bool {
  if family != 6 {
    return false;
  }

  // Granite Rapids (0xAD)
  matches!(model, 0xAD)
}

/// Select tuning preset based on features and pre-extracted CPU info.
///
/// Takes vendor/family/model info extracted from batch CPUID to avoid redundant calls.
/// - `is_amd`: true if vendor string is "AuthenticAMD"
/// - `family`: CPU family (e.g., 25 = Zen 3/4, 26 = Zen 5)
/// - `model`: CPU model within family (used for hybrid detection in std mode)
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[allow(unused_variables)] // `model` only used with std feature for hybrid detection
fn select_x86_tune(caps: Caps, is_amd: bool, family: u32, model: u32) -> Tune {
  use crate::caps::x86;

  // Hybrid Intel note:
  // `detect_x86_64` will clear AVX-512 caps on known hybrid Intel CPUs unless
  // the user explicitly overrides. For tune selection we still want to prefer
  // 256-bit strategies on those systems when AVX2 is available.
  #[cfg(feature = "std")]
  if is_intel_hybrid(is_amd, family, model) && !hybrid_avx512_override() && caps.has(x86::AVX2) {
    return Tune::INTEL_ICL;
  }

  // BLAKE3 (and other vectorized kernels) benefit from AVX-512 on CPUs that
  // support the *base* AVX-512 feature set. Do not gate microarchitecture
  // classification on VPCLMUL/VAES; those are kernel-specific capabilities.
  let has_avx512 = caps.has(x86::AVX512F) && caps.has(x86::AVX512VL);

  if has_avx512 {
    return if is_amd {
      // Zen 5/5c is family 26, Zen 4 is family 25 (models 96-127)
      // Note: Currently no way to differentiate Zen 5c from Zen 5 via CPUID alone.
      // Both share Family 26 (0x1A). Zen 5c is used in EPYC 9005 series and
      // Strix Point APUs with hybrid configurations. For now, use ZEN5 tuning
      // for all Family 26 CPUs. Future: may need OS topology detection or
      // per-core CPUID to identify compact cores in hybrid SKUs.
      if family == 26 {
        Tune::ZEN5
      } else if family == 25 {
        Tune::ZEN4
      } else {
        Tune::DEFAULT
      }
    } else {
      // Intel AVX-512 classification order:
      // 1) Explicit known models (ICL / SPR / GNR)
      // 2) Feature-led fallback (AMX FP16/COMPLEX => GNR; BF16/AMX => SPR)
      // 3) Final fallback => Intel ICL
      if is_intel_icl_model(family, model) {
        Tune::INTEL_ICL
      } else if is_intel_gnr_model(family, model) {
        Tune::INTEL_GNR
      } else if is_intel_spr_model(family, model) {
        Tune::INTEL_SPR
      } else if caps.has(x86::AMX_FP16) || caps.has(x86::AMX_COMPLEX) {
        Tune::INTEL_GNR
      } else if caps.has(x86::AVX512BF16)
        || caps.has(x86::AMX_TILE)
        || caps.has(x86::AMX_BF16)
        || caps.has(x86::AMX_INT8)
      {
        Tune::INTEL_SPR
      } else {
        Tune::INTEL_ICL
      }
    };
  }

  // AVX2-only systems need finer classification than the generic default:
  // - Intel AVX2 parts typically align better with IntelIcl thresholds/policies.
  // - AMD Family 25/26 should stay on Zen-class presets even when AVX-512 is unavailable (e.g.
  //   BIOS-disabled AVX-512, virtualization masks).
  if caps.has(x86::AVX2) {
    if is_amd {
      if family == 26 {
        return Tune::ZEN5;
      }
      if family == 25 {
        return Tune::ZEN4;
      }
      return Tune::DEFAULT;
    }
    return Tune::INTEL_ICL;
  }

  if caps.has(x86::PCLMUL_READY) {
    if !is_amd {
      return Tune::INTEL_ICL;
    }
    return Tune::DEFAULT;
  }

  Tune::PORTABLE
}
