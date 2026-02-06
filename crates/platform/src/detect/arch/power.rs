// Power Detection
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "powerpc64")]
fn detect_power() -> Detected {
  // Start with compile-time detected features.
  #[cfg(feature = "std")]
  let caps = caps_static() | runtime_power();
  #[cfg(not(feature = "std"))]
  let caps = caps_static();

  let tune = select_power_tune(caps);

  Detected {
    caps,
    tune,
    arch: Arch::Power,
  }
}

#[cfg(target_arch = "powerpc64")]
fn select_power_tune(caps: Caps) -> Tune {
  use crate::caps::power;

  if caps.has(power::POWER10_VECTOR) {
    Tune::POWER10
  } else if caps.has(power::POWER9_VECTOR) {
    Tune::POWER9
  } else if caps.has(power::POWER8_VECTOR) {
    Tune::POWER8
  } else if caps.has(power::VSX) {
    Tune::POWER7
  } else {
    Tune::PORTABLE
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Power Runtime Detection
// ─────────────────────────────────────────────────────────────────────────────

/// Runtime Power detection for Linux/Android via /proc/self/auxv.
///
/// This avoids unstable `is_powerpc64_feature_detected!` and keeps the
/// selection logic consistent with the x86/aarch64 batch detectors.
#[cfg(all(
  target_arch = "powerpc64",
  feature = "std",
  any(target_os = "linux", target_os = "android")
))]
fn runtime_power() -> Caps {
  use std::{fs::File, io::Read};

  use crate::caps::power;

  // ELF auxiliary vector entry types
  const AT_HWCAP: u64 = 16;
  const AT_HWCAP2: u64 = 26;

  // HWCAP masks (from linux/arch/powerpc/include/uapi/asm/cputable.h)
  const PPC_FEATURE_HAS_ALTIVEC: u64 = 0x1000_0000;
  const PPC_FEATURE_HAS_VSX: u64 = 0x0000_0080;

  // HWCAP2 masks
  const PPC_FEATURE2_ARCH_2_07: u64 = 0x8000_0000; // POWER8 ISA (v2.07)
  const PPC_FEATURE2_ARCH_3_00: u64 = 0x0080_0000; // POWER9 ISA (v3.00)

  let (hwcap, hwcap2) = (|| -> Option<(u64, u64)> {
    let mut file = File::open("/proc/self/auxv").ok()?;
    let mut buf = [0u8; 4096];
    let n = file.read(&mut buf).ok()?;

    let mut hwcap = 0u64;
    let mut hwcap2 = 0u64;

    for chunk in buf.get(..n)?.chunks_exact(16) {
      let a_type = u64::from_ne_bytes(chunk.get(0..8)?.try_into().ok()?);
      let a_val = u64::from_ne_bytes(chunk.get(8..16)?.try_into().ok()?);

      if a_type == AT_HWCAP {
        hwcap = a_val;
      } else if a_type == AT_HWCAP2 {
        hwcap2 = a_val;
      } else if a_type == 0 {
        break;
      }
    }

    Some((hwcap, hwcap2))
  })()
  .unwrap_or((0, 0));

  let mut caps = Caps::NONE;

  if hwcap & PPC_FEATURE_HAS_ALTIVEC != 0 {
    caps |= power::ALTIVEC;
  }
  if hwcap & PPC_FEATURE_HAS_VSX != 0 {
    caps |= power::VSX;
  }

  // POWER9 implies POWER8 vector/crypto as well.
  if hwcap2 & PPC_FEATURE2_ARCH_3_00 != 0 {
    caps |= power::POWER9_VECTOR | power::POWER8_VECTOR | power::POWER8_CRYPTO;
  } else if hwcap2 & PPC_FEATURE2_ARCH_2_07 != 0 {
    caps |= power::POWER8_VECTOR | power::POWER8_CRYPTO;
  }

  caps
}

/// Runtime Power detection for other platforms.
#[cfg(all(
  target_arch = "powerpc64",
  feature = "std",
  not(any(target_os = "linux", target_os = "android"))
))]
fn runtime_power() -> Caps {
  // No stable runtime detector available on non-Linux today; rely on compile-time
  // `-C target-feature` for static dispatch in those environments.
  Caps::NONE
}

// ─────────────────────────────────────────────────────────────────────────────
