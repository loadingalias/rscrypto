//! Kernel benchmarking API for the unified tuning engine.
//!
//! This module exposes kernel function pointers by name, allowing the tuning
//! engine to directly benchmark specific kernels without going through the
//! cached dispatch system.
//!
//! Used by the `tune` crate to benchmark specific kernels.

use alloc::vec::Vec;

use crate::dispatchers::{Crc16Fn, Crc24Fn, Crc32Fn, Crc64Fn};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CRC-16 Kernel Types
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// CRC-16 kernel lookup result.
#[derive(Clone, Copy)]
pub struct Crc16Kernel {
  /// Kernel name.
  pub name: &'static str,
  /// Kernel function pointer.
  pub func: Crc16Fn,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CRC-24 Kernel Types
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// CRC-24 kernel lookup result.
#[derive(Clone, Copy)]
pub struct Crc24Kernel {
  /// Kernel name.
  pub name: &'static str,
  /// Kernel function pointer.
  pub func: Crc24Fn,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CRC-32 Kernel Types
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// CRC-32 kernel lookup result.
#[derive(Clone, Copy)]
pub struct Crc32Kernel {
  /// Kernel name.
  pub name: &'static str,
  /// Kernel function pointer.
  pub func: Crc32Fn,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CRC-64 Kernel Types
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// CRC-64 kernel lookup result.
#[derive(Clone, Copy)]
pub struct Crc64Kernel {
  /// Kernel name.
  pub name: &'static str,
  /// Kernel function pointer.
  pub func: Crc64Fn,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CRC-16 Kernel Lookup
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Get all available CRC-16 kernel names for the current platform.
#[must_use]
pub fn available_crc16_kernels() -> Vec<&'static str> {
  let mut kernels = Vec::new();

  // Always available
  kernels.push("reference");
  kernels.push("portable/slice4");
  kernels.push("portable/slice8");

  #[cfg(target_arch = "x86_64")]
  {
    use crate::crc16::kernels::x86_64;
    let caps = platform::caps();

    if caps.has(platform::caps::x86::PCLMUL_READY) {
      kernels.extend_from_slice(x86_64::PCLMUL_NAMES);
      kernels.push(x86_64::PCLMUL_SMALL);
    }
    if caps.has(platform::caps::x86::VPCLMUL_READY) {
      kernels.extend_from_slice(x86_64::VPCLMUL_NAMES);
    }
  }

  #[cfg(target_arch = "aarch64")]
  {
    use crate::crc16::kernels::aarch64;
    let caps = platform::caps();

    if caps.has(platform::caps::aarch64::PMULL_READY) {
      kernels.extend_from_slice(aarch64::PMULL_NAMES);
      kernels.push(aarch64::PMULL_SMALL);
    }
    if caps.has(platform::caps::aarch64::PMULL_EOR3_READY) {
      kernels.extend_from_slice(aarch64::PMULL_EOR3_NAMES);
    }
  }

  #[cfg(target_arch = "powerpc64")]
  {
    use crate::crc16::kernels::power;
    let caps = platform::caps();

    if caps.has(platform::caps::power::VPMSUM_READY) {
      kernels.extend_from_slice(power::VPMSUM_NAMES);
    }
  }

  #[cfg(target_arch = "s390x")]
  {
    use crate::crc16::kernels::s390x;
    let caps = platform::caps();

    if caps.has(platform::caps::s390x::VECTOR) {
      kernels.extend_from_slice(s390x::VGFM_NAMES);
    }
  }

  #[cfg(target_arch = "riscv64")]
  {
    use platform::caps::riscv;

    use crate::crc16::kernels::riscv64;
    let caps = platform::caps();

    if caps.has(riscv::ZBC) {
      kernels.extend_from_slice(riscv64::ZBC_NAMES);
    }
    if caps.has(riscv::ZVBC) {
      kernels.extend_from_slice(riscv64::ZVBC_NAMES);
    }
  }

  kernels
}

/// Get a CRC-16/CCITT kernel function by name.
#[must_use]
pub fn get_crc16_ccitt_kernel(name: &str) -> Option<Crc16Kernel> {
  use crate::common::{reference::crc16_bitwise, tables::CRC16_CCITT_POLY};

  // Reference kernel
  if name == "reference" || name == "reference/bitwise" {
    fn reference_ccitt(crc: u16, data: &[u8]) -> u16 {
      crc16_bitwise(CRC16_CCITT_POLY, crc, data)
    }
    return Some(Crc16Kernel {
      name: "reference",
      func: reference_ccitt,
    });
  }

  // Portable kernels
  if name == "portable" || name == "portable/slice4" {
    return Some(Crc16Kernel {
      name: "portable/slice4",
      func: crate::crc16::portable::crc16_ccitt_slice4,
    });
  }
  if name == "portable/slice8" {
    return Some(Crc16Kernel {
      name: "portable/slice8",
      func: crate::crc16::portable::crc16_ccitt_slice8,
    });
  }

  // Architecture-specific kernels
  #[cfg(target_arch = "x86_64")]
  {
    if let Some(k) = get_x86_64_crc16_ccitt_kernel(name) {
      return Some(k);
    }
  }

  #[cfg(target_arch = "aarch64")]
  {
    if let Some(k) = get_aarch64_crc16_ccitt_kernel(name) {
      return Some(k);
    }
  }

  #[cfg(target_arch = "powerpc64")]
  {
    if let Some(k) = get_power_crc16_ccitt_kernel(name) {
      return Some(k);
    }
  }

  #[cfg(target_arch = "s390x")]
  {
    if let Some(k) = get_s390x_crc16_ccitt_kernel(name) {
      return Some(k);
    }
  }

  #[cfg(target_arch = "riscv64")]
  {
    if let Some(k) = get_riscv64_crc16_ccitt_kernel(name) {
      return Some(k);
    }
  }

  None
}

/// Get a CRC-16/IBM kernel function by name.
#[must_use]
pub fn get_crc16_ibm_kernel(name: &str) -> Option<Crc16Kernel> {
  use crate::common::{reference::crc16_bitwise, tables::CRC16_IBM_POLY};

  // Reference kernel
  if name == "reference" || name == "reference/bitwise" {
    fn reference_ibm(crc: u16, data: &[u8]) -> u16 {
      crc16_bitwise(CRC16_IBM_POLY, crc, data)
    }
    return Some(Crc16Kernel {
      name: "reference",
      func: reference_ibm,
    });
  }

  // Portable kernels
  if name == "portable" || name == "portable/slice4" {
    return Some(Crc16Kernel {
      name: "portable/slice4",
      func: crate::crc16::portable::crc16_ibm_slice4,
    });
  }
  if name == "portable/slice8" {
    return Some(Crc16Kernel {
      name: "portable/slice8",
      func: crate::crc16::portable::crc16_ibm_slice8,
    });
  }

  // Architecture-specific kernels
  #[cfg(target_arch = "x86_64")]
  {
    if let Some(k) = get_x86_64_crc16_ibm_kernel(name) {
      return Some(k);
    }
  }

  #[cfg(target_arch = "aarch64")]
  {
    if let Some(k) = get_aarch64_crc16_ibm_kernel(name) {
      return Some(k);
    }
  }

  #[cfg(target_arch = "powerpc64")]
  {
    if let Some(k) = get_power_crc16_ibm_kernel(name) {
      return Some(k);
    }
  }

  #[cfg(target_arch = "s390x")]
  {
    if let Some(k) = get_s390x_crc16_ibm_kernel(name) {
      return Some(k);
    }
  }

  #[cfg(target_arch = "riscv64")]
  {
    if let Some(k) = get_riscv64_crc16_ibm_kernel(name) {
      return Some(k);
    }
  }

  None
}

#[cfg(target_arch = "x86_64")]
fn get_x86_64_crc16_ccitt_kernel(name: &str) -> Option<Crc16Kernel> {
  use crate::crc16::kernels::x86_64;
  let caps = platform::caps();

  if caps.has(platform::caps::x86::PCLMUL_READY) {
    if name == x86_64::PCLMUL_SMALL {
      return Some(Crc16Kernel {
        name: x86_64::PCLMUL_SMALL,
        func: x86_64::CCITT_PCLMUL_SMALL_KERNEL,
      });
    }
    for (&k, &func) in x86_64::PCLMUL_NAMES.iter().zip(x86_64::CCITT_PCLMUL.iter()) {
      if name == k {
        return Some(Crc16Kernel { name: k, func });
      }
    }
  }

  if caps.has(platform::caps::x86::VPCLMUL_READY) {
    for (&k, &func) in x86_64::VPCLMUL_NAMES.iter().zip(x86_64::CCITT_VPCLMUL.iter()) {
      if name == k {
        return Some(Crc16Kernel { name: k, func });
      }
    }
  }

  None
}

#[cfg(target_arch = "x86_64")]
fn get_x86_64_crc16_ibm_kernel(name: &str) -> Option<Crc16Kernel> {
  use crate::crc16::kernels::x86_64;
  let caps = platform::caps();

  if caps.has(platform::caps::x86::PCLMUL_READY) {
    if name == x86_64::PCLMUL_SMALL {
      return Some(Crc16Kernel {
        name: x86_64::PCLMUL_SMALL,
        func: x86_64::IBM_PCLMUL_SMALL_KERNEL,
      });
    }
    for (&k, &func) in x86_64::PCLMUL_NAMES.iter().zip(x86_64::IBM_PCLMUL.iter()) {
      if name == k {
        return Some(Crc16Kernel { name: k, func });
      }
    }
  }

  if caps.has(platform::caps::x86::VPCLMUL_READY) {
    for (&k, &func) in x86_64::VPCLMUL_NAMES.iter().zip(x86_64::IBM_VPCLMUL.iter()) {
      if name == k {
        return Some(Crc16Kernel { name: k, func });
      }
    }
  }

  None
}

#[cfg(target_arch = "aarch64")]
fn get_aarch64_crc16_ccitt_kernel(name: &str) -> Option<Crc16Kernel> {
  use crate::crc16::kernels::aarch64;
  let caps = platform::caps();

  if caps.has(platform::caps::aarch64::PMULL_READY) {
    if name == aarch64::PMULL_SMALL {
      return Some(Crc16Kernel {
        name: aarch64::PMULL_SMALL,
        func: aarch64::CCITT_PMULL_SMALL_KERNEL,
      });
    }
    for (&k, &func) in aarch64::PMULL_NAMES.iter().zip(aarch64::CCITT_PMULL.iter()) {
      if name == k {
        return Some(Crc16Kernel { name: k, func });
      }
    }
  }
  if caps.has(platform::caps::aarch64::PMULL_EOR3_READY) {
    for (&k, &func) in aarch64::PMULL_EOR3_NAMES
      .iter()
      .zip(aarch64::CCITT_PMULL_EOR3.iter())
      .take(3)
    {
      if name == k {
        return Some(Crc16Kernel { name: k, func });
      }
    }
  }

  None
}

#[cfg(target_arch = "aarch64")]
fn get_aarch64_crc16_ibm_kernel(name: &str) -> Option<Crc16Kernel> {
  use crate::crc16::kernels::aarch64;
  let caps = platform::caps();

  if caps.has(platform::caps::aarch64::PMULL_READY) {
    if name == aarch64::PMULL_SMALL {
      return Some(Crc16Kernel {
        name: aarch64::PMULL_SMALL,
        func: aarch64::IBM_PMULL_SMALL_KERNEL,
      });
    }
    for (&k, &func) in aarch64::PMULL_NAMES.iter().zip(aarch64::IBM_PMULL.iter()) {
      if name == k {
        return Some(Crc16Kernel { name: k, func });
      }
    }
  }
  if caps.has(platform::caps::aarch64::PMULL_EOR3_READY) {
    for (&k, &func) in aarch64::PMULL_EOR3_NAMES
      .iter()
      .zip(aarch64::IBM_PMULL_EOR3.iter())
      .take(3)
    {
      if name == k {
        return Some(Crc16Kernel { name: k, func });
      }
    }
  }

  None
}

#[cfg(target_arch = "powerpc64")]
fn get_power_crc16_ccitt_kernel(name: &str) -> Option<Crc16Kernel> {
  use crate::crc16::kernels::power;
  let caps = platform::caps();

  if caps.has(platform::caps::power::VPMSUM_READY) {
    for (&k, &func) in power::VPMSUM_NAMES.iter().zip(power::CCITT_VPMSUM.iter()) {
      if name == k {
        return Some(Crc16Kernel { name: k, func });
      }
    }
  }

  None
}

#[cfg(target_arch = "powerpc64")]
fn get_power_crc16_ibm_kernel(name: &str) -> Option<Crc16Kernel> {
  use crate::crc16::kernels::power;
  let caps = platform::caps();

  if caps.has(platform::caps::power::VPMSUM_READY) {
    for (&k, &func) in power::VPMSUM_NAMES.iter().zip(power::IBM_VPMSUM.iter()) {
      if name == k {
        return Some(Crc16Kernel { name: k, func });
      }
    }
  }

  None
}

#[cfg(target_arch = "s390x")]
fn get_s390x_crc16_ccitt_kernel(name: &str) -> Option<Crc16Kernel> {
  use crate::crc16::kernels::s390x;
  let caps = platform::caps();

  if caps.has(platform::caps::s390x::VECTOR) {
    for (&k, &func) in s390x::VGFM_NAMES.iter().zip(s390x::CCITT_VGFM.iter()) {
      if name == k {
        return Some(Crc16Kernel { name: k, func });
      }
    }
  }

  None
}

#[cfg(target_arch = "s390x")]
fn get_s390x_crc16_ibm_kernel(name: &str) -> Option<Crc16Kernel> {
  use crate::crc16::kernels::s390x;
  let caps = platform::caps();

  if caps.has(platform::caps::s390x::VECTOR) {
    for (&k, &func) in s390x::VGFM_NAMES.iter().zip(s390x::IBM_VGFM.iter()) {
      if name == k {
        return Some(Crc16Kernel { name: k, func });
      }
    }
  }

  None
}

#[cfg(target_arch = "riscv64")]
fn get_riscv64_crc16_ccitt_kernel(name: &str) -> Option<Crc16Kernel> {
  use platform::caps::riscv;

  use crate::crc16::kernels::riscv64;
  let caps = platform::caps();

  if caps.has(riscv::ZBC) {
    for (&k, &func) in riscv64::ZBC_NAMES.iter().zip(riscv64::CCITT_ZBC.iter()) {
      if name == k {
        return Some(Crc16Kernel { name: k, func });
      }
    }
  }

  if caps.has(riscv::ZVBC) {
    for (&k, &func) in riscv64::ZVBC_NAMES.iter().zip(riscv64::CCITT_ZVBC.iter()) {
      if name == k {
        return Some(Crc16Kernel { name: k, func });
      }
    }
  }

  None
}

#[cfg(target_arch = "riscv64")]
fn get_riscv64_crc16_ibm_kernel(name: &str) -> Option<Crc16Kernel> {
  use platform::caps::riscv;

  use crate::crc16::kernels::riscv64;
  let caps = platform::caps();

  if caps.has(riscv::ZBC) {
    for (&k, &func) in riscv64::ZBC_NAMES.iter().zip(riscv64::IBM_ZBC.iter()) {
      if name == k {
        return Some(Crc16Kernel { name: k, func });
      }
    }
  }

  if caps.has(riscv::ZVBC) {
    for (&k, &func) in riscv64::ZVBC_NAMES.iter().zip(riscv64::IBM_ZVBC.iter()) {
      if name == k {
        return Some(Crc16Kernel { name: k, func });
      }
    }
  }

  None
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CRC-24 Kernel Lookup
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Get all available CRC-24 kernel names for the current platform.
#[must_use]
pub fn available_crc24_kernels() -> Vec<&'static str> {
  let mut kernels = Vec::new();

  // Always available
  kernels.push("reference");
  kernels.push("portable/slice4");
  kernels.push("portable/slice8");

  #[cfg(target_arch = "x86_64")]
  {
    use crate::crc24::kernels::x86_64;
    let caps = platform::caps();

    if caps.has(platform::caps::x86::PCLMUL_READY) {
      kernels.extend_from_slice(x86_64::PCLMUL_NAMES);
      kernels.push(x86_64::PCLMUL_SMALL);
    }
    if caps.has(platform::caps::x86::VPCLMUL_READY) {
      kernels.extend_from_slice(x86_64::VPCLMUL_NAMES);
    }
  }

  #[cfg(target_arch = "aarch64")]
  {
    use crate::crc24::kernels::aarch64;
    let caps = platform::caps();

    if caps.has(platform::caps::aarch64::PMULL_READY) {
      kernels.extend_from_slice(aarch64::PMULL_NAMES);
      kernels.push(aarch64::PMULL_SMALL);
    }
  }

  #[cfg(target_arch = "powerpc64")]
  {
    use crate::crc24::kernels::power;
    let caps = platform::caps();

    if caps.has(platform::caps::power::VPMSUM_READY) {
      kernels.extend_from_slice(power::VPMSUM_NAMES);
    }
  }

  #[cfg(target_arch = "s390x")]
  {
    use crate::crc24::kernels::s390x;
    let caps = platform::caps();

    if caps.has(platform::caps::s390x::VECTOR) {
      kernels.extend_from_slice(s390x::VGFM_NAMES);
    }
  }

  #[cfg(target_arch = "riscv64")]
  {
    use platform::caps::riscv;

    use crate::crc24::kernels::riscv64;
    let caps = platform::caps();

    if caps.has(riscv::ZBC) {
      kernels.extend_from_slice(riscv64::ZBC_NAMES);
    }
    if caps.has(riscv::ZVBC) {
      kernels.extend_from_slice(riscv64::ZVBC_NAMES);
    }
  }

  kernels
}

/// Get a CRC-24/OpenPGP kernel function by name.
#[must_use]
pub fn get_crc24_openpgp_kernel(name: &str) -> Option<Crc24Kernel> {
  use crate::common::{reference::crc24_bitwise, tables::CRC24_OPENPGP_POLY};

  // Reference kernel
  if name == "reference" || name == "reference/bitwise" {
    fn reference_openpgp(crc: u32, data: &[u8]) -> u32 {
      crc24_bitwise(CRC24_OPENPGP_POLY, crc, data)
    }
    return Some(Crc24Kernel {
      name: "reference",
      func: reference_openpgp,
    });
  }

  // Portable kernels
  if name == "portable" || name == "portable/slice4" {
    return Some(Crc24Kernel {
      name: "portable/slice4",
      func: crate::crc24::portable::crc24_openpgp_slice4,
    });
  }
  if name == "portable/slice8" {
    return Some(Crc24Kernel {
      name: "portable/slice8",
      func: crate::crc24::portable::crc24_openpgp_slice8,
    });
  }

  #[cfg(target_arch = "x86_64")]
  {
    use crate::crc24::kernels::x86_64;
    let caps = platform::caps();

    if caps.has(platform::caps::x86::PCLMUL_READY) {
      if name == x86_64::PCLMUL_SMALL {
        return Some(Crc24Kernel {
          name: x86_64::PCLMUL_SMALL,
          func: x86_64::OPENPGP_PCLMUL_SMALL_KERNEL,
        });
      }
      for (&k, &func) in x86_64::PCLMUL_NAMES.iter().zip(x86_64::OPENPGP_PCLMUL.iter()) {
        if name == k {
          return Some(Crc24Kernel { name: k, func });
        }
      }
    }

    if caps.has(platform::caps::x86::VPCLMUL_READY) {
      for (&k, &func) in x86_64::VPCLMUL_NAMES.iter().zip(x86_64::OPENPGP_VPCLMUL.iter()) {
        if name == k {
          return Some(Crc24Kernel { name: k, func });
        }
      }
    }
  }

  #[cfg(target_arch = "aarch64")]
  {
    use crate::crc24::kernels::aarch64;
    let caps = platform::caps();

    if caps.has(platform::caps::aarch64::PMULL_READY) {
      if name == aarch64::PMULL_SMALL {
        return Some(Crc24Kernel {
          name: aarch64::PMULL_SMALL,
          func: aarch64::OPENPGP_PMULL_SMALL_KERNEL,
        });
      }
      for (&k, &func) in aarch64::PMULL_NAMES.iter().zip(aarch64::OPENPGP_PMULL.iter()) {
        if name == k {
          return Some(Crc24Kernel { name: k, func });
        }
      }
    }
  }

  #[cfg(target_arch = "powerpc64")]
  {
    use crate::crc24::kernels::power;
    let caps = platform::caps();

    if caps.has(platform::caps::power::VPMSUM_READY) {
      for (&k, &func) in power::VPMSUM_NAMES.iter().zip(power::OPENPGP_VPMSUM.iter()) {
        if name == k {
          return Some(Crc24Kernel { name: k, func });
        }
      }
    }
  }

  #[cfg(target_arch = "s390x")]
  {
    use crate::crc24::kernels::s390x;
    let caps = platform::caps();

    if caps.has(platform::caps::s390x::VECTOR) {
      for (&k, &func) in s390x::VGFM_NAMES.iter().zip(s390x::OPENPGP_VGFM.iter()) {
        if name == k {
          return Some(Crc24Kernel { name: k, func });
        }
      }
    }
  }

  #[cfg(target_arch = "riscv64")]
  {
    use platform::caps::riscv;

    use crate::crc24::kernels::riscv64;
    let caps = platform::caps();

    if caps.has(riscv::ZBC) {
      for (&k, &func) in riscv64::ZBC_NAMES.iter().zip(riscv64::OPENPGP_ZBC.iter()) {
        if name == k {
          return Some(Crc24Kernel { name: k, func });
        }
      }
    }

    if caps.has(riscv::ZVBC) {
      for (&k, &func) in riscv64::ZVBC_NAMES.iter().zip(riscv64::OPENPGP_ZVBC.iter()) {
        if name == k {
          return Some(Crc24Kernel { name: k, func });
        }
      }
    }
  }

  None
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CRC-32 Kernel Lookup
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Get all available CRC-32 IEEE kernel names for the current platform.
#[must_use]
pub fn available_crc32_ieee_kernels() -> Vec<&'static str> {
  let mut kernels = Vec::new();

  // Always available
  kernels.push("reference");
  kernels.push("portable/bytewise");
  kernels.push("portable/slice16");

  #[cfg(target_arch = "x86_64")]
  {
    use crate::crc32::kernels::x86_64;
    let caps = platform::caps();

    if caps.has(platform::caps::x86::PCLMUL_READY) {
      kernels.extend_from_slice(x86_64::CRC32_PCLMUL_NAMES);
      kernels.push(x86_64::CRC32_PCLMUL_SMALL);
    }
    if caps.has(platform::caps::x86::VPCLMUL_READY) {
      kernels.extend_from_slice(x86_64::CRC32_VPCLMUL_NAMES);
      kernels.push(x86_64::CRC32_VPCLMUL_SMALL);
    }
  }

  #[cfg(target_arch = "aarch64")]
  {
    use crate::crc32::kernels::aarch64;
    let caps = platform::caps();

    if caps.has(platform::caps::aarch64::CRC_READY) {
      kernels.push("aarch64/hwcrc");
      kernels.push("aarch64/hwcrc-2way");
      kernels.push("aarch64/hwcrc-3way");
    }
    if caps.has(platform::caps::aarch64::PMULL_READY) && caps.has(platform::caps::aarch64::CRC_READY) {
      kernels.extend_from_slice(aarch64::CRC32_PMULL_NAMES);
      kernels.push(aarch64::PMULL_SMALL);
    }
    if caps.has(platform::caps::aarch64::PMULL_EOR3_READY) && caps.has(platform::caps::aarch64::CRC_READY) {
      kernels.extend_from_slice(aarch64::CRC32_PMULL_EOR3_NAMES);
    }
    if caps.has(platform::caps::aarch64::SVE2_PMULL)
      && caps.has(platform::caps::aarch64::PMULL_READY)
      && caps.has(platform::caps::aarch64::CRC_READY)
    {
      kernels.extend_from_slice(aarch64::CRC32_SVE2_PMULL_NAMES);
      kernels.push(aarch64::SVE2_PMULL_SMALL);
    }
  }

  #[cfg(target_arch = "powerpc64")]
  {
    use crate::crc32::kernels::power;
    let caps = platform::caps();

    if caps.has(platform::caps::power::VPMSUM_READY) {
      kernels.extend_from_slice(power::CRC32_VPMSUM_NAMES);
    }
  }

  #[cfg(target_arch = "s390x")]
  {
    use crate::crc32::kernels::s390x;
    let caps = platform::caps();

    if caps.has(platform::caps::s390x::VECTOR) {
      kernels.extend_from_slice(s390x::CRC32_VGFM_NAMES);
    }
  }

  #[cfg(target_arch = "riscv64")]
  {
    use crate::crc32::kernels::riscv64;
    let caps = platform::caps();

    if caps.has(platform::caps::riscv::ZBC) {
      kernels.extend_from_slice(riscv64::CRC32_ZBC_NAMES);
    }
    if caps.has(platform::caps::riscv::ZVBC) {
      kernels.extend_from_slice(riscv64::CRC32_ZVBC_NAMES);
    }
  }

  kernels
}

/// Get all available CRC-32C kernel names for the current platform.
#[must_use]
pub fn available_crc32c_kernels() -> Vec<&'static str> {
  let mut kernels = Vec::new();

  // Always available
  kernels.push("reference");
  kernels.push("portable/bytewise");
  kernels.push("portable/slice16");

  #[cfg(target_arch = "x86_64")]
  {
    use crate::crc32::kernels::x86_64;
    let caps = platform::caps();

    if caps.has(platform::caps::x86::CRC32C_READY) {
      kernels.extend_from_slice(x86_64::CRC32C_HWCRC_NAMES);
    }
    if caps.has(platform::caps::x86::CRC32C_READY) && caps.has(platform::caps::x86::PCLMUL_READY) {
      kernels.extend_from_slice(x86_64::CRC32C_FUSION_SSE_NAMES);
    }
    if caps.has(platform::caps::x86::AVX512F) && caps.has(platform::caps::x86::PCLMUL_READY) {
      kernels.extend_from_slice(x86_64::CRC32C_FUSION_AVX512_NAMES);
    }
    if caps.has(platform::caps::x86::VPCLMUL_READY) {
      kernels.extend_from_slice(x86_64::CRC32C_FUSION_VPCLMUL_NAMES);
    }
  }

  #[cfg(target_arch = "aarch64")]
  {
    use crate::crc32::kernels::aarch64;
    let caps = platform::caps();

    if caps.has(platform::caps::aarch64::CRC_READY) {
      kernels.push("aarch64/hwcrc");
      kernels.push("aarch64/hwcrc-2way");
      kernels.push("aarch64/hwcrc-3way");
    }
    if caps.has(platform::caps::aarch64::PMULL_READY) && caps.has(platform::caps::aarch64::CRC_READY) {
      kernels.extend_from_slice(aarch64::CRC32C_PMULL_NAMES);
      kernels.push(aarch64::PMULL_SMALL);
    }
    if caps.has(platform::caps::aarch64::PMULL_EOR3_READY) && caps.has(platform::caps::aarch64::CRC_READY) {
      kernels.extend_from_slice(aarch64::CRC32C_PMULL_EOR3_NAMES);
    }
    if caps.has(platform::caps::aarch64::SVE2_PMULL)
      && caps.has(platform::caps::aarch64::PMULL_READY)
      && caps.has(platform::caps::aarch64::CRC_READY)
    {
      kernels.extend_from_slice(aarch64::CRC32C_SVE2_PMULL_NAMES);
      kernels.push(aarch64::SVE2_PMULL_SMALL);
    }
  }

  #[cfg(target_arch = "powerpc64")]
  {
    use crate::crc32::kernels::power;
    let caps = platform::caps();

    if caps.has(platform::caps::power::VPMSUM_READY) {
      kernels.extend_from_slice(power::CRC32C_VPMSUM_NAMES);
    }
  }

  #[cfg(target_arch = "s390x")]
  {
    use crate::crc32::kernels::s390x;
    let caps = platform::caps();

    if caps.has(platform::caps::s390x::VECTOR) {
      kernels.extend_from_slice(s390x::CRC32C_VGFM_NAMES);
    }
  }

  #[cfg(target_arch = "riscv64")]
  {
    use crate::crc32::kernels::riscv64;
    let caps = platform::caps();

    if caps.has(platform::caps::riscv::ZBC) {
      kernels.extend_from_slice(riscv64::CRC32C_ZBC_NAMES);
    }
    if caps.has(platform::caps::riscv::ZVBC) {
      kernels.extend_from_slice(riscv64::CRC32C_ZVBC_NAMES);
    }
  }

  kernels
}

/// Get a CRC-32 IEEE kernel function by name.
#[must_use]
pub fn get_crc32_ieee_kernel(name: &str) -> Option<Crc32Kernel> {
  use crate::common::{reference::crc32_bitwise, tables::CRC32_IEEE_POLY};

  // Reference kernel
  if name == "reference" || name == "reference/bitwise" {
    fn reference_ieee(crc: u32, data: &[u8]) -> u32 {
      crc32_bitwise(CRC32_IEEE_POLY, crc, data)
    }
    return Some(Crc32Kernel {
      name: "reference",
      func: reference_ieee,
    });
  }

  // Portable kernel
  if name == "portable" || name == "portable/slice16" {
    return Some(Crc32Kernel {
      name: "portable/slice16",
      func: crate::crc32::portable::crc32_slice16_ieee,
    });
  }
  if name == "portable/bytewise" {
    return Some(Crc32Kernel {
      name: "portable/bytewise",
      func: crate::crc32::portable::crc32_bytewise_ieee,
    });
  }

  // Architecture-specific kernels
  #[cfg(target_arch = "x86_64")]
  {
    if let Some(k) = get_x86_64_crc32_ieee_kernel(name) {
      return Some(k);
    }
  }

  #[cfg(target_arch = "aarch64")]
  {
    if let Some(k) = get_aarch64_crc32_ieee_kernel(name) {
      return Some(k);
    }
  }

  #[cfg(target_arch = "powerpc64")]
  {
    if let Some(k) = get_power_crc32_ieee_kernel(name) {
      return Some(k);
    }
  }

  #[cfg(target_arch = "s390x")]
  {
    if let Some(k) = get_s390x_crc32_ieee_kernel(name) {
      return Some(k);
    }
  }

  #[cfg(target_arch = "riscv64")]
  {
    if let Some(k) = get_riscv64_crc32_ieee_kernel(name) {
      return Some(k);
    }
  }

  None
}

/// Get a CRC-32C (Castagnoli) kernel function by name.
#[must_use]
pub fn get_crc32c_kernel(name: &str) -> Option<Crc32Kernel> {
  use crate::common::{reference::crc32_bitwise, tables::CRC32C_POLY};

  // Reference kernel
  if name == "reference" || name == "reference/bitwise" {
    fn reference_crc32c(crc: u32, data: &[u8]) -> u32 {
      crc32_bitwise(CRC32C_POLY, crc, data)
    }
    return Some(Crc32Kernel {
      name: "reference",
      func: reference_crc32c,
    });
  }

  // Portable kernel
  if name == "portable" || name == "portable/slice16" {
    return Some(Crc32Kernel {
      name: "portable/slice16",
      func: crate::crc32::portable::crc32c_slice16,
    });
  }
  if name == "portable/bytewise" {
    return Some(Crc32Kernel {
      name: "portable/bytewise",
      func: crate::crc32::portable::crc32c_bytewise,
    });
  }

  // Architecture-specific kernels
  #[cfg(target_arch = "x86_64")]
  {
    if let Some(k) = get_x86_64_crc32c_kernel(name) {
      return Some(k);
    }
  }

  #[cfg(target_arch = "aarch64")]
  {
    if let Some(k) = get_aarch64_crc32c_kernel(name) {
      return Some(k);
    }
  }

  #[cfg(target_arch = "powerpc64")]
  {
    if let Some(k) = get_power_crc32c_kernel(name) {
      return Some(k);
    }
  }

  #[cfg(target_arch = "s390x")]
  {
    if let Some(k) = get_s390x_crc32c_kernel(name) {
      return Some(k);
    }
  }

  #[cfg(target_arch = "riscv64")]
  {
    if let Some(k) = get_riscv64_crc32c_kernel(name) {
      return Some(k);
    }
  }

  None
}

// ─────────────────────────────────────────────────────────────────────────────
// CRC-32 Architecture-specific kernel lookup
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
fn get_x86_64_crc32_ieee_kernel(name: &str) -> Option<Crc32Kernel> {
  use crate::crc32::kernels::x86_64::*;
  let caps = platform::caps();

  // PCLMUL kernels
  if caps.has(platform::caps::x86::PCLMUL_READY) {
    if name == CRC32_PCLMUL_SMALL {
      return Some(Crc32Kernel {
        name: CRC32_PCLMUL_SMALL,
        func: CRC32_PCLMUL_SMALL_KERNEL,
      });
    }
    for (i, &kernel_name) in CRC32_PCLMUL_NAMES.iter().enumerate() {
      if name == kernel_name {
        return Some(Crc32Kernel {
          name: kernel_name,
          func: CRC32_PCLMUL[i],
        });
      }
    }
  }

  // VPCLMUL kernels
  if caps.has(platform::caps::x86::VPCLMUL_READY) {
    if name == CRC32_VPCLMUL_SMALL {
      return Some(Crc32Kernel {
        name: CRC32_VPCLMUL_SMALL,
        func: CRC32_VPCLMUL_SMALL_KERNEL,
      });
    }
    for (i, &kernel_name) in CRC32_VPCLMUL_NAMES.iter().enumerate() {
      if name == kernel_name {
        return Some(Crc32Kernel {
          name: kernel_name,
          func: CRC32_VPCLMUL[i],
        });
      }
    }
  }

  None
}

#[cfg(target_arch = "x86_64")]
fn get_x86_64_crc32c_kernel(name: &str) -> Option<Crc32Kernel> {
  use crate::crc32::kernels::x86_64::*;
  let caps = platform::caps();

  // SSE4.2 CRC32 instruction
  if caps.has(platform::caps::x86::CRC32C_READY) {
    for (i, &kernel_name) in CRC32C_HWCRC_NAMES.iter().enumerate() {
      if name == kernel_name {
        return Some(Crc32Kernel {
          name: kernel_name,
          func: CRC32C_HWCRC[i],
        });
      }
    }
  }

  // Fusion SSE kernels
  if caps.has(platform::caps::x86::CRC32C_READY) && caps.has(platform::caps::x86::PCLMUL_READY) {
    for (i, &kernel_name) in CRC32C_FUSION_SSE_NAMES.iter().enumerate() {
      if name == kernel_name {
        return Some(Crc32Kernel {
          name: kernel_name,
          func: CRC32C_FUSION_SSE[i],
        });
      }
    }
  }

  // Fusion AVX-512 kernels
  if caps.has(platform::caps::x86::AVX512F) && caps.has(platform::caps::x86::PCLMUL_READY) {
    for (i, &kernel_name) in CRC32C_FUSION_AVX512_NAMES.iter().enumerate() {
      if name == kernel_name {
        return Some(Crc32Kernel {
          name: kernel_name,
          func: CRC32C_FUSION_AVX512[i],
        });
      }
    }
  }

  // Fusion VPCLMUL kernels
  if caps.has(platform::caps::x86::VPCLMUL_READY) {
    for (i, &kernel_name) in CRC32C_FUSION_VPCLMUL_NAMES.iter().enumerate() {
      if name == kernel_name {
        return Some(Crc32Kernel {
          name: kernel_name,
          func: CRC32C_FUSION_VPCLMUL[i],
        });
      }
    }
  }

  None
}

#[cfg(target_arch = "aarch64")]
fn get_aarch64_crc32_ieee_kernel(name: &str) -> Option<Crc32Kernel> {
  use crate::crc32::kernels::aarch64;
  let caps = platform::caps();

  // Hardware CRC
  if caps.has(platform::caps::aarch64::CRC_READY) {
    for (i, &kernel_name) in aarch64::CRC32_HWCRC_NAMES.iter().enumerate() {
      if name == kernel_name {
        return Some(Crc32Kernel {
          name: kernel_name,
          func: aarch64::CRC32_HWCRC[i],
        });
      }
    }
  }

  // PMULL kernels
  if caps.has(platform::caps::aarch64::PMULL_READY) && caps.has(platform::caps::aarch64::CRC_READY) {
    if name == aarch64::PMULL_SMALL {
      return Some(Crc32Kernel {
        name: aarch64::PMULL_SMALL,
        func: aarch64::CRC32_PMULL_SMALL_KERNEL,
      });
    }
    for (i, &kernel_name) in aarch64::CRC32_PMULL_NAMES.iter().enumerate() {
      if name == kernel_name {
        return Some(Crc32Kernel {
          name: kernel_name,
          func: aarch64::CRC32_PMULL[i],
        });
      }
    }
  }

  // PMULL+EOR3 kernels
  if caps.has(platform::caps::aarch64::PMULL_EOR3_READY) && caps.has(platform::caps::aarch64::CRC_READY) {
    for (i, &kernel_name) in aarch64::CRC32_PMULL_EOR3_NAMES.iter().enumerate() {
      if name == kernel_name {
        return Some(Crc32Kernel {
          name: kernel_name,
          func: aarch64::CRC32_PMULL_EOR3[i],
        });
      }
    }
  }

  // SVE2 PMULL kernels
  if caps.has(platform::caps::aarch64::SVE2_PMULL)
    && caps.has(platform::caps::aarch64::PMULL_READY)
    && caps.has(platform::caps::aarch64::CRC_READY)
  {
    if name == aarch64::SVE2_PMULL_SMALL {
      return Some(Crc32Kernel {
        name: aarch64::SVE2_PMULL_SMALL,
        func: aarch64::CRC32_SVE2_PMULL_SMALL_KERNEL,
      });
    }
    for (i, &kernel_name) in aarch64::CRC32_SVE2_PMULL_NAMES.iter().enumerate() {
      if name == kernel_name {
        return Some(Crc32Kernel {
          name: kernel_name,
          func: aarch64::CRC32_SVE2_PMULL[i],
        });
      }
    }
  }

  None
}

#[cfg(target_arch = "aarch64")]
fn get_aarch64_crc32c_kernel(name: &str) -> Option<Crc32Kernel> {
  use crate::crc32::kernels::aarch64;
  let caps = platform::caps();

  // Hardware CRC
  if caps.has(platform::caps::aarch64::CRC_READY) {
    for (i, &kernel_name) in aarch64::CRC32C_HWCRC_NAMES.iter().enumerate() {
      if name == kernel_name {
        return Some(Crc32Kernel {
          name: kernel_name,
          func: aarch64::CRC32C_HWCRC[i],
        });
      }
    }
  }

  // PMULL kernels
  if caps.has(platform::caps::aarch64::PMULL_READY) && caps.has(platform::caps::aarch64::CRC_READY) {
    if name == aarch64::PMULL_SMALL {
      return Some(Crc32Kernel {
        name: aarch64::PMULL_SMALL,
        func: aarch64::CRC32C_PMULL_SMALL_KERNEL,
      });
    }
    for (i, &kernel_name) in aarch64::CRC32C_PMULL_NAMES.iter().enumerate() {
      if name == kernel_name {
        return Some(Crc32Kernel {
          name: kernel_name,
          func: aarch64::CRC32C_PMULL[i],
        });
      }
    }
  }

  // PMULL+EOR3 kernels
  if caps.has(platform::caps::aarch64::PMULL_EOR3_READY) && caps.has(platform::caps::aarch64::CRC_READY) {
    for (i, &kernel_name) in aarch64::CRC32C_PMULL_EOR3_NAMES.iter().enumerate() {
      if name == kernel_name {
        return Some(Crc32Kernel {
          name: kernel_name,
          func: aarch64::CRC32C_PMULL_EOR3[i],
        });
      }
    }
  }

  // SVE2 PMULL kernels
  if caps.has(platform::caps::aarch64::SVE2_PMULL)
    && caps.has(platform::caps::aarch64::PMULL_READY)
    && caps.has(platform::caps::aarch64::CRC_READY)
  {
    if name == aarch64::SVE2_PMULL_SMALL {
      return Some(Crc32Kernel {
        name: aarch64::SVE2_PMULL_SMALL,
        func: aarch64::CRC32C_SVE2_PMULL_SMALL_KERNEL,
      });
    }
    for (i, &kernel_name) in aarch64::CRC32C_SVE2_PMULL_NAMES.iter().enumerate() {
      if name == kernel_name {
        return Some(Crc32Kernel {
          name: kernel_name,
          func: aarch64::CRC32C_SVE2_PMULL[i],
        });
      }
    }
  }

  None
}

#[cfg(target_arch = "powerpc64")]
fn get_power_crc32_ieee_kernel(name: &str) -> Option<Crc32Kernel> {
  use crate::crc32::kernels::power::*;
  let caps = platform::caps();

  if caps.has(platform::caps::power::VPMSUM_READY) {
    for (i, &kernel_name) in CRC32_VPMSUM_NAMES.iter().enumerate() {
      if name == kernel_name {
        return Some(Crc32Kernel {
          name: kernel_name,
          func: CRC32_VPMSUM[i],
        });
      }
    }
  }

  None
}

#[cfg(target_arch = "powerpc64")]
fn get_power_crc32c_kernel(name: &str) -> Option<Crc32Kernel> {
  use crate::crc32::kernels::power::*;
  let caps = platform::caps();

  if caps.has(platform::caps::power::VPMSUM_READY) {
    for (i, &kernel_name) in CRC32C_VPMSUM_NAMES.iter().enumerate() {
      if name == kernel_name {
        return Some(Crc32Kernel {
          name: kernel_name,
          func: CRC32C_VPMSUM[i],
        });
      }
    }
  }

  None
}

#[cfg(target_arch = "s390x")]
fn get_s390x_crc32_ieee_kernel(name: &str) -> Option<Crc32Kernel> {
  use crate::crc32::kernels::s390x::*;
  let caps = platform::caps();

  if caps.has(platform::caps::s390x::VECTOR) {
    for (i, &kernel_name) in CRC32_VGFM_NAMES.iter().enumerate() {
      if name == kernel_name {
        return Some(Crc32Kernel {
          name: kernel_name,
          func: CRC32_VGFM[i],
        });
      }
    }
  }

  None
}

#[cfg(target_arch = "s390x")]
fn get_s390x_crc32c_kernel(name: &str) -> Option<Crc32Kernel> {
  use crate::crc32::kernels::s390x::*;
  let caps = platform::caps();

  if caps.has(platform::caps::s390x::VECTOR) {
    for (i, &kernel_name) in CRC32C_VGFM_NAMES.iter().enumerate() {
      if name == kernel_name {
        return Some(Crc32Kernel {
          name: kernel_name,
          func: CRC32C_VGFM[i],
        });
      }
    }
  }

  None
}

#[cfg(target_arch = "riscv64")]
fn get_riscv64_crc32_ieee_kernel(name: &str) -> Option<Crc32Kernel> {
  use crate::crc32::kernels::riscv64::*;
  let caps = platform::caps();

  if caps.has(platform::caps::riscv::ZBC) {
    for (i, &kernel_name) in CRC32_ZBC_NAMES.iter().enumerate() {
      if name == kernel_name {
        return Some(Crc32Kernel {
          name: kernel_name,
          func: CRC32_ZBC[i],
        });
      }
    }
  }

  if caps.has(platform::caps::riscv::ZVBC) {
    for (i, &kernel_name) in CRC32_ZVBC_NAMES.iter().enumerate() {
      if name == kernel_name {
        return Some(Crc32Kernel {
          name: kernel_name,
          func: CRC32_ZVBC[i],
        });
      }
    }
  }

  None
}

#[cfg(target_arch = "riscv64")]
fn get_riscv64_crc32c_kernel(name: &str) -> Option<Crc32Kernel> {
  use crate::crc32::kernels::riscv64::*;
  let caps = platform::caps();

  if caps.has(platform::caps::riscv::ZBC) {
    for (i, &kernel_name) in CRC32C_ZBC_NAMES.iter().enumerate() {
      if name == kernel_name {
        return Some(Crc32Kernel {
          name: kernel_name,
          func: CRC32C_ZBC[i],
        });
      }
    }
  }

  if caps.has(platform::caps::riscv::ZVBC) {
    for (i, &kernel_name) in CRC32C_ZVBC_NAMES.iter().enumerate() {
      if name == kernel_name {
        return Some(Crc32Kernel {
          name: kernel_name,
          func: CRC32C_ZVBC[i],
        });
      }
    }
  }

  None
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CRC-64 Kernel Lookup
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Get all available CRC-64 kernel specs for the current platform.
///
/// Returns a list of kernel names that can be passed to `get_crc64_xz_kernel`
/// or `get_crc64_nvme_kernel`.
#[must_use]
pub fn available_crc64_kernels() -> Vec<&'static str> {
  let mut kernels = Vec::new();

  // Always available
  kernels.push("reference");
  kernels.push("portable/slice16");

  #[cfg(target_arch = "x86_64")]
  {
    use crate::crc64::kernels::x86_64::*;
    let caps = platform::caps();

    if caps.has(platform::caps::x86::PCLMUL_READY) {
      kernels.extend_from_slice(PCLMUL_NAMES);
      kernels.push(PCLMUL_SMALL);
    }
    if caps.has(platform::caps::x86::VPCLMUL_READY) {
      kernels.extend_from_slice(VPCLMUL_NAMES);
      kernels.push(VPCLMUL_4X512);
    }
  }

  #[cfg(target_arch = "aarch64")]
  {
    use crate::crc64::kernels::aarch64::*;
    let caps = platform::caps();

    if caps.has(platform::caps::aarch64::PMULL_READY) {
      kernels.extend_from_slice(PMULL_NAMES);
      kernels.push(PMULL_SMALL);
    }
    if caps.has(platform::caps::aarch64::PMULL_EOR3_READY) {
      kernels.extend_from_slice(PMULL_EOR3_NAMES);
    }
    if caps.has(platform::caps::aarch64::SVE2_PMULL) && caps.has(platform::caps::aarch64::PMULL_READY) {
      kernels.extend_from_slice(SVE2_PMULL_NAMES);
      kernels.push(SVE2_PMULL_SMALL);
    }
  }

  #[cfg(target_arch = "powerpc64")]
  {
    use crate::crc64::kernels::power::*;
    let caps = platform::caps();

    if caps.has(platform::caps::power::VPMSUM_READY) {
      kernels.extend_from_slice(VPMSUM_NAMES);
    }
  }

  #[cfg(target_arch = "s390x")]
  {
    use crate::crc64::kernels::s390x::*;
    let caps = platform::caps();

    if caps.has(platform::caps::s390x::VECTOR) {
      kernels.extend_from_slice(VGFM_NAMES);
    }
  }

  #[cfg(target_arch = "riscv64")]
  {
    use crate::crc64::kernels::riscv64::*;
    let caps = platform::caps();

    if caps.has(platform::caps::riscv::ZBC) {
      kernels.extend_from_slice(ZBC_NAMES);
    }
    if caps.has(platform::caps::riscv::ZVBC) {
      kernels.extend_from_slice(ZVBC_NAMES);
    }
  }

  kernels
}

/// Get a CRC-64-XZ kernel function by name.
///
/// Returns `None` if the kernel is not available on this platform.
#[must_use]
pub fn get_crc64_xz_kernel(name: &str) -> Option<Crc64Kernel> {
  use crate::common::{reference::crc64_bitwise, tables::CRC64_XZ_POLY};

  // Reference kernel
  if name == "reference" || name == "reference/bitwise" {
    fn reference_xz(crc: u64, data: &[u8]) -> u64 {
      crc64_bitwise(CRC64_XZ_POLY, crc, data)
    }
    return Some(Crc64Kernel {
      name: "reference",
      func: reference_xz,
    });
  }

  // Portable kernel
  if name == "portable" || name == "portable/slice16" {
    return Some(Crc64Kernel {
      name: "portable/slice16",
      func: crate::crc64::portable::crc64_slice16_xz,
    });
  }

  // Architecture-specific kernels
  #[cfg(target_arch = "x86_64")]
  {
    if let Some(k) = get_x86_64_xz_kernel(name) {
      return Some(k);
    }
  }

  #[cfg(target_arch = "aarch64")]
  {
    if let Some(k) = get_aarch64_xz_kernel(name) {
      return Some(k);
    }
  }

  #[cfg(target_arch = "powerpc64")]
  {
    if let Some(k) = get_power_xz_kernel(name) {
      return Some(k);
    }
  }

  #[cfg(target_arch = "s390x")]
  {
    if let Some(k) = get_s390x_xz_kernel(name) {
      return Some(k);
    }
  }

  #[cfg(target_arch = "riscv64")]
  {
    if let Some(k) = get_riscv64_xz_kernel(name) {
      return Some(k);
    }
  }

  None
}

/// Get a CRC-64-NVME kernel function by name.
///
/// Returns `None` if the kernel is not available on this platform.
#[must_use]
pub fn get_crc64_nvme_kernel(name: &str) -> Option<Crc64Kernel> {
  use crate::common::{reference::crc64_bitwise, tables::CRC64_NVME_POLY};

  // Reference kernel
  if name == "reference" || name == "reference/bitwise" {
    fn reference_nvme(crc: u64, data: &[u8]) -> u64 {
      crc64_bitwise(CRC64_NVME_POLY, crc, data)
    }
    return Some(Crc64Kernel {
      name: "reference",
      func: reference_nvme,
    });
  }

  // Portable kernel
  if name == "portable" || name == "portable/slice16" {
    return Some(Crc64Kernel {
      name: "portable/slice16",
      func: crate::crc64::portable::crc64_slice16_nvme,
    });
  }

  // Architecture-specific kernels
  #[cfg(target_arch = "x86_64")]
  {
    if let Some(k) = get_x86_64_nvme_kernel(name) {
      return Some(k);
    }
  }

  #[cfg(target_arch = "aarch64")]
  {
    if let Some(k) = get_aarch64_nvme_kernel(name) {
      return Some(k);
    }
  }

  #[cfg(target_arch = "powerpc64")]
  {
    if let Some(k) = get_power_nvme_kernel(name) {
      return Some(k);
    }
  }

  #[cfg(target_arch = "s390x")]
  {
    if let Some(k) = get_s390x_nvme_kernel(name) {
      return Some(k);
    }
  }

  #[cfg(target_arch = "riscv64")]
  {
    if let Some(k) = get_riscv64_nvme_kernel(name) {
      return Some(k);
    }
  }

  None
}

// ─────────────────────────────────────────────────────────────────────────────
// Architecture-specific kernel lookup
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
fn get_x86_64_xz_kernel(name: &str) -> Option<Crc64Kernel> {
  use crate::crc64::kernels::x86_64::*;
  let caps = platform::caps();

  // PCLMUL kernels
  if caps.has(platform::caps::x86::PCLMUL_READY) {
    for (i, &kernel_name) in PCLMUL_NAMES.iter().enumerate() {
      if name == kernel_name {
        return Some(Crc64Kernel {
          name: kernel_name,
          func: XZ_PCLMUL[i],
        });
      }
    }
    if name == PCLMUL_SMALL {
      return Some(Crc64Kernel {
        name: PCLMUL_SMALL,
        func: XZ_PCLMUL_SMALL,
      });
    }
  }

  // VPCLMUL kernels
  if caps.has(platform::caps::x86::VPCLMUL_READY) {
    for (i, &kernel_name) in VPCLMUL_NAMES.iter().enumerate() {
      if name == kernel_name {
        return Some(Crc64Kernel {
          name: kernel_name,
          func: XZ_VPCLMUL[i],
        });
      }
    }
    if name == VPCLMUL_4X512 {
      return Some(Crc64Kernel {
        name: VPCLMUL_4X512,
        func: XZ_VPCLMUL_4X512,
      });
    }
  }

  None
}

#[cfg(target_arch = "x86_64")]
fn get_x86_64_nvme_kernel(name: &str) -> Option<Crc64Kernel> {
  use crate::crc64::kernels::x86_64::*;
  let caps = platform::caps();

  if caps.has(platform::caps::x86::PCLMUL_READY) {
    for (i, &kernel_name) in PCLMUL_NAMES.iter().enumerate() {
      if name == kernel_name {
        return Some(Crc64Kernel {
          name: kernel_name,
          func: NVME_PCLMUL[i],
        });
      }
    }
    if name == PCLMUL_SMALL {
      return Some(Crc64Kernel {
        name: PCLMUL_SMALL,
        func: NVME_PCLMUL_SMALL,
      });
    }
  }

  if caps.has(platform::caps::x86::VPCLMUL_READY) {
    for (i, &kernel_name) in VPCLMUL_NAMES.iter().enumerate() {
      if name == kernel_name {
        return Some(Crc64Kernel {
          name: kernel_name,
          func: NVME_VPCLMUL[i],
        });
      }
    }
    if name == VPCLMUL_4X512 {
      return Some(Crc64Kernel {
        name: VPCLMUL_4X512,
        func: NVME_VPCLMUL_4X512,
      });
    }
  }

  None
}

#[cfg(target_arch = "aarch64")]
fn get_aarch64_xz_kernel(name: &str) -> Option<Crc64Kernel> {
  use crate::crc64::kernels::aarch64::*;
  let caps = platform::caps();

  if caps.has(platform::caps::aarch64::PMULL_READY) {
    for (i, &kernel_name) in PMULL_NAMES.iter().enumerate() {
      if name == kernel_name {
        return Some(Crc64Kernel {
          name: kernel_name,
          func: XZ_PMULL[i],
        });
      }
    }
    if name == PMULL_SMALL {
      return Some(Crc64Kernel {
        name: PMULL_SMALL,
        func: XZ_PMULL_SMALL,
      });
    }
  }

  if caps.has(platform::caps::aarch64::PMULL_EOR3_READY) {
    for (i, &kernel_name) in PMULL_EOR3_NAMES.iter().enumerate() {
      if name == kernel_name {
        return Some(Crc64Kernel {
          name: kernel_name,
          func: XZ_PMULL_EOR3[i],
        });
      }
    }
  }

  if caps.has(platform::caps::aarch64::SVE2_PMULL) && caps.has(platform::caps::aarch64::PMULL_READY) {
    for (i, &kernel_name) in SVE2_PMULL_NAMES.iter().enumerate() {
      if name == kernel_name {
        return Some(Crc64Kernel {
          name: kernel_name,
          func: XZ_SVE2_PMULL[i],
        });
      }
    }
    if name == SVE2_PMULL_SMALL {
      return Some(Crc64Kernel {
        name: SVE2_PMULL_SMALL,
        func: XZ_SVE2_PMULL_SMALL,
      });
    }
  }

  None
}

#[cfg(target_arch = "aarch64")]
fn get_aarch64_nvme_kernel(name: &str) -> Option<Crc64Kernel> {
  use crate::crc64::kernels::aarch64::*;
  let caps = platform::caps();

  if caps.has(platform::caps::aarch64::PMULL_READY) {
    for (i, &kernel_name) in PMULL_NAMES.iter().enumerate() {
      if name == kernel_name {
        return Some(Crc64Kernel {
          name: kernel_name,
          func: NVME_PMULL[i],
        });
      }
    }
    if name == PMULL_SMALL {
      return Some(Crc64Kernel {
        name: PMULL_SMALL,
        func: NVME_PMULL_SMALL,
      });
    }
  }

  if caps.has(platform::caps::aarch64::PMULL_EOR3_READY) {
    for (i, &kernel_name) in PMULL_EOR3_NAMES.iter().enumerate() {
      if name == kernel_name {
        return Some(Crc64Kernel {
          name: kernel_name,
          func: NVME_PMULL_EOR3[i],
        });
      }
    }
  }

  if caps.has(platform::caps::aarch64::SVE2_PMULL) && caps.has(platform::caps::aarch64::PMULL_READY) {
    for (i, &kernel_name) in SVE2_PMULL_NAMES.iter().enumerate() {
      if name == kernel_name {
        return Some(Crc64Kernel {
          name: kernel_name,
          func: NVME_SVE2_PMULL[i],
        });
      }
    }
    if name == SVE2_PMULL_SMALL {
      return Some(Crc64Kernel {
        name: SVE2_PMULL_SMALL,
        func: NVME_SVE2_PMULL_SMALL,
      });
    }
  }

  None
}

#[cfg(target_arch = "powerpc64")]
fn get_power_xz_kernel(name: &str) -> Option<Crc64Kernel> {
  use crate::crc64::kernels::power::*;
  let caps = platform::caps();

  if caps.has(platform::caps::power::VPMSUM_READY) {
    for (i, &kernel_name) in VPMSUM_NAMES.iter().enumerate() {
      if name == kernel_name {
        return Some(Crc64Kernel {
          name: kernel_name,
          func: XZ_VPMSUM[i],
        });
      }
    }
  }

  None
}

#[cfg(target_arch = "powerpc64")]
fn get_power_nvme_kernel(name: &str) -> Option<Crc64Kernel> {
  use crate::crc64::kernels::power::*;
  let caps = platform::caps();

  if caps.has(platform::caps::power::VPMSUM_READY) {
    for (i, &kernel_name) in VPMSUM_NAMES.iter().enumerate() {
      if name == kernel_name {
        return Some(Crc64Kernel {
          name: kernel_name,
          func: NVME_VPMSUM[i],
        });
      }
    }
  }

  None
}

#[cfg(target_arch = "s390x")]
fn get_s390x_xz_kernel(name: &str) -> Option<Crc64Kernel> {
  use crate::crc64::kernels::s390x::*;
  let caps = platform::caps();

  if caps.has(platform::caps::s390x::VECTOR) {
    for (i, &kernel_name) in VGFM_NAMES.iter().enumerate() {
      if name == kernel_name {
        return Some(Crc64Kernel {
          name: kernel_name,
          func: XZ_VGFM[i],
        });
      }
    }
  }

  None
}

#[cfg(target_arch = "s390x")]
fn get_s390x_nvme_kernel(name: &str) -> Option<Crc64Kernel> {
  use crate::crc64::kernels::s390x::*;
  let caps = platform::caps();

  if caps.has(platform::caps::s390x::VECTOR) {
    for (i, &kernel_name) in VGFM_NAMES.iter().enumerate() {
      if name == kernel_name {
        return Some(Crc64Kernel {
          name: kernel_name,
          func: NVME_VGFM[i],
        });
      }
    }
  }

  None
}

#[cfg(target_arch = "riscv64")]
fn get_riscv64_xz_kernel(name: &str) -> Option<Crc64Kernel> {
  use crate::crc64::kernels::riscv64::*;
  let caps = platform::caps();

  if caps.has(platform::caps::riscv::ZBC) {
    for (i, &kernel_name) in ZBC_NAMES.iter().enumerate() {
      if name == kernel_name {
        return Some(Crc64Kernel {
          name: kernel_name,
          func: XZ_ZBC[i],
        });
      }
    }
  }

  if caps.has(platform::caps::riscv::ZVBC) {
    for (i, &kernel_name) in ZVBC_NAMES.iter().enumerate() {
      if name == kernel_name {
        return Some(Crc64Kernel {
          name: kernel_name,
          func: XZ_ZVBC[i],
        });
      }
    }
  }

  None
}

#[cfg(target_arch = "riscv64")]
fn get_riscv64_nvme_kernel(name: &str) -> Option<Crc64Kernel> {
  use crate::crc64::kernels::riscv64::*;
  let caps = platform::caps();

  if caps.has(platform::caps::riscv::ZBC) {
    for (i, &kernel_name) in ZBC_NAMES.iter().enumerate() {
      if name == kernel_name {
        return Some(Crc64Kernel {
          name: kernel_name,
          func: NVME_ZBC[i],
        });
      }
    }
  }

  if caps.has(platform::caps::riscv::ZVBC) {
    for (i, &kernel_name) in ZVBC_NAMES.iter().enumerate() {
      if name == kernel_name {
        return Some(Crc64Kernel {
          name: kernel_name,
          func: NVME_ZVBC[i],
        });
      }
    }
  }

  None
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_reference_kernel_available() {
    let kernel = get_crc64_xz_kernel("reference").expect("reference kernel should be available");
    assert_eq!(kernel.name, "reference");

    // Test it produces correct result
    let data = b"123456789";
    let result = (kernel.func)(!0u64, data) ^ !0u64;
    assert_eq!(result, 0x995DC9BBDF1939FA);
  }

  #[test]
  fn test_portable_kernel_available() {
    let kernel = get_crc64_xz_kernel("portable").expect("portable kernel should be available");
    assert_eq!(kernel.name, "portable/slice16");

    let data = b"123456789";
    let result = (kernel.func)(!0u64, data) ^ !0u64;
    assert_eq!(result, 0x995DC9BBDF1939FA);
  }

  #[test]
  fn test_available_kernels_not_empty() {
    let kernels = available_crc64_kernels();
    assert!(kernels.len() >= 2, "should have at least reference and portable");
    assert!(kernels.contains(&"reference"));
    assert!(kernels.contains(&"portable/slice16"));
  }

  #[test]
  fn test_nvme_kernels() {
    let kernel = get_crc64_nvme_kernel("reference").expect("reference kernel should be available");
    let data = b"123456789";
    let result = (kernel.func)(!0u64, data) ^ !0u64;
    assert_eq!(result, 0xAE8B14860A799888);
  }

  #[test]
  fn test_crc32c_kernels_available() {
    let kernels = available_crc32c_kernels();
    assert!(
      kernels.len() >= 3,
      "should have at least reference and portable kernels, got: {:?}",
      kernels
    );

    // Verify each kernel can be looked up
    for &name in &kernels {
      let kernel = get_crc32c_kernel(name);
      assert!(kernel.is_some(), "kernel '{name}' should be available");
    }
  }
}
