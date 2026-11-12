//! Adler-32 runtime configuration.
//!
//! This module handles force mode selection for Adler-32. The dispatch module
//! handles kernel selection automatically; this module only provides
//! the ability to force specific backends for testing/debugging.

use crate::platform::Caps;

/// Requested backend override for Adler-32.
///
/// Requests are clamped to detected CPU capabilities before use.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Adler32Force {
  /// Use the default auto selector.
  #[default]
  Auto,
  /// Force the portable implementation.
  Portable,
  /// Force the aarch64 NEON kernel (if supported).
  Neon,
  /// Force the aarch64 NEON kernel with the dot-product extension (if supported).
  NeonDotprod,
  /// Force the x86-64 SSSE3 kernel (if supported).
  Ssse3,
  /// Force the x86-64 AVX2 kernel (if supported).
  Avx2,
  /// Force the x86-64 AVX-512 VNNI kernel (if supported).
  Avx512Vnni,
  /// Force the POWER AltiVec `vmsumubm` kernel (if supported).
  Vmsum,
  /// Force the IBM Z vector-facility kernel (if supported).
  Vsum,
  /// Force the riscv64 RVV kernel (if supported).
  Rvv,
}

impl Adler32Force {
  /// Return the stable selector name used by debug and configuration surfaces.
  #[must_use]
  pub const fn as_str(self) -> &'static str {
    match self {
      Self::Auto => "auto",
      Self::Portable => "portable",
      Self::Neon => "neon",
      Self::NeonDotprod => "neon-dotprod",
      Self::Ssse3 => "ssse3",
      Self::Avx2 => "avx2",
      Self::Avx512Vnni => "avx512-vnni",
      Self::Vmsum => "vmsum",
      Self::Vsum => "vsum",
      Self::Rvv => "rvv",
    }
  }
}

#[cfg(feature = "std")]
fn parse_force_env() -> Adler32Force {
  let Ok(value) = std::env::var("RSCRYPTO_ADLER32_FORCE") else {
    return Adler32Force::Auto;
  };
  let value = value.trim();
  if value.is_empty() {
    return Adler32Force::Auto;
  }

  if value.eq_ignore_ascii_case("auto") {
    return Adler32Force::Auto;
  }
  if value.eq_ignore_ascii_case("portable") || value.eq_ignore_ascii_case("scalar") {
    return Adler32Force::Portable;
  }
  if value.eq_ignore_ascii_case("neon") {
    return Adler32Force::Neon;
  }
  if value.eq_ignore_ascii_case("neon-dotprod") || value.eq_ignore_ascii_case("dotprod") {
    return Adler32Force::NeonDotprod;
  }
  for force in [
    Adler32Force::Ssse3,
    Adler32Force::Avx2,
    Adler32Force::Avx512Vnni,
    Adler32Force::Vmsum,
    Adler32Force::Vsum,
    Adler32Force::Rvv,
  ] {
    if value.eq_ignore_ascii_case(force.as_str()) {
      return force;
    }
  }

  Adler32Force::Auto
}

#[inline]
#[must_use]
fn clamp_force_to_caps(requested: Adler32Force, caps: Caps) -> Adler32Force {
  #[cfg(not(any(
    all(target_arch = "aarch64", target_feature = "neon"),
    all(target_arch = "x86_64", target_feature = "sse2"),
    all(target_arch = "powerpc64", not(feature = "portable-only")),
    all(target_arch = "s390x", not(feature = "portable-only")),
    all(target_arch = "riscv64", not(feature = "portable-only"))
  )))]
  let _ = caps;

  match requested {
    Adler32Force::Auto | Adler32Force::Portable => requested,
    Adler32Force::Neon => {
      #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
      {
        if caps.has(crate::platform::caps::aarch64::NEON) {
          return Adler32Force::Neon;
        }
      }
      Adler32Force::Auto
    }
    Adler32Force::NeonDotprod => {
      #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
      {
        if caps.has(crate::platform::caps::aarch64::NEON) && caps.has(crate::platform::caps::aarch64::DOTPROD) {
          return Adler32Force::NeonDotprod;
        }
      }
      Adler32Force::Auto
    }
    Adler32Force::Ssse3 | Adler32Force::Avx2 | Adler32Force::Avx512Vnni => {
      #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
      {
        if super::x86_64::KERNELS
          .iter()
          .any(|kernel| kernel.force == requested && caps.has(kernel.requires))
        {
          return requested;
        }
      }
      Adler32Force::Auto
    }
    Adler32Force::Vmsum => {
      #[cfg(all(target_arch = "powerpc64", not(feature = "portable-only")))]
      {
        if caps.has(crate::platform::caps::power::ALTIVEC) {
          return Adler32Force::Vmsum;
        }
      }
      Adler32Force::Auto
    }
    Adler32Force::Vsum => {
      #[cfg(all(target_arch = "s390x", not(feature = "portable-only")))]
      {
        if caps.has(crate::platform::caps::s390x::VECTOR) {
          return Adler32Force::Vsum;
        }
      }
      Adler32Force::Auto
    }
    Adler32Force::Rvv => {
      #[cfg(all(target_arch = "riscv64", not(feature = "portable-only")))]
      {
        if caps.has(crate::platform::caps::riscv::V) {
          return Adler32Force::Rvv;
        }
      }
      Adler32Force::Auto
    }
  }
}

/// Resolved Adler-32 force configuration.
///
/// `requested_force` is the user or environment request. `effective_force` is
/// the same request clamped to the current platform capabilities.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Adler32Config {
  /// Requested force mode (env/programmatic).
  pub requested_force: Adler32Force,
  /// Force mode clamped to detected CPU capabilities.
  pub effective_force: Adler32Force,
}

/// Compute the effective Adler-32 config for the current process.
#[inline]
#[must_use]
fn config(caps: Caps) -> Adler32Config {
  #[cfg(feature = "std")]
  let requested_force = parse_force_env();
  #[cfg(not(feature = "std"))]
  let requested_force = Adler32Force::Auto;

  let effective_force = clamp_force_to_caps(requested_force, caps);

  Adler32Config {
    requested_force,
    effective_force,
  }
}

/// Return the cached process-wide Adler-32 configuration.
///
/// Under `std`, this reflects `RSCRYPTO_ADLER32_FORCE` clamped to the detected
/// platform capabilities.
#[inline]
#[must_use]
pub(super) fn get() -> Adler32Config {
  #[cfg(feature = "std")]
  {
    static CACHED: crate::backend::cache::OnceCache<Adler32Config> = crate::backend::cache::OnceCache::new();
    CACHED.get_or_init(|| config(crate::platform::caps()))
  }

  #[cfg(not(feature = "std"))]
  {
    config(crate::platform::caps())
  }
}

#[cfg(test)]
mod tests {
  use super::{Adler32Force, clamp_force_to_caps};
  use crate::platform::Caps;

  /// A forced kernel that the CPU lacks must fall back to the auto selector,
  /// never select a kernel whose instructions would fault.
  #[test]
  fn forced_kernels_without_their_capabilities_fall_back_to_auto() {
    for force in [
      Adler32Force::Neon,
      Adler32Force::NeonDotprod,
      Adler32Force::Ssse3,
      Adler32Force::Avx2,
      Adler32Force::Avx512Vnni,
      Adler32Force::Vmsum,
      Adler32Force::Vsum,
      Adler32Force::Rvv,
    ] {
      assert_eq!(
        clamp_force_to_caps(force, Caps::NONE),
        Adler32Force::Auto,
        "{}",
        force.as_str()
      );
    }
    assert_eq!(
      clamp_force_to_caps(Adler32Force::Portable, Caps::NONE),
      Adler32Force::Portable
    );
  }

  #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
  #[test]
  fn neon_without_the_dot_product_extension_keeps_only_the_neon_force() {
    let neon = crate::platform::caps::aarch64::NEON;
    assert_eq!(clamp_force_to_caps(Adler32Force::Neon, neon), Adler32Force::Neon);
    assert_eq!(clamp_force_to_caps(Adler32Force::NeonDotprod, neon), Adler32Force::Auto);
    let dotprod = neon.union(crate::platform::caps::aarch64::DOTPROD);
    assert_eq!(
      clamp_force_to_caps(Adler32Force::NeonDotprod, dotprod),
      Adler32Force::NeonDotprod
    );
  }

  /// Each x86-64 force needs every feature that its kernel's `#[target_feature]` enables.
  #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
  #[test]
  fn each_x86_force_needs_every_feature_of_its_kernel() {
    use crate::platform::caps::x86;

    for (force, features) in [
      (Adler32Force::Ssse3, &[x86::SSSE3][..]),
      (Adler32Force::Avx2, &[x86::AVX2]),
      (
        Adler32Force::Avx512Vnni,
        &[x86::AVX512F, x86::AVX512BW, x86::AVX512VNNI],
      ),
    ] {
      let all = features.iter().fold(Caps::NONE, |caps, &feature| caps.union(feature));
      assert_eq!(clamp_force_to_caps(force, all), force, "{}", force.as_str());
      for &missing in features {
        assert_eq!(
          clamp_force_to_caps(force, all.difference(missing)),
          Adler32Force::Auto,
          "{}",
          force.as_str()
        );
      }
    }
  }
}
