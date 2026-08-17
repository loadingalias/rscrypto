//! AEAD backend targets and dispatch policy.
//!
//! Maps each AEAD primitive to the best available backend for a given platform,
//! based on detected CPU capabilities.
//!
//! Backend selection is derived from detected CPU capabilities.

#[cfg(any(test, feature = "chacha20poly1305", feature = "xchacha20poly1305"))]
use crate::platform::caps::wasm;
use crate::platform::{
  Arch, Caps,
  caps::{aarch64, power, riscv, s390x, x86},
};

/// AEAD primitives on the public surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub(super) enum AeadPrimitive {
  #[cfg(any(test, feature = "xchacha20poly1305"))]
  XChaCha20Poly1305,
  #[cfg(any(test, feature = "chacha20poly1305"))]
  ChaCha20Poly1305,
  #[cfg(any(
    test,
    all(
      feature = "aes-gcm-siv",
      any(
        feature = "diag",
        target_arch = "aarch64",
        target_arch = "powerpc64",
        target_arch = "riscv64",
        target_arch = "s390x",
        target_arch = "x86_64",
      )
    )
  ))]
  Aes256GcmSiv,
  #[cfg(any(test, feature = "aes-gcm"))]
  Aes256Gcm,
  #[cfg(any(test, feature = "aes-gcm"))]
  Aes128Gcm,
  #[cfg(any(
    test,
    all(
      feature = "aes-gcm-siv",
      any(
        feature = "diag",
        target_arch = "aarch64",
        target_arch = "powerpc64",
        target_arch = "riscv64",
        target_arch = "s390x",
        target_arch = "x86_64",
      )
    )
  ))]
  Aes128GcmSiv,
  #[cfg(any(
    test,
    all(
      feature = "aegis256",
      any(
        feature = "diag",
        target_arch = "aarch64",
        all(target_arch = "powerpc64", target_endian = "little"),
        target_arch = "riscv64",
        target_arch = "s390x",
        target_arch = "x86_64",
      )
    )
  ))]
  Aegis256,
}

/// Backend classes selected by live dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub(super) enum AeadBackend {
  Portable,
  WasmPortable,
  #[cfg(any(test, feature = "chacha20poly1305", feature = "xchacha20poly1305"))]
  WasmSimd128,
  #[cfg(any(test, feature = "chacha20poly1305", feature = "xchacha20poly1305"))]
  X86Avx2,
  #[cfg(any(test, feature = "chacha20poly1305", feature = "xchacha20poly1305"))]
  X86Avx512,
  #[cfg(any(
    test,
    all(
      feature = "aegis256",
      any(
        feature = "diag",
        target_arch = "aarch64",
        all(target_arch = "powerpc64", target_endian = "little"),
        target_arch = "riscv64",
        target_arch = "s390x",
        target_arch = "x86_64",
      )
    )
  ))]
  X86Aesni,
  #[cfg(any(test, feature = "aes-gcm", feature = "aes-gcm-siv"))]
  X86AesniPclmul,
  #[cfg(any(test, feature = "aes-gcm", feature = "aes-gcm-siv"))]
  X86VaesVpclmul,
  #[cfg(any(test, feature = "chacha20poly1305", feature = "xchacha20poly1305"))]
  Aarch64Neon,
  #[cfg(any(
    test,
    all(
      feature = "aegis256",
      any(
        feature = "diag",
        target_arch = "aarch64",
        all(target_arch = "powerpc64", target_endian = "little"),
        target_arch = "riscv64",
        target_arch = "s390x",
        target_arch = "x86_64",
      )
    )
  ))]
  Aarch64Aes,
  #[cfg(any(test, feature = "aes-gcm", feature = "aes-gcm-siv"))]
  Aarch64AesPmull,
  #[cfg(any(test, feature = "aes-gcm", feature = "aes-gcm-siv"))]
  Aarch64Sve2AesPmull,
  #[cfg(any(test, feature = "aes-gcm", feature = "aes-gcm-siv"))]
  S390xMsa,
  /// Hamburg vperm AES rounds for AEGIS using register-only z/Vector VPERM.
  /// Used on s390x z13+ where no single-round AES instruction exists.
  #[cfg(any(
    test,
    all(
      feature = "aegis256",
      any(
        feature = "diag",
        target_arch = "aarch64",
        all(target_arch = "powerpc64", target_endian = "little"),
        target_arch = "riscv64",
        target_arch = "s390x",
        target_arch = "x86_64",
      )
    )
  ))]
  S390xVperm,
  #[cfg(any(test, feature = "chacha20poly1305", feature = "xchacha20poly1305"))]
  S390xVector,
  #[cfg(any(test, feature = "aegis256", feature = "aes-gcm", feature = "aes-gcm-siv"))]
  Power8Crypto,
  #[cfg(any(test, feature = "chacha20poly1305", feature = "xchacha20poly1305"))]
  PowerVector,
  #[cfg(any(test, feature = "aegis256", feature = "aes-gcm", feature = "aes-gcm-siv"))]
  Riscv64ScalarCrypto,
  #[cfg(any(test, feature = "aegis256", feature = "aes-gcm", feature = "aes-gcm-siv"))]
  Riscv64VectorCrypto,
  #[cfg(any(test, feature = "chacha20poly1305", feature = "xchacha20poly1305"))]
  Riscv64Vector,
}

impl AeadBackend {
  /// Stable backend label for diagnostics and future benchmark grouping.
  #[cfg(feature = "diag")]
  #[must_use]
  pub(super) const fn name(self) -> &'static str {
    match self {
      Self::Portable => "portable",
      Self::WasmPortable => "wasm32/portable",
      #[cfg(any(test, feature = "chacha20poly1305", feature = "xchacha20poly1305"))]
      Self::WasmSimd128 => "wasm32/simd128",
      #[cfg(any(test, feature = "chacha20poly1305", feature = "xchacha20poly1305"))]
      Self::X86Avx2 => "x86_64/avx2",
      #[cfg(any(test, feature = "chacha20poly1305", feature = "xchacha20poly1305"))]
      Self::X86Avx512 => "x86_64/avx512",
      #[cfg(any(
        test,
        all(
          feature = "aegis256",
          any(
            feature = "diag",
            target_arch = "aarch64",
            all(target_arch = "powerpc64", target_endian = "little"),
            target_arch = "riscv64",
            target_arch = "s390x",
            target_arch = "x86_64",
          )
        )
      ))]
      Self::X86Aesni => "x86_64/aesni",
      #[cfg(any(test, feature = "aes-gcm", feature = "aes-gcm-siv"))]
      Self::X86AesniPclmul => "x86_64/aesni+pclmul",
      #[cfg(any(test, feature = "aes-gcm", feature = "aes-gcm-siv"))]
      Self::X86VaesVpclmul => "x86_64/vaes+vpclmul",
      #[cfg(any(test, feature = "chacha20poly1305", feature = "xchacha20poly1305"))]
      Self::Aarch64Neon => "aarch64/neon",
      #[cfg(any(
        test,
        all(
          feature = "aegis256",
          any(
            feature = "diag",
            target_arch = "aarch64",
            all(target_arch = "powerpc64", target_endian = "little"),
            target_arch = "riscv64",
            target_arch = "s390x",
            target_arch = "x86_64",
          )
        )
      ))]
      Self::Aarch64Aes => "aarch64/aes",
      #[cfg(any(test, feature = "aes-gcm", feature = "aes-gcm-siv"))]
      Self::Aarch64AesPmull => "aarch64/aes+pmull",
      #[cfg(any(test, feature = "aes-gcm", feature = "aes-gcm-siv"))]
      Self::Aarch64Sve2AesPmull => "aarch64/sve2+aes+pmull",
      #[cfg(any(test, feature = "aes-gcm", feature = "aes-gcm-siv"))]
      Self::S390xMsa => "s390x/msa",
      #[cfg(any(
        test,
        all(
          feature = "aegis256",
          any(
            feature = "diag",
            target_arch = "aarch64",
            all(target_arch = "powerpc64", target_endian = "little"),
            target_arch = "riscv64",
            target_arch = "s390x",
            target_arch = "x86_64",
          )
        )
      ))]
      Self::S390xVperm => "s390x/vperm",
      #[cfg(any(test, feature = "chacha20poly1305", feature = "xchacha20poly1305"))]
      Self::S390xVector => "s390x/vector",
      #[cfg(any(test, feature = "aegis256", feature = "aes-gcm", feature = "aes-gcm-siv"))]
      Self::Power8Crypto => "powerpc64/crypto",
      #[cfg(any(test, feature = "chacha20poly1305", feature = "xchacha20poly1305"))]
      Self::PowerVector => "powerpc64/vector",
      #[cfg(any(test, feature = "aegis256", feature = "aes-gcm", feature = "aes-gcm-siv"))]
      Self::Riscv64ScalarCrypto => "riscv64/scalar-crypto",
      #[cfg(any(test, feature = "aegis256", feature = "aes-gcm", feature = "aes-gcm-siv"))]
      Self::Riscv64VectorCrypto => "riscv64/vector-crypto",
      #[cfg(any(test, feature = "chacha20poly1305", feature = "xchacha20poly1305"))]
      Self::Riscv64Vector => "riscv64/vector",
    }
  }
}

/// Select the backend class allowed by the detected architecture, capabilities,
/// and current dispatch policy.
///
/// SIMD classes without accepted target-native evidence resolve to `portable`.
#[must_use]
pub(super) fn select_backend(primitive: AeadPrimitive, arch: Arch, caps: Caps) -> AeadBackend {
  match primitive {
    #[cfg(any(test, feature = "xchacha20poly1305"))]
    AeadPrimitive::XChaCha20Poly1305 => select_chacha_backend(arch, caps),
    #[cfg(any(test, feature = "chacha20poly1305"))]
    AeadPrimitive::ChaCha20Poly1305 => select_chacha_backend(arch, caps),
    #[cfg(any(
      test,
      all(
        feature = "aes-gcm-siv",
        any(
          feature = "diag",
          target_arch = "aarch64",
          target_arch = "powerpc64",
          target_arch = "riscv64",
          target_arch = "s390x",
          target_arch = "x86_64",
        )
      )
    ))]
    AeadPrimitive::Aes256GcmSiv | AeadPrimitive::Aes128GcmSiv => select_gcm_backend(arch, caps),
    #[cfg(any(test, feature = "aes-gcm"))]
    AeadPrimitive::Aes256Gcm | AeadPrimitive::Aes128Gcm => select_gcm_backend(arch, caps),
    #[cfg(any(
      test,
      all(
        feature = "aegis256",
        any(
          feature = "diag",
          target_arch = "aarch64",
          all(target_arch = "powerpc64", target_endian = "little"),
          target_arch = "riscv64",
          target_arch = "s390x",
          target_arch = "x86_64",
        )
      )
    ))]
    AeadPrimitive::Aegis256 => select_aegis_backend(arch, caps),
  }
}

#[cfg(any(test, feature = "chacha20poly1305", feature = "xchacha20poly1305"))]
#[inline]
fn select_chacha_backend(arch: Arch, caps: Caps) -> AeadBackend {
  match arch {
    Arch::X86_64 => {
      if caps.has(x86::AVX512_READY) {
        AeadBackend::X86Avx512
      } else if caps.has(x86::AVX2) {
        AeadBackend::X86Avx2
      } else {
        AeadBackend::Portable
      }
    }
    Arch::Aarch64 => {
      if caps.has(aarch64::NEON) {
        AeadBackend::Aarch64Neon
      } else {
        AeadBackend::Portable
      }
    }
    Arch::Wasm32 | Arch::Wasm64 => {
      if caps.has(wasm::SIMD128) {
        AeadBackend::WasmSimd128
      } else {
        AeadBackend::WasmPortable
      }
    }
    Arch::S390x => {
      if caps.has(s390x::VECTOR) {
        AeadBackend::S390xVector
      } else {
        AeadBackend::Portable
      }
    }
    Arch::Power => {
      if caps.has(power::POWER8_VECTOR) {
        AeadBackend::PowerVector
      } else {
        AeadBackend::Portable
      }
    }
    Arch::Riscv64 => {
      if caps.has(riscv::V) {
        AeadBackend::Riscv64Vector
      } else {
        AeadBackend::Portable
      }
    }
    _ => AeadBackend::Portable,
  }
}

#[cfg(any(test, feature = "aes-gcm", feature = "aes-gcm-siv"))]
#[inline]
fn select_gcm_backend(arch: Arch, caps: Caps) -> AeadBackend {
  match arch {
    Arch::X86_64 => {
      if caps.has(x86::VAES_READY) && caps.has(x86::VPCLMUL_READY) && caps.has(x86::AESNI) {
        AeadBackend::X86VaesVpclmul
      } else if caps.has(x86::AESNI) && caps.has(x86::PCLMULQDQ) {
        AeadBackend::X86AesniPclmul
      } else {
        AeadBackend::Portable
      }
    }
    Arch::Aarch64 => {
      if caps.has(aarch64::AES)
        && caps.has(aarch64::PMULL)
        && caps.has(aarch64::SVE2_AES)
        && caps.has(aarch64::SVE2_PMULL)
      {
        AeadBackend::Aarch64Sve2AesPmull
      } else if caps.has(aarch64::AES) && caps.has(aarch64::PMULL) {
        AeadBackend::Aarch64AesPmull
      } else {
        AeadBackend::Portable
      }
    }
    Arch::S390x => {
      if caps.has(s390x::MSA) {
        AeadBackend::S390xMsa
      } else {
        AeadBackend::Portable
      }
    }
    Arch::Power => {
      if caps.has(power::POWER8_CRYPTO) {
        AeadBackend::Power8Crypto
      } else {
        AeadBackend::Portable
      }
    }
    Arch::Riscv64 => {
      if caps.has(riscv::ZVKNED) && caps.has(riscv::ZVBC) {
        AeadBackend::Riscv64VectorCrypto
      } else if caps.has(riscv::ZKNE) && (caps.has(riscv::ZBC) || caps.has(riscv::ZBKC)) {
        AeadBackend::Riscv64ScalarCrypto
      } else {
        // V-only Hamburg vperm remains available for forced diagnostics but
        // lacks accepted target-native evidence for automatic selection.
        AeadBackend::Portable
      }
    }
    Arch::Wasm32 | Arch::Wasm64 => AeadBackend::WasmPortable,
    _ => AeadBackend::Portable,
  }
}

#[cfg(any(
  test,
  all(
    feature = "aegis256",
    any(
      feature = "diag",
      target_arch = "aarch64",
      all(target_arch = "powerpc64", target_endian = "little"),
      target_arch = "riscv64",
      target_arch = "s390x",
      target_arch = "x86_64",
    )
  )
))]
#[inline]
fn select_aegis_backend(arch: Arch, caps: Caps) -> AeadBackend {
  // VAES-256 is intentionally not used for AEGIS-256. Its six dependent state
  // lanes require cross-lane shuffles in the packed representation; see the
  // XMM-state path in aegis256.rs.
  match arch {
    Arch::X86_64 => {
      if caps.has(x86::AESNI) && caps.has(x86::AVX) {
        AeadBackend::X86Aesni
      } else {
        AeadBackend::Portable
      }
    }
    Arch::Aarch64 => {
      if caps.has(aarch64::AES) {
        AeadBackend::Aarch64Aes
      } else {
        AeadBackend::Portable
      }
    }
    Arch::S390x => {
      // AEGIS needs single AES rounds — CPACF KM/KMA only do full blocks.
      // Hamburg vperm keeps the rounds register-only on z13+ (vector facility).
      if caps.has(s390x::VECTOR) {
        AeadBackend::S390xVperm
      } else {
        AeadBackend::Portable
      }
    }
    Arch::Power => {
      if caps.has(power::POWER8_CRYPTO) {
        AeadBackend::Power8Crypto
      } else {
        AeadBackend::Portable
      }
    }
    Arch::Riscv64 => {
      if caps.has(riscv::ZVKNED) {
        AeadBackend::Riscv64VectorCrypto
      } else if caps.has(riscv::ZKNE) {
        AeadBackend::Riscv64ScalarCrypto
      } else {
        // V-only Hamburg vperm remains available for forced diagnostics but
        // lacks accepted target-native evidence for automatic selection.
        AeadBackend::Portable
      }
    }
    Arch::Wasm32 | Arch::Wasm64 => AeadBackend::WasmPortable,
    _ => AeadBackend::Portable,
  }
}

#[cfg(test)]
mod tests {
  use super::{AeadBackend, AeadPrimitive, select_backend};
  use crate::platform::{
    Arch, Caps,
    caps::{aarch64, power, riscv, s390x, wasm, x86},
  };

  #[test]
  fn gcm_prefers_x86_vaes_then_aesni() {
    let vaes_caps = x86::VAES_READY | x86::VPCLMUL_READY | x86::AESNI;
    assert_eq!(
      select_backend(AeadPrimitive::Aes256Gcm, Arch::X86_64, vaes_caps),
      AeadBackend::X86VaesVpclmul
    );
    assert_eq!(
      select_backend(AeadPrimitive::Aes128Gcm, Arch::X86_64, vaes_caps),
      AeadBackend::X86VaesVpclmul
    );

    let vaes_without_aesni = x86::VAES_READY | x86::VPCLMUL_READY;
    assert_eq!(
      select_backend(AeadPrimitive::Aes256Gcm, Arch::X86_64, vaes_without_aesni),
      AeadBackend::Portable
    );

    let aesni_caps = x86::AESNI | x86::PCLMULQDQ;
    assert_eq!(
      select_backend(AeadPrimitive::Aes256GcmSiv, Arch::X86_64, aesni_caps),
      AeadBackend::X86AesniPclmul
    );
    assert_eq!(
      select_backend(AeadPrimitive::Aes128Gcm, Arch::X86_64, aesni_caps),
      AeadBackend::X86AesniPclmul
    );
    assert_eq!(
      select_backend(AeadPrimitive::Aes128GcmSiv, Arch::X86_64, vaes_caps),
      AeadBackend::X86VaesVpclmul
    );
    assert_eq!(
      select_backend(AeadPrimitive::Aes128GcmSiv, Arch::X86_64, aesni_caps),
      AeadBackend::X86AesniPclmul
    );
  }

  #[test]
  fn gcm_prefers_aarch64_sve2_then_aes_pmull() {
    let sve2_caps = aarch64::AES | aarch64::PMULL | aarch64::SVE2_AES | aarch64::SVE2_PMULL;
    assert_eq!(
      select_backend(AeadPrimitive::Aes256Gcm, Arch::Aarch64, sve2_caps),
      AeadBackend::Aarch64Sve2AesPmull
    );
    assert_eq!(
      select_backend(AeadPrimitive::Aes128Gcm, Arch::Aarch64, sve2_caps),
      AeadBackend::Aarch64Sve2AesPmull
    );

    let aes_pmull_caps = aarch64::AES | aarch64::PMULL;
    assert_eq!(
      select_backend(AeadPrimitive::Aes256GcmSiv, Arch::Aarch64, aes_pmull_caps),
      AeadBackend::Aarch64AesPmull
    );
    assert_eq!(
      select_backend(AeadPrimitive::Aes128Gcm, Arch::Aarch64, aes_pmull_caps),
      AeadBackend::Aarch64AesPmull
    );
    assert_eq!(
      select_backend(AeadPrimitive::Aes128GcmSiv, Arch::Aarch64, sve2_caps),
      AeadBackend::Aarch64Sve2AesPmull
    );
    assert_eq!(
      select_backend(AeadPrimitive::Aes128GcmSiv, Arch::Aarch64, aes_pmull_caps),
      AeadBackend::Aarch64AesPmull
    );

    let sve2_without_sve2_pmull = aarch64::AES | aarch64::PMULL | aarch64::SVE2_AES;
    assert_eq!(
      select_backend(AeadPrimitive::Aes256Gcm, Arch::Aarch64, sve2_without_sve2_pmull),
      AeadBackend::Aarch64AesPmull
    );
    assert_eq!(
      select_backend(AeadPrimitive::Aes256Gcm, Arch::Aarch64, aarch64::AES),
      AeadBackend::Portable
    );
  }

  #[test]
  fn chacha_and_aegis_choose_lane_native_non_aes_and_aes_paths() {
    assert_eq!(
      select_backend(AeadPrimitive::XChaCha20Poly1305, Arch::X86_64, x86::AVX512_READY),
      AeadBackend::X86Avx512
    );
    assert_eq!(
      select_backend(AeadPrimitive::ChaCha20Poly1305, Arch::X86_64, x86::AVX2),
      AeadBackend::X86Avx2
    );
    assert_eq!(
      select_backend(AeadPrimitive::XChaCha20Poly1305, Arch::Aarch64, aarch64::NEON),
      AeadBackend::Aarch64Neon
    );
    assert_eq!(
      select_backend(AeadPrimitive::ChaCha20Poly1305, Arch::Wasm32, wasm::SIMD128),
      AeadBackend::WasmSimd128
    );
    assert_eq!(
      select_backend(AeadPrimitive::Aegis256, Arch::X86_64, x86::AESNI),
      AeadBackend::Portable
    );
    assert_eq!(
      select_backend(AeadPrimitive::Aegis256, Arch::X86_64, x86::AESNI | x86::AVX),
      AeadBackend::X86Aesni
    );
    assert_eq!(
      select_backend(AeadPrimitive::Aegis256, Arch::Aarch64, aarch64::AES),
      AeadBackend::Aarch64Aes
    );
    assert_eq!(
      select_backend(AeadPrimitive::Aegis256, Arch::Riscv64, riscv::ZVKNED),
      AeadBackend::Riscv64VectorCrypto
    );
    assert_eq!(
      select_backend(AeadPrimitive::Aegis256, Arch::Riscv64, riscv::ZKNE),
      AeadBackend::Riscv64ScalarCrypto
    );
    assert_eq!(
      select_backend(AeadPrimitive::Aegis256, Arch::S390x, s390x::VECTOR),
      AeadBackend::S390xVperm
    );
    assert_eq!(
      select_backend(AeadPrimitive::Aegis256, Arch::S390x, Caps::NONE),
      AeadBackend::Portable
    );
  }

  #[test]
  fn s390x_and_power_have_explicit_aes_family_routes() {
    assert_eq!(
      select_backend(AeadPrimitive::Aes256Gcm, Arch::S390x, s390x::MSA),
      AeadBackend::S390xMsa
    );
    assert_eq!(
      select_backend(AeadPrimitive::Aes128Gcm, Arch::S390x, s390x::MSA),
      AeadBackend::S390xMsa
    );
    assert_eq!(
      select_backend(AeadPrimitive::XChaCha20Poly1305, Arch::S390x, s390x::VECTOR),
      AeadBackend::S390xVector
    );
    assert_eq!(
      select_backend(AeadPrimitive::XChaCha20Poly1305, Arch::S390x, s390x::MSA),
      AeadBackend::Portable
    );
    assert_eq!(
      select_backend(AeadPrimitive::Aes256GcmSiv, Arch::Power, power::POWER8_CRYPTO),
      AeadBackend::Power8Crypto
    );
    assert_eq!(
      select_backend(AeadPrimitive::Aes256Gcm, Arch::Power, power::POWER8_CRYPTO),
      AeadBackend::Power8Crypto
    );
    assert_eq!(
      select_backend(AeadPrimitive::Aes128Gcm, Arch::Power, power::POWER8_CRYPTO),
      AeadBackend::Power8Crypto
    );
    assert_eq!(
      select_backend(AeadPrimitive::Aes128GcmSiv, Arch::S390x, s390x::MSA),
      AeadBackend::S390xMsa
    );
    assert_eq!(
      select_backend(AeadPrimitive::Aes128GcmSiv, Arch::Power, power::POWER8_CRYPTO),
      AeadBackend::Power8Crypto
    );
    assert_eq!(
      select_backend(AeadPrimitive::ChaCha20Poly1305, Arch::Power, power::POWER8_VECTOR),
      AeadBackend::PowerVector
    );
    assert_eq!(
      select_backend(AeadPrimitive::Aegis256, Arch::Power, power::POWER8_CRYPTO),
      AeadBackend::Power8Crypto
    );
  }

  #[test]
  fn riscv_multi_tier_dispatch() {
    // Tier 1: full vector crypto
    assert_eq!(
      select_backend(AeadPrimitive::Aes256Gcm, Arch::Riscv64, riscv::ZVKNED | riscv::ZVBC),
      AeadBackend::Riscv64VectorCrypto
    );
    assert_eq!(
      select_backend(AeadPrimitive::Aes128Gcm, Arch::Riscv64, riscv::ZVKNED | riscv::ZVBC),
      AeadBackend::Riscv64VectorCrypto
    );

    // Tier 2: scalar AES + scalar CLMUL
    assert_eq!(
      select_backend(AeadPrimitive::Aes256GcmSiv, Arch::Riscv64, riscv::ZKNE | riscv::ZBC),
      AeadBackend::Riscv64ScalarCrypto
    );
    // Zbkc also qualifies for scalar CLMUL
    assert_eq!(
      select_backend(AeadPrimitive::Aes256Gcm, Arch::Riscv64, riscv::ZKNE | riscv::ZBKC),
      AeadBackend::Riscv64ScalarCrypto
    );
    assert_eq!(
      select_backend(AeadPrimitive::Aes128Gcm, Arch::Riscv64, riscv::ZKNE | riscv::ZBC),
      AeadBackend::Riscv64ScalarCrypto
    );

    // Tier 3: V-only falls back to portable until target-native evidence
    // supports automatic vperm selection.
    assert_eq!(
      select_backend(AeadPrimitive::Aes256Gcm, Arch::Riscv64, riscv::V | riscv::ZBC),
      AeadBackend::Portable
    );
    assert_eq!(
      select_backend(AeadPrimitive::Aes256GcmSiv, Arch::Riscv64, riscv::V),
      AeadBackend::Portable
    );
    assert_eq!(
      select_backend(AeadPrimitive::Aes128Gcm, Arch::Riscv64, riscv::V | riscv::ZBC),
      AeadBackend::Portable
    );
    assert_eq!(
      select_backend(AeadPrimitive::Aes256Gcm, Arch::Riscv64, riscv::ZVKNED),
      AeadBackend::Portable
    );

    // Tier 4: table-free portable fallback (bare scalar, no V, no crypto)
    assert_eq!(
      select_backend(AeadPrimitive::Aes256Gcm, Arch::Riscv64, Caps::NONE),
      AeadBackend::Portable
    );
    assert_eq!(
      select_backend(AeadPrimitive::Aes128Gcm, Arch::Riscv64, Caps::NONE),
      AeadBackend::Portable
    );
    assert_eq!(
      select_backend(AeadPrimitive::Aes128GcmSiv, Arch::Riscv64, riscv::ZVKNED | riscv::ZVBC),
      AeadBackend::Riscv64VectorCrypto
    );
    assert_eq!(
      select_backend(AeadPrimitive::Aes128GcmSiv, Arch::Riscv64, riscv::ZKNE | riscv::ZBC),
      AeadBackend::Riscv64ScalarCrypto
    );
    assert_eq!(
      select_backend(AeadPrimitive::Aes128GcmSiv, Arch::Riscv64, Caps::NONE),
      AeadBackend::Portable
    );

    // AEGIS: Zvkned → Zkne → portable
    assert_eq!(
      select_backend(AeadPrimitive::Aegis256, Arch::Riscv64, riscv::ZVKNED),
      AeadBackend::Riscv64VectorCrypto
    );
    assert_eq!(
      select_backend(AeadPrimitive::Aegis256, Arch::Riscv64, riscv::ZKNE),
      AeadBackend::Riscv64ScalarCrypto
    );
    assert_eq!(
      select_backend(AeadPrimitive::Aegis256, Arch::Riscv64, riscv::V),
      AeadBackend::Portable
    );
    assert_eq!(
      select_backend(AeadPrimitive::Aegis256, Arch::Riscv64, Caps::NONE),
      AeadBackend::Portable
    );

    // ChaCha: V → Riscv64Vector
    assert_eq!(
      select_backend(AeadPrimitive::XChaCha20Poly1305, Arch::Riscv64, riscv::V),
      AeadBackend::Riscv64Vector
    );
  }

  #[test]
  fn wasm_uses_simd128_only_for_chacha() {
    assert_eq!(
      select_backend(AeadPrimitive::ChaCha20Poly1305, Arch::Wasm64, wasm::SIMD128),
      AeadBackend::WasmSimd128
    );
    assert_eq!(
      select_backend(AeadPrimitive::Aes256Gcm, Arch::Wasm32, wasm::SIMD128),
      AeadBackend::WasmPortable
    );
    assert_eq!(
      select_backend(AeadPrimitive::Aegis256, Arch::Wasm64, wasm::SIMD128),
      AeadBackend::WasmPortable
    );
  }
}
