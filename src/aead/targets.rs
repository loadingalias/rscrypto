//! AEAD backend targets and dispatch policy.
//!
//! Maps each AEAD primitive to the best available backend for a given platform,
//! based on detected CPU capabilities.
//!
//! Backend selection is derived from detected CPU capabilities.

use crate::platform::{
  Arch, Caps,
  caps::{aarch64, power, riscv, s390x, wasm, x86},
};

/// AEAD primitives on the public surface.
#[allow(dead_code)] // Reduced-feature test builds can compile only byte wrappers, not live dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AeadPrimitive {
  XChaCha20Poly1305,
  ChaCha20Poly1305,
  Aes256GcmSiv,
  Aes256Gcm,
  AsconAead128,
  Aegis256,
}

/// Backend classes selected by live dispatch.
#[allow(dead_code)] // Some architecture-specific variants are only constructed on their target.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AeadBackend {
  Portable,
  WasmPortable,
  WasmSimd128,
  X86Avx2,
  X86Avx512,
  X86Aesni,
  X86AesniPclmul,
  X86VaesVpclmul,
  Aarch64Neon,
  Aarch64Aes,
  Aarch64AesPmull,
  Aarch64Sve2AesPmull,
  S390xMsa,
  /// Hamburg vperm AES rounds for AEGIS — constant-time via z/Vector VPERM.
  /// Used on s390x z13+ where no single-round AES instruction exists.
  S390xVperm,
  S390xVector,
  Power8Crypto,
  PowerVector,
  Riscv64ScalarCrypto,
  Riscv64VectorCrypto,
  Riscv64Vector,
  /// Hamburg vperm AES via `vrgather.vv` — practically constant-time.
  /// Kept as an explicit backend, but not selected for V-only RISC-V until
  /// it beats the scalar portable path on benchmark hardware.
  Riscv64Vperm,
}

impl AeadBackend {
  /// Stable backend label for diagnostics and future benchmark grouping.
  #[allow(dead_code)] // Used by `diag`; reduced-feature builds can compile dispatch without introspection.
  #[must_use]
  pub const fn name(self) -> &'static str {
    match self {
      Self::Portable => "portable",
      Self::WasmPortable => "wasm32/portable",
      Self::WasmSimd128 => "wasm32/simd128",
      Self::X86Avx2 => "x86_64/avx2",
      Self::X86Avx512 => "x86_64/avx512",
      Self::X86Aesni => "x86_64/aesni",
      Self::X86AesniPclmul => "x86_64/aesni+pclmul",
      Self::X86VaesVpclmul => "x86_64/vaes+vpclmul",
      Self::Aarch64Neon => "aarch64/neon",
      Self::Aarch64Aes => "aarch64/aes",
      Self::Aarch64AesPmull => "aarch64/aes+pmull",
      Self::Aarch64Sve2AesPmull => "aarch64/sve2+aes+pmull",
      Self::S390xMsa => "s390x/msa",
      Self::S390xVperm => "s390x/vperm",
      Self::S390xVector => "s390x/vector",
      Self::Power8Crypto => "powerpc64/crypto",
      Self::PowerVector => "powerpc64/vector",
      Self::Riscv64ScalarCrypto => "riscv64/scalar-crypto",
      Self::Riscv64VectorCrypto => "riscv64/vector-crypto",
      Self::Riscv64Vector => "riscv64/vector",
      Self::Riscv64Vperm => "riscv64/vperm",
    }
  }
}

/// Select the best backend class allowed by the detected architecture and caps.
///
/// This function encodes current dispatch policy, not benchmark fantasy.
/// Unmeasured or unimplemented SIMD classes deliberately resolve to `portable`
/// instead of lying.
#[must_use]
pub fn select_backend(primitive: AeadPrimitive, arch: Arch, caps: Caps) -> AeadBackend {
  match primitive {
    AeadPrimitive::XChaCha20Poly1305 | AeadPrimitive::ChaCha20Poly1305 => select_chacha_backend(arch, caps),
    AeadPrimitive::Aes256GcmSiv | AeadPrimitive::Aes256Gcm => select_gcm_backend(arch, caps),
    AeadPrimitive::AsconAead128 => select_ascon_backend(arch),
    AeadPrimitive::Aegis256 => select_aegis_backend(arch, caps),
  }
}

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
      if caps.has(s390x::MSA) {
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

#[inline]
fn select_gcm_backend(arch: Arch, caps: Caps) -> AeadBackend {
  match arch {
    Arch::X86_64 => {
      if caps.has(x86::VAES_READY) && caps.has(x86::VPCLMUL_READY) {
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
        // Constant-time scalar fallback. V-only Hamburg vperm is currently
        // much slower than the portable path on RISE benchmark hardware, so
        // do not select it automatically.
        AeadBackend::Portable
      }
    }
    Arch::Wasm32 | Arch::Wasm64 => AeadBackend::WasmPortable,
    _ => AeadBackend::Portable,
  }
}

#[inline]
const fn select_ascon_backend(arch: Arch) -> AeadBackend {
  match arch {
    Arch::Wasm32 | Arch::Wasm64 => AeadBackend::WasmPortable,
    _ => AeadBackend::Portable,
  }
}

#[inline]
fn select_aegis_backend(arch: Arch, caps: Caps) -> AeadBackend {
  // VAES-256 is intentionally not used for AEGIS-256. The serial update chain
  // (6 dependent AES rounds per block) makes cross-lane shuffle overhead in
  // the VAES-256 path slower than straight AES-NI. See aegis256.rs encrypt_in_place.
  match arch {
    Arch::X86_64 => {
      if caps.has(x86::AESNI) {
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
      // Hamburg vperm provides constant-time rounds on z13+ (vector facility).
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
        // Constant-time scalar fallback. V-only Hamburg vperm is currently
        // much slower than the portable path on RISE benchmark hardware, so
        // do not select it automatically.
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
    let vaes_caps = x86::VAES_READY | x86::VPCLMUL_READY;
    assert_eq!(
      select_backend(AeadPrimitive::Aes256Gcm, Arch::X86_64, vaes_caps),
      AeadBackend::X86VaesVpclmul
    );

    let aesni_caps = x86::AESNI | x86::PCLMULQDQ;
    assert_eq!(
      select_backend(AeadPrimitive::Aes256GcmSiv, Arch::X86_64, aesni_caps),
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

    let aes_pmull_caps = aarch64::AES | aarch64::PMULL;
    assert_eq!(
      select_backend(AeadPrimitive::Aes256GcmSiv, Arch::Aarch64, aes_pmull_caps),
      AeadBackend::Aarch64AesPmull
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
  }

  #[test]
  fn s390x_and_power_have_explicit_aes_family_routes() {
    assert_eq!(
      select_backend(AeadPrimitive::Aes256Gcm, Arch::S390x, s390x::MSA),
      AeadBackend::S390xMsa
    );
    assert_eq!(
      select_backend(AeadPrimitive::XChaCha20Poly1305, Arch::S390x, s390x::MSA),
      AeadBackend::S390xVector
    );
    assert_eq!(
      select_backend(AeadPrimitive::Aes256GcmSiv, Arch::Power, power::POWER8_CRYPTO),
      AeadBackend::Power8Crypto
    );
    assert_eq!(
      select_backend(AeadPrimitive::ChaCha20Poly1305, Arch::Power, power::POWER8_VECTOR),
      AeadBackend::PowerVector
    );
  }

  #[test]
  fn riscv_multi_tier_dispatch() {
    // Tier 1: full vector crypto
    assert_eq!(
      select_backend(AeadPrimitive::Aes256Gcm, Arch::Riscv64, riscv::ZVKNED | riscv::ZVBC),
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

    // Tier 3: V-only falls back to portable until the vperm AES backend is
    // measured faster than the scalar portable path on benchmark hardware.
    assert_eq!(
      select_backend(AeadPrimitive::Aes256Gcm, Arch::Riscv64, riscv::V | riscv::ZBC),
      AeadBackend::Portable
    );
    assert_eq!(
      select_backend(AeadPrimitive::Aes256GcmSiv, Arch::Riscv64, riscv::V),
      AeadBackend::Portable
    );

    // Tier 4: constant-time portable fallback (bare scalar, no V, no crypto)
    assert_eq!(
      select_backend(AeadPrimitive::Aes256Gcm, Arch::Riscv64, Caps::NONE),
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
  fn wasm_has_simd128_chacha_route() {
    assert_eq!(
      select_backend(AeadPrimitive::ChaCha20Poly1305, Arch::Wasm64, wasm::SIMD128),
      AeadBackend::WasmSimd128
    );
  }

  #[test]
  fn ascon_stays_portable_until_measured_simd_is_proven() {
    assert_eq!(
      select_backend(AeadPrimitive::AsconAead128, Arch::X86_64, x86::AVX2 | x86::VAES_READY),
      AeadBackend::Portable
    );
    assert_eq!(
      select_backend(AeadPrimitive::AsconAead128, Arch::Wasm32, Caps::NONE),
      AeadBackend::WasmPortable
    );
  }
}
