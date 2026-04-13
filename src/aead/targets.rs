//! AEAD rollout targets and backend policy.
//!
//! This is the contract layer between `docs/tasks/aead.md`, platform
//! detection, and future concrete implementations.
//!
//! Two rules matter:
//!
//! - every primitive has an explicit backend goal for every benchmark lane
//! - backend selection is derived from detected CPU capabilities, not wishful thinking

use crate::platform::{
  Arch, Caps,
  caps::{aarch64, power, riscv, s390x, wasm, x86},
};

/// AEAD primitives planned for the public surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AeadPrimitive {
  XChaCha20Poly1305,
  ChaCha20Poly1305,
  Aes256GcmSiv,
  Aes256Gcm,
  Aes128GcmSiv,
  Aes128Gcm,
  AsconAead128,
  Aegis256,
}

impl AeadPrimitive {
  /// Primary AEAD rollout set from `docs/tasks/aead.md`.
  pub const CORE: [Self; 5] = [
    Self::XChaCha20Poly1305,
    Self::Aes256GcmSiv,
    Self::Aes256Gcm,
    Self::AsconAead128,
    Self::Aegis256,
  ];

  /// All planned public AEAD primitives, including interop companions.
  pub const ALL: [Self; 8] = [
    Self::XChaCha20Poly1305,
    Self::ChaCha20Poly1305,
    Self::Aes256GcmSiv,
    Self::Aes256Gcm,
    Self::Aes128GcmSiv,
    Self::Aes128Gcm,
    Self::AsconAead128,
    Self::Aegis256,
  ];

  /// Stable display name.
  #[must_use]
  pub const fn name(self) -> &'static str {
    match self {
      Self::XChaCha20Poly1305 => "xchacha20-poly1305",
      Self::ChaCha20Poly1305 => "chacha20-poly1305",
      Self::Aes256GcmSiv => "aes-256-gcm-siv",
      Self::Aes256Gcm => "aes-256-gcm",
      Self::Aes128GcmSiv => "aes-128-gcm-siv",
      Self::Aes128Gcm => "aes-128-gcm",
      Self::AsconAead128 => "ascon-aead128",
      Self::Aegis256 => "aegis-256",
    }
  }

  /// Returns true for ChaCha20/Poly1305-based profiles.
  #[inline]
  #[must_use]
  pub const fn is_chacha_family(self) -> bool {
    matches!(self, Self::XChaCha20Poly1305 | Self::ChaCha20Poly1305)
  }

  /// Returns true for AES-GCM and AES-GCM-SIV profiles.
  #[inline]
  #[must_use]
  pub const fn is_gcm_family(self) -> bool {
    matches!(
      self,
      Self::Aes256GcmSiv | Self::Aes256Gcm | Self::Aes128GcmSiv | Self::Aes128Gcm
    )
  }
}

/// Backend classes that future implementations are allowed to target.
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
  X86Vaes,
  X86VaesVpclmul,
  Aarch64Neon,
  Aarch64Aes,
  Aarch64AesPmull,
  Aarch64Sve2AesPmull,
  S390xMsa,
  S390xVector,
  Power8Crypto,
  PowerVector,
  Riscv64ScalarCrypto,
  Riscv64VectorCrypto,
  Riscv64Vector,
  /// T-table AES (software) + optional Zbc scalar CLMUL for POLYVAL.
  /// Used on RISC-V without crypto extensions (e.g. SpacemiT K1).
  Riscv64Ttable,
}

impl AeadBackend {
  /// Stable backend label for diagnostics and future benchmark grouping.
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
      Self::X86Vaes => "x86_64/vaes",
      Self::X86VaesVpclmul => "x86_64/vaes+vpclmul",
      Self::Aarch64Neon => "aarch64/neon",
      Self::Aarch64Aes => "aarch64/aes",
      Self::Aarch64AesPmull => "aarch64/aes+pmull",
      Self::Aarch64Sve2AesPmull => "aarch64/sve2+aes+pmull",
      Self::S390xMsa => "s390x/msa",
      Self::S390xVector => "s390x/vector",
      Self::Power8Crypto => "powerpc64/crypto",
      Self::PowerVector => "powerpc64/vector",
      Self::Riscv64ScalarCrypto => "riscv64/scalar-crypto",
      Self::Riscv64VectorCrypto => "riscv64/vector-crypto",
      Self::Riscv64Vector => "riscv64/vector",
      Self::Riscv64Ttable => "riscv64/ttable",
    }
  }
}

/// Benchmark runners from `.github/workflows/bench.yaml`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BenchLane {
  IntelIcl,
  IntelSpr,
  AmdZen4,
  AmdZen5,
  Graviton3,
  Graviton4,
  IbmS390x,
  IbmPower10,
}

impl BenchLane {
  /// All benchmark lanes currently wired into the workflow.
  pub const ALL: [Self; 8] = [
    Self::IntelIcl,
    Self::IntelSpr,
    Self::AmdZen4,
    Self::AmdZen5,
    Self::Graviton3,
    Self::Graviton4,
    Self::IbmS390x,
    Self::IbmPower10,
  ];

  /// Stable lane identifier used by CI planning scripts.
  #[must_use]
  pub const fn platform_name(self) -> &'static str {
    match self {
      Self::IntelIcl => "intel-icl",
      Self::IntelSpr => "intel-spr",
      Self::AmdZen4 => "amd-zen4",
      Self::AmdZen5 => "amd-zen5",
      Self::Graviton3 => "graviton3",
      Self::Graviton4 => "graviton4",
      Self::IbmS390x => "ibm-s390x",
      Self::IbmPower10 => "ibm-power10",
    }
  }

  /// Architecture family for the lane.
  #[must_use]
  pub const fn arch(self) -> Arch {
    match self {
      Self::IntelIcl | Self::IntelSpr | Self::AmdZen4 | Self::AmdZen5 => Arch::X86_64,
      Self::Graviton3 | Self::Graviton4 => Arch::Aarch64,
      Self::IbmS390x => Arch::S390x,
      Self::IbmPower10 => Arch::Power,
    }
  }
}

/// Benchmark-lane backend target for a primitive.
///
/// This is the lane-native ceiling the future implementation is expected to
/// compete with on that runner.
#[must_use]
pub const fn lane_target_backend(primitive: AeadPrimitive, lane: BenchLane) -> AeadBackend {
  match primitive {
    AeadPrimitive::XChaCha20Poly1305 | AeadPrimitive::ChaCha20Poly1305 => match lane {
      BenchLane::IntelIcl | BenchLane::IntelSpr | BenchLane::AmdZen4 | BenchLane::AmdZen5 => AeadBackend::X86Avx512,
      BenchLane::Graviton3 | BenchLane::Graviton4 => AeadBackend::Aarch64Neon,
      BenchLane::IbmS390x => AeadBackend::S390xVector,
      BenchLane::IbmPower10 => AeadBackend::PowerVector,
    },
    AeadPrimitive::Aes256GcmSiv | AeadPrimitive::Aes256Gcm | AeadPrimitive::Aes128GcmSiv | AeadPrimitive::Aes128Gcm => {
      match lane {
        BenchLane::IntelIcl | BenchLane::IntelSpr | BenchLane::AmdZen4 | BenchLane::AmdZen5 => {
          AeadBackend::X86VaesVpclmul
        }
        BenchLane::Graviton3 | BenchLane::Graviton4 => AeadBackend::Aarch64AesPmull,
        BenchLane::IbmS390x => AeadBackend::S390xMsa,
        BenchLane::IbmPower10 => AeadBackend::Power8Crypto,
      }
    }
    AeadPrimitive::AsconAead128 => AeadBackend::Portable,
    AeadPrimitive::Aegis256 => match lane {
      BenchLane::IntelIcl | BenchLane::IntelSpr | BenchLane::AmdZen4 | BenchLane::AmdZen5 => AeadBackend::X86Aesni,
      BenchLane::Graviton3 | BenchLane::Graviton4 => AeadBackend::Aarch64Aes,
      BenchLane::IbmS390x => AeadBackend::S390xMsa,
      BenchLane::IbmPower10 => AeadBackend::Power8Crypto,
    },
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
    AeadPrimitive::Aes256GcmSiv | AeadPrimitive::Aes256Gcm | AeadPrimitive::Aes128GcmSiv | AeadPrimitive::Aes128Gcm => {
      select_gcm_backend(arch, caps)
    }
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
        // T-table AES + optional scalar CLMUL POLYVAL (Zbc/Zbkc).
        AeadBackend::Riscv64Ttable
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
      if caps.has(riscv::ZKNE) {
        AeadBackend::Riscv64ScalarCrypto
      } else {
        // T-table AES rounds — ~200x faster than algebraic GF(2^8) S-box.
        AeadBackend::Riscv64Ttable
      }
    }
    Arch::Wasm32 | Arch::Wasm64 => AeadBackend::WasmPortable,
    _ => AeadBackend::Portable,
  }
}

#[cfg(test)]
mod tests {
  use super::{AeadBackend, AeadPrimitive, BenchLane, lane_target_backend, select_backend};
  use crate::platform::{
    Arch, Caps,
    caps::{aarch64, power, riscv, s390x, wasm, x86},
  };

  #[test]
  fn every_core_primitive_has_a_backend_goal_for_every_bench_lane() {
    for primitive in AeadPrimitive::CORE {
      for lane in BenchLane::ALL {
        let backend = lane_target_backend(primitive, lane);
        assert_ne!(backend.name(), "");
      }
    }
  }

  #[test]
  fn workflow_lane_names_match_manual_matrix_script() {
    let workflow = include_str!("../../.github/workflows/bench.yaml");
    let matrix_script = include_str!("../../scripts/ci/emit-manual-matrix.sh");

    assert!(
      workflow.contains("platforms:"),
      "missing platforms workflow_dispatch input in bench workflow"
    );
    assert!(
      workflow.contains("BENCH_PLATFORMS: ${{ inputs.platforms }}"),
      "bench workflow no longer passes platforms input to the matrix script"
    );

    for lane in BenchLane::ALL {
      let platform = lane.platform_name();
      assert!(
        matrix_script.contains(platform),
        "missing {platform} in bench matrix script"
      );
    }
  }

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
      select_backend(AeadPrimitive::Aes128GcmSiv, Arch::Aarch64, aes_pmull_caps),
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

    // Tier 3: T-table fallback (no crypto extensions)
    assert_eq!(
      select_backend(AeadPrimitive::Aes256Gcm, Arch::Riscv64, riscv::V | riscv::ZBC),
      AeadBackend::Riscv64Ttable
    );
    assert_eq!(
      select_backend(AeadPrimitive::Aes256GcmSiv, Arch::Riscv64, riscv::V),
      AeadBackend::Riscv64Ttable
    );

    // AEGIS: Zkne → T-table
    assert_eq!(
      select_backend(AeadPrimitive::Aegis256, Arch::Riscv64, riscv::ZKNE),
      AeadBackend::Riscv64ScalarCrypto
    );
    assert_eq!(
      select_backend(AeadPrimitive::Aegis256, Arch::Riscv64, riscv::V),
      AeadBackend::Riscv64Ttable
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

  #[test]
  fn helper_sets_remain_stable() {
    assert_eq!(AeadPrimitive::CORE.len(), 5);
    assert_eq!(AeadPrimitive::ALL.len(), 8);
    assert_eq!(BenchLane::ALL.len(), 8);
    assert!(AeadPrimitive::XChaCha20Poly1305.is_chacha_family());
    assert!(AeadPrimitive::Aes256Gcm.is_gcm_family());
    assert!(!AeadPrimitive::AsconAead128.is_gcm_family());
  }
}
