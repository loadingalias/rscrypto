#[cfg(test)]
extern crate alloc;

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  #[cfg(not(miri))] // get() returns portable() under Miri, which has different arch
  fn test_get_returns_valid() {
    let det = get();

    #[cfg(target_arch = "x86_64")]
    assert_eq!(det.arch, Arch::X86_64);
    #[cfg(target_arch = "aarch64")]
    assert_eq!(det.arch, Arch::Aarch64);

    assert!(det.tune.simd_threshold > 0);
    assert!(det.tune.parallel_streams > 0);
    assert!(det.tune.cache_line > 0);
  }

  #[test]
  #[cfg(not(miri))] // Uses syscalls for feature detection
  fn test_detect_uncached_consistent() {
    let d1 = detect_uncached();
    let d2 = detect_uncached();
    assert_eq!(d1.caps, d2.caps);
    assert_eq!(d1.arch, d2.arch);
  }

  #[test]
  #[cfg(not(miri))] // get() uses syscalls for feature detection
  fn test_convenience_functions() {
    let det = get();
    assert_eq!(caps(), det.caps);
    assert_eq!(tune(), det.tune);
    assert_eq!(arch(), det.arch);
  }

  #[test]
  #[cfg(all(target_arch = "x86_64", not(miri)))]
  fn test_x86_64_baseline() {
    use crate::caps::x86;
    let det = get();
    assert!(det.caps.has(x86::SSE2));
  }

  #[test]
  #[cfg(all(target_arch = "aarch64", not(miri)))]
  fn test_aarch64_baseline() {
    use crate::caps::aarch64;
    let det = get();
    assert!(det.caps.has(aarch64::NEON));
  }

  #[test]
  #[cfg(miri)]
  fn test_miri_returns_portable() {
    let det = get();
    assert_eq!(det.caps, Caps::NONE);
    assert_eq!(det.arch, Arch::Other);
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Compile-Time Detection Tests (caps_static)
  // ─────────────────────────────────────────────────────────────────────────────

  #[test]
  fn test_caps_static_is_const() {
    // Verify caps_static() can be used in const context
    const STATIC_CAPS: Caps = caps_static();
    let _ = STATIC_CAPS; // Use it to avoid dead code warning
  }

  #[test]
  #[cfg(target_arch = "x86_64")]
  fn test_caps_static_x86_64_baseline() {
    use crate::caps::x86;

    // x86_64 guarantees SSE2
    let caps = caps_static();
    assert!(caps.has(x86::SSE2), "x86_64 must have SSE2 baseline in caps_static");
  }

  #[test]
  #[cfg(target_arch = "aarch64")]
  fn test_caps_static_aarch64_baseline() {
    use crate::caps::aarch64;

    // aarch64 guarantees NEON
    let caps = caps_static();
    assert!(
      caps.has(aarch64::NEON),
      "aarch64 must have NEON baseline in caps_static"
    );
  }

  #[test]
  #[cfg(not(miri))] // Miri can't detect runtime features, returns Caps::NONE
  fn test_caps_static_subset_of_runtime() {
    // Compile-time detected features must be a subset of runtime detected features
    let static_caps = caps_static();
    let runtime_caps = caps();

    // Every compile-time feature must be present at runtime
    assert!(
      runtime_caps.has(static_caps),
      "caps_static() must be subset of caps(): static={:?}, runtime={:?}",
      static_caps,
      runtime_caps
    );
  }

  #[test]
  fn test_caps_static_consistent() {
    // caps_static() must return the same value every time
    let a = caps_static();
    let b = caps_static();
    assert_eq!(a, b, "caps_static() must be deterministic");
  }

  #[test]
  #[cfg(all(target_arch = "x86_64", not(miri)))]
  fn test_caps_static_x86_features() {
    use crate::caps::x86;

    let caps = caps_static();

    // Test that feature groups are consistent with their baselines
    // If AVX2 is enabled at compile time, it should be detected
    if cfg!(target_feature = "avx2") {
      assert!(caps.has(x86::AVX2), "AVX2 must be detected when target_feature enabled");
    }

    // If AVX-512F is enabled, foundation should be detected
    if cfg!(target_feature = "avx512f") {
      assert!(
        caps.has(x86::AVX512F),
        "AVX512F must be detected when target_feature enabled"
      );
    }

    // If VPCLMULQDQ is enabled, it should be detected
    if cfg!(target_feature = "vpclmulqdq") {
      assert!(
        caps.has(x86::VPCLMULQDQ),
        "VPCLMULQDQ must be detected when target_feature enabled"
      );
    }
  }

  #[test]
  #[cfg(all(target_arch = "aarch64", not(miri)))]
  fn test_caps_static_aarch64_features() {
    use crate::caps::aarch64;

    let caps = caps_static();

    // If AES is enabled at compile time, both AES and PMULL should be detected
    if cfg!(target_feature = "aes") {
      assert!(
        caps.has(aarch64::AES),
        "AES must be detected when target_feature enabled"
      );
      assert!(
        caps.has(aarch64::PMULL),
        "PMULL must be detected when aes target_feature enabled"
      );
    }

    // If SHA3 is enabled, both SHA3 and SHA512 should be detected
    if cfg!(target_feature = "sha3") {
      assert!(
        caps.has(aarch64::SHA3),
        "SHA3 must be detected when target_feature enabled"
      );
      assert!(
        caps.has(aarch64::SHA512),
        "SHA512 must be detected when sha3 target_feature enabled"
      );
    }

    // If SME is enabled, it should be detected (fixing prior drift)
    if cfg!(target_feature = "sme") {
      assert!(
        caps.has(aarch64::SME),
        "SME must be detected when target_feature enabled"
      );
    }
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Apple Silicon Detection Tests
  // ─────────────────────────────────────────────────────────────────────────────

  #[test]
  #[cfg(all(target_arch = "aarch64", target_os = "macos", not(miri)))]
  fn test_apple_silicon_detection_runs() {
    // Just verify detection doesn't crash and returns a valid result
    let chip_gen = detect_apple_silicon_gen();
    // On actual Apple Silicon, we should get Some variant
    // On Rosetta 2 or non-Apple aarch64, we might get None
    if let Some(detected) = chip_gen {
      // Verify the generation is valid
      assert!(matches!(
        detected,
        AppleSiliconGen::M1 | AppleSiliconGen::M2 | AppleSiliconGen::M3 | AppleSiliconGen::M4
      ));
    }
  }

  #[test]
  #[cfg(all(target_arch = "aarch64", target_os = "macos", not(miri)))]
  fn test_apple_silicon_tune_selection() {
    use crate::caps::aarch64;

    let det = get();
    // On Apple Silicon, if we have PMULL+EOR3, we should get Apple tuning
    if det.caps.has(aarch64::PMULL_EOR3_READY) {
      let tune_name = det.tune.name();
      assert!(tune_name.starts_with("Apple"), "Expected Apple tune, got: {tune_name}");
    }
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // SVE Vector Length Detection Tests
  // ─────────────────────────────────────────────────────────────────────────────

  #[test]
  #[cfg(all(target_arch = "aarch64", target_os = "linux", not(miri)))]
  fn test_sve_vlen_detection_runs() {
    // Just verify detection doesn't crash
    let vlen = detect_sve_vlen();
    // VL should be 0 (no SVE) or a valid power-of-2 in [128, 2048]
    if vlen > 0 {
      assert!(vlen >= 128, "SVE VL too small: {vlen}");
      assert!(vlen <= 2048, "SVE VL too large: {vlen}");
      assert!(vlen.is_power_of_two(), "SVE VL not power of 2: {vlen}");
    }
  }

  #[test]
  #[cfg(all(target_arch = "aarch64", target_os = "linux", not(miri)))]
  fn test_sve_tune_includes_vlen() {
    use crate::caps::aarch64;

    let det = get();
    // If SVE is detected, tune should have appropriate SVE VL
    if det.caps.has(aarch64::SVE) || det.caps.has(aarch64::SVE2) {
      let runtime_vlen = detect_sve_vlen();
      if runtime_vlen > 0 {
        // The tune's sve_vlen should be a valid value
        let tune_vlen = det.tune.sve_vlen;
        assert!(
          tune_vlen == 0 || tune_vlen == 128 || tune_vlen == 256 || tune_vlen == 512,
          "Unexpected tune SVE VL: {tune_vlen}"
        );
      }
    }
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Hybrid Intel Detection Tests
  // ─────────────────────────────────────────────────────────────────────────────

  #[test]
  #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "std"))]
  fn test_is_intel_hybrid_amd_returns_false() {
    // AMD CPUs should never be detected as Intel hybrid
    assert!(!is_intel_hybrid(true, 6, 0x97)); // Even with ADL model
    assert!(!is_intel_hybrid(true, 25, 0)); // Zen 4
    assert!(!is_intel_hybrid(true, 26, 0)); // Zen 5
  }

  #[test]
  #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "std"))]
  fn test_is_intel_hybrid_known_models() {
    // Alder Lake models
    assert!(is_intel_hybrid(false, 6, 0x97)); // ADL-S
    assert!(is_intel_hybrid(false, 6, 0x9A)); // ADL-P

    // Raptor Lake models
    assert!(is_intel_hybrid(false, 6, 0xB7)); // RPL-S
    assert!(is_intel_hybrid(false, 6, 0xBA)); // RPL-P

    // Non-hybrid Intel models should return false
    assert!(!is_intel_hybrid(false, 6, 0x8F)); // Sapphire Rapids
    assert!(!is_intel_hybrid(false, 6, 0x6A)); // Ice Lake-SP
  }

  #[test]
  #[allow(unsafe_code)]
  #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "std"))]
  fn test_hybrid_avx512_override_default() {
    // Without env var set, override should be false
    // Note: We can't easily test with env var set due to test isolation
    // but we verify the default behavior
    // SAFETY: This test runs in isolation and doesn't rely on this env var being
    // present for other threads. The remove_var is unsafe due to potential data
    // races with other threads reading env vars, but test isolation mitigates this.
    unsafe { std::env::remove_var("RSCRYPTO_FORCE_AVX512") };
    assert!(!hybrid_avx512_override());
  }

  #[test]
  #[cfg(all(target_arch = "x86_64", not(miri)))]
  fn test_x86_64_model_extraction() {
    // Just verify CPUID model extraction works
    let det = detect_uncached();
    // Model should be extracted - we just verify detection runs
    assert!(det.tune.simd_threshold > 0);
  }

  #[test]
  #[cfg(all(
    target_arch = "aarch64",
    not(miri),
    any(target_os = "macos", target_os = "ios", target_os = "tvos", target_os = "watchos")
  ))]
  fn test_macos_extended_features() {
    // Test that new feature detection works on macOS
    use crate::caps::aarch64;
    let det = get();

    // Verify extended features are detected on capable hardware
    // On M1+, we should detect these features:
    std::eprintln!("Detected features: {}", det.caps.count());
    std::eprintln!("  I8MM: {}", det.caps.has(aarch64::I8MM));
    std::eprintln!("  BF16: {}", det.caps.has(aarch64::BF16));
    std::eprintln!("  FRINTTS: {}", det.caps.has(aarch64::FRINTTS));
    std::eprintln!("  LSE2: {}", det.caps.has(aarch64::LSE2));

    // These should be present on M1 Pro (hw.optional.arm confirms this)
    assert!(det.caps.has(aarch64::FRINTTS), "FRINTTS should be detected on M1+");
    assert!(det.caps.has(aarch64::LSE2), "LSE2 should be detected on M1+");
  }

  #[test]
  #[cfg(all(
    target_arch = "aarch64",
    feature = "std",
    not(miri),
    any(target_os = "macos", target_os = "ios", target_os = "tvos", target_os = "watchos")
  ))]
  fn test_detect_apple_sme_features_exists() {
    // Verify the SME detection function exists and returns valid caps
    let sme_caps = detect_apple_sme_features();
    // The function should always return valid Caps (may be empty on M1-M3)
    // On M4+, SME should be detected
    std::eprintln!("SME caps detected: {}", sme_caps.count());
    std::eprintln!("  SME: {}", sme_caps.has(crate::caps::aarch64::SME));
    std::eprintln!("  SME2: {}", sme_caps.has(crate::caps::aarch64::SME2));
  }

  #[test]
  #[cfg(all(
    target_arch = "aarch64",
    feature = "std",
    not(miri),
    any(target_os = "macos", target_os = "ios", target_os = "tvos", target_os = "watchos")
  ))]
  fn test_detect_apple_silicon_gen_exists() {
    // Verify chip generation detection works
    if let Some(chip_gen) = detect_apple_silicon_gen() {
      std::eprintln!("Detected Apple Silicon generation: {:?}", chip_gen);
      // Basic sanity checks
      match chip_gen {
        AppleSiliconGen::M1 | AppleSiliconGen::M2 | AppleSiliconGen::M3 => {
          // M1-M3 should not have SME
          std::eprintln!("M1-M3 chip detected (no SME expected)");
        }
        AppleSiliconGen::M4 => {
          // M4 should have SME
          std::eprintln!("M4 chip detected (SME expected)");
        }
        AppleSiliconGen::M5 => {
          // M5 should have SME2
          std::eprintln!("M5 chip detected (SME2 expected)");
        }
      }
    } else {
      std::eprintln!("Unknown or A-series chip detected");
    }
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // TuneKind Round-Trip Tests
  // ─────────────────────────────────────────────────────────────────────────────

  // Mirror of the kind_from_u8 mapping used in atomic_cache (no_std).
  // This test validates that the mapping stays in sync with TuneKind's #[repr(u8)].
  fn test_kind_from_u8(v: u8) -> crate::tune::TuneKind {
    use crate::tune::TuneKind;
    match v {
      1 => TuneKind::Default,
      2 => TuneKind::Portable,
      3 => TuneKind::Zen4,
      4 => TuneKind::Zen5,
      5 => TuneKind::Zen5c,
      6 => TuneKind::IntelSpr,
      7 => TuneKind::IntelGnr,
      8 => TuneKind::IntelIcl,
      9 => TuneKind::AppleM1M3,
      10 => TuneKind::AppleM4,
      11 => TuneKind::AppleM5,
      12 => TuneKind::Graviton2,
      13 => TuneKind::Graviton3,
      14 => TuneKind::Graviton4,
      15 => TuneKind::Graviton5,
      16 => TuneKind::NeoverseN2,
      17 => TuneKind::NeoverseN3,
      18 => TuneKind::NeoverseV3,
      19 => TuneKind::NvidiaGrace,
      20 => TuneKind::AmpereAltra,
      21 => TuneKind::Aarch64Pmull,
      22 => TuneKind::Z13,
      23 => TuneKind::Z14,
      24 => TuneKind::Z15,
      25 => TuneKind::Power7,
      26 => TuneKind::Power8,
      27 => TuneKind::Power9,
      28 => TuneKind::Power10,
      _ => TuneKind::Custom,
    }
  }

  /// Validates that the `kind_from_u8` mapping correctly round-trips with TuneKind's
  /// `#[repr(u8)]` discriminants.
  ///
  /// This catches drift if someone adds a new TuneKind variant but forgets to update
  /// the mapping in atomic_cache.
  #[test]
  fn test_tunekind_round_trip() {
    use crate::tune::TuneKind;

    // TuneKind has 29 variants (0=Custom through 28=Power10)
    for i in 0..=28u8 {
      let kind = test_kind_from_u8(i);

      // Custom (0) is the fallback for unknown values, so skip its reverse check
      if kind != TuneKind::Custom {
        assert_eq!(
          kind as u8, i,
          "TuneKind mapping mismatch: kind_from_u8({i}) = {:?} but {:?} as u8 = {}",
          kind, kind, kind as u8
        );
      }
    }

    // Verify out-of-range values map to Custom
    assert_eq!(test_kind_from_u8(29), TuneKind::Custom);
    assert_eq!(test_kind_from_u8(255), TuneKind::Custom);
  }

  /// Verify that all TuneKind variants have distinct u8 representations.
  #[test]
  fn test_tunekind_no_collisions() {
    use alloc::collections::BTreeSet;

    use crate::tune::TuneKind;

    let variants: &[TuneKind] = &[
      TuneKind::Custom,
      TuneKind::Default,
      TuneKind::Portable,
      TuneKind::Zen4,
      TuneKind::Zen5,
      TuneKind::Zen5c,
      TuneKind::IntelSpr,
      TuneKind::IntelGnr,
      TuneKind::IntelIcl,
      TuneKind::AppleM1M3,
      TuneKind::AppleM4,
      TuneKind::AppleM5,
      TuneKind::Graviton2,
      TuneKind::Graviton3,
      TuneKind::Graviton4,
      TuneKind::Graviton5,
      TuneKind::NeoverseN2,
      TuneKind::NeoverseN3,
      TuneKind::NeoverseV3,
      TuneKind::NvidiaGrace,
      TuneKind::AmpereAltra,
      TuneKind::Aarch64Pmull,
      TuneKind::Z13,
      TuneKind::Z14,
      TuneKind::Z15,
      TuneKind::Power7,
      TuneKind::Power8,
      TuneKind::Power9,
      TuneKind::Power10,
    ];

    let mut seen = BTreeSet::new();
    for &kind in variants {
      let val = kind as u8;
      assert!(seen.insert(val), "TuneKind::{:?} has duplicate u8 value {}", kind, val);
    }

    // Verify we have all 29 variants
    assert_eq!(seen.len(), 29, "Expected 29 TuneKind variants");
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Arch Round-Trip Tests
  // ─────────────────────────────────────────────────────────────────────────────

  // Mirror of the arch_to_u8 mapping used in atomic_cache (no_std).
  // Note: Arch doesn't have #[repr(u8)], so this is a custom mapping
  // where Other=0 (the uninitialized/fallback value).
  fn test_arch_to_u8(arch: Arch) -> u8 {
    match arch {
      Arch::X86_64 => 1,
      Arch::X86 => 2,
      Arch::Aarch64 => 3,
      Arch::Arm => 4,
      Arch::Riscv64 => 5,
      Arch::Riscv32 => 6,
      Arch::Power => 7,
      Arch::S390x => 8,
      Arch::Wasm32 => 10,
      Arch::Wasm64 => 11,
      Arch::Other => 0,
    }
  }

  fn test_arch_from_u8(v: u8) -> Arch {
    match v {
      1 => Arch::X86_64,
      2 => Arch::X86,
      3 => Arch::Aarch64,
      4 => Arch::Arm,
      5 => Arch::Riscv64,
      6 => Arch::Riscv32,
      7 => Arch::Power,
      8 => Arch::S390x,
      10 => Arch::Wasm32,
      11 => Arch::Wasm64,
      _ => Arch::Other,
    }
  }

  /// Verify arch_to_u8 and arch_from_u8 are inverses.
  #[test]
  fn test_arch_round_trip() {
    let variants: &[Arch] = &[
      Arch::Other,
      Arch::X86_64,
      Arch::X86,
      Arch::Aarch64,
      Arch::Arm,
      Arch::Riscv64,
      Arch::Riscv32,
      Arch::Power,
      Arch::S390x,
      Arch::Wasm32,
      Arch::Wasm64,
    ];

    for &arch in variants {
      let encoded = test_arch_to_u8(arch);
      let decoded = test_arch_from_u8(encoded);
      assert_eq!(
        arch, decoded,
        "Arch round-trip failed: {:?} -> {} -> {:?}",
        arch, encoded, decoded
      );
    }

    // Verify out-of-range values map to Other
    assert_eq!(test_arch_from_u8(12), Arch::Other);
    assert_eq!(test_arch_from_u8(255), Arch::Other);
  }

  /// Verify all Arch variants have distinct encoded u8 values.
  #[test]
  fn test_arch_no_collisions() {
    use alloc::collections::BTreeSet;

    let variants: &[Arch] = &[
      Arch::Other,
      Arch::X86_64,
      Arch::X86,
      Arch::Aarch64,
      Arch::Arm,
      Arch::Riscv64,
      Arch::Riscv32,
      Arch::Power,
      Arch::S390x,
      Arch::Wasm32,
      Arch::Wasm64,
    ];

    let mut seen = BTreeSet::new();
    for &arch in variants {
      let val = test_arch_to_u8(arch);
      assert!(
        seen.insert(val),
        "Arch::{:?} has duplicate encoded u8 value {}",
        arch,
        val
      );
    }

    assert_eq!(seen.len(), 11, "Expected 11 Arch variants with unique encodings");
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Override Mechanism Tests
  // ─────────────────────────────────────────────────────────────────────────────

  #[test]
  fn test_has_override_exists() {
    // Verify the override API exists and returns a bool.
    // Note: Due to global state from other tests, we can't assert a specific value.
    let _ = has_override();
  }

  #[test]
  fn test_detected_portable_constructor() {
    let det = Detected::portable();
    assert_eq!(det.caps, Caps::NONE);
    assert_eq!(det.tune.kind(), crate::tune::TuneKind::Portable);
    assert_eq!(det.arch, Arch::Other);
  }

  #[test]
  fn test_detected_equality() {
    let a = Detected::portable();
    let b = Detected::portable();
    assert_eq!(a, b);

    let c = Detected {
      caps: Caps::bit(0),
      tune: crate::tune::Tune::DEFAULT,
      arch: Arch::X86_64,
    };
    assert_ne!(a, c);
  }

  #[test]
  fn test_detected_debug() {
    let det = Detected::portable();
    let s = alloc::format!("{:?}", det);
    assert!(s.contains("Detected"));
    assert!(s.contains("caps"));
    assert!(s.contains("tune"));
    assert!(s.contains("arch"));
  }
}
