#[cfg(test)]
extern crate std;

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  #[cfg(not(miri))] // get() returns portable() under Miri, which has different arch
  fn test_get_returns_valid() {
    let det = get();

    assert_eq!(det.arch, Arch::current());
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
  // `feature = "portable-only"` intentionally short-circuits `caps()` to
  // `Caps::NONE`, which would mismatch `get().caps` on a SIMD-capable host.
  // The convenience-function contract under SIMD-on dispatch is what this
  // test asserts; the portable-only override has its own coverage in
  // `test_caps_returns_none_with_portable_only_feature` below.
  #[cfg(not(feature = "portable-only"))]
  fn test_convenience_functions() {
    let det = get();
    assert_eq!(caps(), det.caps);
    assert_eq!(arch(), det.arch);
  }

  #[test]
  #[cfg(all(feature = "portable-only", not(miri)))]
  fn test_caps_returns_none_with_portable_only_feature() {
    // The `portable-only` feature must collapse `caps()` to the empty cap
    // set so every dispatcher falls through to its portable backend.
    assert_eq!(caps(), Caps::NONE, "portable-only must zero out caps()");
    // `arch()` is unaffected — only the cap surface is suppressed.
    let det = get();
    assert_eq!(arch(), det.arch);
  }

  #[test]
  #[cfg(all(target_arch = "x86_64", not(miri)))]
  fn test_x86_64_baseline() {
    use crate::platform::caps::x86;
    let det = get();
    assert!(det.caps.has(x86::SSE2));
  }

  #[test]
  #[cfg(target_arch = "x86_64")]
  fn x86_64_amx_caps_require_process_permission() {
    use crate::platform::caps::x86;

    let caps = x86::SSE2 | X86_ALL_AMX;
    assert_eq!(gate_x86_amx_permission(caps, true), caps);
    assert_eq!(
      gate_x86_amx_permission(caps, false),
      x86::SSE2,
      "denied process permission must remove every AMX capability"
    );
  }

  #[test]
  #[cfg(all(
    target_arch = "x86_64",
    not(feature = "std"),
    any(target_os = "linux", target_os = "android"),
    target_feature = "amx-tile",
    not(miri)
  ))]
  fn no_std_linux_x86_64_masks_compile_time_amx_without_a_permission_probe() {
    use crate::platform::caps::x86;

    assert!(caps_static().has(x86::AMX_TILE));
    assert!(
      detect_uncached().caps.intersection(X86_ALL_AMX).is_empty(),
      "no_std Linux/Android cannot publish AMX without an xcomp permission probe"
    );
  }

  #[test]
  #[cfg(all(target_arch = "aarch64", not(miri)))]
  fn test_aarch64_baseline() {
    use crate::platform::caps::aarch64;
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

  // Compile-Time Detection Tests (caps_static)

  #[test]
  fn test_caps_static_is_const() {
    // Verify caps_static() can be used in const context
    const STATIC_CAPS: Caps = caps_static();
    let _ = STATIC_CAPS; // Use it to avoid dead code warning
  }

  #[test]
  #[cfg(target_arch = "x86_64")]
  fn test_caps_static_x86_64_baseline() {
    use crate::platform::caps::x86;

    // x86_64 guarantees SSE2
    let caps = caps_static();
    assert!(caps.has(x86::SSE2), "x86_64 must have SSE2 baseline in caps_static");
  }

  #[test]
  #[cfg(target_arch = "aarch64")]
  fn test_caps_static_aarch64_baseline() {
    use crate::platform::caps::aarch64;

    // aarch64 guarantees NEON
    let caps = caps_static();
    assert!(
      caps.has(aarch64::NEON),
      "aarch64 must have NEON baseline in caps_static"
    );
  }

  #[test]
  #[cfg(not(miri))] // Miri can't detect runtime features, returns Caps::NONE
  // The "static is a subset of runtime" invariant assumes runtime detection
  // is enabled. With `portable-only`, runtime is intentionally `Caps::NONE`,
  // and `caps_static()` may be non-empty — they're allowed to disagree
  // because the override is the whole point of the feature.
  #[cfg(not(feature = "portable-only"))]
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
    use crate::platform::caps::x86;

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
    use crate::platform::caps::aarch64;

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

  // Apple Silicon Detection Tests

  #[test]
  #[cfg(all(target_arch = "aarch64", target_os = "macos", feature = "std", not(miri)))]
  fn test_apple_silicon_detection_runs() {
    // Just verify detection doesn't crash and returns a valid result
    let chip_gen = detect_apple_silicon_gen();
    // On actual Apple Silicon, we should get Some variant
    // On Rosetta 2 or non-Apple aarch64, we might get None
    if let Some(detected) = chip_gen {
      // Verify the generation is valid
      assert!(matches!(
        detected,
        AppleSiliconGen::M1
          | AppleSiliconGen::M2
          | AppleSiliconGen::M3
          | AppleSiliconGen::M4
          | AppleSiliconGen::M5
      ));
    }
  }

  // SVE Vector Length Detection Tests

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

  // Hybrid Intel Detection Tests

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
  #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "std"))]
  fn test_is_intel_sapphire_rapids_known_model() {
    assert!(is_intel_sapphire_rapids(true, 6, 0x8F));

    assert!(!is_intel_sapphire_rapids(false, 6, 0x8F)); // AMD/unknown vendor
    assert!(!is_intel_sapphire_rapids(true, 6, 0x6A)); // Ice Lake-SP
    assert!(!is_intel_sapphire_rapids(true, 6, 0xCF)); // Emerald Rapids
    assert!(!is_intel_sapphire_rapids(true, 25, 0)); // Non-family-6 CPU
  }

  #[test]
  #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "std"))]
  fn test_hybrid_avx512_override_parser() {
    for value in [None, Some(""), Some("0"), Some("false"), Some("yes"), Some("2")] {
      assert!(!parse_hybrid_avx512_override(value), "accepted {value:?}");
    }
    for value in [Some("1"), Some("true"), Some("TRUE"), Some("TrUe")] {
      assert!(parse_hybrid_avx512_override(value), "rejected {value:?}");
    }
  }

  #[test]
  #[cfg(all(target_arch = "x86_64", not(miri)))]
  fn test_x86_64_model_extraction() {
    // Just verify CPUID model extraction works
    let det = detect_uncached();
    assert_eq!(det.arch, Arch::X86_64);
    assert!(det.caps.count() >= 1);
  }

  #[cfg(all(target_arch = "x86_64", feature = "std"))]
  fn cpuid_feature_snapshot() -> CpuidSnapshot {
    CpuidSnapshot {
      leaf0: CpuidRegisters {
        eax: 0x24,
        ..CpuidRegisters::default()
      },
      leaf1: CpuidRegisters {
        ecx: 1 << 27,
        ..CpuidRegisters::default()
      },
      leaf7_0: CpuidRegisters {
        eax: 1,
        ..CpuidRegisters::default()
      },
      extended_leaf0: CpuidRegisters {
        eax: 0x8000_0001,
        ..CpuidRegisters::default()
      },
      xcr0: 0x6 | 0xe0 | (1 << 17) | (1 << 18) | (1 << 19),
      amx_permission: true,
      ..CpuidSnapshot::default()
    }
  }

  #[cfg(all(target_arch = "x86_64", feature = "std"))]
  fn enable_avx(snapshot: &mut CpuidSnapshot) {
    snapshot.leaf1.ecx |= 1 << 28;
  }

  #[cfg(all(target_arch = "x86_64", feature = "std"))]
  fn enable_avx512(snapshot: &mut CpuidSnapshot) {
    enable_avx(snapshot);
    snapshot.leaf1.ecx |= (1 << 12) | (1 << 29);
    snapshot.leaf7_0.ebx |= 1 << 16;
  }

  #[cfg(all(target_arch = "x86_64", feature = "std"))]
  fn avx_caps() -> Caps {
    use crate::platform::caps::x86;

    x86::AVX
  }

  #[cfg(all(target_arch = "x86_64", feature = "std"))]
  fn avx512_caps() -> Caps {
    use crate::platform::caps::x86;

    x86::AVX | x86::FMA | x86::F16C | x86::AVX512F
  }

  #[test]
  #[cfg(all(target_arch = "x86_64", feature = "std"))]
  fn x86_cpuid_feature_bits_decode_from_their_architectural_registers() {
    use crate::platform::caps::x86;

    fn assert_feature(expected: Caps, configure: impl FnOnce(&mut CpuidSnapshot)) {
      let mut snapshot = cpuid_feature_snapshot();
      configure(&mut snapshot);
      assert_eq!(decode_cpuid_x86_64(snapshot).caps, expected);
    }

    macro_rules! feature_case {
      ($expected:expr, $leaf:ident, $register:ident, $bit:expr) => {
        assert_feature($expected, |snapshot| snapshot.$leaf.$register |= 1 << $bit);
      };
    }

    feature_case!(x86::SSE3, leaf1, ecx, 0);
    feature_case!(x86::PCLMULQDQ, leaf1, ecx, 1);
    feature_case!(x86::SSSE3, leaf1, ecx, 9);
    feature_case!(x86::SSE41, leaf1, ecx, 19);
    feature_case!(x86::SSE42, leaf1, ecx, 20);
    feature_case!(x86::POPCNT, leaf1, ecx, 23);
    feature_case!(x86::AESNI, leaf1, ecx, 25);
    feature_case!(x86::RDRAND, leaf1, ecx, 30);

    feature_case!(x86::BMI1, leaf7_0, ebx, 3);
    feature_case!(x86::BMI2, leaf7_0, ebx, 8);
    feature_case!(x86::RDSEED, leaf7_0, ebx, 18);
    feature_case!(x86::ADX, leaf7_0, ebx, 19);
    feature_case!(x86::SHA, leaf7_0, ebx, 29);

    feature_case!(x86::GFNI, leaf7_0, ecx, 8);
    feature_case!(x86::AMX_BF16, leaf7_0, edx, 22);
    feature_case!(x86::AMX_TILE, leaf7_0, edx, 24);
    feature_case!(x86::AMX_INT8, leaf7_0, edx, 25);
    feature_case!(x86::SHA512, leaf7_1, eax, 0);
    feature_case!(x86::AMX_FP16, leaf7_1, eax, 21);
    feature_case!(x86::AMX_COMPLEX, leaf7_1, edx, 8);
    feature_case!(x86::MOVDIRI, leaf7_0, ecx, 27);
    feature_case!(x86::MOVDIR64B, leaf7_0, ecx, 28);
    feature_case!(x86::SERIALIZE, leaf7_0, edx, 14);

    feature_case!(x86::LZCNT, extended_leaf1, ecx, 5);
    feature_case!(x86::SSE4A, extended_leaf1, ecx, 6);

    assert_feature(avx_caps(), enable_avx);
    assert_feature(x86::AVX | x86::FMA, |snapshot| {
      enable_avx(snapshot);
      snapshot.leaf1.ecx |= 1 << 12;
    });
    assert_feature(x86::AVX | x86::F16C, |snapshot| {
      enable_avx(snapshot);
      snapshot.leaf1.ecx |= 1 << 29;
    });
    assert_feature(x86::AVX | x86::AVX2, |snapshot| {
      enable_avx(snapshot);
      snapshot.leaf7_0.ebx |= 1 << 5;
    });

    let avx512_cases = [
      (x86::AVX512DQ, "ebx", 17),
      (x86::AVX512IFMA, "ebx", 21),
      (x86::AVX512CD, "ebx", 28),
      (x86::AVX512BW, "ebx", 30),
      (x86::AVX512VL, "ebx", 31),
      (x86::AVX512VBMI, "ecx", 1),
      (x86::AVX512VBMI2, "ecx", 6),
      (x86::AVX512VNNI, "ecx", 11),
      (x86::AVX512BITALG, "ecx", 12),
      (x86::AVX512VPOPCNTDQ, "ecx", 14),
      (x86::AVX512VP2INTERSECT, "edx", 8),
    ];
    for (feature, register, bit) in avx512_cases {
      assert_feature(avx512_caps() | feature, |snapshot| {
        enable_avx512(snapshot);
        match register {
          "ebx" => snapshot.leaf7_0.ebx |= 1 << bit,
          "ecx" => snapshot.leaf7_0.ecx |= 1 << bit,
          "edx" => snapshot.leaf7_0.edx |= 1 << bit,
          _ => unreachable!(),
        }
      });
    }
    assert_feature(avx512_caps(), enable_avx512);

    let avx512_bw_caps = avx512_caps() | x86::AVX512BW;
    assert_feature(avx512_bw_caps | x86::AVX512FP16, |snapshot| {
      enable_avx512(snapshot);
      snapshot.leaf7_0.ebx |= 1 << 30;
      snapshot.leaf7_0.edx |= 1 << 23;
    });
    assert_feature(avx512_bw_caps | x86::AVX512BF16, |snapshot| {
      enable_avx512(snapshot);
      snapshot.leaf7_0.ebx |= 1 << 30;
      snapshot.leaf7_1.eax |= 1 << 5;
    });

    assert_feature(x86::AVX | x86::AVX2 | x86::AESNI | x86::VAES, |snapshot| {
      enable_avx(snapshot);
      snapshot.leaf1.ecx |= 1 << 25;
      snapshot.leaf7_0.ebx |= 1 << 5;
      snapshot.leaf7_0.ecx |= 1 << 9;
    });
    assert_feature(x86::AVX | x86::PCLMULQDQ | x86::VPCLMULQDQ, |snapshot| {
      enable_avx(snapshot);
      snapshot.leaf1.ecx |= 1 << 1;
      snapshot.leaf7_0.ecx |= 1 << 10;
    });
    assert_feature(x86::APX, |snapshot| snapshot.leaf7_1.edx |= 1 << 21);
    assert_feature(avx512_caps() | x86::AVX10_1, |snapshot| {
      enable_avx512(snapshot);
      snapshot.leaf7_1.edx |= 1 << 19;
      snapshot.leaf24_0.ebx = 1;
    });
    assert_feature(avx512_caps() | x86::AVX10_1, |snapshot| {
      enable_avx512(snapshot);
      snapshot.leaf7_1.edx |= 1 << 19;
      snapshot.leaf24_0.ebx = 2;
    });
  }

  #[test]
  #[cfg(all(target_arch = "x86_64", feature = "std"))]
  fn x86_cpuid_decoder_rejects_unsupported_leaves_and_missing_os_state() {
    use crate::platform::caps::x86;

    let mut snapshot = cpuid_feature_snapshot();
    snapshot.leaf0.eax = 6;
    snapshot.leaf7_0 = CpuidRegisters {
      eax: u32::MAX,
      ebx: u32::MAX,
      ecx: u32::MAX,
      edx: u32::MAX,
    };
    snapshot.leaf7_1 = snapshot.leaf7_0;
    snapshot.leaf24_0 = snapshot.leaf7_0;
    assert!(decode_cpuid_x86_64(snapshot).caps.is_empty());

    let mut snapshot = cpuid_feature_snapshot();
    snapshot.extended_leaf0.eax = 0x8000_0000;
    snapshot.extended_leaf1.ecx = (1 << 5) | (1 << 6);
    assert!(
      decode_cpuid_x86_64(snapshot)
        .caps
        .intersection(x86::LZCNT.union(x86::SSE4A))
        .is_empty()
    );

    let mut snapshot = cpuid_feature_snapshot();
    snapshot.leaf7_0.eax = 0;
    snapshot.leaf7_1.eax = (1 << 0) | (1 << 4) | (1 << 5) | (1 << 21);
    snapshot.leaf7_1.edx = 1 << 8;
    assert!(decode_cpuid_x86_64(snapshot).caps.is_empty());

    let mut snapshot = cpuid_feature_snapshot();
    snapshot.xcr0 = 0;
    snapshot.leaf7_0.edx = (1 << 22) | (1 << 24) | (1 << 25);
    snapshot.leaf7_1.eax = 1 << 21;
    snapshot.leaf7_1.edx = 1 << 8;
    let amx = x86::AMX_TILE | x86::AMX_BF16 | x86::AMX_INT8 | x86::AMX_FP16 | x86::AMX_COMPLEX;
    assert!(decode_cpuid_x86_64(snapshot).caps.intersection(amx).is_empty());

    let mut snapshot = cpuid_feature_snapshot();
    snapshot.amx_permission = false;
    snapshot.leaf7_0.edx = (1 << 22) | (1 << 24) | (1 << 25);
    snapshot.leaf7_1.eax = 1 << 21;
    snapshot.leaf7_1.edx = 1 << 8;
    assert!(decode_cpuid_x86_64(snapshot).caps.intersection(amx).is_empty());

    let mut snapshot = cpuid_feature_snapshot();
    snapshot.leaf7_0.edx = 1 << 18;
    snapshot.leaf7_1.eax = 1 << 8;
    assert!(
      decode_cpuid_x86_64(snapshot)
        .caps
        .intersection(x86::RDSEED | x86::AMX_COMPLEX)
        .is_empty()
    );

    let mut snapshot = cpuid_feature_snapshot();
    snapshot.leaf7_0.ebx = 1 << 16;
    assert!(decode_cpuid_x86_64(snapshot).caps.intersection(avx512_caps()).is_empty());

    let mut snapshot = cpuid_feature_snapshot();
    snapshot.xcr0 &= !(1 << 19);
    snapshot.leaf7_1.edx = 1 << 21;
    assert!(!decode_cpuid_x86_64(snapshot).caps.has(x86::APX));

    let mut snapshot = cpuid_feature_snapshot();
    enable_avx512(&mut snapshot);
    snapshot.leaf24_0.ebx = 2;
    assert!(
      decode_cpuid_x86_64(snapshot)
        .caps
        .intersection(x86::AVX10_1 | x86::AVX10_2)
        .is_empty()
    );

    for configure in [
      |snapshot: &mut CpuidSnapshot| snapshot.leaf7_0.ecx |= 1 << 9,
      |snapshot: &mut CpuidSnapshot| snapshot.leaf7_0.ecx |= 1 << 10,
      |snapshot: &mut CpuidSnapshot| snapshot.leaf7_0.ebx |= 1 << 5,
    ] {
      let mut snapshot = cpuid_feature_snapshot();
      configure(&mut snapshot);
      assert!(
        decode_cpuid_x86_64(snapshot)
          .caps
          .intersection(x86::VAES | x86::VPCLMULQDQ | x86::AVX2)
          .is_empty()
      );
    }

    let mut snapshot = cpuid_feature_snapshot();
    enable_avx512(&mut snapshot);
    snapshot.leaf7_0.ebx |= 1 << 30;
    snapshot.leaf7_1.eax |= 1 << 4;
    assert!(!decode_cpuid_x86_64(snapshot).caps.has(x86::AVX512BF16));

    let mut snapshot = cpuid_feature_snapshot();
    enable_avx512(&mut snapshot);
    snapshot.leaf7_0.ebx |= 1 << 30;
    snapshot.leaf7_1.eax |= 1 << 5;
    assert!(!decode_cpuid_x86_64(snapshot).caps.has(x86::AVX512FP16));
  }

  #[test]
  #[cfg(all(
    target_arch = "aarch64",
    not(miri),
    any(target_os = "macos", target_os = "ios", target_os = "tvos", target_os = "watchos")
  ))]
  fn test_macos_extended_features() {
    // Test that new feature detection works on macOS
    use crate::platform::caps::aarch64;
    let det = get();

    // Verify extended features are detected on capable hardware
    // On M1+, we should detect these features:
    std::eprintln!("Detected features: {}", det.caps.count());
    std::eprintln!("  I8MM: {}", det.caps.has(aarch64::I8MM));
    std::eprintln!("  BF16: {}", det.caps.has(aarch64::BF16));
    std::eprintln!("  FRINTTS: {}", det.caps.has(aarch64::FRINTTS));
    std::eprintln!("  LSE2: {}", det.caps.has(aarch64::LSE2));

    // FRINTTS is detectable via std::arch on macOS; LSE2 is not exposed by
    // Apple's sysctl and therefore cannot be asserted here.
    assert!(det.caps.has(aarch64::FRINTTS), "FRINTTS should be detected on M1+");
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
    std::eprintln!("  SME: {}", sme_caps.has(crate::platform::caps::aarch64::SME));
    std::eprintln!("  SME2: {}", sme_caps.has(crate::platform::caps::aarch64::SME2));
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

  // Override Mechanism Tests

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
    assert_eq!(det.arch, Arch::Other);
  }

  #[test]
  fn test_detected_equality() {
    let a = Detected::portable();
    let b = Detected::portable();
    assert_eq!(a, b);

    let c = Detected {
      caps: Caps::bit(0),
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
    assert!(s.contains("arch"));
  }

  #[test]
  fn override_validation_accepts_portable_override() {
    assert_eq!(validate_override(Some(Detected::portable())), Ok(Some(Detected::portable())));
  }

  #[test]
  fn override_validation_accepts_current_arch_cap_subset() {
    let det = Detected {
      caps: Caps::NONE,
      arch: Arch::current(),
    };
    assert_eq!(validate_override(Some(det)), Ok(Some(det)));
  }

  #[test]
  fn override_validation_rejects_wrong_arch() {
    let wrong_arch = if Arch::current() == Arch::X86_64 {
      Arch::Aarch64
    } else {
      Arch::X86_64
    };
    let det = Detected {
      caps: Caps::NONE,
      arch: wrong_arch,
    };

    assert_eq!(validate_override(Some(det)), Err(OverrideError::InvalidCapabilities));
  }

  #[test]
  fn override_validation_rejects_impossible_caps() {
    let det = Detected {
      caps: Caps::from_words([u64::MAX; 4]),
      arch: Arch::current(),
    };

    assert_eq!(validate_override(Some(det)), Err(OverrideError::InvalidCapabilities));
  }
}
