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
  #[cfg(all(
    target_arch = "x86_64",
    any(feature = "std", target_os = "linux", target_os = "android")
  ))]
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
  // x86-64 also removes compile-time AMX and hybrid-unsafe AVX-512
  // capabilities when process or CPU-topology safety checks fail.
  #[cfg(all(not(feature = "portable-only"), not(target_arch = "x86_64")))]
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

  // Hybrid Intel Detection Tests

  #[test]
  #[cfg(all(target_arch = "x86_64", feature = "std"))]
  fn test_is_intel_hybrid_amd_returns_false() {
    // AMD CPUs should never be detected as Intel hybrid
    assert!(!is_intel_hybrid(true, 6, 0x97)); // Even with ADL model
    assert!(!is_intel_hybrid(true, 25, 0)); // Zen 4
    assert!(!is_intel_hybrid(true, 26, 0)); // Zen 5
  }

  #[test]
  #[cfg(all(target_arch = "x86_64", feature = "std"))]
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
  #[cfg(all(target_arch = "x86_64", feature = "std"))]
  fn test_is_intel_sapphire_rapids_known_model() {
    assert!(is_intel_sapphire_rapids(true, 6, 0x8F));

    assert!(!is_intel_sapphire_rapids(false, 6, 0x8F)); // AMD/unknown vendor
    assert!(!is_intel_sapphire_rapids(true, 6, 0x6A)); // Ice Lake-SP
    assert!(!is_intel_sapphire_rapids(true, 6, 0xCF)); // Emerald Rapids
    assert!(!is_intel_sapphire_rapids(true, 25, 0)); // Non-family-6 CPU
  }

  #[test]
  #[cfg(not(all(target_arch = "x86_64", feature = "std")))]
  fn hybrid_avx512_opt_in_is_unsupported_without_runtime_avx512_detection() {
    // SAFETY: this build has no runtime AVX-512 detection, so the call cannot
    // enable an instruction.
    assert_eq!(unsafe { allow_hybrid_avx512() }, Err(OverrideError::Unsupported));
  }

  #[test]
  #[cfg(all(target_arch = "x86_64", feature = "std"))]
  fn x86_64_hybrid_avx512_requires_the_process_opt_in() {
    use crate::platform::caps::x86;

    // Hybrid E-cores lack every AVX-512 and AVX10 instruction but implement
    // AVX2, VAES, VPCLMULQDQ, and GFNI, which stay on every path.
    let shared = x86::SSE2 | x86::AVX2 | x86::VAES | x86::VPCLMULQDQ | x86::GFNI;
    let p_core_only = [
      x86::AVX512F,
      x86::AVX512DQ,
      x86::AVX512IFMA,
      x86::AVX512CD,
      x86::AVX512BW,
      x86::AVX512VL,
      x86::AVX512VBMI,
      x86::AVX512VBMI2,
      x86::AVX512VNNI,
      x86::AVX512BITALG,
      x86::AVX512VPOPCNTDQ,
      x86::AVX512BF16,
      x86::AVX512FP16,
      x86::AVX512VP2INTERSECT,
      x86::AVX10_1,
      x86::AVX10_2,
    ];
    let reported = p_core_only.iter().fold(shared, |caps, &cap| caps | cap);

    assert_eq!(
      gate_x86_hybrid_avx512(reported, true, false),
      shared,
      "a hybrid CPU without the opt-in must lose every AVX-512 and AVX10 capability"
    );
    assert_eq!(gate_x86_hybrid_avx512(reported, true, true), reported);
    assert_eq!(gate_x86_hybrid_avx512(reported, false, false), reported);
    assert_eq!(
      gate_x86_hybrid_avx512(shared, true, true),
      shared,
      "the opt-in keeps reported capabilities and never adds one"
    );
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

  /// SSE3, SSSE3, SSE4.1, and SSE4.2, which Rust's `avx` implies.
  #[cfg(all(target_arch = "x86_64", feature = "std"))]
  fn enable_sse42(snapshot: &mut CpuidSnapshot) {
    snapshot.leaf1.ecx |= (1 << 0) | (1 << 9) | (1 << 19) | (1 << 20);
  }

  #[cfg(all(target_arch = "x86_64", feature = "std"))]
  fn enable_avx(snapshot: &mut CpuidSnapshot) {
    enable_sse42(snapshot);
    snapshot.leaf1.ecx |= 1 << 28;
  }

  #[cfg(all(target_arch = "x86_64", feature = "std"))]
  fn enable_avx2(snapshot: &mut CpuidSnapshot) {
    enable_avx(snapshot);
    snapshot.leaf7_0.ebx |= 1 << 5;
  }

  /// AVX-512F with AVX2, FMA, and F16C, which Rust's `avx512f` implies.
  #[cfg(all(target_arch = "x86_64", feature = "std"))]
  fn enable_avx512(snapshot: &mut CpuidSnapshot) {
    enable_avx2(snapshot);
    snapshot.leaf1.ecx |= (1 << 12) | (1 << 29);
    snapshot.leaf7_0.ebx |= 1 << 16;
  }

  #[cfg(all(target_arch = "x86_64", feature = "std"))]
  fn sse42_caps() -> Caps {
    use crate::platform::caps::x86;

    x86::SSE3 | x86::SSSE3 | x86::SSE41 | x86::SSE42
  }

  #[cfg(all(target_arch = "x86_64", feature = "std"))]
  fn avx_caps() -> Caps {
    use crate::platform::caps::x86;

    sse42_caps() | x86::AVX
  }

  #[cfg(all(target_arch = "x86_64", feature = "std"))]
  fn avx2_caps() -> Caps {
    use crate::platform::caps::x86;

    avx_caps() | x86::AVX2
  }

  #[cfg(all(target_arch = "x86_64", feature = "std"))]
  fn avx512_caps() -> Caps {
    use crate::platform::caps::x86;

    avx2_caps() | x86::FMA | x86::F16C | x86::AVX512F
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
    feature_case!(x86::POPCNT, leaf1, ecx, 23);
    feature_case!(x86::AESNI, leaf1, ecx, 25);
    feature_case!(x86::RDRAND, leaf1, ecx, 30);

    feature_case!(x86::BMI1, leaf7_0, ebx, 3);
    feature_case!(x86::BMI2, leaf7_0, ebx, 8);
    feature_case!(x86::RDSEED, leaf7_0, ebx, 18);
    feature_case!(x86::ADX, leaf7_0, ebx, 19);
    feature_case!(x86::SHA, leaf7_0, ebx, 29);

    feature_case!(x86::GFNI, leaf7_0, ecx, 8);
    feature_case!(x86::AMX_TILE, leaf7_0, edx, 24);
    feature_case!(x86::MOVDIRI, leaf7_0, ecx, 27);
    feature_case!(x86::MOVDIR64B, leaf7_0, ecx, 28);
    feature_case!(x86::SERIALIZE, leaf7_0, edx, 14);

    feature_case!(x86::LZCNT, extended_leaf1, ecx, 5);

    // Each feature below also needs the features that Rust's target feature implies.
    assert_feature(x86::SSE3 | x86::SSSE3, |snapshot| snapshot.leaf1.ecx |= (1 << 0) | (1 << 9));
    assert_feature(x86::SSE3 | x86::SSSE3 | x86::SSE41, |snapshot| {
      snapshot.leaf1.ecx |= (1 << 0) | (1 << 9) | (1 << 19);
    });
    assert_feature(sse42_caps(), enable_sse42);
    assert_feature(x86::SSE3 | x86::SSE4A, |snapshot| {
      snapshot.leaf1.ecx |= 1 << 0;
      snapshot.extended_leaf1.ecx |= 1 << 6;
    });
    for (feature, edx, eax) in [
      (x86::AMX_BF16, 1 << 22, 0),
      (x86::AMX_INT8, 1 << 25, 0),
      (x86::AMX_FP16, 0, 1 << 21),
    ] {
      assert_feature(x86::AMX_TILE | feature, |snapshot| {
        snapshot.leaf7_0.edx |= (1 << 24) | edx;
        snapshot.leaf7_1.eax |= eax;
      });
    }
    assert_feature(x86::AMX_TILE | x86::AMX_COMPLEX, |snapshot| {
      snapshot.leaf7_0.edx |= 1 << 24;
      snapshot.leaf7_1.edx |= 1 << 8;
    });
    assert_feature(avx2_caps() | x86::SHA512, |snapshot| {
      enable_avx2(snapshot);
      snapshot.leaf7_1.eax |= 1 << 0;
    });

    assert_feature(avx_caps(), enable_avx);
    assert_feature(avx_caps() | x86::FMA, |snapshot| {
      enable_avx(snapshot);
      snapshot.leaf1.ecx |= 1 << 12;
    });
    assert_feature(avx_caps() | x86::F16C, |snapshot| {
      enable_avx(snapshot);
      snapshot.leaf1.ecx |= 1 << 29;
    });
    assert_feature(avx2_caps(), enable_avx2);

    enum Leaf7Register {
      Ebx,
      Ecx,
      Edx,
    }
    let avx512_cases = [
      (x86::AVX512DQ, Leaf7Register::Ebx, 17),
      (x86::AVX512IFMA, Leaf7Register::Ebx, 21),
      (x86::AVX512CD, Leaf7Register::Ebx, 28),
      (x86::AVX512BW, Leaf7Register::Ebx, 30),
      (x86::AVX512VL, Leaf7Register::Ebx, 31),
      (x86::AVX512VNNI, Leaf7Register::Ecx, 11),
      (x86::AVX512VPOPCNTDQ, Leaf7Register::Ecx, 14),
      (x86::AVX512VP2INTERSECT, Leaf7Register::Edx, 8),
    ];
    for (feature, register, bit) in avx512_cases {
      assert_feature(avx512_caps() | feature, |snapshot| {
        enable_avx512(snapshot);
        match register {
          Leaf7Register::Ebx => snapshot.leaf7_0.ebx |= 1 << bit,
          Leaf7Register::Ecx => snapshot.leaf7_0.ecx |= 1 << bit,
          Leaf7Register::Edx => snapshot.leaf7_0.edx |= 1 << bit,
        }
      });
    }
    assert_feature(avx512_caps(), enable_avx512);

    let avx512_bw_caps = avx512_caps() | x86::AVX512BW;
    for (feature, bit) in [
      (x86::AVX512VBMI, 1),
      (x86::AVX512VBMI2, 6),
      (x86::AVX512BITALG, 12),
    ] {
      assert_feature(avx512_bw_caps | feature, |snapshot| {
        enable_avx512(snapshot);
        snapshot.leaf7_0.ebx |= 1 << 30;
        snapshot.leaf7_0.ecx |= 1 << bit;
      });
    }
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

    assert_feature(avx2_caps() | x86::AESNI | x86::VAES, |snapshot| {
      enable_avx2(snapshot);
      snapshot.leaf1.ecx |= 1 << 25;
      snapshot.leaf7_0.ecx |= 1 << 9;
    });
    assert_feature(avx_caps() | x86::PCLMULQDQ | x86::VPCLMULQDQ, |snapshot| {
      enable_avx(snapshot);
      snapshot.leaf1.ecx |= 1 << 1;
      snapshot.leaf7_0.ecx |= 1 << 10;
    });
    assert_feature(x86::APX, |snapshot| snapshot.leaf7_1.edx |= 1 << 21);
    for version in [1, 2] {
      assert_feature(avx10_1_caps(), |snapshot| {
        enable_avx10_1(snapshot);
        snapshot.leaf24_0.ebx = version;
      });
    }
  }

  /// AVX10.1 with the AVX-512 features that Rust's `avx10.1` implies.
  #[cfg(all(target_arch = "x86_64", feature = "std"))]
  fn enable_avx10_1(snapshot: &mut CpuidSnapshot) {
    enable_avx512(snapshot);
    // BW, CD, DQ, IFMA, and VL; VBMI, VBMI2, VNNI, BITALG, and VPOPCNTDQ; FP16; BF16.
    snapshot.leaf7_0.ebx |= (1 << 17) | (1 << 21) | (1 << 28) | (1 << 30) | (1 << 31);
    snapshot.leaf7_0.ecx |= (1 << 1) | (1 << 6) | (1 << 11) | (1 << 12) | (1 << 14);
    snapshot.leaf7_0.edx |= 1 << 23;
    snapshot.leaf7_1.eax |= 1 << 5;
    snapshot.leaf7_1.edx |= 1 << 19;
    snapshot.leaf24_0.ebx = 1;
  }

  #[cfg(all(target_arch = "x86_64", feature = "std"))]
  fn avx10_1_caps() -> Caps {
    use crate::platform::caps::x86;

    avx512_caps()
      | x86::AVX512BW
      | x86::AVX512CD
      | x86::AVX512DQ
      | x86::AVX512IFMA
      | x86::AVX512VL
      | x86::AVX512VBMI
      | x86::AVX512VBMI2
      | x86::AVX512VNNI
      | x86::AVX512BITALG
      | x86::AVX512VPOPCNTDQ
      | x86::AVX512FP16
      | x86::AVX512BF16
      | x86::AVX10_1
  }

  /// A `#[target_feature]` kernel may execute the instructions of every feature that its
  /// features imply in Rust, so a CPUID that omits an implied feature, as a hypervisor's
  /// CPUID mask can, must not report the feature that implies it.
  #[test]
  #[cfg(all(target_arch = "x86_64", feature = "std"))]
  fn x86_cpuid_decoder_drops_features_whose_implied_features_are_missing() {
    use crate::platform::caps::x86;

    fn decode_without(configure: fn(&mut CpuidSnapshot), clear: impl FnOnce(&mut CpuidSnapshot)) -> Caps {
      let mut snapshot = cpuid_feature_snapshot();
      configure(&mut snapshot);
      clear(&mut snapshot);
      decode_cpuid_x86_64(snapshot).caps
    }

    // AVX-512F without AVX2: no AVX-512 capability at all.
    let caps = decode_without(enable_avx10_1, |snapshot| snapshot.leaf7_0.ebx &= !(1 << 5));
    assert!(!caps.has(x86::AVX2));
    let all_avx512 = avx10_1_caps().difference(avx2_caps().union(x86::FMA).union(x86::F16C));
    assert!(caps.intersection(all_avx512).is_empty(), "AVX-512 without AVX2");

    // AVX-512BW's dependents without AVX-512BW.
    let caps = decode_without(enable_avx10_1, |snapshot| snapshot.leaf7_0.ebx &= !(1 << 30));
    let bw_dependents =
      x86::AVX512VBMI | x86::AVX512VBMI2 | x86::AVX512BITALG | x86::AVX512FP16 | x86::AVX512BF16 | x86::AVX10_1;
    assert!(caps.intersection(bw_dependents).is_empty(), "AVX-512BW dependents");
    assert!(caps.has(x86::AVX512F | x86::AVX512VL | x86::AVX512VNNI));

    // AVX10.1 without one AVX-512 feature that it implies.
    let caps = decode_without(enable_avx10_1, |snapshot| snapshot.leaf7_0.ecx &= !(1 << 11));
    assert!(!caps.has(x86::AVX10_1), "AVX10.1 without AVX-512 VNNI");

    // The SSE chain under AVX: SSE4.2 without SSE4.1 removes SSE4.2 and every AVX feature.
    let caps = decode_without(enable_avx512, |snapshot| snapshot.leaf1.ecx &= !(1 << 19));
    assert_eq!(caps, x86::SSE3 | x86::SSSE3);
    let caps = decode_without(enable_sse42, |snapshot| snapshot.leaf1.ecx &= !(1 << 0));
    assert!(caps.is_empty(), "SSSE3, SSE4.1, and SSE4.2 without SSE3");

    // SHA512 and VAES without AVX2.
    let caps = decode_without(enable_avx, |snapshot| {
      snapshot.leaf1.ecx |= 1 << 25;
      snapshot.leaf7_0.ecx |= 1 << 9;
      snapshot.leaf7_1.eax |= 1 << 0;
    });
    assert_eq!(caps, avx_caps() | x86::AESNI);

    // The AMX extensions without AMX-TILE.
    let caps = decode_without(
      |snapshot| {
        snapshot.leaf7_0.edx |= (1 << 22) | (1 << 25);
        snapshot.leaf7_1.eax |= 1 << 21;
        snapshot.leaf7_1.edx |= 1 << 8;
      },
      |_| {},
    );
    assert!(caps.is_empty(), "AMX extensions without AMX-TILE");
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
    use crate::platform::caps::aarch64;
    let det = get();

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
  fn test_detect_apple_sme_features_are_scoped() {
    use crate::platform::caps::aarch64;

    let sme_caps = detect_apple_sme_features();
    let owned_caps = aarch64::SME
      .union(aarch64::SME2)
      .union(aarch64::SME2P1)
      .union(aarch64::SME_I16I64)
      .union(aarch64::SME_F64F64)
      .union(aarch64::SME_B16B16)
      .union(aarch64::SME_F16F16);
    assert!(sme_caps.difference(owned_caps).is_empty());
  }

  /// Racing first callers must agree on one published value, even when each probe returns a
  /// different result, and a published cache must not run detection again.
  #[test]
  #[cfg(feature = "std")]
  fn cache_publishes_one_result_to_concurrent_first_callers() {
    use core::sync::atomic::{AtomicU8, Ordering};
    use std::sync::Barrier;

    static PROBES: AtomicU8 = AtomicU8::new(0);
    fn distinct() -> Detected {
      let probe = PROBES.fetch_add(1, Ordering::Relaxed);
      Detected {
        caps: Caps::bit(probe % 64),
        arch: Arch::current(),
      }
    }

    const THREADS: usize = 8;
    let cache = atomic_cache::Cache::new();
    let barrier = Barrier::new(THREADS);
    let results: alloc::vec::Vec<Detected> = std::thread::scope(|scope| {
      let handles: alloc::vec::Vec<_> = (0..THREADS)
        .map(|_| {
          scope.spawn(|| {
            barrier.wait();
            cache.get_or_init(distinct)
          })
        })
        .collect();
      handles
        .into_iter()
        .map(|handle| handle.join().expect("first caller must not panic"))
        .collect()
    });

    let published = results[0];
    assert!(results.iter().all(|result| *result == published), "{results:?}");
    let probes = PROBES.load(Ordering::Relaxed);
    assert_eq!(cache.get_or_init(distinct), published);
    assert_eq!(PROBES.load(Ordering::Relaxed), probes, "a published cache detected again");
  }

  /// A panicking probe must leave the cache uninitialized rather than wedged, so the next
  /// caller detects and publishes normally.
  #[test]
  #[cfg(all(feature = "std", panic = "unwind"))]
  fn cache_survives_a_panicking_probe() {
    #[expect(clippy::panic, reason = "the test needs a probe that panics")]
    fn panics() -> Detected {
      panic!("probe failure")
    }
    fn portable() -> Detected {
      Detected::portable()
    }

    let cache = atomic_cache::Cache::new();
    let outcome = std::panic::catch_unwind(core::panic::AssertUnwindSafe(|| cache.get_or_init(panics)));
    let _panic = outcome.expect_err("the probe must panic");
    assert_eq!(cache.get_or_init(portable), Detected::portable());
  }

  /// The owned sysctl reads must report exactly what the standard library's Darwin detector
  /// reports for every capability the replaced macro calls produced.
  #[test]
  #[cfg(all(
    target_arch = "aarch64",
    feature = "std",
    not(miri),
    any(target_os = "macos", target_os = "ios", target_os = "tvos", target_os = "watchos")
  ))]
  fn apple_owned_detection_matches_std_detect() {
    use std::arch::is_aarch64_feature_detected as std_has;

    use crate::platform::caps::aarch64;

    let owned = caps_static() | runtime_aarch64();
    let expected = [
      ("aes", aarch64::AES, std_has!("aes")),
      ("pmull", aarch64::PMULL, std_has!("pmull")),
      ("sha2", aarch64::SHA2, std_has!("sha2")),
      ("sha3", aarch64::SHA3, std_has!("sha3")),
      ("sha3 (SHA512)", aarch64::SHA512, std_has!("sha3")),
      ("sm4 (SM3)", aarch64::SM3, std_has!("sm4")),
      ("sm4", aarch64::SM4, std_has!("sm4")),
      ("crc", aarch64::CRC, std_has!("crc")),
      ("dotprod", aarch64::DOTPROD, std_has!("dotprod")),
      ("fp16", aarch64::FP16, std_has!("fp16")),
      ("i8mm", aarch64::I8MM, std_has!("i8mm")),
      ("bf16", aarch64::BF16, std_has!("bf16")),
      ("frintts", aarch64::FRINTTS, std_has!("frintts")),
      ("sve", aarch64::SVE, std_has!("sve")),
      ("sve2", aarch64::SVE2, std_has!("sve2")),
      ("lse", aarch64::LSE, std_has!("lse")),
      ("lse2", aarch64::LSE2, std_has!("lse2")),
      ("rand", aarch64::RNG, std_has!("rand")),
    ];
    for (feature, cap, std_detected) in expected {
      assert_eq!(owned.has(cap), std_detected, "{feature}");
    }
  }

  /// The owned DIT probe behind `DataIndependentTiming` must agree with the standard library.
  #[test]
  #[cfg(all(target_arch = "aarch64", feature = "std", not(miri)))]
  fn dit_probe_matches_std_detect() {
    assert_eq!(probe_dit(), std::arch::is_aarch64_feature_detected!("dit"));
    assert_eq!(dit_supported(), probe_dit());
  }

  /// A probed DIT state must be nonzero, or every guard would probe again, and it must record the
  /// probe's answer. Compile-time DIT targets never touch the cache.
  #[test]
  #[cfg(all(target_arch = "aarch64", feature = "std", not(miri)))]
  fn dit_cache_records_the_probe() {
    use core::sync::atomic::Ordering;

    let present = probe_dit();
    assert_eq!(dit_supported(), present);
    assert_eq!(dit_supported(), present);
    let state = DIT_STATE.load(Ordering::Relaxed);
    if cfg!(target_feature = "dit") {
      assert_eq!(state, 0, "a compile-time DIT target must not read or write the cache");
    } else {
      assert_eq!(state, DIT_KNOWN | if present { DIT_PRESENT } else { 0 });
    }
  }

  /// `getauxval` must return the kernel's `AT_HWCAP` and `AT_HWCAP2` entries, read here
  /// independently from `/proc/self/auxv`.
  #[test]
  #[cfg(all(
    feature = "std",
    not(miri),
    any(target_arch = "aarch64", target_arch = "powerpc64", target_arch = "s390x"),
    any(target_os = "linux", target_os = "android")
  ))]
  fn linux_hwcaps_match_proc_self_auxv() {
    let auxv = std::fs::read("/proc/self/auxv").expect("the kernel exposes /proc/self/auxv");
    let (mut hwcap, mut hwcap2) = (0u64, 0u64);
    for entry in auxv.as_chunks::<16>().0 {
      let (kind, value) = entry.split_at(8);
      let value = u64::from_ne_bytes(value.try_into().expect("8-byte value"));
      match u64::from_ne_bytes(kind.try_into().expect("8-byte type")) {
        0 => break,
        16 => hwcap = value,
        26 => hwcap2 = value,
        _ => {}
      }
    }
    assert_ne!(hwcap, 0, "the auxiliary vector has no AT_HWCAP entry");
    assert_eq!(linux_hwcaps(), (hwcap, hwcap2));
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
