//! Forced-kernel correctness tests for Argon2 BlaMka SIMD kernels.
//!
//! Every per-arch kernel under `src/auth/argon2/` must produce
//! bit-identical output to the portable kernel for both
//! `xor_into = false` (first pass) and `xor_into = true` (v1.3 pass > 0)
//! paths. These tests pin each kernel explicitly via the `diag` feature
//! hooks and cross-check against the portable oracle.
//!
//! The in-module test `auth::argon2::dispatch::tests::forced_kernels_match_portable`
//! covers kernel-level differential already; this file layers
//! end-to-end hash differentials on top — if a kernel diverges only on a
//! specific input distribution (e.g. carries in the BlaMka multiply), a
//! full hash with a non-trivial cost matrix surfaces it.

#![cfg(all(feature = "argon2", feature = "diag"))]
#![allow(clippy::unwrap_used)]

use rscrypto::{
  Argon2Params, Argon2Version,
  auth::{argon2, argon2::Argon2Variant},
};

fn params(m_kib: u32, t: u32, p: u32, out_len: u32) -> Argon2Params {
  Argon2Params::new()
    .memory_cost_kib(m_kib)
    .time_cost(t)
    .parallelism(p)
    .output_len(out_len)
    .version(Argon2Version::V0x13)
    .build()
    .unwrap()
}

const PASSWORD: &[u8] = b"correct horse battery staple";
const SALT: &[u8] = b"rscrypto-salt-16b!!!";

/// Every kernel must agree at a range of cost points.
fn all_kernels_agree(variant: Argon2Variant) {
  for &(m, t, p) in &[(16u32, 1u32, 1u32), (32, 2, 1), (64, 1, 2), (32, 3, 1)] {
    let params = params(m, t, p, 32);
    let mut expected = [0u8; 32];
    argon2::diag_hash_portable(&params, PASSWORD, SALT, variant, &mut expected).unwrap();

    let mut active_out = [0u8; 32];
    argon2::diag_hash_active(&params, PASSWORD, SALT, variant, &mut active_out).unwrap();
    assert_eq!(
      active_out, expected,
      "active kernel diverged on m={m} t={t} p={p} variant={variant:?}"
    );

    #[cfg(target_arch = "aarch64")]
    {
      let mut neon_out = [0u8; 32];
      argon2::diag_hash_aarch64_neon(&params, PASSWORD, SALT, variant, &mut neon_out).unwrap();
      assert_eq!(
        neon_out, expected,
        "aarch64-neon diverged on m={m} t={t} p={p} variant={variant:?}"
      );
    }

    #[cfg(target_arch = "x86_64")]
    {
      let host = rscrypto::platform::caps();

      if host.has(rscrypto::auth::argon2::required_caps(argon2::KernelId::X86Avx2)) {
        let mut out = [0u8; 32];
        argon2::diag_hash_x86_avx2(&params, PASSWORD, SALT, variant, &mut out).unwrap();
        assert_eq!(
          out, expected,
          "x86-avx2 diverged on m={m} t={t} p={p} variant={variant:?}"
        );
      }

      if host.has(rscrypto::auth::argon2::required_caps(argon2::KernelId::X86Avx512)) {
        let mut out = [0u8; 32];
        argon2::diag_hash_x86_avx512(&params, PASSWORD, SALT, variant, &mut out).unwrap();
        assert_eq!(
          out, expected,
          "x86-avx512 diverged on m={m} t={t} p={p} variant={variant:?}"
        );
      }
    }

    #[cfg(target_arch = "powerpc64")]
    if rscrypto::platform::caps().has(rscrypto::auth::argon2::required_caps(argon2::KernelId::PowerVsx)) {
      let mut out = [0u8; 32];
      argon2::diag_hash_power_vsx(&params, PASSWORD, SALT, variant, &mut out).unwrap();
      assert_eq!(
        out, expected,
        "power-vsx diverged on m={m} t={t} p={p} variant={variant:?}"
      );
    }

    #[cfg(target_arch = "s390x")]
    if rscrypto::platform::caps().has(rscrypto::auth::argon2::required_caps(argon2::KernelId::S390xVector)) {
      let mut out = [0u8; 32];
      argon2::diag_hash_s390x_vector(&params, PASSWORD, SALT, variant, &mut out).unwrap();
      assert_eq!(
        out, expected,
        "s390x-vector diverged on m={m} t={t} p={p} variant={variant:?}"
      );
    }

    #[cfg(target_arch = "riscv64")]
    if rscrypto::platform::caps().has(rscrypto::auth::argon2::required_caps(argon2::KernelId::Riscv64V)) {
      let mut out = [0u8; 32];
      argon2::diag_hash_riscv64_v(&params, PASSWORD, SALT, variant, &mut out).unwrap();
      assert_eq!(
        out, expected,
        "riscv64-v diverged on m={m} t={t} p={p} variant={variant:?}"
      );
    }

    #[cfg(target_arch = "wasm32")]
    if rscrypto::platform::caps().has(rscrypto::auth::argon2::required_caps(argon2::KernelId::WasmSimd128)) {
      let mut out = [0u8; 32];
      argon2::diag_hash_wasm_simd128(&params, PASSWORD, SALT, variant, &mut out).unwrap();
      assert_eq!(
        out, expected,
        "wasm-simd128 diverged on m={m} t={t} p={p} variant={variant:?}"
      );
    }
  }
}

#[test]
fn argon2d_all_kernels_agree() {
  all_kernels_agree(Argon2Variant::Argon2d);
}

#[test]
fn argon2i_all_kernels_agree() {
  all_kernels_agree(Argon2Variant::Argon2i);
}

#[test]
fn argon2id_all_kernels_agree() {
  all_kernels_agree(Argon2Variant::Argon2id);
}

/// Compress a single block via every kernel and assert bit-identical
/// output. Exercises both `xor_into` paths with pathological inputs.
#[test]
fn single_block_compress_matches_across_kernels() {
  use argon2::DIAG_BLOCK_WORDS;
  let x: [u64; DIAG_BLOCK_WORDS] = core::array::from_fn(|i| (i as u64).wrapping_mul(0x0f0f_0f0f_0f0f_0f0f));
  let y: [u64; DIAG_BLOCK_WORDS] =
    core::array::from_fn(|i| (i as u64).wrapping_mul(0xa5a5_a5a5_a5a5_a5a5).rotate_left(i as u32));

  for xor_into in [false, true] {
    let mut expected = if xor_into {
      [0xdead_beef_feed_faceu64; DIAG_BLOCK_WORDS]
    } else {
      [0u64; DIAG_BLOCK_WORDS]
    };
    argon2::diag_compress_portable(&mut expected, &x, &y, xor_into);

    #[cfg(target_arch = "aarch64")]
    {
      let mut neon = if xor_into {
        [0xdead_beef_feed_faceu64; DIAG_BLOCK_WORDS]
      } else {
        [0u64; DIAG_BLOCK_WORDS]
      };
      argon2::diag_compress_aarch64_neon(&mut neon, &x, &y, xor_into);
      assert_eq!(
        neon, expected,
        "aarch64-neon single-block diverged (xor_into = {xor_into})"
      );
    }

    #[cfg(target_arch = "x86_64")]
    {
      let host = rscrypto::platform::caps();

      if host.has(rscrypto::auth::argon2::required_caps(argon2::KernelId::X86Avx2)) {
        let mut out = if xor_into {
          [0xdead_beef_feed_faceu64; DIAG_BLOCK_WORDS]
        } else {
          [0u64; DIAG_BLOCK_WORDS]
        };
        argon2::diag_compress_x86_avx2(&mut out, &x, &y, xor_into);
        assert_eq!(out, expected, "x86-avx2 single-block diverged (xor_into = {xor_into})");
      }

      if host.has(rscrypto::auth::argon2::required_caps(argon2::KernelId::X86Avx512)) {
        let mut out = if xor_into {
          [0xdead_beef_feed_faceu64; DIAG_BLOCK_WORDS]
        } else {
          [0u64; DIAG_BLOCK_WORDS]
        };
        argon2::diag_compress_x86_avx512(&mut out, &x, &y, xor_into);
        assert_eq!(
          out, expected,
          "x86-avx512 single-block diverged (xor_into = {xor_into})"
        );
      }
    }

    #[cfg(target_arch = "powerpc64")]
    if rscrypto::platform::caps().has(rscrypto::auth::argon2::required_caps(argon2::KernelId::PowerVsx)) {
      let mut out = if xor_into {
        [0xdead_beef_feed_faceu64; DIAG_BLOCK_WORDS]
      } else {
        [0u64; DIAG_BLOCK_WORDS]
      };
      argon2::diag_compress_power_vsx(&mut out, &x, &y, xor_into);
      assert_eq!(out, expected, "power-vsx single-block diverged (xor_into = {xor_into})");
    }

    #[cfg(target_arch = "s390x")]
    if rscrypto::platform::caps().has(rscrypto::auth::argon2::required_caps(argon2::KernelId::S390xVector)) {
      let mut out = if xor_into {
        [0xdead_beef_feed_faceu64; DIAG_BLOCK_WORDS]
      } else {
        [0u64; DIAG_BLOCK_WORDS]
      };
      argon2::diag_compress_s390x_vector(&mut out, &x, &y, xor_into);
      assert_eq!(
        out, expected,
        "s390x-vector single-block diverged (xor_into = {xor_into})"
      );
    }

    #[cfg(target_arch = "riscv64")]
    if rscrypto::platform::caps().has(rscrypto::auth::argon2::required_caps(argon2::KernelId::Riscv64V)) {
      let mut out = if xor_into {
        [0xdead_beef_feed_faceu64; DIAG_BLOCK_WORDS]
      } else {
        [0u64; DIAG_BLOCK_WORDS]
      };
      argon2::diag_compress_riscv64_v(&mut out, &x, &y, xor_into);
      assert_eq!(out, expected, "riscv64-v single-block diverged (xor_into = {xor_into})");
    }

    #[cfg(target_arch = "wasm32")]
    if rscrypto::platform::caps().has(rscrypto::auth::argon2::required_caps(argon2::KernelId::WasmSimd128)) {
      let mut out = if xor_into {
        [0xdead_beef_feed_faceu64; DIAG_BLOCK_WORDS]
      } else {
        [0u64; DIAG_BLOCK_WORDS]
      };
      argon2::diag_compress_wasm_simd128(&mut out, &x, &y, xor_into);
      assert_eq!(
        out, expected,
        "wasm-simd128 single-block diverged (xor_into = {xor_into})"
      );
    }
  }
}

/// Active kernel on this host is the expected choice.
///
/// Skipped under `feature = "portable-only"` — the override forces
/// `caps()` to `Caps::NONE`, so dispatch picks `Portable` regardless of
/// host. That's the desired behaviour, and `test_caps_returns_none_with_portable_only_feature`
/// covers it. Re-asserting the per-arch SIMD selection here would
/// contradict the override.
#[test]
#[cfg(not(feature = "portable-only"))]
fn active_kernel_is_reported_correctly() {
  let active = argon2::diag_active_kernel();

  #[cfg(target_arch = "aarch64")]
  assert_eq!(
    active,
    argon2::KernelId::Aarch64Neon,
    "expected aarch64-neon to win dispatch"
  );

  #[cfg(target_arch = "x86_64")]
  {
    let host = rscrypto::platform::caps();
    let avx512_caps = rscrypto::auth::argon2::required_caps(argon2::KernelId::X86Avx512);
    let avx2_caps = rscrypto::auth::argon2::required_caps(argon2::KernelId::X86Avx2);

    if host.has(avx512_caps) {
      assert_eq!(active, argon2::KernelId::X86Avx512, "expected AVX-512 to win dispatch");
    } else if host.has(avx2_caps) {
      assert_eq!(active, argon2::KernelId::X86Avx2, "expected AVX2 to win dispatch");
    } else {
      assert_eq!(active, argon2::KernelId::Portable);
    }
  }

  #[cfg(target_arch = "powerpc64")]
  if rscrypto::platform::caps().has(rscrypto::auth::argon2::required_caps(argon2::KernelId::PowerVsx)) {
    assert_eq!(active, argon2::KernelId::PowerVsx, "expected POWER VSX to win dispatch");
  } else {
    assert_eq!(active, argon2::KernelId::Portable);
  }

  #[cfg(target_arch = "s390x")]
  if rscrypto::platform::caps().has(rscrypto::auth::argon2::required_caps(argon2::KernelId::S390xVector)) {
    assert_eq!(
      active,
      argon2::KernelId::S390xVector,
      "expected s390x z/Vector to win dispatch"
    );
  } else {
    assert_eq!(active, argon2::KernelId::Portable);
  }

  #[cfg(target_arch = "riscv64")]
  if rscrypto::platform::caps().has(rscrypto::auth::argon2::required_caps(argon2::KernelId::Riscv64V)) {
    assert_eq!(active, argon2::KernelId::Riscv64V, "expected RVV to win dispatch");
  } else {
    assert_eq!(active, argon2::KernelId::Portable);
  }

  #[cfg(target_arch = "wasm32")]
  if rscrypto::platform::caps().has(rscrypto::auth::argon2::required_caps(argon2::KernelId::WasmSimd128)) {
    assert_eq!(
      active,
      argon2::KernelId::WasmSimd128,
      "expected wasm simd128 to win dispatch"
    );
  } else {
    assert_eq!(active, argon2::KernelId::Portable);
  }

  #[cfg(not(any(
    target_arch = "aarch64",
    target_arch = "x86_64",
    target_arch = "powerpc64",
    target_arch = "s390x",
    target_arch = "riscv64",
    target_arch = "wasm32",
  )))]
  assert_eq!(active, argon2::KernelId::Portable);
}
