//! Argon2 BlaMka compression kernel dispatch.
//!
//! Exposes the [`KernelId`] enum and the runtime selector that picks the
//! highest-throughput kernel whose required caps are present on the host.
//!
//! The dispatcher is invoked once per [`super::argon2_hash`] call and the
//! resulting [`kernels::CompressFn`] pointer is threaded down through the
//! fill engine. A single function-pointer call per 1 KiB block keeps the
//! hot path branch-free across the 2·p·t × segment iterations.

use super::kernels::{self, CompressFn};
use crate::{
  backend::cache::OnceCache,
  platform::{Caps, caps},
};

/// Compression kernel identifier.
///
/// Every shipped kernel has a variant here plus a forced-kernel test that
/// pins it against the portable oracle. Variants are `#[cfg]`-gated by
/// target architecture: the enum width stays minimal on each target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum KernelId {
  /// Pure-Rust portable implementation. Always available.
  Portable,

  /// aarch64 NEON (4-way parallel BlaMka).
  #[cfg(target_arch = "aarch64")]
  Aarch64Neon,

  /// x86_64 AVX2 (4-way parallel BlaMka across YMM registers).
  #[cfg(target_arch = "x86_64")]
  X86Avx2,

  /// x86_64 AVX-512F + AVX-512VL (asymmetric ZMM/YMM design: 8-way
  /// 2-row batched row pass + 4-way YMM column pass with native VPRORQ).
  #[cfg(target_arch = "x86_64")]
  X86Avx512,

  /// powerpc64 VSX (4-way parallel BlaMka over `core::simd::u64x2`).
  #[cfg(target_arch = "powerpc64")]
  PowerVsx,

  /// s390x z/Vector facility (4-way parallel BlaMka with native
  /// `verllg` rotations).
  #[cfg(target_arch = "s390x")]
  S390xVector,

  /// riscv64 RVV (4-way parallel BlaMka via compiler auto-vectorisation
  /// at VL=2 / SEW=64).
  #[cfg(target_arch = "riscv64")]
  Riscv64V,

  /// wasm32 simd128 (4-way parallel BlaMka over `v128`).
  #[cfg(target_arch = "wasm32")]
  WasmSimd128,
}

impl KernelId {
  /// Kernel name for diagnostics and forced-kernel test plumbing.
  #[must_use]
  pub const fn as_str(self) -> &'static str {
    match self {
      Self::Portable => "portable",
      #[cfg(target_arch = "aarch64")]
      Self::Aarch64Neon => "aarch64-neon",
      #[cfg(target_arch = "x86_64")]
      Self::X86Avx2 => "x86-avx2",
      #[cfg(target_arch = "x86_64")]
      Self::X86Avx512 => "x86-avx512",
      #[cfg(target_arch = "powerpc64")]
      Self::PowerVsx => "power-vsx",
      #[cfg(target_arch = "s390x")]
      Self::S390xVector => "s390x-vector",
      #[cfg(target_arch = "riscv64")]
      Self::Riscv64V => "riscv64-v",
      #[cfg(target_arch = "wasm32")]
      Self::WasmSimd128 => "wasm-simd128",
    }
  }
}

/// All compiled-in kernels, ordered best-first.
///
/// The dispatcher walks this slice and picks the first entry whose
/// [`required_caps`] is a subset of the host's [`crate::platform::caps()`].
/// Portable is always the last entry — it has empty caps, so it always
/// matches.
pub const ALL_KERNELS: &[KernelId] = &[
  #[cfg(target_arch = "x86_64")]
  KernelId::X86Avx512,
  #[cfg(target_arch = "x86_64")]
  KernelId::X86Avx2,
  #[cfg(target_arch = "aarch64")]
  KernelId::Aarch64Neon,
  #[cfg(target_arch = "powerpc64")]
  KernelId::PowerVsx,
  #[cfg(target_arch = "s390x")]
  KernelId::S390xVector,
  #[cfg(target_arch = "riscv64")]
  KernelId::Riscv64V,
  #[cfg(target_arch = "wasm32")]
  KernelId::WasmSimd128,
  KernelId::Portable,
];

/// Capabilities required for `kernel` to be callable on the current host.
#[must_use]
pub const fn required_caps(kernel: KernelId) -> Caps {
  match kernel {
    KernelId::Portable => Caps::from_words([0; 4]),
    #[cfg(target_arch = "aarch64")]
    // NEON is baseline on aarch64 — the cap is always present, but we
    // encode it explicitly so the dispatcher treats all kernels uniformly.
    KernelId::Aarch64Neon => crate::platform::caps::aarch64::NEON,
    #[cfg(target_arch = "x86_64")]
    KernelId::X86Avx2 => crate::platform::caps::x86::AVX2,
    #[cfg(target_arch = "x86_64")]
    // AVX-512F powers the ZMM-wide row pass; AVX-512VL gives the YMM-form
    // VPRORQ used by the 4-way column pass. Both are required.
    KernelId::X86Avx512 => crate::platform::caps::x86::AVX512F.union(crate::platform::caps::x86::AVX512VL),
    #[cfg(target_arch = "powerpc64")]
    KernelId::PowerVsx => crate::platform::caps::power::VSX,
    #[cfg(target_arch = "s390x")]
    KernelId::S390xVector => crate::platform::caps::s390x::VECTOR,
    #[cfg(target_arch = "riscv64")]
    KernelId::Riscv64V => crate::platform::caps::riscv::V,
    #[cfg(target_arch = "wasm32")]
    KernelId::WasmSimd128 => crate::platform::caps::wasm::SIMD128,
  }
}

/// Return the compression function for `kernel`.
///
/// Does **not** check `required_caps`; the caller must have verified the
/// host supports the kernel (the dispatcher does this, and the forced-
/// kernel tests do it explicitly).
#[inline]
#[must_use]
pub(super) fn compress_fn_for(kernel: KernelId) -> CompressFn {
  match kernel {
    KernelId::Portable => kernels::compress_portable,
    #[cfg(target_arch = "aarch64")]
    KernelId::Aarch64Neon => super::aarch64::compress_neon,
    #[cfg(target_arch = "x86_64")]
    KernelId::X86Avx2 => super::x86_64::compress_avx2,
    #[cfg(target_arch = "x86_64")]
    KernelId::X86Avx512 => super::x86_64::compress_avx512,
    #[cfg(target_arch = "powerpc64")]
    KernelId::PowerVsx => super::power::compress_vsx,
    #[cfg(target_arch = "s390x")]
    KernelId::S390xVector => super::s390x::compress_vector,
    #[cfg(target_arch = "riscv64")]
    KernelId::Riscv64V => super::riscv64::compress_rvv,
    #[cfg(target_arch = "wasm32")]
    KernelId::WasmSimd128 => super::wasm::compress_simd128,
  }
}

/// Cached active kernel id. The first call to [`active_kernel`] walks
/// `ALL_KERNELS` against [`crate::platform::caps()`]; every subsequent call
/// returns the cached result directly. Mirrors the CRC64 / Blake2 / Keccak
/// dispatch pattern; saves a linear cap-set walk on every Argon2 hash.
static ACTIVE_KERNEL: OnceCache<KernelId> = OnceCache::new();

/// Select the best kernel available on the current host.
///
/// First call resolves; subsequent calls hit [`ACTIVE_KERNEL`] and return
/// without re-walking `ALL_KERNELS`. The cache value is `KernelId` (Copy),
/// so the cached read is a single atomic-acquire load.
///
/// # Gate: macOS aarch64
///
/// Single-block Blake2b NEON loses to portable scalar on Apple Silicon
/// because the 2-u64 NEON width cannot match the wide OoO scalar core
/// on one 128-byte block. Argon2 BlaMka is a *different* shape — each
/// compression is 1024 bytes with 16 P-rounds of 4 independent GBs, so
/// a 4-way SIMD kernel has real parallelism to extract even on M-series.
/// The gate on this primitive is measured, not inherited: the NEON kernel
/// is the active one on all aarch64 targets unless a future measurement
/// flips the polarity for a specific host.
///
/// # Gate: x86_64
///
/// AVX-512 sits ahead of AVX2 in `ALL_KERNELS`. The dispatcher only picks
/// it when the host has both AVX-512F and AVX-512VL caps — see
/// [`required_caps`]. If `AVX-512` is unavailable, AVX2 is the fallback;
/// only when neither is present does the dispatcher fall through to the
/// portable kernel.
#[must_use]
pub(super) fn active_kernel() -> KernelId {
  ACTIVE_KERNEL.get_or_init(|| {
    let host = caps();
    for &id in ALL_KERNELS {
      if host.has(required_caps(id)) {
        return id;
      }
    }
    KernelId::Portable
  })
}

/// Resolve the active [`CompressFn`] for this host. Threaded through the
/// fill engine and called per Argon2 hash; backed by [`ACTIVE_KERNEL`] so the
/// per-call cost after the first resolve is one cache load + one match arm
/// (no linear walk over `ALL_KERNELS`, no caps-bitset comparison).
#[inline]
pub(super) fn active_compress() -> CompressFn {
  compress_fn_for(active_kernel())
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn portable_has_no_required_caps() {
    assert!(required_caps(KernelId::Portable).is_empty());
  }

  #[test]
  fn all_kernels_terminate_on_portable() {
    assert_eq!(*ALL_KERNELS.last().unwrap(), KernelId::Portable);
  }

  #[test]
  fn portable_kernel_name() {
    assert_eq!(KernelId::Portable.as_str(), "portable");
  }

  #[cfg(target_arch = "aarch64")]
  #[test]
  fn aarch64_neon_kernel_name() {
    assert_eq!(KernelId::Aarch64Neon.as_str(), "aarch64-neon");
  }

  #[cfg(target_arch = "aarch64")]
  #[test]
  fn aarch64_neon_required_caps_are_neon() {
    assert_eq!(
      required_caps(KernelId::Aarch64Neon),
      crate::platform::caps::aarch64::NEON
    );
  }

  #[cfg(target_arch = "x86_64")]
  #[test]
  fn x86_avx2_kernel_name() {
    assert_eq!(KernelId::X86Avx2.as_str(), "x86-avx2");
  }

  #[cfg(target_arch = "x86_64")]
  #[test]
  fn x86_avx2_required_caps_are_avx2() {
    assert_eq!(required_caps(KernelId::X86Avx2), crate::platform::caps::x86::AVX2);
  }

  #[cfg(target_arch = "x86_64")]
  #[test]
  fn x86_avx512_kernel_name() {
    assert_eq!(KernelId::X86Avx512.as_str(), "x86-avx512");
  }

  #[cfg(target_arch = "x86_64")]
  #[test]
  fn x86_avx512_required_caps_are_f_and_vl() {
    let caps = required_caps(KernelId::X86Avx512);
    assert!(caps.has(crate::platform::caps::x86::AVX512F));
    assert!(caps.has(crate::platform::caps::x86::AVX512VL));
  }

  #[cfg(target_arch = "x86_64")]
  #[test]
  fn x86_kernels_ordered_avx512_then_avx2_then_portable() {
    // The dispatcher walks ALL_KERNELS in order and picks the first whose
    // caps are present on the host. AVX-512 must precede AVX2 must precede
    // portable so the highest-throughput available kernel wins.
    let avx512_pos = ALL_KERNELS
      .iter()
      .position(|&k| k == KernelId::X86Avx512)
      .expect("avx-512 in table");
    let avx2_pos = ALL_KERNELS
      .iter()
      .position(|&k| k == KernelId::X86Avx2)
      .expect("avx2 in table");
    let portable_pos = ALL_KERNELS
      .iter()
      .position(|&k| k == KernelId::Portable)
      .expect("portable in table");
    assert!(avx512_pos < avx2_pos);
    assert!(avx2_pos < portable_pos);
  }

  #[cfg(target_arch = "powerpc64")]
  #[test]
  fn power_vsx_kernel_name_and_caps() {
    assert_eq!(KernelId::PowerVsx.as_str(), "power-vsx");
    assert_eq!(required_caps(KernelId::PowerVsx), crate::platform::caps::power::VSX);
  }

  #[cfg(target_arch = "s390x")]
  #[test]
  fn s390x_vector_kernel_name_and_caps() {
    assert_eq!(KernelId::S390xVector.as_str(), "s390x-vector");
    assert_eq!(
      required_caps(KernelId::S390xVector),
      crate::platform::caps::s390x::VECTOR
    );
  }

  #[cfg(target_arch = "riscv64")]
  #[test]
  fn riscv64_v_kernel_name_and_caps() {
    assert_eq!(KernelId::Riscv64V.as_str(), "riscv64-v");
    assert_eq!(required_caps(KernelId::Riscv64V), crate::platform::caps::riscv::V);
  }

  #[cfg(target_arch = "wasm32")]
  #[test]
  fn wasm_simd128_kernel_name_and_caps() {
    assert_eq!(KernelId::WasmSimd128.as_str(), "wasm-simd128");
    assert_eq!(
      required_caps(KernelId::WasmSimd128),
      crate::platform::caps::wasm::SIMD128
    );
  }

  #[test]
  fn active_kernel_is_in_all_kernels() {
    let id = active_kernel();
    assert!(ALL_KERNELS.contains(&id));
  }

  #[test]
  fn forced_kernels_match_portable() {
    // Correctness-spot-check: every kernel applied to the same input must
    // produce the same output. The full RFC 9106 Appendix A vectors run
    // in tests/argon2_vectors.rs against the active kernel; this test is
    // the per-kernel differential.
    let x: [u64; super::super::BLOCK_WORDS] = core::array::from_fn(|i| i as u64 * 0x0101_0101_0101_0101);
    let y: [u64; super::super::BLOCK_WORDS] = core::array::from_fn(|i| (i as u64).wrapping_mul(0xdead_beef_feed_face));
    let mut expected = [0u64; super::super::BLOCK_WORDS];
    // SAFETY: portable kernel has no preconditions.
    unsafe { kernels::compress_portable(&mut expected, &x, &y, false) };

    for &id in ALL_KERNELS {
      let host = crate::platform::caps();
      if !host.has(required_caps(id)) {
        continue;
      }
      let kernel = compress_fn_for(id);
      let mut got = [0u64; super::super::BLOCK_WORDS];
      // SAFETY: cap check above confirms the kernel's required_caps
      // are present on the host.
      unsafe { kernel(&mut got, &x, &y, false) };
      assert_eq!(got, expected, "kernel {} diverged from portable", id.as_str());

      // XOR-into variant.
      let mut acc = [0xa5a5_a5a5_a5a5_a5a5u64; super::super::BLOCK_WORDS];
      let mut expected_xor = acc;
      // SAFETY: kernel caps already validated; portable has no preconditions.
      unsafe {
        kernels::compress_portable(&mut expected_xor, &x, &y, true);
        kernel(&mut acc, &x, &y, true);
      }
      assert_eq!(
        acc,
        expected_xor,
        "kernel {} xor_into diverged from portable",
        id.as_str()
      );
    }
  }
}
