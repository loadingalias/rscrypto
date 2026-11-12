//! Adler-32 from RFC 1950 (**NOT CRYPTO**).
//!
//! This module provides [`Adler32`], the checksum of zlib streams.
//!
//! # Kernels
//!
//! - Portable Rust on every target, the sole semantic authority.
//! - aarch64: NEON, and NEON with the dot-product extension where the CPU has it.
//! - x86-64: SSSE3, AVX2, and AVX-512 VNNI, the fastest that the CPU has.
//! - POWER: AltiVec `vmsumubm`. IBM Z: the vector facility, where the CPU has it.
//! - riscv64: RVV where the CPU has the V extension.
//! - wasm32: SIMD128 when the artifact requires it.

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub(in crate::checksum) mod aarch64;
#[cfg(any(
  all(target_arch = "x86_64", target_feature = "sse2"),
  all(target_arch = "powerpc64", not(feature = "portable-only")),
  all(target_arch = "s390x", not(feature = "portable-only"))
))]
mod blocks;
pub(crate) mod config;
pub(in crate::checksum) mod portable;
#[cfg(all(target_arch = "powerpc64", not(feature = "portable-only")))]
pub(in crate::checksum) mod power;
#[cfg(all(target_arch = "riscv64", not(feature = "portable-only")))]
pub(in crate::checksum) mod riscv64;
#[cfg(all(target_arch = "s390x", not(feature = "portable-only")))]
pub(in crate::checksum) mod s390x;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128", not(feature = "portable-only")))]
pub(in crate::checksum) mod wasm32;
#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
pub(in crate::checksum) mod x86_64;

pub use config::{Adler32Config, Adler32Force};
use portable::{MOD, MOD64};

use crate::traits::{Checksum, ChecksumCombine};

// Dispatch

type Adler32DispatchFn = crate::checksum::dispatchers::Adler32Fn;

/// One Adler-32 kernel: its entry point and its canonical name.
#[derive(Clone, Copy)]
struct Kernel {
  update: Adler32DispatchFn,
  name: &'static str,
}

const PORTABLE: Kernel = Kernel {
  update: portable::update,
  name: portable::KERNEL_NAME,
};

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
const NEON: Kernel = Kernel {
  update: aarch64::update_safe,
  name: aarch64::KERNEL_NAME,
};

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
const NEON_DOTPROD: Kernel = Kernel {
  update: adler32_neon_dotprod,
  name: aarch64::DOTPROD_KERNEL_NAME,
};

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
fn adler32_neon_dotprod(adler: u32, data: &[u8]) -> u32 {
  // SAFETY: NEON is enabled at compile time. Only `select` returns this function, and only when
  // cached detection reports the dot-product extension: automatic selection checks it, and
  // `config::get()` keeps `Adler32Force::NeonDotprod` only then.
  unsafe { aarch64::update_dotprod(adler, data) }
}

#[cfg(all(target_arch = "powerpc64", not(feature = "portable-only")))]
const VMSUM: Kernel = Kernel {
  update: power::update_safe,
  name: power::KERNEL_NAME,
};

#[cfg(all(target_arch = "s390x", not(feature = "portable-only")))]
const VSUM: Kernel = Kernel {
  update: s390x::update_safe,
  name: s390x::KERNEL_NAME,
};

#[cfg(all(target_arch = "riscv64", not(feature = "portable-only")))]
const RVV: Kernel = Kernel {
  update: adler32_rvv,
  name: riscv64::KERNEL_NAME,
};

#[cfg(all(target_arch = "riscv64", not(feature = "portable-only")))]
fn adler32_rvv(adler: u32, data: &[u8]) -> u32 {
  // SAFETY: Only `select` returns this function, and only when cached detection reports the V
  // extension: automatic selection checks it, and `config::get()` keeps `Adler32Force::Rvv`
  // only then.
  unsafe { riscv64::update(adler, data) }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128", not(feature = "portable-only")))]
const WASM_SIMD128: Kernel = Kernel {
  update: wasm32::update,
  name: wasm32::KERNEL_NAME,
};

/// The kernel for `force`, which comes from `config::get()`: a forced kernel survives it only
/// when cached detection reports every capability that the kernel needs.
///
/// This is the only kernel selection of Adler-32. [`kernel`] caches its result, and updates and
/// kernel names both read that cache, so a reported name is the kernel that runs.
fn select(force: Adler32Force) -> Kernel {
  #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
  if let Some(kernel) = x86_64::KERNELS.iter().find(|kernel| kernel.force == force) {
    return Kernel {
      update: kernel.update,
      name: kernel.name,
    };
  }
  match force {
    Adler32Force::Portable => PORTABLE,
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    Adler32Force::Neon => NEON,
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    Adler32Force::NeonDotprod => NEON_DOTPROD,
    #[cfg(all(target_arch = "powerpc64", not(feature = "portable-only")))]
    Adler32Force::Vmsum => VMSUM,
    #[cfg(all(target_arch = "s390x", not(feature = "portable-only")))]
    Adler32Force::Vsum => VSUM,
    #[cfg(all(target_arch = "riscv64", not(feature = "portable-only")))]
    Adler32Force::Rvv => RVV,
    _ => select_auto(),
  }
}

/// Automatic selection: the x86-64 kernel that cached detection allows, the aarch64 dot-product
/// kernel when detection reports `FEAT_DotProd` and the NEON kernel when it reports NEON, the
/// POWER kernel when detection reports AltiVec, the IBM Z kernel when detection reports the
/// vector facility, the riscv64 RVV kernel when detection reports V, the wasm32 SIMD128 kernel
/// when the artifact requires SIMD128, and otherwise the portable kernel. `portable-only`, Miri,
/// and a portable detection override report no capability, so they select the portable kernel.
/// Every Adler-32 kernel handles every length, so one kernel serves every call.
fn select_auto() -> Kernel {
  let caps = crate::platform::caps();
  #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
  if let Some(kernel) = x86_64::select(caps) {
    return Kernel {
      update: kernel.update,
      name: kernel.name,
    };
  }
  #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
  if caps.has(crate::platform::caps::aarch64::DOTPROD) {
    return NEON_DOTPROD;
  }
  #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
  if caps.has(crate::platform::caps::aarch64::NEON) {
    return NEON;
  }
  #[cfg(all(target_arch = "powerpc64", not(feature = "portable-only")))]
  if caps.has(crate::platform::caps::power::ALTIVEC) {
    return VMSUM;
  }
  #[cfg(all(target_arch = "s390x", not(feature = "portable-only")))]
  if caps.has(crate::platform::caps::s390x::VECTOR) {
    return VSUM;
  }
  #[cfg(all(target_arch = "riscv64", not(feature = "portable-only")))]
  if caps.has(crate::platform::caps::riscv::V) {
    return RVV;
  }
  #[cfg(all(target_arch = "wasm32", target_feature = "simd128", not(feature = "portable-only")))]
  if caps.has(crate::platform::caps::wasm::SIMD128) {
    return WASM_SIMD128;
  }
  let _ = caps;
  PORTABLE
}

/// The kernel of this process, selected once for the effective force.
#[inline]
fn kernel() -> Kernel {
  static KERNEL: crate::backend::cache::OnceCache<Kernel> = crate::backend::cache::OnceCache::new();
  KERNEL.get_or_init(|| select(config::get().effective_force))
}

/// Adler-32 through [`kernel`], for a hasher that `const` construction cannot resolve.
fn adler32_dispatch(adler: u32, data: &[u8]) -> u32 {
  (kernel().update)(adler, data)
}

/// Adler-32 checksum from [RFC 1950](https://www.rfc-editor.org/rfc/rfc1950.html) (**NOT CRYPTO**).
///
/// zlib streams store Adler-32 big-endian: the checksum of the uncompressed data
/// after the compressed data, and the checksum of a preset dictionary in the
/// header. It detects accidental corruption only and gives no authentication.
///
/// [`Checksum::with_initial`] resumes from the checksum of the data so far, and
/// [`ChecksumCombine::combine`] joins the checksums of two parts, as zlib's
/// `adler32` and `adler32_combine` do. The state is two 16-bit sums and keeps
/// no input.
///
/// # Examples
///
/// ```rust
/// use rscrypto::checksum::{Adler32, Checksum, ChecksumCombine};
///
/// assert_eq!(Adler32::checksum(b"Wikipedia"), 0x11E6_0398);
///
/// let (a, b) = b"Wikipedia".split_at(4);
/// let combined = Adler32::combine(Adler32::checksum(a), Adler32::checksum(b), b.len());
/// assert_eq!(combined, 0x11E6_0398);
/// ```
#[derive(Clone, Copy)]
pub struct Adler32 {
  state: u32,
  dispatch: Adler32DispatchFn,
}

impl Adler32 {
  const INIT: u32 = 1;

  /// Create a hasher to resume from a previous Adler-32 value.
  #[inline]
  #[must_use]
  pub const fn resume(adler: u32) -> Self {
    Self {
      state: (((adler >> 16) % MOD) << 16) | ((adler & 0xffff) % MOD),
      // `resume` is const, so the kernel resolves at the first update.
      dispatch: adler32_dispatch,
    }
  }

  /// Get the effective Adler-32 configuration.
  #[must_use]
  pub fn config() -> Adler32Config {
    config::get()
  }

  /// Returns the kernel name that the selector would choose for `len`.
  ///
  /// Every length runs the same kernel.
  #[must_use]
  pub fn kernel_name_for_len(_len: usize) -> &'static str {
    kernel().name
  }
}

impl Checksum for Adler32 {
  const OUTPUT_SIZE: usize = 4;
  type Output = u32;

  #[inline]
  fn new() -> Self {
    Self {
      state: Self::INIT,
      dispatch: kernel().update,
    }
  }

  #[inline]
  fn with_initial(initial: u32) -> Self {
    let (a, b) = sums(initial);
    Self {
      state: (b << 16) | a,
      dispatch: kernel().update,
    }
  }

  #[inline]
  fn update(&mut self, data: &[u8]) {
    self.state = (self.dispatch)(self.state, data);
  }

  #[inline]
  fn update_vectored(&mut self, bufs: &[&[u8]]) {
    for &buf in bufs {
      if !buf.is_empty() {
        self.state = (self.dispatch)(self.state, buf);
      }
    }
  }

  #[inline]
  fn finalize(&self) -> u32 {
    self.state
  }

  #[inline]
  fn reset(&mut self) {
    self.state = Self::INIT;
  }
}

impl core::fmt::Debug for Adler32 {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("Adler32").finish_non_exhaustive()
  }
}

impl Default for Adler32 {
  #[inline]
  fn default() -> Self {
    <Self as Checksum>::new()
  }
}

impl ChecksumCombine for Adler32 {
  /// Computes zlib's `adler32_combine`: over `A || B`, `a = a(A) + a(B) - 1`
  /// and `b = b(A) + b(B) + len(B)·(a(A) - 1)`, both modulo 65521. Only
  /// `len(B)` modulo 65521 matters, so every `usize` length is exact.
  fn combine(crc_a: u32, crc_b: u32, len_b: usize) -> u32 {
    let (a1, b1) = sums(crc_a);
    let (a2, b2) = sums(crc_b);
    let (a1, b1, a2, b2) = (u64::from(a1), u64::from(b1), u64::from(a2), u64::from(b2));
    let len = (len_b as u64) % MOD64;
    let a = portable::reduce(a1.strict_add(a2).strict_add(MOD64 - 1));
    let b = portable::reduce(
      b1.strict_add(b2)
        .strict_add(len.strict_mul(a1))
        .strict_add(MOD64.strict_sub(len)),
    );
    (b << 16) | a
  }
}

/// Split a checksum into its reduced sums `a` (low half) and `b` (high half).
#[inline]
fn sums(value: u32) -> (u32, u32) {
  ((value & 0xffff) % MOD, (value >> 16) % MOD)
}

// Buffered Wrapper

// Buffered wrappers use this to decide when to flush/process in larger chunks.
//
// Each threshold is the shortest update that reaches the selected kernel's block loop, so the
// block setup and the reduction at its end pay back. They come from the kernels' structure:
// local timing on a loaded host could not confirm the crossover, which needs a quiet host.
#[cfg(feature = "alloc")]
#[inline]
#[must_use]
fn adler32_buffered_threshold() -> usize {
  adler32_buffered_threshold_impl()
}

/// The NEON and SIMD128 kernels run 64-byte blocks; shorter updates take 16-byte vectors and
/// the byte tail.
#[cfg(all(
  feature = "alloc",
  any(
    all(target_arch = "aarch64", target_feature = "neon"),
    all(target_arch = "wasm32", target_feature = "simd128", not(feature = "portable-only"))
  )
))]
#[inline]
#[must_use]
fn adler32_buffered_threshold_impl() -> usize {
  const THRESHOLD: usize = 64;
  THRESHOLD
}

/// The portable kernel runs 16-byte steps; shorter updates take the byte tail.
#[cfg(all(
  feature = "alloc",
  not(any(
    all(target_arch = "aarch64", target_feature = "neon"),
    all(target_arch = "wasm32", target_feature = "simd128", not(feature = "portable-only"))
  ))
))]
#[inline]
#[must_use]
fn adler32_buffered_threshold_impl() -> usize {
  const THRESHOLD: usize = 16;
  THRESHOLD
}

/// Large enough to clear any warmup threshold, small enough to stay cache-friendly.
#[cfg(feature = "alloc")]
const BUFFERED_ADLER32_BUFFER_SIZE: usize = 512;

#[cfg(feature = "alloc")]
define_buffered_crc! {
  /// A buffering wrapper around [`Adler32`] for streaming small chunks.
  pub struct BufferedAdler32<Adler32> {
    buffer_size: BUFFERED_ADLER32_BUFFER_SIZE,
    threshold_fn: adler32_buffered_threshold,
  }
}

#[cfg(feature = "alloc")]
impl Adler32 {
  /// Creates a buffering wrapper that coalesces short updates before Adler-32 dispatch.
  #[must_use]
  pub fn buffered() -> BufferedAdler32 {
    BufferedAdler32::new()
  }
}

impl_std_io_write_for_checksum!(Adler32);

// Kernel Introspection

#[cfg(feature = "diag")]
impl crate::checksum::introspect::KernelIntrospect for Adler32 {
  fn kernel_name_for_len(len: usize) -> &'static str {
    Self::kernel_name_for_len(len)
  }
}

/// Automatic selection runs, and names, the fastest kernel that the host's capabilities allow.
/// The list below states that policy apart from `select_auto`, so on its native host a reported
/// capability that does not select its kernel fails here. The kernel must also match the
/// portable kernel through the public API.
#[cfg(all(test, not(miri)))]
#[test]
fn automatic_selection_runs_the_kernel_that_the_capabilities_call_for() {
  use crate::platform::Caps;

  if config::get().effective_force != Adler32Force::Auto {
    return;
  }
  let policy: &[(Caps, &str)] = &[
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    (
      crate::platform::caps::x86::AVX512F
        .union(crate::platform::caps::x86::AVX512BW)
        .union(crate::platform::caps::x86::AVX512VNNI),
      "x86_64/avx512-vnni",
    ),
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    (crate::platform::caps::x86::AVX2, "x86_64/avx2"),
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    (crate::platform::caps::x86::SSSE3, "x86_64/ssse3"),
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    (crate::platform::caps::aarch64::DOTPROD, "aarch64/neon-dotprod"),
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    (crate::platform::caps::aarch64::NEON, "aarch64/neon"),
    #[cfg(all(target_arch = "powerpc64", not(feature = "portable-only")))]
    (crate::platform::caps::power::ALTIVEC, "power/vmsum"),
    #[cfg(all(target_arch = "s390x", not(feature = "portable-only")))]
    (crate::platform::caps::s390x::VECTOR, "s390x/vsum"),
    #[cfg(all(target_arch = "riscv64", not(feature = "portable-only")))]
    (crate::platform::caps::riscv::V, "riscv64/v"),
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128", not(feature = "portable-only")))]
    (crate::platform::caps::wasm::SIMD128, "wasm32/simd128"),
  ];
  let host = crate::platform::caps();
  let expected = policy
    .iter()
    .find(|&&(needs, _)| host.has(needs))
    .map_or("portable/unroll16", |&(_, name)| name);
  assert_eq!(Adler32::kernel_name_for_len(0), expected);

  let data: alloc::vec::Vec<u8> = (0..70_000u32)
    .map(|i| (i.wrapping_mul(131) >> 3).to_le_bytes()[0])
    .collect();
  for len in [0, 1, 15, 16, 63, 64, 65, 255, 256, 5553, 16_385, 70_000] {
    let input = &data[..len];
    assert_eq!(
      Adler32::checksum(input),
      portable::update(1, input),
      "{expected}: len={len}"
    );
  }
}

#[cfg(all(
  test,
  not(miri),
  any(
    all(target_arch = "aarch64", target_feature = "neon"),
    all(target_arch = "riscv64", not(feature = "portable-only"))
  )
))]
mod tests {
  use alloc::{vec, vec::Vec};

  #[cfg(target_arch = "aarch64")]
  use super::aarch64;
  use super::portable;
  #[cfg(target_arch = "riscv64")]
  use super::riscv64;
  use crate::checksum::dispatchers::Adler32Fn;

  /// The aarch64 kernels that this CPU can run, with their names.
  #[cfg(target_arch = "aarch64")]
  fn kernels() -> Vec<(&'static str, Adler32Fn)> {
    let mut kernels: Vec<(&'static str, Adler32Fn)> = vec![(aarch64::KERNEL_NAME, aarch64::update_safe)];
    if crate::platform::caps().has(crate::platform::caps::aarch64::DOTPROD) {
      kernels.push((aarch64::DOTPROD_KERNEL_NAME, |adler, data| {
        // SAFETY: NEON is enabled at compile time, and the capability check above establishes
        // the dot-product extension.
        unsafe { aarch64::update_dotprod(adler, data) }
      }));
    }
    kernels
  }

  /// The riscv64 kernel, with its name, when the CPU has the V extension. A host without V
  /// tests none here; the selection policy test proves that a host with V runs it.
  #[cfg(target_arch = "riscv64")]
  fn kernels() -> Vec<(&'static str, Adler32Fn)> {
    if !crate::platform::caps().has(crate::platform::caps::riscv::V) {
      return Vec::new();
    }
    vec![(riscv64::KERNEL_NAME, |adler, data| {
      // SAFETY: The capability check above establishes the V extension.
      unsafe { riscv64::update(adler, data) }
    })]
  }

  /// Lengths that end on and around every block, vector, and tail size of the kernels, the
  /// 5,552-byte portable and the 16 KiB NEON reduction lengths, the RVV reduction lengths of
  /// 1,024 rows of 16 to 256 bytes, and 8 KiB, a few times over.
  fn lengths() -> Vec<usize> {
    let mut lengths: Vec<usize> = (0..=640).collect();
    for base in [5552usize, 8192, 16_384, 131_072] {
      for multiple in 1..=3 {
        let at = base.strict_mul(multiple);
        lengths.extend([
          at.strict_sub(65),
          at.strict_sub(1),
          at,
          at.strict_add(1),
          at.strict_add(15),
          at.strict_add(63),
        ]);
      }
    }
    lengths.sort_unstable();
    lengths.dedup();
    lengths
  }

  fn patterned(len: usize) -> Vec<u8> {
    (0..len)
      .map(|i| i.wrapping_mul(131).wrapping_add(i >> 9).to_le_bytes()[0])
      .collect()
  }

  /// Starting values with reduced sums near 0 and near 65520.
  const STARTS: [u32; 3] = [1, 0xfff0_fff0, 0x0001_fff0];

  /// All-0xff input drives every lane sum to its bound; all-zero input keeps every sum at its
  /// starting value.
  #[test]
  fn kernels_match_portable_for_every_length_alignment_and_start() {
    let longest = *lengths().last().expect("lengths are not empty");
    let all: Vec<usize> = (0..64).collect();
    for fill in [None, Some(0xff), Some(0)] {
      let buffer = match fill {
        Some(byte) => vec![byte; longest + 64],
        None => patterned(longest + 64),
      };
      for len in lengths() {
        // Every alignment for the short lengths; a spread for the long ones keeps the test fast.
        let offsets: &[usize] = if len <= 640 { &all } else { &[0, 1, 31, 63] };
        for &offset in offsets {
          let data = &buffer[offset..offset + len];
          for start in STARTS {
            let expected = portable::update(start, data);
            for (name, kernel) in kernels() {
              assert_eq!(
                kernel(start, data),
                expected,
                "{name}: fill={fill:?}, len={len}, offset={offset}, start={start:#x}"
              );
            }
          }
        }
      }
    }
  }

  /// Writes that straddle the 16 KiB NEON reduction length or span several reductions.
  #[test]
  fn chunked_kernel_writes_match_portable_one_shot() {
    for fill in [Some(0xff), None] {
      let data = match fill {
        Some(byte) => vec![byte; 3 * 131_072 + 17],
        None => patterned(3 * 131_072 + 17),
      };
      let expected = portable::update(1, &data);
      for (name, kernel) in kernels() {
        for chunk in [1, 63, 64, 65, 16_383, 16_385, 131_071, 131_073] {
          let streamed = data.chunks(chunk).fold(1, kernel);
          assert_eq!(streamed, expected, "{name}: fill={fill:?}, chunk={chunk}");
        }
      }
    }
  }
}
