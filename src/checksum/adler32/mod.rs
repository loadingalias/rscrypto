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
#[cfg(feature = "std")]
type Adler32DispatchVectoredFn = fn(u32, &[&[u8]]) -> u32;

#[inline]
fn adler32_apply_kernel_vectored(mut adler: u32, bufs: &[&[u8]], kernel: Adler32DispatchFn) -> u32 {
  for &buf in bufs {
    if !buf.is_empty() {
      adler = kernel(adler, buf);
    }
  }
  adler
}

#[inline]
fn adler32_dispatch_auto(adler: u32, data: &[u8]) -> u32 {
  crate::checksum::kernel_table::adler32(crate::checksum::kernel_table::active_table(), adler, data)
}

#[inline]
fn adler32_dispatch_auto_vectored(adler: u32, bufs: &[&[u8]]) -> u32 {
  adler32_apply_kernel_vectored(adler, bufs, adler32_dispatch_auto)
}

#[cfg(feature = "std")]
#[inline]
fn adler32_dispatch_portable(adler: u32, data: &[u8]) -> u32 {
  portable::update(adler, data)
}

#[cfg(feature = "std")]
#[inline]
fn adler32_dispatch_portable_vectored(adler: u32, bufs: &[&[u8]]) -> u32 {
  adler32_apply_kernel_vectored(adler, bufs, portable::update)
}

#[cfg(all(feature = "std", target_arch = "aarch64", target_feature = "neon"))]
#[inline]
fn adler32_dispatch_neon(adler: u32, data: &[u8]) -> u32 {
  aarch64::update_safe(adler, data)
}

#[cfg(all(feature = "std", target_arch = "aarch64", target_feature = "neon"))]
#[inline]
fn adler32_dispatch_neon_vectored(adler: u32, bufs: &[&[u8]]) -> u32 {
  adler32_apply_kernel_vectored(adler, bufs, aarch64::update_safe)
}

#[cfg(all(feature = "std", target_arch = "aarch64", target_feature = "neon"))]
#[inline]
fn adler32_dispatch_neon_dotprod(adler: u32, data: &[u8]) -> u32 {
  // SAFETY: NEON is enabled at compile time. Two paths select this wrapper, and both establish
  // the dot-product extension: `Adler32Force::NeonDotprod`, which survives `config::get()` only
  // when cached detection reports it, and `adler32_auto_kernel`, which checks it.
  unsafe { aarch64::update_dotprod(adler, data) }
}

#[cfg(all(feature = "std", target_arch = "aarch64", target_feature = "neon"))]
#[inline]
fn adler32_dispatch_neon_dotprod_vectored(adler: u32, bufs: &[&[u8]]) -> u32 {
  adler32_apply_kernel_vectored(adler, bufs, adler32_dispatch_neon_dotprod)
}

/// The kernel that automatic selection runs on this CPU, resolved once into the dispatch cache.
///
/// It applies the selection of `kernel_table::adler32`: the x86-64 kernel that cached detection
/// allows, the aarch64 dot-product kernel when detection reports `FEAT_DotProd`, the wasm32
/// SIMD128 kernel when the artifact requires SIMD128, and otherwise the kernel table's row. Every Adler-32 kernel handles every length, and each kernel table
/// names one Adler-32 kernel for all of its size classes, so one kernel serves every call.
#[cfg(feature = "std")]
#[inline]
fn adler32_auto_kernel() -> Adler32DispatchFn {
  let caps = crate::platform::caps();
  #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
  if let Some(kernel) = x86_64::select(caps) {
    return kernel.update;
  }
  #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
  if caps.has(crate::platform::caps::aarch64::DOTPROD) {
    return adler32_dispatch_neon_dotprod;
  }
  #[cfg(all(target_arch = "wasm32", target_feature = "simd128", not(feature = "portable-only")))]
  if caps.has(crate::platform::caps::wasm::SIMD128) {
    return wasm32::update;
  }
  let _ = caps;
  crate::checksum::kernel_table::active_table()
    .select_fns(usize::MAX)
    .adler32
}

// x86-64: a forced kernel survives `config::get()` only when cached detection reports every
// capability in its `x86_64::KERNELS` entry, and only that force selects its kernel here.

#[cfg(all(feature = "std", target_arch = "x86_64", target_feature = "sse2"))]
#[inline]
fn adler32_dispatch_ssse3_vectored(adler: u32, bufs: &[&[u8]]) -> u32 {
  adler32_apply_kernel_vectored(adler, bufs, x86_64::adler32_ssse3_safe)
}

#[cfg(all(feature = "std", target_arch = "x86_64", target_feature = "sse2"))]
#[inline]
fn adler32_dispatch_avx2_vectored(adler: u32, bufs: &[&[u8]]) -> u32 {
  adler32_apply_kernel_vectored(adler, bufs, x86_64::adler32_avx2_safe)
}

#[cfg(all(feature = "std", target_arch = "x86_64", target_feature = "sse2"))]
#[inline]
fn adler32_dispatch_avx512_vnni_vectored(adler: u32, bufs: &[&[u8]]) -> u32 {
  adler32_apply_kernel_vectored(adler, bufs, x86_64::adler32_avx512_vnni_safe)
}

// POWER and IBM Z: a forced kernel survives `config::get()` only when cached detection reports
// AltiVec or the vector facility, and only that force selects its kernel here.

#[cfg(all(feature = "std", target_arch = "powerpc64", not(feature = "portable-only")))]
#[inline]
fn adler32_dispatch_vmsum_vectored(adler: u32, bufs: &[&[u8]]) -> u32 {
  adler32_apply_kernel_vectored(adler, bufs, power::update_safe)
}

#[cfg(all(feature = "std", target_arch = "s390x", not(feature = "portable-only")))]
#[inline]
fn adler32_dispatch_vsum_vectored(adler: u32, bufs: &[&[u8]]) -> u32 {
  adler32_apply_kernel_vectored(adler, bufs, s390x::update_safe)
}

define_crc_dispatch! {
  word_ty: u32,
  dispatch_fn_ty: Adler32DispatchFn,
  dispatch_vectored_fn_ty: Adler32DispatchVectoredFn,
  auto_force: Adler32Force::Auto,
  force_expr: config::get().effective_force,
  active_table: crate::checksum::kernel_table::active_table(),
  auto_dispatch: adler32_dispatch_auto,
  auto_vectored_dispatch: adler32_dispatch_auto_vectored,
  dispatch_cache: ADLER32_DISPATCH,
  dispatch_vectored_cache: ADLER32_DISPATCH_VECTORED,
  resolve_dispatch: resolve_adler32_dispatch,
  resolve_dispatch_vectored: resolve_adler32_dispatch_vectored,
  dispatch: adler32_dispatch,
  dispatch_vectored: adler32_dispatch_vectored,
  resolved_dispatch: adler32_resolved_dispatch,
  runtime_paths: adler32_runtime_paths,
  resolve_match: {
    Adler32Force::Portable => adler32_dispatch_portable,
    Adler32Force::Auto => adler32_auto_kernel(),
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    Adler32Force::Neon => adler32_dispatch_neon,
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    Adler32Force::NeonDotprod => adler32_dispatch_neon_dotprod,
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    Adler32Force::Ssse3 => x86_64::adler32_ssse3_safe,
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    Adler32Force::Avx2 => x86_64::adler32_avx2_safe,
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    Adler32Force::Avx512Vnni => x86_64::adler32_avx512_vnni_safe,
    #[cfg(all(target_arch = "powerpc64", not(feature = "portable-only")))]
    Adler32Force::Vmsum => power::update_safe,
    #[cfg(all(target_arch = "s390x", not(feature = "portable-only")))]
    Adler32Force::Vsum => s390x::update_safe,
    _ => adler32_dispatch_auto,
  },
  resolve_vectored_match: {
    Adler32Force::Portable => adler32_dispatch_portable_vectored,
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    Adler32Force::Neon => adler32_dispatch_neon_vectored,
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    Adler32Force::NeonDotprod => adler32_dispatch_neon_dotprod_vectored,
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    Adler32Force::Ssse3 => adler32_dispatch_ssse3_vectored,
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    Adler32Force::Avx2 => adler32_dispatch_avx2_vectored,
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    Adler32Force::Avx512Vnni => adler32_dispatch_avx512_vnni_vectored,
    #[cfg(all(target_arch = "powerpc64", not(feature = "portable-only")))]
    Adler32Force::Vmsum => adler32_dispatch_vmsum_vectored,
    #[cfg(all(target_arch = "s390x", not(feature = "portable-only")))]
    Adler32Force::Vsum => adler32_dispatch_vsum_vectored,
    _ => adler32_dispatch_auto_vectored,
  }
}

// Kernel Name Introspection

/// Get the name of the kernel that would be selected for a given buffer length.
#[inline]
#[must_use]
fn adler32_selected_kernel_name(len: usize) -> &'static str {
  match config::get().effective_force {
    Adler32Force::Portable => return portable::KERNEL_NAME,
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    Adler32Force::Neon => return aarch64::KERNEL_NAME,
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    Adler32Force::NeonDotprod => return aarch64::DOTPROD_KERNEL_NAME,
    #[cfg(all(target_arch = "powerpc64", not(feature = "portable-only")))]
    Adler32Force::Vmsum => return power::KERNEL_NAME,
    #[cfg(all(target_arch = "s390x", not(feature = "portable-only")))]
    Adler32Force::Vsum => return s390x::KERNEL_NAME,
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    force => {
      if let Some(kernel) = x86_64::KERNELS.iter().find(|kernel| kernel.force == force) {
        return kernel.name;
      }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "sse2")))]
    _ => {}
  }
  crate::checksum::kernel_table::adler32_name(crate::checksum::kernel_table::active_table(), len)
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
  auto_table: Option<&'static crate::checksum::kernel_table::KernelTable>,
}

impl Adler32 {
  const INIT: u32 = 1;

  /// Create a hasher to resume from a previous Adler-32 value.
  #[inline]
  #[must_use]
  pub const fn resume(adler: u32) -> Self {
    Self {
      state: (((adler >> 16) % MOD) << 16) | ((adler & 0xffff) % MOD),
      // `resume` is const, so keep runtime force semantics via wrapper dispatch.
      dispatch: adler32_dispatch,
      auto_table: None,
    }
  }

  /// Get the effective Adler-32 configuration.
  #[must_use]
  pub fn config() -> Adler32Config {
    config::get()
  }

  /// Returns the kernel name that the selector would choose for `len`.
  #[must_use]
  pub fn kernel_name_for_len(len: usize) -> &'static str {
    adler32_selected_kernel_name(len)
  }
}

/// The dispatch of a new hasher. With `std`, every force, automatic selection included,
/// resolves once into the dispatch cache, so a call pays one cached load and one indirect call.
#[inline]
fn adler32_paths() -> (
  Adler32DispatchFn,
  Option<&'static crate::checksum::kernel_table::KernelTable>,
) {
  if cfg!(feature = "std") {
    (adler32_resolved_dispatch(), None)
  } else {
    adler32_runtime_paths()
  }
}

impl Checksum for Adler32 {
  const OUTPUT_SIZE: usize = 4;
  type Output = u32;

  #[inline]
  fn new() -> Self {
    let (dispatch, auto_table) = adler32_paths();
    Self {
      state: Self::INIT,
      dispatch,
      auto_table,
    }
  }

  #[inline]
  fn with_initial(initial: u32) -> Self {
    let (a, b) = sums(initial);
    let (dispatch, auto_table) = adler32_paths();
    Self {
      state: (b << 16) | a,
      dispatch,
      auto_table,
    }
  }

  #[inline]
  fn update(&mut self, data: &[u8]) {
    self.state = match self.auto_table {
      Some(table) => crate::checksum::kernel_table::adler32(table, self.state, data),
      None => (self.dispatch)(self.state, data),
    };
  }

  #[inline]
  fn update_vectored(&mut self, bufs: &[&[u8]]) {
    self.state = adler32_dispatch_vectored(self.state, bufs);
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

/// Automatic selection resolves one kernel for every length, which holds only while the active
/// kernel table names one Adler-32 kernel for all of its size classes.
#[cfg(all(test, feature = "std"))]
#[test]
fn the_active_table_names_one_adler32_kernel_for_every_size_class() {
  let table = crate::checksum::kernel_table::active_table();
  let largest = table.select_names(usize::MAX).adler32_name;
  for len in [0, 1, 64, 65, 256, 4096, 65_536, 1 << 20] {
    assert_eq!(table.select_names(len).adler32_name, largest, "len={len}");
  }
}

#[cfg(all(test, target_arch = "aarch64", target_feature = "neon", not(miri)))]
mod tests {
  use alloc::{vec, vec::Vec};

  use super::{aarch64, portable};
  use crate::checksum::dispatchers::Adler32Fn;

  /// The aarch64 kernels that this CPU can run, with their names.
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

  /// Lengths that end on and around every block, vector, and tail size of the kernels, the
  /// 5,552-byte portable and the 16 KiB NEON reduction lengths, and 8 KiB and 128 KiB, a few
  /// times over.
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
  fn aarch64_kernels_match_portable_for_every_length_alignment_and_start() {
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
  fn chunked_aarch64_writes_match_portable_one_shot() {
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
