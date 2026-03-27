//! CRC-24 implementations with optional hardware acceleration.
//!
//! This module provides:
//! - [`Crc24OpenPgp`] - CRC-24/OPENPGP (RFC 4880)
//!
//! # Quick Start
//!
//! ```rust
//! use rscrypto::checksum::{Checksum, ChecksumCombine, Crc24OpenPgp};
//!
//! let data = b"123456789";
//! assert_eq!(Crc24OpenPgp::checksum(data), 0x21CF02);
//!
//! let (a, b) = data.split_at(4);
//! let combined = Crc24OpenPgp::combine(
//!   Crc24OpenPgp::checksum(a),
//!   Crc24OpenPgp::checksum(b),
//!   b.len(),
//! );
//! assert_eq!(combined, Crc24OpenPgp::checksum(data));
//! ```

pub(crate) mod config;
pub(crate) mod kernels;
pub(crate) mod keys;
pub(crate) mod portable;
#[cfg(any(
  target_arch = "x86_64",
  target_arch = "aarch64",
  target_arch = "powerpc64",
  target_arch = "s390x",
  target_arch = "riscv64"
))]
mod reflected;

#[cfg(all(test, feature = "alloc"))]
pub mod kernel_test;

#[allow(unused_imports)]
pub use config::{Crc24Config, Crc24Force};

#[cfg(any(test, feature = "std"))]
use crate::checksum::common::reference::crc24_bitwise;
#[cfg(any(test, feature = "testing"))]
use crate::checksum::common::tables::generate_crc24_tables_4;
use crate::checksum::common::{
  combine::{Gf2Matrix24, combine_crc24, generate_shift8_matrix_24},
  tables::{CRC24_OPENPGP_POLY, generate_crc24_tables_8},
};
// Re-export traits for test modules (`use super::*`).
#[allow(unused_imports)]
pub(super) use crate::traits::{Checksum, ChecksumCombine};

#[cfg(target_arch = "aarch64")]
mod aarch64;
#[cfg(target_arch = "powerpc64")]
mod power;
#[cfg(target_arch = "riscv64")]
mod riscv64;
#[cfg(target_arch = "s390x")]
mod s390x;
#[cfg(target_arch = "x86_64")]
mod x86_64;

// ─────────────────────────────────────────────────────────────────────────────
// Kernel Tables (compile-time)
// ─────────────────────────────────────────────────────────────────────────────

mod kernel_tables {
  use super::*;
  #[cfg(any(test, feature = "testing"))]
  pub static OPENPGP_TABLES_4: [[u32; 256]; 4] = generate_crc24_tables_4(CRC24_OPENPGP_POLY);
  pub static OPENPGP_TABLES_8: [[u32; 256]; 8] = generate_crc24_tables_8(CRC24_OPENPGP_POLY);
}

// ─────────────────────────────────────────────────────────────────────────────
// Reference Kernel Wrapper
// ─────────────────────────────────────────────────────────────────────────────

/// Bitwise reference implementation for CRC-24/OPENPGP.
#[cfg(any(test, feature = "std"))]
#[inline]
fn crc24_openpgp_reference(crc: u32, data: &[u8]) -> u32 {
  crc24_bitwise(CRC24_OPENPGP_POLY, crc, data)
}

// ─────────────────────────────────────────────────────────────────────────────
// Auto Dispatch Function (using new dispatch module)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "std")]
type Crc24DispatchFn = crate::checksum::dispatchers::Crc24Fn;
#[cfg(feature = "std")]
type Crc24DispatchVectoredFn = fn(u32, &[&[u8]]) -> u32;

#[cfg(feature = "std")]
#[inline]
fn crc24_apply_kernel_vectored(mut crc: u32, bufs: &[&[u8]], kernel: Crc24DispatchFn) -> u32 {
  for &buf in bufs {
    if !buf.is_empty() {
      crc = kernel(crc, buf);
    }
  }
  crc
}

#[inline]
fn crc24_openpgp_dispatch_auto(crc: u32, data: &[u8]) -> u32 {
  let table = crate::checksum::dispatch::active_table();
  let kernel = table.select_fns(data.len()).crc24_openpgp;
  kernel(crc, data)
}

#[inline]
fn crc24_openpgp_dispatch_auto_vectored(crc: u32, bufs: &[&[u8]]) -> u32 {
  crc_vectored_dispatch!(crate::checksum::dispatch::active_table(), crc, crc24_openpgp, bufs)
}

#[cfg(feature = "std")]
#[inline]
fn crc24_openpgp_dispatch_reference(crc: u32, data: &[u8]) -> u32 {
  crc24_openpgp_reference(crc, data)
}

#[cfg(feature = "std")]
#[inline]
fn crc24_openpgp_dispatch_portable(crc: u32, data: &[u8]) -> u32 {
  portable::crc24_openpgp_slice8(crc, data)
}

#[cfg(feature = "std")]
#[inline]
fn crc24_openpgp_dispatch_reference_vectored(crc: u32, bufs: &[&[u8]]) -> u32 {
  crc24_apply_kernel_vectored(crc, bufs, crc24_openpgp_reference)
}

#[cfg(feature = "std")]
#[inline]
fn crc24_openpgp_dispatch_portable_vectored(crc: u32, bufs: &[&[u8]]) -> u32 {
  crc24_apply_kernel_vectored(crc, bufs, portable::crc24_openpgp_slice8)
}

#[cfg(feature = "std")]
static CRC24_OPENPGP_DISPATCH: crate::backend::cache::OnceCache<Crc24DispatchFn> =
  crate::backend::cache::OnceCache::new();
#[cfg(feature = "std")]
static CRC24_OPENPGP_DISPATCH_VECTORED: crate::backend::cache::OnceCache<Crc24DispatchVectoredFn> =
  crate::backend::cache::OnceCache::new();

#[cfg(feature = "std")]
#[inline]
fn resolve_crc24_openpgp_dispatch() -> Crc24DispatchFn {
  match config::get().effective_force {
    Crc24Force::Reference => crc24_openpgp_dispatch_reference,
    Crc24Force::Portable => crc24_openpgp_dispatch_portable,
    _ => crc24_openpgp_dispatch_auto,
  }
}

#[cfg(feature = "std")]
#[inline]
fn resolve_crc24_openpgp_dispatch_vectored() -> Crc24DispatchVectoredFn {
  match config::get().effective_force {
    Crc24Force::Reference => crc24_openpgp_dispatch_reference_vectored,
    Crc24Force::Portable => crc24_openpgp_dispatch_portable_vectored,
    _ => crc24_openpgp_dispatch_auto_vectored,
  }
}

/// CRC-24/OPENPGP dispatch - hot path uses one-time resolved dispatch function.
#[inline]
fn crc24_openpgp_dispatch(crc: u32, data: &[u8]) -> u32 {
  #[cfg(feature = "std")]
  {
    let dispatch = CRC24_OPENPGP_DISPATCH.get_or_init(resolve_crc24_openpgp_dispatch);
    dispatch(crc, data)
  }

  #[cfg(not(feature = "std"))]
  {
    crc24_openpgp_dispatch_auto(crc, data)
  }
}

/// CRC-24/OPENPGP vectored dispatch (processes multiple buffers in order).
#[inline]
fn crc24_openpgp_dispatch_vectored(crc: u32, bufs: &[&[u8]]) -> u32 {
  #[cfg(feature = "std")]
  {
    let dispatch = CRC24_OPENPGP_DISPATCH_VECTORED.get_or_init(resolve_crc24_openpgp_dispatch_vectored);
    dispatch(crc, bufs)
  }

  #[cfg(not(feature = "std"))]
  {
    crc24_openpgp_dispatch_auto_vectored(crc, bufs)
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Introspection
// ─────────────────────────────────────────────────────────────────────────────

/// Get the name of the CRC-24/OPENPGP kernel that would be selected for a given buffer length.
///
/// This is primarily for diagnostics and testing. The actual kernel selection
/// happens via the dispatch module's pre-computed tables.
#[inline]
#[must_use]
pub(crate) fn crc24_openpgp_selected_kernel_name(len: usize) -> &'static str {
  let cfg = config::get();

  // Handle forced modes
  if cfg.effective_force == Crc24Force::Reference {
    return kernels::REFERENCE;
  }
  if cfg.effective_force == Crc24Force::Portable {
    return kernels::PORTABLE_SLICE8;
  }

  // For auto mode, return the specific kernel name from the dispatch table
  let table = crate::checksum::dispatch::active_table();
  table.select_names(len).crc24_openpgp_name
}

// ─────────────────────────────────────────────────────────────────────────────
// CRC-24 Types
// ─────────────────────────────────────────────────────────────────────────────

/// CRC-24/OPENPGP checksum.
///
/// Used by OpenPGP (RFC 4880) for ASCII armor / Radix-64 integrity.
///
/// # Properties
///
/// - **Polynomial**: 0x864CFB (normal)
/// - **Initial value**: 0xB704CE
/// - **Final XOR**: 0x000000
/// - **Reflect input/output**: No
///
/// # Examples
///
/// ```rust
/// use rscrypto::checksum::{Checksum, Crc24OpenPgp};
///
/// let crc = Crc24OpenPgp::checksum(b"123456789");
/// assert_eq!(crc, 0x21CF02);
/// ```
#[derive(Clone, Copy)]
pub struct Crc24OpenPgp {
  state: u32,
}

impl Crc24OpenPgp {
  const MASK: u32 = 0x00FF_FFFF;
  const INIT: u32 = 0x00B7_04CE;
  const XOROUT: u32 = 0x0000_0000;
  const INIT_XOROUT: u32 = Self::INIT ^ Self::XOROUT;

  /// Pre-computed shift-by-8 matrix for combine.
  const SHIFT8_MATRIX: Gf2Matrix24 = generate_shift8_matrix_24(CRC24_OPENPGP_POLY);

  /// Create a hasher to resume from a previous CRC value.
  #[inline]
  #[must_use]
  pub const fn resume(crc: u32) -> Self {
    Self {
      state: (crc ^ Self::XOROUT) & Self::MASK,
    }
  }

  /// Get the effective CRC-24 configuration.
  #[must_use]
  pub fn config() -> Crc24Config {
    config::get()
  }

  /// Returns the kernel name that the selector would choose for `len`.
  #[must_use]
  pub fn kernel_name_for_len(len: usize) -> &'static str {
    crc24_openpgp_selected_kernel_name(len)
  }
}

impl crate::traits::Checksum for Crc24OpenPgp {
  const OUTPUT_SIZE: usize = 3;
  type Output = u32;

  #[inline]
  fn new() -> Self {
    Self { state: Self::INIT }
  }

  #[inline]
  fn with_initial(initial: u32) -> Self {
    Self {
      state: (initial ^ Self::XOROUT) & Self::MASK,
    }
  }

  #[inline]
  fn update(&mut self, data: &[u8]) {
    if data.len() <= 7 {
      self.state = portable::crc24_openpgp_bytewise(self.state, data);
      return;
    }
    self.state = crc24_openpgp_dispatch(self.state, data) & Self::MASK;
  }

  #[inline]
  fn update_vectored(&mut self, bufs: &[&[u8]]) {
    self.state = crc24_openpgp_dispatch_vectored(self.state, bufs) & Self::MASK;
  }

  #[inline]
  fn finalize(&self) -> u32 {
    (self.state ^ Self::XOROUT) & Self::MASK
  }

  #[inline]
  fn reset(&mut self) {
    self.state = Self::INIT;
  }

  #[inline]
  fn checksum(data: &[u8]) -> u32 {
    crate::checksum::dispatch::crc24_openpgp(data)
  }
}

impl core::fmt::Debug for Crc24OpenPgp {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("Crc24OpenPgp").finish_non_exhaustive()
  }
}

impl Default for Crc24OpenPgp {
  fn default() -> Self {
    <Self as crate::traits::Checksum>::new()
  }
}

impl crate::traits::ChecksumCombine for Crc24OpenPgp {
  fn combine(crc_a: u32, crc_b: u32, len_b: usize) -> u32 {
    combine_crc24(crc_a, crc_b, len_b, Self::SHIFT8_MATRIX, Self::INIT_XOROUT)
  }
}

#[cfg(feature = "alloc")]
impl Crc24OpenPgp {
  #[must_use]
  pub fn buffered() -> crate::checksum::buffered::BufferedCrc24OpenPgp {
    crate::checksum::buffered::BufferedCrc24OpenPgp::new()
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Buffered CRC-24 Wrapper
// ─────────────────────────────────────────────────────────────────────────────

/// Buffer size for buffered CRC-24 wrappers.
#[cfg(feature = "alloc")]
const BUFFERED_CRC24_BUFFER_SIZE: usize = 256;

#[cfg(feature = "alloc")]
#[inline]
#[must_use]
fn crc24_buffered_threshold() -> usize {
  64
}

#[cfg(feature = "alloc")]
define_buffered_crc! {
  /// A buffering wrapper around [`Crc24OpenPgp`] for streaming small chunks.
  pub struct BufferedCrc24OpenPgp<Crc24OpenPgp> {
    buffer_size: BUFFERED_CRC24_BUFFER_SIZE,
    threshold_fn: crc24_buffered_threshold,
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel Introspection
// ─────────────────────────────────────────────────────────────────────────────

impl crate::checksum::introspect::KernelIntrospect for Crc24OpenPgp {
  fn kernel_name_for_len(len: usize) -> &'static str {
    Self::kernel_name_for_len(len)
  }
}

#[cfg(test)]
mod tests {
  extern crate std;

  use super::*;

  #[test]
  fn test_vectors_crc24_openpgp() {
    assert_eq!(Crc24OpenPgp::checksum(b"123456789"), 0x0021_CF02);
    assert_eq!(Crc24OpenPgp::checksum(b""), 0x00B7_04CE);
  }

  #[test]
  fn test_combine_all_splits_openpgp() {
    let data = b"The quick brown fox jumps over the lazy dog";
    let full = Crc24OpenPgp::checksum(data);
    for split in 0..=data.len() {
      let (a, b) = data.split_at(split);
      let combined = Crc24OpenPgp::combine(Crc24OpenPgp::checksum(a), Crc24OpenPgp::checksum(b), b.len());
      assert_eq!(combined, full, "split={split}");
    }
  }

  #[test]
  fn test_streaming_resume_openpgp() {
    let data = b"The quick brown fox jumps over the lazy dog";
    let oneshot = Crc24OpenPgp::checksum(data);

    for &split in &[1, data.len() / 4, data.len() / 2, data.len().strict_sub(1)] {
      let (a, b) = data.split_at(split);
      let crc_a = Crc24OpenPgp::checksum(a);
      let mut resumed = Crc24OpenPgp::resume(crc_a);
      resumed.update(b);
      assert_eq!(
        resumed.finalize(),
        oneshot,
        "Crc24OpenPgp resume failed at split={split}"
      );
    }
  }

  #[cfg(feature = "alloc")]
  #[test]
  fn test_buffered_openpgp_matches_unbuffered() {
    let data = b"The quick brown fox jumps over the lazy dog";
    let expected = Crc24OpenPgp::checksum(data);

    let mut buffered = BufferedCrc24OpenPgp::new();
    for chunk in data.chunks(3) {
      buffered.update(chunk);
    }
    assert_eq!(buffered.finalize(), expected);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Cross-Check Tests: All accelerated kernels vs bitwise reference
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod cross_check {
  extern crate std;

  use super::*;
  use crate::checksum::common::tests::{STREAMING_CHUNK_SIZES, TEST_LENGTHS, generate_test_data};

  // ─────────────────────────────────────────────────────────────────────────
  // CRC-24/OPENPGP Cross-Check Tests
  // ─────────────────────────────────────────────────────────────────────────

  const INIT: u32 = 0x00B7_04CE;
  const MASK: u32 = 0x00FF_FFFF;

  #[test]
  fn openpgp_all_lengths() {
    for &len in TEST_LENGTHS {
      let data = generate_test_data(len);
      let reference = crc24_openpgp_reference(INIT, &data) & MASK;
      let actual = Crc24OpenPgp::checksum(&data);
      assert_eq!(actual, reference, "CRC-24/OPENPGP mismatch at len={len}");
    }
  }

  #[test]
  fn openpgp_all_single_bytes() {
    for byte in 0u8..=255 {
      let data = [byte];
      let reference = crc24_openpgp_reference(INIT, &data) & MASK;
      let actual = Crc24OpenPgp::checksum(&data);
      assert_eq!(actual, reference, "CRC-24/OPENPGP mismatch for byte={byte:#04X}");
    }
  }

  #[test]
  fn openpgp_streaming_all_chunk_sizes() {
    let data = generate_test_data(4096);
    let reference = crc24_openpgp_reference(INIT, &data) & MASK;

    for &chunk_size in STREAMING_CHUNK_SIZES {
      let mut hasher = Crc24OpenPgp::new();
      for chunk in data.chunks(chunk_size) {
        hasher.update(chunk);
      }
      let actual = hasher.finalize();
      assert_eq!(
        actual, reference,
        "CRC-24/OPENPGP streaming mismatch with chunk_size={chunk_size}"
      );
    }
  }

  #[test]
  fn openpgp_combine_all_splits() {
    let data = generate_test_data(1024);
    let reference = crc24_openpgp_reference(INIT, &data) & MASK;

    for split in [0, 1, 15, 16, 17, 127, 128, 129, 511, 512, 513, 1023, 1024] {
      if split > data.len() {
        continue;
      }
      let (a, b) = data.split_at(split);
      let crc_a = Crc24OpenPgp::checksum(a);
      let crc_b = Crc24OpenPgp::checksum(b);
      let combined = Crc24OpenPgp::combine(crc_a, crc_b, b.len());
      assert_eq!(combined, reference, "CRC-24/OPENPGP combine mismatch at split={split}");
    }
  }

  #[test]
  fn openpgp_unaligned_offsets() {
    let data = generate_test_data(4096 + 64);

    for offset in 1..=16 {
      let slice = &data[offset..offset + 4096];
      let reference = crc24_openpgp_reference(INIT, slice) & MASK;
      let actual = Crc24OpenPgp::checksum(slice);
      assert_eq!(
        actual, reference,
        "CRC-24/OPENPGP unaligned mismatch at offset={offset}"
      );
    }
  }

  #[test]
  fn openpgp_byte_at_a_time_streaming() {
    let data = generate_test_data(256);
    let reference = crc24_openpgp_reference(INIT, &data) & MASK;

    let mut hasher = Crc24OpenPgp::new();
    for &byte in &data {
      hasher.update(&[byte]);
    }
    assert_eq!(hasher.finalize(), reference, "CRC-24/OPENPGP byte-at-a-time mismatch");
  }

  #[test]
  fn openpgp_reference_kernel_accessible() {
    let data = b"123456789";
    let expected = 0x0021_CF02_u32;
    let reference = crc24_openpgp_reference(INIT, data) & MASK;
    assert_eq!(reference, expected, "Reference kernel check value mismatch");
  }

  #[test]
  fn openpgp_portable_matches_reference() {
    for &len in TEST_LENGTHS {
      let data = generate_test_data(len);
      let reference = crc24_openpgp_reference(INIT, &data) & MASK;
      let portable = portable::crc24_openpgp_slice8(INIT, &data) & MASK;
      assert_eq!(portable, reference, "CRC-24/OPENPGP portable mismatch at len={len}");
    }
  }
}

#[cfg(test)]
crate::define_crc_property_tests!(crc24_openpgp_props, Crc24OpenPgp);
