#[cfg(any(test, feature = "std"))]
use super::{
  Blake3, Blake3Xof, CHUNK_LEN, DERIVE_KEY_CONTEXT, DERIVE_KEY_MATERIAL, IV, KEY_LEN, KEYED_HASH, OUT_LEN, control,
  digest_oneshot, digest_oneshot_words, kernels, single_chunk_output, words8_from_le_bytes_32,
};
use crate::traits::{Digest, Xof};

#[cfg(any(test, feature = "std"))]
const STREAM_BENCH_KEY: [u8; KEY_LEN] = [
  0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10, 0x11, 0x12,
  0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F,
];

#[cfg(any(test, feature = "std"))]
const STREAM_BENCH_CONTEXT: &str = "rscrypto-blake3-stream-bench";

#[cfg(any(test, feature = "std"))]
#[inline]
fn squeeze_finalize_xof(mut xof: Blake3Xof) -> [u8; OUT_LEN] {
  let mut out = [0u8; OUT_LEN];
  xof.squeeze(&mut out);
  out
}

impl Blake3 {
  #[inline]
  #[allow(clippy::manual_saturating_arithmetic)] // Explicit clamp semantics are preferred here.
  fn clamp_add_usize(lhs: usize, rhs: usize) -> usize {
    lhs.checked_add(rhs).unwrap_or(usize::MAX)
  }

  /// One-shot hash using an explicitly selected kernel.
  ///
  /// This is crate-internal glue for `hashes::bench`.
  #[inline]
  #[must_use]
  #[cfg(any(test, feature = "std"))]
  pub(crate) fn digest_with_kernel_id(id: kernels::Blake3KernelId, data: &[u8]) -> [u8; OUT_LEN] {
    let kernel = kernels::kernel(id);
    digest_oneshot(kernel, IV, 0, data)
  }

  /// One-shot keyed hash using an explicitly selected kernel.
  ///
  /// This is crate-internal glue for `hashes::bench`.
  #[inline]
  #[must_use]
  #[cfg(any(test, feature = "std"))]
  pub(crate) fn keyed_digest_with_kernel_id(id: kernels::Blake3KernelId, key: &[u8; 32], data: &[u8]) -> [u8; OUT_LEN] {
    let kernel = kernels::kernel(id);
    let key_words = words8_from_le_bytes_32(key);
    digest_oneshot(kernel, key_words, KEYED_HASH, data)
  }

  /// One-shot derive-key hash using an explicitly selected kernel.
  ///
  /// This is crate-internal glue for `hashes::bench`.
  #[inline]
  #[must_use]
  #[cfg(any(test, feature = "std"))]
  pub(crate) fn derive_key_with_kernel_id(
    id: kernels::Blake3KernelId,
    context: &str,
    key_material: &[u8],
  ) -> [u8; OUT_LEN] {
    let kernel = kernels::kernel(id);
    let context_key_words = digest_oneshot_words(kernel, IV, DERIVE_KEY_CONTEXT, context.as_bytes());
    digest_oneshot(kernel, context_key_words, DERIVE_KEY_MATERIAL, key_material)
  }

  #[inline]
  #[cfg(any(test, feature = "std"))]
  fn stream_chunks_pattern_with_kernel_pair_and_state(
    stream_id: kernels::Blake3KernelId,
    bulk_id: kernels::Blake3KernelId,
    chunk_pattern: &[usize],
    key_words: [u32; 8],
    flags: u32,
    xof_mode: bool,
    data: &[u8],
  ) -> [u8; OUT_LEN] {
    let mut hasher = Self::new_internal_with(key_words, flags, stream_id);
    if chunk_pattern.is_empty() {
      hasher.update_with(data, stream_id, bulk_id);
    } else {
      let mut offset = 0usize;
      let mut idx = 0usize;
      while offset < data.len() {
        let step = chunk_pattern[idx % chunk_pattern.len()].max(1);
        let end = Self::clamp_add_usize(offset, step).min(data.len());
        hasher.update_with(&data[offset..end], stream_id, bulk_id);
        offset = end;
        idx = idx.strict_add(1);
      }
    }

    if xof_mode {
      squeeze_finalize_xof(hasher.finalize_xof())
    } else {
      hasher.finalize()
    }
  }

  #[inline]
  #[cfg(any(test, feature = "std"))]
  fn stream_chunks_with_kernel_pair_and_state(
    stream_id: kernels::Blake3KernelId,
    bulk_id: kernels::Blake3KernelId,
    chunk_size: usize,
    key_words: [u32; 8],
    flags: u32,
    data: &[u8],
  ) -> [u8; OUT_LEN] {
    Self::stream_chunks_pattern_with_kernel_pair_and_state(
      stream_id,
      bulk_id,
      core::slice::from_ref(&chunk_size),
      key_words,
      flags,
      false,
      data,
    )
  }

  /// Streaming hash with explicit `(stream, bulk)` kernel IDs.
  ///
  /// This is crate-internal glue for `hashes::bench`.
  #[inline]
  #[must_use]
  #[cfg(any(test, feature = "std"))]
  pub(crate) fn stream_chunks_with_kernel_pair_id(
    stream_id: kernels::Blake3KernelId,
    bulk_id: kernels::Blake3KernelId,
    chunk_size: usize,
    data: &[u8],
  ) -> [u8; OUT_LEN] {
    Self::stream_chunks_with_kernel_pair_and_state(stream_id, bulk_id, chunk_size, IV, 0, data)
  }

  /// Keyed streaming hash with explicit `(stream, bulk)` kernel IDs.
  ///
  /// This is crate-internal glue for `hashes::bench`.
  #[inline]
  #[must_use]
  #[cfg(any(test, feature = "std"))]
  pub(crate) fn stream_chunks_keyed_with_kernel_pair_id(
    stream_id: kernels::Blake3KernelId,
    bulk_id: kernels::Blake3KernelId,
    chunk_size: usize,
    data: &[u8],
  ) -> [u8; OUT_LEN] {
    let key_words = words8_from_le_bytes_32(&STREAM_BENCH_KEY);
    Self::stream_chunks_with_kernel_pair_and_state(stream_id, bulk_id, chunk_size, key_words, KEYED_HASH, data)
  }

  /// Derive-key-material streaming hash with explicit `(stream, bulk)` kernel IDs.
  ///
  /// This is crate-internal glue for `hashes::bench`.
  #[inline]
  #[must_use]
  #[cfg(any(test, feature = "std"))]
  pub(crate) fn stream_chunks_derive_with_kernel_pair_id(
    stream_id: kernels::Blake3KernelId,
    bulk_id: kernels::Blake3KernelId,
    chunk_size: usize,
    data: &[u8],
  ) -> [u8; OUT_LEN] {
    let stream = kernels::kernel(stream_id);
    let context_key_words = digest_oneshot_words(stream, IV, DERIVE_KEY_CONTEXT, STREAM_BENCH_CONTEXT.as_bytes());
    Self::stream_chunks_with_kernel_pair_and_state(
      stream_id,
      bulk_id,
      chunk_size,
      context_key_words,
      DERIVE_KEY_MATERIAL,
      data,
    )
  }

  /// Streaming XOF with explicit `(stream, bulk)` kernel IDs.
  ///
  /// This is crate-internal glue for `hashes::bench`.
  #[inline]
  #[must_use]
  #[cfg(any(test, feature = "std"))]
  pub(crate) fn stream_chunks_xof_with_kernel_pair_id(
    stream_id: kernels::Blake3KernelId,
    bulk_id: kernels::Blake3KernelId,
    chunk_size: usize,
    data: &[u8],
  ) -> [u8; OUT_LEN] {
    Self::stream_chunks_pattern_with_kernel_pair_and_state(
      stream_id,
      bulk_id,
      core::slice::from_ref(&chunk_size),
      IV,
      0,
      true,
      data,
    )
  }

  /// Streaming mixed-pattern hash with explicit `(stream, bulk)` kernel IDs.
  ///
  /// This is crate-internal glue for `hashes::bench`.
  #[inline]
  #[must_use]
  #[cfg(any(test, feature = "std"))]
  pub(crate) fn stream_chunks_mixed_with_kernel_pair_id(
    stream_id: kernels::Blake3KernelId,
    bulk_id: kernels::Blake3KernelId,
    chunk_pattern: &[usize],
    data: &[u8],
  ) -> [u8; OUT_LEN] {
    Self::stream_chunks_pattern_with_kernel_pair_and_state(stream_id, bulk_id, chunk_pattern, IV, 0, false, data)
  }

  /// Keyed streaming mixed-pattern hash with explicit `(stream, bulk)` kernel IDs.
  ///
  /// This is crate-internal glue for `hashes::bench`.
  #[inline]
  #[must_use]
  #[cfg(any(test, feature = "std"))]
  pub(crate) fn stream_chunks_mixed_keyed_with_kernel_pair_id(
    stream_id: kernels::Blake3KernelId,
    bulk_id: kernels::Blake3KernelId,
    chunk_pattern: &[usize],
    data: &[u8],
  ) -> [u8; OUT_LEN] {
    let key_words = words8_from_le_bytes_32(&STREAM_BENCH_KEY);
    Self::stream_chunks_pattern_with_kernel_pair_and_state(
      stream_id,
      bulk_id,
      chunk_pattern,
      key_words,
      KEYED_HASH,
      false,
      data,
    )
  }

  /// Derive-key-material streaming mixed-pattern hash with explicit `(stream, bulk)` kernel IDs.
  ///
  /// This is crate-internal glue for `hashes::bench`.
  #[inline]
  #[must_use]
  #[cfg(any(test, feature = "std"))]
  pub(crate) fn stream_chunks_mixed_derive_with_kernel_pair_id(
    stream_id: kernels::Blake3KernelId,
    bulk_id: kernels::Blake3KernelId,
    chunk_pattern: &[usize],
    data: &[u8],
  ) -> [u8; OUT_LEN] {
    let stream = kernels::kernel(stream_id);
    let context_key_words = digest_oneshot_words(stream, IV, DERIVE_KEY_CONTEXT, STREAM_BENCH_CONTEXT.as_bytes());
    Self::stream_chunks_pattern_with_kernel_pair_and_state(
      stream_id,
      bulk_id,
      chunk_pattern,
      context_key_words,
      DERIVE_KEY_MATERIAL,
      false,
      data,
    )
  }

  /// Streaming mixed-pattern XOF with explicit `(stream, bulk)` kernel IDs.
  ///
  /// This is crate-internal glue for `hashes::bench`.
  #[inline]
  #[must_use]
  #[cfg(any(test, feature = "std"))]
  pub(crate) fn stream_chunks_mixed_xof_with_kernel_pair_id(
    stream_id: kernels::Blake3KernelId,
    bulk_id: kernels::Blake3KernelId,
    chunk_pattern: &[usize],
    data: &[u8],
  ) -> [u8; OUT_LEN] {
    Self::stream_chunks_pattern_with_kernel_pair_and_state(stream_id, bulk_id, chunk_pattern, IV, 0, true, data)
  }

  #[inline]
  #[must_use]
  #[cfg(any(test, feature = "std"))]
  pub(crate) fn digest_portable(data: &[u8]) -> [u8; OUT_LEN] {
    let kernel = kernels::kernel(kernels::Blake3KernelId::Portable);
    if data.len() <= CHUNK_LEN {
      let output = single_chunk_output(kernel, IV, 0, 0, data);
      output.root_hash_bytes()
    } else {
      let mut hasher = Self::new_internal_with(IV, 0, kernels::Blake3KernelId::Portable);
      hasher.update_with(
        data,
        kernels::Blake3KernelId::Portable,
        kernels::Blake3KernelId::Portable,
      );
      hasher.finalize()
    }
  }
}

/// Benchmark-only hooks (not part of the stable API).
#[cfg(feature = "std")]
#[doc(hidden)]
pub mod __bench {
  pub use super::control::bench::*;
}
