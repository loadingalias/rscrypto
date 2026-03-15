//! Internal macros for CRC variant generation.
//!
//! These macros eliminate boilerplate when defining buffered CRC wrappers.

/// Generate a buffered CRC wrapper for a given inner CRC type.
///
/// This macro creates:
/// - The struct definition with `inner`, `buffer`, and `len` fields
/// - `new()`, `update()`, `finalize()`, `reset()`, `backend_name()` methods
/// - `Default` trait implementation
///
/// # Arguments
///
/// - `$name`: The wrapper type name (e.g., `BufferedCrc64Xz`)
/// - `$inner`: The inner CRC type (e.g., `Crc64Xz`)
/// - `$buffer_size`: The buffer size constant
/// - `$threshold_fn`: Function that returns the SIMD threshold
#[cfg(feature = "alloc")]
macro_rules! define_buffered_crc {
  (
    $(#[$outer:meta])*
    $vis:vis struct $name:ident<$inner:ty> {
      buffer_size: $buffer_size:expr,
      threshold_fn: $threshold_fn:expr,
    }
  ) => {
    $(#[$outer])*
    $vis struct $name {
      inner: $inner,
      buffer: alloc::boxed::Box<[u8; $buffer_size]>,
      len: usize,
    }

    impl $name {
      /// Create a new buffered CRC hasher.
      #[must_use]
      pub fn new() -> Self {
        Self {
          inner: <$inner as $crate::Checksum>::new(),
          buffer: alloc::boxed::Box::new([0u8; $buffer_size]),
          len: 0,
        }
      }

      /// Update the CRC with more data.
      ///
      /// Data is buffered internally until enough accumulates for efficient
      /// SIMD processing.
      #[allow(clippy::indexing_slicing)]
      // Safety: All slice indices are bounds-checked by the algorithm:
      // - self.len < buffer_size (invariant maintained by this function)
      // - fill = min(input.len(), space), so input[..fill] and buffer[len..len+fill] are valid
      // - aligned <= input.len() by construction
      pub fn update(&mut self, data: &[u8]) {
        let threshold = $threshold_fn();
        let mut input = data;

        // If we have buffered data, try to fill and flush
        if self.len > 0 {
          let space = $buffer_size - self.len;
          let fill = input.len().min(space);
          self.buffer[self.len..self.len + fill].copy_from_slice(&input[..fill]);
          self.len += fill;
          input = &input[fill..];

          // Flush if buffer is full or we have enough for SIMD
          if self.len >= $buffer_size || (self.len >= threshold && input.is_empty()) {
            <$inner as $crate::Checksum>::update(&mut self.inner, &self.buffer[..self.len]);
            self.len = 0;
          }
        }

        // Process large chunks directly
        if input.len() >= threshold {
          // Find largest aligned chunk
          let aligned = (input.len() / threshold) * threshold;
          <$inner as $crate::Checksum>::update(&mut self.inner, &input[..aligned]);
          input = &input[aligned..];
        }

        // Buffer remainder
        if !input.is_empty() {
          self.buffer[..input.len()].copy_from_slice(input);
          self.len = input.len();
        }
      }

      /// Update the CRC with multiple non-contiguous buffers.
      #[inline]
      pub fn update_vectored(&mut self, bufs: &[&[u8]]) {
        for &buf in bufs {
          self.update(buf);
        }
      }

      /// Update the CRC with `std::io::IoSlice` buffers.
      #[cfg(feature = "std")]
      #[inline]
      pub fn update_io_slices(&mut self, bufs: &[std::io::IoSlice<'_>]) {
        for buf in bufs {
          self.update(buf);
        }
      }

      /// Finalize and return the CRC value.
      ///
      /// Flushes any remaining buffered data before computing the final CRC.
      #[must_use]
      #[allow(clippy::indexing_slicing)]
      // Safety: self.len < buffer_size (invariant)
      pub fn finalize(&self) -> <$inner as $crate::Checksum>::Output {
        if self.len > 0 {
          // Clone inner to avoid mutating self
          let mut inner = self.inner.clone();
          <$inner as $crate::Checksum>::update(&mut inner, &self.buffer[..self.len]);
          <$inner as $crate::Checksum>::finalize(&inner)
        } else {
          <$inner as $crate::Checksum>::finalize(&self.inner)
        }
      }

      /// Reset the hasher to initial state.
      pub fn reset(&mut self) {
        <$inner as $crate::Checksum>::reset(&mut self.inner);
        self.len = 0;
      }

      /// Get the name of the currently selected backend.
      #[must_use]
      pub fn backend_name() -> &'static str {
        <$inner>::backend_name()
      }
    }

    impl Default for $name {
      fn default() -> Self {
        Self::new()
      }
    }
  };
}
