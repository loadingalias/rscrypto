//! Internal macros for CRC variant generation.
//!
//! These macros eliminate boilerplate when defining multiple polynomial variants
//! of CRC algorithms (e.g., CRC-64-XZ and CRC-64-NVME share identical structure
//! but different polynomials and tables).

/// Generate a CRC-64 variant type with all trait implementations.
///
/// This macro creates:
/// - The struct definition with `state: u64`
/// - `resume()`, `backend_name()`, `config()`, `tunables()`, `kernel_name_for_len()` methods
/// - `Checksum` trait implementation
/// - `ChecksumCombine` trait implementation
///
/// # Arguments
///
/// - `$name`: The type name (e.g., `Crc64Xz`)
/// - `$poly`: The polynomial constant (e.g., `CRC64_XZ_POLY`)
/// - `$dispatcher`: The dispatcher static (e.g., `CRC64_XZ_DISPATCHER`)
/// - `$doc`: Doc string for the type
macro_rules! define_crc64_type {
  (
    $(#[$outer:meta])*
    $vis:vis struct $name:ident {
      poly: $poly:expr,
      dispatcher: $dispatcher:expr,
      portable: $portable:expr,
    }
  ) => {
    $(#[$outer])*
    #[derive(Clone)]
    $vis struct $name {
      state: u64,
      kernel: $crate::dispatchers::Crc64Fn,
      initialized: bool,
    }

    impl $name {
      /// Pre-computed shift-by-8 matrix for combine.
      const SHIFT8_MATRIX: $crate::common::combine::Gf2Matrix64 =
        $crate::common::combine::generate_shift8_matrix_64($poly);

      /// Create a hasher to resume from a previous CRC value.
      #[inline]
      #[must_use]
      pub const fn resume(crc: u64) -> Self {
        Self {
          state: crc ^ !0,
          kernel: $portable,
          initialized: false,
        }
      }

      /// Get the name of the currently selected backend.
      ///
      /// Returns the dispatcher name (e.g., "portable/slice16", "x86_64/auto").
      #[must_use]
      pub fn backend_name() -> &'static str {
        $dispatcher.backend_name()
      }

      /// Get the effective CRC-64 configuration (force mode).
      #[must_use]
      pub fn config() -> $crate::crc64::Crc64Config {
        $crate::crc64::config::get()
      }

      /// Returns the kernel name that the selector would choose for `len`.
      ///
      /// This is intended for debugging/benchmarking and does not allocate.
      #[must_use]
      pub fn kernel_name_for_len(len: usize) -> &'static str {
        $crate::crc64::crc64_selected_kernel_name(len)
      }
    }

    impl $crate::Checksum for $name {
      const OUTPUT_SIZE: usize = 8;
      type Output = u64;

      #[inline]
      fn new() -> Self {
        Self {
          state: !0,
          kernel: $dispatcher.kernel(),
          initialized: true,
        }
      }

      #[inline]
      fn with_initial(initial: u64) -> Self {
        Self {
          state: initial ^ !0,
          kernel: $dispatcher.kernel(),
          initialized: true,
        }
      }

      #[inline]
      fn update(&mut self, data: &[u8]) {
        if !self.initialized {
          self.kernel = $dispatcher.kernel();
          self.initialized = true;
        }
        self.state = (self.kernel)(self.state, data);
      }

      #[inline]
      fn finalize(&self) -> u64 {
        self.state ^ !0
      }

      #[inline]
      fn reset(&mut self) {
        self.state = !0;
      }
    }

    impl Default for $name {
      #[inline]
      fn default() -> Self {
        <$name as $crate::Checksum>::new()
      }
    }

    impl $crate::ChecksumCombine for $name {
      fn combine(crc_a: u64, crc_b: u64, len_b: usize) -> u64 {
        $crate::common::combine::combine_crc64(crc_a, crc_b, len_b, Self::SHIFT8_MATRIX)
      }
    }
  };
}

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
