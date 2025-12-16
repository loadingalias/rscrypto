//! CRC32-C (Castagnoli) checksum.
//!
//! CRC32-C uses polynomial 0x1EDC6F41, which was specifically designed to have
//! excellent error detection properties for data storage and networking.
//!
//! # Performance
//!
//! | Platform | Throughput | Implementation |
//! |----------|------------|----------------|
//! | Sapphire Rapids (AVX-512) | ~100 GB/s | VPCLMULQDQ + hybrid |
//! | Apple M1/M2/M3 | ~85 GB/s | PMULL + EOR3 |
//! | Ice Lake (AVX-512) | ~64 GB/s | VPCLMULQDQ |
//! | Haswell (AVX2) | ~25 GB/s | PCLMULQDQ |
//! | Portable | ~500 MB/s | Slicing-by-8 |
//!
//! # Usage
//!
//! ```
//! use checksum::Crc32c;
//!
//! // One-shot computation (fastest for single buffers)
//! let crc = Crc32c::checksum(b"hello world");
//!
//! // Incremental computation (for streaming)
//! let mut hasher = Crc32c::new();
//! hasher.update(b"hello ");
//! hasher.update(b"world");
//! assert_eq!(hasher.finalize(), crc);
//! ```
//!
//! # Hardware Acceleration
//!
//! The implementation automatically uses the fastest available instruction set:
//!
//! - **x86_64**: VPCLMULQDQ (AVX-512) → PCLMULQDQ → SSE4.2 → portable
//! - **aarch64**: PMULL+EOR3 → PMULL → CRC32 extension → portable

pub(crate) mod portable;

// Future SIMD modules:
#[cfg(target_arch = "aarch64")]
pub(crate) mod aarch64;

#[cfg(target_arch = "x86_64")]
pub(crate) mod x86_64;

use traits::{Checksum, ChecksumCombine};

/// CRC32-C (Castagnoli) checksum.
///
/// This struct implements streaming CRC32-C computation with automatic
/// hardware acceleration when available.
///
/// # Thread Safety
///
/// `Crc32c` is `Send` and `Sync`. Multiple hashers can operate in parallel
/// on different data, and results can be combined using [`combine`](Self::combine).
#[derive(Clone, Debug)]
pub struct Crc32c {
  /// Current CRC state (inverted - XOR applied on finalize)
  state: u32,
  /// Initial value for reset
  initial: u32,
}

impl Crc32c {
  /// Initial value for CRC32-C (all ones).
  const INIT: u32 = 0xFFFF_FFFF;

  /// Create a new hasher with the default initial value.
  #[inline]
  #[must_use]
  pub const fn new() -> Self {
    Self {
      state: Self::INIT,
      initial: Self::INIT,
    }
  }

  /// Create a new hasher that will resume from a previous CRC.
  ///
  /// This is useful for continuing a checksum computation that was
  /// interrupted or for implementing parallel computation.
  ///
  /// # Example
  ///
  /// ```
  /// use checksum::Crc32c;
  ///
  /// let data = b"hello world";
  /// let (first, second) = data.split_at(6);
  ///
  /// // Compute CRC of first part
  /// let crc1 = Crc32c::checksum(first);
  ///
  /// // Resume from that CRC
  /// let mut hasher = Crc32c::resume(crc1);
  /// hasher.update(second);
  ///
  /// // Result matches one-shot computation
  /// assert_eq!(hasher.finalize(), Crc32c::checksum(data));
  /// ```
  #[inline]
  #[must_use]
  pub const fn resume(crc: u32) -> Self {
    Self {
      // Invert back to internal state
      state: crc ^ Self::INIT,
      initial: crc ^ Self::INIT,
    }
  }

  /// Compute CRC32-C of data in one shot.
  ///
  /// This is the fastest path for data that fits in memory.
  ///
  /// # Example
  ///
  /// ```
  /// use checksum::Crc32c;
  ///
  /// assert_eq!(Crc32c::checksum(b"123456789"), 0xE3069283);
  /// ```
  #[inline]
  #[must_use]
  pub fn checksum(data: &[u8]) -> u32 {
    dispatch(Self::INIT, data) ^ Self::INIT
  }

  /// Update the hasher with additional data.
  #[inline]
  pub fn update(&mut self, data: &[u8]) {
    self.state = dispatch(self.state, data);
  }

  /// Finalize and return the checksum.
  ///
  /// This does not consume the hasher, allowing further updates.
  #[inline]
  #[must_use]
  pub const fn finalize(&self) -> u32 {
    self.state ^ Self::INIT
  }

  /// Reset the hasher to its initial state.
  #[inline]
  pub fn reset(&mut self) {
    self.state = self.initial;
  }

  /// Get the current CRC state.
  ///
  /// This returns the same value as [`finalize`](Self::finalize) and is
  /// provided for API consistency with other checksum implementations.
  #[inline]
  #[must_use]
  pub const fn state(&self) -> u32 {
    self.finalize()
  }

  /// Combine two CRCs: `crc(A || B)` from `crc(A)`, `crc(B)`, `len(B)`.
  ///
  /// This operation runs in O(log n) time where n is `len_b`.
  ///
  /// # Example
  ///
  /// ```
  /// use checksum::Crc32c;
  ///
  /// let data = b"hello world";
  /// let (a, b) = data.split_at(6);
  ///
  /// let crc_a = Crc32c::checksum(a);
  /// let crc_b = Crc32c::checksum(b);
  /// let crc_ab = Crc32c::checksum(data);
  ///
  /// assert_eq!(Crc32c::combine(crc_a, crc_b, b.len()), crc_ab);
  /// ```
  #[inline]
  #[must_use]
  pub fn combine(crc_a: u32, crc_b: u32, len_b: usize) -> u32 {
    crate::combine::crc32c_combine(crc_a, crc_b, len_b)
  }
}

impl Default for Crc32c {
  #[inline]
  fn default() -> Self {
    Self::new()
  }
}

impl Checksum for Crc32c {
  const OUTPUT_SIZE: usize = 4;
  type Output = u32;

  #[inline]
  fn new() -> Self {
    Crc32c::new()
  }

  #[inline]
  fn with_initial(initial: Self::Output) -> Self {
    Self {
      state: initial ^ Self::INIT,
      initial: initial ^ Self::INIT,
    }
  }

  #[inline]
  fn update(&mut self, data: &[u8]) {
    Crc32c::update(self, data);
  }

  #[inline]
  fn finalize(&self) -> Self::Output {
    Crc32c::finalize(self)
  }

  #[inline]
  fn reset(&mut self) {
    Crc32c::reset(self);
  }

  #[inline]
  fn checksum(data: &[u8]) -> Self::Output {
    Crc32c::checksum(data)
  }
}

impl ChecksumCombine for Crc32c {
  #[inline]
  fn combine(crc_a: Self::Output, crc_b: Self::Output, len_b: usize) -> Self::Output {
    Crc32c::combine(crc_a, crc_b, len_b)
  }
}

/// Returns the CRC32C backend this build will use on the current machine.
///
/// This is intended for diagnostics and benchmarking.
#[doc(hidden)]
#[inline]
#[must_use]
#[cfg(all(
  target_arch = "aarch64",
  target_feature = "aes",
  target_feature = "crc",
  target_feature = "sha3"
))]
pub fn selected_backend() -> &'static str {
  "aarch64/pmull+eor3 (compile-time)"
}

#[doc(hidden)]
#[inline]
#[must_use]
#[cfg(all(
  target_arch = "aarch64",
  target_feature = "aes",
  target_feature = "crc",
  not(target_feature = "sha3")
))]
pub fn selected_backend() -> &'static str {
  "aarch64/pmull (compile-time)"
}

#[doc(hidden)]
#[inline]
#[must_use]
#[cfg(all(target_arch = "aarch64", target_feature = "crc", not(target_feature = "aes")))]
pub fn selected_backend() -> &'static str {
  "aarch64/crc (compile-time)"
}

#[doc(hidden)]
#[inline]
#[must_use]
#[cfg(all(
  target_arch = "x86_64",
  target_feature = "avx512f",
  target_feature = "avx512vl",
  target_feature = "avx512bw",
  target_feature = "vpclmulqdq",
  target_feature = "pclmulqdq"
))]
pub fn selected_backend() -> &'static str {
  "x86_64/vpclmul (compile-time)"
}

#[doc(hidden)]
#[inline]
#[must_use]
#[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq", target_feature = "ssse3"))]
pub fn selected_backend() -> &'static str {
  "x86_64/pclmul (compile-time)"
}

#[doc(hidden)]
#[inline]
#[must_use]
#[cfg(all(target_arch = "x86_64", target_feature = "sse4.2"))]
pub fn selected_backend() -> &'static str {
  "x86_64/sse4.2 (compile-time)"
}

#[doc(hidden)]
#[inline]
#[must_use]
#[cfg(target_arch = "wasm32")]
pub fn selected_backend() -> &'static str {
  #[cfg(feature = "no-tables")]
  return "wasm32/bitwise";

  #[cfg(not(feature = "no-tables"))]
  "wasm32/slicing-by-8"
}

#[doc(hidden)]
#[inline]
#[must_use]
#[cfg(not(any(
  all(
    target_arch = "aarch64",
    target_feature = "aes",
    target_feature = "crc",
    target_feature = "sha3"
  ),
  all(
    target_arch = "aarch64",
    target_feature = "aes",
    target_feature = "crc",
    not(target_feature = "sha3")
  ),
  all(target_arch = "aarch64", target_feature = "crc", not(target_feature = "aes")),
  all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vl",
    target_feature = "avx512bw",
    target_feature = "vpclmulqdq",
    target_feature = "pclmulqdq"
  ),
  all(target_arch = "x86_64", target_feature = "pclmulqdq", target_feature = "ssse3"),
  all(target_arch = "x86_64", target_feature = "sse4.2"),
  target_arch = "wasm32",
)))]
pub fn selected_backend() -> &'static str {
  #[cfg(all(feature = "std", target_arch = "aarch64"))]
  {
    if std::arch::is_aarch64_feature_detected!("aes")
      && std::arch::is_aarch64_feature_detected!("crc")
      && std::arch::is_aarch64_feature_detected!("sha3")
    {
      return "aarch64/pmull+eor3 (runtime)";
    }
    if std::arch::is_aarch64_feature_detected!("aes") && std::arch::is_aarch64_feature_detected!("crc") {
      return "aarch64/pmull (runtime)";
    }
    if std::arch::is_aarch64_feature_detected!("crc") {
      return "aarch64/crc (runtime)";
    }
  }

  #[cfg(all(feature = "std", target_arch = "x86_64"))]
  {
    use platform::x86_64::{MicroArch, detect_microarch};

    let arch = detect_microarch();

    if std::arch::is_x86_feature_detected!("vpclmulqdq")
      && std::arch::is_x86_feature_detected!("avx512f")
      && std::arch::is_x86_feature_detected!("avx512vl")
      && std::arch::is_x86_feature_detected!("avx512bw")
      && std::arch::is_x86_feature_detected!("pclmulqdq")
    {
      // AMD Zen 4/5 use hybrid scalar+SIMD for better throughput.
      if std::arch::is_x86_feature_detected!("sse4.2") {
        match arch {
          MicroArch::Zen5 => return "x86_64/hybrid-zen5 (7-way crc32q + vpclmul)",
          MicroArch::Zen4 => return "x86_64/hybrid-zen4 (3-way crc32q + vpclmul)",
          _ => {}
        }
      }
      return "x86_64/vpclmul (runtime)";
    }
    if std::arch::is_x86_feature_detected!("pclmulqdq") && std::arch::is_x86_feature_detected!("ssse3") {
      return "x86_64/pclmul (runtime)";
    }
    if std::arch::is_x86_feature_detected!("sse4.2") {
      return "x86_64/sse4.2 (runtime)";
    }
  }

  "portable/table"
}

#[cfg(feature = "std")]
impl std::io::Write for Crc32c {
  #[inline]
  fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
    self.update(buf);
    Ok(buf.len())
  }

  #[inline]
  fn flush(&mut self) -> std::io::Result<()> {
    Ok(())
  }
}

/// Dispatch to the fastest available implementation.
///
/// This function selects the optimal implementation based on available
/// hardware features, detected at compile time or runtime.
#[inline]
fn dispatch(crc: u32, data: &[u8]) -> u32 {
  // Tier 1: Compile-time target features.
  #[cfg(all(
    target_arch = "aarch64",
    target_feature = "aes",
    target_feature = "crc",
    target_feature = "sha3"
  ))]
  {
    // Mirrors the best-known Apple M-series strategy:
    // - very small: CRC32 extension only
    // - small/medium: PMULL baseline
    // - large: PMULL+EOR3 fusion (v9s3x2e_s3)
    if data.len() < 256 {
      return aarch64::compute_crc_enabled(crc, data);
    }
    if data.len() <= 1024 {
      return crate::simd::aarch64::pmull::compute_pmull_enabled(crc, data);
    }
    crate::simd::aarch64::pmull::compute_pmull_eor3_enabled(crc, data)
  }

  #[cfg(all(
    target_arch = "aarch64",
    target_feature = "aes",
    target_feature = "crc",
    not(target_feature = "sha3")
  ))]
  {
    if data.len() < 256 {
      return aarch64::compute_crc_enabled(crc, data);
    }
    crate::simd::aarch64::pmull::compute_pmull_enabled(crc, data)
  }

  #[cfg(all(target_arch = "aarch64", target_feature = "crc", not(target_feature = "aes")))]
  {
    aarch64::compute_crc_enabled(crc, data)
  }

  // Everything below is only relevant when `target_feature="crc"` is not
  // enabled at compile time on aarch64.
  #[cfg(not(all(target_arch = "aarch64", target_feature = "crc")))]
  {
    // Tier 1: compile-time x86_64 target features.
    #[cfg(all(
      target_arch = "x86_64",
      target_feature = "avx512f",
      target_feature = "avx512vl",
      target_feature = "avx512bw",
      target_feature = "vpclmulqdq",
      target_feature = "pclmulqdq"
    ))]
    {
      // On x86, the dedicated CRC32 instruction is excellent for small buffers,
      // even when a folding engine is available.
      #[cfg(target_feature = "sse4.2")]
      {
        if data.len() < 256 {
          return x86_64::compute_sse42_enabled(crc, data);
        }
      }
      crate::simd::x86_64::vpclmul::compute_vpclmul_enabled(crc, data)
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq", target_feature = "ssse3"))]
    {
      #[cfg(target_feature = "sse4.2")]
      {
        if data.len() < 256 {
          return x86_64::compute_sse42_enabled(crc, data);
        }
      }
      crate::simd::x86_64::pclmul::compute_pclmul_enabled(crc, data)
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse4.2"))]
    {
      x86_64::compute_sse42_enabled(crc, data)
    }

    // Tier 2: Runtime detection (std only), cached with OnceLock.
    #[cfg(all(feature = "std", target_arch = "aarch64", not(target_feature = "crc")))]
    {
      use std::sync::OnceLock;
      static DISPATCH: OnceLock<fn(u32, &[u8]) -> u32> = OnceLock::new();
      let f = DISPATCH.get_or_init(crate::simd::aarch64::detect_crc32c_best);
      f(crc, data)
    }

    #[cfg(all(feature = "std", target_arch = "x86_64", not(target_feature = "sse4.2")))]
    {
      use std::sync::OnceLock;
      static DISPATCH: OnceLock<fn(u32, &[u8]) -> u32> = OnceLock::new();
      let f = DISPATCH.get_or_init(crate::simd::x86_64::detect_crc32c_best);
      f(crc, data)
    }

    // Tier 3: wasm32 with parallel streams optimization.
    #[cfg(target_arch = "wasm32")]
    {
      crate::simd::wasm32::compute_crc32c(crc, data)
    }

    // Tier 4: Portable fallback.
    #[cfg(not(any(
      all(feature = "std", target_arch = "aarch64", not(target_feature = "crc")),
      all(feature = "std", target_arch = "x86_64", not(target_feature = "sse4.2")),
      target_arch = "wasm32",
    )))]
    {
      portable::compute(crc, data)
    }
  }
}

#[cfg(test)]
mod tests {
  extern crate std;

  use alloc::vec::Vec;

  use super::*;

  #[test]
  fn test_checksum() {
    assert_eq!(Crc32c::checksum(b"123456789"), 0xE306_9283);
  }

  #[test]
  fn test_empty() {
    assert_eq!(Crc32c::checksum(b""), 0x0000_0000);
  }

  #[test]
  fn test_zeros() {
    assert_eq!(Crc32c::checksum(&[0u8; 32]), 0x8A91_36AA);
  }

  #[test]
  fn test_ones() {
    assert_eq!(Crc32c::checksum(&[0xFFu8; 32]), 0x62A8_AB43);
  }

  #[test]
  fn test_incremental() {
    let mut hasher = Crc32c::new();
    hasher.update(b"1234");
    hasher.update(b"56789");
    assert_eq!(hasher.finalize(), 0xE306_9283);
  }

  #[test]
  fn test_resume() {
    let data = b"hello world";
    let (first, second) = data.split_at(6);

    let crc1 = Crc32c::checksum(first);
    let mut hasher = Crc32c::resume(crc1);
    hasher.update(second);

    assert_eq!(hasher.finalize(), Crc32c::checksum(data));
  }

  #[test]
  fn test_reset() {
    let mut hasher = Crc32c::new();
    hasher.update(b"garbage");
    hasher.reset();
    hasher.update(b"123456789");
    assert_eq!(hasher.finalize(), 0xE306_9283);
  }

  #[test]
  fn test_clone() {
    let mut hasher = Crc32c::new();
    hasher.update(b"1234");

    let mut clone = hasher.clone();
    hasher.update(b"56789");
    clone.update(b"56789");

    assert_eq!(hasher.finalize(), clone.finalize());
  }

  #[test]
  fn test_trait_impl() {
    fn check_trait<T: Checksum>() {}
    fn check_combine<T: ChecksumCombine>() {}

    check_trait::<Crc32c>();
    check_combine::<Crc32c>();
  }

  #[cfg(feature = "std")]
  fn gen_bytes(len: usize, seed: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(len);
    let mut x = seed;
    for _ in 0..len {
      // xorshift64*
      x ^= x << 13;
      x ^= x >> 7;
      x ^= x << 17;
      out.push((x as u8).wrapping_add((x >> 8) as u8));
    }
    out
  }

  #[test]
  #[cfg(all(feature = "std", target_arch = "x86_64"))]
  fn test_simd_matches_portable_x86() {
    let lengths = [0usize, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 255, 256, 1024];
    let inits = [0u32, 0xFFFF_FFFFu32, 0x0123_4567u32];

    for &len in &lengths {
      let data = gen_bytes(len, len as u64 ^ 0x9E37_79B9_7F4A_7C15);
      for &init in &inits {
        let expected = portable::compute(init, &data);

        if std::arch::is_x86_feature_detected!("sse4.2") {
          let got = x86_64::compute_sse42_runtime(init, &data);
          assert_eq!(got, expected, "sse4.2 mismatch at len={}", len);
        }

        if std::arch::is_x86_feature_detected!("pclmulqdq") && std::arch::is_x86_feature_detected!("ssse3") {
          let got = crate::simd::x86_64::pclmul::compute_pclmul_runtime(init, &data);
          assert_eq!(got, expected, "pclmul mismatch at len={}", len);
        }

        if std::arch::is_x86_feature_detected!("vpclmulqdq")
          && std::arch::is_x86_feature_detected!("avx512f")
          && std::arch::is_x86_feature_detected!("avx512vl")
          && std::arch::is_x86_feature_detected!("avx512bw")
          && std::arch::is_x86_feature_detected!("pclmulqdq")
        {
          let got = crate::simd::x86_64::vpclmul::compute_vpclmul_runtime(init, &data);
          assert_eq!(got, expected, "vpclmul mismatch at len={}", len);

          // Also test hybrid kernels (they should match portable for all sizes).
          if std::arch::is_x86_feature_detected!("sse4.2") {
            let got_zen4 = crate::simd::x86_64::hybrid::compute_hybrid_zen4_runtime(init, &data);
            assert_eq!(got_zen4, expected, "hybrid-zen4 mismatch at len={}", len);

            let got_zen5 = crate::simd::x86_64::hybrid::compute_hybrid_zen5_runtime(init, &data);
            assert_eq!(got_zen5, expected, "hybrid-zen5 mismatch at len={}", len);
          }
        }
      }
    }
  }

  #[test]
  #[cfg(all(feature = "std", target_arch = "aarch64"))]
  fn test_simd_matches_portable_aarch64() {
    let lengths = [0usize, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 255, 256, 1024];
    let inits = [0u32, 0xFFFF_FFFFu32, 0x89AB_CDEFu32];

    for &len in &lengths {
      let data = gen_bytes(len, len as u64 ^ 0xD1B5_4A32_D192_ED03);
      for &init in &inits {
        let expected = portable::compute(init, &data);

        if std::arch::is_aarch64_feature_detected!("crc") {
          #[cfg(target_feature = "crc")]
          let got = aarch64::compute_crc_enabled(init, &data);
          #[cfg(not(target_feature = "crc"))]
          let got = aarch64::compute_crc_runtime(init, &data);
          assert_eq!(got, expected, "crc ext mismatch at len={}", len);
        }

        if std::arch::is_aarch64_feature_detected!("aes") && std::arch::is_aarch64_feature_detected!("crc") {
          // If this binary is compiled with `sha3`, it will always execute the EOR3 codegen.
          #[cfg(all(target_feature = "aes", target_feature = "crc", target_feature = "sha3"))]
          {
            let got = crate::simd::aarch64::pmull::compute_pmull_eor3_enabled(init, &data);
            assert_eq!(got, expected, "pmull+eor3 mismatch at len={}", len);
          }

          #[cfg(not(all(target_feature = "aes", target_feature = "crc", target_feature = "sha3")))]
          {
            if std::arch::is_aarch64_feature_detected!("sha3") {
              let got = crate::simd::aarch64::pmull::compute_pmull_eor3_runtime(init, &data);
              assert_eq!(got, expected, "pmull+eor3 mismatch at len={}", len);
            } else {
              #[cfg(all(target_feature = "aes", target_feature = "crc"))]
              let got = crate::simd::aarch64::pmull::compute_pmull_enabled(init, &data);
              #[cfg(not(all(target_feature = "aes", target_feature = "crc")))]
              let got = crate::simd::aarch64::pmull::compute_pmull_runtime(init, &data);
              assert_eq!(got, expected, "pmull mismatch at len={}", len);
            }
          }
        }
      }
    }
  }
}
