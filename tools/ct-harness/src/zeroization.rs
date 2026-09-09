//! Optimized cleanup sentinel for the production fixed-size secret owner.

/// Consume a production owner without observing its storage after destruction.
///
/// The returned byte keeps secret initialization live. The caller never reads
/// cleared storage, so dead-store elimination remains free to expose a missing
/// volatile wipe. This covers this concrete drop path, not all secret owners.
#[unsafe(no_mangle)]
#[inline(never)]
pub fn zeroize_entry_secret_bytes_32(input: &[u8; 32]) -> u8 {
  // A fixed alignment keeps the sentinel straight-line and reviewable. The
  // production destructor still owns every write; unaligned paths are outside
  // this sentinel's scope.
  #[repr(align(8))]
  struct Aligned(rscrypto::SecretBytes<32>);
  let secret = Aligned(rscrypto::SecretBytes::new(*input));
  core::hint::black_box(&secret).0.as_bytes()[0]
}
