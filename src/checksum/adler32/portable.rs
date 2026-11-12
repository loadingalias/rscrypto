//! Portable Adler-32 kernel: zlib's loop with deferred reduction.

/// The largest prime below 2^16.
pub(super) const MOD: u32 = 65_521;
pub(super) const MOD64: u64 = 65_521;

/// Canonical name of the portable kernel.
pub(in crate::checksum) const KERNEL_NAME: &str = "portable/unroll16";

/// Bytes of the unrolled inner step.
const STEP: usize = 16;
/// zlib's NMAX: the most bytes between two reductions.
const NMAX: usize = 5552;
/// From reduced sums, `n` bytes of 0xff raise `b` to its largest value,
/// `255·n·(n + 1)/2 + (n + 1)·(MOD - 1)`, which stays at most `u32::MAX` for `n = NMAX`.
const NMAX_FITS: () = {
  let n = NMAX as u64;
  assert!(255 * n * (n + 1) / 2 + (n + 1) * (MOD64 - 1) <= u32::MAX as u64);
};

/// Add `data` to the Adler-32 value `adler`, whose two sums are reduced.
///
/// `a` and `b` stay in registers, so each byte costs one load and two adds. Both sums
/// reduce once per `NMAX` bytes.
pub(in crate::checksum) fn update(adler: u32, data: &[u8]) -> u32 {
  let mut sums = (adler & 0xffff, adler >> 16);
  for run in data.chunks(NMAX) {
    let (steps, tail) = run.as_chunks::<STEP>();
    for step in steps {
      sums = add(sums, step);
    }
    let (a, b) = add(sums, tail);
    sums = (a % MOD, b % MOD);
  }
  (sums.1 << 16) | sums.0
}

/// Add `bytes` to the sums `(a, b)`. The bytes since their last reduction total at most `NMAX`.
#[inline(always)]
fn add((mut a, mut b): (u32, u32), bytes: &[u8]) -> (u32, u32) {
  // `NMAX_FITS` bounds both sums of a run by `u32::MAX`, so each wrapping add equals the
  // exact add.
  let () = NMAX_FITS;
  for &byte in bytes {
    a = a.wrapping_add(u32::from(byte));
    b = b.wrapping_add(a);
  }
  (a, b)
}

/// Reduce `value` modulo 65521.
#[inline(always)]
pub(super) fn reduce(value: u64) -> u32 {
  let [r0, r1, r2, r3, ..] = (value % MOD64).to_le_bytes();
  u32::from_le_bytes([r0, r1, r2, r3])
}
