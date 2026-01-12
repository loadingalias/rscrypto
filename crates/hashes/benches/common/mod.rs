use core::hint::black_box;

/// Deterministic, fast pseudo-random generator suitable for benchmarks.
///
/// This is *not* cryptographically secure; it's only used to avoid unrealistic
/// all-zero / highly-structured benchmark inputs.
#[inline]
fn xorshift64star(state: &mut u64) -> u64 {
  let mut x = *state;
  x ^= x >> 12;
  x ^= x << 25;
  x ^= x >> 27;
  *state = x;
  x.wrapping_mul(0x2545F4914F6CDD1D)
}

pub fn pseudo_random_bytes(len: usize, seed: u64) -> Vec<u8> {
  let mut state = seed ^ (len as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
  let mut out = vec![0u8; len];
  for b in &mut out {
    *b = (xorshift64star(&mut state) >> 56) as u8;
  }
  // Help ensure the compiler can't assume anything about the contents.
  black_box(&out);
  out
}

pub fn sized_inputs() -> Vec<(usize, Vec<u8>)> {
  // Includes a few edge cases and a selection of "real-world-ish" payload sizes.
  let sizes = [
    0usize,
    1,
    3,
    8,
    16,
    31,
    32,
    63,
    64,
    65,
    128,
    256,
    1024,
    4 * 1024,
    16 * 1024,
    64 * 1024,
    1024 * 1024,
  ];
  sizes
    .into_iter()
    .map(|len| (len, pseudo_random_bytes(len, 0xD1CE_B00C_D15C_0FFE)))
    .collect()
}

pub fn set_throughput(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>, len: usize) {
  if len == 0 {
    group.throughput(criterion::Throughput::Elements(1));
  } else {
    group.throughput(criterion::Throughput::Bytes(len as u64));
  }
}
