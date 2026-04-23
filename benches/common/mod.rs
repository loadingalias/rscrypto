use criterion::{BenchmarkGroup, Throughput, measurement::WallTime};

/// Deterministic pseudo-random bytes for reproducible benchmarks.
pub fn random_bytes(len: usize) -> Vec<u8> {
  let mut state: u64 = (len as u64) ^ 0x517c_c1b7_2722_0a95;
  (0..len)
    .map(|_| {
      state ^= state << 13;
      state ^= state >> 7;
      state ^= state << 17;
      state = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
      (state >> 56) as u8
    })
    .collect()
}

/// Focused size matrix for comparison benchmarks.
///
/// Covers: overhead (0B, 1B), small (32B), block boundary (64B),
/// medium (256B, 1 KiB), page-aligned (4 KiB, 16 KiB), in-cache
/// (64 KiB, 256 KiB), throughput (1 MiB).
pub fn comp_sizes() -> Vec<(usize, Vec<u8>)> {
  [0, 1, 32, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576]
    .into_iter()
    .map(|len| (len, random_bytes(len)))
    .collect()
}

/// Set criterion throughput for a benchmark group.
pub fn set_throughput(group: &mut BenchmarkGroup<'_, WallTime>, len: usize) {
  if len > 0 {
    group.throughput(Throughput::Bytes(len as u64));
  }
}
