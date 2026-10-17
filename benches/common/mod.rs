use criterion::{BenchmarkGroup, Throughput, measurement::WallTime};

/// Deterministic pseudo-random bytes for reproducible benchmarks.
pub(crate) fn random_bytes(len: usize) -> Vec<u8> {
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

/// Focused size matrix for comparison benchmarks, plus `boundaries`.
///
/// Covers: overhead (0B, 1B), small (32B), block boundary (64B),
/// medium (256B, 1 KiB), page-sized (4 KiB, 16 KiB), in-cache
/// (64 KiB, 256 KiB), throughput (1 MiB). `boundaries` adds the lengths at
/// which one primitive changes regime, such as the longest message whose
/// padding fits one block and the shortest that needs another. Lengths stay
/// in ascending order.
pub(crate) fn comp_sizes(boundaries: &[usize]) -> Vec<(usize, Vec<u8>)> {
  let mut lengths = [0, 1, 32, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576].to_vec();
  lengths.extend_from_slice(boundaries);
  lengths.sort_unstable();
  lengths.dedup();
  lengths.into_iter().map(|len| (len, random_bytes(len))).collect()
}

/// Set criterion throughput for a benchmark group.
pub(crate) fn set_throughput(group: &mut BenchmarkGroup<'_, WallTime>, len: usize) {
  if len > 0 {
    group.throughput(Throughput::Bytes(len as u64));
  }
}
