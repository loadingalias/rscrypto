use criterion::{BenchmarkGroup, measurement::WallTime};

#[cfg(any(
  all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "aes",
    target_feature = "sse2"
  ),
  all(
    any(target_arch = "arm", target_arch = "aarch64"),
    target_feature = "aes",
    target_feature = "neon"
  ),
))]
pub fn bench_gxhash64(group: &mut BenchmarkGroup<'_, WallTime>, len: usize, data: &[u8]) {
  group.bench_with_input(criterion::BenchmarkId::new("gxhash", len), data, |b, d| {
    b.iter(|| core::hint::black_box(gxhash::gxhash64(core::hint::black_box(d), 0)))
  });
}

#[cfg(not(any(
  all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "aes",
    target_feature = "sse2"
  ),
  all(
    any(target_arch = "arm", target_arch = "aarch64"),
    target_feature = "aes",
    target_feature = "neon"
  ),
)))]
pub fn bench_gxhash64(_group: &mut BenchmarkGroup<'_, WallTime>, _len: usize, _data: &[u8]) {}

#[cfg(any(
  all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "aes",
    target_feature = "sse2"
  ),
  all(
    any(target_arch = "arm", target_arch = "aarch64"),
    target_feature = "aes",
    target_feature = "neon"
  ),
))]
pub fn bench_gxhash128(group: &mut BenchmarkGroup<'_, WallTime>, len: usize, data: &[u8]) {
  group.bench_with_input(criterion::BenchmarkId::new("gxhash", len), data, |b, d| {
    b.iter(|| core::hint::black_box(gxhash::gxhash128(core::hint::black_box(d), 0)))
  });
}

#[cfg(not(any(
  all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "aes",
    target_feature = "sse2"
  ),
  all(
    any(target_arch = "arm", target_arch = "aarch64"),
    target_feature = "aes",
    target_feature = "neon"
  ),
)))]
pub fn bench_gxhash128(_group: &mut BenchmarkGroup<'_, WallTime>, _len: usize, _data: &[u8]) {}
