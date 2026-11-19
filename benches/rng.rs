//! `ChaCha20Rng` generation benchmarks against the `chacha20` crate's generator.
//!
//! Both generators run the 20-round ChaCha block function from a seed. rscrypto's
//! generator also rekeys from its own output and clears each byte that it
//! returns, so it does more work per byte. The two streams differ after the
//! first block, so each row checks the first 64 bytes of the timed generator
//! before timing: from the zero seed, both give RFC 8439 Appendix A.1 test
//! vector 1, the keystream block of the zero key, zero nonce, and counter 0.

#[path = "common/criterion.rs"]
#[expect(dead_code, reason = "each row checks the first block of the generator that it times")]
mod bench_config;

use core::hint::black_box;

use bench_config::checked_routine;
use criterion::{BenchmarkId, Criterion, Throughput};
use rand_core::{Rng as _, SeedableRng as _};

/// RFC 8439 Appendix A.1, test vector 1.
const BLOCK_ZERO: [u8; 64] = [
  0x76, 0xb8, 0xe0, 0xad, 0xa0, 0xf1, 0x3d, 0x90, 0x40, 0x5d, 0x6a, 0xe5, 0x53, 0x86, 0xbd, 0x28, 0xbd, 0xd2, 0x19,
  0xb8, 0xa0, 0x8d, 0xed, 0x1a, 0xa8, 0x36, 0xef, 0xcc, 0x8b, 0x77, 0x0d, 0xc7, 0xda, 0x41, 0x59, 0x7c, 0x51, 0x57,
  0x48, 0x8d, 0x77, 0x24, 0xe0, 0x3f, 0xb8, 0xd8, 0x4a, 0x37, 0x6a, 0x43, 0xb8, 0xf4, 0x15, 0x18, 0xa1, 0x1c, 0xc3,
  0x87, 0xb6, 0x69, 0xb2, 0xee, 0x65, 0x86,
];
/// Fill lengths: a 128-bit token, a key, one block, a record, and a large buffer.
const FILL_LENGTHS: [usize; 5] = [16, 32, 64, 1024, 16384];

/// Times `draw` on a generator from the zero seed, after its first 64 bytes
/// match [`BLOCK_ZERO`].
fn row<R, T>(
  rng: R,
  fill: fn(&mut R, &mut [u8]),
  mut draw: impl FnMut(&mut R) -> T,
) -> impl FnMut(&mut criterion::Bencher<'_>) {
  checked_routine(
    rng,
    move |rng| {
      let mut first = [0u8; 64];
      fill(rng, &mut first);
      assert_eq!(first, BLOCK_ZERO);
    },
    move |b, rng| b.iter(|| draw(black_box(&mut *rng))),
  )
}

fn rscrypto_rng() -> rscrypto::ChaCha20Rng {
  rscrypto::ChaCha20Rng::from_seed([0; 32])
}

fn chacha20_rng() -> chacha20::ChaCha20Rng {
  chacha20::ChaCha20Rng::from_seed([0; 32])
}

fn rscrypto_fill(rng: &mut rscrypto::ChaCha20Rng, out: &mut [u8]) {
  rng.fill_bytes(out);
}

fn chacha20_fill(rng: &mut chacha20::ChaCha20Rng, out: &mut [u8]) {
  rng.fill_bytes(out);
}

fn chacha20_rng_fill(c: &mut Criterion) {
  if !bench_config::selected("chacha20-rng/fill/") {
    return;
  }
  let mut group = c.benchmark_group("chacha20-rng/fill");
  for len in FILL_LENGTHS {
    group.throughput(Throughput::Bytes(len as u64));
    let mut ours = vec![0u8; len];
    group.bench_function(
      BenchmarkId::new("rscrypto", len),
      row(rscrypto_rng(), rscrypto_fill, move |rng| rng.fill_bytes(&mut ours)),
    );
    let mut theirs = vec![0u8; len];
    group.bench_function(
      BenchmarkId::new("chacha20", len),
      row(chacha20_rng(), chacha20_fill, move |rng| rng.fill_bytes(&mut theirs)),
    );
  }
  group.finish();
}

fn chacha20_rng_words(c: &mut Criterion) {
  if !bench_config::selected("chacha20-rng/next/") {
    return;
  }
  let mut group = c.benchmark_group("chacha20-rng/next");
  group.bench_function(
    BenchmarkId::new("rscrypto", "u32"),
    row(rscrypto_rng(), rscrypto_fill, |rng| rng.next_u32()),
  );
  group.bench_function(
    BenchmarkId::new("chacha20", "u32"),
    row(chacha20_rng(), chacha20_fill, |rng| rng.next_u32()),
  );
  group.bench_function(
    BenchmarkId::new("rscrypto", "u64"),
    row(rscrypto_rng(), rscrypto_fill, |rng| rng.next_u64()),
  );
  group.bench_function(
    BenchmarkId::new("chacha20", "u64"),
    row(chacha20_rng(), chacha20_fill, |rng| rng.next_u64()),
  );
  group.finish();
}

fn main() {
  bench_config::run(&[chacha20_rng_fill, chacha20_rng_words]);
}
