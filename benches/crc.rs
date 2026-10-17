//! CRC comparison benchmarks: rscrypto vs competitor crates.
//!
//! Before timing, every row must return the reference crate's checksum for its input. Combination rows must return
//! the reference crate's checksum of both parts concatenated.

#[path = "common/criterion.rs"]
mod bench_config;

mod common;

use core::{fmt::Debug, hint::black_box};

use bench_config::checked;
use crc_fast::CrcAlgorithm;
use criterion::{BenchmarkId, Criterion, Throughput};
use rscrypto::{Checksum, ChecksumCombine};

fn crc32_ieee(c: &mut Criterion) {
  if !bench_config::selected("crc32") {
    return;
  }
  let inputs = common::comp_sizes(&[]);
  let mut g = c.benchmark_group("crc32");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    let expected = crc32fast::hash(data);

    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked(
        || rscrypto::Crc32::checksum(black_box(data)),
        |out| assert_eq!(out, expected),
      ),
    );
    g.bench_function(
      BenchmarkId::new("crc32fast", len),
      checked(|| crc32fast::hash(black_box(data)), |out| assert_eq!(out, expected)),
    );
    g.bench_function(
      BenchmarkId::new("crc-fast", len),
      checked(
        || crc_fast::crc32_iso_hdlc(black_box(data)),
        |out| assert_eq!(out, expected),
      ),
    );
  }

  g.finish();
}

fn crc32c(c: &mut Criterion) {
  if !bench_config::selected("crc32c") {
    return;
  }
  let inputs = common::comp_sizes(&[]);
  let mut g = c.benchmark_group("crc32c");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    let expected = crc32c::crc32c(data);

    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked(
        || rscrypto::Crc32C::checksum(black_box(data)),
        |out| assert_eq!(out, expected),
      ),
    );
    g.bench_function(
      BenchmarkId::new("crc32c", len),
      checked(|| crc32c::crc32c(black_box(data)), |out| assert_eq!(out, expected)),
    );
    g.bench_function(
      BenchmarkId::new("crc-fast", len),
      checked(
        || crc_fast::crc32_iscsi(black_box(data)),
        |out| assert_eq!(out, expected),
      ),
    );
  }

  g.finish();
}

fn crc64_xz(c: &mut Criterion) {
  if !bench_config::selected("crc64-xz") {
    return;
  }
  let inputs = common::comp_sizes(&[]);
  let crc64fast = |data: &[u8]| {
    let mut h = crc64fast::Digest::new();
    h.write(data);
    h.sum64()
  };
  let mut g = c.benchmark_group("crc64-xz");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    let expected = crc64fast(data);

    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked(
        || rscrypto::Crc64::checksum(black_box(data)),
        |out| assert_eq!(out, expected),
      ),
    );
    g.bench_function(
      BenchmarkId::new("crc64fast", len),
      checked(|| crc64fast(black_box(data)), |out| assert_eq!(out, expected)),
    );
  }

  g.finish();
}

fn crc64_nvme(c: &mut Criterion) {
  if !bench_config::selected("crc64-nvme") {
    return;
  }
  let inputs = common::comp_sizes(&[]);
  let mut g = c.benchmark_group("crc64-nvme");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    let expected = crc_fast::crc64_nvme(data);

    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked(
        || rscrypto::Crc64Nvme::checksum(black_box(data)),
        |out| assert_eq!(out, expected),
      ),
    );
    g.bench_function(
      BenchmarkId::new("crc-fast", len),
      checked(
        || crc_fast::crc64_nvme(black_box(data)),
        |out| assert_eq!(out, expected),
      ),
    );
  }

  g.finish();
}

fn crc16(c: &mut Criterion) {
  if !bench_config::selected("crc16-ccitt") {
    return;
  }
  let inputs = common::comp_sizes(&[]);
  let crc_algo = crc::Crc::<u16>::new(&crc::CRC_16_IBM_SDLC);
  let mut g = c.benchmark_group("crc16-ccitt");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    let expected = crc_algo.checksum(data);

    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked(
        || rscrypto::Crc16Ccitt::checksum(black_box(data)),
        |out| assert_eq!(out, expected),
      ),
    );
    g.bench_function(
      BenchmarkId::new("crc", len),
      checked(|| crc_algo.checksum(black_box(data)), |out| assert_eq!(out, expected)),
    );
  }

  g.finish();
}

fn crc16_ibm(c: &mut Criterion) {
  if !bench_config::selected("crc16-ibm") {
    return;
  }
  let inputs = common::comp_sizes(&[]);
  let crc_algo = crc::Crc::<u16>::new(&crc::CRC_16_ARC);
  let mut g = c.benchmark_group("crc16-ibm");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    let expected = crc_algo.checksum(data);

    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked(
        || rscrypto::Crc16Ibm::checksum(black_box(data)),
        |out| assert_eq!(out, expected),
      ),
    );
    g.bench_function(
      BenchmarkId::new("crc", len),
      checked(|| crc_algo.checksum(black_box(data)), |out| assert_eq!(out, expected)),
    );
  }

  g.finish();
}

fn crc24(c: &mut Criterion) {
  if !bench_config::selected("crc24-openpgp") {
    return;
  }
  let inputs = common::comp_sizes(&[]);
  let crc_algo = crc::Crc::<u32>::new(&crc::CRC_24_OPENPGP);
  let mut g = c.benchmark_group("crc24-openpgp");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    let expected = crc_algo.checksum(data);

    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked(
        || rscrypto::Crc24OpenPgp::checksum(black_box(data)),
        |out| assert_eq!(out, expected),
      ),
    );
    g.bench_function(
      BenchmarkId::new("crc", len),
      checked(|| crc_algo.checksum(black_box(data)), |out| assert_eq!(out, expected)),
    );
  }

  g.finish();
}

/// Streaming input length and update sizes: 1 MiB through updates of one cache line or one page.
const STREAM_LEN: usize = 1 << 20;
const STREAM_CHUNKS: [usize; 2] = [64, 4096];

/// One library's streaming checksum: construct a state, update it with `data` in chunks of the given size, and
/// finalize.
type StreamRow<'a, T> = (&'a str, &'a dyn Fn(&[u8], usize) -> T);

/// Registers `group`: 1 MiB through updates of each chunk size. Timed: state construction, every update, and
/// finalization. Every row must return `reference`'s one-shot checksum.
fn streaming_group<T: Copy + Debug + PartialEq>(
  c: &mut Criterion,
  group: &str,
  reference: impl Fn(&[u8]) -> T,
  rows: &[StreamRow<'_, T>],
) {
  if !bench_config::selected(group) {
    return;
  }
  let data = common::random_bytes(STREAM_LEN);
  let expected = reference(&data);
  let mut g = c.benchmark_group(group);
  g.throughput(Throughput::Bytes(STREAM_LEN as u64));
  for chunk_size in STREAM_CHUNKS {
    for &(library, stream) in rows {
      g.bench_function(
        format!("{library}/{chunk_size}B"),
        checked(|| stream(black_box(&data), chunk_size), |out| assert_eq!(out, expected)),
      );
    }
  }
  g.finish();
}

/// Lengths of the second part for combination: a short packet, and one below 64 KiB and 16 MiB. Their set bits make
/// square-and-multiply combination do the most multiplications for their bit length.
const COMBINE_LENGTHS: [usize; 3] = [64, 65_535, 16_777_215];
/// Length of the first part, which combination does not read.
const COMBINE_PREFIX_LEN: usize = 1024;

/// One library's combination of `crc(A)` and `crc(B)` into `crc(A || B)`, given `len(B)`.
type CombineRow<'a, T> = (&'a str, &'a dyn Fn(T, T, usize) -> T);

/// Registers `group`: combine the checksums of two parts. Timed: the combination, including any state that the
/// library's API needs. Every row must return `reference`'s checksum of the concatenated parts.
fn combine_group<T: Copy + Debug + PartialEq>(
  c: &mut Criterion,
  group: &str,
  reference: impl Fn(&[u8]) -> T,
  rows: &[CombineRow<'_, T>],
) {
  if !bench_config::selected(group) {
    return;
  }
  let prefix = common::random_bytes(COMBINE_PREFIX_LEN);
  let crc_a = reference(&prefix);
  let mut g = c.benchmark_group(group);
  for len_b in COMBINE_LENGTHS {
    let suffix = common::random_bytes(len_b);
    let crc_b = reference(&suffix);
    let expected = reference(&[prefix.as_slice(), suffix.as_slice()].concat());
    for &(library, combine) in rows {
      g.bench_function(
        BenchmarkId::new(library, len_b),
        checked(
          || combine(black_box(crc_a), black_box(crc_b), black_box(len_b)),
          |out| assert_eq!(out, expected),
        ),
      );
    }
  }
  g.finish();
}

/// Streams `data` through `update` calls of `chunk_size` bytes.
fn rscrypto_stream<C: Checksum>(data: &[u8], chunk_size: usize) -> C::Output {
  let mut state = C::new();
  for chunk in data.chunks(chunk_size) {
    state.update(black_box(chunk));
  }
  state.finalize()
}

/// Streams `data` through crc-fast's `Digest` in `chunk_size` updates.
fn crc_fast_stream(algorithm: CrcAlgorithm, data: &[u8], chunk_size: usize) -> u64 {
  let mut digest = crc_fast::Digest::new(algorithm);
  for chunk in data.chunks(chunk_size) {
    digest.update(black_box(chunk));
  }
  digest.finalize()
}

/// Narrows crc-fast's 64-bit result to a checksum width; crc-fast keeps narrower results in the low bits.
fn narrow<T: TryFrom<u64>>(value: u64) -> T
where
  T::Error: Debug,
{
  T::try_from(value).expect("crc-fast keeps a narrow checksum in the low bits")
}

fn crc_streaming(c: &mut Criterion) {
  let crc16_ccitt = crc::Crc::<u16>::new(&crc::CRC_16_IBM_SDLC);
  let crc16_ibm = crc::Crc::<u16>::new(&crc::CRC_16_ARC);
  let crc24 = crc::Crc::<u32>::new(&crc::CRC_24_OPENPGP);
  let crc64fast = |data: &[u8]| {
    let mut digest = crc64fast::Digest::new();
    digest.write(data);
    digest.sum64()
  };

  streaming_group(
    c,
    "crc32/streaming",
    crc32fast::hash,
    &[
      ("rscrypto", &rscrypto_stream::<rscrypto::Crc32>),
      ("crc32fast", &|data, chunk_size| {
        let mut hasher = crc32fast::Hasher::new();
        for chunk in data.chunks(chunk_size) {
          hasher.update(black_box(chunk));
        }
        hasher.finalize()
      }),
      ("crc-fast", &|data, chunk_size| {
        narrow(crc_fast_stream(CrcAlgorithm::Crc32IsoHdlc, data, chunk_size))
      }),
    ],
  );
  streaming_group(
    c,
    "crc32c/streaming",
    crc32c::crc32c,
    &[
      ("rscrypto", &rscrypto_stream::<rscrypto::Crc32C>),
      ("crc32c", &|data, chunk_size| {
        data
          .chunks(chunk_size)
          .fold(0, |crc, chunk| crc32c::crc32c_append(crc, black_box(chunk)))
      }),
      ("crc-fast", &|data, chunk_size| {
        narrow(crc_fast_stream(CrcAlgorithm::Crc32Iscsi, data, chunk_size))
      }),
    ],
  );
  streaming_group(
    c,
    "crc64-xz/streaming",
    crc64fast,
    &[
      ("rscrypto", &rscrypto_stream::<rscrypto::Crc64>),
      ("crc64fast", &|data, chunk_size| {
        let mut digest = crc64fast::Digest::new();
        for chunk in data.chunks(chunk_size) {
          digest.write(black_box(chunk));
        }
        digest.sum64()
      }),
    ],
  );
  streaming_group(
    c,
    "crc64-nvme/streaming",
    crc_fast::crc64_nvme,
    &[
      ("rscrypto", &rscrypto_stream::<rscrypto::Crc64Nvme>),
      ("crc-fast", &|data, chunk_size| {
        crc_fast_stream(CrcAlgorithm::Crc64Nvme, data, chunk_size)
      }),
    ],
  );
  streaming_group(
    c,
    "crc16-ccitt/streaming",
    |data| crc16_ccitt.checksum(data),
    &[
      ("rscrypto", &rscrypto_stream::<rscrypto::Crc16Ccitt>),
      ("crc", &|data, chunk_size| {
        let mut digest = crc16_ccitt.digest();
        for chunk in data.chunks(chunk_size) {
          digest.update(black_box(chunk));
        }
        digest.finalize()
      }),
    ],
  );
  streaming_group(
    c,
    "crc16-ibm/streaming",
    |data| crc16_ibm.checksum(data),
    &[
      ("rscrypto", &rscrypto_stream::<rscrypto::Crc16Ibm>),
      ("crc", &|data, chunk_size| {
        let mut digest = crc16_ibm.digest();
        for chunk in data.chunks(chunk_size) {
          digest.update(black_box(chunk));
        }
        digest.finalize()
      }),
    ],
  );
  streaming_group(
    c,
    "crc24-openpgp/streaming",
    |data| crc24.checksum(data),
    &[
      ("rscrypto", &rscrypto_stream::<rscrypto::Crc24OpenPgp>),
      ("crc", &|data, chunk_size| {
        let mut digest = crc24.digest();
        for chunk in data.chunks(chunk_size) {
          digest.update(black_box(chunk));
        }
        digest.finalize()
      }),
    ],
  );
}

/// crc-fast's combination for one algorithm, narrowed to the checksum width.
fn crc_fast_combine<T: Into<u64> + TryFrom<u64>>(algorithm: CrcAlgorithm, crc_a: T, crc_b: T, len_b: usize) -> T
where
  T::Error: Debug,
{
  let len_b = u64::try_from(len_b).expect("combination lengths fit u64");
  narrow(crc_fast::checksum_combine(algorithm, crc_a.into(), crc_b.into(), len_b))
}

// CRC-24/OpenPGP has no comparison library with combination; its rscrypto row keeps the API measured.
fn crc_combine(c: &mut Criterion) {
  let crc16_ccitt = crc::Crc::<u16>::new(&crc::CRC_16_IBM_SDLC);
  let crc16_ibm = crc::Crc::<u16>::new(&crc::CRC_16_ARC);
  let crc24 = crc::Crc::<u32>::new(&crc::CRC_24_OPENPGP);
  let crc64fast = |data: &[u8]| {
    let mut digest = crc64fast::Digest::new();
    digest.write(data);
    digest.sum64()
  };

  combine_group(
    c,
    "crc32/combine",
    crc32fast::hash,
    &[
      ("rscrypto", &rscrypto::Crc32::combine),
      ("crc32fast", &|crc_a, crc_b, len_b| {
        // crc32fast combines hasher states that carry their checksum and length.
        let len_a = u64::try_from(COMBINE_PREFIX_LEN).expect("combination lengths fit u64");
        let len_b = u64::try_from(len_b).expect("combination lengths fit u64");
        let mut hasher = crc32fast::Hasher::new_with_initial_len(crc_a, len_a);
        hasher.combine(&crc32fast::Hasher::new_with_initial_len(crc_b, len_b));
        hasher.finalize()
      }),
      ("crc-fast", &|crc_a, crc_b, len_b| {
        crc_fast_combine(CrcAlgorithm::Crc32IsoHdlc, crc_a, crc_b, len_b)
      }),
    ],
  );
  combine_group(
    c,
    "crc32c/combine",
    crc32c::crc32c,
    &[
      ("rscrypto", &rscrypto::Crc32C::combine),
      ("crc32c", &crc32c::crc32c_combine),
      ("crc-fast", &|crc_a, crc_b, len_b| {
        crc_fast_combine(CrcAlgorithm::Crc32Iscsi, crc_a, crc_b, len_b)
      }),
    ],
  );
  combine_group(
    c,
    "crc64-xz/combine",
    crc64fast,
    &[
      ("rscrypto", &rscrypto::Crc64::combine),
      ("crc-fast", &|crc_a, crc_b, len_b| {
        crc_fast_combine(CrcAlgorithm::Crc64Xz, crc_a, crc_b, len_b)
      }),
    ],
  );
  combine_group(
    c,
    "crc64-nvme/combine",
    crc_fast::crc64_nvme,
    &[
      ("rscrypto", &rscrypto::Crc64Nvme::combine),
      ("crc-fast", &|crc_a, crc_b, len_b| {
        crc_fast_combine(CrcAlgorithm::Crc64Nvme, crc_a, crc_b, len_b)
      }),
    ],
  );
  combine_group(
    c,
    "crc16-ccitt/combine",
    |data| crc16_ccitt.checksum(data),
    &[
      ("rscrypto", &rscrypto::Crc16Ccitt::combine),
      ("crc-fast", &|crc_a, crc_b, len_b| {
        crc_fast_combine(CrcAlgorithm::Crc16IbmSdlc, crc_a, crc_b, len_b)
      }),
    ],
  );
  combine_group(
    c,
    "crc16-ibm/combine",
    |data| crc16_ibm.checksum(data),
    &[
      ("rscrypto", &rscrypto::Crc16Ibm::combine),
      ("crc-fast", &|crc_a, crc_b, len_b| {
        crc_fast_combine(CrcAlgorithm::Crc16Arc, crc_a, crc_b, len_b)
      }),
    ],
  );
  combine_group(
    c,
    "crc24-openpgp/combine",
    |data| crc24.checksum(data),
    &[("rscrypto", &rscrypto::Crc24OpenPgp::combine)],
  );
}

fn main() {
  bench_config::run(&[
    crc32_ieee,
    crc32c,
    crc64_xz,
    crc64_nvme,
    crc16,
    crc16_ibm,
    crc24,
    crc_streaming,
    crc_combine,
  ]);
}
