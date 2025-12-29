use core::hint::black_box;
use std::sync::Once;

use checksum::{
  BufferedCrc16Ccitt, BufferedCrc16Ibm, BufferedCrc24OpenPgp, BufferedCrc32, BufferedCrc32C, BufferedCrc64Nvme,
  BufferedCrc64Xz, Checksum, Crc16Ccitt, Crc16Ibm, Crc24OpenPgp, Crc32, Crc32C, Crc64, Crc64Nvme,
};
use crc::Crc as RefCrc;
use crc_fast::{CrcAlgorithm as CrcFastAlgorithm, Digest as CrcFastDigest};
use crc32c::crc32c as crc32c_oneshot;
use crc32fast::Hasher as Crc32FastHasher;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

static REF_CRC24_OPENPGP: RefCrc<u32> = RefCrc::<u32>::new(&crc::CRC_24_OPENPGP);

const BUFFERED_CHUNK_BYTES: usize = 31;

/// Print platform detection info once at benchmark start.
fn print_platform_info() {
  static ONCE: Once = Once::new();
  ONCE.call_once(|| {
    let tune = platform::tune();
    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║                   PLATFORM DETECTION INFO                    ║");
    eprintln!("╠══════════════════════════════════════════════════════════════╣");
    eprintln!("║ Platform: {}", platform::describe());
    eprintln!("║ Tune Kind: {:?}", tune.kind());
    eprintln!("║ PCLMUL threshold: {} bytes", tune.pclmul_threshold);
    eprintln!("║ SIMD width: {} bits", tune.effective_simd_width);
    eprintln!("║ Fast wide ops: {}", tune.fast_wide_ops);
    eprintln!("║ Parallel streams: {}", tune.parallel_streams);
    eprintln!("╠══════════════════════════════════════════════════════════════╣");
    eprintln!("║ Kernel selection by size:");
    for &(label, size) in CASES {
      let crc64 = Crc64::kernel_name_for_len(size);
      let crc64_nvme = Crc64Nvme::kernel_name_for_len(size);
      let crc32 = Crc32::kernel_name_for_len(size);
      let crc32c = Crc32C::kernel_name_for_len(size);
      let crc16_ccitt = Crc16Ccitt::kernel_name_for_len(size);
      let crc16_ibm = Crc16Ibm::kernel_name_for_len(size);
      let crc24 = Crc24OpenPgp::kernel_name_for_len(size);
      eprintln!(
        "║   {:>3} ({:>7} B): crc64/xz={crc64}  crc64/nvme={crc64_nvme}  crc32={crc32}  crc32c={crc32c}  \
         crc16/ccitt={crc16_ccitt}  crc16/ibm={crc16_ibm}  crc24/openpgp={crc24}",
        label, size
      );
    }
    eprintln!("╚══════════════════════════════════════════════════════════════╝");
  });
}

const CASES: &[(&str, usize)] = &[
  ("xs", 64),
  ("s", 256),
  ("m", 4usize.strict_mul(1024)),
  ("l", 64usize.strict_mul(1024)),
  ("xl", 1024usize.strict_mul(1024)),
];

fn make_data(len: usize) -> Vec<u8> {
  (0..len)
    .map(|i| (i as u8).wrapping_mul(31).wrapping_add(i.strict_shr(8) as u8))
    .collect()
}

fn bench_crc64_xz_comp(c: &mut Criterion) {
  print_platform_info();
  let base_rs = Crc64::new();
  let mut buffered_rs = BufferedCrc64Xz::new();
  let base_ref = crc64fast::Digest::new();
  let base_fast = CrcFastDigest::new(CrcFastAlgorithm::Crc64Xz);

  let mut group = c.benchmark_group("crc64/xz");
  for &(label, size) in CASES {
    let data = make_data(size);
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input(BenchmarkId::new("rscrypto/checksum", label), &data, |b, data| {
      b.iter(|| {
        let mut hasher = base_rs.clone();
        hasher.update(black_box(data));
        black_box(hasher.finalize());
      });
    });

    group.bench_with_input(BenchmarkId::new("rscrypto/buffered", label), &data, |b, data| {
      b.iter(|| {
        buffered_rs.reset();
        for chunk in data.chunks(BUFFERED_CHUNK_BYTES) {
          buffered_rs.update(black_box(chunk));
        }
        black_box(buffered_rs.finalize());
      });
    });

    group.bench_with_input(BenchmarkId::new("crc64fast/auto", label), &data, |b, data| {
      b.iter(|| {
        let mut hasher = base_ref.clone();
        hasher.write(black_box(data));
        black_box(hasher.sum64());
      });
    });

    group.bench_with_input(BenchmarkId::new("crc-fast/auto", label), &data, |b, data| {
      b.iter(|| {
        let mut hasher = base_fast;
        hasher.update(black_box(data));
        black_box(hasher.finalize());
      });
    });
  }
  group.finish();
}

fn bench_crc64_nvme_comp(c: &mut Criterion) {
  let base_rs = Crc64Nvme::new();
  let mut buffered_rs = BufferedCrc64Nvme::new();
  let base_ref = crc64fast_nvme::Digest::new();
  let base_fast = CrcFastDigest::new(CrcFastAlgorithm::Crc64Nvme);

  let mut group = c.benchmark_group("crc64/nvme");
  for &(label, size) in CASES {
    let data = make_data(size);
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input(BenchmarkId::new("rscrypto/checksum", label), &data, |b, data| {
      b.iter(|| {
        let mut hasher = base_rs.clone();
        hasher.update(black_box(data));
        black_box(hasher.finalize());
      });
    });

    group.bench_with_input(BenchmarkId::new("rscrypto/buffered", label), &data, |b, data| {
      b.iter(|| {
        buffered_rs.reset();
        for chunk in data.chunks(BUFFERED_CHUNK_BYTES) {
          buffered_rs.update(black_box(chunk));
        }
        black_box(buffered_rs.finalize());
      });
    });

    group.bench_with_input(BenchmarkId::new("crc64fast-nvme/auto", label), &data, |b, data| {
      b.iter(|| {
        let mut hasher = base_ref.clone();
        hasher.write(black_box(data));
        black_box(hasher.sum64());
      });
    });

    group.bench_with_input(BenchmarkId::new("crc-fast/auto", label), &data, |b, data| {
      b.iter(|| {
        let mut hasher = base_fast;
        hasher.update(black_box(data));
        black_box(hasher.finalize());
      });
    });
  }
  group.finish();
}

fn bench_crc32_ieee_comp(c: &mut Criterion) {
  print_platform_info();
  let base_rs = Crc32::new();
  let mut buffered_rs = BufferedCrc32::new();
  let base_fast = CrcFastDigest::new(CrcFastAlgorithm::Crc32IsoHdlc);
  let base_crc32fast = Crc32FastHasher::new();

  let mut group = c.benchmark_group("crc32/ieee");
  for &(label, size) in CASES {
    let data = make_data(size);
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input(BenchmarkId::new("rscrypto/checksum", label), &data, |b, data| {
      b.iter(|| {
        let mut hasher = base_rs.clone();
        hasher.update(black_box(data));
        black_box(hasher.finalize());
      });
    });

    group.bench_with_input(BenchmarkId::new("rscrypto/buffered", label), &data, |b, data| {
      b.iter(|| {
        buffered_rs.reset();
        for chunk in data.chunks(BUFFERED_CHUNK_BYTES) {
          buffered_rs.update(black_box(chunk));
        }
        black_box(buffered_rs.finalize());
      });
    });

    group.bench_with_input(BenchmarkId::new("crc-fast/auto", label), &data, |b, data| {
      b.iter(|| {
        let mut hasher = base_fast;
        hasher.update(black_box(data));
        black_box(hasher.finalize());
      });
    });

    group.bench_with_input(BenchmarkId::new("crc32fast/auto", label), &data, |b, data| {
      b.iter(|| {
        let mut hasher = base_crc32fast.clone();
        hasher.update(black_box(data));
        black_box(hasher.finalize());
      });
    });
  }
  group.finish();
}

fn bench_crc32c_castagnoli_comp(c: &mut Criterion) {
  print_platform_info();
  let base_rs = Crc32C::new();
  let mut buffered_rs = BufferedCrc32C::new();
  let base_fast = CrcFastDigest::new(CrcFastAlgorithm::Crc32Iscsi);

  let mut group = c.benchmark_group("crc32c/castagnoli");
  for &(label, size) in CASES {
    let data = make_data(size);
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input(BenchmarkId::new("rscrypto/checksum", label), &data, |b, data| {
      b.iter(|| {
        let mut hasher = base_rs.clone();
        hasher.update(black_box(data));
        black_box(hasher.finalize());
      });
    });

    group.bench_with_input(BenchmarkId::new("rscrypto/buffered", label), &data, |b, data| {
      b.iter(|| {
        buffered_rs.reset();
        for chunk in data.chunks(BUFFERED_CHUNK_BYTES) {
          buffered_rs.update(black_box(chunk));
        }
        black_box(buffered_rs.finalize());
      });
    });

    group.bench_with_input(BenchmarkId::new("crc-fast/auto", label), &data, |b, data| {
      b.iter(|| {
        let mut hasher = base_fast;
        hasher.update(black_box(data));
        black_box(hasher.finalize());
      });
    });

    group.bench_with_input(BenchmarkId::new("crc32c/oneshot", label), &data, |b, data| {
      b.iter(|| black_box(crc32c_oneshot(black_box(data))));
    });
  }
  group.finish();
}

fn bench_crc16_ccitt_comp(c: &mut Criterion) {
  print_platform_info();
  let base_rs = Crc16Ccitt::new();
  let mut buffered_rs = BufferedCrc16Ccitt::new();
  let base_fast = CrcFastDigest::new(CrcFastAlgorithm::Crc16IbmSdlc);

  let mut group = c.benchmark_group("crc16/ccitt");
  for &(label, size) in CASES {
    let data = make_data(size);
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input(BenchmarkId::new("rscrypto/checksum", label), &data, |b, data| {
      b.iter(|| {
        let mut hasher = base_rs.clone();
        hasher.update(black_box(data));
        black_box(hasher.finalize());
      });
    });

    group.bench_with_input(BenchmarkId::new("rscrypto/buffered", label), &data, |b, data| {
      b.iter(|| {
        buffered_rs.reset();
        for chunk in data.chunks(BUFFERED_CHUNK_BYTES) {
          buffered_rs.update(black_box(chunk));
        }
        black_box(buffered_rs.finalize());
      });
    });

    group.bench_with_input(BenchmarkId::new("crc-fast/auto", label), &data, |b, data| {
      b.iter(|| {
        let mut hasher = base_fast;
        hasher.update(black_box(data));
        black_box(hasher.finalize());
      });
    });
  }
  group.finish();
}

fn bench_crc16_ibm_comp(c: &mut Criterion) {
  print_platform_info();
  let base_rs = Crc16Ibm::new();
  let mut buffered_rs = BufferedCrc16Ibm::new();
  let base_fast = CrcFastDigest::new(CrcFastAlgorithm::Crc16Arc);

  let mut group = c.benchmark_group("crc16/ibm");
  for &(label, size) in CASES {
    let data = make_data(size);
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input(BenchmarkId::new("rscrypto/checksum", label), &data, |b, data| {
      b.iter(|| {
        let mut hasher = base_rs.clone();
        hasher.update(black_box(data));
        black_box(hasher.finalize());
      });
    });

    group.bench_with_input(BenchmarkId::new("rscrypto/buffered", label), &data, |b, data| {
      b.iter(|| {
        buffered_rs.reset();
        for chunk in data.chunks(BUFFERED_CHUNK_BYTES) {
          buffered_rs.update(black_box(chunk));
        }
        black_box(buffered_rs.finalize());
      });
    });

    group.bench_with_input(BenchmarkId::new("crc-fast/auto", label), &data, |b, data| {
      b.iter(|| {
        let mut hasher = base_fast;
        hasher.update(black_box(data));
        black_box(hasher.finalize());
      });
    });
  }
  group.finish();
}

fn crc24_openpgp_reference(data: &[u8]) -> u32 {
  // MSB-first OpenPGP using a 32-bit expanded register (top 24 bits).
  const POLY: u32 = 0x0086_4CFB;
  const INIT: u32 = 0x00B7_04CE;
  let poly_aligned = POLY << 8;

  let mut state: u32 = INIT << 8;
  for &byte in data {
    state ^= (byte as u32) << 24;
    for _ in 0..8 {
      if state & 0x8000_0000 != 0 {
        state = (state << 1) ^ poly_aligned;
      } else {
        state <<= 1;
      }
    }
  }
  (state >> 8) & 0x00FF_FFFF
}

fn bench_crc24_openpgp_comp(c: &mut Criterion) {
  print_platform_info();
  let base_rs = Crc24OpenPgp::new();
  let mut buffered_rs = BufferedCrc24OpenPgp::new();

  let mut group = c.benchmark_group("crc24/openpgp");
  for &(label, size) in CASES {
    let data = make_data(size);
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input(BenchmarkId::new("rscrypto/checksum", label), &data, |b, data| {
      b.iter(|| {
        let mut hasher = base_rs.clone();
        hasher.update(black_box(data));
        black_box(hasher.finalize());
      });
    });

    group.bench_with_input(BenchmarkId::new("rscrypto/buffered", label), &data, |b, data| {
      b.iter(|| {
        buffered_rs.reset();
        for chunk in data.chunks(BUFFERED_CHUNK_BYTES) {
          buffered_rs.update(black_box(chunk));
        }
        black_box(buffered_rs.finalize());
      });
    });

    group.bench_with_input(BenchmarkId::new("crc/auto", label), &data, |b, data| {
      b.iter(|| {
        let mut digest = REF_CRC24_OPENPGP.digest();
        digest.update(black_box(data));
        black_box(digest.finalize() & 0x00FF_FFFF);
      });
    });

    // Bitwise reference is too slow for large buffers; keep it as a sanity/overhead baseline.
    if size <= 4096 {
      group.bench_with_input(BenchmarkId::new("reference/bitwise", label), &data, |b, data| {
        b.iter(|| black_box(crc24_openpgp_reference(black_box(data))));
      });
    }
  }
  group.finish();
}

criterion_group!(
  benches,
  bench_crc64_xz_comp,
  bench_crc64_nvme_comp,
  bench_crc32_ieee_comp,
  bench_crc32c_castagnoli_comp,
  bench_crc16_ccitt_comp,
  bench_crc16_ibm_comp,
  bench_crc24_openpgp_comp
);
criterion_main!(benches);
