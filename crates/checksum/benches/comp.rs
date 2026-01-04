use core::hint::black_box;

use checksum::{
  BufferedCrc16Ccitt, BufferedCrc16Ibm, BufferedCrc24OpenPgp, BufferedCrc32, BufferedCrc32C, BufferedCrc64Nvme,
  BufferedCrc64Xz, Checksum, Crc16Ccitt, Crc16Ibm, Crc24OpenPgp, Crc32, Crc32C, Crc64, Crc64Nvme,
};
use crc::Crc as RefCrc;
use crc_fast::{CrcAlgorithm as CrcFastAlgorithm, Digest as CrcFastDigest};
use crc32c::crc32c as crc32c_oneshot;
use crc32fast::Hasher as Crc32FastHasher;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

mod util;

static REF_CRC24_OPENPGP: RefCrc<u32> = RefCrc::<u32>::new(&crc::CRC_24_OPENPGP);

fn bench_crc64_xz_comp(c: &mut Criterion) {
  util::print_platform_info();
  let base_rs = Crc64::new();
  let mut buffered_rs = BufferedCrc64Xz::new();
  let base_ref = crc64fast::Digest::new();
  let base_fast = CrcFastDigest::new(CrcFastAlgorithm::Crc64Xz);

  let mut group = c.benchmark_group("crc64/xz");
  for &(label, size) in util::CASES {
    let data = util::make_alignment_variants(util::make_data(size));
    group.throughput(Throughput::Bytes(size as u64));

    let kernel = Crc64::kernel_name_for_len(size);
    for variant in &data {
      let data = variant.as_slice();
      let param = util::bench_param_label(label, variant.alignment());

      group.bench_with_input(
        BenchmarkId::new(format!("rscrypto/checksum[{kernel}]"), &param),
        &data,
        |b, data| {
          b.iter(|| {
            let mut hasher = base_rs.clone();
            hasher.update(black_box(data));
            black_box(hasher.finalize());
          });
        },
      );

      group.bench_with_input(
        BenchmarkId::new(format!("rscrypto/buffered[{kernel}]"), &param),
        &data,
        |b, data| {
          b.iter(|| {
            buffered_rs.reset();
            for chunk in data.chunks(util::BUFFERED_CHUNK_BYTES) {
              buffered_rs.update(black_box(chunk));
            }
            black_box(buffered_rs.finalize());
          });
        },
      );

      group.bench_with_input(BenchmarkId::new("crc64fast/auto", &param), &data, |b, data| {
        b.iter(|| {
          let mut hasher = base_ref.clone();
          hasher.write(black_box(data));
          black_box(hasher.sum64());
        });
      });

      group.bench_with_input(BenchmarkId::new("crc-fast/auto", &param), &data, |b, data| {
        b.iter(|| {
          let mut hasher = base_fast;
          hasher.update(black_box(data));
          black_box(hasher.finalize());
        });
      });
    }
  }
  group.finish();
}

fn bench_crc64_nvme_comp(c: &mut Criterion) {
  let base_rs = Crc64Nvme::new();
  let mut buffered_rs = BufferedCrc64Nvme::new();
  let base_ref = crc64fast_nvme::Digest::new();
  let base_fast = CrcFastDigest::new(CrcFastAlgorithm::Crc64Nvme);

  let mut group = c.benchmark_group("crc64/nvme");
  for &(label, size) in util::CASES {
    let data = util::make_alignment_variants(util::make_data(size));
    group.throughput(Throughput::Bytes(size as u64));

    let kernel = Crc64Nvme::kernel_name_for_len(size);
    for variant in &data {
      let data = variant.as_slice();
      let param = util::bench_param_label(label, variant.alignment());

      group.bench_with_input(
        BenchmarkId::new(format!("rscrypto/checksum[{kernel}]"), &param),
        &data,
        |b, data| {
          b.iter(|| {
            let mut hasher = base_rs.clone();
            hasher.update(black_box(data));
            black_box(hasher.finalize());
          });
        },
      );

      group.bench_with_input(
        BenchmarkId::new(format!("rscrypto/buffered[{kernel}]"), &param),
        &data,
        |b, data| {
          b.iter(|| {
            buffered_rs.reset();
            for chunk in data.chunks(util::BUFFERED_CHUNK_BYTES) {
              buffered_rs.update(black_box(chunk));
            }
            black_box(buffered_rs.finalize());
          });
        },
      );

      group.bench_with_input(BenchmarkId::new("crc64fast-nvme/auto", &param), &data, |b, data| {
        b.iter(|| {
          let mut hasher = base_ref.clone();
          hasher.write(black_box(data));
          black_box(hasher.sum64());
        });
      });

      group.bench_with_input(BenchmarkId::new("crc-fast/auto", &param), &data, |b, data| {
        b.iter(|| {
          let mut hasher = base_fast;
          hasher.update(black_box(data));
          black_box(hasher.finalize());
        });
      });
    }
  }
  group.finish();
}

fn bench_crc32_ieee_comp(c: &mut Criterion) {
  util::print_platform_info();
  let base_rs = Crc32::new();
  let mut buffered_rs = BufferedCrc32::new();
  let base_fast = CrcFastDigest::new(CrcFastAlgorithm::Crc32IsoHdlc);
  let base_crc32fast = Crc32FastHasher::new();

  let mut group = c.benchmark_group("crc32/ieee");
  for &(label, size) in util::CASES {
    let data = util::make_alignment_variants(util::make_data(size));
    group.throughput(Throughput::Bytes(size as u64));

    let kernel = Crc32::kernel_name_for_len(size);
    for variant in &data {
      let data = variant.as_slice();
      let param = util::bench_param_label(label, variant.alignment());

      group.bench_with_input(
        BenchmarkId::new(format!("rscrypto/checksum[{kernel}]"), &param),
        &data,
        |b, data| {
          b.iter(|| {
            let mut hasher = base_rs.clone();
            hasher.update(black_box(data));
            black_box(hasher.finalize());
          });
        },
      );

      group.bench_with_input(
        BenchmarkId::new(format!("rscrypto/buffered[{kernel}]"), &param),
        &data,
        |b, data| {
          b.iter(|| {
            buffered_rs.reset();
            for chunk in data.chunks(util::BUFFERED_CHUNK_BYTES) {
              buffered_rs.update(black_box(chunk));
            }
            black_box(buffered_rs.finalize());
          });
        },
      );

      group.bench_with_input(BenchmarkId::new("crc-fast/auto", &param), &data, |b, data| {
        b.iter(|| {
          let mut hasher = base_fast;
          hasher.update(black_box(data));
          black_box(hasher.finalize());
        });
      });

      group.bench_with_input(BenchmarkId::new("crc32fast/auto", &param), &data, |b, data| {
        b.iter(|| {
          let mut hasher = base_crc32fast.clone();
          hasher.update(black_box(data));
          black_box(hasher.finalize());
        });
      });
    }
  }
  group.finish();
}

fn bench_crc32c_castagnoli_comp(c: &mut Criterion) {
  util::print_platform_info();
  let base_rs = Crc32C::new();
  let mut buffered_rs = BufferedCrc32C::new();
  let base_fast = CrcFastDigest::new(CrcFastAlgorithm::Crc32Iscsi);

  let mut group = c.benchmark_group("crc32c/castagnoli");
  for &(label, size) in util::CASES {
    let data = util::make_alignment_variants(util::make_data(size));
    group.throughput(Throughput::Bytes(size as u64));

    let kernel = Crc32C::kernel_name_for_len(size);
    for variant in &data {
      let data = variant.as_slice();
      let param = util::bench_param_label(label, variant.alignment());

      group.bench_with_input(
        BenchmarkId::new(format!("rscrypto/checksum[{kernel}]"), &param),
        &data,
        |b, data| {
          b.iter(|| {
            let mut hasher = base_rs.clone();
            hasher.update(black_box(data));
            black_box(hasher.finalize());
          });
        },
      );

      group.bench_with_input(
        BenchmarkId::new(format!("rscrypto/buffered[{kernel}]"), &param),
        &data,
        |b, data| {
          b.iter(|| {
            buffered_rs.reset();
            for chunk in data.chunks(util::BUFFERED_CHUNK_BYTES) {
              buffered_rs.update(black_box(chunk));
            }
            black_box(buffered_rs.finalize());
          });
        },
      );

      group.bench_with_input(BenchmarkId::new("crc-fast/auto", &param), &data, |b, data| {
        b.iter(|| {
          let mut hasher = base_fast;
          hasher.update(black_box(data));
          black_box(hasher.finalize());
        });
      });

      group.bench_with_input(BenchmarkId::new("crc32c/oneshot", &param), &data, |b, data| {
        b.iter(|| black_box(crc32c_oneshot(black_box(data))));
      });
    }
  }
  group.finish();
}

fn bench_crc16_ccitt_comp(c: &mut Criterion) {
  util::print_platform_info();
  let base_rs = Crc16Ccitt::new();
  let mut buffered_rs = BufferedCrc16Ccitt::new();
  let base_fast = CrcFastDigest::new(CrcFastAlgorithm::Crc16IbmSdlc);

  let mut group = c.benchmark_group("crc16/ccitt");
  for &(label, size) in util::CASES {
    let data = util::make_alignment_variants(util::make_data(size));
    group.throughput(Throughput::Bytes(size as u64));

    let kernel = Crc16Ccitt::kernel_name_for_len(size);
    for variant in &data {
      let data = variant.as_slice();
      let param = util::bench_param_label(label, variant.alignment());

      group.bench_with_input(
        BenchmarkId::new(format!("rscrypto/checksum[{kernel}]"), &param),
        &data,
        |b, data| {
          b.iter(|| {
            let mut hasher = base_rs.clone();
            hasher.update(black_box(data));
            black_box(hasher.finalize());
          });
        },
      );

      group.bench_with_input(
        BenchmarkId::new(format!("rscrypto/buffered[{kernel}]"), &param),
        &data,
        |b, data| {
          b.iter(|| {
            buffered_rs.reset();
            for chunk in data.chunks(util::BUFFERED_CHUNK_BYTES) {
              buffered_rs.update(black_box(chunk));
            }
            black_box(buffered_rs.finalize());
          });
        },
      );

      group.bench_with_input(BenchmarkId::new("crc-fast/auto", &param), &data, |b, data| {
        b.iter(|| {
          let mut hasher = base_fast;
          hasher.update(black_box(data));
          black_box(hasher.finalize());
        });
      });
    }
  }
  group.finish();
}

fn bench_crc16_ibm_comp(c: &mut Criterion) {
  util::print_platform_info();
  let base_rs = Crc16Ibm::new();
  let mut buffered_rs = BufferedCrc16Ibm::new();
  let base_fast = CrcFastDigest::new(CrcFastAlgorithm::Crc16Arc);

  let mut group = c.benchmark_group("crc16/ibm");
  for &(label, size) in util::CASES {
    let data = util::make_alignment_variants(util::make_data(size));
    group.throughput(Throughput::Bytes(size as u64));

    let kernel = Crc16Ibm::kernel_name_for_len(size);
    for variant in &data {
      let data = variant.as_slice();
      let param = util::bench_param_label(label, variant.alignment());

      group.bench_with_input(
        BenchmarkId::new(format!("rscrypto/checksum[{kernel}]"), &param),
        &data,
        |b, data| {
          b.iter(|| {
            let mut hasher = base_rs.clone();
            hasher.update(black_box(data));
            black_box(hasher.finalize());
          });
        },
      );

      group.bench_with_input(
        BenchmarkId::new(format!("rscrypto/buffered[{kernel}]"), &param),
        &data,
        |b, data| {
          b.iter(|| {
            buffered_rs.reset();
            for chunk in data.chunks(util::BUFFERED_CHUNK_BYTES) {
              buffered_rs.update(black_box(chunk));
            }
            black_box(buffered_rs.finalize());
          });
        },
      );

      group.bench_with_input(BenchmarkId::new("crc-fast/auto", &param), &data, |b, data| {
        b.iter(|| {
          let mut hasher = base_fast;
          hasher.update(black_box(data));
          black_box(hasher.finalize());
        });
      });
    }
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
  util::print_platform_info();
  let base_rs = Crc24OpenPgp::new();
  let mut buffered_rs = BufferedCrc24OpenPgp::new();

  let mut group = c.benchmark_group("crc24/openpgp");
  for &(label, size) in util::CASES {
    let data = util::make_alignment_variants(util::make_data(size));
    group.throughput(Throughput::Bytes(size as u64));

    let kernel = Crc24OpenPgp::kernel_name_for_len(size);
    for variant in &data {
      let data = variant.as_slice();
      let param = util::bench_param_label(label, variant.alignment());

      group.bench_with_input(
        BenchmarkId::new(format!("rscrypto/checksum[{kernel}]"), &param),
        &data,
        |b, data| {
          b.iter(|| {
            let mut hasher = base_rs.clone();
            hasher.update(black_box(data));
            black_box(hasher.finalize());
          });
        },
      );

      group.bench_with_input(
        BenchmarkId::new(format!("rscrypto/buffered[{kernel}]"), &param),
        &data,
        |b, data| {
          b.iter(|| {
            buffered_rs.reset();
            for chunk in data.chunks(util::BUFFERED_CHUNK_BYTES) {
              buffered_rs.update(black_box(chunk));
            }
            black_box(buffered_rs.finalize());
          });
        },
      );

      group.bench_with_input(BenchmarkId::new("crc/auto", &param), &data, |b, data| {
        b.iter(|| {
          let mut digest = REF_CRC24_OPENPGP.digest();
          digest.update(black_box(data));
          black_box(digest.finalize() & 0x00FF_FFFF);
        });
      });

      // Bitwise reference is too slow for large buffers; keep it as a sanity/overhead baseline.
      if size <= 4096 {
        group.bench_with_input(BenchmarkId::new("reference/bitwise", &param), &data, |b, data| {
          b.iter(|| black_box(crc24_openpgp_reference(black_box(data))));
        });
      }
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
