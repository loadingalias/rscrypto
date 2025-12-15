//! CRC Library Comparison Benchmarks
//!
//! Compares rscrypto against popular Rust CRC implementations.
//!
//! Run all comparisons:
//!   cargo bench -p checksum -- comp
//!
//! Run with native CPU optimizations:
//!   RUSTFLAGS='-C target-cpu=native' cargo bench -p checksum -- comp
//!
//! Run specific algorithm:
//!   cargo bench -p checksum -- comp/crc32c

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

const SIZES: [usize; 7] = [64, 256, 1024, 4096, 16384, 65536, 1048576];

// ============================================================================
// CRC32-C (Castagnoli)
// ============================================================================

fn comp_crc32c(c: &mut Criterion) {
  let mut group = c.benchmark_group("comp/crc32c");
  eprintln!("rscrypto crc32c: {}", checksum::crc32c::selected_backend());

  for size in SIZES {
    let data = vec![0u8; size];
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input(BenchmarkId::new("rscrypto", size), &data, |b, data| {
      b.iter(|| core::hint::black_box(checksum::Crc32c::checksum(data)));
    });

    group.bench_with_input(BenchmarkId::new("crc32c", size), &data, |b, data| {
      b.iter(|| core::hint::black_box(crc32c::crc32c(data)));
    });

    let crc_generic = crc::Crc::<u32>::new(&crc::CRC_32_ISCSI);
    group.bench_with_input(BenchmarkId::new("crc", size), &data, |b, data| {
      b.iter(|| core::hint::black_box(crc_generic.checksum(data)));
    });

    group.bench_with_input(BenchmarkId::new("crc-fast", size), &data, |b, data| {
      b.iter(|| core::hint::black_box(crc_fast::checksum(crc_fast::CrcAlgorithm::Crc32Iscsi, data) as u32));
    });
  }

  group.finish();
}

// ============================================================================
// CRC32 (ISO-HDLC)
// ============================================================================

fn comp_crc32(c: &mut Criterion) {
  let mut group = c.benchmark_group("comp/crc32");
  eprintln!("rscrypto crc32: {}", checksum::crc32::selected_backend());

  for size in SIZES {
    let data = vec![0u8; size];
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input(BenchmarkId::new("rscrypto", size), &data, |b, data| {
      b.iter(|| core::hint::black_box(checksum::Crc32::checksum(data)));
    });

    group.bench_with_input(BenchmarkId::new("crc32fast", size), &data, |b, data| {
      b.iter(|| core::hint::black_box(crc32fast::hash(data)));
    });

    let crc_generic = crc::Crc::<u32>::new(&crc::CRC_32_ISO_HDLC);
    group.bench_with_input(BenchmarkId::new("crc", size), &data, |b, data| {
      b.iter(|| core::hint::black_box(crc_generic.checksum(data)));
    });

    group.bench_with_input(BenchmarkId::new("crc-fast", size), &data, |b, data| {
      b.iter(|| core::hint::black_box(crc_fast::checksum(crc_fast::CrcAlgorithm::Crc32IsoHdlc, data) as u32));
    });
  }

  group.finish();
}

// ============================================================================
// CRC64/XZ (ECMA)
// ============================================================================

fn comp_crc64_xz(c: &mut Criterion) {
  let mut group = c.benchmark_group("comp/crc64-xz");
  eprintln!("rscrypto crc64/xz: {}", checksum::crc64::selected_backend());

  for size in SIZES {
    let data = vec![0u8; size];
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input(BenchmarkId::new("rscrypto", size), &data, |b, data| {
      b.iter(|| core::hint::black_box(checksum::Crc64::checksum(data)));
    });

    group.bench_with_input(BenchmarkId::new("crc64fast", size), &data, |b, data| {
      b.iter(|| {
        let mut digest = crc64fast::Digest::new();
        digest.write(data);
        core::hint::black_box(digest.sum64())
      });
    });

    let crc_generic = crc::Crc::<u64>::new(&crc::CRC_64_XZ);
    group.bench_with_input(BenchmarkId::new("crc", size), &data, |b, data| {
      b.iter(|| core::hint::black_box(crc_generic.checksum(data)));
    });

    group.bench_with_input(BenchmarkId::new("crc-fast", size), &data, |b, data| {
      b.iter(|| core::hint::black_box(crc_fast::checksum(crc_fast::CrcAlgorithm::Crc64Xz, data)));
    });
  }

  group.finish();
}

// ============================================================================
// CRC64/NVMe
// ============================================================================

fn comp_crc64_nvme(c: &mut Criterion) {
  let mut group = c.benchmark_group("comp/crc64-nvme");
  eprintln!("rscrypto crc64/nvme: {}", checksum::crc64::selected_backend_nvme());

  for size in SIZES {
    let data = vec![0u8; size];
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input(BenchmarkId::new("rscrypto", size), &data, |b, data| {
      b.iter(|| core::hint::black_box(checksum::Crc64Nvme::checksum(data)));
    });

    group.bench_with_input(BenchmarkId::new("crc64fast-nvme", size), &data, |b, data| {
      b.iter(|| {
        let mut digest = crc64fast_nvme::Digest::new();
        digest.write(data);
        core::hint::black_box(digest.sum64())
      });
    });

    group.bench_with_input(BenchmarkId::new("crc-fast", size), &data, |b, data| {
      b.iter(|| core::hint::black_box(crc_fast::checksum(crc_fast::CrcAlgorithm::Crc64Nvme, data)));
    });
  }

  group.finish();
}

// ============================================================================
// CRC16/IBM
// ============================================================================

fn comp_crc16_ibm(c: &mut Criterion) {
  let mut group = c.benchmark_group("comp/crc16-ibm");
  eprintln!("rscrypto crc16/ibm: {}", checksum::crc16::ibm::selected_backend());

  for size in SIZES {
    let data = vec![0u8; size];
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input(BenchmarkId::new("rscrypto", size), &data, |b, data| {
      b.iter(|| core::hint::black_box(checksum::Crc16Ibm::checksum(data)));
    });

    let crc_generic = crc::Crc::<u16>::new(&crc::CRC_16_ARC);
    group.bench_with_input(BenchmarkId::new("crc", size), &data, |b, data| {
      b.iter(|| core::hint::black_box(crc_generic.checksum(data)));
    });
  }

  group.finish();
}

// ============================================================================
// CRC16/CCITT-FALSE
// ============================================================================

fn comp_crc16_ccitt(c: &mut Criterion) {
  let mut group = c.benchmark_group("comp/crc16-ccitt");
  eprintln!(
    "rscrypto crc16/ccitt: {}",
    checksum::crc16::ccitt_false::selected_backend()
  );

  for size in SIZES {
    let data = vec![0u8; size];
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input(BenchmarkId::new("rscrypto", size), &data, |b, data| {
      b.iter(|| core::hint::black_box(checksum::Crc16CcittFalse::checksum(data)));
    });

    let crc_generic = crc::Crc::<u16>::new(&crc::CRC_16_IBM_3740);
    group.bench_with_input(BenchmarkId::new("crc", size), &data, |b, data| {
      b.iter(|| core::hint::black_box(crc_generic.checksum(data)));
    });
  }

  group.finish();
}

// ============================================================================
// CRC24/OpenPGP
// ============================================================================

fn comp_crc24(c: &mut Criterion) {
  let mut group = c.benchmark_group("comp/crc24");
  eprintln!("rscrypto crc24: {}", checksum::crc24::selected_backend());

  for size in SIZES {
    let data = vec![0u8; size];
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input(BenchmarkId::new("rscrypto", size), &data, |b, data| {
      b.iter(|| core::hint::black_box(checksum::Crc24::checksum(data)));
    });

    let crc_generic = crc::Crc::<u32>::new(&crc::CRC_24_OPENPGP);
    group.bench_with_input(BenchmarkId::new("crc", size), &data, |b, data| {
      b.iter(|| core::hint::black_box(crc_generic.checksum(data)));
    });
  }

  group.finish();
}

criterion_group!(
  benches,
  comp_crc32c,
  comp_crc32,
  comp_crc64_xz,
  comp_crc64_nvme,
  comp_crc16_ibm,
  comp_crc16_ccitt,
  comp_crc24,
);
criterion_main!(benches);
