//! Address alignment controls for existing full-chunk production workers.

use core::ops::Range;

fn range_at_offset(storage: &[u8], len: usize, offset: usize) -> Range<usize> {
  assert!(offset < 64);
  let base_offset = storage.as_ptr() as usize % 64;
  let start = (64usize.strict_sub(base_offset) % 64).strict_add(offset);
  let range = start..start.strict_add(len);
  assert_eq!(storage[range.clone()].as_ptr() as usize % 64, offset);
  range
}

pub(super) fn fixture(len: usize, offset: usize) -> (Vec<u8>, Range<usize>) {
  let mut storage = vec![0u8; len.strict_add(128)];
  let range = range_at_offset(&storage, len, offset);
  for (i, byte) in storage[range.clone()].iter_mut().enumerate() {
    *byte = u8::try_from(i.strict_add(1) % 251).expect("fixture remainder fits");
  }
  (storage, range)
}

#[cfg(all(rscrypto_internal, feature = "diag"))]
pub(super) fn bench(c: &mut criterion::Criterion) {
  use core::hint::black_box;

  use rscrypto::hashes::crypto::blake3::{
    Blake3DiagKernel, diag_blake3_chunk_cvs_with_kernel, diag_blake3_kernel_available,
  };

  use super::bench_config::checked_with;

  if !super::bench_config::selected("blake3/leaf-alignment-cvs/") {
    return;
  }
  let kernels = [
    Blake3DiagKernel::Portable,
    #[cfg(target_arch = "aarch64")]
    Blake3DiagKernel::Aarch64Neon,
  ];
  for chunks in 1usize..=8 {
    let len = chunks.strict_mul(1024);
    for input_offset in [0usize, 1] {
      let (storage, input_range) = fixture(len, input_offset);
      let data = &storage[input_range];
      let expected = super::upstream_chunk_cvs(data);

      for output_offset in [0usize, 1] {
        let name = format!("blake3/leaf-alignment-cvs/chunks-{chunks}/in-{input_offset}/out-{output_offset}");
        let mut g = c.benchmark_group(name);
        super::common::set_throughput(&mut g, len);
        for kernel in kernels {
          if !diag_blake3_kernel_available(kernel) {
            continue;
          }
          // Guard bytes on both sides of the aligned output detect writes outside it.
          let mut out = vec![0xa5; expected.len().strict_add(128)];
          let range = range_at_offset(&out, expected.len(), output_offset);
          let (head, rest) = out.split_at_mut(range.start);
          let (output, tail) = rest.split_at_mut(range.len());
          g.bench_function(
            format!("rscrypto-{}", kernel.label()),
            checked_with(
              output,
              |output| {
                diag_blake3_chunk_cvs_with_kernel(kernel, black_box(data), black_box(&mut **output))
                  .expect("selected kernel is available");
                black_box(&**output);
              },
              |(), output| {
                assert_eq!(**output, *expected, "{kernel:?}, chunks={chunks}");
                assert!(head.iter().all(|&byte| byte == 0xa5));
                assert!(tail.iter().all(|&byte| byte == 0xa5));
              },
            ),
          );
        }
        g.finish();
      }
    }
  }
}
