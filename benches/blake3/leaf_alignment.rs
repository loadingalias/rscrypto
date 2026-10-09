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

  use blake3::hazmat::HasherExt as _;
  use rscrypto::hashes::crypto::blake3::{
    Blake3DiagKernel, diag_blake3_chunk_cvs_with_kernel, diag_blake3_kernel_available,
  };

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
      let mut expected = Vec::with_capacity(chunks.strict_mul(32));
      for (index, chunk) in data.chunks_exact(1024).enumerate() {
        let mut hasher = blake3::Hasher::new();
        let offset = u64::try_from(index.strict_mul(1024)).expect("fixture offset fits");
        hasher.set_input_offset(offset).update(chunk);
        expected.extend_from_slice(&hasher.finalize_non_root());
      }
      let mut portable = vec![0u8; expected.len()];
      diag_blake3_chunk_cvs_with_kernel(Blake3DiagKernel::Portable, data, &mut portable)
        .expect("portable kernel is available");
      assert_eq!(portable, expected, "portable CV oracle, chunks={chunks}");

      for output_offset in [0usize, 1] {
        let name = format!("blake3/leaf-alignment-cvs/chunks-{chunks}/in-{input_offset}/out-{output_offset}");
        let mut g = c.benchmark_group(name);
        super::common::set_throughput(&mut g, len);
        for kernel in kernels {
          if !diag_blake3_kernel_available(kernel) {
            continue;
          }
          let mut out = vec![0xa5; expected.len().strict_add(128)];
          let range = range_at_offset(&out, expected.len(), output_offset);
          diag_blake3_chunk_cvs_with_kernel(kernel, data, &mut out[range.clone()])
            .expect("selected kernel is available");
          assert_eq!(out[range.clone()], expected, "{kernel:?}, chunks={chunks}");
          assert!(out[..range.start].iter().all(|&byte| byte == 0xa5));
          assert!(out[range.end..].iter().all(|&byte| byte == 0xa5));
          let output = &mut out[range];
          g.bench_function(format!("rscrypto-{}", kernel.label()), |b| {
            b.iter(|| {
              diag_blake3_chunk_cvs_with_kernel(kernel, black_box(data), black_box(&mut *output))
                .expect("selected kernel is available");
              black_box(&*output);
            })
          });
        }
        g.finish();
      }
    }
  }
}
