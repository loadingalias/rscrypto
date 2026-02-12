extern crate alloc;

use alloc::vec::Vec;

use traits::Digest as _;

use super::{
  Sha256,
  kernels::{ALL, Sha256KernelId, compress_blocks_fn, required_caps},
};
use crate::crypto::dispatch_util::SizeClassDispatch;

#[derive(Clone, Debug)]
pub struct KernelResult {
  pub name: &'static str,
  pub digest: [u8; 32],
}

fn hasher_for_kernel(id: Sha256KernelId) -> Sha256 {
  let compress = compress_blocks_fn(id);
  Sha256 {
    compress_blocks: compress,
    dispatch: Some(SizeClassDispatch {
      boundaries: [usize::MAX; 3],
      xs: compress,
      s: compress,
      m: compress,
      l: compress,
    }),
    ..Default::default()
  }
}

fn digest_with_kernel(id: Sha256KernelId, data: &[u8]) -> [u8; 32] {
  let mut h = hasher_for_kernel(id);
  h.update(data);
  h.finalize()
}

#[must_use]
pub fn run_all_sha256_kernels(data: &[u8]) -> Vec<KernelResult> {
  let caps = platform::caps();
  let mut out = Vec::with_capacity(ALL.len());
  for &id in ALL {
    if caps.has(required_caps(id)) {
      out.push(KernelResult {
        name: id.as_str(),
        digest: digest_with_kernel(id, data),
      });
    }
  }
  out
}

pub fn verify_sha256_kernels(data: &[u8]) -> Result<(), &'static str> {
  let results = run_all_sha256_kernels(data);
  let Some(first) = results.first() else {
    return Ok(());
  };
  for r in &results[1..] {
    if r.digest != first.digest {
      return Err("sha256 kernel mismatch");
    }
  }
  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;

  fn pattern(len: usize) -> Vec<u8> {
    (0..len)
      .map(|i| (i as u8).wrapping_mul(31).wrapping_add((i >> 8) as u8))
      .collect()
  }

  fn all_chunk_sizes() -> &'static [usize] {
    &[
      1, 2, 3, 7, 8, 15, 16, 31, 32, 63, 64, 65, 127, 128, 255, 256, 1024, 4096,
    ]
  }

  #[test]
  fn all_kernels_match_sha2_oracle_and_streaming_splits() {
    let caps = platform::caps();

    let lens = [
      0usize, 1, 2, 3, 55, 56, 57, 63, 64, 65, 119, 120, 121, 127, 128, 129, 1000,
    ];

    for &id in ALL {
      if !caps.has(required_caps(id)) {
        continue;
      }

      for &len in &lens {
        let msg = pattern(len);
        let ours = digest_with_kernel(id, &msg);

        use sha2::Digest as _;
        let expected = sha2::Sha256::digest(&msg);
        let mut exp = [0u8; 32];
        exp.copy_from_slice(&expected);
        assert_eq!(ours, exp, "sha256 oracle mismatch for kernel={}", id.as_str());

        // Streaming chunking patterns.
        for &chunk in all_chunk_sizes() {
          let mut h = hasher_for_kernel(id);
          for part in msg.chunks(chunk) {
            h.update(part);
          }
          assert_eq!(
            h.finalize(),
            ours,
            "sha256 streaming mismatch kernel={} len={} chunk={}",
            id.as_str(),
            len,
            chunk
          );
        }

        // Exhaustive two-split for small buffers (padding edges).
        if len <= 256 {
          for split in 0..=len {
            let (a, b) = msg.split_at(split);
            let mut h = hasher_for_kernel(id);
            h.update(a);
            h.update(b);
            assert_eq!(
              h.finalize(),
              ours,
              "sha256 split mismatch kernel={} len={} split={}",
              id.as_str(),
              len,
              split
            );
          }
        }
      }
    }
  }

  #[test]
  fn run_all_agree() {
    verify_sha256_kernels(b"abc").expect("kernels should agree");
    verify_sha256_kernels(&pattern(4096)).expect("kernels should agree");
  }
}
