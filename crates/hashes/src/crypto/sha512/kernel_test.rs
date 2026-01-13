extern crate alloc;

use alloc::vec::Vec;

use traits::Digest as _;

use super::{
  Sha512,
  kernels::{ALL, Sha512KernelId, compress_blocks_fn, required_caps},
};
use crate::crypto::dispatch_util::SizeClassDispatch;

#[derive(Clone, Debug)]
pub struct KernelResult {
  pub name: &'static str,
  pub digest: [u8; 64],
}

fn hasher_for_kernel(id: Sha512KernelId) -> Sha512 {
  let compress = compress_blocks_fn(id);
  Sha512 {
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

fn digest_with_kernel(id: Sha512KernelId, data: &[u8]) -> [u8; 64] {
  let mut h = hasher_for_kernel(id);
  h.update(data);
  h.finalize()
}

#[must_use]
pub fn run_all_sha512_kernels(data: &[u8]) -> Vec<KernelResult> {
  let caps = platform::caps();
  let mut out = Vec::new();
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

pub fn verify_sha512_kernels(data: &[u8]) -> Result<(), &'static str> {
  let results = run_all_sha512_kernels(data);
  let Some(first) = results.first() else {
    return Ok(());
  };
  for r in &results[1..] {
    if r.digest != first.digest {
      return Err("sha512 kernel mismatch");
    }
  }
  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;

  fn pattern(len: usize) -> Vec<u8> {
    (0..len)
      .map(|i| (i as u8).wrapping_mul(13).wrapping_add((i >> 8) as u8))
      .collect()
  }

  #[test]
  fn all_kernels_match_sha2_oracle_and_streaming_splits() {
    let caps = platform::caps();
    let lens = [
      0usize, 1, 2, 3, 111, 112, 113, 127, 128, 129, 239, 240, 241, 255, 256, 257, 1000,
    ];

    for &id in ALL {
      if !caps.has(required_caps(id)) {
        continue;
      }

      for &len in &lens {
        let msg = pattern(len);
        let ours = digest_with_kernel(id, &msg);

        use sha2::Digest as _;
        let expected = sha2::Sha512::digest(&msg);
        let mut exp = [0u8; 64];
        exp.copy_from_slice(&expected);
        assert_eq!(ours, exp, "sha512 oracle mismatch for kernel={}", id.as_str());

        for &chunk in &[1usize, 7, 31, 32, 63, 64, 65, 127, 128, 129, 1024, 4096] {
          let mut h = hasher_for_kernel(id);
          for part in msg.chunks(chunk) {
            h.update(part);
          }
          assert_eq!(
            h.finalize(),
            ours,
            "sha512 streaming mismatch kernel={} len={} chunk={}",
            id.as_str(),
            len,
            chunk
          );
        }
      }
    }
  }
}
