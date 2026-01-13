extern crate alloc;

use alloc::vec::Vec;

use traits::Digest as _;

use super::{
  Blake3,
  kernels::{ALL, Blake3KernelId, kernel as kernel_for_id, required_caps},
};
use crate::crypto::dispatch_util::SizeClassDispatch;

#[derive(Clone, Debug)]
pub struct KernelResult {
  pub name: &'static str,
  pub digest: [u8; 32],
}

fn force_hasher_kernel(mut h: Blake3, id: Blake3KernelId) -> Blake3 {
  let kernel = kernel_for_id(id);
  h.kernel = kernel;
  h.dispatch = Some(SizeClassDispatch {
    boundaries: [usize::MAX; 3],
    xs: kernel,
    s: kernel,
    m: kernel,
    l: kernel,
  });
  h.chunk_state.kernel = kernel;
  h
}

fn hasher_for_kernel(id: Blake3KernelId) -> Blake3 {
  force_hasher_kernel(Blake3::new(), id)
}

fn digest_with_kernel(id: Blake3KernelId, data: &[u8]) -> [u8; 32] {
  let mut h = hasher_for_kernel(id);
  h.update(data);
  h.finalize()
}

#[must_use]
pub fn run_all_blake3_kernels(data: &[u8]) -> Vec<KernelResult> {
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

pub fn verify_blake3_kernels(data: &[u8]) -> Result<(), &'static str> {
  let results = run_all_blake3_kernels(data);
  let Some(first) = results.first() else {
    return Ok(());
  };
  for r in &results[1..] {
    if r.digest != first.digest {
      return Err("blake3 kernel mismatch");
    }
  }
  Ok(())
}

#[cfg(test)]
mod tests {
  use traits::Xof as _;

  use super::*;

  const KEY: &[u8; 32] = b"whats the Elvish word for friend";
  const CONTEXT: &str = "BLAKE3 2019-12-27 16:29:52 test vectors context";

  fn keyed_hasher_for_kernel(id: Blake3KernelId, key: &[u8; 32]) -> Blake3 {
    force_hasher_kernel(Blake3::new_keyed(key), id)
  }

  fn derive_hasher_for_kernel(id: Blake3KernelId, context: &str) -> Blake3 {
    force_hasher_kernel(Blake3::new_derive_key(context), id)
  }

  fn pattern(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i % 251) as u8).collect()
  }

  #[test]
  fn all_kernels_match_official_crate_and_streaming_splits() {
    let caps = platform::caps();
    let lens = [0usize, 1, 2, 3, 63, 64, 65, 1023, 1024, 1025, 2047, 2048, 2049, 10_000];

    for &id in ALL {
      if !caps.has(required_caps(id)) {
        continue;
      }

      for &len in &lens {
        let msg = pattern(len);

        // Hash mode.
        let ours = digest_with_kernel(id, &msg);
        let expected = *blake3::hash(&msg).as_bytes();
        assert_eq!(ours, expected, "blake3 hash mismatch for kernel={}", id.as_str());

        // Streaming chunking patterns.
        for &chunk in &[1usize, 7, 31, 32, 63, 64, 65, 256, 1024, 4096] {
          let mut h = hasher_for_kernel(id);
          for part in msg.chunks(chunk) {
            h.update(part);
          }
          assert_eq!(
            h.finalize(),
            ours,
            "blake3 streaming mismatch kernel={} len={} chunk={}",
            id.as_str(),
            len,
            chunk
          );
        }

        // Keyed hash mode.
        {
          let mut h = keyed_hasher_for_kernel(id, KEY);
          for part in msg.chunks(63) {
            h.update(part);
          }
          let ours = h.finalize();
          let expected = *blake3::keyed_hash(KEY, &msg).as_bytes();
          assert_eq!(ours, expected, "blake3 keyed mismatch kernel={}", id.as_str());
        }

        // Derive-key mode.
        {
          let mut h = derive_hasher_for_kernel(id, CONTEXT);
          for part in msg.chunks(65) {
            h.update(part);
          }
          let ours = h.finalize();
          let expected = {
            let mut hh = blake3::Hasher::new_derive_key(CONTEXT);
            hh.update(&msg);
            *hh.finalize().as_bytes()
          };
          assert_eq!(ours, expected, "blake3 derive mismatch kernel={}", id.as_str());
        }
      }
    }
  }

  #[test]
  fn xof_prefix_matches_official_crate() {
    let caps = platform::caps();
    let data = pattern(1234);

    for &id in ALL {
      if !caps.has(required_caps(id)) {
        continue;
      }

      let mut ours = [0u8; 131];
      {
        let mut h = hasher_for_kernel(id);
        h.update(&data);
        let mut xof = h.finalize_xof();
        xof.squeeze(&mut ours);
      }

      let mut expected = [0u8; 131];
      {
        let mut h = blake3::Hasher::new();
        h.update(&data);
        let mut out = h.finalize_xof();
        out.fill(&mut expected);
      }

      assert_eq!(ours, expected, "blake3 xof mismatch kernel={}", id.as_str());
    }
  }

  #[test]
  fn run_all_agree() {
    verify_blake3_kernels(b"abc").expect("kernels should agree");
    verify_blake3_kernels(&pattern(8192)).expect("kernels should agree");
  }
}
