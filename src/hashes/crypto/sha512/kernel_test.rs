use alloc::vec::Vec;

use super::{
  Sha512,
  kernels::{ALL, Sha512KernelId, compress_blocks_fn, required_caps},
};
use crate::{hashes::crypto::dispatch_util::SizeClassDispatch, traits::Digest as _};

#[allow(dead_code)]
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

fn digest_oneshot_with_kernel(id: Sha512KernelId, data: &[u8]) -> [u8; 64] {
  let compress = compress_blocks_fn(id);
  let mut state = super::H0;

  let (blocks, rest) = data.as_chunks::<{ super::BLOCK_LEN }>();
  if !blocks.is_empty() {
    compress(&mut state, &data[..blocks.len().strict_mul(super::BLOCK_LEN)]);
  }

  let total_bits = (data.len() as u128) << 3;
  let mut block = [0u8; super::BLOCK_LEN];
  block[..rest.len()].copy_from_slice(rest);
  block[rest.len()] = 0x80;

  if rest.len() >= 112 {
    compress(&mut state, &block);
    block = [0u8; super::BLOCK_LEN];
  }

  block[112..128].copy_from_slice(&total_bits.to_be_bytes());
  compress(&mut state, &block);

  let mut out = [0u8; 64];
  for (chunk, &word) in out.chunks_exact_mut(8).zip(state.iter()) {
    chunk.copy_from_slice(&word.to_be_bytes());
  }
  out
}

#[allow(dead_code)]
#[must_use]
pub fn run_all_sha512_kernels(data: &[u8]) -> Vec<KernelResult> {
  let caps = crate::platform::caps();
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

#[allow(dead_code)]
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
  fn all_kernels_match_sha2_oracle_for_oneshot_and_streaming_splits() {
    let caps = crate::platform::caps();
    #[cfg(not(miri))]
    let lens = [
      0usize, 1, 2, 3, 111, 112, 113, 127, 128, 129, 239, 240, 241, 255, 256, 257, 1000,
    ];
    #[cfg(miri)]
    let lens = [0usize, 1, 111, 112, 113, 127, 128, 129, 255, 256, 257];
    #[cfg(not(miri))]
    let chunks = [1usize, 7, 31, 32, 63, 64, 65, 127, 128, 129, 1024, 4096];
    #[cfg(miri)]
    let chunks = [1usize, 31, 32, 63, 64, 65, 127, 128, 129];

    for &id in ALL {
      if !caps.has(required_caps(id)) {
        continue;
      }

      for &len in &lens {
        let msg = pattern(len);
        let oneshot = digest_oneshot_with_kernel(id, &msg);
        let ours = digest_with_kernel(id, &msg);

        use sha2::Digest as _;
        let expected = sha2::Sha512::digest(&msg);
        let mut exp = [0u8; 64];
        exp.copy_from_slice(&expected);
        assert_eq!(
          oneshot,
          exp,
          "sha512 oneshot oracle mismatch for kernel={}",
          id.as_str()
        );
        assert_eq!(ours, exp, "sha512 oracle mismatch for kernel={}", id.as_str());

        for &chunk in &chunks {
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

        // Exhaustive two-split for small buffers (padding edges).
        #[cfg(not(miri))]
        let split_limit = 256;
        #[cfg(miri)]
        let split_limit = 128;
        if len <= split_limit {
          for split in 0..=len {
            let (a, b) = msg.split_at(split);
            let mut h = hasher_for_kernel(id);
            h.update(a);
            h.update(b);
            assert_eq!(
              h.finalize(),
              ours,
              "sha512 split mismatch kernel={} len={} split={}",
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
    verify_sha512_kernels(b"abc").expect("kernels should agree");
    verify_sha512_kernels(&pattern(4096)).expect("kernels should agree");
  }
}
