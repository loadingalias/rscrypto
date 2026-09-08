use crate::traits::Digest as _;
use alloc::vec::Vec;

use super::{
  Sha224,
  kernels::{ALL, Sha224KernelId, compress_blocks_fn, required_caps},
};

fn digest_with_kernel<'a>(id: Sha224KernelId, chunks: impl IntoIterator<Item = &'a [u8]>) -> [u8; 28] {
  let compress = compress_blocks_fn(id);
  let mut h = Sha224::default();
  for chunk in chunks {
    if !chunk.is_empty() {
      h.update_with(chunk, compress);
    }
  }
  h.finalize_inner_with(compress)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  #[cfg(target_arch = "x86_64")]
  fn x86_dispatch_requires_sha_and_sse41() {
    use crate::platform::{Caps, caps::x86};

    let required = required_caps(Sha224KernelId::X86Sha);
    for (caps, expected) in [
      (Caps::NONE, Sha224KernelId::Portable),
      (x86::SHA, Sha224KernelId::Portable),
      (x86::SSE41, Sha224KernelId::Portable),
      (x86::SHA | x86::SSE41, Sha224KernelId::X86Sha),
    ] {
      assert_eq!(caps.has(required), expected == Sha224KernelId::X86Sha);
      let kernel = super::super::dispatch_policy::select_runtime_kernel(caps);
      assert_eq!(kernel, expected);
    }
  }

  #[test]
  fn compression_kernels_match_portable_at_all_alignments() {
    let caps = crate::platform::caps();
    let input = pattern(4096 + 63);
    for &id in ALL {
      if !caps.has(required_caps(id)) {
        continue;
      }
      let compress = compress_blocks_fn(id);
      for offset in 0..64 {
        for len in [0, 64, 128, 256, 4096] {
          let blocks = &input[offset..offset + len];
          for initial in [super::super::H0, crate::hashes::crypto::sha256::H0] {
            let mut expected = initial;
            let mut actual = initial;
            crate::hashes::crypto::sha256::Sha256::compress_blocks_portable(&mut expected, blocks);
            compress(&mut actual, blocks);
            assert_eq!(actual, expected, "kernel={} offset={offset} len={len}", id.as_str());
          }
        }
      }
    }
  }

  fn pattern(len: usize) -> Vec<u8> {
    (0..len)
      .map(|i| {
        let bytes = i.to_le_bytes();
        bytes[0].wrapping_mul(17).wrapping_add(bytes[1])
      })
      .collect()
  }

  #[test]
  fn all_kernels_match_sha2_oracle_and_streaming_splits() {
    let caps = crate::platform::caps();
    #[cfg(not(miri))]
    let lens = [
      0usize, 1, 2, 3, 55, 56, 57, 63, 64, 65, 119, 120, 121, 127, 128, 129, 1000, 4096,
    ];
    #[cfg(miri)]
    let lens = [0usize, 1, 55, 56, 57, 63, 64, 65, 127, 128, 129];
    #[cfg(not(miri))]
    let chunks = [1usize, 7, 31, 32, 63, 64, 65, 128, 1024];
    #[cfg(miri)]
    let chunks = [1usize, 31, 32, 63, 64, 65, 128];

    for &id in ALL {
      if !caps.has(required_caps(id)) {
        continue;
      }

      for &len in &lens {
        let msg = pattern(len);
        let ours = digest_with_kernel(id, [msg.as_slice()]);

        use sha2::Digest as _;
        let expected = sha2::Sha224::digest(&msg);
        let mut exp = [0u8; 28];
        exp.copy_from_slice(&expected);
        assert_eq!(ours, exp, "sha224 oracle mismatch for kernel={} len={len}", id.as_str());

        for &chunk in &chunks {
          assert_eq!(
            digest_with_kernel(id, msg.chunks(chunk)),
            ours,
            "sha224 chunk mismatch kernel={} len={} chunk={}",
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
            assert_eq!(
              digest_with_kernel(id, [a, b]),
              ours,
              "sha224 split mismatch kernel={} len={} split={}",
              id.as_str(),
              len,
              split
            );
          }
        }
      }
    }
  }
}

#[test]
fn forced_backend_survives_initialization_and_clone_until_reset() {
  fn observed(state: &mut [u32; 8], _blocks: &[u8]) {
    state[0] = 42;
  }

  let mut hasher = Sha224 {
    compress_blocks: observed,
    dispatch_initialized: true,
    ..Default::default()
  };
  let mut cloned = hasher.clone();
  for h in [&mut hasher, &mut cloned] {
    let mut state = [0; 8];
    h.select_compress()(&mut state, &[]);
    assert_eq!(state[0], 42, "forced callback was replaced during selection");
    h.reset();
    assert!(!h.dispatch_initialized);
    h.update(b"abc");
    assert_eq!(h.finalize(), Sha224::digest(b"abc"));
  }
}
