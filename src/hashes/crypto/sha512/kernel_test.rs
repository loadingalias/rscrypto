use alloc::vec::Vec;

use super::{
  Sha512,
  kernels::{ALL, Sha512KernelId, compress_blocks_fn, required_caps},
};
use crate::traits::Digest as _;

fn hasher_for_kernel(id: Sha512KernelId) -> Sha512 {
  let compress = compress_blocks_fn(id);
  Sha512 {
    compress_blocks: compress,
    dispatch_initialized: true,
    ..Default::default()
  }
}

fn digest_with_kernel(id: Sha512KernelId, data: &[u8]) -> [u8; 64] {
  let mut h = hasher_for_kernel(id);
  h.update(data);
  h.finalize()
}

fn digest_64_byte_prefix_with_kernel(id: Sha512KernelId, prefix: &[u8; 64], data: &[u8]) -> [u8; 64] {
  Sha512::digest_64_byte_prefix_with(prefix, data, compress_blocks_fn(id))
}

#[cfg(test)]
mod tests {
  use super::*;

  fn pattern(len: usize) -> Vec<u8> {
    (0..len)
      .map(|i| {
        let bytes = i.to_le_bytes();
        bytes[0].wrapping_mul(13).wrapping_add(bytes[1])
      })
      .collect()
  }

  #[test]
  fn all_kernels_match_sha2_oracle_for_oneshot_and_streaming_splits() {
    let caps = crate::platform::caps();
    #[cfg(not(miri))]
    let lens = [
      0usize, 1, 2, 3, 111, 112, 113, 127, 128, 129, 239, 240, 241, 255, 256, 257, 1000, 4096,
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
        let oneshot = super::super::dispatch::digest_oneshot(&msg, compress_blocks_fn(id));
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
  fn all_kernels_match_sha2_oracle_for_64_byte_prefix() {
    let caps = crate::platform::caps();
    #[cfg(not(miri))]
    let lens = [
      0usize, 1, 47, 48, 49, 63, 64, 65, 111, 112, 113, 127, 128, 129, 191, 192, 193, 255, 256, 257, 1000, 4096,
    ];
    #[cfg(miri)]
    let lens = [0usize, 1, 63, 64, 65, 111, 112, 113, 127, 128, 129, 255, 256];

    let mut prefix = [0u8; 64];
    for (i, byte) in prefix.iter_mut().enumerate() {
      *byte = u8::try_from(i)
        .expect("SHA-512 prefix index fits in u8")
        .wrapping_mul(17)
        .wrapping_add(0x5a);
    }

    for &id in ALL {
      if !caps.has(required_caps(id)) {
        continue;
      }

      for &len in &lens {
        let msg = pattern(len);
        let mut oracle_input = Vec::with_capacity(prefix.len().strict_add(msg.len()));
        oracle_input.extend_from_slice(&prefix);
        oracle_input.extend_from_slice(&msg);

        let actual = digest_64_byte_prefix_with_kernel(id, &prefix, &msg);
        let expected = super::super::dispatch::digest_oneshot(&oracle_input, compress_blocks_fn(id));

        assert_eq!(
          actual,
          expected,
          "sha512 64-byte-prefix mismatch kernel={} len={}",
          id.as_str(),
          len
        );
      }
    }
  }
}

#[test]
fn forced_backend_survives_initialization_and_clone_until_reset() {
  fn observed(state: &mut [u64; 8], _blocks: &[u8]) {
    state[0] = 42;
  }

  let mut hasher = Sha512 {
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
    assert_eq!(h.finalize(), Sha512::digest(b"abc"));
  }
}
