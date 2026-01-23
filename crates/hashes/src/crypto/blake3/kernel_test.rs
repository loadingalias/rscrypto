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
  use alloc::vec;

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
  fn bench_helpers_match_forced_kernel_streaming() {
    let caps = platform::caps();
    let msg = pattern(4096 + 17);

    for &id in ALL {
      if !caps.has(required_caps(id)) {
        continue;
      }

      let oneshot = Blake3::digest_with_kernel_id(id, &msg);
      assert_eq!(
        oneshot,
        digest_with_kernel(id, &msg),
        "digest_with_kernel_id mismatch for kernel={}",
        id.as_str()
      );

      for &chunk in &[64usize, 4096] {
        let helper_stream = Blake3::stream_chunks_with_kernel_id(id, chunk, &msg);

        let mut h = hasher_for_kernel(id);
        for part in msg.chunks(chunk) {
          h.update(part);
        }
        assert_eq!(
          helper_stream,
          h.finalize(),
          "stream_chunks_with_kernel_id mismatch for kernel={} chunk={}",
          id.as_str(),
          chunk
        );
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

  #[test]
  fn oneshot_digest_matches_official_crate() {
    // This exercises the `dispatch::digest` fast path (including the multi-chunk
    // oneshot implementation) rather than the streaming `update` API.
    let lens = [
      0usize, 1, 2, 3, 63, 64, 65, 1023, 1024, 1025, 2047, 2048, 2049, 4096, 8192, 65_536, 1_048_576,
    ];

    for &len in &lens {
      let msg = pattern(len);
      let ours = Blake3::digest(&msg);
      let expected = *blake3::hash(&msg).as_bytes();
      assert_eq!(ours, expected, "blake3 oneshot mismatch len={len}");
    }
  }

  #[test]
  fn oneshot_keyed_and_derive_match_official_crate() {
    let lens = [0usize, 1, 3, 64, 65, 1024, 4096, 10_000];

    for &len in &lens {
      let msg = pattern(len);

      let ours = Blake3::keyed_digest(KEY, &msg);
      let expected = *blake3::keyed_hash(KEY, &msg).as_bytes();
      assert_eq!(ours, expected, "blake3 keyed oneshot mismatch len={len}");

      let ours = Blake3::derive_key(CONTEXT, &msg);
      let expected = blake3::derive_key(CONTEXT, &msg);
      assert_eq!(ours, expected, "blake3 derive-key oneshot mismatch len={len}");
    }
  }

  /// Test that all hash_many kernel implementations produce identical output.
  /// This tests the multi-chunk throughput path against the portable reference.
  #[test]
  fn hash_many_kernels_agree() {
    use super::super::{CHUNK_LEN, IV, OUT_LEN};

    let caps = platform::caps();

    let num_chunks = 8usize;

    // One contiguous buffer containing `num_chunks` full chunks.
    let mut input = vec![0u8; num_chunks * CHUNK_LEN];
    for chunk_idx in 0..num_chunks {
      let base = chunk_idx * CHUNK_LEN;
      for i in 0..CHUNK_LEN {
        input[base + i] = ((i % 251) as u8).wrapping_add(chunk_idx as u8);
      }
    }

    // Get portable kernel output as reference
    let portable_kernel = kernel_for_id(Blake3KernelId::Portable);
    let mut reference_out = vec![0u8; num_chunks * OUT_LEN];
    // SAFETY: `input` contains `num_chunks` full chunks, and `reference_out` is `num_chunks * OUT_LEN`.
    unsafe {
      (portable_kernel.hash_many_contiguous)(input.as_ptr(), num_chunks, &IV, 0, 0, reference_out.as_mut_ptr());
    }

    // Compare all other kernels against portable
    for &id in ALL {
      if id == Blake3KernelId::Portable {
        continue;
      }
      if !caps.has(required_caps(id)) {
        continue;
      }

      let k = kernel_for_id(id);
      let mut out = vec![0u8; num_chunks * OUT_LEN];
      // SAFETY: `input` contains `num_chunks` full chunks, and `out` is `num_chunks * OUT_LEN`.
      unsafe { (k.hash_many_contiguous)(input.as_ptr(), num_chunks, &IV, 0, 0, out.as_mut_ptr()) };

      assert_eq!(
        out,
        reference_out,
        "hash_many_contiguous mismatch: kernel={} differs from portable",
        id.as_str()
      );
    }
  }

  /// Test hash_many with various input sizes and configurations.
  #[test]
  fn hash_many_various_sizes() {
    use super::super::{CHUNK_LEN, IV, OUT_LEN};

    let caps = platform::caps();

    // Test with 1, 2, 3, 4, 5, 7, 8 chunks to cover edge cases
    for num_chunks in [1, 2, 3, 4, 5, 7, 8] {
      let mut input = vec![0u8; num_chunks * CHUNK_LEN];
      for chunk_idx in 0..num_chunks {
        let base = chunk_idx * CHUNK_LEN;
        for i in 0..CHUNK_LEN {
          input[base + i] = ((i % 251) as u8).wrapping_add(chunk_idx as u8);
        }
      }

      // Get portable reference
      let portable_kernel = kernel_for_id(Blake3KernelId::Portable);
      let mut reference_out = vec![0u8; num_chunks * OUT_LEN];
      // SAFETY: `input` contains `num_chunks` full chunks, and `reference_out` is `num_chunks * OUT_LEN`.
      unsafe {
        (portable_kernel.hash_many_contiguous)(input.as_ptr(), num_chunks, &IV, 0, 0, reference_out.as_mut_ptr());
      }

      // Compare all kernels
      for &id in ALL {
        if !caps.has(required_caps(id)) {
          continue;
        }

        let k = kernel_for_id(id);
        let mut out = vec![0u8; num_chunks * OUT_LEN];
        // SAFETY: `input` contains `num_chunks` full chunks, and `out` is `num_chunks * OUT_LEN`.
        unsafe { (k.hash_many_contiguous)(input.as_ptr(), num_chunks, &IV, 0, 0, out.as_mut_ptr()) };

        assert_eq!(
          out,
          reference_out,
          "hash_many_contiguous mismatch: kernel={} num_chunks={}",
          id.as_str(),
          num_chunks
        );
      }
    }
  }
}
