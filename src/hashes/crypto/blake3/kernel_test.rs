use alloc::vec::Vec;

use super::{
  Blake3,
  kernels::{Blake3KernelId, kernel as kernel_for_id, required_caps},
};
use crate::traits::Digest as _;

const ALL: &[Blake3KernelId] = &[
  Blake3KernelId::Portable,
  #[cfg(target_arch = "x86_64")]
  Blake3KernelId::X86Sse41,
  #[cfg(target_arch = "x86_64")]
  Blake3KernelId::X86Avx2,
  #[cfg(target_arch = "x86_64")]
  Blake3KernelId::X86Avx512,
  #[cfg(target_arch = "aarch64")]
  Blake3KernelId::Aarch64Neon,
  #[cfg(all(target_arch = "s390x", not(feature = "portable-only")))]
  Blake3KernelId::S390xVector,
  #[cfg(all(target_arch = "powerpc64", not(feature = "portable-only")))]
  Blake3KernelId::PowerVsx,
  #[cfg(all(target_arch = "riscv64", not(feature = "portable-only")))]
  Blake3KernelId::RiscvV,
  #[cfg(all(target_arch = "wasm32", target_feature = "simd128", not(feature = "portable-only")))]
  Blake3KernelId::WasmSimd128,
];

#[derive(Clone, Debug)]
pub(super) struct KernelResult {
  pub digest: [u8; 32],
}

fn force_hasher_kernel(mut h: Blake3, id: Blake3KernelId) -> Blake3 {
  let kernel = kernel_for_id(id);
  h.bulk_kernel_id = kernel.id;
  h.chunk_state.kernel_id = kernel.id;
  h
}

fn hasher_for_kernel(id: Blake3KernelId) -> Blake3 {
  force_hasher_kernel(Blake3::new(), id)
}

fn digest_with_kernel(id: Blake3KernelId, data: &[u8]) -> [u8; 32] {
  let kernel = kernel_for_id(id);
  let mut h = hasher_for_kernel(id);
  h.update_with(data, kernel.id, kernel.id);
  h.finalize()
}

#[must_use]
pub(super) fn run_all_blake3_kernels(data: &[u8]) -> Vec<KernelResult> {
  let caps = crate::platform::caps();
  let mut out = Vec::with_capacity(ALL.len());
  for &id in ALL {
    if caps.has(required_caps(id)) {
      out.push(KernelResult {
        digest: digest_with_kernel(id, data),
      });
    }
  }
  out
}

pub(super) fn verify_blake3_kernels(data: &[u8]) -> Result<(), &'static str> {
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

  use super::*;
  use crate::traits::Xof as _;

  const KEY: &[u8; 32] = b"whats the Elvish word for friend";
  const CONTEXT: &str = "BLAKE3 2019-12-27 16:29:52 test vectors context";

  fn keyed_hasher_for_kernel(id: Blake3KernelId, key: &[u8; 32]) -> Blake3 {
    force_hasher_kernel(Blake3::new_keyed(key), id)
  }

  fn derive_hasher_for_kernel(id: Blake3KernelId, context: &str) -> Blake3 {
    force_hasher_kernel(Blake3::new_derive_key(context), id)
  }

  fn pattern(len: usize) -> Vec<u8> {
    (0..len)
      .map(|i| u8::try_from(i % 251).expect("test pattern byte fits in u8"))
      .collect()
  }

  #[test]
  fn all_kernels_match_official_crate_and_streaming_splits() {
    let caps = crate::platform::caps();
    #[cfg(not(miri))]
    let lens = [0usize, 1, 2, 3, 63, 64, 65, 1023, 1024, 1025, 2047, 2048, 2049, 10_000];
    #[cfg(miri)]
    let lens = [0usize, 1, 63, 64, 65, 1023, 1024, 1025, 2048];
    #[cfg(not(miri))]
    let chunks = [1usize, 7, 31, 32, 63, 64, 65, 256, 1024, 4096];
    #[cfg(miri)]
    let chunks = [1usize, 31, 32, 63, 64, 65, 256];

    for &id in ALL {
      if !caps.has(required_caps(id)) {
        continue;
      }

      let kernel = kernel_for_id(id);
      for &len in &lens {
        let msg = pattern(len);

        // Hash mode.
        let ours = digest_with_kernel(id, &msg);
        let expected = *blake3::hash(&msg).as_bytes();
        assert_eq!(ours, expected, "blake3 hash mismatch for kernel={}", id.as_str());

        // Streaming chunking patterns.
        for &chunk in &chunks {
          let mut h = hasher_for_kernel(id);
          for part in msg.chunks(chunk) {
            h.update_with(part, kernel.id, kernel.id);
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
            h.update_with(part, kernel.id, kernel.id);
          }
          let ours = h.finalize();
          let expected = *blake3::keyed_hash(KEY, &msg).as_bytes();
          assert_eq!(ours, expected, "blake3 keyed mismatch kernel={}", id.as_str());
        }

        // Derive-key mode.
        {
          let mut h = derive_hasher_for_kernel(id, CONTEXT);
          for part in msg.chunks(65) {
            h.update_with(part, kernel.id, kernel.id);
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

  #[cfg(target_arch = "aarch64")]
  #[test]
  fn oneshot_1024_fast_path_matches_official_crate() {
    let caps = crate::platform::caps();
    let id = Blake3KernelId::Aarch64Neon;
    if !caps.has(required_caps(id)) {
      return;
    }

    let msg = pattern(1024);

    // Forced-kernel oneshot (exercises the len==1024 fast path).
    let ours = digest_with_kernel(id, &msg);
    let expected = *blake3::hash(&msg).as_bytes();
    assert_eq!(ours, expected, "blake3 oneshot mismatch kernel={}", id.as_str());

    // Keyed oneshot (uses length-based dispatch).
    let ours_keyed = Blake3::keyed_digest(KEY, &msg);
    let expected_keyed = *blake3::keyed_hash(KEY, &msg).as_bytes();
    assert_eq!(ours_keyed.as_bytes(), &expected_keyed, "blake3 keyed oneshot mismatch");

    // Derive-key oneshot (uses length-based dispatch for key material).
    let ours_derived = Blake3::derive_key(CONTEXT, &msg);
    let expected_derived = {
      let mut h = blake3::Hasher::new_derive_key(CONTEXT);
      h.update(&msg);
      *h.finalize().as_bytes()
    };
    assert_eq!(ours_derived, expected_derived, "blake3 derive-key oneshot mismatch");
  }

  #[test]
  fn xof_prefix_matches_official_crate() {
    let caps = crate::platform::caps();
    let data = pattern(1234);

    for &id in ALL {
      if !caps.has(required_caps(id)) {
        continue;
      }

      let mut ours = [0u8; 131];
      {
        let kernel = kernel_for_id(id);
        let mut h = hasher_for_kernel(id);
        h.update_with(&data, kernel.id, kernel.id);
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
  fn single_chunk_xof_prefix_matches_official_crate_for_forced_kernels() {
    let caps = crate::platform::caps();
    let lens = [0usize, 1, 63, 64, 65, 511, 1024];

    for &id in ALL {
      if !caps.has(required_caps(id)) {
        continue;
      }

      let kernel = kernel_for_id(id);
      for &len in &lens {
        let msg = pattern(len);

        let mut ours = [0u8; 96];
        {
          let mut h = hasher_for_kernel(id);
          for part in msg.chunks(13) {
            h.update_with(part, kernel.id, kernel.id);
          }
          let mut xof = h.finalize_xof();
          xof.squeeze(&mut ours[..32]);
          xof.squeeze(&mut ours[32..]);
        }

        let mut expected = [0u8; 96];
        {
          let mut h = blake3::Hasher::new();
          for part in msg.chunks(13) {
            h.update(part);
          }
          let mut xof = h.finalize_xof();
          xof.fill(&mut expected[..32]);
          xof.fill(&mut expected[32..]);
        }

        assert_eq!(
          ours,
          expected,
          "single-chunk xof mismatch kernel={} len={len}",
          id.as_str()
        );
      }
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
    #[cfg(not(miri))]
    let lens = [
      0usize, 1, 2, 3, 63, 64, 65, 1023, 1024, 1025, 2047, 2048, 2049, 4096, 8192, 65_536, 1_048_576,
    ];
    #[cfg(miri)]
    let lens = [0usize, 1, 63, 64, 65, 1023, 1024, 1025, 4096, 8192];

    for &len in &lens {
      let msg = pattern(len);
      let ours = Blake3::digest(&msg);
      let expected = *blake3::hash(&msg).as_bytes();
      assert_eq!(ours, expected, "blake3 oneshot mismatch len={len}");
    }
  }

  #[test]
  fn oneshot_keyed_and_derive_match_official_crate() {
    #[cfg(not(miri))]
    let lens = [0usize, 1, 3, 64, 65, 1024, 4096, 10_000];
    #[cfg(miri)]
    let lens = [0usize, 1, 64, 65, 1024, 4096];

    for &len in &lens {
      let msg = pattern(len);

      let ours = Blake3::keyed_digest(KEY, &msg);
      let expected = *blake3::keyed_hash(KEY, &msg).as_bytes();
      assert_eq!(ours.as_bytes(), &expected, "blake3 keyed oneshot mismatch len={len}");

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

    let caps = crate::platform::caps();

    let num_chunks = 8usize;

    // One contiguous buffer containing `num_chunks` full chunks.
    let mut input = vec![0u8; num_chunks * CHUNK_LEN];
    for chunk_idx in 0..num_chunks {
      let base = chunk_idx * CHUNK_LEN;
      for i in 0..CHUNK_LEN {
        let byte = u8::try_from(i % 251).expect("test pattern byte fits in u8");
        let chunk = u8::try_from(chunk_idx).expect("test chunk index fits in u8");
        input[base + i] = byte.wrapping_add(chunk);
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

    let caps = crate::platform::caps();

    // Test with 1, 2, 3, 4, 5, 7, 8 chunks to cover edge cases
    for num_chunks in [1, 2, 3, 4, 5, 7, 8] {
      let mut input = vec![0u8; num_chunks * CHUNK_LEN];
      for chunk_idx in 0..num_chunks {
        let base = chunk_idx * CHUNK_LEN;
        for i in 0..CHUNK_LEN {
          let byte = u8::try_from(i % 251).expect("test pattern byte fits in u8");
          let chunk = u8::try_from(chunk_idx).expect("test chunk index fits in u8");
          input[base + i] = byte.wrapping_add(chunk);
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

  #[test]
  fn hash_many_unaligned_tails_match_independent_cvs() {
    use blake3::hazmat::HasherExt as _;

    use super::super::{CHUNK_LEN, DERIVE_KEY_MATERIAL, IV, KEYED_HASH, OUT_LEN};

    let caps = crate::platform::caps();
    let context_key = blake3::hazmat::hash_derive_key_context(CONTEXT);
    let words = |bytes: &[u8; 32]| {
      core::array::from_fn(|i| u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().expect("key word")))
    };
    let modes = [
      (0, IV, blake3::Hasher::new()),
      (KEYED_HASH, words(KEY), blake3::Hasher::new_keyed(KEY)),
      (
        DERIVE_KEY_MATERIAL,
        words(&context_key),
        blake3::Hasher::new_from_context_key(&context_key),
      ),
    ];
    let portable = kernel_for_id(Blake3KernelId::Portable);

    // Include each remainder before and after four/eight-leaf groups. The high
    // counter crosses the low-word boundary inside both vector groups and tails.
    for num_chunks in 1usize..=11 {
      let message = pattern(num_chunks * CHUNK_LEN);
      let out_len = num_chunks * OUT_LEN;
      for counter in [0u64, (1u64 << 32) - 3] {
        for (flags, key, initial) in &modes {
          let mut expected = Vec::with_capacity(out_len);
          for (index, chunk) in message.as_chunks::<CHUNK_LEN>().0.iter().enumerate() {
            let mut oracle = initial.clone();
            let chunk_counter = counter + u64::try_from(index).expect("chunk index fits");
            oracle.set_input_offset(chunk_counter * 1024).update(chunk);
            expected.extend_from_slice(&oracle.finalize_non_root());
          }
          let mut reference = vec![0u8; out_len];
          // SAFETY: message and reference cover exactly num_chunks chunks/CVs.
          unsafe {
            (portable.hash_many_contiguous)(
              message.as_ptr(),
              num_chunks,
              key,
              counter,
              *flags,
              reference.as_mut_ptr(),
            );
          }
          assert_eq!(
            reference, expected,
            "Portable CVs, chunks={num_chunks}, counter={counter}, flags={flags}"
          );

          for input_offset in [0usize, 1, 4, 7, 8, 15] {
            let mut input = vec![0u8; message.len() + 32];
            let input_start = (16 - input.as_ptr() as usize % 16) % 16 + input_offset;
            input[input_start..input_start + message.len()].copy_from_slice(&message);
            for output_offset in [0usize, 1, 3, 4, 7] {
              for &id in ALL {
                if !caps.has(required_caps(id)) {
                  continue;
                }
                let mut output = vec![0xa5u8; out_len + 32];
                let start = (16 - output.as_ptr() as usize % 16) % 16 + output_offset;
                let end = start + out_len;
                // SAFETY: input/output are disjoint initialized allocations with
                // full chunk/CV ranges at every offset; required CPU caps passed.
                unsafe {
                  (kernel_for_id(id).hash_many_contiguous)(
                    input.as_ptr().add(input_start),
                    num_chunks,
                    key,
                    counter,
                    *flags,
                    output.as_mut_ptr().add(start),
                  );
                }
                assert_eq!(
                  output[start..end],
                  expected,
                  "CVs: kernel={}, chunks={num_chunks}, counter={counter}, flags={flags}, in={input_offset}, out={output_offset}",
                  id.as_str()
                );
                assert!(
                  output[..start].iter().all(|&byte| byte == 0xa5),
                  "output prefix overwritten"
                );
                assert!(
                  output[end..].iter().all(|&byte| byte == 0xa5),
                  "output suffix overwritten"
                );
              }
            }
          }
        }
      }
    }
  }

  #[cfg(all(feature = "std", not(miri)))]
  #[test]
  fn large_inputs_match_official_crate() {
    // This is sized to reliably cross the std-only parallelization thresholds.
    // If the test runner only exposes 1 CPU, we still validate correctness,
    // but the parallel path may not execute.
    let lens = [512 * 1024 + 123, 4 * 1024 * 1024 + 17];

    for &len in &lens {
      let msg = pattern(len);

      // One-shot hashing (may take the parallel path).
      let ours = Blake3::digest(&msg);
      let expected = *blake3::hash(&msg).as_bytes();
      assert_eq!(ours, expected, "blake3 oneshot mismatch len={}", len);

      // Streaming hashing with a single large update (may take the parallel path).
      let mut h = Blake3::new();
      h.update(&msg);
      let ours_stream = h.finalize();
      assert_eq!(ours_stream, expected, "blake3 streaming mismatch len={}", len);

      // Keyed hash one-shot.
      let ours_keyed = Blake3::keyed_digest(KEY, &msg);
      let expected_keyed = *blake3::keyed_hash(KEY, &msg).as_bytes();
      assert_eq!(
        ours_keyed.as_bytes(),
        &expected_keyed,
        "blake3 keyed mismatch len={}",
        len
      );

      // Derive-key one-shot.
      let ours_derived = Blake3::derive_key(CONTEXT, &msg);
      let expected_derived = {
        let mut hh = blake3::Hasher::new_derive_key(CONTEXT);
        hh.update(&msg);
        *hh.finalize().as_bytes()
      };
      assert_eq!(ours_derived, expected_derived, "blake3 derive-key mismatch len={}", len);
    }
  }

  #[cfg(any(
    all(target_arch = "aarch64", target_endian = "little", not(feature = "portable-only")),
    all(target_arch = "x86_64", target_feature = "sse2")
  ))]
  #[test]
  fn partial_forests_match_portable_and_upstream() {
    use blake3::hazmat::HasherExt as _;

    use super::super::{CHUNK_LEN, DERIVE_KEY_CONTEXT, DERIVE_KEY_MATERIAL, IV, KEYED_HASH, words8_to_le_bytes};

    #[cfg(target_arch = "aarch64")]
    let kernels = [(Blake3KernelId::Aarch64Neon, vec![3usize, 7, 11, 15])];
    #[cfg(target_arch = "x86_64")]
    let kernels = {
      let mut shapes = Vec::new();
      if super::super::kernels::avx512vl_partial_lane_available() {
        shapes.push(3usize);
      }
      if super::super::kernels::avx512_partial_lane_available() {
        shapes.push(15);
      }
      [(Blake3KernelId::X86Avx512, shapes)]
    };
    let portable = Blake3KernelId::Portable;
    let words = |bytes: &[u8; 32]| {
      core::array::from_fn(|i| u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().expect("key word")))
    };
    let context_key = blake3::hazmat::hash_derive_key_context(CONTEXT);
    let modes = [
      (0, IV, Some(blake3::Hasher::new())),
      (KEYED_HASH, words(KEY), Some(blake3::Hasher::new_keyed(KEY))),
      (
        DERIVE_KEY_MATERIAL,
        words(&context_key),
        Some(blake3::Hasher::new_from_context_key(&context_key)),
      ),
      (DERIVE_KEY_CONTEXT, IV, None),
    ];
    let storage = pattern(17 * CHUNK_LEN + 15);
    for (id, shapes) in kernels {
      if !crate::platform::caps().has(required_caps(id)) {
        continue;
      }
      // Shapes outside a kernel's exact lane groups keep the existing route.
      for full in [3usize, 7, 11, 15].into_iter().filter(|full| !shapes.contains(full)) {
        let mut declined = force_hasher_kernel(Blake3::new(), id);
        assert_eq!(
          declined.try_partial_forest_update(&storage[..full * CHUNK_LEN + 1]),
          None
        );
      }
      for full in shapes {
        for offset in [0usize, 1, 7, 15] {
          let data = &storage[offset..];
          for counter in [0u64, 16, (1 << 32) - 16, (1 << 54) - 16] {
            for (flags, key, initial) in &modes {
              let mut seed = Blake3::new_internal(*key, *flags);
              seed.chunk_state.chunk_counter = counter;
              let mut expected_frontier = seed.clone();
              expected_frontier.update_with(&data[..full * CHUNK_LEN + 1], portable, portable);
              if let Some(initial) = initial {
                let mut start = 0usize;
                // Portable's canonical frontier is checked against independent
                // upstream subtrees before it supplies the state expectation.
                for (index, level) in (0..4).rev().filter(|level| full & (1 << level) != 0).enumerate() {
                  let count = 1usize << level;
                  let mut oracle = initial.clone();
                  oracle.set_input_offset((counter + u64::try_from(start).expect("offset")) * 1024);
                  oracle.update(&data[start * CHUNK_LEN..(start + count) * CHUNK_LEN]);
                  // SAFETY: the Portable update initialized each live stack slot.
                  let cv = unsafe { expected_frontier.cv_stack[index].assume_init_ref() };
                  assert_eq!(words8_to_le_bytes(cv), oracle.finalize_non_root());
                  start += count;
                }
              }
              for partial in 1usize..CHUNK_LEN {
                let end = full * CHUNK_LEN + partial;
                let mut actual = force_hasher_kernel(seed.clone(), id);
                assert_eq!(actual.try_partial_forest_update(&data[..end]), Some(end));
                let mut expected = expected_frontier.clone();
                expected.update_with(&data[full * CHUNK_LEN + 1..end], portable, portable);
                let context = alloc::format!("{id:?}/{full}/{offset}/{counter}/{flags}/{partial}");
                assert_eq!(
                  actual.chunk_state.chunk_counter, expected.chunk_state.chunk_counter,
                  "{context}"
                );
                assert_eq!(
                  actual.chunk_state.chaining_value, expected.chunk_state.chaining_value,
                  "{context}"
                );
                assert_eq!(
                  actual.chunk_state.blocks_compressed, expected.chunk_state.blocks_compressed,
                  "{context}"
                );
                assert_eq!(
                  actual.chunk_state.block_len, expected.chunk_state.block_len,
                  "{context}"
                );
                assert_eq!(actual.chunk_state.block, expected.chunk_state.block, "{context}");
                assert!(actual.pending_chunk_cv.is_none(), "{context}");
                assert_eq!(actual.cv_stack_len, expected.cv_stack_len, "{context}");
                for index in 0..usize::from(actual.cv_stack_len) {
                  // SAFETY: both updates initialized their independently tracked
                  // live slots, and the equal lengths bound this shared read.
                  unsafe {
                    assert_eq!(
                      actual.cv_stack[index].assume_init_ref(),
                      expected.cv_stack[index].assume_init_ref(),
                      "{context}"
                    );
                  }
                }
                if counter == 0 {
                  let mut got = [0u8; 131];
                  let mut want = [0u8; 131];
                  actual.clone().finalize_xof().squeeze(&mut got);
                  expected.clone().finalize_xof().squeeze(&mut want);
                  assert_eq!(got, want, "{context}");
                  if let Some(initial) = initial {
                    initial.clone().update(&data[..end]).finalize_xof().fill(&mut want);
                    assert_eq!(got, want, "{context}");
                  }
                  actual.update_with(&[], id, id);
                  actual.update_with(&data[end..end + 70], id, id);
                  expected.update_with(&data[end..end + 70], portable, portable);
                  assert_eq!(actual.finalize(), expected.finalize(), "{context}");
                }
              }
            }
          }
        }
      }
    }
  }
}
