use alloc::vec::Vec;

use super::{
  Sha256,
  kernels::{Sha256KernelId, required_caps},
};

pub(crate) const ALL: &[Sha256KernelId] = &[
  Sha256KernelId::Portable,
  #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
  Sha256KernelId::X86Sha,
  #[cfg(target_arch = "aarch64")]
  Sha256KernelId::Aarch64Sha2,
  #[cfg(all(
    any(target_arch = "riscv64", target_arch = "riscv32"),
    not(feature = "portable-only")
  ))]
  Sha256KernelId::RiscvZknh,
  #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
  Sha256KernelId::WasmSimd128,
  #[cfg(target_arch = "s390x")]
  Sha256KernelId::S390xKimd,
];

fn digest_with_kernel<'a>(id: Sha256KernelId, chunks: impl IntoIterator<Item = &'a [u8]>) -> [u8; 32] {
  let mut h = Sha256::default();
  for chunk in chunks {
    if !chunk.is_empty() {
      h.update_with(chunk, id);
    }
  }
  h.finalize_inner_with::<false>(id)
}

#[cfg(test)]
mod tests {
  use super::*;

  fn pattern(len: usize) -> Vec<u8> {
    (0..len)
      .map(|i| {
        let bytes = i.to_le_bytes();
        bytes[0].wrapping_mul(31).wrapping_add(bytes[1])
      })
      .collect()
  }

  fn all_chunk_sizes() -> &'static [usize] {
    #[cfg(not(miri))]
    {
      &[
        1, 2, 3, 7, 8, 15, 16, 31, 32, 63, 64, 65, 127, 128, 255, 256, 1024, 4096,
      ]
    }
    #[cfg(miri)]
    {
      &[1, 7, 31, 32, 63, 64, 65, 127, 128]
    }
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

    for &id in ALL {
      if !caps.has(required_caps(id)) {
        continue;
      }

      for &len in &lens {
        let msg = pattern(len);
        let ours = digest_with_kernel(id, [msg.as_slice()]);

        use sha2::Digest as _;
        let expected = sha2::Sha256::digest(&msg);
        let mut exp = [0u8; 32];
        exp.copy_from_slice(&expected);
        assert_eq!(ours, exp, "sha256 oracle mismatch for kernel={} len={len}", id.as_str());

        // Streaming chunking patterns.
        for &chunk in all_chunk_sizes() {
          assert_eq!(
            digest_with_kernel(id, msg.chunks(chunk)),
            ours,
            "sha256 streaming mismatch kernel={} len={} chunk={}",
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
  #[cfg(target_arch = "x86_64")]
  fn x86_sha_kernel_requires_sha_and_sse41() {
    let required = required_caps(Sha256KernelId::X86Sha);

    assert!(required.has(crate::platform::caps::x86::SHA));
    assert!(required.has(crate::platform::caps::x86::SSE41));
    assert!(!crate::platform::caps::x86::SHA.has(required));
  }
}

#[test]
fn forced_backend_survives_initialization_and_clone_until_reset() {
  let caps = crate::platform::caps();
  for &id in ALL.iter().filter(|&&id| caps.has(required_caps(id))) {
    let mut hasher = Sha256 {
      compress_blocks: id,
      dispatch_initialized: true,
      ..Default::default()
    };
    let mut cloned = hasher.clone();
    for h in [&mut hasher, &mut cloned] {
      assert!(
        h.select_compress() == id,
        "forced kernel {} was replaced during selection",
        id.as_str()
      );
      h.reset();
      assert!(!h.dispatch_initialized);
      h.update(b"abc");
      assert_eq!(h.finalize(), Sha256::digest(b"abc"));
    }
  }
}
