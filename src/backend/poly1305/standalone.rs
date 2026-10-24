//! Arithmetic owner and backend dispatch for the standalone Poly1305 authenticator.
//!
//! On 64-bit targets the engine keeps a radix-2^64 accumulator. Short inputs run the single-chain
//! scalar kernel inline. Longer inputs leave the caller's frame for one out-of-line dispatch: the
//! AVX-512 IFMA kernel on x86-64 with AVX-512F and AVX-512 IFMA, the NEON kernel (alone, or beside
//! four scalar chains for long inputs) on AArch64, and four scalar chains elsewhere. Dispatch is
//! resolved once per call from the cached CPU capabilities. Other targets, and builds with
//! `portable-only`, run the radix-2^26 portable authority.

#[cfg(not(all(target_pointer_width = "64", not(feature = "portable-only"))))]
use super::State as Inner;
#[cfg(all(target_pointer_width = "64", not(feature = "portable-only")))]
use super::scalar64::State as Inner;

/// Clamped key, accumulator, and pad of one standalone authenticator. Dropping it clears them.
pub(crate) struct Engine(Inner);

/// Fewest blocks that leave the single-chain scalar kernel on this target.
#[cfg(all(target_pointer_width = "64", not(feature = "portable-only")))]
const LONG_MIN_BLOCKS: usize = {
  let wide = super::scalar64::WIDE_MIN_BLOCKS;
  #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
  let vector = super::x86_64_ifma::MIN_BLOCKS;
  #[cfg(all(target_arch = "aarch64", target_endian = "little", target_feature = "neon"))]
  let vector = super::aarch64_neon::MIN_BLOCKS;
  #[cfg(not(any(
    all(target_arch = "x86_64", target_feature = "sse2"),
    all(target_arch = "aarch64", target_endian = "little", target_feature = "neon")
  )))]
  let vector = wide;
  if vector < wide { vector } else { wide }
};

#[cfg(all(
  any(
    all(target_arch = "x86_64", target_feature = "sse2"),
    all(target_arch = "aarch64", target_endian = "little", target_feature = "neon")
  ),
  target_pointer_width = "64",
  not(feature = "portable-only")
))]
#[inline(always)]
fn current_caps() -> crate::platform::Caps {
  #[cfg(feature = "std")]
  {
    crate::platform::caps()
  }

  #[cfg(not(feature = "std"))]
  {
    crate::platform::caps_static()
  }
}

#[cfg(all(target_pointer_width = "64", not(feature = "portable-only")))]
impl Engine {
  #[inline(always)]
  pub(crate) fn new(key: &[u8; 32]) -> Self {
    Self(Inner::new(key))
  }

  /// Absorbs full blocks, each with its 2^128 bit.
  #[inline]
  pub(crate) fn absorb_blocks(&mut self, blocks: &[[u8; 16]]) {
    if blocks.len() < LONG_MIN_BLOCKS {
      self.0.absorb_blocks(blocks);
    } else {
      self.absorb_long(blocks);
    }
  }

  /// Dispatches inputs of at least [`LONG_MIN_BLOCKS`] blocks. Keeping it out of line keeps the
  /// vector kernels' register saves out of the short path.
  #[inline(never)]
  fn absorb_long(&mut self, blocks: &[[u8; 16]]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    if blocks.len() >= super::x86_64_ifma::MIN_BLOCKS {
      use crate::platform::caps::x86;
      let caps = current_caps();
      if caps.has(x86::AVX512F.union(x86::AVX512IFMA)) {
        // AMD cores issue 512-bit integer work on four ports with two-cycle latency, so the shorter
        // dependency chain of the fused final pass pays there.
        let fuse_final = caps.has(x86::AMD);
        // SAFETY: the capability check above establishes AVX-512F and AVX-512 IFMA.
        unsafe { super::x86_64_ifma::absorb_blocks(&mut self.0, blocks, fuse_final) };
        return;
      }
    }

    #[cfg(all(target_arch = "aarch64", target_endian = "little", target_feature = "neon"))]
    if blocks.len() >= super::aarch64_neon::MIN_BLOCKS {
      use crate::platform::caps::aarch64;
      // NEON is a compile-time feature here; the check keeps capability overrides authoritative.
      if current_caps().has(aarch64::NEON) {
        if blocks.len() >= super::aarch64_neon::HYBRID_MIN_BLOCKS {
          super::aarch64_neon::absorb_blocks_hybrid(&mut self.0, blocks);
        } else {
          super::aarch64_neon::absorb_blocks(&mut self.0, blocks);
        }
        return;
      }
    }

    if blocks.len() >= super::scalar64::WIDE_MIN_BLOCKS {
      self.0.absorb_blocks_wide(blocks);
    } else {
      self.0.absorb_blocks(blocks);
    }
  }

  /// Absorbs a caller-framed final fragment without the 2^128 bit.
  #[inline(always)]
  pub(crate) fn absorb_final_fragment(&mut self, block: &[u8; 16]) {
    self.0.absorb_final_fragment(block);
  }

  /// Absorbs a final fragment of 1 to 15 message bytes, framed here with a byte of one.
  #[inline(always)]
  pub(crate) fn absorb_final_bytes(&mut self, fragment: &[u8]) {
    self.0.absorb_final_bytes(fragment);
  }

  #[inline(always)]
  pub(crate) fn tag(&self) -> [u8; 16] {
    self.0.tag()
  }
}

#[cfg(not(all(target_pointer_width = "64", not(feature = "portable-only"))))]
impl Engine {
  #[inline(always)]
  pub(crate) fn new(key: &[u8; 32]) -> Self {
    Self(Inner::new(key))
  }

  /// Absorbs full blocks, each with its 2^128 bit.
  #[inline]
  pub(crate) fn absorb_blocks(&mut self, blocks: &[[u8; 16]]) {
    for block in blocks {
      self.0.compute_block_portable(block, false);
    }
  }

  /// Absorbs a caller-framed final fragment without the 2^128 bit.
  #[inline(always)]
  pub(crate) fn absorb_final_fragment(&mut self, block: &[u8; 16]) {
    self.0.compute_block_portable(block, true);
  }

  /// Absorbs a final fragment of 1 to 15 message bytes, framed here with a byte of one. The framed
  /// stack copy of the fragment is cleared after use.
  #[inline(always)]
  pub(crate) fn absorb_final_bytes(&mut self, fragment: &[u8]) {
    let mut block = [0u8; 16];
    block[..fragment.len()].copy_from_slice(fragment);
    block[fragment.len()] = 1;
    self.0.compute_block_portable(&block, true);
    crate::traits::ct::zeroize(&mut block);
  }

  #[inline(always)]
  pub(crate) fn tag(&self) -> [u8; 16] {
    self.0.tag()
  }
}

#[cfg(all(test, target_pointer_width = "64", not(feature = "portable-only")))]
mod tests {
  use alloc::{vec, vec::Vec};

  use super::super::{State as Portable, scalar64::State as Scalar};

  /// The radix-2^26 portable authority's tag for `message`.
  fn portable_tag(key: &[u8; 32], message: &[u8]) -> [u8; 16] {
    let mut state = Portable::new(key);
    let (blocks, rest) = message.as_chunks::<16>();
    for block in blocks {
      state.compute_block_portable(block, false);
    }
    if !rest.is_empty() {
      let mut block = [0u8; 16];
      block[..rest.len()].copy_from_slice(rest);
      block[rest.len()] = 1;
      state.compute_block_portable(&block, true);
    }
    state.tag()
  }

  /// The tag of `message` with its full blocks absorbed by `absorb`.
  fn backend_tag(key: &[u8; 32], message: &[u8], absorb: fn(&mut Scalar, &[[u8; 16]])) -> [u8; 16] {
    let mut state = Scalar::new(key);
    let (blocks, rest) = message.as_chunks::<16>();
    absorb(&mut state, blocks);
    if !rest.is_empty() {
      let mut block = [0u8; 16];
      block[..rest.len()].copy_from_slice(rest);
      block[rest.len()] = 1;
      state.absorb_final_fragment(&block);
    }
    state.tag()
  }

  fn keys() -> Vec<[u8; 32]> {
    let mut patterned = [0u8; 32];
    for (index, byte) in patterned.iter_mut().enumerate() {
      let [low, ..] = index.strict_mul(29).strict_add(13).to_le_bytes();
      *byte = low;
    }
    // All-ones keys keep every clamped limb and the pad at their bounds; zero keys check the pad
    // and the zero multiplier.
    vec![[0xff; 32], [0; 32], patterned, [0x5a; 32]]
  }

  /// Compares `absorb` with the portable authority on lengths, alignments, and carry-heavy inputs.
  ///
  /// Lengths 0 to 1100 cross every vector group and remainder size; offsets move the input across
  /// every alignment within a cache line. All-ones messages keep every limb at its bound.
  fn exercise(absorb: fn(&mut Scalar, &[[u8; 16]])) {
    let mut buffer = vec![0u8; 70_000];
    for (index, byte) in buffer.iter_mut().enumerate() {
      let [low, ..] = index.strict_mul(131).strict_add(index >> 7).to_le_bytes();
      *byte = low;
    }
    let ones = vec![0xffu8; 70_000];
    for key in keys() {
      for len in 0..=1100usize {
        for offset in [0usize, 1, 7, 33] {
          let message = &buffer[offset..offset.strict_add(len)];
          assert_eq!(
            backend_tag(&key, message, absorb),
            portable_tag(&key, message),
            "len {len}, offset {offset}"
          );
        }
        assert_eq!(
          backend_tag(&key, &ones[..len], absorb),
          portable_tag(&key, &ones[..len]),
          "ones {len}"
        );
      }
      for len in [127usize, 128, 129, 255, 256, 257, 271, 1024, 1039] {
        for offset in 0..64usize {
          let message = &buffer[offset..offset.strict_add(len)];
          assert_eq!(
            backend_tag(&key, message, absorb),
            portable_tag(&key, message),
            "len {len}, offset {offset}"
          );
        }
      }
      for len in [2047usize, 2048, 4111, 16_384, 16_401, 65_541] {
        let message = &buffer[3..3usize.strict_add(len)];
        assert_eq!(
          backend_tag(&key, message, absorb),
          portable_tag(&key, message),
          "len {len}"
        );
        assert_eq!(
          backend_tag(&key, &ones[..len], absorb),
          portable_tag(&key, &ones[..len]),
          "ones {len}"
        );
      }
    }
  }

  #[test]
  fn scalar_matches_portable() {
    exercise(Scalar::absorb_blocks);
  }

  #[test]
  fn wide_scalar_matches_portable() {
    exercise(Scalar::absorb_blocks_wide);
  }

  #[test]
  fn dispatched_engine_matches_portable() {
    exercise(|state, blocks| {
      let mut engine = super::Engine(Scalar::new(&[0; 32]));
      core::mem::swap(&mut engine.0, state);
      engine.absorb_blocks(blocks);
      core::mem::swap(&mut engine.0, state);
    });
  }

  #[test]
  #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
  fn avx512_ifma_matches_portable() {
    use crate::platform::caps::x86;
    if !crate::platform::caps().has(x86::AVX512F.union(x86::AVX512IFMA)) {
      return;
    }
    exercise(|state, blocks| {
      // SAFETY: the capability check above establishes AVX-512F and AVX-512 IFMA.
      unsafe { super::super::x86_64_ifma::absorb_blocks(state, blocks, false) }
    });
    exercise(|state, blocks| {
      // SAFETY: the capability check above establishes AVX-512F and AVX-512 IFMA.
      unsafe { super::super::x86_64_ifma::absorb_blocks(state, blocks, true) }
    });
  }

  #[test]
  #[cfg(all(target_arch = "aarch64", target_endian = "little", target_feature = "neon"))]
  fn neon_matches_portable() {
    exercise(super::super::aarch64_neon::absorb_blocks);
  }

  #[test]
  #[cfg(all(target_arch = "aarch64", target_endian = "little", target_feature = "neon"))]
  fn neon_scalar_hybrid_matches_portable() {
    exercise(super::super::aarch64_neon::absorb_blocks_hybrid);
  }
}
