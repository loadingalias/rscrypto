use super::{State, compute_block_wasm_simd128};

#[inline]
pub(super) fn compute_block(state: &mut State, block: &[u8; 16], partial: bool) {
  // SAFETY: backend selection guarantees `simd128` is available before this wrapper is chosen.
  unsafe { compute_block_wasm_simd128(state, block, partial) }
}
