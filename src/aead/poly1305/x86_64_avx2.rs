use super::{State, compute_block_x86_avx2};

define_target_feature_forwarder! {
  pub(super) fn compute_block(state: &mut State, block: &[u8; 16], partial: bool) {
    feature = "avx2";
    outer_safety = "backend selection guarantees AVX2 is available before this wrapper is chosen.";
    inner_safety = "this wrapper enables AVX2 before calling the shared AVX2 body.";
    call = compute_block_x86_avx2(state, block, partial);
  }
}
