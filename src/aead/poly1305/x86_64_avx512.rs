use super::{State, compute_block_x86_avx512};

define_target_feature_forwarder! {
  pub(super) fn compute_block(state: &mut State, block: &[u8; 16], partial: bool) {
    feature = "avx512f,avx512vl,avx512bw,avx512dq";
    outer_safety = "backend selection guarantees the AVX-512 feature set required by this kernel.";
    inner_safety = "this wrapper enables the AVX-512 feature set required by the shared AVX-512 body.";
    call = compute_block_x86_avx512(state, block, partial);
  }
}
