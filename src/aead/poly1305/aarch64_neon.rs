use super::{State, compute_block_aarch64_neon};

define_target_feature_forwarder! {
  pub(super) fn compute_block(state: &mut State, block: &[u8; 16], partial: bool) {
    feature = "neon";
    outer_safety = "backend selection guarantees NEON is available before this wrapper is chosen.";
    inner_safety = "this wrapper enables NEON before calling the shared NEON body.";
    call = compute_block_aarch64_neon(state, block, partial);
  }
}
