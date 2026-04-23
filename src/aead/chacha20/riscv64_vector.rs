use super::{KEY_SIZE, NONCE_SIZE, xor_keystream_u32x4_impl};

define_target_feature_forwarder! {
  pub(super) fn xor_keystream(
    key: &[u8; KEY_SIZE],
    initial_counter: u32,
    nonce: &[u8; NONCE_SIZE],
    buffer: &mut [u8]
  ) {
    feature = "v";
    outer_safety = "backend selection guarantees the vector extension before this wrapper is chosen.";
    inner_safety = "the wrapper only reaches this function when the RISC-V vector extension is available.";
    call = xor_keystream_u32x4_impl(key, initial_counter, nonce, buffer);
  }
}
