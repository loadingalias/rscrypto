#![no_main]

#[path = "../target_impls/aead_quic_lb.rs"]
mod target_impl;

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
  target_impl::run(data);
});
