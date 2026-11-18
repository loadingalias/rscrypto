#![no_main]

#[path = "../target_impls/provider_quic_header.rs"]
mod target_impl;

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
  target_impl::run(data);
});
