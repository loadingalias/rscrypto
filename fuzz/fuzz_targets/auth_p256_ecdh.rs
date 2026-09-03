#![no_main]

#[path = "../target_impls/auth_p256_ecdh.rs"]
mod target_impl;

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
  target_impl::run(data);
});
