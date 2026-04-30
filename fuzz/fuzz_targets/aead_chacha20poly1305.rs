#![no_main]

#[path = "../target_impls/aead_chacha20poly1305.rs"]
mod target_impl;

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    target_impl::run(data);
});
