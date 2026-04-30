#![no_main]

#[path = "../../../fuzz/target_impls/aead_nonce_counter.rs"]
mod target_impl;

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    target_impl::run(data);
});
