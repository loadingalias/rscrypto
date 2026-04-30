#![no_main]

#[path = "../../../fuzz/target_impls/auth_kmac256.rs"]
mod target_impl;

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    target_impl::run(data);
});
