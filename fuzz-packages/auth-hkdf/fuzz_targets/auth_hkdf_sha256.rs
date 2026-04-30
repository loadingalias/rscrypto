#![no_main]

#[path = "../../../fuzz/target_impls/auth_hkdf_sha256.rs"]
mod target_impl;

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    target_impl::run(data);
});
