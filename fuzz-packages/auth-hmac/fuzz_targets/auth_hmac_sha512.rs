#![no_main]

#[path = "../../../fuzz/target_impls/auth_hmac_sha512.rs"]
mod target_impl;

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    target_impl::run(data);
});
