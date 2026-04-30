#![no_main]

#[path = "../../../fuzz/target_impls/hash_sha3.rs"]
mod target_impl;

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    target_impl::run(data);
});
