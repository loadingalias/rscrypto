#![no_main]

#[path = "../../../fuzz/target_impls/hash_ascon_cxof.rs"]
mod target_impl;

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    target_impl::run(data);
});
