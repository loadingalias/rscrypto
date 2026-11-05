#![no_main]

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
  #[cfg(fuzzing)]
  rscrypto_rustls::quic::fuzzing::token(data);
  #[cfg(not(fuzzing))]
  let _unused_in_nonfuzz_build = data;
});
