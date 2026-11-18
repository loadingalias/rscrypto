// `compat.py` builds this wrapper to prove that an embedder can build `BLAKE3_PORTABLE_SOURCE` alone.
#![no_std]
#![forbid(unsafe_code)]

#[allow(dead_code)]
mod blake3 {
  include!("../../src/hashes/crypto/blake3/portable.rs");
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
  loop {}
}
