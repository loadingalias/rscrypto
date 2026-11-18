// The `compatibility` lane of `scripts/check/compat.py` builds this embedder of the portable BLAKE3 text.
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
