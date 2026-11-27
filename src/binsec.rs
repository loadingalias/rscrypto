//! Fixed inputs and stop address for the library test proof binary.

use core::ptr;

#[cfg(any(
  feature = "aegis256",
  feature = "aes-gcm",
  feature = "aes-gcm-siv",
  feature = "argon2",
  feature = "ascon-aead",
  feature = "chacha20poly1305",
  feature = "rsa",
  feature = "x25519"
))]
include!("binsec/input.rs");

#[unsafe(no_mangle)]
#[used]
pub(crate) static mut CT_BINSEC_RHS_16: [u8; 16] = [0u8; 16];

#[unsafe(no_mangle)]
#[used]
pub(crate) static mut CT_BINSEC_KEY_32: [u8; 32] = [0u8; 32];

#[unsafe(no_mangle)]
#[used]
pub(crate) static mut CT_BINSEC_KEY_16: [u8; 16] = [0u8; 16];

#[unsafe(no_mangle)]
#[used]
pub(crate) static mut CT_BINSEC_TAG_16: [u8; 16] = [0u8; 16];

#[unsafe(no_mangle)]
#[used]
pub(crate) static mut CT_BINSEC_AEAD_BUFFER_32: [u8; 32] = [0u8; 32];

#[unsafe(no_mangle)]
#[used]
pub(crate) static mut CT_BINSEC_NONCE_16: [u8; 16] = [0u8; 16];

#[unsafe(no_mangle)]
#[used]
pub(crate) static mut CT_BINSEC_AEGIS_NONCE_32: [u8; 32] = [0u8; 32];

#[unsafe(no_mangle)]
#[used]
pub(crate) static mut CT_BINSEC_ARGON2_DST: [u64; 128] = [0u64; 128];

#[unsafe(no_mangle)]
#[used]
pub(crate) static mut CT_BINSEC_ARGON2_X: [u64; 128] = [0u64; 128];

#[unsafe(no_mangle)]
#[used]
pub(crate) static mut CT_BINSEC_ARGON2_Y: [u64; 128] = [0u64; 128];

#[unsafe(no_mangle)]
#[used]
pub(crate) static mut CT_BINSEC_POLY1305_KEY: [u8; 32] = [0u8; 32];

#[unsafe(no_mangle)]
#[used]
pub(crate) static mut CT_BINSEC_POLY1305_BLOCK: [u8; 16] = [0u8; 16];

#[unsafe(no_mangle)]
#[used]
pub(crate) static mut CT_BINSEC_POLY1305_PARTIAL: u8 = 0;

#[unsafe(no_mangle)]
#[used]
pub(crate) static mut CT_BINSEC_CURVE25519_LHS: [u64; 5] = [0u64; 5];

#[unsafe(no_mangle)]
#[used]
pub(crate) static mut CT_BINSEC_CURVE25519_RHS: [u64; 5] = [0u64; 5];

#[unsafe(no_mangle)]
#[used]
pub(crate) static mut CT_BINSEC_CURVE25519_SWAP: u8 = 0;

#[unsafe(no_mangle)]
#[used]
pub(crate) static mut CT_BINSEC_ED25519_DIGIT: i8 = 0;

#[unsafe(no_mangle)]
#[used]
pub(crate) static mut CT_BINSEC_ECDSA_DIGIT: u8 = 0;

#[unsafe(no_mangle)]
#[used]
pub(crate) static mut CT_BINSEC_P256_ECDH_DIGIT: u8 = 0;

#[unsafe(no_mangle)]
#[used]
pub(crate) static mut CT_BINSEC_P256_ECDH_SIGNED_DIGIT: u8 = 0;

#[unsafe(no_mangle)]
#[used]
pub(crate) static mut CT_BINSEC_P384_ECDH_GENERATOR_DIGIT: u8 = 0;

#[unsafe(no_mangle)]
#[used]
pub(crate) static mut CT_BINSEC_P384_ECDH_WINDOW_DIGIT: u8 = 0;

#[unsafe(no_mangle)]
#[used]
pub(crate) static mut CT_BINSEC_RSA_WINDOW_TABLE: [u64; 64] = [0u64; 64];

#[unsafe(no_mangle)]
#[used]
pub(crate) static mut CT_BINSEC_RSA_WINDOW: u8 = 0;

#[cfg(target_arch = "aarch64")]
#[unsafe(no_mangle)]
#[used]
pub(crate) static mut CT_BINSEC_RSA_PAIR_A: [u64; 48] = [0u64; 48];

#[cfg(target_arch = "aarch64")]
#[unsafe(no_mangle)]
#[used]
pub(crate) static mut CT_BINSEC_RSA_PAIR_B: [u64; 48] = [0u64; 48];

#[cfg(target_arch = "aarch64")]
#[unsafe(no_mangle)]
#[used]
pub(crate) static mut CT_BINSEC_RSA_PAIR_MODULI: [u64; 48] = [0u64; 48];

#[cfg(target_arch = "aarch64")]
#[unsafe(no_mangle)]
#[used]
pub(crate) static mut CT_BINSEC_RSA_PAIR_CONSTANTS: [u64; 4] = [0u64; 4];

#[unsafe(no_mangle)]
#[used]
pub(crate) static mut CT_BINSEC_RSA_COMPONENT_32: [u8; 32] = [0u8; 32];

#[unsafe(no_mangle)]
#[used]
pub(crate) static mut CT_BINSEC_RSA_UPPER_32: [u8; 32] = [0u8; 32];

#[unsafe(no_mangle)]
#[used]
pub(crate) static mut CT_BINSEC_RSA_OTHER_32: [u8; 32] = [0u8; 32];

#[unsafe(no_mangle)]
#[used]
pub(crate) static mut CT_BINSEC_RSA_EXPONENT_256: [u8; 256] = [0u8; 256];

#[unsafe(no_mangle)]
#[used]
pub(crate) static mut CT_BINSEC_RSA_CIPHERTEXT_256: [u8; 256] = [0u8; 256];

#[unsafe(no_mangle)]
#[used]
pub(crate) static mut CT_BINSEC_RSA_ENCODED_256: [u8; 256] = [0u8; 256];

#[unsafe(no_mangle)]
#[used]
pub(crate) static mut CT_BINSEC_RSA_CANDIDATES_256: [u8; 256] = [0u8; 256];

#[unsafe(no_mangle)]
#[used]
pub(crate) static mut CT_BINSEC_RSA_ALTERNATIVE_256: [u8; 256] = [0u8; 256];

#[unsafe(no_mangle)]
#[used]
pub(crate) static mut CT_BINSEC_RESULT: u8 = 0;

#[unsafe(no_mangle)]
#[used]
pub(crate) static mut CT_BINSEC_MLDSA_OPERANDS: [u32; 2] = [0; 2];

#[unsafe(no_mangle)]
#[used]
pub(crate) static mut CT_BINSEC_MLDSA_PRODUCT: u32 = 0;

#[unsafe(no_mangle)]
#[inline(never)]
pub(crate) extern "C" fn ct_binsec_done(result: u8) -> ! {
  // SAFETY: `CT_BINSEC_RESULT` is a harness-owned global used only to keep the
  // computed result observable in the analyzed binary.
  unsafe { ptr::write_volatile(ptr::addr_of_mut!(CT_BINSEC_RESULT), result) };
  loop {
    core::hint::spin_loop();
  }
}
