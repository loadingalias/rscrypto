#![allow(
  suspicious_runtime_symbol_definitions,
  reason = "the freestanding Binsec artifact intentionally provides its own C memory-runtime ABI"
)]

#[path = "../../../src/binsec/input.rs"]
mod input;
mod memory;

use input::array_from_global;

use core::ptr;

use rscrypto::aead::expert::AeadWithNonce;
use rscrypto::{
  Aegis256, Aegis256Key, Aes128Gcm, Aes128GcmKey, Aes128GcmSiv, Aes128GcmSivKey, Aes256Gcm, Aes256GcmKey, Aes256GcmSiv,
  Aes256GcmSivKey, AsconAead128, AsconAead128Key, Blake2b256, Blake2bKey, Blake2s256, Blake2sKey, Blake3,
  Blake3KeyedHash, ChaCha20Poly1305, ChaCha20Poly1305Key, ChaCha20Rng, HmacSha256, HmacSha256Tag, HmacSha384,
  HmacSha384Tag, HmacSha512, HmacSha512Tag, Kmac256, Mac, X25519SecretKey, XChaCha20Poly1305, XChaCha20Poly1305Key,
  aead::{Nonce96, Nonce128, Nonce192, Nonce256},
};

const LEN_16: usize = 16;
const LEN_32: usize = 32;
const LEN_48: usize = 48;
const LEN_64: usize = 64;

#[unsafe(no_mangle)]
#[used]
pub static mut CT_BINSEC_LHS_16: [u8; LEN_16] = [0u8; LEN_16];

#[unsafe(no_mangle)]
#[used]
pub static mut CT_BINSEC_RHS_16: [u8; LEN_16] = [0u8; LEN_16];

#[unsafe(no_mangle)]
#[used]
pub static mut CT_BINSEC_LHS_32: [u8; LEN_32] = [0u8; LEN_32];

#[unsafe(no_mangle)]
#[used]
pub static mut CT_BINSEC_RHS_32: [u8; LEN_32] = [0u8; LEN_32];

#[unsafe(no_mangle)]
#[used]
pub static mut CT_BINSEC_LHS_48: [u8; LEN_48] = [0u8; LEN_48];

#[unsafe(no_mangle)]
#[used]
pub static mut CT_BINSEC_RHS_48: [u8; LEN_48] = [0u8; LEN_48];

#[unsafe(no_mangle)]
#[used]
pub static mut CT_BINSEC_LHS_64: [u8; LEN_64] = [0u8; LEN_64];

#[unsafe(no_mangle)]
#[used]
pub static mut CT_BINSEC_RHS_64: [u8; LEN_64] = [0u8; LEN_64];

#[unsafe(no_mangle)]
#[used]
pub static mut CT_BINSEC_KEY_32: [u8; 32] = [0u8; 32];

#[unsafe(no_mangle)]
#[used]
pub static mut CT_BINSEC_KEY_16: [u8; 16] = [0u8; 16];

#[unsafe(no_mangle)]
#[used]
pub static mut CT_BINSEC_KEY_48: [u8; 48] = [0u8; 48];

#[unsafe(no_mangle)]
#[used]
pub static mut CT_BINSEC_KEY_64: [u8; 64] = [0u8; 64];

#[unsafe(no_mangle)]
#[used]
pub static mut CT_BINSEC_TAG_16: [u8; 16] = [0u8; 16];

#[unsafe(no_mangle)]
#[used]
pub static mut CT_BINSEC_TAG_32: [u8; 32] = [0u8; 32];

#[unsafe(no_mangle)]
#[used]
pub static mut CT_BINSEC_TAG_48: [u8; 48] = [0u8; 48];

#[unsafe(no_mangle)]
#[used]
pub static mut CT_BINSEC_TAG_64: [u8; 64] = [0u8; 64];

#[unsafe(no_mangle)]
#[used]
pub static mut CT_BINSEC_PASSWORD_32: [u8; 32] = [0u8; 32];

#[unsafe(no_mangle)]
#[used]
pub static mut CT_BINSEC_PASSWORD_64: [u8; 64] = [0u8; 64];

#[unsafe(no_mangle)]
#[used]
pub static mut CT_BINSEC_IKM_32: [u8; 32] = [0u8; 32];

#[unsafe(no_mangle)]
#[used]
pub static mut CT_BINSEC_IKM_48: [u8; 48] = [0u8; 48];

#[unsafe(no_mangle)]
#[used]
pub static mut CT_BINSEC_AEAD_BUFFER_32: [u8; 32] = [0u8; 32];

#[unsafe(no_mangle)]
#[used]
pub static mut CT_BINSEC_RESULT: u8 = 0;

#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn ct_binsec_done(result: u8) -> ! {
  // SAFETY: `CT_BINSEC_RESULT` is a harness-owned global used only to keep the
  // computed result observable in the analyzed binary.
  unsafe { ptr::write_volatile(ptr::addr_of_mut!(CT_BINSEC_RESULT), result) };
  loop {
    core::hint::spin_loop();
  }
}

#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn ct_binsec_owner_eq_64() -> ! {
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let lhs = HmacSha512Tag::from_bytes(unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_LHS_64)) });
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let rhs = HmacSha512Tag::from_bytes(unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_RHS_64)) });
  ct_binsec_done(u8::from(lhs.ct_eq(&rhs).declassify()))
}

#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn ct_binsec_owner_eq_48() -> ! {
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let lhs = HmacSha384Tag::from_bytes(unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_LHS_48)) });
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let rhs = HmacSha384Tag::from_bytes(unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_RHS_48)) });
  ct_binsec_done(u8::from(lhs.ct_eq(&rhs).declassify()))
}

#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn ct_binsec_owner_eq_32() -> ! {
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let lhs = X25519SecretKey::from_bytes(unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_LHS_32)) });
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let rhs = X25519SecretKey::from_bytes(unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_RHS_32)) });
  ct_binsec_done(u8::from(lhs.ct_eq(&rhs).declassify()))
}

#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn ct_binsec_owner_eq_16() -> ! {
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let lhs = Aes128GcmKey::from_bytes(unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_LHS_16)) });
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let rhs = Aes128GcmKey::from_bytes(unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_RHS_16)) });
  ct_binsec_done(u8::from(lhs.ct_eq(&rhs).declassify()))
}

#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn ct_binsec_hmac_sha256_verify() -> ! {
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let key = unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_KEY_32)) };
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let expected = unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_TAG_32)) };
  let mut mac = HmacSha256::new(&key);
  mac.update(b"binsec");
  let ok = mac.verify(&HmacSha256Tag::from_bytes(expected));
  ct_binsec_done(u8::from(ok.is_ok()))
}

#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn ct_binsec_hmac_sha256_verify_truncated_64() -> ! {
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let key = unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_KEY_32)) };
  // SAFETY: This pointer references a fixed harness global with static storage.
  let expected = unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_TAG_32)) };
  let expected = expected
    .first_chunk::<8>()
    .expect("HMAC-SHA256 tags contain eight bytes");
  let ok = HmacSha256::verify_truncated_tag_64(&key, b"binsec", expected);
  ct_binsec_done(u8::from(ok.is_ok()))
}

#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn ct_binsec_hmac_sha384_verify() -> ! {
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let key = unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_KEY_48)) };
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let expected = unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_TAG_48)) };
  let mut mac = HmacSha384::new(&key);
  mac.update(b"binsec");
  let ok = mac.verify(&HmacSha384Tag::from_bytes(expected));
  ct_binsec_done(u8::from(ok.is_ok()))
}

#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn ct_binsec_hmac_sha512_verify() -> ! {
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let key = unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_KEY_64)) };
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let expected = unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_TAG_64)) };
  let mut mac = HmacSha512::new(&key);
  mac.update(b"binsec");
  let ok = mac.verify(&HmacSha512Tag::from_bytes(expected));
  ct_binsec_done(u8::from(ok.is_ok()))
}

#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn ct_binsec_kmac256_verify() -> ! {
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let key = unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_KEY_32)) };
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let expected = unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_TAG_32)) };
  let ok = Kmac256::verify_tag(&key, b"ct", b"binsec", &expected).is_ok();
  ct_binsec_done(u8::from(ok))
}

#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn ct_binsec_blake3_verify_keyed() -> ! {
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let key = unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_KEY_32)) };
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let expected = unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_TAG_32)) };
  let expected = Blake3KeyedHash::from_bytes(expected);
  ct_binsec_done(u8::from(Blake3::verify_keyed(&key, b"binsec", &expected).is_ok()))
}

#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn ct_binsec_hkdf_sha256_derive() -> ! {
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let ikm = unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_IKM_32)) };
  let okm = rscrypto::HkdfSha256::derive_array::<32>(b"salt", &ikm, b"info").expect("one HKDF output block");
  let mut acc = 0u8;
  for byte in okm {
    acc ^= byte;
  }
  ct_binsec_done(acc)
}

#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn ct_binsec_hkdf_sha384_derive() -> ! {
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let ikm = unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_IKM_48)) };
  let okm = rscrypto::HkdfSha384::derive_array::<48>(b"salt", &ikm, b"info").expect("one HKDF output block");
  let mut acc = 0u8;
  for byte in okm {
    acc ^= byte;
  }
  ct_binsec_done(acc)
}

#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn ct_binsec_pbkdf2_sha256_verify() -> ! {
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let password = unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_PASSWORD_32)) };
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let expected = unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_TAG_32)) };
  let ok = rscrypto::Pbkdf2Sha256::verify_password_primitive(&password, b"salt", 1, &expected).is_ok();
  ct_binsec_done(u8::from(ok))
}

#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn ct_binsec_pbkdf2_sha512_verify() -> ! {
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let password = unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_PASSWORD_64)) };
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let expected = unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_TAG_64)) };
  let ok = rscrypto::Pbkdf2Sha512::verify_password_primitive(&password, b"salt", 1, &expected).is_ok();
  ct_binsec_done(u8::from(ok))
}

macro_rules! aead_seal_entry {
  ($name:ident, $cipher:ty, $key:ty, $nonce:ty, $key_global:ident, $nonce_byte:expr) => {
    #[unsafe(no_mangle)]
    #[inline(never)]
    pub extern "C" fn $name() -> ! {
      // SAFETY: This pointer references a fixed harness global with static storage.
      let key = unsafe { array_from_global(ptr::addr_of!($key_global)) };
      // SAFETY: This pointer references a fixed harness global with static storage.
      let mut buffer = unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_AEAD_BUFFER_32)) };
      let key = <$key>::from_bytes(key);
      let nonce = <$nonce>::from_bytes([$nonce_byte; <$nonce>::LENGTH]);
      let cipher = <$cipher>::new(&key);
      let mut acc = 0u8;
      if let Ok(tag) = cipher.encrypt_in_place(&nonce, b"binsec", &mut buffer) {
        for byte in tag.as_ref().iter().copied().chain(buffer) {
          acc ^= byte;
        }
      }
      ct_binsec_done(acc)
    }
  };
}

macro_rules! aead_open_entry {
  ($name:ident, $cipher:ty, $key:ty, $nonce:ty, $key_global:ident, $nonce_byte:expr) => {
    #[unsafe(no_mangle)]
    #[inline(never)]
    pub extern "C" fn $name() -> ! {
      // SAFETY: This pointer references a fixed harness global with static storage.
      let key = unsafe { array_from_global(ptr::addr_of!($key_global)) };
      // SAFETY: This pointer references a fixed harness global with static storage.
      let tag = unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_TAG_16)) };
      // SAFETY: This pointer references a fixed harness global with static storage.
      let mut buffer = unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_AEAD_BUFFER_32)) };
      let key = <$key>::from_bytes(key);
      let nonce = <$nonce>::from_bytes([$nonce_byte; <$nonce>::LENGTH]);
      let cipher = <$cipher>::new(&key);
      let tag = <$cipher>::tag_from_slice(&tag);
      let ok = if let Ok(tag) = tag {
        cipher.decrypt_in_place(&nonce, b"binsec", &mut buffer, &tag).is_ok()
      } else {
        false
      };
      let mut acc = u8::from(ok);
      for byte in buffer {
        acc ^= byte;
      }
      ct_binsec_done(acc)
    }
  };
}

aead_seal_entry!(
  ct_binsec_aes128gcm_seal,
  Aes128Gcm,
  Aes128GcmKey,
  Nonce96,
  CT_BINSEC_KEY_16,
  0xA1
);
aead_seal_entry!(
  ct_binsec_aes256gcm_seal,
  Aes256Gcm,
  Aes256GcmKey,
  Nonce96,
  CT_BINSEC_KEY_32,
  0xA2
);
aead_seal_entry!(
  ct_binsec_aes128gcmsiv_seal,
  Aes128GcmSiv,
  Aes128GcmSivKey,
  Nonce96,
  CT_BINSEC_KEY_16,
  0xA3
);
aead_seal_entry!(
  ct_binsec_aes256gcmsiv_seal,
  Aes256GcmSiv,
  Aes256GcmSivKey,
  Nonce96,
  CT_BINSEC_KEY_32,
  0xA4
);
aead_seal_entry!(
  ct_binsec_chacha20poly1305_seal_api,
  ChaCha20Poly1305,
  ChaCha20Poly1305Key,
  Nonce96,
  CT_BINSEC_KEY_32,
  0xA5
);
aead_seal_entry!(
  ct_binsec_xchacha20poly1305_seal,
  XChaCha20Poly1305,
  XChaCha20Poly1305Key,
  Nonce192,
  CT_BINSEC_KEY_32,
  0xA6
);
aead_seal_entry!(
  ct_binsec_aegis256_seal,
  Aegis256,
  Aegis256Key,
  Nonce256,
  CT_BINSEC_KEY_32,
  0xA7
);
aead_seal_entry!(
  ct_binsec_ascon_aead128_seal,
  AsconAead128,
  AsconAead128Key,
  Nonce128,
  CT_BINSEC_KEY_16,
  0xA8
);

aead_open_entry!(
  ct_binsec_aes128gcm_open,
  Aes128Gcm,
  Aes128GcmKey,
  Nonce96,
  CT_BINSEC_KEY_16,
  0xB1
);
aead_open_entry!(
  ct_binsec_aes256gcm_open,
  Aes256Gcm,
  Aes256GcmKey,
  Nonce96,
  CT_BINSEC_KEY_32,
  0xB2
);
aead_open_entry!(
  ct_binsec_aes128gcmsiv_open,
  Aes128GcmSiv,
  Aes128GcmSivKey,
  Nonce96,
  CT_BINSEC_KEY_16,
  0xB3
);
aead_open_entry!(
  ct_binsec_aes256gcmsiv_open,
  Aes256GcmSiv,
  Aes256GcmSivKey,
  Nonce96,
  CT_BINSEC_KEY_32,
  0xB4
);
aead_open_entry!(
  ct_binsec_chacha20poly1305_open,
  ChaCha20Poly1305,
  ChaCha20Poly1305Key,
  Nonce96,
  CT_BINSEC_KEY_32,
  0xB5
);
aead_open_entry!(
  ct_binsec_xchacha20poly1305_open,
  XChaCha20Poly1305,
  XChaCha20Poly1305Key,
  Nonce192,
  CT_BINSEC_KEY_32,
  0xB6
);
aead_open_entry!(
  ct_binsec_aegis256_open,
  Aegis256,
  Aegis256Key,
  Nonce256,
  CT_BINSEC_KEY_32,
  0xB7
);
aead_open_entry!(
  ct_binsec_ascon_aead128_open,
  AsconAead128,
  AsconAead128Key,
  Nonce128,
  CT_BINSEC_KEY_16,
  0xB8
);

#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn ct_binsec_chacha20_rng_fill() -> ! {
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let seed = unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_KEY_32)) };
  // One read takes the first refill: eight portable ChaCha20 blocks, key replacement, and clears.
  let mut out = [0u8; 32];
  ChaCha20Rng::from_seed(seed).fill_bytes(&mut out);
  let mut acc = 0u8;
  for byte in out {
    acc ^= byte;
  }
  ct_binsec_done(acc)
}

#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn ct_binsec_blake2_blake3_keyed_digest() -> ! {
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let key = unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_KEY_32)) };
  let b2b256 = Blake2b256::keyed_digest(Blake2bKey::new(&key).expect("32-byte key"), b"binsec");
  let b2s256 = Blake2s256::keyed_digest(Blake2sKey::new(&key).expect("32-byte key"), b"binsec");
  let b3 = Blake3::keyed_digest(&key, b"binsec").to_bytes();

  let mut acc = 0u8;
  for byte in b2b256.into_iter().chain(b2s256).chain(b3) {
    acc ^= byte;
  }
  ct_binsec_done(acc)
}

#[unsafe(no_mangle)]
#[used]
pub static CT_BINSEC_ENTRYPOINTS: [extern "C" fn() -> !; 32] = [
  ct_binsec_owner_eq_16,
  ct_binsec_owner_eq_32,
  ct_binsec_owner_eq_48,
  ct_binsec_owner_eq_64,
  ct_binsec_hmac_sha256_verify,
  ct_binsec_hmac_sha256_verify_truncated_64,
  ct_binsec_hmac_sha384_verify,
  ct_binsec_hmac_sha512_verify,
  ct_binsec_kmac256_verify,
  ct_binsec_blake3_verify_keyed,
  ct_binsec_hkdf_sha256_derive,
  ct_binsec_hkdf_sha384_derive,
  ct_binsec_pbkdf2_sha256_verify,
  ct_binsec_pbkdf2_sha512_verify,
  ct_binsec_aes128gcm_seal,
  ct_binsec_aes256gcm_seal,
  ct_binsec_aes128gcmsiv_seal,
  ct_binsec_aes256gcmsiv_seal,
  ct_binsec_chacha20poly1305_seal_api,
  ct_binsec_xchacha20poly1305_seal,
  ct_binsec_aegis256_seal,
  ct_binsec_ascon_aead128_seal,
  ct_binsec_aes128gcm_open,
  ct_binsec_aes256gcm_open,
  ct_binsec_aes128gcmsiv_open,
  ct_binsec_aes256gcmsiv_open,
  ct_binsec_chacha20poly1305_open,
  ct_binsec_xchacha20poly1305_open,
  ct_binsec_aegis256_open,
  ct_binsec_ascon_aead128_open,
  ct_binsec_chacha20_rng_fill,
  ct_binsec_blake2_blake3_keyed_digest,
];

fn main() {
  ct_binsec_owner_eq_64();
}
