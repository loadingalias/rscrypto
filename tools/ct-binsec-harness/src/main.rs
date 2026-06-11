use core::{ptr, slice};

use rscrypto::{
  Aegis256, Aegis256Key, Aes128Gcm, Aes128GcmKey, Aes128GcmSiv, Aes128GcmSivKey, Aes256Gcm, Aes256GcmKey,
  Aes256GcmSiv, Aes256GcmSivKey, AsconAead128, AsconAead128Key, Blake3KeyedHash, ChaCha20Poly1305,
  ChaCha20Poly1305Key, Kmac256, SecretBytes, XChaCha20Poly1305, XChaCha20Poly1305Key,
  aead::{Nonce96, Nonce128, Nonce192, Nonce256},
  traits::ct,
};

const LEN_16: usize = 16;
const LEN_32: usize = 32;
const LEN_64: usize = 64;
const CURVE25519_LIMBS: usize = 5;
const ARGON2_BLOCK_WORDS: usize = 128;
const RSA_WINDOW_LIMBS: usize = 4;
const RSA_WINDOW_TABLE_LIMBS: usize = 16 * RSA_WINDOW_LIMBS;

#[unsafe(no_mangle)]
#[inline(never)]
pub unsafe extern "C" fn memset(dst: *mut u8, value: i32, len: usize) -> *mut u8 {
  let mut offset = 0usize;
  while offset < len {
    // SAFETY: C callers require `dst..dst+len` to be writable. Volatile writes
    // keep zeroization visible to BINSEC and prevent this shim from becoming a
    // recursive compiler intrinsic.
    unsafe { ptr::write_volatile(dst.add(offset), value as u8) };
    offset += 1;
  }
  dst
}

#[unsafe(no_mangle)]
#[inline(never)]
pub unsafe extern "C" fn memmove(dst: *mut u8, src: *const u8, len: usize) -> *mut u8 {
  if (dst as usize) <= (src as usize) {
    let mut offset = 0usize;
    while offset < len {
      // SAFETY: C callers require source and destination ranges to be valid.
      let byte = unsafe { ptr::read(src.add(offset)) };
      // SAFETY: C callers require source and destination ranges to be valid.
      unsafe { ptr::write(dst.add(offset), byte) };
      offset += 1;
    }
  } else {
    let mut remaining = len;
    while remaining != 0 {
      remaining -= 1;
      // SAFETY: C callers require source and destination ranges to be valid.
      let byte = unsafe { ptr::read(src.add(remaining)) };
      // SAFETY: C callers require source and destination ranges to be valid.
      unsafe { ptr::write(dst.add(remaining), byte) };
    }
  }
  dst
}

#[unsafe(no_mangle)]
#[inline(never)]
pub unsafe extern "C" fn memcpy(dst: *mut u8, src: *const u8, len: usize) -> *mut u8 {
  // SAFETY: `memcpy` has the same validity contract as `memmove`, with the
  // additional non-overlap precondition. The `memmove` implementation is valid
  // for both cases and keeps the harness small.
  unsafe { memmove(dst, src, len) }
}

#[unsafe(no_mangle)]
#[inline(never)]
pub unsafe extern "C" fn bcmp(lhs: *const u8, rhs: *const u8, len: usize) -> i32 {
  let mut acc = 0u8;
  let mut offset = 0usize;
  while offset < len {
    // SAFETY: C callers require both ranges to be readable for `len` bytes.
    let l = unsafe { ptr::read(lhs.add(offset)) };
    // SAFETY: C callers require both ranges to be readable for `len` bytes.
    let r = unsafe { ptr::read(rhs.add(offset)) };
    acc |= l ^ r;
    offset += 1;
  }
  i32::from(acc)
}

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
pub static mut CT_BINSEC_NONCE_16: [u8; 16] = [0u8; 16];

#[unsafe(no_mangle)]
#[used]
pub static mut CT_BINSEC_AEGIS_NONCE_32: [u8; 32] = [0u8; 32];

#[unsafe(no_mangle)]
#[used]
pub static mut CT_BINSEC_ARGON2_DST: [u64; ARGON2_BLOCK_WORDS] = [0u64; ARGON2_BLOCK_WORDS];

#[unsafe(no_mangle)]
#[used]
pub static mut CT_BINSEC_ARGON2_X: [u64; ARGON2_BLOCK_WORDS] = [0u64; ARGON2_BLOCK_WORDS];

#[unsafe(no_mangle)]
#[used]
pub static mut CT_BINSEC_ARGON2_Y: [u64; ARGON2_BLOCK_WORDS] = [0u64; ARGON2_BLOCK_WORDS];

#[unsafe(no_mangle)]
#[used]
pub static mut CT_BINSEC_POLY1305_KEY: [u8; 32] = [0u8; 32];

#[unsafe(no_mangle)]
#[used]
pub static mut CT_BINSEC_POLY1305_BLOCK: [u8; 16] = [0u8; 16];

#[unsafe(no_mangle)]
#[used]
pub static mut CT_BINSEC_POLY1305_PARTIAL: u8 = 0;

#[unsafe(no_mangle)]
#[used]
pub static mut CT_BINSEC_CURVE25519_LHS: [u64; CURVE25519_LIMBS] = [0u64; CURVE25519_LIMBS];

#[unsafe(no_mangle)]
#[used]
pub static mut CT_BINSEC_CURVE25519_RHS: [u64; CURVE25519_LIMBS] = [0u64; CURVE25519_LIMBS];

#[unsafe(no_mangle)]
#[used]
pub static mut CT_BINSEC_CURVE25519_SWAP: u8 = 0;

#[unsafe(no_mangle)]
#[used]
pub static mut CT_BINSEC_ED25519_DIGIT: i8 = 0;

#[unsafe(no_mangle)]
#[used]
pub static mut CT_BINSEC_ECDSA_DIGIT: u8 = 0;

#[unsafe(no_mangle)]
#[used]
pub static mut CT_BINSEC_RSA_WINDOW_TABLE: [u64; RSA_WINDOW_TABLE_LIMBS] = [0u64; RSA_WINDOW_TABLE_LIMBS];

#[unsafe(no_mangle)]
#[used]
pub static mut CT_BINSEC_RSA_WINDOW: u8 = 0;

#[unsafe(no_mangle)]
#[used]
pub static mut CT_BINSEC_RSA_COMPONENT_32: [u8; 32] = [0u8; 32];

#[unsafe(no_mangle)]
#[used]
pub static mut CT_BINSEC_RSA_UPPER_32: [u8; 32] = [0u8; 32];

#[unsafe(no_mangle)]
#[used]
pub static mut CT_BINSEC_RSA_OTHER_32: [u8; 32] = [0u8; 32];

#[unsafe(no_mangle)]
#[used]
pub static mut CT_BINSEC_RESULT: u8 = 0;

#[inline(always)]
unsafe fn slice_from_global<const N: usize>(ptr: *const [u8; N]) -> &'static [u8] {
  // SAFETY: BINSEC harness globals are fixed-size byte arrays with static
  // storage duration. The caller passes the address of one such global.
  unsafe { slice::from_raw_parts(ptr.cast::<u8>(), N) }
}

#[inline(always)]
unsafe fn array_from_global<const N: usize>(ptr: *const [u8; N]) -> [u8; N] {
  // SAFETY: BINSEC harness globals are fixed-size byte arrays with static
  // storage duration. `read_volatile` keeps the symbolic bytes observable.
  unsafe { ptr::read_volatile(ptr) }
}

#[inline(always)]
unsafe fn limbs_from_global<const N: usize>(ptr: *const [u64; N]) -> [u64; N] {
  // SAFETY: BINSEC harness globals are fixed-size limb arrays with static
  // storage duration. `read_volatile` keeps the symbolic limbs observable.
  unsafe { ptr::read_volatile(ptr) }
}

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
pub extern "C" fn ct_binsec_constant_time_eq_64() -> ! {
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let lhs = unsafe { slice_from_global(ptr::addr_of!(CT_BINSEC_LHS_64)) };
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let rhs = unsafe { slice_from_global(ptr::addr_of!(CT_BINSEC_RHS_64)) };
  ct_binsec_done(u8::from(ct::constant_time_eq(lhs, rhs)))
}

#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn ct_binsec_constant_time_eq_32() -> ! {
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let lhs = unsafe { slice_from_global(ptr::addr_of!(CT_BINSEC_LHS_32)) };
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let rhs = unsafe { slice_from_global(ptr::addr_of!(CT_BINSEC_RHS_32)) };
  ct_binsec_done(u8::from(ct::constant_time_eq(lhs, rhs)))
}

#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn ct_binsec_constant_time_eq_16() -> ! {
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let lhs = unsafe { slice_from_global(ptr::addr_of!(CT_BINSEC_LHS_16)) };
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let rhs = unsafe { slice_from_global(ptr::addr_of!(CT_BINSEC_RHS_16)) };
  ct_binsec_done(u8::from(ct::constant_time_eq(lhs, rhs)))
}

#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn ct_binsec_secret_bytes32_eq() -> ! {
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let lhs = SecretBytes::<32>::from(unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_LHS_32)) });
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let rhs = SecretBytes::<32>::from(unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_RHS_32)) });
  ct_binsec_done(u8::from(lhs == rhs))
}

#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn ct_binsec_hmac_sha256_verify() -> ! {
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let key = unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_KEY_32)) };
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let expected = unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_TAG_32)) };
  let ok = rscrypto::diag_hmac_sha256_verify_portable(&key, &expected);
  ct_binsec_done(u8::from(ok))
}

#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn ct_binsec_hmac_sha384_verify() -> ! {
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let key = unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_KEY_48)) };
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let expected = unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_TAG_48)) };
  let ok = rscrypto::diag_hmac_sha384_verify_portable(&key, &expected);
  ct_binsec_done(u8::from(ok))
}

#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn ct_binsec_hmac_sha512_verify() -> ! {
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let key = unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_KEY_64)) };
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let expected = unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_TAG_64)) };
  let ok = rscrypto::diag_hmac_sha512_verify_portable(&key, &expected);
  ct_binsec_done(u8::from(ok))
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
  let actual = rscrypto::hashes::crypto::diag_blake3_keyed_digest_portable(&key);
  let ok = actual == expected;
  ct_binsec_done(u8::from(ok))
}

#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn ct_binsec_hkdf_sha256_derive() -> ! {
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let ikm = unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_IKM_32)) };
  let okm = rscrypto::diag_hkdf_sha256_derive_portable(&ikm);
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
  let okm = rscrypto::diag_hkdf_sha384_derive_portable(&ikm);
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
  let ok = rscrypto::diag_pbkdf2_sha256_verify_portable(&password, &expected);
  ct_binsec_done(u8::from(ok))
}

#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn ct_binsec_pbkdf2_sha512_verify() -> ! {
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let password = unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_PASSWORD_64)) };
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let expected = unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_TAG_64)) };
  let ok = rscrypto::diag_pbkdf2_sha512_verify_portable(&password, &expected);
  ct_binsec_done(u8::from(ok))
}

#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn ct_binsec_aes_round_portable() -> ! {
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let block = unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_TAG_16)) };
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let round_key = unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_KEY_16)) };
  let out = rscrypto::aead::diag_aes_enc_round_portable(&block, &round_key);
  let mut acc = 0u8;
  for byte in out {
    acc ^= byte;
  }
  ct_binsec_done(acc)
}

#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn ct_binsec_ghash_block_portable() -> ! {
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let h = unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_KEY_16)) };
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let block = unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_TAG_16)) };
  let out = rscrypto::aead::diag_ghash_block_portable(&h, &block);
  let mut acc = 0u8;
  for byte in out {
    acc ^= byte;
  }
  ct_binsec_done(acc)
}

#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn ct_binsec_polyval_reduce_portable() -> ! {
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let a = unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_KEY_16)) };
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let b = unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_TAG_16)) };
  let out = rscrypto::aead::diag_polyval_reduce_portable(&a, &b);
  let mut acc = 0u8;
  for byte in out {
    acc ^= byte;
  }
  ct_binsec_done(acc)
}

#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn ct_binsec_aegis256_update_portable() -> ! {
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let key = unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_KEY_32)) };
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let nonce = unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_AEGIS_NONCE_32)) };
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let block = unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_TAG_16)) };
  let out = rscrypto::aead::diag_aegis256_update_portable(&key, &nonce, &block);
  let mut acc = 0u8;
  for byte in out {
    acc ^= byte;
  }
  ct_binsec_done(acc)
}

#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn ct_binsec_ascon_aead128_tag_portable() -> ! {
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let key = unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_KEY_16)) };
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let nonce = unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_NONCE_16)) };
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let block = unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_TAG_16)) };
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let expected = unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_RHS_16)) };
  let ok = rscrypto::aead::diag_ascon_aead128_tag_portable(&key, &nonce, &block, &expected);
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

aead_seal_entry!(ct_binsec_aes128gcm_seal, Aes128Gcm, Aes128GcmKey, Nonce96, CT_BINSEC_KEY_16, 0xA1);
aead_seal_entry!(ct_binsec_aes256gcm_seal, Aes256Gcm, Aes256GcmKey, Nonce96, CT_BINSEC_KEY_32, 0xA2);
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
aead_seal_entry!(ct_binsec_aegis256_seal, Aegis256, Aegis256Key, Nonce256, CT_BINSEC_KEY_32, 0xA7);
aead_seal_entry!(
  ct_binsec_ascon_aead128_seal,
  AsconAead128,
  AsconAead128Key,
  Nonce128,
  CT_BINSEC_KEY_16,
  0xA8
);

aead_open_entry!(ct_binsec_aes128gcm_open, Aes128Gcm, Aes128GcmKey, Nonce96, CT_BINSEC_KEY_16, 0xB1);
aead_open_entry!(ct_binsec_aes256gcm_open, Aes256Gcm, Aes256GcmKey, Nonce96, CT_BINSEC_KEY_32, 0xB2);
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
aead_open_entry!(ct_binsec_aegis256_open, Aegis256, Aegis256Key, Nonce256, CT_BINSEC_KEY_32, 0xB7);
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
pub extern "C" fn ct_binsec_argon2i_hash() -> ! {
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let mut dst = unsafe { limbs_from_global(ptr::addr_of!(CT_BINSEC_ARGON2_DST)) };
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let x = unsafe { limbs_from_global(ptr::addr_of!(CT_BINSEC_ARGON2_X)) };
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let y = unsafe { limbs_from_global(ptr::addr_of!(CT_BINSEC_ARGON2_Y)) };
  rscrypto::auth::argon2::diag_compress_portable(&mut dst, &x, &y, true);

  let mut acc = 0u64;
  for word in dst {
    acc ^= word;
  }
  ct_binsec_done((acc | (acc >> 8) | (acc >> 16) | (acc >> 24) | (acc >> 32) | (acc >> 40) | (acc >> 48) | (acc >> 56)) as u8)
}

#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn ct_binsec_chacha20poly1305_seal() -> ! {
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let key = unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_KEY_32)) };
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let mut buffer = unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_AEAD_BUFFER_32)) };
  rscrypto::aead::diag_chacha20_xor_keystream_portable(&key, 1, &[7u8; 12], &mut buffer);
  let mut acc = 0u8;
  for byte in buffer {
    acc ^= byte;
  }
  ct_binsec_done(acc)
}

#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn ct_binsec_blake2_blake3_keyed_digest() -> ! {
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let key = unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_KEY_32)) };
  let b2b256 = rscrypto::hashes::crypto::diag_blake2b256_keyed_digest_portable(&key);
  let b2s256 = rscrypto::hashes::crypto::diag_blake2s256_keyed_digest_portable(&key);
  let b3 = rscrypto::hashes::crypto::diag_blake3_keyed_digest_portable(&key).to_bytes();

  let mut acc = 0u8;
  for byte in b2b256.into_iter().chain(b2s256).chain(b3) {
    acc ^= byte;
  }
  ct_binsec_done(acc)
}

#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn ct_binsec_poly1305_block_portable() -> ! {
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let key = unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_POLY1305_KEY)) };
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let block = unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_POLY1305_BLOCK)) };
  // SAFETY: This pointer references a fixed harness global with static storage.
  let partial = unsafe { ptr::read_volatile(ptr::addr_of!(CT_BINSEC_POLY1305_PARTIAL)) } != 0;
  let tag = rscrypto::aead::diag_poly1305_block_portable_digest(&key, &block, partial);

  let mut acc = 0u8;
  for byte in tag {
    acc ^= byte;
  }
  ct_binsec_done(acc)
}

#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn ct_binsec_curve25519_conditional_swap() -> ! {
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let mut lhs = unsafe { limbs_from_global(ptr::addr_of!(CT_BINSEC_CURVE25519_LHS)) };
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let mut rhs = unsafe { limbs_from_global(ptr::addr_of!(CT_BINSEC_CURVE25519_RHS)) };
  // SAFETY: This pointer references a fixed harness global with static storage.
  let swap = unsafe { ptr::read_volatile(ptr::addr_of!(CT_BINSEC_CURVE25519_SWAP)) };

  rscrypto::diag_curve25519_conditional_swap(&mut lhs, &mut rhs, swap);

  let mut acc = 0u64;
  for limb in lhs.into_iter().chain(rhs) {
    acc ^= limb;
  }
  ct_binsec_done((acc | (acc >> 8) | (acc >> 16) | (acc >> 24) | (acc >> 32) | (acc >> 40) | (acc >> 48) | (acc >> 56)) as u8)
}

#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn ct_binsec_ed25519_select_basepoint_cached() -> ! {
  // SAFETY: This pointer references a fixed harness global with static storage.
  let digit = unsafe { ptr::read_volatile(ptr::addr_of!(CT_BINSEC_ED25519_DIGIT)) };
  let limbs = rscrypto::diag_ed25519_select_basepoint_cached_limb_digest(digit);

  let mut acc = 0u64;
  for limb in limbs {
    acc ^= limb;
  }
  ct_binsec_done((acc | (acc >> 8) | (acc >> 16) | (acc >> 24) | (acc >> 32) | (acc >> 40) | (acc >> 48) | (acc >> 56)) as u8)
}

#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn ct_binsec_ecdsa_p256_select_signing_generator_affine() -> ! {
  // SAFETY: This pointer references a fixed harness global with static storage.
  let digit = unsafe { ptr::read_volatile(ptr::addr_of!(CT_BINSEC_ECDSA_DIGIT)) };
  let limbs = rscrypto::diag_ecdsa_p256_select_signing_generator_affine_limb_digest(digit);

  let mut acc = 0u64;
  for limb in limbs {
    acc ^= limb;
  }
  ct_binsec_done((acc | (acc >> 8) | (acc >> 16) | (acc >> 24) | (acc >> 32) | (acc >> 40) | (acc >> 48) | (acc >> 56)) as u8)
}

#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn ct_binsec_ecdsa_p384_select_signing_generator_affine() -> ! {
  // SAFETY: This pointer references a fixed harness global with static storage.
  let digit = unsafe { ptr::read_volatile(ptr::addr_of!(CT_BINSEC_ECDSA_DIGIT)) };
  let limbs = rscrypto::diag_ecdsa_p384_select_signing_generator_affine_limb_digest(digit);

  let mut acc = 0u64;
  for limb in limbs {
    acc ^= limb;
  }
  ct_binsec_done((acc | (acc >> 8) | (acc >> 16) | (acc >> 24) | (acc >> 32) | (acc >> 40) | (acc >> 48) | (acc >> 56)) as u8)
}

#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn ct_binsec_rsa_private_select_window_power_4() -> ! {
  // SAFETY: This pointer references a fixed harness global with static storage.
  let table = unsafe { limbs_from_global(ptr::addr_of!(CT_BINSEC_RSA_WINDOW_TABLE)) };
  // SAFETY: This pointer references a fixed harness global with static storage.
  let window = unsafe { ptr::read_volatile(ptr::addr_of!(CT_BINSEC_RSA_WINDOW)) };
  let limbs = rscrypto::diag_rsa_private_select_window_power_4(&table, window);

  let mut acc = 0u64;
  for limb in limbs {
    acc ^= limb;
  }
  ct_binsec_done((acc | (acc >> 8) | (acc >> 16) | (acc >> 24) | (acc >> 32) | (acc >> 40) | (acc >> 48) | (acc >> 56)) as u8)
}

#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn ct_binsec_rsa_private_component_validation_32() -> ! {
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let component = unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_RSA_COMPONENT_32)) };
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let upper = unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_RSA_UPPER_32)) };
  // SAFETY: These pointers reference fixed harness globals with static storage.
  let other = unsafe { array_from_global(ptr::addr_of!(CT_BINSEC_RSA_OTHER_32)) };
  let ok = rscrypto::auth::diag_rsa_private_component_validation_32(&component, &upper, &other);
  ct_binsec_done(ok)
}

#[cfg(target_arch = "x86_64")]
#[unsafe(no_mangle)]
#[inline(never)]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn ct_binsec_ed25519_select_basepoint_cached_avx2() -> ! {
  // SAFETY: This pointer references a fixed harness global with static storage.
  let digit = unsafe { ptr::read_volatile(ptr::addr_of!(CT_BINSEC_ED25519_DIGIT)) };
  // SAFETY: This entrypoint is compiled with AVX2 enabled and is selected only
  // for the x86_64 AVX2 BINSEC kernel.
  let limbs = unsafe { rscrypto::diag_ed25519_select_basepoint_cached_avx2_limb_digest(digit) };

  let mut acc = 0u64;
  for limb in limbs {
    acc ^= limb;
  }
  ct_binsec_done((acc | (acc >> 8) | (acc >> 16) | (acc >> 24) | (acc >> 32) | (acc >> 40) | (acc >> 48) | (acc >> 56)) as u8)
}

#[cfg(target_arch = "x86_64")]
#[unsafe(no_mangle)]
#[inline(never)]
#[target_feature(enable = "avx2,avx512ifma,avx512vl")]
pub unsafe extern "C" fn ct_binsec_ed25519_select_basepoint_cached_ifma() -> ! {
  // SAFETY: This pointer references a fixed harness global with static storage.
  let digit = unsafe { ptr::read_volatile(ptr::addr_of!(CT_BINSEC_ED25519_DIGIT)) };
  // SAFETY: This entrypoint is compiled with AVX2, AVX-512 IFMA, and AVX-512 VL
  // enabled and is selected only for the x86_64 IFMA BINSEC kernel.
  let limbs = unsafe { rscrypto::diag_ed25519_select_basepoint_cached_ifma_limb_digest(digit) };

  let mut acc = 0u64;
  for limb in limbs {
    acc ^= limb;
  }
  ct_binsec_done((acc | (acc >> 8) | (acc >> 16) | (acc >> 24) | (acc >> 32) | (acc >> 40) | (acc >> 48) | (acc >> 56)) as u8)
}

#[unsafe(no_mangle)]
#[used]
pub static CT_BINSEC_ENTRYPOINTS: [extern "C" fn() -> !; 44] = [
  ct_binsec_constant_time_eq_64,
  ct_binsec_constant_time_eq_32,
  ct_binsec_constant_time_eq_16,
  ct_binsec_secret_bytes32_eq,
  ct_binsec_hmac_sha256_verify,
  ct_binsec_hmac_sha384_verify,
  ct_binsec_hmac_sha512_verify,
  ct_binsec_kmac256_verify,
  ct_binsec_blake3_verify_keyed,
  ct_binsec_hkdf_sha256_derive,
  ct_binsec_hkdf_sha384_derive,
  ct_binsec_pbkdf2_sha256_verify,
  ct_binsec_pbkdf2_sha512_verify,
  ct_binsec_aes_round_portable,
  ct_binsec_ghash_block_portable,
  ct_binsec_polyval_reduce_portable,
  ct_binsec_aegis256_update_portable,
  ct_binsec_ascon_aead128_tag_portable,
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
  ct_binsec_argon2i_hash,
  ct_binsec_chacha20poly1305_seal,
  ct_binsec_blake2_blake3_keyed_digest,
  ct_binsec_poly1305_block_portable,
  ct_binsec_curve25519_conditional_swap,
  ct_binsec_ed25519_select_basepoint_cached,
  ct_binsec_ecdsa_p256_select_signing_generator_affine,
  ct_binsec_ecdsa_p384_select_signing_generator_affine,
  ct_binsec_rsa_private_select_window_power_4,
  ct_binsec_rsa_private_component_validation_32,
];

#[cfg(target_arch = "x86_64")]
#[unsafe(no_mangle)]
#[used]
pub static CT_BINSEC_X86_ENTRYPOINTS: [unsafe extern "C" fn() -> !; 2] = [
  ct_binsec_ed25519_select_basepoint_cached_avx2,
  ct_binsec_ed25519_select_basepoint_cached_ifma,
];

fn main() {
  ct_binsec_constant_time_eq_64();
}
