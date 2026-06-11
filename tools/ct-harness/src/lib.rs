//! Stable C ABI entrypoints for constant-time artifact generation.
//!
//! These functions are audit harnesses. They call production `rscrypto` APIs
//! through narrow pointer/length boundaries so IR, assembly, object, and timing
//! tools can target stable symbols without analyzing ergonomic public APIs.

use core::{ptr, slice};
use std::format;

use rscrypto::{
  Aegis256, Aegis256Key, Aes128Gcm, Aes128GcmKey, Aes128GcmSiv, Aes128GcmSivKey, Aes256Gcm, Aes256GcmKey, Aes256GcmSiv,
  Aes256GcmSivKey, Argon2Params, Argon2d, Argon2i, Argon2id, AsconAead128, AsconAead128Key, Blake2b256, Blake2b512,
  Blake2s128, Blake2s256, Blake3, Blake3KeyedHash, ChaCha20Poly1305, ChaCha20Poly1305Key, Crc32, EcdsaP256SecretKey,
  EcdsaP384SecretKey, Ed25519PublicKey, Ed25519SecretKey, Ed25519Signature, HkdfSha256, HkdfSha384, HmacSha256,
  HmacSha384, HmacSha512, Kmac256, Pbkdf2Sha256, Pbkdf2Sha512, RsaOaepProfile, RsaPkcs1v15Profile, RsaPrivateKey,
  RsaPssProfile, Scrypt, ScryptParams, SecretBytes, Sha256, X25519PublicKey, X25519SecretKey, XChaCha20Poly1305,
  XChaCha20Poly1305Key,
  aead::{Nonce96, Nonce128, Nonce192, Nonce256},
  checksum::Checksum,
  traits::ct,
};

const STATUS_ERR: u8 = 0;
const STATUS_OK: u8 = 1;

/// Convert a nullable C pointer/length pair into a shared byte slice.
///
/// # Safety
///
/// If `len != 0`, `ptr` must be valid for reads of `len` bytes and must remain
/// live for the returned borrow.
unsafe fn input_slice<'a>(ptr: *const u8, len: usize) -> Option<&'a [u8]> {
  if ptr.is_null() {
    return (len == 0).then_some(&[]);
  }

  // SAFETY: Constructs a shared slice from caller-provided FFI input because:
  // 1. The caller contract requires `ptr` to be valid for reads of `len` bytes.
  // 2. The returned borrow is tied to the harness function call and does not escape.
  // 3. Null was handled above, including the zero-length case.
  Some(unsafe { slice::from_raw_parts(ptr, len) })
}

/// Convert a nullable C pointer/length pair into a mutable byte slice.
///
/// # Safety
///
/// If `len != 0`, `ptr` must be valid for writes of `len` bytes, uniquely
/// borrowed for the duration of the call, and must remain live for the returned
/// borrow.
unsafe fn output_slice<'a>(ptr: *mut u8, len: usize) -> Option<&'a mut [u8]> {
  if ptr.is_null() {
    return (len == 0).then_some(&mut []);
  }

  // SAFETY: Constructs a mutable slice from caller-provided FFI output because:
  // 1. The caller contract requires `ptr` to be valid for writes of `len` bytes.
  // 2. The caller contract requires unique access for the duration of this call.
  // 3. The returned borrow is used only inside the harness function and does not escape.
  // 4. Null was handled above, including the zero-length case.
  Some(unsafe { slice::from_raw_parts_mut(ptr, len) })
}

/// Read a fixed-size input array by value from a C pointer.
///
/// # Safety
///
/// `ptr` must be valid for reads of `N` bytes and may be unaligned.
unsafe fn read_array<const N: usize>(ptr: *const u8) -> Option<[u8; N]> {
  if ptr.is_null() {
    return None;
  }

  let mut out = [0u8; N];
  // SAFETY: Copies a fixed-size input array from FFI memory because:
  // 1. The caller contract requires `ptr` to be valid for reads of `N` bytes.
  // 2. `out` is a stack-owned `[u8; N]` valid for writes of `N` bytes.
  // 3. `copy_nonoverlapping` is correct even for unaligned byte pointers.
  // 4. Source and destination cannot overlap because `out` is newly allocated stack storage.
  unsafe { ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), N) };
  Some(out)
}

/// Write a fixed-size output array to a C pointer.
///
/// # Safety
///
/// `ptr` must be valid for writes of `N` bytes and may be unaligned.
unsafe fn write_array<const N: usize>(ptr: *mut u8, value: &[u8; N]) -> bool {
  if ptr.is_null() {
    return false;
  }

  // SAFETY: Copies a fixed-size output array into FFI memory because:
  // 1. The caller contract requires `ptr` to be valid for writes of `N` bytes.
  // 2. `value` is a valid shared reference to exactly `N` initialized bytes.
  // 3. `copy_nonoverlapping` is correct even for unaligned byte pointers.
  // 4. Caller-provided output must not overlap `value`; `value` is owned by this stack frame or
  //    production API result.
  unsafe { ptr::copy_nonoverlapping(value.as_ptr(), ptr, N) };
  true
}

/// Constant-time equality harness.
#[unsafe(no_mangle)]
pub extern "C" fn ct_entry_constant_time_eq(a: *const u8, a_len: usize, b: *const u8, b_len: usize) -> u8 {
  // SAFETY: FFI inputs are converted under `input_slice`'s contract; invalid
  // null/nonzero pairs return an error status instead of dereferencing.
  let Some(a) = (unsafe { input_slice(a, a_len) }) else {
    return STATUS_ERR;
  };
  // SAFETY: Same reasoning as above for `b`.
  let Some(b) = (unsafe { input_slice(b, b_len) }) else {
    return STATUS_ERR;
  };

  u8::from(ct::constant_time_eq(a, b))
}

/// HMAC-SHA256 tag verification harness.
#[unsafe(no_mangle)]
pub extern "C" fn ct_entry_hmac_sha256_verify(
  key: *const u8,
  key_len: usize,
  data: *const u8,
  data_len: usize,
  expected_tag: *const u8,
) -> u8 {
  // SAFETY: FFI input pointers are validated by `input_slice` / `read_array`.
  let Some(key) = (unsafe { input_slice(key, key_len) }) else {
    return STATUS_ERR;
  };
  // SAFETY: FFI input pointers are validated by `input_slice` / `read_array`.
  let Some(data) = (unsafe { input_slice(data, data_len) }) else {
    return STATUS_ERR;
  };
  // SAFETY: The expected tag pointer must reference exactly 32 readable bytes.
  let Some(expected_tag) = (unsafe { read_array::<32>(expected_tag) }) else {
    return STATUS_ERR;
  };

  HmacSha256::verify_tag(key, data, &expected_tag)
    .map(|()| STATUS_OK)
    .unwrap_or(STATUS_ERR)
}

/// BLAKE3 keyed-tag verification harness.
#[unsafe(no_mangle)]
pub extern "C" fn ct_entry_blake3_verify_keyed(
  key: *const u8,
  data: *const u8,
  data_len: usize,
  expected_tag: *const u8,
) -> u8 {
  // SAFETY: The key pointer must reference exactly 32 readable bytes.
  let Some(key) = (unsafe { read_array::<32>(key) }) else {
    return STATUS_ERR;
  };
  // SAFETY: FFI input pointers are validated by `input_slice`.
  let Some(data) = (unsafe { input_slice(data, data_len) }) else {
    return STATUS_ERR;
  };
  // SAFETY: The expected tag pointer must reference exactly 32 readable bytes.
  let Some(expected_tag) = (unsafe { read_array::<32>(expected_tag) }) else {
    return STATUS_ERR;
  };

  let expected = Blake3KeyedHash::from_bytes(expected_tag);
  Blake3::verify_keyed(&key, data, &expected)
    .map(|()| STATUS_OK)
    .unwrap_or(STATUS_ERR)
}

/// ChaCha20-Poly1305 open/authentication harness.
#[unsafe(no_mangle)]
pub extern "C" fn ct_entry_chacha20poly1305_open(
  key: *const u8,
  nonce: *const u8,
  aad: *const u8,
  aad_len: usize,
  buffer: *mut u8,
  buffer_len: usize,
  tag: *const u8,
) -> u8 {
  // SAFETY: Fixed-size FFI inputs are copied by value after null checks.
  let Some(key) = (unsafe { read_array::<32>(key) }) else {
    return STATUS_ERR;
  };
  // SAFETY: Fixed-size FFI inputs are copied by value after null checks.
  let Some(nonce) = (unsafe { read_array::<12>(nonce) }) else {
    return STATUS_ERR;
  };
  // SAFETY: FFI input pointers are validated by `input_slice`.
  let Some(aad) = (unsafe { input_slice(aad, aad_len) }) else {
    return STATUS_ERR;
  };
  // SAFETY: FFI output pointer is validated by `output_slice`.
  let Some(buffer) = (unsafe { output_slice(buffer, buffer_len) }) else {
    return STATUS_ERR;
  };
  // SAFETY: Fixed-size FFI input is copied by value after null check.
  let Some(tag) = (unsafe { read_array::<16>(tag) }) else {
    return STATUS_ERR;
  };

  let key = ChaCha20Poly1305Key::from_bytes(key);
  let nonce = Nonce96::from_bytes(nonce);
  let cipher = ChaCha20Poly1305::new(&key);
  let Ok(tag) = ChaCha20Poly1305::tag_from_slice(&tag) else {
    return STATUS_ERR;
  };

  cipher
    .decrypt_in_place(&nonce, aad, buffer, &tag)
    .map(|()| STATUS_OK)
    .unwrap_or(STATUS_ERR)
}

/// X25519 scalar multiplication / shared-secret harness.
#[unsafe(no_mangle)]
pub extern "C" fn ct_entry_x25519(out: *mut u8, scalar: *const u8, point: *const u8) -> u8 {
  // SAFETY: Fixed-size FFI inputs are copied by value after null checks.
  let Some(scalar) = (unsafe { read_array::<32>(scalar) }) else {
    return STATUS_ERR;
  };
  // SAFETY: Fixed-size FFI inputs are copied by value after null checks.
  let Some(point) = (unsafe { read_array::<32>(point) }) else {
    return STATUS_ERR;
  };

  let secret = X25519SecretKey::from_bytes(scalar);
  let public = X25519PublicKey::from_bytes(point);
  let Ok(shared) = secret.diffie_hellman(&public) else {
    return STATUS_ERR;
  };
  // SAFETY: The output pointer must reference exactly 32 writable bytes.
  if unsafe { write_array(out, shared.as_bytes()) } {
    STATUS_OK
  } else {
    STATUS_ERR
  }
}

/// Ed25519 signing harness.
#[unsafe(no_mangle)]
pub extern "C" fn ct_entry_ed25519_sign(
  out: *mut u8,
  secret_key: *const u8,
  message: *const u8,
  message_len: usize,
) -> u8 {
  // SAFETY: Fixed-size FFI input is copied by value after null check.
  let Some(secret_key) = (unsafe { read_array::<32>(secret_key) }) else {
    return STATUS_ERR;
  };
  // SAFETY: FFI input pointer is validated by `input_slice`.
  let Some(message) = (unsafe { input_slice(message, message_len) }) else {
    return STATUS_ERR;
  };

  let secret_key = Ed25519SecretKey::from_bytes(secret_key);
  let signature = secret_key.sign(message);
  // SAFETY: The output pointer must reference exactly 64 writable bytes.
  if unsafe { write_array(out, signature.as_bytes()) } {
    STATUS_OK
  } else {
    STATUS_ERR
  }
}

/// ECDSA/P-256 signing harness with caller-supplied projective blinding.
#[unsafe(no_mangle)]
pub extern "C" fn ct_entry_ecdsa_p256_sign(
  out: *mut u8,
  secret_key: *const u8,
  blind: *const u8,
  message: *const u8,
  message_len: usize,
) -> u8 {
  // SAFETY: Fixed-size FFI input is copied by value after null check.
  let Some(secret_key) = (unsafe { read_array::<32>(secret_key) }) else {
    return STATUS_ERR;
  };
  // SAFETY: Fixed-size FFI input is copied by value after null check.
  let Some(blind) = (unsafe { read_array::<64>(blind) }) else {
    return STATUS_ERR;
  };
  // SAFETY: FFI input pointer is validated by `input_slice`.
  let Some(message) = (unsafe { input_slice(message, message_len) }) else {
    return STATUS_ERR;
  };

  let Ok(secret_key) = EcdsaP256SecretKey::from_bytes(secret_key) else {
    return STATUS_ERR;
  };
  let Ok(signature) = secret_key.try_sign_blinded(message, |out| out.copy_from_slice(&blind)) else {
    return STATUS_ERR;
  };
  let signature = signature.to_bytes();

  // SAFETY: The output pointer must reference exactly 64 writable bytes.
  if unsafe { write_array(out, &signature) } {
    STATUS_OK
  } else {
    STATUS_ERR
  }
}

/// ECDSA/P-384 signing harness with caller-supplied projective blinding.
#[unsafe(no_mangle)]
pub extern "C" fn ct_entry_ecdsa_p384_sign(
  out: *mut u8,
  secret_key: *const u8,
  blind: *const u8,
  message: *const u8,
  message_len: usize,
) -> u8 {
  // SAFETY: Fixed-size FFI input is copied by value after null check.
  let Some(secret_key) = (unsafe { read_array::<48>(secret_key) }) else {
    return STATUS_ERR;
  };
  // SAFETY: Fixed-size FFI input is copied by value after null check.
  let Some(blind) = (unsafe { read_array::<96>(blind) }) else {
    return STATUS_ERR;
  };
  // SAFETY: FFI input pointer is validated by `input_slice`.
  let Some(message) = (unsafe { input_slice(message, message_len) }) else {
    return STATUS_ERR;
  };

  let Ok(secret_key) = EcdsaP384SecretKey::from_bytes(secret_key) else {
    return STATUS_ERR;
  };
  let Ok(signature) = secret_key.try_sign_blinded(message, |out| out.copy_from_slice(&blind)) else {
    return STATUS_ERR;
  };
  let signature = signature.to_bytes();

  // SAFETY: The output pointer must reference exactly 96 writable bytes.
  if unsafe { write_array(out, &signature) } {
    STATUS_OK
  } else {
    STATUS_ERR
  }
}

/// PBKDF2-HMAC-SHA256 verification harness.
#[unsafe(no_mangle)]
pub extern "C" fn ct_entry_pbkdf2_sha256_verify(
  password: *const u8,
  password_len: usize,
  salt: *const u8,
  salt_len: usize,
  iterations: u32,
  expected: *const u8,
  expected_len: usize,
) -> u8 {
  // SAFETY: FFI input pointers are validated by `input_slice`.
  let Some(password) = (unsafe { input_slice(password, password_len) }) else {
    return STATUS_ERR;
  };
  // SAFETY: FFI input pointers are validated by `input_slice`.
  let Some(salt) = (unsafe { input_slice(salt, salt_len) }) else {
    return STATUS_ERR;
  };
  // SAFETY: FFI input pointers are validated by `input_slice`.
  let Some(expected) = (unsafe { input_slice(expected, expected_len) }) else {
    return STATUS_ERR;
  };

  Pbkdf2Sha256::verify_password(password, salt, iterations, expected)
    .map(|()| STATUS_OK)
    .unwrap_or(STATUS_ERR)
}

macro_rules! fixed_tag_verify_entry {
  ($name:ident, $ty:ty, $tag_len:literal) => {
    #[unsafe(no_mangle)]
    pub extern "C" fn $name(
      key: *const u8,
      key_len: usize,
      data: *const u8,
      data_len: usize,
      expected_tag: *const u8,
    ) -> u8 {
      // SAFETY: FFI input pointers are validated by `input_slice` / `read_array`.
      let Some(key) = (unsafe { input_slice(key, key_len) }) else {
        return STATUS_ERR;
      };
      // SAFETY: FFI input pointers are validated by `input_slice` / `read_array`.
      let Some(data) = (unsafe { input_slice(data, data_len) }) else {
        return STATUS_ERR;
      };
      // SAFETY: The expected tag pointer must reference exactly `$tag_len` readable bytes.
      let Some(expected_tag) = (unsafe { read_array::<$tag_len>(expected_tag) }) else {
        return STATUS_ERR;
      };

      <$ty>::verify_tag(key, data, &expected_tag)
        .map(|()| STATUS_OK)
        .unwrap_or(STATUS_ERR)
    }
  };
}

fixed_tag_verify_entry!(ct_entry_hmac_sha384_verify, HmacSha384, 48);
fixed_tag_verify_entry!(ct_entry_hmac_sha512_verify, HmacSha512, 64);

#[unsafe(no_mangle)]
pub extern "C" fn ct_entry_secret_bytes32_eq(a: *const u8, b: *const u8) -> u8 {
  // SAFETY: Fixed-size FFI inputs are copied by value after null checks.
  let Some(a) = (unsafe { read_array::<32>(a) }) else {
    return STATUS_ERR;
  };
  // SAFETY: Fixed-size FFI inputs are copied by value after null checks.
  let Some(b) = (unsafe { read_array::<32>(b) }) else {
    return STATUS_ERR;
  };

  u8::from(SecretBytes::<32>::from(a) == SecretBytes::<32>::from(b))
}

#[unsafe(no_mangle)]
pub extern "C" fn ct_entry_x25519_secret_eq(a: *const u8, b: *const u8) -> u8 {
  // SAFETY: Fixed-size FFI inputs are copied by value after null checks.
  let Some(a) = (unsafe { read_array::<32>(a) }) else {
    return STATUS_ERR;
  };
  // SAFETY: Fixed-size FFI inputs are copied by value after null checks.
  let Some(b) = (unsafe { read_array::<32>(b) }) else {
    return STATUS_ERR;
  };

  u8::from(X25519SecretKey::from_bytes(a) == X25519SecretKey::from_bytes(b))
}

#[unsafe(no_mangle)]
pub extern "C" fn ct_entry_ed25519_secret_eq(a: *const u8, b: *const u8) -> u8 {
  // SAFETY: Fixed-size FFI inputs are copied by value after null checks.
  let Some(a) = (unsafe { read_array::<32>(a) }) else {
    return STATUS_ERR;
  };
  // SAFETY: Fixed-size FFI inputs are copied by value after null checks.
  let Some(b) = (unsafe { read_array::<32>(b) }) else {
    return STATUS_ERR;
  };

  u8::from(Ed25519SecretKey::from_bytes(a) == Ed25519SecretKey::from_bytes(b))
}

#[unsafe(no_mangle)]
pub extern "C" fn ct_entry_secret_bytes32_debug_masked(secret: *const u8) -> u8 {
  // SAFETY: Fixed-size FFI input is copied by value after null check.
  let Some(secret) = (unsafe { read_array::<32>(secret) }) else {
    return STATUS_ERR;
  };

  let formatted = format!("{:?}", SecretBytes::<32>::from(secret));
  u8::from(formatted == "SecretBytes(****)")
}

#[unsafe(no_mangle)]
pub extern "C" fn ct_entry_kmac256_verify(
  key: *const u8,
  key_len: usize,
  customization: *const u8,
  customization_len: usize,
  data: *const u8,
  data_len: usize,
  expected_tag: *const u8,
  expected_tag_len: usize,
) -> u8 {
  // SAFETY: FFI input pointers are validated by `input_slice`.
  let Some(key) = (unsafe { input_slice(key, key_len) }) else {
    return STATUS_ERR;
  };
  // SAFETY: FFI input pointers are validated by `input_slice`.
  let Some(customization) = (unsafe { input_slice(customization, customization_len) }) else {
    return STATUS_ERR;
  };
  // SAFETY: FFI input pointers are validated by `input_slice`.
  let Some(data) = (unsafe { input_slice(data, data_len) }) else {
    return STATUS_ERR;
  };
  // SAFETY: FFI input pointers are validated by `input_slice`.
  let Some(expected_tag) = (unsafe { input_slice(expected_tag, expected_tag_len) }) else {
    return STATUS_ERR;
  };

  Kmac256::verify_tag(key, customization, data, expected_tag)
    .map(|()| STATUS_OK)
    .unwrap_or(STATUS_ERR)
}

macro_rules! aead_open_entry {
  ($name:ident, $cipher:ty, $key:ty, $nonce:ty, $key_len:literal, $nonce_len:literal) => {
    #[unsafe(no_mangle)]
    pub extern "C" fn $name(
      key: *const u8,
      nonce: *const u8,
      aad: *const u8,
      aad_len: usize,
      buffer: *mut u8,
      buffer_len: usize,
      tag: *const u8,
    ) -> u8 {
      // SAFETY: Fixed-size FFI inputs are copied by value after null checks.
      let Some(key) = (unsafe { read_array::<$key_len>(key) }) else {
        return STATUS_ERR;
      };
      // SAFETY: Fixed-size FFI inputs are copied by value after null checks.
      let Some(nonce) = (unsafe { read_array::<$nonce_len>(nonce) }) else {
        return STATUS_ERR;
      };
      // SAFETY: FFI input pointer is validated by `input_slice`.
      let Some(aad) = (unsafe { input_slice(aad, aad_len) }) else {
        return STATUS_ERR;
      };
      // SAFETY: FFI output pointer is validated by `output_slice`.
      let Some(buffer) = (unsafe { output_slice(buffer, buffer_len) }) else {
        return STATUS_ERR;
      };
      // SAFETY: Fixed-size FFI input is copied by value after null check.
      let Some(tag) = (unsafe { read_array::<16>(tag) }) else {
        return STATUS_ERR;
      };

      let key = <$key>::from_bytes(key);
      let nonce = <$nonce>::from_bytes(nonce);
      let cipher = <$cipher>::new(&key);
      let Ok(tag) = <$cipher>::tag_from_slice(&tag) else {
        return STATUS_ERR;
      };

      cipher
        .decrypt_in_place(&nonce, aad, buffer, &tag)
        .map(|()| STATUS_OK)
        .unwrap_or(STATUS_ERR)
    }
  };
}

aead_open_entry!(ct_entry_aes128gcm_open, Aes128Gcm, Aes128GcmKey, Nonce96, 16, 12);
aead_open_entry!(ct_entry_aes256gcm_open, Aes256Gcm, Aes256GcmKey, Nonce96, 32, 12);
aead_open_entry!(
  ct_entry_aes128gcmsiv_open,
  Aes128GcmSiv,
  Aes128GcmSivKey,
  Nonce96,
  16,
  12
);
aead_open_entry!(
  ct_entry_aes256gcmsiv_open,
  Aes256GcmSiv,
  Aes256GcmSivKey,
  Nonce96,
  32,
  12
);
aead_open_entry!(
  ct_entry_xchacha20poly1305_open,
  XChaCha20Poly1305,
  XChaCha20Poly1305Key,
  Nonce192,
  32,
  24
);
aead_open_entry!(ct_entry_aegis256_open, Aegis256, Aegis256Key, Nonce256, 32, 32);
aead_open_entry!(
  ct_entry_ascon_aead128_open,
  AsconAead128,
  AsconAead128Key,
  Nonce128,
  16,
  16
);

macro_rules! hkdf_derive_entry {
  ($name:ident, $ty:ty) => {
    #[unsafe(no_mangle)]
    pub extern "C" fn $name(
      salt: *const u8,
      salt_len: usize,
      ikm: *const u8,
      ikm_len: usize,
      info: *const u8,
      info_len: usize,
      out: *mut u8,
      out_len: usize,
    ) -> u8 {
      // SAFETY: FFI input/output pointers are validated by slice helpers.
      let Some(salt) = (unsafe { input_slice(salt, salt_len) }) else {
        return STATUS_ERR;
      };
      // SAFETY: FFI input/output pointers are validated by slice helpers.
      let Some(ikm) = (unsafe { input_slice(ikm, ikm_len) }) else {
        return STATUS_ERR;
      };
      // SAFETY: FFI input/output pointers are validated by slice helpers.
      let Some(info) = (unsafe { input_slice(info, info_len) }) else {
        return STATUS_ERR;
      };
      // SAFETY: FFI input/output pointers are validated by slice helpers.
      let Some(out) = (unsafe { output_slice(out, out_len) }) else {
        return STATUS_ERR;
      };

      <$ty>::derive(salt, ikm, info, out)
        .map(|()| STATUS_OK)
        .unwrap_or(STATUS_ERR)
    }
  };
}

hkdf_derive_entry!(ct_entry_hkdf_sha256_derive, HkdfSha256);
hkdf_derive_entry!(ct_entry_hkdf_sha384_derive, HkdfSha384);

macro_rules! pbkdf2_entry {
  ($derive_name:ident, $verify_name:ident, $ty:ty) => {
    #[unsafe(no_mangle)]
    pub extern "C" fn $derive_name(
      password: *const u8,
      password_len: usize,
      salt: *const u8,
      salt_len: usize,
      iterations: u32,
      out: *mut u8,
      out_len: usize,
    ) -> u8 {
      // SAFETY: FFI input/output pointers are validated by slice helpers.
      let Some(password) = (unsafe { input_slice(password, password_len) }) else {
        return STATUS_ERR;
      };
      // SAFETY: FFI input/output pointers are validated by slice helpers.
      let Some(salt) = (unsafe { input_slice(salt, salt_len) }) else {
        return STATUS_ERR;
      };
      // SAFETY: FFI input/output pointers are validated by slice helpers.
      let Some(out) = (unsafe { output_slice(out, out_len) }) else {
        return STATUS_ERR;
      };

      <$ty>::derive_key(password, salt, iterations, out)
        .map(|()| STATUS_OK)
        .unwrap_or(STATUS_ERR)
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn $verify_name(
      password: *const u8,
      password_len: usize,
      salt: *const u8,
      salt_len: usize,
      iterations: u32,
      expected: *const u8,
      expected_len: usize,
    ) -> u8 {
      // SAFETY: FFI input pointers are validated by `input_slice`.
      let Some(password) = (unsafe { input_slice(password, password_len) }) else {
        return STATUS_ERR;
      };
      // SAFETY: FFI input pointers are validated by `input_slice`.
      let Some(salt) = (unsafe { input_slice(salt, salt_len) }) else {
        return STATUS_ERR;
      };
      // SAFETY: FFI input pointers are validated by `input_slice`.
      let Some(expected) = (unsafe { input_slice(expected, expected_len) }) else {
        return STATUS_ERR;
      };

      <$ty>::verify_password(password, salt, iterations, expected)
        .map(|()| STATUS_OK)
        .unwrap_or(STATUS_ERR)
    }
  };
}

pbkdf2_entry!(
  ct_entry_pbkdf2_sha256_derive,
  ct_entry_pbkdf2_sha256_verify_full,
  Pbkdf2Sha256
);
pbkdf2_entry!(
  ct_entry_pbkdf2_sha512_derive,
  ct_entry_pbkdf2_sha512_verify,
  Pbkdf2Sha512
);

macro_rules! argon2_entry {
  ($derive_name:ident, $verify_name:ident, $ty:ty) => {
    #[unsafe(no_mangle)]
    pub extern "C" fn $derive_name(
      password: *const u8,
      password_len: usize,
      salt: *const u8,
      salt_len: usize,
      memory_cost_kib: u32,
      time_cost: u32,
      out: *mut u8,
      out_len: usize,
    ) -> u8 {
      // SAFETY: FFI input/output pointers are validated by slice helpers.
      let Some(password) = (unsafe { input_slice(password, password_len) }) else {
        return STATUS_ERR;
      };
      // SAFETY: FFI input/output pointers are validated by slice helpers.
      let Some(salt) = (unsafe { input_slice(salt, salt_len) }) else {
        return STATUS_ERR;
      };
      // SAFETY: FFI input/output pointers are validated by slice helpers.
      let Some(out) = (unsafe { output_slice(out, out_len) }) else {
        return STATUS_ERR;
      };
      let Ok(params) = Argon2Params::new()
        .memory_cost_kib(memory_cost_kib)
        .time_cost(time_cost)
        .output_len(out_len as u32)
        .build()
      else {
        return STATUS_ERR;
      };

      <$ty>::hash(&params, password, salt, out)
        .map(|()| STATUS_OK)
        .unwrap_or(STATUS_ERR)
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn $verify_name(
      password: *const u8,
      password_len: usize,
      salt: *const u8,
      salt_len: usize,
      memory_cost_kib: u32,
      time_cost: u32,
      expected: *const u8,
      expected_len: usize,
    ) -> u8 {
      // SAFETY: FFI input pointers are validated by `input_slice`.
      let Some(password) = (unsafe { input_slice(password, password_len) }) else {
        return STATUS_ERR;
      };
      // SAFETY: FFI input pointers are validated by `input_slice`.
      let Some(salt) = (unsafe { input_slice(salt, salt_len) }) else {
        return STATUS_ERR;
      };
      // SAFETY: FFI input pointers are validated by `input_slice`.
      let Some(expected) = (unsafe { input_slice(expected, expected_len) }) else {
        return STATUS_ERR;
      };
      let Ok(params) = Argon2Params::new()
        .memory_cost_kib(memory_cost_kib)
        .time_cost(time_cost)
        .output_len(expected_len as u32)
        .build()
      else {
        return STATUS_ERR;
      };

      <$ty>::verify(&params, password, salt, expected)
        .map(|()| STATUS_OK)
        .unwrap_or(STATUS_ERR)
    }
  };
}

argon2_entry!(ct_entry_argon2i_hash, ct_entry_argon2i_verify, Argon2i);
argon2_entry!(ct_entry_argon2d_hash, ct_entry_argon2d_verify, Argon2d);
argon2_entry!(ct_entry_argon2id_hash, ct_entry_argon2id_verify, Argon2id);

#[unsafe(no_mangle)]
pub extern "C" fn ct_entry_scrypt_verify(
  password: *const u8,
  password_len: usize,
  salt: *const u8,
  salt_len: usize,
  log_n: u8,
  r: u32,
  p: u32,
  expected: *const u8,
  expected_len: usize,
) -> u8 {
  // SAFETY: FFI input pointers are validated by `input_slice`.
  let Some(password) = (unsafe { input_slice(password, password_len) }) else {
    return STATUS_ERR;
  };
  // SAFETY: FFI input pointers are validated by `input_slice`.
  let Some(salt) = (unsafe { input_slice(salt, salt_len) }) else {
    return STATUS_ERR;
  };
  // SAFETY: FFI input pointers are validated by `input_slice`.
  let Some(expected) = (unsafe { input_slice(expected, expected_len) }) else {
    return STATUS_ERR;
  };
  let Ok(params) = ScryptParams::new()
    .log_n(log_n)
    .r(r)
    .p(p)
    .output_len(expected_len as u32)
    .build()
  else {
    return STATUS_ERR;
  };

  Scrypt::verify(&params, password, salt, expected)
    .map(|()| STATUS_OK)
    .unwrap_or(STATUS_ERR)
}

#[unsafe(no_mangle)]
pub extern "C" fn ct_entry_ed25519_verify(
  public_key: *const u8,
  message: *const u8,
  message_len: usize,
  signature: *const u8,
) -> u8 {
  // SAFETY: Fixed-size FFI inputs are copied by value after null checks.
  let Some(public_key) = (unsafe { read_array::<32>(public_key) }) else {
    return STATUS_ERR;
  };
  // SAFETY: FFI input pointer is validated by `input_slice`.
  let Some(message) = (unsafe { input_slice(message, message_len) }) else {
    return STATUS_ERR;
  };
  // SAFETY: Fixed-size FFI input is copied by value after null check.
  let Some(signature) = (unsafe { read_array::<64>(signature) }) else {
    return STATUS_ERR;
  };

  Ed25519PublicKey::from_bytes(public_key)
    .verify(message, &Ed25519Signature::from_bytes(signature))
    .map(|()| STATUS_OK)
    .unwrap_or(STATUS_ERR)
}

#[unsafe(no_mangle)]
pub extern "C" fn ct_entry_rsa_pkcs1v15_sign_fixed_blinding(
  out: *mut u8,
  out_len: usize,
  pkcs8_der: *const u8,
  pkcs8_der_len: usize,
  message: *const u8,
  message_len: usize,
  blinding_factor: *const u8,
  blinding_factor_len: usize,
  blinding_inverse: *const u8,
  blinding_inverse_len: usize,
) -> u8 {
  // SAFETY: FFI input/output pointers are validated by slice helpers.
  let Some(out) = (unsafe { output_slice(out, out_len) }) else {
    return STATUS_ERR;
  };
  // SAFETY: FFI input/output pointers are validated by slice helpers.
  let Some(pkcs8_der) = (unsafe { input_slice(pkcs8_der, pkcs8_der_len) }) else {
    return STATUS_ERR;
  };
  // SAFETY: FFI input/output pointers are validated by slice helpers.
  let Some(message) = (unsafe { input_slice(message, message_len) }) else {
    return STATUS_ERR;
  };
  // SAFETY: FFI input/output pointers are validated by slice helpers.
  let Some(blinding_factor) = (unsafe { input_slice(blinding_factor, blinding_factor_len) }) else {
    return STATUS_ERR;
  };
  // SAFETY: FFI input/output pointers are validated by slice helpers.
  let Some(blinding_inverse) = (unsafe { input_slice(blinding_inverse, blinding_inverse_len) }) else {
    return STATUS_ERR;
  };

  let Ok(key) = RsaPrivateKey::from_pkcs8_der(pkcs8_der) else {
    return STATUS_ERR;
  };
  let mut scratch = key.private_scratch();
  key
    .sign_pkcs1v15_with_blinding_factor_and_scratch(
      RsaPkcs1v15Profile::Sha256,
      message,
      blinding_factor,
      blinding_inverse,
      out,
      &mut scratch,
    )
    .map(|()| STATUS_OK)
    .unwrap_or(STATUS_ERR)
}

#[unsafe(no_mangle)]
pub extern "C" fn ct_entry_rsa_pss_sign_fixed_blinding(
  out: *mut u8,
  out_len: usize,
  pkcs8_der: *const u8,
  pkcs8_der_len: usize,
  message: *const u8,
  message_len: usize,
  salt: *const u8,
  salt_len: usize,
  blinding_factor: *const u8,
  blinding_factor_len: usize,
  blinding_inverse: *const u8,
  blinding_inverse_len: usize,
) -> u8 {
  // SAFETY: FFI input/output pointers are validated by slice helpers.
  let Some(out) = (unsafe { output_slice(out, out_len) }) else {
    return STATUS_ERR;
  };
  // SAFETY: FFI input/output pointers are validated by slice helpers.
  let Some(pkcs8_der) = (unsafe { input_slice(pkcs8_der, pkcs8_der_len) }) else {
    return STATUS_ERR;
  };
  // SAFETY: FFI input/output pointers are validated by slice helpers.
  let Some(message) = (unsafe { input_slice(message, message_len) }) else {
    return STATUS_ERR;
  };
  // SAFETY: FFI input/output pointers are validated by slice helpers.
  let Some(salt) = (unsafe { input_slice(salt, salt_len) }) else {
    return STATUS_ERR;
  };
  // SAFETY: FFI input/output pointers are validated by slice helpers.
  let Some(blinding_factor) = (unsafe { input_slice(blinding_factor, blinding_factor_len) }) else {
    return STATUS_ERR;
  };
  // SAFETY: FFI input/output pointers are validated by slice helpers.
  let Some(blinding_inverse) = (unsafe { input_slice(blinding_inverse, blinding_inverse_len) }) else {
    return STATUS_ERR;
  };

  let Ok(key) = RsaPrivateKey::from_pkcs8_der(pkcs8_der) else {
    return STATUS_ERR;
  };
  let mut scratch = key.private_scratch();
  key
    .sign_pss_with_salt_and_blinding_factor_and_scratch(
      RsaPssProfile::Sha256,
      message,
      salt,
      blinding_factor,
      blinding_inverse,
      out,
      &mut scratch,
    )
    .map(|()| STATUS_OK)
    .unwrap_or(STATUS_ERR)
}

#[unsafe(no_mangle)]
pub extern "C" fn ct_entry_rsa_oaep_decrypt_fixed_blinding(
  out: *mut u8,
  out_len: usize,
  pkcs8_der: *const u8,
  pkcs8_der_len: usize,
  label: *const u8,
  label_len: usize,
  ciphertext: *const u8,
  ciphertext_len: usize,
  blinding_factor: *const u8,
  blinding_factor_len: usize,
  blinding_inverse: *const u8,
  blinding_inverse_len: usize,
) -> usize {
  // SAFETY: FFI input/output pointers are validated by slice helpers.
  let Some(out) = (unsafe { output_slice(out, out_len) }) else {
    return usize::MAX;
  };
  // SAFETY: FFI input/output pointers are validated by slice helpers.
  let Some(pkcs8_der) = (unsafe { input_slice(pkcs8_der, pkcs8_der_len) }) else {
    return usize::MAX;
  };
  // SAFETY: FFI input/output pointers are validated by slice helpers.
  let Some(label) = (unsafe { input_slice(label, label_len) }) else {
    return usize::MAX;
  };
  // SAFETY: FFI input/output pointers are validated by slice helpers.
  let Some(ciphertext) = (unsafe { input_slice(ciphertext, ciphertext_len) }) else {
    return usize::MAX;
  };
  // SAFETY: FFI input/output pointers are validated by slice helpers.
  let Some(blinding_factor) = (unsafe { input_slice(blinding_factor, blinding_factor_len) }) else {
    return usize::MAX;
  };
  // SAFETY: FFI input/output pointers are validated by slice helpers.
  let Some(blinding_inverse) = (unsafe { input_slice(blinding_inverse, blinding_inverse_len) }) else {
    return usize::MAX;
  };

  let Ok(key) = RsaPrivateKey::from_pkcs8_der(pkcs8_der) else {
    return usize::MAX;
  };
  let mut scratch = key.private_scratch();
  key
    .decrypt_oaep_with_blinding_factor_and_scratch(
      RsaOaepProfile::Sha256,
      label,
      ciphertext,
      blinding_factor,
      blinding_inverse,
      out,
      &mut scratch,
    )
    .unwrap_or(usize::MAX)
}

#[unsafe(no_mangle)]
pub extern "C" fn ct_entry_rsa_pkcs1v15_decrypt_fixed_blinding(
  out: *mut u8,
  out_len: usize,
  pkcs8_der: *const u8,
  pkcs8_der_len: usize,
  ciphertext: *const u8,
  ciphertext_len: usize,
  blinding_factor: *const u8,
  blinding_factor_len: usize,
  blinding_inverse: *const u8,
  blinding_inverse_len: usize,
) -> usize {
  // SAFETY: FFI input/output pointers are validated by slice helpers.
  let Some(out) = (unsafe { output_slice(out, out_len) }) else {
    return usize::MAX;
  };
  // SAFETY: FFI input/output pointers are validated by slice helpers.
  let Some(pkcs8_der) = (unsafe { input_slice(pkcs8_der, pkcs8_der_len) }) else {
    return usize::MAX;
  };
  // SAFETY: FFI input/output pointers are validated by slice helpers.
  let Some(ciphertext) = (unsafe { input_slice(ciphertext, ciphertext_len) }) else {
    return usize::MAX;
  };
  // SAFETY: FFI input/output pointers are validated by slice helpers.
  let Some(blinding_factor) = (unsafe { input_slice(blinding_factor, blinding_factor_len) }) else {
    return usize::MAX;
  };
  // SAFETY: FFI input/output pointers are validated by slice helpers.
  let Some(blinding_inverse) = (unsafe { input_slice(blinding_inverse, blinding_inverse_len) }) else {
    return usize::MAX;
  };

  let Ok(key) = RsaPrivateKey::from_pkcs8_der(pkcs8_der) else {
    return usize::MAX;
  };
  let mut scratch = key.private_scratch();
  key
    .decrypt_pkcs1v15_with_blinding_factor_and_scratch(
      ciphertext,
      blinding_factor,
      blinding_inverse,
      out,
      &mut scratch,
    )
    .unwrap_or(usize::MAX)
}

#[unsafe(no_mangle)]
pub extern "C" fn ct_entry_rsa_private_key_pkcs8_roundtrip(
  out: *mut u8,
  out_len: usize,
  pkcs8_der: *const u8,
  pkcs8_der_len: usize,
) -> usize {
  // SAFETY: FFI input/output pointers are validated by slice helpers.
  let Some(out) = (unsafe { output_slice(out, out_len) }) else {
    return usize::MAX;
  };
  // SAFETY: FFI input/output pointers are validated by slice helpers.
  let Some(pkcs8_der) = (unsafe { input_slice(pkcs8_der, pkcs8_der_len) }) else {
    return usize::MAX;
  };

  let Ok(key) = RsaPrivateKey::from_pkcs8_der(pkcs8_der) else {
    return usize::MAX;
  };
  let der = key.to_pkcs8_der();
  if der.len() > out.len() {
    return usize::MAX;
  }
  out[..der.len()].copy_from_slice(&der);
  der.len()
}

macro_rules! blake2_keyed_entry {
  ($name:ident, $ty:ty, $out_len:literal) => {
    #[unsafe(no_mangle)]
    pub extern "C" fn $name(key: *const u8, key_len: usize, data: *const u8, data_len: usize, out: *mut u8) -> u8 {
      // SAFETY: FFI input pointers are validated by slice helpers.
      let Some(key) = (unsafe { input_slice(key, key_len) }) else {
        return STATUS_ERR;
      };
      // SAFETY: FFI input pointers are validated by slice helpers.
      let Some(data) = (unsafe { input_slice(data, data_len) }) else {
        return STATUS_ERR;
      };

      let digest = <$ty>::keyed_digest(key, data);
      // SAFETY: The output pointer must reference exactly `$out_len` writable bytes.
      if unsafe { write_array::<$out_len>(out, &digest) } {
        STATUS_OK
      } else {
        STATUS_ERR
      }
    }
  };
}

blake2_keyed_entry!(ct_entry_blake2b256_keyed_digest, Blake2b256, 32);
blake2_keyed_entry!(ct_entry_blake2b512_keyed_digest, Blake2b512, 64);
blake2_keyed_entry!(ct_entry_blake2s128_keyed_digest, Blake2s128, 16);
blake2_keyed_entry!(ct_entry_blake2s256_keyed_digest, Blake2s256, 32);

#[unsafe(no_mangle)]
pub extern "C" fn ct_entry_blake3_keyed_digest(key: *const u8, data: *const u8, data_len: usize, out: *mut u8) -> u8 {
  // SAFETY: Fixed-size FFI input is copied by value after null check.
  let Some(key) = (unsafe { read_array::<32>(key) }) else {
    return STATUS_ERR;
  };
  // SAFETY: FFI input pointer is validated by `input_slice`.
  let Some(data) = (unsafe { input_slice(data, data_len) }) else {
    return STATUS_ERR;
  };

  let digest = Blake3::keyed_digest(&key, data);
  // SAFETY: The output pointer must reference exactly 32 writable bytes.
  if unsafe { write_array(out, digest.as_bytes()) } {
    STATUS_OK
  } else {
    STATUS_ERR
  }
}

#[unsafe(no_mangle)]
pub extern "C" fn ct_entry_public_sha256_digest(data: *const u8, data_len: usize, out: *mut u8) -> u8 {
  // SAFETY: FFI input pointer is validated by `input_slice`.
  let Some(data) = (unsafe { input_slice(data, data_len) }) else {
    return STATUS_ERR;
  };
  let digest = Sha256::digest(data);
  // SAFETY: The output pointer must reference exactly 32 writable bytes.
  if unsafe { write_array(out, &digest) } {
    STATUS_OK
  } else {
    STATUS_ERR
  }
}

#[unsafe(no_mangle)]
pub extern "C" fn ct_entry_public_crc32_checksum(data: *const u8, data_len: usize) -> u32 {
  // SAFETY: FFI input pointer is validated by `input_slice`.
  let Some(data) = (unsafe { input_slice(data, data_len) }) else {
    return 0;
  };

  Crc32::checksum(data)
}

#[unsafe(no_mangle)]
pub extern "C" fn ct_entry_classification_public_only() -> u8 {
  STATUS_OK
}

#[unsafe(no_mangle)]
pub extern "C" fn ct_entry_classification_best_effort() -> u8 {
  STATUS_OK
}
