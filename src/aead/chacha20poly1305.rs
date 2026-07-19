#![allow(clippy::indexing_slicing)]

//! ChaCha20-Poly1305 public AEAD surface.

use core::fmt;

use super::{
  AeadBufferError, LengthOverflow, Nonce96, OpenError, SealError, chacha20, poly1305, targets::AeadPrimitive,
};
use crate::traits::{Aead, ct};

const KEY_SIZE: usize = chacha20::KEY_SIZE;
const TAG_SIZE: usize = 16;
const NONCE_SIZE: usize = Nonce96::LENGTH;
const MAX_PLAINTEXT_LEN: u64 = (u32::MAX as u64) * (chacha20::BLOCK_SIZE as u64);
#[cfg(any(target_arch = "x86_64", all(target_arch = "powerpc64", target_endian = "little")))]
const SMALL_AAD_FAST_MAX: usize = 63;
#[cfg(all(target_arch = "powerpc64", target_endian = "little"))]
const POWER_SHORT_FAST_MAX: usize = chacha20::BLOCK_SIZE;
#[cfg(target_arch = "aarch64")]
const AARCH64_INTERLEAVED_MIN: usize = 1024;
#[cfg(target_arch = "aarch64")]
const AARCH64_INTERLEAVED_CHUNK: usize = 1024 * 1024;
#[cfg(all(
  feature = "diag",
  target_arch = "aarch64",
  any(target_os = "linux", target_os = "macos")
))]
const AARCH64_OWNED_PAR4_DIAG_CHUNK: usize = chacha20::BLOCK_SIZE * 8;
#[cfg(all(
  target_arch = "x86_64",
  target_os = "linux",
  any(test, all(not(debug_assertions), not(feature = "portable-only")))
))]
const X86_64_ASM_ZEN5_MAX: usize = 1024;
#[cfg(all(
  target_arch = "x86_64",
  target_os = "linux",
  any(test, all(not(debug_assertions), not(feature = "portable-only")))
))]
const X86_64_ASM_SPR_MAX: usize = 256;
#[cfg(all(
  target_arch = "x86_64",
  target_os = "linux",
  any(test, all(not(debug_assertions), not(feature = "portable-only")))
))]
const X86_64_OPEN_ASM_SHORT_MAX: usize = 256;

#[cfg(all(
  target_arch = "aarch64",
  any(target_os = "linux", target_os = "macos"),
  not(debug_assertions),
  not(feature = "portable-only")
))]
#[path = "chacha20poly1305/aarch64_asm.rs"]
mod aarch64_asm;
#[cfg(all(
  target_arch = "x86_64",
  target_os = "linux",
  any(feature = "diag", all(not(debug_assertions), not(feature = "portable-only")))
))]
#[path = "chacha20poly1305/x86_64_asm.rs"]
mod x86_64_asm;

define_aead_key_type!(ChaCha20Poly1305Key, KEY_SIZE, "ChaCha20-Poly1305 secret key bytes.");

define_aead_tag_type!(
  ChaCha20Poly1305Tag,
  TAG_SIZE,
  "ChaCha20-Poly1305 authentication tag bytes."
);

/// Portable ChaCha20-Poly1305 AEAD.
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "getrandom")]
/// # {
/// use rscrypto::{Aead, ChaCha20Poly1305, ChaCha20Poly1305Key};
///
/// let key = ChaCha20Poly1305Key::from_bytes([0x42; 32]);
/// let cipher = ChaCha20Poly1305::new(&key);
///
/// let mut buf = *b"hello";
/// let (nonce, tag) = cipher.seal_random_in_place(b"", &mut buf)?;
/// cipher.decrypt_in_place(&nonce, b"", &mut buf, &tag)?;
/// assert_eq!(&buf, b"hello");
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// Tampering is reported as an opaque verification failure.
///
/// ```
/// # #[cfg(feature = "getrandom")]
/// # {
/// use rscrypto::{Aead, ChaCha20Poly1305, ChaCha20Poly1305Key, aead::OpenError};
///
/// let key = ChaCha20Poly1305Key::from_bytes([0x42; 32]);
/// let cipher = ChaCha20Poly1305::new(&key);
///
/// let mut sealed = [0u8; 5 + ChaCha20Poly1305::TAG_SIZE];
/// let nonce = cipher.seal_random(b"", b"hello", &mut sealed)?;
/// sealed[0] ^= 1;
///
/// let mut opened = [0u8; 5];
/// assert_eq!(
///   cipher.decrypt(&nonce, b"", &sealed, &mut opened),
///   Err(OpenError::verification())
/// );
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct ChaCha20Poly1305 {
  key: ChaCha20Poly1305Key,
}

impl fmt::Debug for ChaCha20Poly1305 {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("ChaCha20Poly1305").finish_non_exhaustive()
  }
}

impl ChaCha20Poly1305 {
  /// Key length in bytes.
  pub const KEY_SIZE: usize = KEY_SIZE;

  /// Nonce length in bytes.
  pub const NONCE_SIZE: usize = NONCE_SIZE;

  /// Tag length in bytes.
  pub const TAG_SIZE: usize = TAG_SIZE;

  /// Construct a new ChaCha20-Poly1305 instance from `key`.
  #[inline]
  #[must_use]
  pub fn new(key: &ChaCha20Poly1305Key) -> Self {
    <Self as Aead>::new(key)
  }

  /// Rebuild a typed tag from raw tag bytes.
  #[inline]
  pub fn tag_from_slice(bytes: &[u8]) -> Result<ChaCha20Poly1305Tag, AeadBufferError> {
    <Self as Aead>::tag_from_slice(bytes)
  }

  /// Encrypt `buffer` in place and return the detached authentication tag.
  #[inline]
  pub fn encrypt_in_place(
    &self,
    nonce: &Nonce96,
    aad: &[u8],
    buffer: &mut [u8],
  ) -> Result<ChaCha20Poly1305Tag, SealError> {
    <Self as Aead>::encrypt_in_place(self, nonce, aad, buffer)
  }

  /// Decrypt `buffer` in place and verify the detached authentication tag.
  #[inline]
  pub fn decrypt_in_place(
    &self,
    nonce: &Nonce96,
    aad: &[u8],
    buffer: &mut [u8],
    tag: &ChaCha20Poly1305Tag,
  ) -> Result<(), OpenError> {
    <Self as Aead>::decrypt_in_place(self, nonce, aad, buffer, tag)
  }

  /// Encrypt `plaintext` into `out` as `ciphertext || tag`.
  #[inline]
  pub fn encrypt(&self, nonce: &Nonce96, aad: &[u8], plaintext: &[u8], out: &mut [u8]) -> Result<(), SealError> {
    <Self as Aead>::encrypt(self, nonce, aad, plaintext, out)
  }

  /// Decrypt a combined `ciphertext || tag` into `out`.
  #[inline]
  pub fn decrypt(
    &self,
    nonce: &Nonce96,
    aad: &[u8],
    ciphertext_and_tag: &[u8],
    out: &mut [u8],
  ) -> Result<(), OpenError> {
    <Self as Aead>::decrypt(self, nonce, aad, ciphertext_and_tag, out)
  }

  fn compute_tag(&self, nonce: &Nonce96, aad: &[u8], ciphertext: &[u8]) -> Result<[u8; TAG_SIZE], LengthOverflow> {
    let mut poly_key = chacha20::poly1305_key_gen(self.key.as_bytes(), nonce.as_bytes());
    let tag = poly1305::authenticate_aead(AeadPrimitive::ChaCha20Poly1305, aad, ciphertext, &poly_key);
    ct::zeroize(&mut poly_key);
    tag
  }

  #[cfg(any(target_arch = "x86_64", all(target_arch = "powerpc64", target_endian = "little")))]
  fn encrypt_empty_text_fast(&self, nonce: &Nonce96, aad: &[u8]) -> Option<Result<ChaCha20Poly1305Tag, SealError>> {
    if aad.len() > SMALL_AAD_FAST_MAX {
      return None;
    }

    let mut poly_key = chacha20::poly1305_key_gen(self.key.as_bytes(), nonce.as_bytes());
    let tag = poly1305::authenticate_aead_empty_text_portable(aad, &poly_key);
    ct::zeroize(&mut poly_key);
    Some(Ok(ChaCha20Poly1305Tag::from_bytes(tag)))
  }

  #[cfg(any(target_arch = "x86_64", all(target_arch = "powerpc64", target_endian = "little")))]
  fn decrypt_empty_text_fast(
    &self,
    nonce: &Nonce96,
    aad: &[u8],
    tag: &ChaCha20Poly1305Tag,
  ) -> Option<Result<(), OpenError>> {
    if aad.len() > SMALL_AAD_FAST_MAX {
      return None;
    }

    let mut poly_key = chacha20::poly1305_key_gen(self.key.as_bytes(), nonce.as_bytes());
    let expected = poly1305::authenticate_aead_empty_text_portable(aad, &poly_key);
    ct::zeroize(&mut poly_key);
    if !ct::fixed_eq(&expected, tag.as_bytes()) {
      return Some(Err(OpenError::verification()));
    }
    Some(Ok(()))
  }

  #[cfg(all(target_arch = "powerpc64", target_endian = "little"))]
  fn encrypt_short_text_power_fast(
    &self,
    nonce: &Nonce96,
    aad: &[u8],
    buffer: &mut [u8],
  ) -> Option<Result<ChaCha20Poly1305Tag, SealError>> {
    if buffer.is_empty() || buffer.len() > POWER_SHORT_FAST_MAX || aad.len() > SMALL_AAD_FAST_MAX {
      return None;
    }

    let mut poly_key = chacha20::poly1305_key_gen(self.key.as_bytes(), nonce.as_bytes());
    chacha20::xor_keystream_first_block_portable(self.key.as_bytes(), 1, nonce.as_bytes(), buffer);
    let tag = poly1305::authenticate_aead_short_text_portable(aad, buffer, &poly_key);
    ct::zeroize(&mut poly_key);
    Some(Ok(ChaCha20Poly1305Tag::from_bytes(tag)))
  }

  #[cfg(all(target_arch = "powerpc64", target_endian = "little"))]
  fn decrypt_short_text_power_fast(
    &self,
    nonce: &Nonce96,
    aad: &[u8],
    buffer: &mut [u8],
    tag: &ChaCha20Poly1305Tag,
  ) -> Option<Result<(), OpenError>> {
    if buffer.is_empty() || buffer.len() > POWER_SHORT_FAST_MAX || aad.len() > SMALL_AAD_FAST_MAX {
      return None;
    }

    let mut poly_key = chacha20::poly1305_key_gen(self.key.as_bytes(), nonce.as_bytes());
    let expected = poly1305::authenticate_aead_short_text_portable(aad, buffer, &poly_key);
    ct::zeroize(&mut poly_key);
    if !ct::fixed_eq(&expected, tag.as_bytes()) {
      ct::zeroize(buffer);
      return Some(Err(OpenError::verification()));
    }

    chacha20::xor_keystream_first_block_portable(self.key.as_bytes(), 1, nonce.as_bytes(), buffer);
    Some(Ok(()))
  }

  #[cfg(all(
    target_arch = "x86_64",
    target_os = "linux",
    any(test, feature = "diag", all(not(debug_assertions), not(feature = "portable-only")))
  ))]
  #[inline]
  fn x86_64_asm_caps_available(caps: crate::platform::Caps) -> bool {
    use crate::platform::caps::x86;

    caps.has(x86::AVX2.union(x86::BMI2))
  }

  #[cfg(all(
    target_arch = "x86_64",
    target_os = "linux",
    any(test, all(not(debug_assertions), not(feature = "portable-only")))
  ))]
  #[inline]
  fn x86_64_asm_recommended(caps: crate::platform::Caps, plaintext_len: usize) -> bool {
    use crate::platform::caps::x86;

    if plaintext_len == 0 || !Self::x86_64_asm_caps_available(caps) {
      return false;
    }

    // Measured on the 2026-07-01 x86_64 Linux bench run:
    // - Sapphire Rapids regresses beyond 256 bytes.
    // - AMD Zen5 regresses beyond 1024 bytes.
    // - The remaining measured AVX2+BMI2 lanes, Zen4 and Ice Lake, benefit from the integrated pass
    //   shape across the sampled non-empty sizes.
    if caps.has(x86::INTEL_SAPPHIRE_RAPIDS) {
      plaintext_len <= X86_64_ASM_SPR_MAX
    } else if caps.has(x86::AMD_ZEN5) {
      plaintext_len <= X86_64_ASM_ZEN5_MAX
    } else {
      true
    }
  }

  #[cfg(all(
    target_arch = "x86_64",
    target_os = "linux",
    any(test, all(not(debug_assertions), not(feature = "portable-only")))
  ))]
  #[inline]
  fn x86_64_open_asm_recommended(caps: crate::platform::Caps, ciphertext_len: usize) -> bool {
    use crate::platform::caps::x86;

    if ciphertext_len == 0 || !Self::x86_64_asm_caps_available(caps) {
      return false;
    }

    // Measured on the 2026-07-01 decrypt bench:
    // - all sampled x86_64 CPUs lose 1..=256 bytes on the generic split open path.
    // - AMD Zen4 loses through the full sampled matrix; the repo only has a Zen5-specific AMD
    //   discriminator, so AMD without Zen5 takes integrated open for every non-empty size.
    if caps.has(x86::AMD) && !caps.has(x86::AMD_ZEN5) {
      true
    } else {
      ciphertext_len <= X86_64_OPEN_ASM_SHORT_MAX
    }
  }

  #[cfg(all(feature = "diag", target_arch = "x86_64", target_os = "linux"))]
  fn encrypt_in_place_asm_x86_64_forced(
    &self,
    nonce: &Nonce96,
    aad: &[u8],
    buffer: &mut [u8],
  ) -> Option<Result<ChaCha20Poly1305Tag, SealError>> {
    if buffer.is_empty() {
      return None;
    }

    #[cfg(feature = "std")]
    let caps = crate::platform::caps();
    #[cfg(not(feature = "std"))]
    let caps = crate::platform::caps_static();

    if !Self::x86_64_asm_caps_available(caps) {
      return None;
    }

    let tag = x86_64_asm::seal_in_place(self.key.as_bytes(), nonce.as_bytes(), aad, buffer);
    Some(Ok(ChaCha20Poly1305Tag::from_bytes(tag)))
  }

  #[cfg(all(feature = "diag", target_arch = "x86_64", target_os = "linux"))]
  fn decrypt_in_place_asm_x86_64_forced(
    &self,
    nonce: &Nonce96,
    aad: &[u8],
    buffer: &mut [u8],
    tag: &ChaCha20Poly1305Tag,
  ) -> Option<Result<(), OpenError>> {
    if buffer.is_empty() {
      return None;
    }

    #[cfg(feature = "std")]
    let caps = crate::platform::caps();
    #[cfg(not(feature = "std"))]
    let caps = crate::platform::caps_static();

    if !Self::x86_64_asm_caps_available(caps) {
      return None;
    }

    let expected = x86_64_asm::open_in_place(self.key.as_bytes(), nonce.as_bytes(), aad, buffer);
    if !ct::fixed_eq(&expected, tag.as_bytes()) {
      ct::zeroize(buffer);
      return Some(Err(OpenError::verification()));
    }
    Some(Ok(()))
  }

  #[cfg(all(
    target_arch = "x86_64",
    target_os = "linux",
    not(debug_assertions),
    not(feature = "portable-only")
  ))]
  fn encrypt_in_place_asm_x86_64(
    &self,
    nonce: &Nonce96,
    aad: &[u8],
    buffer: &mut [u8],
  ) -> Option<Result<ChaCha20Poly1305Tag, SealError>> {
    #[cfg(feature = "std")]
    let caps = crate::platform::caps();
    #[cfg(not(feature = "std"))]
    let caps = crate::platform::caps_static();

    if !Self::x86_64_asm_recommended(caps, buffer.len()) {
      return None;
    }

    let tag = x86_64_asm::seal_in_place(self.key.as_bytes(), nonce.as_bytes(), aad, buffer);
    Some(Ok(ChaCha20Poly1305Tag::from_bytes(tag)))
  }

  #[cfg(all(
    target_arch = "x86_64",
    target_os = "linux",
    not(debug_assertions),
    not(feature = "portable-only")
  ))]
  fn decrypt_in_place_asm_x86_64(
    &self,
    nonce: &Nonce96,
    aad: &[u8],
    buffer: &mut [u8],
    tag: &ChaCha20Poly1305Tag,
  ) -> Option<Result<(), OpenError>> {
    #[cfg(feature = "std")]
    let caps = crate::platform::caps();
    #[cfg(not(feature = "std"))]
    let caps = crate::platform::caps_static();

    if !Self::x86_64_open_asm_recommended(caps, buffer.len()) {
      return None;
    }

    let expected = x86_64_asm::open_in_place(self.key.as_bytes(), nonce.as_bytes(), aad, buffer);
    if !ct::fixed_eq(&expected, tag.as_bytes()) {
      ct::zeroize(buffer);
      return Some(Err(OpenError::verification()));
    }
    Some(Ok(()))
  }

  #[cfg(all(
    target_arch = "aarch64",
    any(target_os = "linux", target_os = "macos"),
    not(debug_assertions),
    not(feature = "portable-only")
  ))]
  fn encrypt_in_place_asm_aarch64(
    &self,
    nonce: &Nonce96,
    aad: &[u8],
    buffer: &mut [u8],
  ) -> Option<Result<ChaCha20Poly1305Tag, SealError>> {
    if buffer.is_empty() {
      return None;
    }

    let tag = aarch64_asm::seal_in_place(self.key.as_bytes(), nonce.as_bytes(), aad, buffer);
    Some(Ok(ChaCha20Poly1305Tag::from_bytes(tag)))
  }

  #[cfg(all(
    target_arch = "aarch64",
    any(target_os = "linux", target_os = "macos"),
    not(debug_assertions),
    not(feature = "portable-only")
  ))]
  fn decrypt_in_place_asm_aarch64(
    &self,
    nonce: &Nonce96,
    aad: &[u8],
    buffer: &mut [u8],
    tag: &ChaCha20Poly1305Tag,
  ) -> Option<Result<(), OpenError>> {
    if buffer.is_empty() {
      return None;
    }

    let expected = aarch64_asm::open_in_place(self.key.as_bytes(), nonce.as_bytes(), aad, buffer);
    if !ct::fixed_eq(&expected, tag.as_bytes()) {
      ct::zeroize(buffer);
      return Some(Err(OpenError::verification()));
    }
    Some(Ok(()))
  }

  #[cfg(target_arch = "aarch64")]
  fn encrypt_in_place_interleaved_aarch64(
    &self,
    nonce: &Nonce96,
    aad: &[u8],
    buffer: &mut [u8],
  ) -> Option<Result<ChaCha20Poly1305Tag, SealError>> {
    use crate::platform::caps::aarch64;

    if buffer.len() < AARCH64_INTERLEAVED_MIN {
      return None;
    }

    #[cfg(feature = "std")]
    let caps = crate::platform::caps();
    #[cfg(not(feature = "std"))]
    let caps = crate::platform::caps_static();

    if !caps.has(aarch64::NEON) {
      return None;
    }

    let lengths = match super::AeadByteLengths::try_new(aad.len(), buffer.len()) {
      Ok(lengths) => lengths,
      Err(_) => return Some(Err(SealError::too_large())),
    };

    let mut poly_key = chacha20::poly1305_key_gen(self.key.as_bytes(), nonce.as_bytes());
    let mut authenticator = poly1305::aarch64_neon::AeadPar4::new(&poly_key);
    authenticator.update_padded_segment(aad);

    let mut counter = 1u32;
    let mut chunks = buffer.chunks_exact_mut(AARCH64_INTERLEAVED_CHUNK);
    for chunk in &mut chunks {
      chacha20::xor_keystream_aarch64_neon(self.key.as_bytes(), counter, nonce.as_bytes(), chunk);
      authenticator.update_padded_segment(chunk);
      counter = counter.wrapping_add((AARCH64_INTERLEAVED_CHUNK / chacha20::BLOCK_SIZE) as u32);
    }

    let remainder = chunks.into_remainder();
    if !remainder.is_empty() {
      chacha20::xor_keystream_aarch64_neon(self.key.as_bytes(), counter, nonce.as_bytes(), remainder);
      authenticator.update_padded_segment(remainder);
    }

    let tag = ChaCha20Poly1305Tag::from_bytes(authenticator.finalize(lengths));
    ct::zeroize(&mut poly_key);
    Some(Ok(tag))
  }

  #[cfg(target_arch = "aarch64")]
  fn decrypt_in_place_interleaved_aarch64(
    &self,
    nonce: &Nonce96,
    aad: &[u8],
    buffer: &mut [u8],
    tag: &ChaCha20Poly1305Tag,
  ) -> Option<Result<(), OpenError>> {
    use crate::platform::caps::aarch64;

    if buffer.len() < AARCH64_INTERLEAVED_MIN {
      return None;
    }

    #[cfg(feature = "std")]
    let caps = crate::platform::caps();
    #[cfg(not(feature = "std"))]
    let caps = crate::platform::caps_static();

    if !caps.has(aarch64::NEON) {
      return None;
    }

    let lengths = match super::AeadByteLengths::try_new(aad.len(), buffer.len()) {
      Ok(lengths) => lengths,
      Err(_) => return Some(Err(OpenError::too_large())),
    };

    let mut poly_key = chacha20::poly1305_key_gen(self.key.as_bytes(), nonce.as_bytes());
    let mut authenticator = poly1305::aarch64_neon::AeadPar4::new(&poly_key);
    authenticator.update_padded_segment(aad);

    let mut counter = 1u32;
    let mut chunks = buffer.chunks_exact_mut(AARCH64_INTERLEAVED_CHUNK);
    for chunk in &mut chunks {
      authenticator.update_padded_segment(chunk);
      chacha20::xor_keystream_aarch64_neon(self.key.as_bytes(), counter, nonce.as_bytes(), chunk);
      counter = counter.wrapping_add((AARCH64_INTERLEAVED_CHUNK / chacha20::BLOCK_SIZE) as u32);
    }

    let remainder = chunks.into_remainder();
    if !remainder.is_empty() {
      authenticator.update_padded_segment(remainder);
      chacha20::xor_keystream_aarch64_neon(self.key.as_bytes(), counter, nonce.as_bytes(), remainder);
    }

    let expected = authenticator.finalize(lengths);
    ct::zeroize(&mut poly_key);
    if !ct::fixed_eq(&expected, tag.as_bytes()) {
      ct::zeroize(buffer);
      return Some(Err(OpenError::verification()));
    }
    Some(Ok(()))
  }

  #[cfg(all(
    feature = "diag",
    target_arch = "aarch64",
    any(target_os = "linux", target_os = "macos")
  ))]
  fn encrypt_in_place_owned_par4_aarch64_diag(
    &self,
    nonce: &Nonce96,
    aad: &[u8],
    buffer: &mut [u8],
  ) -> Result<ChaCha20Poly1305Tag, SealError> {
    let lengths = super::AeadByteLengths::try_new(aad.len(), buffer.len()).map_err(|_| SealError::too_large())?;

    let mut poly_key = chacha20::poly1305_key_gen(self.key.as_bytes(), nonce.as_bytes());
    let mut authenticator = poly1305::aarch64_asm::AeadPar4Asm::new(&poly_key);
    authenticator.update_padded_segment(aad);

    let bulk_len = buffer
      .len()
      .strict_div(AARCH64_OWNED_PAR4_DIAG_CHUNK)
      .strict_mul(AARCH64_OWNED_PAR4_DIAG_CHUNK);
    let (bulk, remainder) = buffer.split_at_mut(bulk_len);
    let mut counter = 1u32;
    if !bulk.is_empty() {
      chacha20::diag_chacha20_xor_keystream_aarch64_owned_asm(self.key.as_bytes(), counter, nonce.as_bytes(), bulk);
      authenticator.update_padded_segment(bulk);
      counter = counter.wrapping_add(bulk_len.strict_div(chacha20::BLOCK_SIZE) as u32);
    }

    if !remainder.is_empty() {
      chacha20::xor_keystream_aarch64_neon(self.key.as_bytes(), counter, nonce.as_bytes(), remainder);
      authenticator.update_padded_segment(remainder);
    }

    let tag = ChaCha20Poly1305Tag::from_bytes(authenticator.finalize(lengths));
    ct::zeroize(&mut poly_key);
    Ok(tag)
  }

  fn encrypt_in_place_owned_unchecked(
    &self,
    nonce: &Nonce96,
    aad: &[u8],
    buffer: &mut [u8],
  ) -> Result<ChaCha20Poly1305Tag, SealError> {
    #[cfg(target_arch = "aarch64")]
    if let Some(result) = self.encrypt_in_place_interleaved_aarch64(nonce, aad, buffer) {
      return result;
    }

    chacha20::xor_keystream(
      AeadPrimitive::ChaCha20Poly1305,
      self.key.as_bytes(),
      1,
      nonce.as_bytes(),
      buffer,
    )
    .map_err(|_| SealError::too_large())?;

    let mut poly_key = chacha20::poly1305_key_gen(self.key.as_bytes(), nonce.as_bytes());
    let tag = ChaCha20Poly1305Tag::from_bytes(
      poly1305::authenticate_aead(AeadPrimitive::ChaCha20Poly1305, aad, buffer, &poly_key)
        .map_err(|_| SealError::too_large())?,
    );
    ct::zeroize(&mut poly_key);
    Ok(tag)
  }

  fn decrypt_in_place_owned_unchecked(
    &self,
    nonce: &Nonce96,
    aad: &[u8],
    buffer: &mut [u8],
    tag: &ChaCha20Poly1305Tag,
  ) -> Result<(), OpenError> {
    #[cfg(target_arch = "aarch64")]
    if let Some(result) = self.decrypt_in_place_interleaved_aarch64(nonce, aad, buffer, tag) {
      return result;
    }

    let expected = self
      .compute_tag(nonce, aad, buffer)
      .map_err(|_| OpenError::too_large())?;
    if !ct::fixed_eq(&expected, tag.as_bytes()) {
      ct::zeroize(buffer);
      return Err(OpenError::verification());
    }

    chacha20::xor_keystream(
      AeadPrimitive::ChaCha20Poly1305,
      self.key.as_bytes(),
      1,
      nonce.as_bytes(),
      buffer,
    )
    .map_err(|_| OpenError::too_large())?;
    Ok(())
  }
}

#[cfg(feature = "diag")]
pub fn diag_chacha20poly1305_encrypt_in_place_owned(
  cipher: &ChaCha20Poly1305,
  nonce: &Nonce96,
  aad: &[u8],
  buffer: &mut [u8],
) -> Result<ChaCha20Poly1305Tag, SealError> {
  super::seal_bounded_length_as_u64(buffer.len(), MAX_PLAINTEXT_LEN)?;
  cipher.encrypt_in_place_owned_unchecked(nonce, aad, buffer)
}

#[cfg(all(
  feature = "diag",
  target_arch = "aarch64",
  any(target_os = "linux", target_os = "macos")
))]
pub fn diag_chacha20poly1305_encrypt_in_place_owned_par4_aarch64(
  cipher: &ChaCha20Poly1305,
  nonce: &Nonce96,
  aad: &[u8],
  buffer: &mut [u8],
) -> Result<ChaCha20Poly1305Tag, SealError> {
  super::seal_bounded_length_as_u64(buffer.len(), MAX_PLAINTEXT_LEN)?;
  cipher.encrypt_in_place_owned_par4_aarch64_diag(nonce, aad, buffer)
}

#[cfg(all(feature = "diag", target_arch = "x86_64", target_os = "linux"))]
pub fn diag_chacha20poly1305_encrypt_in_place_x86_64_asm(
  cipher: &ChaCha20Poly1305,
  nonce: &Nonce96,
  aad: &[u8],
  buffer: &mut [u8],
) -> Option<Result<ChaCha20Poly1305Tag, SealError>> {
  if let Err(err) = super::seal_bounded_length_as_u64(buffer.len(), MAX_PLAINTEXT_LEN) {
    return Some(Err(err));
  }

  cipher.encrypt_in_place_asm_x86_64_forced(nonce, aad, buffer)
}

#[cfg(all(feature = "diag", target_arch = "x86_64", target_os = "linux"))]
pub fn diag_chacha20poly1305_decrypt_in_place_x86_64_asm(
  cipher: &ChaCha20Poly1305,
  nonce: &Nonce96,
  aad: &[u8],
  buffer: &mut [u8],
  tag: &ChaCha20Poly1305Tag,
) -> Option<Result<(), OpenError>> {
  if let Err(err) = super::open_bounded_length_as_u64(buffer.len(), MAX_PLAINTEXT_LEN) {
    return Some(Err(err));
  }

  cipher.decrypt_in_place_asm_x86_64_forced(nonce, aad, buffer, tag)
}

#[cfg(feature = "diag")]
pub fn diag_chacha20poly1305_decrypt_in_place_owned(
  cipher: &ChaCha20Poly1305,
  nonce: &Nonce96,
  aad: &[u8],
  buffer: &mut [u8],
  tag: &ChaCha20Poly1305Tag,
) -> Result<(), OpenError> {
  super::open_bounded_length_as_u64(buffer.len(), MAX_PLAINTEXT_LEN)?;
  cipher.decrypt_in_place_owned_unchecked(nonce, aad, buffer, tag)
}

impl Aead for ChaCha20Poly1305 {
  const KEY_SIZE: usize = KEY_SIZE;
  const NONCE_SIZE: usize = NONCE_SIZE;
  const TAG_SIZE: usize = TAG_SIZE;

  type Key = ChaCha20Poly1305Key;
  type Nonce = Nonce96;
  type Tag = ChaCha20Poly1305Tag;

  fn new(key: &Self::Key) -> Self {
    Self {
      key: key.duplicate_secret(),
    }
  }

  fn tag_from_slice(bytes: &[u8]) -> Result<Self::Tag, AeadBufferError> {
    if bytes.len() != TAG_SIZE {
      return Err(AeadBufferError::new());
    }

    let mut tag = [0u8; TAG_SIZE];
    tag.copy_from_slice(bytes);
    Ok(ChaCha20Poly1305Tag::from_bytes(tag))
  }

  fn encrypt_in_place(&self, nonce: &Self::Nonce, aad: &[u8], buffer: &mut [u8]) -> Result<Self::Tag, SealError> {
    super::seal_bounded_length_as_u64(buffer.len(), MAX_PLAINTEXT_LEN)?;

    #[cfg(any(target_arch = "x86_64", all(target_arch = "powerpc64", target_endian = "little")))]
    if buffer.is_empty()
      && let Some(result) = self.encrypt_empty_text_fast(nonce, aad)
    {
      return result;
    }

    #[cfg(all(target_arch = "powerpc64", target_endian = "little"))]
    if let Some(result) = self.encrypt_short_text_power_fast(nonce, aad, buffer) {
      return result;
    }

    #[cfg(all(
      target_arch = "aarch64",
      any(target_os = "linux", target_os = "macos"),
      not(debug_assertions),
      not(feature = "portable-only")
    ))]
    if let Some(result) = self.encrypt_in_place_asm_aarch64(nonce, aad, buffer) {
      return result;
    }

    #[cfg(all(
      target_arch = "x86_64",
      target_os = "linux",
      not(debug_assertions),
      not(feature = "portable-only")
    ))]
    if let Some(result) = self.encrypt_in_place_asm_x86_64(nonce, aad, buffer) {
      return result;
    }

    self.encrypt_in_place_owned_unchecked(nonce, aad, buffer)
  }

  fn decrypt_in_place(
    &self,
    nonce: &Self::Nonce,
    aad: &[u8],
    buffer: &mut [u8],
    tag: &Self::Tag,
  ) -> Result<(), OpenError> {
    super::open_bounded_length_as_u64(buffer.len(), MAX_PLAINTEXT_LEN)?;

    #[cfg(any(target_arch = "x86_64", all(target_arch = "powerpc64", target_endian = "little")))]
    if buffer.is_empty()
      && let Some(result) = self.decrypt_empty_text_fast(nonce, aad, tag)
    {
      return result;
    }

    #[cfg(all(target_arch = "powerpc64", target_endian = "little"))]
    if let Some(result) = self.decrypt_short_text_power_fast(nonce, aad, buffer, tag) {
      return result;
    }

    #[cfg(all(
      target_arch = "x86_64",
      target_os = "linux",
      not(debug_assertions),
      not(feature = "portable-only")
    ))]
    if let Some(result) = self.decrypt_in_place_asm_x86_64(nonce, aad, buffer, tag) {
      return result;
    }

    #[cfg(all(
      target_arch = "aarch64",
      any(target_os = "linux", target_os = "macos"),
      not(debug_assertions),
      not(feature = "portable-only")
    ))]
    if let Some(result) = self.decrypt_in_place_asm_aarch64(nonce, aad, buffer, tag) {
      return result;
    }

    self.decrypt_in_place_owned_unchecked(nonce, aad, buffer, tag)
  }
}

#[cfg(test)]
mod tests {
  #[cfg(any(target_arch = "x86_64", all(target_arch = "powerpc64", target_endian = "little")))]
  use alloc::vec::Vec;

  use super::*;

  #[test]
  fn round_trip() {
    let key = ChaCha20Poly1305Key::from_bytes([0x42u8; 32]);
    let nonce = Nonce96::from_bytes([0x07u8; 12]);
    let cipher = ChaCha20Poly1305::new(&key);

    let mut buf = *b"hello chacha";
    let tag = cipher.encrypt_in_place(&nonce, b"aad", &mut buf).unwrap();
    cipher.decrypt_in_place(&nonce, b"aad", &mut buf, &tag).unwrap();
    assert_eq!(&buf, b"hello chacha");
  }

  #[test]
  fn wrong_nonce_rejected() {
    let key = ChaCha20Poly1305Key::from_bytes([0x42u8; 32]);
    let nonce = Nonce96::from_bytes([0x07u8; 12]);
    let cipher = ChaCha20Poly1305::new(&key);

    let mut buf = *b"nonce test";
    let tag = cipher.encrypt_in_place(&nonce, b"aad", &mut buf).unwrap();

    let wrong_nonce = Nonce96::from_bytes([0x08u8; 12]);
    let result = cipher.decrypt_in_place(&wrong_nonce, b"aad", &mut buf, &tag);
    assert!(result.is_err());
  }

  #[test]
  fn buffer_zeroed_on_auth_failure() {
    let key = ChaCha20Poly1305Key::from_bytes([0x42u8; 32]);
    let nonce = Nonce96::from_bytes([0x07u8; 12]);
    let cipher = ChaCha20Poly1305::new(&key);

    let mut buf = *b"zero me on failure";
    let tag = cipher.encrypt_in_place(&nonce, b"aad", &mut buf).unwrap();

    let mut bad_tag = tag.to_bytes();
    bad_tag[0] ^= 0xFF;
    let bad_tag = ChaCha20Poly1305Tag::from_bytes(bad_tag);

    let result = cipher.decrypt_in_place(&nonce, b"aad", &mut buf, &bad_tag);
    assert!(result.is_err());
    assert!(buf.iter().all(|&b| b == 0), "buffer not zeroed on auth failure");
  }

  #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
  #[test]
  fn x86_64_asm_policy_matches_measured_thresholds() {
    use crate::platform::{Caps, caps::x86};

    let avx2_bmi2 = x86::AVX2 | x86::BMI2;
    let amd_zen4 = avx2_bmi2 | x86::AMD;
    let zen5 = avx2_bmi2 | x86::AMD | x86::AMD_ZEN5;
    let spr = avx2_bmi2 | x86::INTEL_SAPPHIRE_RAPIDS;

    assert!(!ChaCha20Poly1305::x86_64_asm_recommended(Caps::NONE, 1));
    assert!(!ChaCha20Poly1305::x86_64_asm_recommended(avx2_bmi2, 0));

    assert!(ChaCha20Poly1305::x86_64_asm_recommended(avx2_bmi2, 16_384));

    assert!(ChaCha20Poly1305::x86_64_asm_recommended(zen5, X86_64_ASM_ZEN5_MAX));
    assert!(!ChaCha20Poly1305::x86_64_asm_recommended(zen5, X86_64_ASM_ZEN5_MAX + 1));

    assert!(ChaCha20Poly1305::x86_64_asm_recommended(spr, X86_64_ASM_SPR_MAX));
    assert!(!ChaCha20Poly1305::x86_64_asm_recommended(spr, X86_64_ASM_SPR_MAX + 1));

    assert!(!ChaCha20Poly1305::x86_64_open_asm_recommended(Caps::NONE, 1));
    assert!(!ChaCha20Poly1305::x86_64_open_asm_recommended(avx2_bmi2, 0));

    assert!(ChaCha20Poly1305::x86_64_open_asm_recommended(
      avx2_bmi2,
      X86_64_OPEN_ASM_SHORT_MAX
    ));
    assert!(!ChaCha20Poly1305::x86_64_open_asm_recommended(
      avx2_bmi2,
      X86_64_OPEN_ASM_SHORT_MAX + 1
    ));

    assert!(ChaCha20Poly1305::x86_64_open_asm_recommended(amd_zen4, 16_384));
    assert!(ChaCha20Poly1305::x86_64_open_asm_recommended(
      zen5,
      X86_64_OPEN_ASM_SHORT_MAX
    ));
    assert!(!ChaCha20Poly1305::x86_64_open_asm_recommended(
      zen5,
      X86_64_OPEN_ASM_SHORT_MAX + 1
    ));
    assert!(ChaCha20Poly1305::x86_64_open_asm_recommended(
      spr,
      X86_64_OPEN_ASM_SHORT_MAX
    ));
    assert!(!ChaCha20Poly1305::x86_64_open_asm_recommended(
      spr,
      X86_64_OPEN_ASM_SHORT_MAX + 1
    ));
  }

  #[cfg(any(target_arch = "x86_64", all(target_arch = "powerpc64", target_endian = "little")))]
  #[test]
  fn empty_text_fast_decrypt_matches_owned_path() {
    let key = ChaCha20Poly1305Key::from_bytes([0x42; KEY_SIZE]);
    let nonce = Nonce96::from_bytes([0x24; NONCE_SIZE]);
    let cipher = ChaCha20Poly1305::new(&key);

    for aad_len in [0usize, 1, 14, 15, 16, 17, 31, 32, 33, 63, 64] {
      let aad = (0..aad_len)
        .map(|index| 0xa7u8.wrapping_add((index as u8).wrapping_mul(7)))
        .collect::<Vec<_>>();
      let mut ciphertext = Vec::new();

      let expected_tag = cipher
        .encrypt_in_place_owned_unchecked(&nonce, &aad, &mut ciphertext)
        .unwrap();
      let actual = cipher.decrypt_empty_text_fast(&nonce, &aad, &expected_tag);

      if aad_len > SMALL_AAD_FAST_MAX {
        assert!(
          actual.is_none(),
          "empty decrypt fast path applied outside its measured gate: aad_len={aad_len}"
        );
        continue;
      }

      actual
        .expect("empty decrypt fast path must apply inside its measured gate")
        .unwrap();

      let mut bad_tag = expected_tag.to_bytes();
      bad_tag[0] ^= 0x80;
      assert_eq!(
        cipher
          .decrypt_empty_text_fast(&nonce, &aad, &ChaCha20Poly1305Tag::from_bytes(bad_tag))
          .expect("empty decrypt fast path must apply inside its measured gate"),
        Err(OpenError::verification())
      );
    }
  }

  #[cfg(all(feature = "diag", target_arch = "x86_64", target_os = "linux"))]
  #[test]
  fn x86_64_open_asm_matches_owned_path() {
    if !ChaCha20Poly1305::x86_64_asm_caps_available(crate::platform::caps()) {
      return;
    }

    let key = ChaCha20Poly1305Key::from_bytes([0x42; KEY_SIZE]);
    let nonce = Nonce96::from_bytes([0x24; NONCE_SIZE]);
    let cipher = ChaCha20Poly1305::new(&key);

    for plaintext_len in [
      1usize, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 1024,
    ] {
      let plaintext = (0..plaintext_len)
        .map(|index| 0x51u8.wrapping_add((index as u8).wrapping_mul(13)))
        .collect::<Vec<_>>();

      for aad_len in [0usize, 1, 13, 14, 15, 16, 17, 31, 32, 63, 64] {
        let aad = (0..aad_len)
          .map(|index| 0xa7u8.wrapping_add((index as u8).wrapping_mul(7)))
          .collect::<Vec<_>>();

        let mut ciphertext = plaintext.clone();
        let tag = cipher
          .encrypt_in_place_owned_unchecked(&nonce, &aad, &mut ciphertext)
          .unwrap();
        let mut actual = ciphertext.clone();
        let actual_tag = x86_64_asm::open_in_place(key.as_bytes(), nonce.as_bytes(), &aad, &mut actual);

        assert_eq!(
          actual, plaintext,
          "x86 open asm plaintext mismatch plaintext_len={plaintext_len} aad_len={aad_len}"
        );
        assert_eq!(
          actual_tag,
          tag.to_bytes(),
          "x86 open asm tag mismatch plaintext_len={plaintext_len} aad_len={aad_len}"
        );
      }
    }
  }

  #[cfg(all(target_arch = "powerpc64", target_endian = "little"))]
  #[test]
  fn power_short_fast_encrypt_matches_owned_path() {
    let key = ChaCha20Poly1305Key::from_bytes([0x42; KEY_SIZE]);
    let nonce = Nonce96::from_bytes([0x24; NONCE_SIZE]);
    let cipher = ChaCha20Poly1305::new(&key);

    for plaintext_len in [0usize, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65] {
      let plaintext = (0..plaintext_len)
        .map(|index| 0x51u8.wrapping_add((index as u8).wrapping_mul(13)))
        .collect::<Vec<_>>();

      for aad_len in [0usize, 1, 14, 15, 16, 17, 31, 32, 33, 63, 64] {
        let aad = (0..aad_len)
          .map(|index| 0xa7u8.wrapping_add((index as u8).wrapping_mul(7)))
          .collect::<Vec<_>>();

        let mut actual = plaintext.clone();
        let actual_tag = cipher.encrypt_short_text_power_fast(&nonce, &aad, &mut actual);
        if plaintext_len == 0 || plaintext_len > POWER_SHORT_FAST_MAX || aad_len > SMALL_AAD_FAST_MAX {
          assert!(
            actual_tag.is_none(),
            "Power short fast path applied outside its measured gate: plaintext_len={plaintext_len} aad_len={aad_len}"
          );
          assert_eq!(actual, plaintext);
          continue;
        }

        let mut expected = plaintext.clone();
        let expected_tag = cipher
          .encrypt_in_place_owned_unchecked(&nonce, &aad, &mut expected)
          .unwrap();
        let actual_tag = actual_tag
          .expect("Power short fast path must apply inside its measured gate")
          .unwrap();

        assert_eq!(
          actual, expected,
          "Power short ciphertext mismatch plaintext_len={plaintext_len} aad_len={aad_len}"
        );
        assert_eq!(
          actual_tag, expected_tag,
          "Power short tag mismatch plaintext_len={plaintext_len} aad_len={aad_len}"
        );
      }
    }
  }

  #[cfg(all(target_arch = "powerpc64", target_endian = "little"))]
  #[test]
  fn power_short_fast_decrypt_matches_owned_path() {
    let key = ChaCha20Poly1305Key::from_bytes([0x42; KEY_SIZE]);
    let nonce = Nonce96::from_bytes([0x24; NONCE_SIZE]);
    let cipher = ChaCha20Poly1305::new(&key);

    for plaintext_len in [0usize, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65] {
      let plaintext = (0..plaintext_len)
        .map(|index| 0x51u8.wrapping_add((index as u8).wrapping_mul(13)))
        .collect::<Vec<_>>();

      for aad_len in [0usize, 1, 14, 15, 16, 17, 31, 32, 33, 63, 64] {
        let aad = (0..aad_len)
          .map(|index| 0xa7u8.wrapping_add((index as u8).wrapping_mul(7)))
          .collect::<Vec<_>>();

        let mut ciphertext = plaintext.clone();
        let tag = cipher
          .encrypt_in_place_owned_unchecked(&nonce, &aad, &mut ciphertext)
          .unwrap();
        let mut actual = ciphertext.clone();
        let actual_result = cipher.decrypt_short_text_power_fast(&nonce, &aad, &mut actual, &tag);

        if plaintext_len == 0 || plaintext_len > POWER_SHORT_FAST_MAX || aad_len > SMALL_AAD_FAST_MAX {
          assert!(
            actual_result.is_none(),
            "Power short decrypt fast path applied outside its measured gate: plaintext_len={plaintext_len} \
             aad_len={aad_len}"
          );
          assert_eq!(actual, ciphertext);
          continue;
        }

        actual_result
          .expect("Power short decrypt fast path must apply inside its measured gate")
          .unwrap();
        assert_eq!(
          actual, plaintext,
          "Power short plaintext mismatch plaintext_len={plaintext_len} aad_len={aad_len}"
        );

        let mut bad_tag = tag.to_bytes();
        bad_tag[0] ^= 0x80;
        let mut rejected = ciphertext.clone();
        assert_eq!(
          cipher
            .decrypt_short_text_power_fast(&nonce, &aad, &mut rejected, &ChaCha20Poly1305Tag::from_bytes(bad_tag),)
            .expect("Power short decrypt fast path must apply inside its measured gate"),
          Err(OpenError::verification())
        );
        assert!(
          rejected.iter().all(|&byte| byte == 0),
          "Power short decrypt fast path did not zeroize rejected buffer"
        );
      }
    }
  }

  #[test]
  fn wrong_aad_rejected() {
    let key = ChaCha20Poly1305Key::from_bytes([0x42u8; 32]);
    let nonce = Nonce96::from_bytes([0x07u8; 12]);
    let cipher = ChaCha20Poly1305::new(&key);

    let mut buf = *b"aad test";
    let tag = cipher.encrypt_in_place(&nonce, b"correct", &mut buf).unwrap();

    let result = cipher.decrypt_in_place(&nonce, b"wrong", &mut buf, &tag);
    assert!(result.is_err());
  }
}
