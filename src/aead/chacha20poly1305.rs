//! ChaCha20-Poly1305 public AEAD surface.

use core::fmt;

use super::{AeadBufferError, Nonce96, OpenError, SealError, poly1305, targets::AeadPrimitive};
use crate::backend::chacha20;
use crate::traits::{Aead, AeadGather, ct};

mod sealing_key;
pub use sealing_key::ChaCha20Poly1305SealingKey;

const KEY_SIZE: usize = chacha20::KEY_SIZE;
const TAG_SIZE: usize = 16;
const NONCE_SIZE: usize = Nonce96::LENGTH;
const MAX_PLAINTEXT_LEN: u64 = (u32::MAX as u64) * (chacha20::BLOCK_SIZE as u64);
#[cfg(any(
  all(target_arch = "x86_64", target_feature = "sse2"),
  all(target_arch = "powerpc64", not(feature = "portable-only"), target_endian = "little")
))]
const SMALL_AAD_FAST_MAX: usize = 63;
#[cfg(all(target_arch = "powerpc64", not(feature = "portable-only"), target_endian = "little"))]
const POWER_SHORT_FAST_MAX: usize = chacha20::BLOCK_SIZE;
#[cfg(target_arch = "aarch64")]
const AARCH64_INTERLEAVED_MIN: usize = 1024;
#[cfg(target_arch = "aarch64")]
const AARCH64_INTERLEAVED_BLOCKS: u32 = 16 * 1024;
#[cfg(target_arch = "aarch64")]
const AARCH64_INTERLEAVED_CHUNK: usize = (AARCH64_INTERLEAVED_BLOCKS as usize) * chacha20::BLOCK_SIZE;
#[cfg(all(
  target_arch = "x86_64",
  target_feature = "sse2",
  target_os = "linux",
  any(test, all(not(debug_assertions), not(feature = "portable-only")))
))]
const X86_64_ASM_ZEN5_MAX: usize = 1024;
#[cfg(all(
  target_arch = "x86_64",
  target_feature = "sse2",
  target_os = "linux",
  any(test, all(not(debug_assertions), not(feature = "portable-only")))
))]
const X86_64_ASM_SPR_MAX: usize = 256;
#[cfg(all(
  target_arch = "x86_64",
  target_feature = "sse2",
  target_os = "linux",
  any(test, all(not(debug_assertions), not(feature = "portable-only")))
))]
const X86_64_OPEN_ASM_SHORT_MAX: usize = 256;
/// Gathered messages up to this length seal their leading span through the
/// integrated assembly. Longer gathers keep the vector stream and separate
/// Poly1305 path: on Granite Rapids it was within a few percent of the
/// integrated kernel for 1 KiB records and faster for 16 KiB records.
#[cfg(all(
  target_arch = "x86_64",
  target_feature = "sse2",
  target_os = "linux",
  not(debug_assertions),
  not(miri),
  not(feature = "portable-only")
))]
const X86_64_GATHER_ASM_MAX: usize = 1024;

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
  target_feature = "sse2",
  target_os = "linux",
  all(any(test, not(debug_assertions)), not(feature = "portable-only"))
))]
#[path = "chacha20poly1305/x86_64_asm.rs"]
mod x86_64_asm;
#[cfg(all(target_arch = "x86_64", target_feature = "sse2", not(feature = "portable-only")))]
#[path = "chacha20poly1305/x86_64_avx512.rs"]
mod x86_64_avx512;

define_aead_key_type!(ChaCha20Poly1305Key, KEY_SIZE, "ChaCha20-Poly1305 secret key bytes.");

define_aead_tag_type!(
  ChaCha20Poly1305Tag,
  TAG_SIZE,
  "ChaCha20-Poly1305 authentication tag bytes."
);

/// ChaCha20-Poly1305 with 96-bit nonces.
///
/// Use [`ChaCha20Poly1305SealingKey`] to seal messages and this type to open them.
/// Each nonce must be unique under its key. RFC 8439 rules out random generation.
/// Protocols that derive unique nonces can use [`AeadWithNonce`](super::expert::AeadWithNonce).
/// See [RFC 8439, section 2.6](https://www.rfc-editor.org/rfc/rfc8439.html#section-2.6).
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

  fn compute_tag(&self, nonce: &Nonce96, aad: &[u8], ciphertext: &[u8]) -> [u8; TAG_SIZE] {
    let mut poly_key = chacha20::poly1305_key_gen(self.key.as_bytes(), nonce.as_bytes());
    let tag = poly1305::authenticate_aead(AeadPrimitive::ChaCha20Poly1305, aad, ciphertext, &poly_key);
    ct::zeroize(&mut poly_key);
    tag
  }

  #[cfg(any(
    all(target_arch = "x86_64", target_feature = "sse2"),
    all(target_arch = "powerpc64", not(feature = "portable-only"), target_endian = "little")
  ))]
  fn encrypt_empty_text_fast(&self, nonce: &Nonce96, aad: &[u8]) -> Option<Result<ChaCha20Poly1305Tag, SealError>> {
    if aad.len() > SMALL_AAD_FAST_MAX {
      return None;
    }

    let mut poly_key = chacha20::poly1305_key_gen(self.key.as_bytes(), nonce.as_bytes());
    let tag = poly1305::authenticate_aead_empty_text_portable(aad, &poly_key);
    ct::zeroize(&mut poly_key);
    Some(Ok(ChaCha20Poly1305Tag::from_bytes(tag)))
  }

  #[cfg(any(
    all(target_arch = "x86_64", target_feature = "sse2"),
    all(target_arch = "powerpc64", not(feature = "portable-only"), target_endian = "little")
  ))]
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
    if !ct::fixed_eq(&expected, tag.as_bytes()).declassify() {
      return Some(Err(OpenError::verification()));
    }
    Some(Ok(()))
  }

  #[cfg(all(target_arch = "powerpc64", not(feature = "portable-only"), target_endian = "little"))]
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

  #[cfg(all(target_arch = "powerpc64", not(feature = "portable-only"), target_endian = "little"))]
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
    if !ct::fixed_eq(&expected, tag.as_bytes()).declassify() {
      ct::zeroize(buffer);
      return Some(Err(OpenError::verification()));
    }

    chacha20::xor_keystream_first_block_portable(self.key.as_bytes(), 1, nonce.as_bytes(), buffer);
    Some(Ok(()))
  }

  /// Whether the owned AVX-512 kernels may run: AVX512F, VL, BW, and DQ.
  #[cfg(all(target_arch = "x86_64", target_feature = "sse2", not(feature = "portable-only")))]
  #[inline]
  fn x86_64_avx512_available() -> bool {
    #[cfg(feature = "std")]
    let caps = crate::platform::caps();
    #[cfg(not(feature = "std"))]
    let caps = crate::platform::caps_static();
    caps.has(crate::platform::caps::x86::AVX512_READY)
  }

  /// # Safety
  ///
  /// The CPU must support AVX2 and BMI2. The total length must exceed 320 bytes and fit the ChaCha20 counter.
  #[cfg(all(
    target_arch = "x86_64",
    target_feature = "sse2",
    target_os = "linux",
    not(feature = "portable-only"),
    any(test, not(debug_assertions))
  ))]
  unsafe fn open_segments_avx2(
    &self,
    nonce: &Nonce96,
    aad: &[u8],
    segments: &mut [&mut [u8]],
    tag: &ChaCha20Poly1305Tag,
  ) -> Option<Result<(), OpenError>> {
    let expected = super::segments::spans(segments, |spans| {
      // SAFETY: The caller establishes the CPU and length requirements. The spans borrow disjoint segments.
      unsafe { x86_64_asm::open_scatter(self.key.as_bytes(), nonce.as_bytes(), aad, spans) }
    })?;
    Some(super::segments::verdict(
      segments,
      ct::fixed_eq(&expected, tag.as_bytes()).declassify(),
    ))
  }

  #[cfg(all(
    target_arch = "x86_64",
    target_feature = "sse2",
    target_os = "linux",
    any(test, all(not(debug_assertions), not(feature = "portable-only")))
  ))]
  #[inline]
  fn x86_64_asm_caps_available(caps: crate::platform::Caps) -> bool {
    use crate::platform::caps::x86;

    caps.has(x86::AVX2.union(x86::BMI2))
  }

  #[cfg(all(
    target_arch = "x86_64",
    target_feature = "sse2",
    target_os = "linux",
    any(test, all(not(debug_assertions), not(feature = "portable-only")))
  ))]
  #[inline]
  fn x86_64_asm_recommended(caps: crate::platform::Caps, plaintext_len: usize) -> bool {
    use crate::platform::caps::x86;

    if plaintext_len == 0 || !Self::x86_64_asm_caps_available(caps) {
      return false;
    }

    // The manually maintained x86 policy bounds the integrated assembly path
    // on the two microarchitecture classes with explicit crossover limits.
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
    target_feature = "sse2",
    target_os = "linux",
    any(test, all(not(debug_assertions), not(feature = "portable-only")))
  ))]
  #[inline]
  fn x86_64_open_asm_recommended(caps: crate::platform::Caps, ciphertext_len: usize) -> bool {
    use crate::platform::caps::x86;

    if ciphertext_len == 0 || !Self::x86_64_asm_caps_available(caps) {
      return false;
    }

    // AMD without the Zen5 discriminator keeps the integrated open path for
    // every non-empty size; other x86 classes use the configured short-input
    // boundary.
    if caps.has(x86::AMD) && !caps.has(x86::AMD_ZEN5) {
      true
    } else {
      ciphertext_len <= X86_64_OPEN_ASM_SHORT_MAX
    }
  }

  #[cfg(all(
    target_arch = "x86_64",
    target_feature = "sse2",
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

    // SAFETY: the public AEAD entry validated the ChaCha20 length bound, and the recommendation gate proves a
    // nonempty buffer plus AVX2+BMI2 availability.
    let tag = unsafe { x86_64_asm::seal_in_place(self.key.as_bytes(), nonce.as_bytes(), aad, buffer) };
    Some(Ok(ChaCha20Poly1305Tag::from_bytes(tag)))
  }

  #[cfg(all(
    target_arch = "x86_64",
    target_feature = "sse2",
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

    // SAFETY: the public AEAD entry validated the ChaCha20 length bound, and the recommendation gate proves a
    // nonempty buffer plus AVX2+BMI2 availability.
    let expected = unsafe { x86_64_asm::open_in_place(self.key.as_bytes(), nonce.as_bytes(), aad, buffer) };
    if !ct::fixed_eq(&expected, tag.as_bytes()).declassify() {
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
    if !ct::fixed_eq(&expected, tag.as_bytes()).declassify() {
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
  ) -> Option<ChaCha20Poly1305Tag> {
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

    let lengths = super::AeadByteLengths::from_usize(aad.len(), buffer.len());

    let mut poly_key = chacha20::poly1305_key_gen(self.key.as_bytes(), nonce.as_bytes());
    let mut authenticator = poly1305::aarch64_neon::AeadPar4::new(&poly_key);
    authenticator.update_padded_segment(aad);

    let mut counter = 1u32;
    let (chunks, remainder) = buffer.as_chunks_mut::<AARCH64_INTERLEAVED_CHUNK>();
    for chunk in chunks {
      // SAFETY: the capability gate proves NEON. Every caller reaches this private helper after the public or
      // diagnostic length bound, or from a same-module test with a bounded buffer; the fixed-size chunks and exact
      // counter advance preserve that whole-buffer bound for this segment.
      unsafe { chacha20::xor_keystream_aarch64_neon(self.key.as_bytes(), counter, nonce.as_bytes(), chunk) };
      authenticator.update_padded_segment(chunk);
      counter = counter.wrapping_add(AARCH64_INTERLEAVED_BLOCKS);
    }

    if !remainder.is_empty() {
      // SAFETY: the capability gate proves NEON. Every caller reaches this private helper after the public or
      // diagnostic length bound, or from a same-module test with a bounded buffer; `counter` tracks the preceding
      // full chunks exactly, so the final remainder cannot exhaust the counter range.
      unsafe { chacha20::xor_keystream_aarch64_neon(self.key.as_bytes(), counter, nonce.as_bytes(), remainder) };
      authenticator.update_padded_segment(remainder);
    }

    let tag = ChaCha20Poly1305Tag::from_bytes(authenticator.finalize(lengths));
    ct::zeroize(&mut poly_key);
    Some(tag)
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

    let lengths = super::AeadByteLengths::from_usize(aad.len(), buffer.len());

    let mut poly_key = chacha20::poly1305_key_gen(self.key.as_bytes(), nonce.as_bytes());
    let mut authenticator = poly1305::aarch64_neon::AeadPar4::new(&poly_key);
    authenticator.update_padded_segment(aad);

    let mut counter = 1u32;
    let (chunks, remainder) = buffer.as_chunks_mut::<AARCH64_INTERLEAVED_CHUNK>();
    for chunk in chunks {
      authenticator.update_padded_segment(chunk);
      // SAFETY: the capability gate proves NEON. Every caller reaches this private helper after the public or
      // diagnostic length bound, or from a same-module test with a bounded buffer; the fixed-size chunks and exact
      // counter advance preserve that whole-buffer bound for this segment.
      unsafe { chacha20::xor_keystream_aarch64_neon(self.key.as_bytes(), counter, nonce.as_bytes(), chunk) };
      counter = counter.wrapping_add(AARCH64_INTERLEAVED_BLOCKS);
    }

    if !remainder.is_empty() {
      authenticator.update_padded_segment(remainder);
      // SAFETY: the capability gate proves NEON. Every caller reaches this private helper after the public or
      // diagnostic length bound, or from a same-module test with a bounded buffer; `counter` tracks the preceding
      // full chunks exactly, so the final remainder cannot exhaust the counter range.
      unsafe { chacha20::xor_keystream_aarch64_neon(self.key.as_bytes(), counter, nonce.as_bytes(), remainder) };
    }

    let expected = authenticator.finalize(lengths);
    ct::zeroize(&mut poly_key);
    if !ct::fixed_eq(&expected, tag.as_bytes()).declassify() {
      ct::zeroize(buffer);
      return Some(Err(OpenError::verification()));
    }
    Some(Ok(()))
  }

  fn encrypt_in_place_owned_unchecked(
    &self,
    nonce: &Nonce96,
    aad: &[u8],
    buffer: &mut [u8],
  ) -> Result<ChaCha20Poly1305Tag, SealError> {
    #[cfg(target_arch = "aarch64")]
    if let Some(tag) = self.encrypt_in_place_interleaved_aarch64(nonce, aad, buffer) {
      return Ok(tag);
    }

    chacha20::xor_keystream(self.key.as_bytes(), 1, nonce.as_bytes(), buffer).map_err(|_| SealError::too_large())?;

    let mut poly_key = chacha20::poly1305_key_gen(self.key.as_bytes(), nonce.as_bytes());
    let tag = ChaCha20Poly1305Tag::from_bytes(poly1305::authenticate_aead(
      AeadPrimitive::ChaCha20Poly1305,
      aad,
      buffer,
      &poly_key,
    ));
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

    let expected = self.compute_tag(nonce, aad, buffer);
    if !ct::fixed_eq(&expected, tag.as_bytes()).declassify() {
      ct::zeroize(buffer);
      return Err(OpenError::verification());
    }

    chacha20::xor_keystream(self.key.as_bytes(), 1, nonce.as_bytes(), buffer).map_err(|_| OpenError::too_large())?;
    Ok(())
  }
}

impl AeadGather for ChaCha20Poly1305 {
  fn __validate_gather_lengths(_aad_len: usize, plaintext_len: usize) -> Result<(), SealError> {
    super::seal_bounded_length_as_u64(plaintext_len, MAX_PLAINTEXT_LEN)?;
    Ok(())
  }

  fn __decrypt_segments_in_place(
    &self,
    nonce: &Self::Nonce,
    aad: &[u8],
    segments: &mut [&mut [u8]],
    tag: &Self::Tag,
  ) -> Result<(), OpenError> {
    if let Some(buffer) = super::segments::contiguous(segments) {
      return <Self as Aead>::decrypt_in_place(self, nonce, aad, buffer, tag);
    }
    let len = super::segments::total_len(segments).ok_or_else(OpenError::too_large)?;
    super::open_bounded_length_as_u64(len, MAX_PLAINTEXT_LEN)?;
    if Self::whole_message_open(len) {
      let matched = super::segments::staged::<STAGED_OPEN_MAX>(segments, len, |message| {
        <Self as Aead>::decrypt_in_place(self, nonce, aad, message, tag).is_ok()
      });
      return super::segments::verdict(segments, matched);
    }
    #[cfg(all(
      target_arch = "x86_64",
      target_feature = "sse2",
      target_os = "linux",
      not(debug_assertions),
      not(feature = "portable-only")
    ))]
    if Self::x86_64_open_asm_selected(len)
      // SAFETY: Selection checks AVX2 and BMI2. The caller checked the message length above.
      && let Some(result) = unsafe { self.open_segments_avx2(nonce, aad, segments, tag) }
    {
      return result;
    }
    #[cfg(all(
      target_arch = "aarch64",
      any(target_os = "linux", target_os = "macos"),
      not(debug_assertions),
      not(feature = "portable-only")
    ))]
    if let Some(expected) = super::segments::spans(segments, |spans| {
      aarch64_asm::open_scatter(self.key.as_bytes(), nonce.as_bytes(), aad, spans)
    }) {
      let matched = ct::fixed_eq(&expected, tag.as_bytes()).declassify();
      return super::segments::verdict(segments, matched);
    }
    let matched = self.open_segments_composed(nonce, aad, segments, len, tag);
    super::segments::verdict(segments, matched)
  }

  fn __encrypt_gather_with_nonce<'input, 'output>(
    &self,
    nonce: &Self::Nonce,
    aad: &[u8],
    plaintext: impl IntoIterator<Item = &'input [u8]>,
    out: &'output mut [core::mem::MaybeUninit<u8>],
    _token: crate::traits::aead::SealToken,
  ) -> Result<(Self::Tag, &'output mut [u8]), SealError> {
    Self::__validate_gather_lengths(aad.len(), out.len())?;
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2", not(feature = "portable-only")))]
    if Self::x86_64_avx512_available() {
      // A TLS record's payload and content type seal like one contiguous
      // message; other shapes take the stream path.
      let len = out.len();
      return super::gather::head_tail_spans(
        plaintext,
        out,
        16,
        |head, tail, dst| {
          let source = chacha20::X86Source {
            head: head.as_ptr(),
            head_len: head.len(),
            tail,
          };
          // SAFETY: the capability check proves AVX512F/VL/BW/DQ. The spans are
          // shared input disjoint from the exclusive output, whose length the
          // driver checked against the validated message.
          let tag = unsafe { x86_64_avx512::seal(self.key.as_bytes(), nonce.as_bytes(), aad, source, dst, len) };
          Ok(ChaCha20Poly1305Tag::from_bytes(tag))
        },
        |rest, out| self.__seal_gather_fallback(nonce, aad, rest, out),
      );
    }
    #[cfg(all(
      target_arch = "aarch64",
      target_endian = "little",
      any(target_os = "macos", target_os = "linux"),
      not(debug_assertions),
      not(miri),
      not(feature = "portable-only")
    ))]
    if !out.is_empty() {
      return super::gather::slices(
        plaintext,
        out,
        |input, output| {
          let (tag, ciphertext) = aarch64_asm::seal_gather(self.key.as_bytes(), nonce.as_bytes(), aad, input, output);
          (ChaCha20Poly1305Tag::from_bytes(tag), ciphertext)
        },
        |prefix, rest, output| {
          self.__seal_gather_fallback(
            nonce,
            aad,
            prefix.iter().map(super::gather::Slice::as_slice).chain(rest),
            output,
          )
        },
      );
    }
    #[cfg(all(
      target_arch = "x86_64",
      target_feature = "sse2",
      target_os = "linux",
      not(debug_assertions),
      not(miri),
      not(feature = "portable-only")
    ))]
    if out.len() <= X86_64_GATHER_ASM_MAX && {
      #[cfg(feature = "std")]
      let caps = crate::platform::caps();
      #[cfg(not(feature = "std"))]
      let caps = crate::platform::caps_static();
      Self::x86_64_asm_recommended(caps, out.len())
    } {
      // A TLS record's payload span enters the integrated kernel directly; its
      // short remainder, such as the content type, follows as extra ciphertext.
      return super::gather::head_tail::<_, _, { chacha20::BLOCK_SIZE }, { chacha20::BLOCK_SIZE }>(
        plaintext,
        out,
        1,
        |counter, stream| {
          stream.fill(0);
          chacha20::xor_keystream(self.key.as_bytes(), counter, nonce.as_bytes(), stream)
            .map_err(|_| SealError::too_large())
        },
        |head, output, tail| {
          // SAFETY: the recommendation gate proves AVX2 and BMI2. The driver
          // passes a nonempty head of `output.len()` bytes and the tail
          // ciphertext at the following stream offset; the public entry
          // validated the complete length against ChaCha20's block limit.
          let (tag, ciphertext) =
            unsafe { x86_64_asm::seal_with_extra(self.key.as_bytes(), nonce.as_bytes(), aad, head, output, tail) };
          (ChaCha20Poly1305Tag::from_bytes(tag), ciphertext)
        },
        |rest, output| self.__seal_gather_fallback(nonce, aad, rest, output),
      );
    }
    self.__seal_gather_fallback(nonce, aad, plaintext, out)
  }
}

/// Longest segmented message that opens on a staged copy: the AVX-512 one-pass open.
const STAGED_OPEN_MAX: usize = 448;

/// XOR the keystream from block `counter` into `data`, a range of a validated message.
#[inline(always)]
fn xor_message_range(key: &ChaCha20Poly1305Key, nonce: &Nonce96, counter: u64, data: &mut [u8]) {
  if data.is_empty() {
    return;
  }
  let counter = u32::try_from(counter).expect("a message block's counter fits 32 bits");
  chacha20::xor_keystream(key.as_bytes(), counter, nonce.as_bytes(), data)
    .expect("the validated message length bounds every block counter");
}

/// The segmented open over the Poly1305 state of its message: each range is
/// authenticated, then decrypted in place. Returns whether the tag matched.
struct OpenSegments<'a, 'b> {
  key: &'a ChaCha20Poly1305Key,
  nonce: &'a Nonce96,
  segments: &'a mut [&'b mut [u8]],
  len: usize,
  tag: &'a ChaCha20Poly1305Tag,
}

impl poly1305::WithAeadMac for OpenSegments<'_, '_> {
  type Output = bool;

  fn run<K: poly1305::AeadKernel>(self, mac: poly1305::AeadMac<'_, K>) -> bool {
    let Self {
      key,
      nonce,
      segments,
      len,
      tag,
    } = self;
    // The block counter starts at 1; block 0 gave the Poly1305 key.
    super::segments::open::<{ chacha20::BLOCK_SIZE }, _>(
      segments,
      len,
      (mac, 1u64),
      |(mac, counter), data| {
        mac.update(data);
        xor_message_range(key, nonce, *counter, data);
        *counter = counter.strict_add(data.len().strict_div(chacha20::BLOCK_SIZE) as u64);
      },
      |(mut mac, counter), rest| {
        mac.update(rest);
        let matched = ct::fixed_eq(&mac.finish(), tag.as_bytes()).declassify();
        if matched {
          xor_message_range(key, nonce, counter, rest);
        }
        matched
      },
    )
  }
}

impl ChaCha20Poly1305 {
  /// Whether `decrypt_in_place` opens a `len`-byte message of at most
  /// [`STAGED_OPEN_MAX`] bytes with a whole-message kernel that cannot run per
  /// segment. Such a message opens on a staged copy.
  #[inline]
  fn whole_message_open(len: usize) -> bool {
    if len > STAGED_OPEN_MAX {
      return false;
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2", not(feature = "portable-only")))]
    if Self::x86_64_avx512_available() {
      return len <= x86_64_avx512::ONE_PASS_MAX;
    }
    #[cfg(all(
      target_arch = "x86_64",
      target_feature = "sse2",
      target_os = "linux",
      not(debug_assertions),
      not(feature = "portable-only")
    ))]
    if Self::x86_64_open_asm_selected(len) {
      return true;
    }
    #[cfg(all(target_arch = "powerpc64", not(feature = "portable-only"), target_endian = "little"))]
    if len <= POWER_SHORT_FAST_MAX {
      return true;
    }
    false
  }

  /// Whether `decrypt_in_place` opens a `len`-byte message with the integrated
  /// x86-64 assembly: AVX-512 is absent and the assembly is recommended.
  #[cfg(all(
    target_arch = "x86_64",
    target_feature = "sse2",
    target_os = "linux",
    not(debug_assertions),
    not(feature = "portable-only")
  ))]
  #[inline]
  fn x86_64_open_asm_selected(len: usize) -> bool {
    #[cfg(feature = "std")]
    let caps = crate::platform::caps();
    #[cfg(not(feature = "std"))]
    let caps = crate::platform::caps_static();
    !Self::x86_64_avx512_available() && Self::x86_64_open_asm_recommended(caps, len)
  }

  /// Open `len` ciphertext bytes across `segments` with the selected Poly1305
  /// and ChaCha20 kernels: each range is authenticated, then decrypted in place.
  fn open_segments_composed(
    &self,
    nonce: &Nonce96,
    aad: &[u8],
    segments: &mut [&mut [u8]],
    len: usize,
    tag: &ChaCha20Poly1305Tag,
  ) -> bool {
    let mut poly_key = chacha20::poly1305_key_gen(self.key.as_bytes(), nonce.as_bytes());
    let open = OpenSegments {
      key: &self.key,
      nonce,
      segments,
      len,
      tag,
    };
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2", not(feature = "portable-only")))]
    let matched = if Self::x86_64_avx512_available() {
      x86_64_avx512::with_mac(&poly_key, aad, len, open)
    } else {
      poly1305::with_aead_mac(AeadPrimitive::ChaCha20Poly1305, &poly_key, aad, len, open)
    };
    #[cfg(not(all(target_arch = "x86_64", target_feature = "sse2", not(feature = "portable-only"))))]
    let matched = poly1305::with_aead_mac(AeadPrimitive::ChaCha20Poly1305, &poly_key, aad, len, open);
    ct::zeroize(&mut poly_key);
    matched
  }

  fn __seal_gather_fallback<'input, 'output>(
    &self,
    nonce: &Nonce96,
    aad: &[u8],
    plaintext: impl IntoIterator<Item = &'input [u8]>,
    out: &'output mut [core::mem::MaybeUninit<u8>],
  ) -> Result<(ChaCha20Poly1305Tag, &'output mut [u8]), SealError> {
    let ciphertext = super::gather::xor::<64, 1024>(plaintext, out, 1, |counter, stream| {
      stream.fill(0);
      chacha20::xor_keystream(self.key.as_bytes(), counter, nonce.as_bytes(), stream)
        .map_err(|_| SealError::too_large())
    })?;
    let tag = ChaCha20Poly1305Tag::from_bytes(self.compute_tag(nonce, aad, ciphertext));
    Ok((tag, ciphertext))
  }
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

  fn __encrypt_in_place_with_nonce(
    &self,
    nonce: &Self::Nonce,
    aad: &[u8],
    buffer: &mut [u8],
    _token: crate::traits::aead::SealToken,
  ) -> Result<Self::Tag, SealError> {
    super::seal_bounded_length_as_u64(buffer.len(), MAX_PLAINTEXT_LEN)?;

    #[cfg(any(
      all(target_arch = "x86_64", target_feature = "sse2"),
      all(target_arch = "powerpc64", not(feature = "portable-only"), target_endian = "little")
    ))]
    if buffer.is_empty()
      && let Some(result) = self.encrypt_empty_text_fast(nonce, aad)
    {
      return result;
    }

    #[cfg(all(target_arch = "powerpc64", not(feature = "portable-only"), target_endian = "little"))]
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

    #[cfg(all(target_arch = "x86_64", target_feature = "sse2", not(feature = "portable-only")))]
    if Self::x86_64_avx512_available() {
      let data = buffer.as_mut_ptr();
      let source = chacha20::X86Source {
        head: data,
        head_len: buffer.len(),
        tail: &[],
      };
      // SAFETY: the capability check proves AVX512F/VL/BW/DQ, and the entry
      // validated the ChaCha20 length bound. `buffer` is sealed in place.
      let tag = unsafe { x86_64_avx512::seal(self.key.as_bytes(), nonce.as_bytes(), aad, source, data, buffer.len()) };
      return Ok(ChaCha20Poly1305Tag::from_bytes(tag));
    }

    #[cfg(all(
      target_arch = "x86_64",
      target_feature = "sse2",
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

    #[cfg(any(
      all(target_arch = "x86_64", target_feature = "sse2"),
      all(target_arch = "powerpc64", not(feature = "portable-only"), target_endian = "little")
    ))]
    if buffer.is_empty()
      && let Some(result) = self.decrypt_empty_text_fast(nonce, aad, tag)
    {
      return result;
    }

    #[cfg(all(target_arch = "powerpc64", not(feature = "portable-only"), target_endian = "little"))]
    if let Some(result) = self.decrypt_short_text_power_fast(nonce, aad, buffer, tag) {
      return result;
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse2", not(feature = "portable-only")))]
    if Self::x86_64_avx512_available() {
      // SAFETY: the capability check proves AVX512F/VL/BW/DQ, and the entry
      // validated the ChaCha20 length bound. The kernel zeroes `buffer` on a mismatch.
      let matched = unsafe {
        x86_64_avx512::open(
          self.key.as_bytes(),
          nonce.as_bytes(),
          aad,
          buffer.as_mut_ptr(),
          buffer.len(),
          tag.as_bytes(),
        )
      };
      return if matched {
        Ok(())
      } else {
        Err(OpenError::verification())
      };
    }

    #[cfg(all(
      target_arch = "x86_64",
      target_feature = "sse2",
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
  #[cfg(any(
    all(target_arch = "x86_64", target_feature = "sse2"),
    all(target_arch = "powerpc64", not(feature = "portable-only"), target_endian = "little")
  ))]
  use alloc::vec::Vec;

  use super::*;
  use crate::aead::expert::AeadWithNonce;

  #[test]
  fn wrong_nonce_rejected() {
    let key = ChaCha20Poly1305Key::from_bytes([0x42u8; 32]);
    let nonce = Nonce96::from_bytes([0x07u8; 12]);
    let cipher = ChaCha20Poly1305::new(&key);

    let mut buf = *b"nonce test";
    let tag = cipher
      .encrypt_in_place(&nonce, b"aad", &mut buf)
      .expect("valid input must encrypt");

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
    let tag = cipher
      .encrypt_in_place(&nonce, b"aad", &mut buf)
      .expect("valid input must encrypt");

    let mut bad_tag = tag.to_bytes();
    bad_tag[0] ^= 0xFF;
    let bad_tag = ChaCha20Poly1305Tag::from_bytes(bad_tag);

    let result = cipher.decrypt_in_place(&nonce, b"aad", &mut buf, &bad_tag);
    assert!(result.is_err());
    assert!(buf.iter().all(|&b| b == 0), "buffer not zeroed on auth failure");
  }

  #[cfg(all(target_arch = "x86_64", target_feature = "sse2", target_os = "linux"))]
  #[test]
  fn x86_64_asm_policy_matches_configured_thresholds() {
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

  #[cfg(any(
    all(target_arch = "x86_64", target_feature = "sse2"),
    all(target_arch = "powerpc64", not(feature = "portable-only"), target_endian = "little")
  ))]
  #[test]
  fn empty_text_fast_decrypt_matches_owned_path() {
    let key = ChaCha20Poly1305Key::from_bytes([0x42; KEY_SIZE]);
    let nonce = Nonce96::from_bytes([0x24; NONCE_SIZE]);
    let cipher = ChaCha20Poly1305::new(&key);

    for aad_len in [0usize, 1, 14, 15, 16, 17, 31, 32, 33, 63, 64] {
      let aad = (0u8..)
        .take(aad_len)
        .map(|index| 0xa7u8.wrapping_add(index.wrapping_mul(7)))
        .collect::<Vec<_>>();
      let mut ciphertext = Vec::new();

      let expected_tag = cipher
        .encrypt_in_place_owned_unchecked(&nonce, &aad, &mut ciphertext)
        .expect("owned path must produce an empty-text tag");
      let actual = cipher.decrypt_empty_text_fast(&nonce, &aad, &expected_tag);

      if aad_len > SMALL_AAD_FAST_MAX {
        assert!(
          actual.is_none(),
          "empty decrypt fast path applied outside its configured gate: aad_len={aad_len}"
        );
        continue;
      }

      assert_eq!(
        actual.expect("empty decrypt fast path must apply inside its configured gate"),
        Ok(())
      );

      let mut bad_tag = expected_tag.to_bytes();
      bad_tag[0] ^= 0x80;
      assert_eq!(
        cipher
          .decrypt_empty_text_fast(&nonce, &aad, &ChaCha20Poly1305Tag::from_bytes(bad_tag))
          .expect("empty decrypt fast path must apply inside its configured gate"),
        Err(OpenError::verification())
      );
    }
  }

  #[cfg(all(target_arch = "powerpc64", not(feature = "portable-only"), target_endian = "little"))]
  #[test]
  fn power_short_fast_encrypt_matches_owned_path() {
    let key = ChaCha20Poly1305Key::from_bytes([0x42; KEY_SIZE]);
    let nonce = Nonce96::from_bytes([0x24; NONCE_SIZE]);
    let cipher = ChaCha20Poly1305::new(&key);

    for plaintext_len in [0usize, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65] {
      let plaintext = (0u8..)
        .take(plaintext_len)
        .map(|index| 0x51u8.wrapping_add(index.wrapping_mul(13)))
        .collect::<Vec<_>>();

      for aad_len in [0usize, 1, 14, 15, 16, 17, 31, 32, 33, 63, 64] {
        let aad = (0u8..)
          .take(aad_len)
          .map(|index| 0xa7u8.wrapping_add(index.wrapping_mul(7)))
          .collect::<Vec<_>>();

        let mut actual = plaintext.clone();
        let actual_tag = cipher.encrypt_short_text_power_fast(&nonce, &aad, &mut actual);
        if plaintext_len == 0 || plaintext_len > POWER_SHORT_FAST_MAX || aad_len > SMALL_AAD_FAST_MAX {
          assert!(
            actual_tag.is_none(),
            "Power short fast path applied outside its configured gate: plaintext_len={plaintext_len} \
             aad_len={aad_len}"
          );
          assert_eq!(actual, plaintext);
          continue;
        }

        let mut expected = plaintext.clone();
        let expected_tag = cipher
          .encrypt_in_place_owned_unchecked(&nonce, &aad, &mut expected)
          .expect("bounded test input must encrypt through the owned path");
        let actual_tag = actual_tag
          .expect("Power short fast path must apply inside its configured gate")
          .expect("bounded test input must encrypt through the Power fast path");

        assert_eq!(
          actual, expected,
          "Power short ciphertext mismatch plaintext_len={plaintext_len} aad_len={aad_len}"
        );
        assert!(
          actual_tag.ct_eq(&expected_tag).declassify(),
          "Power short tag mismatch plaintext_len={plaintext_len} aad_len={aad_len}"
        );
      }
    }
  }

  #[cfg(all(target_arch = "powerpc64", not(feature = "portable-only"), target_endian = "little"))]
  #[test]
  fn power_short_fast_decrypt_matches_owned_path() {
    let key = ChaCha20Poly1305Key::from_bytes([0x42; KEY_SIZE]);
    let nonce = Nonce96::from_bytes([0x24; NONCE_SIZE]);
    let cipher = ChaCha20Poly1305::new(&key);

    for plaintext_len in [0usize, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65] {
      let plaintext = (0u8..)
        .take(plaintext_len)
        .map(|index| 0x51u8.wrapping_add(index.wrapping_mul(13)))
        .collect::<Vec<_>>();

      for aad_len in [0usize, 1, 14, 15, 16, 17, 31, 32, 33, 63, 64] {
        let aad = (0u8..)
          .take(aad_len)
          .map(|index| 0xa7u8.wrapping_add(index.wrapping_mul(7)))
          .collect::<Vec<_>>();

        let mut ciphertext = plaintext.clone();
        let tag = cipher
          .encrypt_in_place_owned_unchecked(&nonce, &aad, &mut ciphertext)
          .expect("bounded test input must encrypt through the owned path");
        let mut actual = ciphertext.clone();
        let actual_result = cipher.decrypt_short_text_power_fast(&nonce, &aad, &mut actual, &tag);

        if plaintext_len == 0 || plaintext_len > POWER_SHORT_FAST_MAX || aad_len > SMALL_AAD_FAST_MAX {
          assert!(
            actual_result.is_none(),
            "Power short decrypt fast path applied outside its configured gate: plaintext_len={plaintext_len} \
             aad_len={aad_len}"
          );
          assert_eq!(actual, ciphertext);
          continue;
        }

        assert_eq!(
          actual_result.expect("Power short decrypt fast path must apply inside its configured gate"),
          Ok(())
        );
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
            .expect("Power short decrypt fast path must apply inside its configured gate"),
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
    let tag = cipher
      .encrypt_in_place(&nonce, b"correct", &mut buf)
      .expect("valid input must encrypt");

    let result = cipher.decrypt_in_place(&nonce, b"wrong", &mut buf, &tag);
    assert!(result.is_err());
  }

  #[cfg(all(
    target_arch = "x86_64",
    target_feature = "sse2",
    target_os = "linux",
    not(feature = "portable-only"),
    not(miri)
  ))]
  #[test]
  fn avx2_span_open_matches_oracle_and_clears_only_segments_on_failure() {
    use ::chacha20poly1305::{
      ChaCha20Poly1305 as Oracle, KeyInit,
      aead::{Aead as _, Payload, array::Array},
    };
    use alloc::vec;

    const LENS: &[usize] = &[
      321, 448, 449, 511, 512, 513, 575, 576, 577, 1088, 1089, 1601, 4097, 16_385,
    ];

    let caps = crate::platform::caps();
    if !caps.has(crate::platform::caps::x86::AVX2.union(crate::platform::caps::x86::BMI2)) {
      return;
    }

    let key_bytes = [0x42; ChaCha20Poly1305::KEY_SIZE];
    let nonce_bytes = [0x24; Nonce96::LENGTH];
    let cipher = ChaCha20Poly1305::new(&ChaCha20Poly1305Key::from_bytes(key_bytes));
    let nonce = Nonce96::from_bytes(nonce_bytes);
    let oracle = Oracle::new(&Array(key_bytes));
    let aad = [0xa7; 13];

    for &len in LENS {
      let plaintext: Vec<_> = (0..len).map(|i| i.to_le_bytes()[0].wrapping_mul(17)).collect();
      let sealed = oracle
        .encrypt(
          &Array(nonce_bytes),
          Payload {
            msg: &plaintext,
            aad: &aad,
          },
        )
        .expect("oracle seals the fixture");
      let (ciphertext, tag) = sealed.split_at(len);
      let tag = ChaCha20Poly1305::tag_from_slice(tag).expect("16-byte tag");
      let mut cut_sets: Vec<Vec<usize>> = [
        1usize,
        63,
        64,
        65,
        575,
        576,
        577,
        1087,
        1088,
        1089,
        len / 2 | 1,
        len - 1,
      ]
      .into_iter()
      .filter(|cut| (1..len).contains(cut))
      .map(|cut| vec![cut])
      .collect();
      // At most 128 segments take the span assembly.
      cut_sets.push((1..len).step_by((len / 120).max(97)).collect());
      cut_sets.push((560..len.min(600)).collect());
      cut_sets.retain(|cuts| !cuts.is_empty());
      for cuts in &cut_sets {
        for corrupt in [false, true] {
          let mut ends = cuts.clone();
          ends.insert(0, 0);
          ends.push(len);
          // Guards and independent allocations keep each segment a separate range.
          let mut storage: Vec<Vec<u8>> = ends
            .windows(2)
            .map(|range| [&[0xa5], &ciphertext[range[0]..range[1]], &[0x5a][..]].concat())
            .collect();
          if corrupt {
            storage[1][1] ^= 0x01;
          }
          let mut segments: Vec<&mut [u8]> = storage
            .iter_mut()
            .map(|bytes| {
              let end = bytes.len() - 1;
              &mut bytes[1..end]
            })
            .collect();
          // SAFETY: The CPU check covers AVX2 and BMI2. Every fixture has 321 to 16,385 bytes.
          let result =
            unsafe { cipher.open_segments_avx2(&nonce, &aad, &mut segments, &tag) }.expect("at most 128 spans");
          let opened: Vec<u8> = storage
            .iter()
            .flat_map(|bytes| bytes[1..bytes.len() - 1].to_vec())
            .collect();
          if corrupt {
            assert_eq!(result, Err(OpenError::verification()), "len {len}, cuts {cuts:?}");
            assert!(opened.iter().all(|&byte| byte == 0), "len {len}, cuts {cuts:?}");
          } else {
            result.expect("the span assembly opens the oracle ciphertext");
            assert_eq!(opened, plaintext, "len {len}, cuts {cuts:?}");
          }
          assert!(
            storage
              .iter()
              .all(|bytes| bytes[0] == 0xa5 && bytes[bytes.len() - 1] == 0x5a),
            "len {len}, cuts {cuts:?}"
          );
        }
      }
    }
  }
}

#[cfg(all(test, feature = "std"))]
mod kernel_tests {
  use super::*;
  use crate::aead::expert::AeadWithNonce as _;
  fn pattern_bytes(len: usize, seed: u8) -> Vec<u8> {
    let mut out = vec![0u8; len];
    for (index, byte) in out.iter_mut().enumerate() {
      let index = index.to_le_bytes()[0];
      *byte = seed
        .wrapping_add(index.wrapping_mul(17))
        .wrapping_add(index.rotate_left(1));
    }
    out
  }
  #[test]
  fn owned_decrypt_large_inputs_match_normal_path() {
    const PLAINTEXT_LENS: &[usize] = &[1024, 1025, 4095, 4096, 4097, 16_384];

    let key = ChaCha20Poly1305Key::from_bytes([0x42; ChaCha20Poly1305::KEY_SIZE]);
    let nonce = Nonce96::from_bytes([0x24; Nonce96::LENGTH]);
    let cipher = ChaCha20Poly1305::new(&key);

    for &plaintext_len in PLAINTEXT_LENS {
      let plaintext = pattern_bytes(plaintext_len, 0x51);
      let aad = pattern_bytes(257, 0xa7);
      let mut ciphertext = plaintext.clone();
      let tag = cipher
        .encrypt_in_place(&nonce, &aad, &mut ciphertext)
        .expect("diagnostic ChaCha20-Poly1305 fixture must seal");

      cipher
        .decrypt_in_place_owned_unchecked(&nonce, &aad, &mut ciphertext, &tag)
        .expect("owned diagnostic ChaCha20-Poly1305 path must open its fixture");

      assert_eq!(
        ciphertext, plaintext,
        "owned diagnostic decrypt mismatch at plaintext_len={plaintext_len}"
      );
    }
  }
  #[test]
  fn owned_decrypt_zeroes_large_buffer_on_bad_tag() {
    let key = ChaCha20Poly1305Key::from_bytes([0x11; ChaCha20Poly1305::KEY_SIZE]);
    let nonce = Nonce96::from_bytes([0x22; Nonce96::LENGTH]);
    let cipher = ChaCha20Poly1305::new(&key);

    let plaintext = pattern_bytes(4097, 0x6d);
    let aad = pattern_bytes(33, 0x95);
    let mut ciphertext = plaintext;
    let mut tag = cipher
      .encrypt_in_place(&nonce, &aad, &mut ciphertext)
      .expect("diagnostic ChaCha20-Poly1305 bad-tag fixture must seal")
      .to_bytes();
    tag[7] ^= 0x80;

    let result =
      cipher.decrypt_in_place_owned_unchecked(&nonce, &aad, &mut ciphertext, &ChaCha20Poly1305Tag::from_bytes(tag));

    result.expect_err("owned diagnostic ChaCha20-Poly1305 path must reject a modified tag");
    assert!(
      ciphertext.iter().all(|&byte| byte == 0),
      "owned diagnostic decrypt must zero caller buffer on verification failure"
    );
  }
  #[cfg(all(
    not(feature = "portable-only"),
    not(miri),
    target_arch = "x86_64",
    target_os = "linux"
  ))]
  #[test]
  fn x86_asm_encrypt_matches_owned_path() {
    const PLAINTEXT_LENS: &[usize] = &[
      0, 1, 2, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 191, 192, 193, 255, 256, 257, 319, 320, 321, 511,
      512, 513, 1024, 1025, 4095, 4096, 4097, 16_384,
    ];
    const AAD_LENS: &[usize] = &[0, 1, 13, 14, 15, 16, 17, 31, 32, 33, 63, 64, 65, 257];

    let caps = crate::platform::caps();
    if !caps.has(crate::platform::caps::x86::AVX2.union(crate::platform::caps::x86::BMI2)) {
      return;
    }

    let key = ChaCha20Poly1305Key::from_bytes([0x42; ChaCha20Poly1305::KEY_SIZE]);
    let nonce = Nonce96::from_bytes([0x24; Nonce96::LENGTH]);
    let cipher = ChaCha20Poly1305::new(&key);

    for &plaintext_len in PLAINTEXT_LENS {
      let plaintext = pattern_bytes(plaintext_len, 0x51);

      for &aad_len in AAD_LENS {
        let aad = pattern_bytes(aad_len, 0xa7);
        let mut actual = plaintext.clone();
        if actual.is_empty() {
          continue;
        }
        // SAFETY: The host supports AVX2 and BMI2. The initialized buffer is nonempty and below the counter limit.
        let actual_tag = unsafe { x86_64_asm::seal_in_place(key.as_bytes(), nonce.as_bytes(), &aad, &mut actual) };

        let mut expected = plaintext.clone();
        let expected_tag = cipher
          .encrypt_in_place_owned_unchecked(&nonce, &aad, &mut expected)
          .expect("owned diagnostic ChaCha20-Poly1305 path must seal");
        assert_eq!(
          actual, expected,
          "x86 asm ciphertext mismatch plaintext_len={plaintext_len} aad_len={aad_len}"
        );
        assert!(
          actual_tag == *expected_tag.as_bytes(),
          "x86 asm tag mismatch plaintext_len={plaintext_len} aad_len={aad_len}"
        );
      }
    }
  }
}
