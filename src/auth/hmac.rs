//! HMAC-SHA2 family (RFC 2104, FIPS 198-1).

use crate::{
  hashes::crypto::{
    Sha256, Sha384, Sha512,
    dispatch_util::len_hint_from_u64,
    sha256::{H0 as SHA256_H0, Sha256Prefix, dispatch as sha256_dispatch},
    sha384::{H0 as SHA384_H0, Sha384Prefix, dispatch as sha384_dispatch},
    sha512::{H0 as SHA512_H0, Sha512Prefix, dispatch as sha512_dispatch},
  },
  traits::{Digest, Mac, VerificationError, ct},
};

const SHA256_BLOCK_SIZE: usize = 64;
const SHA256_TAG_SIZE: usize = 32;
const SHA512_FAMILY_BLOCK_SIZE: usize = 128;
const SHA384_TAG_SIZE: usize = 48;
const SHA512_TAG_SIZE: usize = 64;
const HMAC_IPAD_WORD: u64 = 0x3636_3636_3636_3636;
const HMAC_OPAD_WORD: u64 = 0x5c5c_5c5c_5c5c_5c5c;
const HMAC_PAD_DELTA_WORD: u64 = HMAC_IPAD_WORD ^ HMAC_OPAD_WORD;

macro_rules! define_hmac_tag_type {
  ($name:ident, $len:expr, $doc:literal) => {
    #[doc = $doc]
    #[derive(Clone, Copy)]
    pub struct $name([u8; Self::LENGTH]);

    impl PartialEq for $name {
      #[inline]
      fn eq(&self, other: &Self) -> bool {
        ct::fixed_eq(&self.0, &other.0)
      }
    }

    impl Eq for $name {}

    impl core::hash::Hash for $name {
      #[inline]
      fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        core::hash::Hash::hash(&self.0, state);
      }
    }

    impl $name {
      /// Tag length in bytes.
      pub const LENGTH: usize = $len;

      /// Construct a typed tag from raw bytes.
      #[inline]
      #[must_use]
      pub const fn from_bytes(bytes: [u8; Self::LENGTH]) -> Self {
        Self(bytes)
      }

      /// Return the tag bytes.
      #[inline]
      #[must_use]
      pub const fn to_bytes(self) -> [u8; Self::LENGTH] {
        self.0
      }

      /// Return the tag bytes.
      #[inline]
      #[must_use]
      pub const fn into_bytes(self) -> [u8; Self::LENGTH] {
        self.0
      }

      /// Borrow the tag bytes as a fixed-size array.
      #[inline]
      #[must_use]
      pub const fn as_bytes(&self) -> &[u8; Self::LENGTH] {
        &self.0
      }

      /// Borrow the tag bytes as a slice.
      #[inline]
      #[must_use]
      pub fn as_slice(&self) -> &[u8] {
        &self.0
      }
    }

    impl Default for $name {
      #[inline]
      fn default() -> Self {
        Self([0u8; Self::LENGTH])
      }
    }

    impl From<[u8; $len]> for $name {
      #[inline]
      fn from(bytes: [u8; $len]) -> Self {
        Self::from_bytes(bytes)
      }
    }

    impl From<$name> for [u8; $len] {
      #[inline]
      fn from(tag: $name) -> Self {
        tag.to_bytes()
      }
    }

    impl TryFrom<&[u8]> for $name {
      type Error = core::array::TryFromSliceError;

      #[inline]
      fn try_from(bytes: &[u8]) -> Result<Self, Self::Error> {
        Ok(Self::from_bytes(bytes.try_into()?))
      }
    }

    impl AsRef<[u8]> for $name {
      #[inline]
      fn as_ref(&self) -> &[u8] {
        &self.0
      }
    }

    impl AsRef<[u8; $len]> for $name {
      #[inline]
      fn as_ref(&self) -> &[u8; $len] {
        &self.0
      }
    }

    impl core::fmt::Debug for $name {
      fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}(", stringify!($name))?;
        for byte in self.0 {
          write!(f, "{byte:02x}")?;
        }
        write!(f, ")")
      }
    }

    #[cfg(feature = "serde")]
    #[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
    impl serde::Serialize for $name {
      fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_bytes(&self.0)
      }
    }

    #[cfg(feature = "serde")]
    #[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
    impl<'de> serde::Deserialize<'de> for $name {
      fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct ByteVisitor;

        impl<'de> serde::de::Visitor<'de> for ByteVisitor {
          type Value = $name;

          fn expecting(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            write!(f, "{} bytes", <$name>::LENGTH)
          }

          fn visit_bytes<E: serde::de::Error>(self, v: &[u8]) -> Result<Self::Value, E> {
            let arr: [u8; <$name>::LENGTH] = v.try_into().map_err(|_| E::invalid_length(v.len(), &self))?;
            Ok(<$name>::from_bytes(arr))
          }

          fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
            let mut arr = [0u8; <$name>::LENGTH];
            for (i, byte) in arr.iter_mut().enumerate() {
              *byte = seq
                .next_element()?
                .ok_or_else(|| serde::de::Error::invalid_length(i, &self))?;
            }
            Ok(<$name>::from_bytes(arr))
          }
        }

        deserializer.deserialize_bytes(ByteVisitor)
      }
    }
  };
}

define_hmac_tag_type!(HmacSha256Tag, SHA256_TAG_SIZE, "HMAC-SHA256 authentication tag.");
define_hmac_tag_type!(HmacSha384Tag, SHA384_TAG_SIZE, "HMAC-SHA384 authentication tag.");
define_hmac_tag_type!(HmacSha512Tag, SHA512_TAG_SIZE, "HMAC-SHA512 authentication tag.");

#[cfg(target_arch = "x86_64")]
#[inline]
fn hmac_sha256_oneshot_prefers_streaming(caps: crate::platform::Caps) -> bool {
  caps.has(crate::platform::caps::x86::INTEL_SAPPHIRE_RAPIDS)
}

#[inline]
fn hmac_sha256_oneshot_should_stream() -> bool {
  #[cfg(target_arch = "x86_64")]
  return hmac_sha256_oneshot_prefers_streaming(crate::platform::caps());

  #[cfg(not(target_arch = "x86_64"))]
  return false;
}

#[inline]
pub(crate) fn hmac_prefix_state<const BLOCK_SIZE: usize, T>(
  key_block: &mut [u8; BLOCK_SIZE],
  build: impl FnOnce(&[u8; BLOCK_SIZE], &[u8; BLOCK_SIZE]) -> T,
) -> T {
  let mut ipad = [0x36u8; BLOCK_SIZE];
  let mut opad = [0x5Cu8; BLOCK_SIZE];
  if BLOCK_SIZE == SHA256_BLOCK_SIZE || BLOCK_SIZE == SHA512_FAMILY_BLOCK_SIZE {
    hmac_prefix_pad_words(key_block, &mut ipad, &mut opad);
  } else {
    for ((ipad_byte, opad_byte), key_byte) in ipad.iter_mut().zip(opad.iter_mut()).zip(key_block.iter().copied()) {
      *ipad_byte ^= key_byte;
      *opad_byte ^= key_byte;
    }
  }

  let result = build(&ipad, &opad);

  ct::zeroize_no_fence(key_block);
  ct::zeroize_no_fence(&mut ipad);
  ct::zeroize_no_fence(&mut opad);
  core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);

  result
}

#[inline(always)]
fn read_u64_unaligned<const N: usize>(bytes: &[u8; N], offset: usize) -> u64 {
  assert!(offset.strict_add(8) <= N);
  // SAFETY: `offset + 8 <= N` above keeps the 8-byte read in bounds. The source is a byte array, so
  // it may be unaligned and `read_unaligned` is required.
  unsafe { core::ptr::read_unaligned(bytes.as_ptr().add(offset).cast::<u64>()) }
}

#[inline(always)]
fn write_u64_unaligned<const N: usize>(bytes: &mut [u8; N], offset: usize, word: u64) {
  assert!(offset.strict_add(8) <= N);
  // SAFETY: `offset + 8 <= N` above keeps the 8-byte write in bounds. The destination is a byte
  // array, so it may be unaligned and `write_unaligned` is required.
  unsafe { core::ptr::write_unaligned(bytes.as_mut_ptr().add(offset).cast::<u64>(), word) };
}

#[inline(always)]
fn hmac_prefix_pad_word<const BLOCK_SIZE: usize>(
  key_block: &[u8; BLOCK_SIZE],
  ipad: &mut [u8; BLOCK_SIZE],
  opad: &mut [u8; BLOCK_SIZE],
  offset: usize,
) {
  let key = read_u64_unaligned(key_block, offset);
  write_u64_unaligned(ipad, offset, HMAC_IPAD_WORD ^ key);
  write_u64_unaligned(opad, offset, HMAC_OPAD_WORD ^ key);
}

#[inline(always)]
fn hmac_prefix_pad_words<const BLOCK_SIZE: usize>(
  key_block: &[u8; BLOCK_SIZE],
  ipad: &mut [u8; BLOCK_SIZE],
  opad: &mut [u8; BLOCK_SIZE],
) {
  hmac_prefix_pad_word(key_block, ipad, opad, 0);
  hmac_prefix_pad_word(key_block, ipad, opad, 8);
  hmac_prefix_pad_word(key_block, ipad, opad, 16);
  hmac_prefix_pad_word(key_block, ipad, opad, 24);
  hmac_prefix_pad_word(key_block, ipad, opad, 32);
  hmac_prefix_pad_word(key_block, ipad, opad, 40);
  hmac_prefix_pad_word(key_block, ipad, opad, 48);
  hmac_prefix_pad_word(key_block, ipad, opad, 56);

  if BLOCK_SIZE == SHA512_FAMILY_BLOCK_SIZE {
    hmac_prefix_pad_word(key_block, ipad, opad, 64);
    hmac_prefix_pad_word(key_block, ipad, opad, 72);
    hmac_prefix_pad_word(key_block, ipad, opad, 80);
    hmac_prefix_pad_word(key_block, ipad, opad, 88);
    hmac_prefix_pad_word(key_block, ipad, opad, 96);
    hmac_prefix_pad_word(key_block, ipad, opad, 104);
    hmac_prefix_pad_word(key_block, ipad, opad, 112);
    hmac_prefix_pad_word(key_block, ipad, opad, 120);
  }
}

#[inline(always)]
fn hmac_outer_pad_word<const OUT_SIZE: usize, const BLOCK_SIZE: usize>(
  outer: &mut [u8; OUT_SIZE],
  ipad: &[u8; BLOCK_SIZE],
  offset: usize,
) {
  let word = read_u64_unaligned(ipad, offset) ^ HMAC_PAD_DELTA_WORD;
  write_u64_unaligned(outer, offset, word);
}

#[inline(always)]
fn hmac_outer_pad_words<const OUT_SIZE: usize, const BLOCK_SIZE: usize>(
  outer: &mut [u8; OUT_SIZE],
  ipad: &[u8; BLOCK_SIZE],
) {
  hmac_outer_pad_word(outer, ipad, 0);
  hmac_outer_pad_word(outer, ipad, 8);
  hmac_outer_pad_word(outer, ipad, 16);
  hmac_outer_pad_word(outer, ipad, 24);
  hmac_outer_pad_word(outer, ipad, 32);
  hmac_outer_pad_word(outer, ipad, 40);
  hmac_outer_pad_word(outer, ipad, 48);
  hmac_outer_pad_word(outer, ipad, 56);

  if BLOCK_SIZE == SHA512_FAMILY_BLOCK_SIZE {
    hmac_outer_pad_word(outer, ipad, 64);
    hmac_outer_pad_word(outer, ipad, 72);
    hmac_outer_pad_word(outer, ipad, 80);
    hmac_outer_pad_word(outer, ipad, 88);
    hmac_outer_pad_word(outer, ipad, 96);
    hmac_outer_pad_word(outer, ipad, 104);
    hmac_outer_pad_word(outer, ipad, 112);
    hmac_outer_pad_word(outer, ipad, 120);
  }
}

/// HMAC-SHA256 authentication state.
///
/// # Examples
///
/// ```rust
/// use rscrypto::{HmacSha256, Mac};
///
/// let key = b"shared-secret";
/// let data = b"auth message";
///
/// let tag = HmacSha256::mac(key, data);
///
/// let mut mac = HmacSha256::new(key);
/// mac.update(b"auth ");
/// mac.update(b"message");
/// assert!(mac.verify(&tag).is_ok());
/// ```
#[derive(Clone)]
pub struct HmacSha256 {
  inner: Sha256,
  inner_init: Sha256Prefix,
  outer_init: Sha256Prefix,
}

impl core::fmt::Debug for HmacSha256 {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("HmacSha256").finish_non_exhaustive()
  }
}

impl HmacSha256 {
  /// HMAC-SHA256 block size in bytes.
  pub const BLOCK_SIZE: usize = SHA256_BLOCK_SIZE;

  /// HMAC-SHA256 tag size in bytes.
  pub const TAG_SIZE: usize = SHA256_TAG_SIZE;

  /// Compute the HMAC tag of `data` in one shot.
  #[inline]
  #[must_use]
  pub fn mac(key: &[u8], data: &[u8]) -> HmacSha256Tag {
    <Self as Mac>::mac(key, data)
  }

  /// Verify `expected` against the HMAC tag of `data` in constant time.
  #[inline]
  #[must_use = "HMAC verification must be checked; a dropped Result silently accepts a forged tag"]
  pub fn verify_tag(key: &[u8], data: &[u8], expected: &HmacSha256Tag) -> Result<(), VerificationError> {
    <Self as Mac>::verify_tag(key, data, expected)
  }

  #[cfg(any(test, feature = "diag"))]
  #[allow(dead_code)]
  pub(crate) fn new_with_compress_for_test(
    key: &[u8],
    compress: crate::hashes::crypto::sha256::kernels::CompressBlocksFn,
  ) -> Self {
    let mut key_block = [0u8; SHA256_BLOCK_SIZE];
    if key.len() > SHA256_BLOCK_SIZE {
      let digest = Sha256::digest(key);
      key_block[..SHA256_TAG_SIZE].copy_from_slice(&digest);
    } else if let Some(dst) = key_block.get_mut(..key.len()) {
      dst.copy_from_slice(key);
    }

    let (inner, inner_init, outer_init) = hmac_prefix_state(&mut key_block, |ipad, opad| {
      let mut inner = Sha256::new_with_compress_for_test(compress);
      inner.update(ipad);
      let inner_init = inner.aligned_prefix();

      let mut outer = Sha256::new_with_compress_for_test(compress);
      outer.update(opad);
      let outer_init = outer.aligned_prefix();

      (inner, inner_init, outer_init)
    });

    Self {
      inner,
      inner_init,
      outer_init,
    }
  }

  #[cfg(any(test, feature = "diag"))]
  #[allow(dead_code)]
  pub(crate) fn mac_with_compress_for_test(
    key: &[u8],
    data: &[u8],
    compress: crate::hashes::crypto::sha256::kernels::CompressBlocksFn,
  ) -> [u8; SHA256_TAG_SIZE] {
    let mut mac = Self::new_with_compress_for_test(key, compress);
    mac.update(data);
    mac.finalize().to_bytes()
  }
}

#[cfg(feature = "diag")]
#[must_use]
pub fn diag_hmac_sha256_verify_portable(key: &[u8; SHA256_TAG_SIZE], expected: &[u8; SHA256_TAG_SIZE]) -> bool {
  let compress = crate::hashes::crypto::sha256::kernels::compress_blocks_fn(
    crate::hashes::crypto::sha256::kernels::Sha256KernelId::Portable,
  );
  let tag = HmacSha256::mac_with_compress_for_test(key, b"binsec", compress);
  ct::fixed_eq(&tag, expected)
}

impl Mac for HmacSha256 {
  const TAG_SIZE: usize = SHA256_TAG_SIZE;
  type Tag = HmacSha256Tag;

  fn new(key: &[u8]) -> Self {
    let mut key_block = [0u8; SHA256_BLOCK_SIZE];
    if key.len() > SHA256_BLOCK_SIZE {
      let digest = Sha256::digest(key);
      for (dst, src) in key_block.iter_mut().zip(digest.iter()) {
        *dst = *src;
      }
    } else {
      for (dst, src) in key_block.iter_mut().zip(key.iter()) {
        *dst = *src;
      }
    }

    let (inner_init, inner_init_prefix, outer_init_prefix) = hmac_prefix_state(&mut key_block, |ipad, opad| {
      let mut inner_init = Sha256::new();
      inner_init.update(ipad);

      let mut outer_init = Sha256::new();
      outer_init.update(opad);

      let inner_init_prefix = inner_init.aligned_prefix();
      let outer_init_prefix = outer_init.aligned_prefix();

      (inner_init, inner_init_prefix, outer_init_prefix)
    });

    Self {
      inner: inner_init,
      inner_init: inner_init_prefix,
      outer_init: outer_init_prefix,
    }
  }

  #[inline]
  fn update(&mut self, data: &[u8]) {
    self.inner.update(data);
  }

  #[inline]
  fn finalize(&self) -> Self::Tag {
    let inner_hash = self.inner.finalize();
    let mut outer = Sha256::from_aligned_prefix(self.outer_init);
    outer.update(&inner_hash);
    HmacSha256Tag::from_bytes(outer.finalize())
  }

  #[inline]
  fn reset(&mut self) {
    self.inner.reset_to_aligned_prefix(self.inner_init);
  }

  /// Oneshot HMAC-SHA256: merges compress calls for small inputs and batches
  /// zeroization under a single compiler fence.
  ///
  /// For data <= 256 B the entire padded inner message is built on the stack
  /// and compressed in one call, eliminating per-call overhead (function-pointer
  /// dispatch, state save/restore) that dominates on fast SHA2-CE cores.
  /// The outer hash is always merged into a single 128-byte (2-block) call.
  #[inline]
  #[allow(clippy::indexing_slicing)] // All indices bounded by prior length checks + fixed-size arrays.
  fn mac(key: &[u8], data: &[u8]) -> Self::Tag {
    if hmac_sha256_oneshot_should_stream() {
      // Sapphire Rapids regresses badly on the fused stack-buffer shape; the
      // public streaming path keeps the same semantics with the measured fast call shape.
      let mut mac = Self::new(key);
      mac.update(data);
      return mac.finalize();
    }

    let mut ipad = [0x36u8; SHA256_BLOCK_SIZE];
    if key.len() > SHA256_BLOCK_SIZE {
      let digest = Sha256::digest(key);
      for (ip, &kb) in ipad[..SHA256_TAG_SIZE].iter_mut().zip(digest.iter()) {
        *ip = kb ^ 0x36;
      }
    } else {
      for (ip, &kb) in ipad[..key.len()].iter_mut().zip(key.iter()) {
        *ip = kb ^ 0x36;
      }
    }

    let total_inner = (SHA256_BLOCK_SIZE as u64).strict_add(data.len() as u64);
    let compress = sha256_dispatch::compress_dispatch().select(len_hint_from_u64(total_inner));
    let total_inner_bits = total_inner.strict_mul(8);

    let mut state = SHA256_H0;

    const INLINE_DATA_MAX: usize = 256;

    if data.len() <= INLINE_DATA_MAX {
      let data_end = SHA256_BLOCK_SIZE.strict_add(data.len());
      let padded = data_end.strict_add(9).strict_add(63).strict_div(64).strict_mul(64);

      macro_rules! compress_inline_inner {
        ($len:expr) => {{
          let mut inner_buf = [0u8; $len];
          inner_buf[..SHA256_BLOCK_SIZE].copy_from_slice(&ipad);
          inner_buf[SHA256_BLOCK_SIZE..data_end].copy_from_slice(data);
          inner_buf[data_end] = 0x80;
          inner_buf[padded.strict_sub(8)..padded].copy_from_slice(&total_inner_bits.to_be_bytes());
          compress(&mut state, &inner_buf);
          ct::zeroize_no_fence(&mut inner_buf);
        }};
      }

      match padded {
        128 => compress_inline_inner!(128),
        192 => compress_inline_inner!(192),
        256 => compress_inline_inner!(256),
        320 => compress_inline_inner!(320),
        384 => compress_inline_inner!(384),
        _ => unreachable!("HMAC-SHA256 inline inner padding is bounded to 128..=384 bytes"),
      }
    } else {
      compress(&mut state, &ipad);

      let full_len = data.len().strict_sub(data.len() % SHA256_BLOCK_SIZE);
      if full_len != 0 {
        compress(&mut state, &data[..full_len]);
      }
      let rest = &data[full_len..];

      let mut inner_buf = [0u8; SHA256_BLOCK_SIZE];
      inner_buf[..rest.len()].copy_from_slice(rest);
      inner_buf[rest.len()] = 0x80;
      if rest.len() >= 56 {
        compress(&mut state, &inner_buf[..SHA256_BLOCK_SIZE]);
        inner_buf[..SHA256_BLOCK_SIZE].fill(0);
      }
      inner_buf[56..SHA256_BLOCK_SIZE].copy_from_slice(&total_inner_bits.to_be_bytes());
      compress(&mut state, &inner_buf[..SHA256_BLOCK_SIZE]);
      ct::zeroize_no_fence(&mut inner_buf);
    }

    let mut outer = [0u8; 128];
    hmac_outer_pad_words(&mut outer, &ipad);
    for (i, &word) in state.iter().enumerate() {
      let off = SHA256_BLOCK_SIZE.strict_add(i.strict_mul(4));
      outer[off..off.strict_add(4)].copy_from_slice(&word.to_be_bytes());
    }
    outer[SHA256_BLOCK_SIZE.strict_add(SHA256_TAG_SIZE)] = 0x80;
    outer[120..128].copy_from_slice(&768u64.to_be_bytes());

    state = SHA256_H0;
    compress(&mut state, &outer);

    let mut tag = [0u8; SHA256_TAG_SIZE];
    for (chunk, &word) in tag.chunks_exact_mut(4).zip(state.iter()) {
      chunk.copy_from_slice(&word.to_be_bytes());
    }

    ct::zeroize_no_fence(&mut ipad);
    ct::zeroize_no_fence(&mut outer);
    for word in state.iter_mut() {
      // SAFETY: word is a valid, aligned, dereferenceable pointer to initialized memory.
      unsafe { core::ptr::write_volatile(word, 0) };
    }
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);

    HmacSha256Tag::from_bytes(tag)
  }

  #[inline]
  fn verify(&self, expected: &Self::Tag) -> Result<(), VerificationError> {
    if self.finalize() == *expected {
      Ok(())
    } else {
      Err(VerificationError::new())
    }
  }
}

impl Drop for HmacSha256 {
  fn drop(&mut self) {
    self.inner_init.zeroize();
    self.outer_init.zeroize();
  }
}

/// HMAC-SHA384 authentication state.
///
/// # Examples
///
/// ```rust
/// use rscrypto::{HmacSha384, Mac};
///
/// let key = b"shared-secret";
/// let data = b"auth message";
///
/// let tag = HmacSha384::mac(key, data);
///
/// let mut mac = HmacSha384::new(key);
/// mac.update(b"auth ");
/// mac.update(b"message");
/// assert!(mac.verify(&tag).is_ok());
/// assert!(HmacSha384::verify_tag(key, data, &tag).is_ok());
/// ```
#[derive(Clone)]
pub struct HmacSha384 {
  inner: Sha384,
  inner_init: Sha384Prefix,
  outer_init: Sha384Prefix,
}

impl core::fmt::Debug for HmacSha384 {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("HmacSha384").finish_non_exhaustive()
  }
}

impl HmacSha384 {
  /// HMAC-SHA384 block size in bytes.
  pub const BLOCK_SIZE: usize = SHA512_FAMILY_BLOCK_SIZE;

  /// HMAC-SHA384 tag size in bytes.
  pub const TAG_SIZE: usize = SHA384_TAG_SIZE;

  /// Compute the HMAC tag of `data` in one shot.
  #[inline]
  #[must_use]
  pub fn mac(key: &[u8], data: &[u8]) -> HmacSha384Tag {
    <Self as Mac>::mac(key, data)
  }

  /// Verify `expected` against the HMAC tag of `data` in constant time.
  #[inline]
  #[must_use = "HMAC verification must be checked; a dropped Result silently accepts a forged tag"]
  pub fn verify_tag(key: &[u8], data: &[u8], expected: &HmacSha384Tag) -> Result<(), VerificationError> {
    <Self as Mac>::verify_tag(key, data, expected)
  }

  #[cfg(any(test, feature = "diag"))]
  pub(crate) fn new_with_compress_for_test(
    key: &[u8],
    compress: crate::hashes::crypto::sha384::kernels::CompressBlocksFn,
  ) -> Self {
    let mut key_block = [0u8; SHA512_FAMILY_BLOCK_SIZE];
    if key.len() > SHA512_FAMILY_BLOCK_SIZE {
      let digest = Sha384::digest(key);
      key_block[..SHA384_TAG_SIZE].copy_from_slice(&digest);
    } else if let Some(dst) = key_block.get_mut(..key.len()) {
      dst.copy_from_slice(key);
    }

    let (inner, inner_init, outer_init) = hmac_prefix_state(&mut key_block, |ipad, opad| {
      let mut inner = Sha384::new_with_compress_for_test(compress);
      inner.update(ipad);
      let inner_init = inner.aligned_prefix();

      let mut outer = Sha384::new_with_compress_for_test(compress);
      outer.update(opad);
      let outer_init = outer.aligned_prefix();

      (inner, inner_init, outer_init)
    });

    Self {
      inner,
      inner_init,
      outer_init,
    }
  }

  #[cfg(any(test, feature = "diag"))]
  pub(crate) fn mac_with_compress_for_test(
    key: &[u8],
    data: &[u8],
    compress: crate::hashes::crypto::sha384::kernels::CompressBlocksFn,
  ) -> [u8; SHA384_TAG_SIZE] {
    let mut mac = Self::new_with_compress_for_test(key, compress);
    mac.update(data);
    mac.finalize().to_bytes()
  }
}

#[cfg(feature = "diag")]
#[must_use]
pub fn diag_hmac_sha384_verify_portable(key: &[u8; SHA384_TAG_SIZE], expected: &[u8; SHA384_TAG_SIZE]) -> bool {
  let compress = crate::hashes::crypto::sha384::kernels::compress_blocks_fn(
    crate::hashes::crypto::sha384::kernels::Sha384KernelId::Portable,
  );
  let tag = HmacSha384::mac_with_compress_for_test(key, b"binsec", compress);
  ct::fixed_eq(&tag, expected)
}

impl Mac for HmacSha384 {
  const TAG_SIZE: usize = SHA384_TAG_SIZE;
  type Tag = HmacSha384Tag;

  fn new(key: &[u8]) -> Self {
    let mut key_block = [0u8; SHA512_FAMILY_BLOCK_SIZE];
    if key.len() > SHA512_FAMILY_BLOCK_SIZE {
      let digest = Sha384::digest(key);
      for (dst, src) in key_block.iter_mut().zip(digest.iter()) {
        *dst = *src;
      }
    } else {
      for (dst, src) in key_block.iter_mut().zip(key.iter()) {
        *dst = *src;
      }
    }

    let (inner_init, inner_init_prefix, outer_init_prefix) = hmac_prefix_state(&mut key_block, |ipad, opad| {
      let mut inner_init = Sha384::new();
      inner_init.update(ipad);

      let mut outer_init = Sha384::new();
      outer_init.update(opad);

      let inner_init_prefix = inner_init.aligned_prefix();
      let outer_init_prefix = outer_init.aligned_prefix();

      (inner_init, inner_init_prefix, outer_init_prefix)
    });

    Self {
      inner: inner_init,
      inner_init: inner_init_prefix,
      outer_init: outer_init_prefix,
    }
  }

  #[inline]
  fn update(&mut self, data: &[u8]) {
    self.inner.update(data);
  }

  #[inline]
  fn finalize(&self) -> Self::Tag {
    let inner_hash = self.inner.finalize();
    let mut outer = Sha384::from_aligned_prefix(self.outer_init);
    outer.update(&inner_hash);
    HmacSha384Tag::from_bytes(outer.finalize())
  }

  #[inline]
  fn reset(&mut self) {
    self.inner.reset_to_aligned_prefix(self.inner_init);
  }

  #[inline]
  #[allow(clippy::indexing_slicing)] // All indices bounded by prior length checks + fixed-size arrays.
  fn mac(key: &[u8], data: &[u8]) -> Self::Tag {
    const INLINE_DATA_MAX: usize = 256;

    let mut ipad = [0x36u8; SHA512_FAMILY_BLOCK_SIZE];
    if key.len() > SHA512_FAMILY_BLOCK_SIZE {
      let digest = Sha384::digest(key);
      for (ip, &kb) in ipad[..SHA384_TAG_SIZE].iter_mut().zip(digest.iter()) {
        *ip = kb ^ 0x36;
      }
    } else {
      for (ip, &kb) in ipad[..key.len()].iter_mut().zip(key.iter()) {
        *ip = kb ^ 0x36;
      }
    }

    let total_inner = (SHA512_FAMILY_BLOCK_SIZE as u64).strict_add(data.len() as u64);
    let compress = sha384_dispatch::compress_dispatch().select(len_hint_from_u64(total_inner));
    let total_inner_bits = total_inner.strict_mul(8);

    let mut state = SHA384_H0;

    if data.len() <= INLINE_DATA_MAX {
      let data_end = SHA512_FAMILY_BLOCK_SIZE.strict_add(data.len());
      let padded = data_end.strict_add(17).strict_add(127).strict_div(128).strict_mul(128);

      macro_rules! compress_inline_inner {
        ($len:expr) => {{
          let mut inner_buf = [0u8; $len];
          inner_buf[..SHA512_FAMILY_BLOCK_SIZE].copy_from_slice(&ipad);
          inner_buf[SHA512_FAMILY_BLOCK_SIZE..data_end].copy_from_slice(data);
          inner_buf[data_end] = 0x80;
          inner_buf[padded.strict_sub(8)..padded].copy_from_slice(&total_inner_bits.to_be_bytes());
          compress(&mut state, &inner_buf);
          ct::zeroize_no_fence(&mut inner_buf);
        }};
      }

      match padded {
        256 => compress_inline_inner!(256),
        384 => compress_inline_inner!(384),
        512 => compress_inline_inner!(512),
        _ => unreachable!("HMAC-SHA384 inline inner padding is bounded to 256..=512 bytes"),
      }
    } else {
      compress(&mut state, &ipad);

      let full_len = data.len().strict_sub(data.len() % SHA512_FAMILY_BLOCK_SIZE);
      if full_len != 0 {
        compress(&mut state, &data[..full_len]);
      }
      let rest = &data[full_len..];

      let mut inner_buf = [0u8; SHA512_FAMILY_BLOCK_SIZE];
      inner_buf[..rest.len()].copy_from_slice(rest);
      inner_buf[rest.len()] = 0x80;
      if rest.len() >= 112 {
        compress(&mut state, &inner_buf[..SHA512_FAMILY_BLOCK_SIZE]);
        inner_buf[..SHA512_FAMILY_BLOCK_SIZE].fill(0);
      }
      inner_buf[120..SHA512_FAMILY_BLOCK_SIZE].copy_from_slice(&total_inner_bits.to_be_bytes());
      compress(&mut state, &inner_buf[..SHA512_FAMILY_BLOCK_SIZE]);
      ct::zeroize_no_fence(&mut inner_buf);
    }

    let mut outer = [0u8; 256];
    hmac_outer_pad_words(&mut outer, &ipad);
    for (i, &word) in state.iter().take(6).enumerate() {
      let off = SHA512_FAMILY_BLOCK_SIZE.strict_add(i.strict_mul(8));
      outer[off..off.strict_add(8)].copy_from_slice(&word.to_be_bytes());
    }
    outer[SHA512_FAMILY_BLOCK_SIZE.strict_add(SHA384_TAG_SIZE)] = 0x80;
    outer[248..256].copy_from_slice(&1408u64.to_be_bytes());

    state = SHA384_H0;
    compress(&mut state, &outer);

    let mut tag = [0u8; SHA384_TAG_SIZE];
    for (chunk, &word) in tag.chunks_exact_mut(8).zip(state.iter()) {
      chunk.copy_from_slice(&word.to_be_bytes());
    }

    ct::zeroize_no_fence(&mut ipad);
    ct::zeroize_no_fence(&mut outer);
    for word in state.iter_mut() {
      // SAFETY: word is a valid, aligned, dereferenceable pointer to initialized memory.
      unsafe { core::ptr::write_volatile(word, 0) };
    }
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);

    HmacSha384Tag::from_bytes(tag)
  }

  #[inline]
  fn verify(&self, expected: &Self::Tag) -> Result<(), VerificationError> {
    if self.finalize() == *expected {
      Ok(())
    } else {
      Err(VerificationError::new())
    }
  }
}

impl Drop for HmacSha384 {
  fn drop(&mut self) {
    self.inner_init.zeroize();
    self.outer_init.zeroize();
  }
}

/// HMAC-SHA512 authentication state.
///
/// # Examples
///
/// ```rust
/// use rscrypto::{HmacSha512, Mac};
///
/// let key = b"shared-secret";
/// let data = b"auth message";
///
/// let tag = HmacSha512::mac(key, data);
///
/// let mut mac = HmacSha512::new(key);
/// mac.update(b"auth ");
/// mac.update(b"message");
/// assert!(mac.verify(&tag).is_ok());
/// assert!(HmacSha512::verify_tag(key, data, &tag).is_ok());
/// ```
#[derive(Clone)]
pub struct HmacSha512 {
  inner: Sha512,
  inner_init: Sha512Prefix,
  outer_init: Sha512Prefix,
}

impl core::fmt::Debug for HmacSha512 {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("HmacSha512").finish_non_exhaustive()
  }
}

impl HmacSha512 {
  /// HMAC-SHA512 block size in bytes.
  pub const BLOCK_SIZE: usize = SHA512_FAMILY_BLOCK_SIZE;

  /// HMAC-SHA512 tag size in bytes.
  pub const TAG_SIZE: usize = SHA512_TAG_SIZE;

  /// Compute the HMAC tag of `data` in one shot.
  #[inline]
  #[must_use]
  pub fn mac(key: &[u8], data: &[u8]) -> HmacSha512Tag {
    <Self as Mac>::mac(key, data)
  }

  /// Verify `expected` against the HMAC tag of `data` in constant time.
  #[inline]
  #[must_use = "HMAC verification must be checked; a dropped Result silently accepts a forged tag"]
  pub fn verify_tag(key: &[u8], data: &[u8], expected: &HmacSha512Tag) -> Result<(), VerificationError> {
    <Self as Mac>::verify_tag(key, data, expected)
  }

  #[cfg(any(test, feature = "diag"))]
  pub(crate) fn new_with_compress_for_test(
    key: &[u8],
    compress: crate::hashes::crypto::sha512::kernels::CompressBlocksFn,
  ) -> Self {
    let mut key_block = [0u8; SHA512_FAMILY_BLOCK_SIZE];
    if key.len() > SHA512_FAMILY_BLOCK_SIZE {
      let digest = Sha512::digest(key);
      key_block[..SHA512_TAG_SIZE].copy_from_slice(&digest);
    } else if let Some(dst) = key_block.get_mut(..key.len()) {
      dst.copy_from_slice(key);
    }

    let (inner, inner_init, outer_init) = hmac_prefix_state(&mut key_block, |ipad, opad| {
      let mut inner = Sha512::new_with_compress_for_test(compress);
      inner.update(ipad);
      let inner_init = inner.aligned_prefix();

      let mut outer = Sha512::new_with_compress_for_test(compress);
      outer.update(opad);
      let outer_init = outer.aligned_prefix();

      (inner, inner_init, outer_init)
    });

    Self {
      inner,
      inner_init,
      outer_init,
    }
  }

  #[cfg(any(test, feature = "diag"))]
  pub(crate) fn mac_with_compress_for_test(
    key: &[u8],
    data: &[u8],
    compress: crate::hashes::crypto::sha512::kernels::CompressBlocksFn,
  ) -> [u8; SHA512_TAG_SIZE] {
    let mut mac = Self::new_with_compress_for_test(key, compress);
    mac.update(data);
    mac.finalize().to_bytes()
  }
}

#[cfg(feature = "diag")]
#[must_use]
pub fn diag_hmac_sha512_verify_portable(key: &[u8; SHA512_TAG_SIZE], expected: &[u8; SHA512_TAG_SIZE]) -> bool {
  let compress = crate::hashes::crypto::sha512::kernels::compress_blocks_fn(
    crate::hashes::crypto::sha512::kernels::Sha512KernelId::Portable,
  );
  let tag = HmacSha512::mac_with_compress_for_test(key, b"binsec", compress);
  ct::fixed_eq(&tag, expected)
}

impl Mac for HmacSha512 {
  const TAG_SIZE: usize = SHA512_TAG_SIZE;
  type Tag = HmacSha512Tag;

  fn new(key: &[u8]) -> Self {
    let mut key_block = [0u8; SHA512_FAMILY_BLOCK_SIZE];
    if key.len() > SHA512_FAMILY_BLOCK_SIZE {
      let digest = Sha512::digest(key);
      for (dst, src) in key_block.iter_mut().zip(digest.iter()) {
        *dst = *src;
      }
    } else {
      for (dst, src) in key_block.iter_mut().zip(key.iter()) {
        *dst = *src;
      }
    }

    let (inner_init, inner_init_prefix, outer_init_prefix) = hmac_prefix_state(&mut key_block, |ipad, opad| {
      let mut inner_init = Sha512::new();
      inner_init.update(ipad);

      let mut outer_init = Sha512::new();
      outer_init.update(opad);

      let inner_init_prefix = inner_init.aligned_prefix();
      let outer_init_prefix = outer_init.aligned_prefix();

      (inner_init, inner_init_prefix, outer_init_prefix)
    });

    Self {
      inner: inner_init,
      inner_init: inner_init_prefix,
      outer_init: outer_init_prefix,
    }
  }

  #[inline]
  fn update(&mut self, data: &[u8]) {
    self.inner.update(data);
  }

  #[inline]
  fn finalize(&self) -> Self::Tag {
    let inner_hash = self.inner.finalize();
    let mut outer = Sha512::from_aligned_prefix(self.outer_init);
    outer.update(&inner_hash);
    HmacSha512Tag::from_bytes(outer.finalize())
  }

  #[inline]
  fn reset(&mut self) {
    self.inner.reset_to_aligned_prefix(self.inner_init);
  }

  #[inline]
  #[allow(clippy::indexing_slicing)] // All indices bounded by prior length checks + fixed-size arrays.
  fn mac(key: &[u8], data: &[u8]) -> Self::Tag {
    const INLINE_DATA_MAX: usize = 256;

    let mut ipad = [0x36u8; SHA512_FAMILY_BLOCK_SIZE];
    if key.len() > SHA512_FAMILY_BLOCK_SIZE {
      let digest = Sha512::digest(key);
      for (ip, &kb) in ipad[..SHA512_TAG_SIZE].iter_mut().zip(digest.iter()) {
        *ip = kb ^ 0x36;
      }
    } else {
      for (ip, &kb) in ipad[..key.len()].iter_mut().zip(key.iter()) {
        *ip = kb ^ 0x36;
      }
    }

    let total_inner = (SHA512_FAMILY_BLOCK_SIZE as u64).strict_add(data.len() as u64);
    let compress = sha512_dispatch::compress_dispatch().select(len_hint_from_u64(total_inner));
    let total_inner_bits = total_inner.strict_mul(8);

    let mut state = SHA512_H0;

    if data.len() <= INLINE_DATA_MAX {
      let data_end = SHA512_FAMILY_BLOCK_SIZE.strict_add(data.len());
      let padded = data_end.strict_add(17).strict_add(127).strict_div(128).strict_mul(128);

      macro_rules! compress_inline_inner {
        ($len:expr) => {{
          let mut inner_buf = [0u8; $len];
          inner_buf[..SHA512_FAMILY_BLOCK_SIZE].copy_from_slice(&ipad);
          inner_buf[SHA512_FAMILY_BLOCK_SIZE..data_end].copy_from_slice(data);
          inner_buf[data_end] = 0x80;
          inner_buf[padded.strict_sub(8)..padded].copy_from_slice(&total_inner_bits.to_be_bytes());
          compress(&mut state, &inner_buf);
          ct::zeroize_no_fence(&mut inner_buf);
        }};
      }

      match padded {
        256 => compress_inline_inner!(256),
        384 => compress_inline_inner!(384),
        512 => compress_inline_inner!(512),
        _ => unreachable!("HMAC-SHA512 inline inner padding is bounded to 256..=512 bytes"),
      }
    } else {
      compress(&mut state, &ipad);

      let full_len = data.len().strict_sub(data.len() % SHA512_FAMILY_BLOCK_SIZE);
      if full_len != 0 {
        compress(&mut state, &data[..full_len]);
      }
      let rest = &data[full_len..];

      let mut inner_buf = [0u8; SHA512_FAMILY_BLOCK_SIZE];
      inner_buf[..rest.len()].copy_from_slice(rest);
      inner_buf[rest.len()] = 0x80;
      if rest.len() >= 112 {
        compress(&mut state, &inner_buf[..SHA512_FAMILY_BLOCK_SIZE]);
        inner_buf[..SHA512_FAMILY_BLOCK_SIZE].fill(0);
      }
      inner_buf[120..SHA512_FAMILY_BLOCK_SIZE].copy_from_slice(&total_inner_bits.to_be_bytes());
      compress(&mut state, &inner_buf[..SHA512_FAMILY_BLOCK_SIZE]);
      ct::zeroize_no_fence(&mut inner_buf);
    }

    let mut outer = [0u8; 256];
    hmac_outer_pad_words(&mut outer, &ipad);
    for (i, &word) in state.iter().enumerate() {
      let off = SHA512_FAMILY_BLOCK_SIZE.strict_add(i.strict_mul(8));
      outer[off..off.strict_add(8)].copy_from_slice(&word.to_be_bytes());
    }
    outer[SHA512_FAMILY_BLOCK_SIZE.strict_add(SHA512_TAG_SIZE)] = 0x80;
    outer[248..256].copy_from_slice(&1536u64.to_be_bytes());

    state = SHA512_H0;
    compress(&mut state, &outer);

    let mut tag = [0u8; SHA512_TAG_SIZE];
    for (chunk, &word) in tag.chunks_exact_mut(8).zip(state.iter()) {
      chunk.copy_from_slice(&word.to_be_bytes());
    }

    ct::zeroize_no_fence(&mut ipad);
    ct::zeroize_no_fence(&mut outer);
    for word in state.iter_mut() {
      // SAFETY: word is a valid, aligned, dereferenceable pointer to initialized memory.
      unsafe { core::ptr::write_volatile(word, 0) };
    }
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);

    HmacSha512Tag::from_bytes(tag)
  }

  #[inline]
  fn verify(&self, expected: &Self::Tag) -> Result<(), VerificationError> {
    if self.finalize() == *expected {
      Ok(())
    } else {
      Err(VerificationError::new())
    }
  }
}

impl Drop for HmacSha512 {
  fn drop(&mut self) {
    self.inner_init.zeroize();
    self.outer_init.zeroize();
  }
}

#[cfg(test)]
mod tests {
  use alloc::vec::Vec;

  use hmac::{Hmac, Mac as _, digest::KeyInit};

  use super::*;
  use crate::hashes::crypto::{
    sha384::kernels::{
      ALL as SHA384_KERNELS, Sha384KernelId, compress_blocks_fn as sha384_compress_blocks_fn,
      required_caps as sha384_required_caps,
    },
    sha512::kernels::{
      ALL as SHA512_KERNELS, Sha512KernelId, compress_blocks_fn as sha512_compress_blocks_fn,
      required_caps as sha512_required_caps,
    },
  };

  type RustCryptoHmacSha384 = Hmac<sha2::Sha384>;
  type RustCryptoHmacSha512 = Hmac<sha2::Sha512>;

  fn pattern(len: usize, mul: u8, add: u8) -> Vec<u8> {
    (0..len)
      .map(|i| {
        (i as u8)
          .wrapping_mul(mul)
          .wrapping_add(((i >> 3) as u8).wrapping_add(add))
      })
      .collect()
  }

  fn oracle_hmac_sha384(key: &[u8], data: &[u8]) -> [u8; SHA384_TAG_SIZE] {
    let mut mac = RustCryptoHmacSha384::new_from_slice(key).unwrap();
    mac.update(data);
    let bytes = mac.finalize().into_bytes();
    let mut tag = [0u8; SHA384_TAG_SIZE];
    tag.copy_from_slice(&bytes);
    tag
  }

  fn oracle_hmac_sha512(key: &[u8], data: &[u8]) -> [u8; SHA512_TAG_SIZE] {
    let mut mac = RustCryptoHmacSha512::new_from_slice(key).unwrap();
    mac.update(data);
    let bytes = mac.finalize().into_bytes();
    let mut tag = [0u8; SHA512_TAG_SIZE];
    tag.copy_from_slice(&bytes);
    tag
  }

  #[test]
  #[cfg(target_arch = "x86_64")]
  fn hmac_sha256_sapphire_rapids_oneshot_policy_uses_streaming() {
    let spr = crate::platform::caps::x86::INTEL_SAPPHIRE_RAPIDS;
    assert!(hmac_sha256_oneshot_prefers_streaming(spr));

    let amd_zen5 = crate::platform::caps::x86::AMD | crate::platform::caps::x86::AMD_ZEN5;
    assert!(!hmac_sha256_oneshot_prefers_streaming(amd_zen5));

    assert!(!hmac_sha256_oneshot_prefers_streaming(crate::platform::Caps::NONE));
  }

  fn assert_hmac_sha384_kernel(id: Sha384KernelId) {
    let compress = sha384_compress_blocks_fn(id);
    let cases = [
      (0usize, 0usize, 1usize),
      (1, 1, 1),
      (16, 31, 7),
      (48, 127, 31),
      (80, 128, 64),
      (160, 129, 65),
      (256, 255, 128),
      (300, 1024, 257),
    ];

    for &(key_len, data_len, chunk_len) in &cases {
      let key = pattern(key_len, 17, 3);
      let data = pattern(data_len, 29, 11);
      let expected = oracle_hmac_sha384(&key, &data);

      assert_eq!(
        HmacSha384::mac(&key, &data).as_bytes(),
        &expected,
        "sha384 public oneshot mismatch kernel={} key_len={} data_len={}",
        id.as_str(),
        key_len,
        data_len
      );
      assert_eq!(
        HmacSha384::mac_with_compress_for_test(&key, &data, compress),
        expected,
        "sha384 forced oneshot mismatch kernel={} key_len={} data_len={}",
        id.as_str(),
        key_len,
        data_len
      );

      let mut streaming = HmacSha384::new_with_compress_for_test(&key, compress);
      for chunk in data.chunks(chunk_len) {
        streaming.update(chunk);
      }
      assert_eq!(
        streaming.finalize().as_bytes(),
        &expected,
        "sha384 forced streaming mismatch kernel={} key_len={} data_len={} chunk_len={}",
        id.as_str(),
        key_len,
        data_len,
        chunk_len
      );

      streaming.reset();
      streaming.update(&data);
      assert_eq!(
        streaming.finalize().as_bytes(),
        &expected,
        "sha384 forced reset mismatch kernel={} key_len={} data_len={}",
        id.as_str(),
        key_len,
        data_len
      );
    }
  }

  fn assert_hmac_sha512_kernel(id: Sha512KernelId) {
    let compress = sha512_compress_blocks_fn(id);
    let cases = [
      (0usize, 0usize, 1usize),
      (1, 1, 1),
      (32, 63, 7),
      (64, 127, 31),
      (96, 128, 64),
      (192, 129, 65),
      (256, 255, 128),
      (320, 1024, 257),
    ];

    for &(key_len, data_len, chunk_len) in &cases {
      let key = pattern(key_len, 23, 7);
      let data = pattern(data_len, 37, 13);
      let expected = oracle_hmac_sha512(&key, &data);

      assert_eq!(
        HmacSha512::mac(&key, &data).as_bytes(),
        &expected,
        "sha512 public oneshot mismatch kernel={} key_len={} data_len={}",
        id.as_str(),
        key_len,
        data_len
      );
      assert_eq!(
        HmacSha512::mac_with_compress_for_test(&key, &data, compress),
        expected,
        "sha512 forced oneshot mismatch kernel={} key_len={} data_len={}",
        id.as_str(),
        key_len,
        data_len
      );

      let mut streaming = HmacSha512::new_with_compress_for_test(&key, compress);
      for chunk in data.chunks(chunk_len) {
        streaming.update(chunk);
      }
      assert_eq!(
        streaming.finalize().as_bytes(),
        &expected,
        "sha512 forced streaming mismatch kernel={} key_len={} data_len={} chunk_len={}",
        id.as_str(),
        key_len,
        data_len,
        chunk_len
      );

      streaming.reset();
      streaming.update(&data);
      assert_eq!(
        streaming.finalize().as_bytes(),
        &expected,
        "sha512 forced reset mismatch kernel={} key_len={} data_len={}",
        id.as_str(),
        key_len,
        data_len
      );
    }
  }

  #[test]
  fn hmac_sha384_forced_kernels_match_oracle() {
    let caps = crate::platform::caps();
    for &id in SHA384_KERNELS {
      if caps.has(sha384_required_caps(id)) {
        assert_hmac_sha384_kernel(id);
      }
    }
  }

  #[test]
  fn hmac_sha512_forced_kernels_match_oracle() {
    let caps = crate::platform::caps();
    for &id in SHA512_KERNELS {
      if caps.has(sha512_required_caps(id)) {
        assert_hmac_sha512_kernel(id);
      }
    }
  }
}
