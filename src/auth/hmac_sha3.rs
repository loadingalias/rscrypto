//! HMAC-SHA3 family (RFC 2104, FIPS 198-1, FIPS 202).

#![allow(clippy::indexing_slicing)] // HMAC pads are fixed-size arrays bounded by rate constants.

use crate::{
  hashes::crypto::keccak::KeccakCore,
  traits::{Mac, VerificationError, ct},
};

const SHA3_224_RATE: usize = 144;
const SHA3_224_TAG_SIZE: usize = 28;
const SHA3_256_RATE: usize = 136;
const SHA3_256_TAG_SIZE: usize = 32;
const SHA3_384_RATE: usize = 104;
const SHA3_384_TAG_SIZE: usize = 48;
const SHA3_512_RATE: usize = 72;
const SHA3_512_TAG_SIZE: usize = 64;
const SHA3_PAD: u8 = 0x06;

macro_rules! define_hmac_sha3_tag_type {
  ($name:ident, $len:expr, $doc:literal) => {
    #[doc = $doc]
    #[derive(Clone, Copy)]
    pub struct $name([u8; Self::LENGTH]);

    impl PartialEq for $name {
      #[inline]
      fn eq(&self, other: &Self) -> bool {
        ct::constant_time_eq(&self.0, &other.0)
      }
    }

    impl PartialEq<[u8; $len]> for $name {
      #[inline]
      fn eq(&self, other: &[u8; $len]) -> bool {
        ct::constant_time_eq(&self.0, other)
      }
    }

    impl PartialEq<$name> for [u8; $len] {
      #[inline]
      fn eq(&self, other: &$name) -> bool {
        ct::constant_time_eq(self, &other.0)
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

    impl crate::traits::ConstantTimeEq for $name {
      #[inline]
      fn ct_eq(&self, other: &Self) -> bool {
        ct::constant_time_eq(&self.0, &other.0)
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

define_hmac_sha3_tag_type!(HmacSha3_224Tag, SHA3_224_TAG_SIZE, "HMAC-SHA3-224 authentication tag.");
define_hmac_sha3_tag_type!(HmacSha3_256Tag, SHA3_256_TAG_SIZE, "HMAC-SHA3-256 authentication tag.");
define_hmac_sha3_tag_type!(HmacSha3_384Tag, SHA3_384_TAG_SIZE, "HMAC-SHA3-384 authentication tag.");
define_hmac_sha3_tag_type!(HmacSha3_512Tag, SHA3_512_TAG_SIZE, "HMAC-SHA3-512 authentication tag.");

#[inline]
fn sha3_digest<const RATE: usize, const OUT: usize>(data: &[u8]) -> [u8; OUT] {
  let mut state = KeccakCore::<RATE>::default();
  state.update(data);
  sha3_finalize(&state)
}

#[inline]
fn sha3_finalize<const RATE: usize, const OUT: usize>(state: &KeccakCore<RATE>) -> [u8; OUT] {
  let mut out = [0u8; OUT];
  state.finalize_into_fixed(SHA3_PAD, &mut out);
  out
}

#[inline]
fn hmac_sha3_prefix_state<const RATE: usize>(
  key_block: &mut [u8; RATE],
) -> (KeccakCore<RATE>, KeccakCore<RATE>, KeccakCore<RATE>) {
  let mut ipad = [0x36u8; RATE];
  let mut opad = [0x5cu8; RATE];
  for ((ipad_byte, opad_byte), key_byte) in ipad.iter_mut().zip(opad.iter_mut()).zip(key_block.iter().copied()) {
    *ipad_byte ^= key_byte;
    *opad_byte ^= key_byte;
  }

  let mut inner = KeccakCore::<RATE>::default();
  inner.update(&ipad);
  let inner_init = inner.clone();

  let mut outer_init = KeccakCore::<RATE>::default();
  outer_init.update(&opad);

  ct::zeroize_no_fence(key_block);
  ct::zeroize_no_fence(&mut ipad);
  ct::zeroize_no_fence(&mut opad);
  core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);

  (inner, inner_init, outer_init)
}

macro_rules! define_hmac_sha3 {
  ($name:ident, $tag:ident, $rate:expr, $tag_size:expr, $label:literal) => {
    #[doc = concat!("HMAC-", $label, " authentication state.")]
    #[derive(Clone)]
    pub struct $name {
      inner: KeccakCore<$rate>,
      inner_init: KeccakCore<$rate>,
      outer_init: KeccakCore<$rate>,
    }

    impl core::fmt::Debug for $name {
      fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct(stringify!($name)).finish_non_exhaustive()
      }
    }

    impl $name {
      #[doc = concat!("HMAC-", $label, " block size in bytes.")]
      pub const BLOCK_SIZE: usize = $rate;

      #[doc = concat!("HMAC-", $label, " tag size in bytes.")]
      pub const TAG_SIZE: usize = $tag_size;

      /// Compute the HMAC tag of `data` in one shot.
      #[inline]
      #[must_use]
      pub fn mac(key: &[u8], data: &[u8]) -> $tag {
        <Self as Mac>::mac(key, data)
      }

      /// Verify `expected` against the HMAC tag of `data` in constant time.
      #[inline]
      #[must_use = "HMAC verification must be checked; a dropped Result silently accepts a forged tag"]
      pub fn verify_tag(key: &[u8], data: &[u8], expected: &$tag) -> Result<(), VerificationError> {
        <Self as Mac>::verify_tag(key, data, expected)
      }
    }

    impl Mac for $name {
      const TAG_SIZE: usize = $tag_size;
      type Tag = $tag;

      fn new(key: &[u8]) -> Self {
        let mut key_block = [0u8; $rate];
        if key.len() > $rate {
          let digest = sha3_digest::<$rate, $tag_size>(key);
          key_block[..$tag_size].copy_from_slice(&digest);
        } else {
          key_block[..key.len()].copy_from_slice(key);
        }

        let (inner, inner_init, outer_init) = hmac_sha3_prefix_state(&mut key_block);

        Self {
          inner,
          inner_init,
          outer_init,
        }
      }

      #[inline]
      fn update(&mut self, data: &[u8]) {
        self.inner.update(data);
      }

      #[inline]
      fn finalize(&self) -> Self::Tag {
        let inner_hash = sha3_finalize::<$rate, $tag_size>(&self.inner);
        let mut outer = self.outer_init.clone();
        outer.update(&inner_hash);
        $tag::from_bytes(sha3_finalize::<$rate, $tag_size>(&outer))
      }

      #[inline]
      fn reset(&mut self) {
        self.inner = self.inner_init.clone();
      }
    }
  };
}

define_hmac_sha3!(
  HmacSha3_224,
  HmacSha3_224Tag,
  SHA3_224_RATE,
  SHA3_224_TAG_SIZE,
  "SHA3-224"
);
define_hmac_sha3!(
  HmacSha3_256,
  HmacSha3_256Tag,
  SHA3_256_RATE,
  SHA3_256_TAG_SIZE,
  "SHA3-256"
);
define_hmac_sha3!(
  HmacSha3_384,
  HmacSha3_384Tag,
  SHA3_384_RATE,
  SHA3_384_TAG_SIZE,
  "SHA3-384"
);
define_hmac_sha3!(
  HmacSha3_512,
  HmacSha3_512Tag,
  SHA3_512_RATE,
  SHA3_512_TAG_SIZE,
  "SHA3-512"
);
