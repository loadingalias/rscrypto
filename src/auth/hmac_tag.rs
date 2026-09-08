//! Shared representation and conversions for the distinct HMAC tag types.

macro_rules! define_hmac_tag_type {
  ($name:ident, $len:expr, $doc:literal) => {
    #[doc = $doc]
    #[derive(Clone, Copy)]
    pub struct $name([u8; Self::LENGTH]);

    impl core::hash::Hash for $name {
      #[inline]
      fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        core::hash::Hash::hash(&self.0, state);
      }
    }

    impl $name {
      /// Tag length in bytes.
      pub const LENGTH: usize = $len;

      /// Compare two tags without exposing a branchable boolean.
      #[inline]
      pub fn ct_eq(&self, other: &Self) -> $crate::traits::ct::CtDecision {
        $crate::traits::ct::fixed_eq(&self.0, &other.0)
      }

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

pub(super) use define_hmac_tag_type;
