//! Serde preserves strict import validation and explicit secret export policy.

use super::*;

macro_rules! encoded {
  ($type:ident, $bytes:expr) => {
    impl serde::Serialize for $type {
      fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let bytes: &[u8] = $bytes(self);
        serializer.serialize_bytes(bytes)
      }
    }

    impl<'de> serde::Deserialize<'de> for $type {
      fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Visitor;
        impl<'de> serde::de::Visitor<'de> for Visitor {
          type Value = $type;
          fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "a canonical {}-byte {}", $type::LENGTH, stringify!($type))
          }
          fn visit_bytes<E: serde::de::Error>(self, bytes: &[u8]) -> Result<Self::Value, E> {
            $type::try_from_slice(bytes).map_err(E::custom)
          }
          fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
            // Also guards partially read secret sequences when deserialization fails.
            let mut bytes = ZeroizingBytes::<{ $type::LENGTH }>::zeroed();
            for (i, byte) in bytes.as_mut_array().iter_mut().enumerate() {
              *byte = seq
                .next_element()?
                .ok_or_else(|| serde::de::Error::invalid_length(i, &self))?;
            }
            if seq.next_element::<u8>()?.is_some() {
              return Err(serde::de::Error::invalid_length($type::LENGTH.strict_add(1), &self));
            }
            self.visit_bytes(bytes.as_array())
          }
        }
        deserializer.deserialize_bytes(Visitor)
      }
    }
  };
}

encoded!(MlDsa44PublicKey, MlDsa44PublicKey::as_ref);
encoded!(MlDsa65PublicKey, MlDsa65PublicKey::as_ref);
encoded!(MlDsa87PublicKey, MlDsa87PublicKey::as_ref);
encoded!(MlDsa44Signature, MlDsa44Signature::as_ref);
encoded!(MlDsa65Signature, MlDsa65Signature::as_ref);
encoded!(MlDsa87Signature, MlDsa87Signature::as_ref);

#[cfg(feature = "serde-secrets")]
mod secrets {
  use super::*;
  encoded!(MlDsa44SecretKey, MlDsa44SecretKey::encoded_secret);
  encoded!(MlDsa65SecretKey, MlDsa65SecretKey::encoded_secret);
  encoded!(MlDsa87SecretKey, MlDsa87SecretKey::encoded_secret);
}
