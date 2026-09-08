//! Public tag contracts shared by independently enabled HMAC families.

#![cfg(any(feature = "hmac", feature = "hmac-sha3"))]

macro_rules! tag_contract {
  ($test:ident, $tag:path, $len:expr, $label:literal) => {
    #[test]
    fn $test() {
      type Tag = $tag;
      let bytes = [0x5au8; $len];
      let tag = Tag::from_bytes(bytes);
      assert_eq!(Tag::LENGTH, $len);
      assert_eq!(tag.to_bytes(), bytes);
      assert_eq!(tag.into_bytes(), bytes);
      assert_eq!(*tag.as_bytes(), bytes);
      assert_eq!(tag.as_slice(), &bytes);
      assert_eq!(<Tag as AsRef<[u8]>>::as_ref(&tag), &bytes);
      assert_eq!(<Tag as AsRef<[u8; $len]>>::as_ref(&tag), &bytes);
      assert_eq!(<[u8; $len]>::from(Tag::from(bytes)), bytes);
      assert_eq!(Tag::try_from(bytes.as_slice()).unwrap().to_bytes(), bytes);
      assert!(Tag::try_from(&bytes[..Tag::LENGTH.strict_sub(1)]).is_err());
      assert!(Tag::try_from([0u8; Tag::LENGTH.strict_add(1)].as_slice()).is_err());
      assert_eq!(Tag::default().to_bytes(), [0u8; $len]);
      assert!(tag.ct_eq(&Tag::from_bytes(bytes)).declassify());
      for i in 0..$len {
        let mut altered = bytes;
        altered[i] ^= 1;
        assert!(!tag.ct_eq(&Tag::from_bytes(altered)).declassify());
      }
      assert_eq!(format!("{tag:?}"), format!("{}({})", $label, "5a".repeat($len)));
      use core::hash::{Hash as _, Hasher as _};
      let mut tag_hash = std::hash::DefaultHasher::new();
      let mut bytes_hash = std::hash::DefaultHasher::new();
      tag.hash(&mut tag_hash);
      bytes.hash(&mut bytes_hash);
      assert_eq!(tag_hash.finish(), bytes_hash.finish());

      #[cfg(feature = "serde")]
      {
        use serde::Deserialize as _;
        use serde::de::value::{BorrowedBytesDeserializer, Error};

        let json = serde_json::to_string(&tag).unwrap();
        assert_eq!(json, serde_json::to_string(bytes.as_slice()).unwrap());
        assert_eq!(serde_json::from_str::<Tag>(&json).unwrap().to_bytes(), bytes);
        assert!(
          serde_json::from_str::<Tag>(&serde_json::to_string(&bytes[..Tag::LENGTH.strict_sub(1)]).unwrap()).is_err()
        );
        assert!(
          serde_json::from_str::<Tag>(&serde_json::to_string([0u8; Tag::LENGTH.strict_add(1)].as_slice()).unwrap())
            .is_err()
        );
        assert_eq!(
          Tag::deserialize(BorrowedBytesDeserializer::<Error>::new(&bytes))
            .unwrap()
            .to_bytes(),
          bytes
        );
        assert!(
          Tag::deserialize(BorrowedBytesDeserializer::<Error>::new(
            &bytes[..Tag::LENGTH.strict_sub(1)]
          ))
          .is_err()
        );
        assert!(
          Tag::deserialize(BorrowedBytesDeserializer::<Error>::new(
            &[0u8; Tag::LENGTH.strict_add(1)]
          ))
          .is_err()
        );
      }
    }
  };
}

#[cfg(feature = "hmac")]
tag_contract!(sha256, rscrypto::HmacSha256Tag, 32, "HmacSha256Tag");
#[cfg(feature = "hmac")]
tag_contract!(sha384, rscrypto::HmacSha384Tag, 48, "HmacSha384Tag");
#[cfg(feature = "hmac")]
tag_contract!(sha512, rscrypto::HmacSha512Tag, 64, "HmacSha512Tag");
#[cfg(feature = "hmac-sha3")]
tag_contract!(sha3_224, rscrypto::HmacSha3_224Tag, 28, "HmacSha3_224Tag");
#[cfg(feature = "hmac-sha3")]
tag_contract!(sha3_256, rscrypto::HmacSha3_256Tag, 32, "HmacSha3_256Tag");
#[cfg(feature = "hmac-sha3")]
tag_contract!(sha3_384, rscrypto::HmacSha3_384Tag, 48, "HmacSha3_384Tag");
#[cfg(feature = "hmac-sha3")]
tag_contract!(sha3_512, rscrypto::HmacSha3_512Tag, 64, "HmacSha3_512Tag");

#[cfg(all(feature = "hmac", feature = "hmac-sha3"))]
#[test]
fn equal_length_tags_remain_distinct_types() {
  assert_ne!(
    core::any::TypeId::of::<rscrypto::HmacSha256Tag>(),
    core::any::TypeId::of::<rscrypto::HmacSha3_256Tag>()
  );
  assert_ne!(
    core::any::TypeId::of::<rscrypto::HmacSha384Tag>(),
    core::any::TypeId::of::<rscrypto::HmacSha3_384Tag>()
  );
  assert_ne!(
    core::any::TypeId::of::<rscrypto::HmacSha512Tag>(),
    core::any::TypeId::of::<rscrypto::HmacSha3_512Tag>()
  );
}
