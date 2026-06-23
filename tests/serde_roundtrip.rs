//! Serde round-trip tests for byte-wrapper serde implementations.
//!
//! Each type is serialized to JSON and deserialized back, asserting byte-level
//! equality. This catches encoding bugs that would silently corrupt tags,
//! nonces, public keys, signatures, and explicitly opted-in secrets.

#![cfg(feature = "serde")]

macro_rules! serde_roundtrip {
  ($name:ident, $ty:ty, $len:expr) => {
    #[test]
    fn $name() {
      let bytes = {
        let mut b = [0u8; $len];
        for (i, v) in b.iter_mut().enumerate() {
          *v = (i as u8).wrapping_mul(0x37).wrapping_add(0x11);
        }
        b
      };
      let original = <$ty>::from_bytes(bytes);
      let json = serde_json::to_string(&original).expect("serialize");
      let recovered: $ty = serde_json::from_str(&json).expect("deserialize");
      assert_eq!(original.as_bytes(), recovered.as_bytes());
    }
  };
}

#[cfg(feature = "aead")]
mod aead_nonces {
  use rscrypto::aead::{Nonce96, Nonce128, Nonce192, Nonce256};

  serde_roundtrip!(nonce96, Nonce96, 12);
  serde_roundtrip!(nonce128, Nonce128, 16);
  serde_roundtrip!(nonce192, Nonce192, 24);
  serde_roundtrip!(nonce256, Nonce256, 32);
}

#[cfg(feature = "chacha20poly1305")]
mod chacha20poly1305_serde {
  #[cfg(feature = "serde-secrets")]
  use rscrypto::aead::ChaCha20Poly1305Key;
  use rscrypto::aead::ChaCha20Poly1305Tag;

  #[cfg(feature = "serde-secrets")]
  serde_roundtrip!(key, ChaCha20Poly1305Key, 32);
  serde_roundtrip!(tag, ChaCha20Poly1305Tag, 16);
}

#[cfg(feature = "xchacha20poly1305")]
mod xchacha20poly1305_serde {
  #[cfg(feature = "serde-secrets")]
  use rscrypto::aead::XChaCha20Poly1305Key;
  use rscrypto::aead::XChaCha20Poly1305Tag;

  #[cfg(feature = "serde-secrets")]
  serde_roundtrip!(key, XChaCha20Poly1305Key, 32);
  serde_roundtrip!(tag, XChaCha20Poly1305Tag, 16);
}

#[cfg(feature = "aes-gcm")]
mod aes128gcm_serde {
  #[cfg(feature = "serde-secrets")]
  use rscrypto::aead::Aes128GcmKey;
  use rscrypto::aead::Aes128GcmTag;

  #[cfg(feature = "serde-secrets")]
  serde_roundtrip!(key, Aes128GcmKey, 16);
  serde_roundtrip!(tag, Aes128GcmTag, 16);
}

#[cfg(feature = "aes-gcm")]
mod aes256gcm_serde {
  #[cfg(feature = "serde-secrets")]
  use rscrypto::aead::Aes256GcmKey;
  use rscrypto::aead::Aes256GcmTag;

  #[cfg(feature = "serde-secrets")]
  serde_roundtrip!(key, Aes256GcmKey, 32);
  serde_roundtrip!(tag, Aes256GcmTag, 16);
}

#[cfg(feature = "aes-gcm-siv")]
mod aes128gcmsiv_serde {
  #[cfg(feature = "serde-secrets")]
  use rscrypto::aead::Aes128GcmSivKey;
  use rscrypto::aead::Aes128GcmSivTag;

  #[cfg(feature = "serde-secrets")]
  serde_roundtrip!(key, Aes128GcmSivKey, 16);
  serde_roundtrip!(tag, Aes128GcmSivTag, 16);
}

#[cfg(feature = "aes-gcm-siv")]
mod aes256gcmsiv_serde {
  #[cfg(feature = "serde-secrets")]
  use rscrypto::aead::Aes256GcmSivKey;
  use rscrypto::aead::Aes256GcmSivTag;

  #[cfg(feature = "serde-secrets")]
  serde_roundtrip!(key, Aes256GcmSivKey, 32);
  serde_roundtrip!(tag, Aes256GcmSivTag, 16);
}

#[cfg(feature = "ascon-aead")]
mod ascon128_serde {
  #[cfg(feature = "serde-secrets")]
  use rscrypto::aead::AsconAead128Key;
  use rscrypto::aead::AsconAead128Tag;

  #[cfg(feature = "serde-secrets")]
  serde_roundtrip!(key, AsconAead128Key, 16);
  serde_roundtrip!(tag, AsconAead128Tag, 16);
}

#[cfg(feature = "aegis256")]
mod aegis256_serde {
  #[cfg(feature = "serde-secrets")]
  use rscrypto::aead::Aegis256Key;
  use rscrypto::aead::Aegis256Tag;

  #[cfg(feature = "serde-secrets")]
  serde_roundtrip!(key, Aegis256Key, 32);
  serde_roundtrip!(tag, Aegis256Tag, 16);
}

#[cfg(feature = "ed25519")]
mod ed25519_serde {
  #[cfg(feature = "serde-secrets")]
  use rscrypto::Ed25519SecretKey;
  use rscrypto::{Ed25519PublicKey, Ed25519Signature};

  #[cfg(feature = "serde-secrets")]
  serde_roundtrip!(secret_key, Ed25519SecretKey, 32);
  serde_roundtrip!(public_key, Ed25519PublicKey, 32);
  serde_roundtrip!(signature, Ed25519Signature, 64);
}

#[cfg(feature = "x25519")]
mod x25519_serde {
  use rscrypto::X25519PublicKey;
  #[cfg(feature = "serde-secrets")]
  use rscrypto::{X25519SecretKey, X25519SharedSecret};

  #[cfg(feature = "serde-secrets")]
  serde_roundtrip!(secret_key, X25519SecretKey, 32);
  serde_roundtrip!(public_key, X25519PublicKey, 32);
  #[cfg(feature = "serde-secrets")]
  serde_roundtrip!(shared_secret, X25519SharedSecret, 32);
}

#[cfg(feature = "aead")]
#[test]
fn wrong_length_bytes_rejected() {
  use rscrypto::aead::Nonce96;

  // Nonce96 is 12 bytes; feeding 11 or 13 must fail.
  let short = serde_json::to_string(&[0u8; 11]).unwrap();
  let long = serde_json::to_string(&[0u8; 13]).unwrap();
  assert!(serde_json::from_str::<Nonce96>(&short).is_err());
  assert!(serde_json::from_str::<Nonce96>(&long).is_err());
}
