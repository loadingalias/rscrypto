//! Serde round-trip tests for all types with `impl_serde_bytes!`.
//!
//! Each type is serialized to JSON and deserialized back, asserting byte-level
//! equality. This catches encoding bugs that would silently corrupt keys, tags,
//! nonces, and signatures over the wire.

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

// ── AEAD nonces ─────────────────────────────────────────────────────────────
#[cfg(feature = "aead")]
mod aead_nonces {
  use rscrypto::aead::{Nonce96, Nonce128, Nonce192, Nonce256};

  serde_roundtrip!(nonce96, Nonce96, 12);
  serde_roundtrip!(nonce128, Nonce128, 16);
  serde_roundtrip!(nonce192, Nonce192, 24);
  serde_roundtrip!(nonce256, Nonce256, 32);
}

// ── AEAD keys and tags ──────────────────────────────────────────────────────
#[cfg(feature = "chacha20poly1305")]
mod chacha20poly1305_serde {
  use rscrypto::aead::{ChaCha20Poly1305Key, ChaCha20Poly1305Tag};

  serde_roundtrip!(key, ChaCha20Poly1305Key, 32);
  serde_roundtrip!(tag, ChaCha20Poly1305Tag, 16);
}

#[cfg(feature = "xchacha20poly1305")]
mod xchacha20poly1305_serde {
  use rscrypto::aead::{XChaCha20Poly1305Key, XChaCha20Poly1305Tag};

  serde_roundtrip!(key, XChaCha20Poly1305Key, 32);
  serde_roundtrip!(tag, XChaCha20Poly1305Tag, 16);
}

#[cfg(feature = "aes-gcm")]
mod aes256gcm_serde {
  use rscrypto::aead::{Aes256GcmKey, Aes256GcmTag};

  serde_roundtrip!(key, Aes256GcmKey, 32);
  serde_roundtrip!(tag, Aes256GcmTag, 16);
}

#[cfg(feature = "aes-gcm-siv")]
mod aes256gcmsiv_serde {
  use rscrypto::aead::{Aes256GcmSivKey, Aes256GcmSivTag};

  serde_roundtrip!(key, Aes256GcmSivKey, 32);
  serde_roundtrip!(tag, Aes256GcmSivTag, 16);
}

#[cfg(feature = "ascon-aead")]
mod ascon128_serde {
  use rscrypto::aead::{AsconAead128Key, AsconAead128Tag};

  serde_roundtrip!(key, AsconAead128Key, 16);
  serde_roundtrip!(tag, AsconAead128Tag, 16);
}

#[cfg(feature = "aegis256")]
mod aegis256_serde {
  use rscrypto::aead::{Aegis256Key, Aegis256Tag};

  serde_roundtrip!(key, Aegis256Key, 32);
  serde_roundtrip!(tag, Aegis256Tag, 16);
}

// ── Auth keys and signatures ────────────────────────────────────────────────
#[cfg(feature = "ed25519")]
mod ed25519_serde {
  use rscrypto::{Ed25519PublicKey, Ed25519SecretKey, Ed25519Signature};

  serde_roundtrip!(secret_key, Ed25519SecretKey, 32);
  serde_roundtrip!(public_key, Ed25519PublicKey, 32);
  serde_roundtrip!(signature, Ed25519Signature, 64);
}

#[cfg(feature = "x25519")]
mod x25519_serde {
  use rscrypto::{X25519PublicKey, X25519SecretKey, X25519SharedSecret};

  serde_roundtrip!(secret_key, X25519SecretKey, 32);
  serde_roundtrip!(public_key, X25519PublicKey, 32);
  serde_roundtrip!(shared_secret, X25519SharedSecret, 32);
}

// ── Wrong-length deserialization must fail ───────────────────────────────────
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
