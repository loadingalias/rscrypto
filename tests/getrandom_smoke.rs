//! Smoke tests for `getrandom`-backed fallible constructors.
//!
//! Validates that the OS entropy source is reachable and that successive calls
//! produce distinct output (ruling out an all-zero or constant return).

#![cfg(feature = "getrandom")]

#[cfg(any(
  feature = "chacha20poly1305",
  feature = "xchacha20poly1305",
  feature = "aes-gcm",
  feature = "aes-gcm-siv",
  feature = "ascon-aead",
  feature = "aegis256",
  feature = "aead",
  feature = "ed25519",
  feature = "x25519"
))]
macro_rules! random_smoke {
  ($name:ident, $ty:ty) => {
    #[test]
    fn $name() {
      let a = <$ty>::try_random().expect("OS entropy must produce the first value");
      let b = <$ty>::try_random().expect("OS entropy must produce the second value");

      // Must not be all-zero (overwhelmingly unlikely from a CSPRNG).
      assert!(a.as_bytes().iter().any(|&b| b != 0), "try_random() returned all zeros");
      // Two calls must differ (probability of collision: 2^-256 for 32-byte types).
      assert_ne!(
        a.as_bytes(),
        b.as_bytes(),
        "two try_random() calls returned identical output"
      );
    }
  };
}

#[cfg(feature = "chacha20poly1305")]
mod chacha20poly1305_rng {
  use rscrypto::aead::ChaCha20Poly1305Key;
  random_smoke!(key, ChaCha20Poly1305Key);
}

#[cfg(feature = "xchacha20poly1305")]
mod xchacha20poly1305_rng {
  use rscrypto::aead::XChaCha20Poly1305Key;
  random_smoke!(key, XChaCha20Poly1305Key);
}

#[cfg(feature = "aes-gcm")]
mod aes128gcm_rng {
  use rscrypto::aead::Aes128GcmKey;
  random_smoke!(key, Aes128GcmKey);
}

#[cfg(feature = "aes-gcm")]
mod aes256gcm_rng {
  use rscrypto::aead::Aes256GcmKey;
  random_smoke!(key, Aes256GcmKey);
}

#[cfg(feature = "aes-gcm-siv")]
mod aes128gcmsiv_rng {
  use rscrypto::aead::Aes128GcmSivKey;
  random_smoke!(key, Aes128GcmSivKey);
}

#[cfg(feature = "aes-gcm-siv")]
mod aes256gcmsiv_rng {
  use rscrypto::aead::Aes256GcmSivKey;
  random_smoke!(key, Aes256GcmSivKey);
}

#[cfg(feature = "ascon-aead")]
mod ascon128_rng {
  use rscrypto::aead::AsconAead128Key;
  random_smoke!(key, AsconAead128Key);
}

#[cfg(feature = "aegis256")]
mod aegis256_rng {
  use rscrypto::aead::Aegis256Key;
  random_smoke!(key, Aegis256Key);
}

#[cfg(feature = "aead")]
mod nonce_rng {
  use rscrypto::aead::{Nonce96, Nonce128, Nonce192, Nonce256};
  random_smoke!(nonce96, Nonce96);
  random_smoke!(nonce128, Nonce128);
  random_smoke!(nonce192, Nonce192);
  random_smoke!(nonce256, Nonce256);
}

#[cfg(feature = "ed25519")]
mod ed25519_rng {
  use rscrypto::{Ed25519Keypair, Ed25519SecretKey};
  random_smoke!(secret_key, Ed25519SecretKey);

  #[test]
  fn keypair_try_generate() {
    let keypair = Ed25519Keypair::try_generate().expect("OS entropy must generate an Ed25519 keypair");
    assert!(keypair.secret_key().as_bytes().iter().any(|&b| b != 0));
  }
}

#[cfg(feature = "x25519")]
mod x25519_rng {
  use rscrypto::X25519SecretKey;
  random_smoke!(secret_key, X25519SecretKey);

  #[test]
  fn secret_key_try_generate() {
    let secret = X25519SecretKey::try_generate().expect("OS entropy must generate an X25519 secret key");
    assert!(secret.as_bytes().iter().any(|&b| b != 0));
  }
}

#[cfg(feature = "chacha20-rng")]
mod chacha20_rng {
  use rscrypto::ChaCha20Rng;

  #[test]
  fn generator_try_generate_uses_platform_entropy() {
    let mut first = ChaCha20Rng::try_generate().expect("OS entropy must seed the first generator");
    let mut second = ChaCha20Rng::try_generate().expect("OS entropy must seed the second generator");
    assert_ne!(
      first.next_u64(),
      second.next_u64(),
      "independently seeded generators agreed"
    );
  }
}

#[cfg(feature = "rapidhash")]
mod rapidhash_rng {
  use core::hash::BuildHasher;

  use rscrypto::RapidRandomState;

  #[test]
  fn random_state_uses_platform_entropy() {
    let first = RapidRandomState::try_new().expect("OS entropy must initialize the first random state");
    let second = RapidRandomState::try_new().expect("OS entropy must initialize the second random state");
    assert_ne!(
      first.hash_one(b"collection key"),
      second.hash_one(b"collection key"),
      "successive randomized states produced the same hash"
    );
  }
}

#[cfg(feature = "ecdsa-p256")]
mod ecdsa_p256_rng {
  use rscrypto::EcdsaP256SecretKey;

  /// Every valid scalar is nonzero, so only a second key shows that the source is not constant.
  #[test]
  fn secret_key_try_generate() {
    let first = EcdsaP256SecretKey::try_generate().expect("OS entropy must generate a P-256 secret key");
    let second = EcdsaP256SecretKey::try_generate().expect("OS entropy must generate a second P-256 secret key");
    assert_ne!(
      first.as_bytes(),
      second.as_bytes(),
      "two generated P-256 keys are equal"
    );
  }
}

#[cfg(feature = "ecdsa-p384")]
mod ecdsa_p384_rng {
  use rscrypto::EcdsaP384SecretKey;

  /// Every valid scalar is nonzero, so only a second key shows that the source is not constant.
  #[test]
  fn secret_key_try_generate() {
    let first = EcdsaP384SecretKey::try_generate().expect("OS entropy must generate a P-384 secret key");
    let second = EcdsaP384SecretKey::try_generate().expect("OS entropy must generate a second P-384 secret key");
    assert_ne!(
      first.as_bytes(),
      second.as_bytes(),
      "two generated P-384 keys are equal"
    );
  }
}

#[cfg(feature = "ml-kem")]
mod mlkem_rng {
  use rscrypto::{MlKem512, MlKem768, MlKem1024};

  /// Keys from any seed, even all zeros, have nonzero bytes, so only a second
  /// keypair shows that the source is not constant.
  #[test]
  fn keypair_try_generate() {
    macro_rules! distinct {
      ($kem:ty) => {{
        let (first, _) = <$kem>::try_generate_keypair().expect("OS entropy must generate a keypair");
        let (second, _) = <$kem>::try_generate_keypair().expect("OS entropy must generate a second keypair");
        assert_ne!(
          first.as_bytes(),
          second.as_bytes(),
          "two generated {} keys are equal",
          stringify!($kem)
        );
      }};
    }
    distinct!(MlKem512);
    distinct!(MlKem768);
    distinct!(MlKem1024);
  }
}
