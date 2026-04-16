//! Smoke tests for `getrandom`-backed `random()` methods.
//!
//! Validates that the OS entropy source is reachable and that successive calls
//! produce distinct output (ruling out an all-zero or constant return).

#![cfg(feature = "getrandom")]

macro_rules! random_smoke {
  ($name:ident, $ty:ty) => {
    #[test]
    fn $name() {
      let a = <$ty>::random();
      let b = <$ty>::random();

      // Must not be all-zero (overwhelmingly unlikely from a CSPRNG).
      assert!(a.as_bytes().iter().any(|&b| b != 0), "random() returned all zeros");
      // Two calls must differ (probability of collision: 2^-256 for 32-byte types).
      assert_ne!(
        a.as_bytes(),
        b.as_bytes(),
        "two random() calls returned identical output"
      );
    }
  };
}

// ── AEAD keys ───────────────────────────────────────────────────────────────
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
mod aes256gcm_rng {
  use rscrypto::aead::Aes256GcmKey;
  random_smoke!(key, Aes256GcmKey);
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

// ── AEAD nonces ─────────────────────────────────────────────────────────────
#[cfg(feature = "aead")]
mod nonce_rng {
  use rscrypto::aead::{Nonce96, Nonce128, Nonce192, Nonce256};
  random_smoke!(nonce96, Nonce96);
  random_smoke!(nonce128, Nonce128);
  random_smoke!(nonce192, Nonce192);
  random_smoke!(nonce256, Nonce256);
}

// ── Auth keys ───────────────────────────────────────────────────────────────
#[cfg(feature = "ed25519")]
mod ed25519_rng {
  use rscrypto::Ed25519SecretKey;
  random_smoke!(secret_key, Ed25519SecretKey);
}

#[cfg(feature = "x25519")]
mod x25519_rng {
  use rscrypto::X25519SecretKey;
  random_smoke!(secret_key, X25519SecretKey);
}
