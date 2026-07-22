use core::fmt::Debug;

fn assert_debug_snapshot(value: &impl Debug, expected: &str) {
  assert_eq!(format!("{value:?}"), expected);
}

#[test]
fn generic_secret_owner_debug_is_redacted() {
  assert_debug_snapshot(&rscrypto::SecretBytes::new([0x53; 16]), "SecretBytes(****)");
}

#[test]
fn keyed_state_debug_snapshots_are_redacted() {
  const KEY_16: [u8; 16] = [0x53; 16];
  const KEY_32: [u8; 32] = [0x53; 32];

  #[cfg(feature = "aes-gcm")]
  {
    use rscrypto::{Aes128Gcm, Aes128GcmKey, Aes256Gcm, Aes256GcmKey};

    let key = Aes128GcmKey::from_bytes(KEY_16);
    assert_debug_snapshot(&key, "Aes128GcmKey(****)");
    assert_debug_snapshot(&Aes128Gcm::new(&key), "Aes128Gcm { .. }");

    let key = Aes256GcmKey::from_bytes(KEY_32);
    assert_debug_snapshot(&key, "Aes256GcmKey(****)");
    assert_debug_snapshot(&Aes256Gcm::new(&key), "Aes256Gcm { .. }");
  }

  #[cfg(feature = "aes-gcm-siv")]
  {
    use rscrypto::{Aes128GcmSiv, Aes128GcmSivKey, Aes256GcmSiv, Aes256GcmSivKey};

    let key = Aes128GcmSivKey::from_bytes(KEY_16);
    assert_debug_snapshot(&key, "Aes128GcmSivKey(****)");
    assert_debug_snapshot(&Aes128GcmSiv::new(&key), "Aes128GcmSiv { .. }");

    let key = Aes256GcmSivKey::from_bytes(KEY_32);
    assert_debug_snapshot(&key, "Aes256GcmSivKey(****)");
    assert_debug_snapshot(&Aes256GcmSiv::new(&key), "Aes256GcmSiv { .. }");
  }

  #[cfg(feature = "chacha20poly1305")]
  {
    use rscrypto::{ChaCha20Poly1305, ChaCha20Poly1305Key};

    let key = ChaCha20Poly1305Key::from_bytes(KEY_32);
    assert_debug_snapshot(&key, "ChaCha20Poly1305Key(****)");
    assert_debug_snapshot(&ChaCha20Poly1305::new(&key), "ChaCha20Poly1305 { .. }");
  }

  #[cfg(feature = "xchacha20poly1305")]
  {
    use rscrypto::{XChaCha20Poly1305, XChaCha20Poly1305Key};

    let key = XChaCha20Poly1305Key::from_bytes(KEY_32);
    assert_debug_snapshot(&key, "XChaCha20Poly1305Key(****)");
    assert_debug_snapshot(&XChaCha20Poly1305::new(&key), "XChaCha20Poly1305 { .. }");
  }

  #[cfg(feature = "aegis256")]
  {
    use rscrypto::{Aegis256, Aegis256Key};

    let key = Aegis256Key::from_bytes(KEY_32);
    assert_debug_snapshot(&key, "Aegis256Key(****)");
    assert_debug_snapshot(&Aegis256::new(&key), "Aegis256 { .. }");
  }

  #[cfg(feature = "ascon-aead")]
  {
    use rscrypto::{AsconAead128, AsconAead128Key};

    let key = AsconAead128Key::from_bytes(KEY_16);
    assert_debug_snapshot(&key, "AsconAead128Key(****)");
    assert_debug_snapshot(&AsconAead128::new(&key), "AsconAead128 { .. }");
  }

  #[cfg(feature = "hmac")]
  {
    use rscrypto::{HmacSha256, HmacSha384, HmacSha512, Mac};

    assert_debug_snapshot(&HmacSha256::new(&KEY_32), "HmacSha256 { .. }");
    assert_debug_snapshot(&HmacSha384::new(&KEY_32), "HmacSha384 { .. }");
    assert_debug_snapshot(&HmacSha512::new(&KEY_32), "HmacSha512 { .. }");
  }

  #[cfg(feature = "hmac-sha3")]
  {
    use rscrypto::{HmacSha3_224, HmacSha3_256, HmacSha3_384, HmacSha3_512, Mac};

    assert_debug_snapshot(&HmacSha3_224::new(&KEY_32), "HmacSha3_224 { .. }");
    assert_debug_snapshot(&HmacSha3_256::new(&KEY_32), "HmacSha3_256 { .. }");
    assert_debug_snapshot(&HmacSha3_384::new(&KEY_32), "HmacSha3_384 { .. }");
    assert_debug_snapshot(&HmacSha3_512::new(&KEY_32), "HmacSha3_512 { .. }");
  }

  #[cfg(feature = "hkdf")]
  {
    use rscrypto::{HkdfSha256, HkdfSha384, HkdfSha512};

    assert_debug_snapshot(&HkdfSha256::new(b"salt", &KEY_32), "HkdfSha256 { .. }");
    assert_debug_snapshot(&HkdfSha384::new(b"salt", &KEY_32), "HkdfSha384 { .. }");
    assert_debug_snapshot(&HkdfSha512::new(b"salt", &KEY_32), "HkdfSha512 { .. }");
  }

  #[cfg(feature = "kmac")]
  {
    use rscrypto::{Kmac128, Kmac256};

    assert_debug_snapshot(&Kmac128::new(&KEY_32, b"custom"), "Kmac128 { .. }");
    assert_debug_snapshot(&Kmac256::new(&KEY_32, b"custom"), "Kmac256 { .. }");
  }

  #[cfg(feature = "pbkdf2")]
  {
    use rscrypto::{Pbkdf2Sha256, Pbkdf2Sha512};

    assert_debug_snapshot(&Pbkdf2Sha256::new(&KEY_32), "Pbkdf2Sha256 { .. }");
    assert_debug_snapshot(&Pbkdf2Sha512::new(&KEY_32), "Pbkdf2Sha512 { .. }");
  }

  #[cfg(feature = "poly1305")]
  {
    use rscrypto::{Poly1305, Poly1305OneTimeKey};

    let key = Poly1305OneTimeKey::from_bytes(KEY_32);
    assert_debug_snapshot(&key, "Poly1305OneTimeKey(****)");
    assert_debug_snapshot(&Poly1305::new(key), "Poly1305 { .. }");
  }
}

#[test]
fn keyed_hash_debug_snapshots_are_redacted() {
  const KEY: [u8; 32] = [0x53; 32];

  #[cfg(feature = "blake2s")]
  {
    let params = rscrypto::Blake2sParams::new().key(&KEY);
    assert_debug_snapshot(
      &params,
      "Blake2sParams { key_len: 32, salt: [0, 0, 0, 0, 0, 0, 0, 0], personal: [0, 0, 0, 0, 0, 0, 0, 0] }",
    );
    assert_debug_snapshot(&params.build_128(), "Blake2s128 { .. }");
    assert_debug_snapshot(&params.build_256(), "Blake2s256 { .. }");
  }

  #[cfg(feature = "blake2b")]
  {
    let params = rscrypto::Blake2bParams::new().key(&KEY);
    assert_debug_snapshot(
      &params,
      "Blake2bParams { key_len: 32, salt: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], personal: [0, 0, 0, 0, 0, \
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] }",
    );
    assert_debug_snapshot(&params.build_256(), "Blake2b256 { .. }");
    assert_debug_snapshot(&params.build_512(), "Blake2b512 { .. }");
    assert_debug_snapshot(&params.build(32), "Blake2b { output_len: 32, .. }");
  }

  #[cfg(feature = "blake3")]
  {
    assert_debug_snapshot(&rscrypto::Blake3::new_keyed(&KEY), "Blake3 { .. }");
    assert_debug_snapshot(&rscrypto::Blake3::keyed_xof(&KEY, b"message"), "Blake3XofReader { .. }");
  }
}

#[test]
fn private_key_and_shared_secret_debug_snapshots_are_redacted() {
  #[cfg(feature = "ecdsa-p256")]
  {
    let secret = rscrypto::EcdsaP256SecretKey::from_bytes([1u8; 32]).expect("valid P-256 scalar");
    assert_debug_snapshot(&secret, "EcdsaP256SecretKey(****)");
    let keypair = rscrypto::EcdsaP256Keypair::from_secret_key(secret);
    assert_debug_snapshot(
      &keypair,
      &format!("EcdsaP256Keypair {{ public: {:?}, .. }}", keypair.public_key()),
    );
  }

  #[cfg(feature = "ecdsa-p384")]
  {
    let secret = rscrypto::EcdsaP384SecretKey::from_bytes([1u8; 48]).expect("valid P-384 scalar");
    assert_debug_snapshot(&secret, "EcdsaP384SecretKey(****)");
    let keypair = rscrypto::EcdsaP384Keypair::from_secret_key(secret);
    assert_debug_snapshot(
      &keypair,
      &format!("EcdsaP384Keypair {{ public: {:?}, .. }}", keypair.public_key()),
    );
  }

  #[cfg(feature = "ed25519")]
  {
    let secret = rscrypto::Ed25519SecretKey::from_bytes([0x53; 32]);
    assert_debug_snapshot(&secret, "Ed25519SecretKey(****)");
    let keypair = rscrypto::Ed25519Keypair::from_secret_key(secret);
    assert_debug_snapshot(
      &keypair,
      &format!("Ed25519Keypair {{ public: {:?}, .. }}", keypair.public_key()),
    );
  }

  #[cfg(feature = "x25519")]
  {
    let secret = rscrypto::X25519SecretKey::from_bytes([0x53; 32]);
    assert_debug_snapshot(&secret, "X25519SecretKey(****)");
    let shared = secret
      .diffie_hellman(&rscrypto::X25519PublicKey::basepoint())
      .expect("basepoint exchange must produce a nonzero shared secret");
    assert_debug_snapshot(&shared, "X25519SharedSecret(****)");
  }

  #[cfg(feature = "ml-kem")]
  {
    assert_debug_snapshot(
      &rscrypto::MlKem512DecapsulationKey::from_bytes([0x53; rscrypto::MlKem512DecapsulationKey::LENGTH]),
      "MlKem512DecapsulationKey(****)",
    );
    assert_debug_snapshot(
      &rscrypto::MlKem768DecapsulationKey::from_bytes([0x53; rscrypto::MlKem768DecapsulationKey::LENGTH]),
      "MlKem768DecapsulationKey(****)",
    );
    assert_debug_snapshot(
      &rscrypto::MlKem1024DecapsulationKey::from_bytes([0x53; rscrypto::MlKem1024DecapsulationKey::LENGTH]),
      "MlKem1024DecapsulationKey(****)",
    );
    assert_debug_snapshot(
      &rscrypto::MlKem512SharedSecret::from_bytes([0x53; rscrypto::MlKem512SharedSecret::LENGTH]),
      "MlKem512SharedSecret(****)",
    );
    assert_debug_snapshot(
      &rscrypto::MlKem768SharedSecret::from_bytes([0x53; rscrypto::MlKem768SharedSecret::LENGTH]),
      "MlKem768SharedSecret(****)",
    );
    assert_debug_snapshot(
      &rscrypto::MlKem1024SharedSecret::from_bytes([0x53; rscrypto::MlKem1024SharedSecret::LENGTH]),
      "MlKem1024SharedSecret(****)",
    );

    macro_rules! assert_prepared_decapsulation_key_redacted {
      ($profile:ty, $expected:literal) => {{
        use rscrypto::Kem as _;

        let (_, key) = <$profile>::generate_keypair(|out| {
          for (index, byte) in out.iter_mut().enumerate() {
            *byte = 0x53u8.wrapping_add(index as u8);
          }
          Ok::<(), rscrypto::MlKemError>(())
        })
        .expect("deterministic ML-KEM key generation must succeed");
        let prepared = key.prepare().expect("generated ML-KEM key must validate");
        assert_debug_snapshot(&prepared, $expected);
      }};
    }

    assert_prepared_decapsulation_key_redacted!(rscrypto::MlKem512, "MlKem512PreparedDecapsulationKey(****)");
    assert_prepared_decapsulation_key_redacted!(rscrypto::MlKem768, "MlKem768PreparedDecapsulationKey(****)");
    assert_prepared_decapsulation_key_redacted!(rscrypto::MlKem1024, "MlKem1024PreparedDecapsulationKey(****)");
  }
}

#[test]
fn borrowed_secret_context_debug_snapshot_is_redacted() {
  #[cfg(feature = "argon2")]
  assert_debug_snapshot(
    &rscrypto::Argon2Context::new(b"pepper-value", b"associated-value"),
    "Argon2Context { secret: \"[REDACTED]\", secret_len: 12, associated_data_len: 16 }",
  );
}

#[test]
fn secret_input_error_snapshots_do_not_echo_input_bytes() {
  #[cfg(feature = "aes-gcm")]
  {
    let error = "5353535353535353535353535353535z"
      .parse::<rscrypto::Aes128GcmKey>()
      .expect_err("invalid secret hex must fail");
    assert_debug_snapshot(&error, "InvalidChar { index: 31, .. }");
    assert_eq!(error.to_string(), "invalid hex character at index 31");
  }

  #[cfg(feature = "ecdsa-p256")]
  {
    let error = rscrypto::EcdsaP256SecretKey::from_bytes([0u8; 32]).expect_err("zero scalar must fail");
    assert_debug_snapshot(&error, "InvalidSecretKey");
    assert_eq!(error.to_string(), "invalid ECDSA secret key");

    struct SeedBearingError;

    impl Debug for SeedBearingError {
      fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("SeedBearingError(53535353)")
      }
    }

    impl core::fmt::Display for SeedBearingError {
      fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("seed-bearing error: 53535353")
      }
    }

    impl core::error::Error for SeedBearingError {}

    let generation_error = rscrypto::EcdsaP256SecretKey::try_generate_with(|_| Err(SeedBearingError))
      .expect_err("random-source failure must be preserved as an opaque error");
    assert_debug_snapshot(&generation_error, "Random(..)");
    assert_eq!(
      generation_error.to_string(),
      "ECDSA key-generation random source failed"
    );
    assert!(core::error::Error::source(&generation_error).is_none());
  }

  #[cfg(feature = "ml-kem")]
  {
    let error = rscrypto::MlKem512DecapsulationKey::try_from_slice(&[0x53; 31])
      .expect_err("wrong-length decapsulation key must fail");
    assert_debug_snapshot(&error, "InvalidDecapsulationKey");
    assert_eq!(error.to_string(), "ML-KEM decapsulation key failed validation");
  }

  #[cfg(feature = "kmac")]
  {
    let error =
      rscrypto::Kmac256::verify_tag(b"secret key", b"custom", b"message", &[]).expect_err("empty KMAC tag must fail");
    assert_debug_snapshot(&error, "VerificationError");
    assert_eq!(error.to_string(), "verification failed");
  }

  #[cfg(feature = "aes-gcm")]
  {
    let error = rscrypto::OpenError::verification();
    assert_debug_snapshot(&error, "Verification(VerificationError)");
    assert_eq!(error.to_string(), "verification failed");
  }
}
