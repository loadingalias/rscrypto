use super::*;
use crate::auth::rsa::tests::*;

/// Request counts of one signature through a workspace.
#[cfg(feature = "getrandom")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Requests {
  salt: usize,
  blinding: usize,
}

/// Sign through `workspace` with a salt source that fills `0x5a` and a
/// blinding source that gives the factor 2 for every request (blinding
/// candidate and inverse mask). Returns the result and the requests of each
/// source.
#[cfg(feature = "getrandom")]
fn sign_with_workspace_and_factor_two(
  key: &RsaPrivateKey,
  profile: RsaSignatureProfile,
  message: &[u8],
  out: &mut [u8],
  workspace: &mut RsaSigningWorkspace,
) -> (Result<(), RsaPrivateOpError>, Requests) {
  let (factor, _) = factor_two_and_inverse(key.public_key().modulus());
  let mut requests = Requests { salt: 0, blinding: 0 };
  let result = key.sign_signature_with_random_fill_and_workspace(
    profile,
    message,
    out,
    workspace,
    |salt| {
      requests.salt = requests.salt.strict_add(1);
      salt.fill(0x5a);
      Ok::<(), !>(())
    },
    |blinding| {
      requests.blinding = requests.blinding.strict_add(1);
      blinding.copy_from_slice(&factor);
      Ok::<(), !>(())
    },
  );
  (result, requests)
}

/// The pair that the `use_in_epoch`-th signature of an epoch uses when its
/// refresh drew the factor 2: `(2^(2^(k-1)))ᵉ mod n` and `(2⁻¹)^(2^(k-1)) mod n`,
/// computed with num-bigint.
#[cfg(feature = "getrandom")]
fn factor_two_epoch_pair(key: &RsaPrivateKey, use_in_epoch: usize) -> (Vec<u8>, Vec<u8>) {
  let modulus = BigUint::from_bytes_be(key.public_key().modulus());
  let squarings = u32::try_from(use_in_epoch.strict_sub(1)).expect("an epoch has at most 32 signatures");
  let doubling = 1u64.strict_shl(squarings);
  let exponent = key.public_key().public_exponent().as_u64().strict_mul(doubling);
  let power = BigUint::from(2u8).modpow(&BigUint::from(exponent), &modulus);
  let (_, half) = factor_two_and_inverse(key.public_key().modulus());
  let inverse = BigUint::from_bytes_be(&half).modpow(&BigUint::from(doubling), &modulus);
  let mut power_bytes = vec![0u8; key.signature_len()];
  let mut inverse_bytes = vec![0u8; key.signature_len()];
  left_pad_be(&power.to_bytes_be(), &mut power_bytes).expect("the blinding power must fit the modulus width");
  left_pad_be(&inverse.to_bytes_be(), &mut inverse_bytes).expect("the blinding inverse must fit the modulus width");
  (power_bytes, inverse_bytes)
}

#[cfg(feature = "getrandom")]
fn assert_workspace_pair_is_clear(workspace: &RsaSigningWorkspace) {
  assert_eq!(workspace.remaining, 0);
  assert!(workspace.blinding_power.as_slice().iter().all(|&byte| byte == 0));
  assert!(workspace.blinding_inverse.as_slice().iter().all(|&byte| byte == 0));
}

#[test]
fn rsa_miri_proof_zeroizes_private_storage_at_every_word_alignment() {
  const GUARD: u8 = 0x3c;
  const SECRET: u8 = 0xa5;
  const LENGTHS: [usize; 11] = [0, 1, 7, 8, 9, 15, 16, 17, 127, 128, 129];
  let word_bytes = core::mem::size_of::<u64>();

  for offset in 0..word_bytes {
    for len in LENGTHS {
      let end = offset.strict_add(len);
      let mut storage = vec![GUARD; end.strict_add(word_bytes)];
      storage[offset..end].fill(SECRET);

      ct::zeroize(&mut storage[offset..end]);

      assert!(storage[..offset].iter().all(|&byte| byte == GUARD));
      assert!(storage[offset..end].iter().all(|&byte| byte == 0));
      assert!(storage[end..].iter().all(|&byte| byte == GUARD));
    }
  }
}

#[test]
fn private_key_signs_pkcs1v15_and_pss_end_to_end() {
  let key = prevalidated_test_private_key();
  let message = b"rscrypto RSA private signing roundtrip";
  let (blinding_factor, blinding_factor_inverse) = factor_two_and_inverse(key.public_key().modulus());

  let mut pkcs1v15_signature = vec![0u8; key.public_key().modulus().len()];
  key
    .sign_pkcs1v15_with_blinding_factor(
      RsaPkcs1v15Profile::Sha256,
      message,
      RsaBlindingPair::new(&blinding_factor, &blinding_factor_inverse),
      &mut pkcs1v15_signature,
    )
    .expect("PKCS#1 v1.5 signing with a valid blinding pair must succeed");
  key
    .public_key()
    .verify_pkcs1v15(RsaPkcs1v15Profile::Sha256, message, &pkcs1v15_signature)
    .expect("the generated PKCS#1 v1.5 signature must verify");
  assert_eq!(
    key
      .public_key()
      .verify_pkcs1v15(RsaPkcs1v15Profile::Sha256, b"wrong message", &pkcs1v15_signature),
    Err(VerificationError::new())
  );
  let salt = [0xa5; Sha256::OUTPUT_SIZE];
  let mut pss_signature = vec![0u8; key.public_key().modulus().len()];
  key
    .sign_pss_with_salt_and_blinding_factor(
      RsaPssProfile::Sha256,
      message,
      &salt,
      RsaBlindingPair::new(&blinding_factor, &blinding_factor_inverse),
      &mut pss_signature,
    )
    .expect("PSS signing with explicit salt and a valid blinding pair must succeed");
  key
    .public_key()
    .verify_pss_with_salt_len(RsaPssProfile::Sha256, salt.len(), message, &pss_signature)
    .expect("the generated PSS signature must verify with its explicit salt length");
  assert_eq!(
    key
      .public_key()
      .verify_pss_with_salt_len(RsaPssProfile::Sha256, salt.len(), b"wrong message", &pss_signature),
    Err(VerificationError::new())
  );
  let label = b"rscrypto-oaep-label";
  let plaintext = b"rsa oaep decrypt roundtrip";
  let mut ciphertext = vec![0u8; key.signature_len()];
  let seed = [0x53; Sha256::OUTPUT_SIZE];
  key
    .public_key()
    .encrypt_oaep_with_random_fill(
      RsaOaepProfile::Sha256,
      label,
      plaintext,
      &mut ciphertext,
      seed_fill(&seed),
    )
    .expect("OAEP encryption with a correctly sized deterministic seed must succeed");
  let mut decrypted = vec![0u8; key.signature_len()];
  let decrypted_len = key
    .decrypt_oaep_with_blinding_factor(
      RsaOaepProfile::Sha256,
      label,
      &ciphertext,
      RsaBlindingPair::new(&blinding_factor, &blinding_factor_inverse),
      &mut decrypted,
    )
    .expect("OAEP decryption with the matching label and blinding pair must succeed");
  assert_eq!(&decrypted[..decrypted_len], plaintext);
  assert_eq!(
    key.decrypt_oaep_with_blinding_factor(
      RsaOaepProfile::Sha256,
      b"wrong label",
      &ciphertext,
      RsaBlindingPair::new(&blinding_factor, &blinding_factor_inverse),
      &mut decrypted,
    ),
    Err(RsaPrivateOpError::DecryptionFailed)
  );
  let pkcs1v15_plaintext = b"rsaes pkcs1v15 decrypt roundtrip";
  let pkcs1v15_padding = vec![0xa7; key.signature_len().strict_sub(pkcs1v15_plaintext.len()).strict_sub(3)];
  let mut pkcs1v15_ciphertext = vec![0u8; key.signature_len()];
  key
    .public_key()
    .encrypt_pkcs1v15_with_random_fill(
      pkcs1v15_plaintext,
      &mut pkcs1v15_ciphertext,
      seed_fill(&pkcs1v15_padding),
    )
    .expect("PKCS#1 v1.5 encryption with a valid deterministic padding string must succeed");
  let mut pkcs1v15_decrypted = vec![0u8; key.signature_len()];
  let pkcs1v15_decrypted_len = key
    .decrypt_pkcs1v15_with_blinding_factor(
      &pkcs1v15_ciphertext,
      RsaBlindingPair::new(&blinding_factor, &blinding_factor_inverse),
      &mut pkcs1v15_decrypted,
    )
    .expect("PKCS#1 v1.5 decryption with a valid blinding pair must succeed");
  assert_eq!(&pkcs1v15_decrypted[..pkcs1v15_decrypted_len], pkcs1v15_plaintext);
  let mut scratch = key.private_scratch();
  let pkcs1v15_decrypted_len = key
    .decrypt_pkcs1v15_with_blinding_factor_and_scratch(
      &pkcs1v15_ciphertext,
      RsaBlindingPair::new(&blinding_factor, &blinding_factor_inverse),
      &mut pkcs1v15_decrypted,
      &mut scratch,
    )
    .expect("scratch-backed PKCS#1 v1.5 decryption with a valid blinding pair must succeed");
  assert_eq!(&pkcs1v15_decrypted[..pkcs1v15_decrypted_len], pkcs1v15_plaintext);
}

#[test]
fn private_key_caller_random_signing_rejects_candidates_then_roundtrips() {
  let key = prevalidated_test_private_key();
  let message = b"rscrypto caller-random rejection stream";
  let (factor, _) = factor_two_and_inverse(key.public_key().modulus());
  let mut modulus_candidate = key.public_key().modulus().to_vec();
  let mut noninvertible_candidate = vec![0u8; key.signature_len()];
  left_pad_be(&rsa_private_prime_p(), &mut noninvertible_candidate).expect("prime factor must fit modulus width");
  let salt = [0x5au8; Sha256::OUTPUT_SIZE];
  let mut calls = 0usize;
  let mut signature = vec![0u8; key.signature_len()];
  let mut scratch = key.private_scratch();

  key
    .sign_pss_with_random_fill_and_scratch(RsaPssProfile::Sha256, message, &mut signature, &mut scratch, |out| {
      match calls {
        0 => {
          assert_eq!(out.len(), salt.len());
          out.copy_from_slice(&salt);
        }
        1 => out.fill(0),
        2 => out.copy_from_slice(&modulus_candidate),
        3 => out.copy_from_slice(&noninvertible_candidate),
        // The inverse mask for the noninvertible candidate.
        4 => out.copy_from_slice(&factor),
        5 => out.copy_from_slice(&factor),
        // An inverse mask equal to the modulus is redrawn.
        6 => out.copy_from_slice(&modulus_candidate),
        7 => out.copy_from_slice(&factor),
        _ => return Err(()),
      }
      calls = calls.strict_add(1);
      Ok::<(), ()>(())
    })
    .expect("a deterministic stream ending in an invertible factor must sign");

  modulus_candidate.fill(0);
  assert_eq!(calls, 8);
  key
    .public_key()
    .verify_pss(RsaPssProfile::Sha256, message, &signature)
    .expect("the caller-random signature must verify");
  assert_private_scratch_is_clear(&scratch);
}

#[test]
fn private_key_caller_random_signing_covers_profiles_and_tls_mappings() {
  let key = prevalidated_test_private_key();
  let message = b"rscrypto caller-random RSA profile coverage";
  let (factor, _) = factor_two_and_inverse(key.public_key().modulus());
  let mut signature = vec![0u8; key.signature_len()];
  let mut scratch = key.private_scratch();

  for profile in [RsaPssProfile::Sha256, RsaPssProfile::Sha384, RsaPssProfile::Sha512] {
    let mut calls = 0usize;
    key
      .sign_pss_with_random_fill_and_scratch(profile, message, &mut signature, &mut scratch, |out| {
        if calls == 0 {
          assert_eq!(out.len(), profile.digest_len());
          out.fill(0x3c);
        } else {
          assert_eq!(out.len(), factor.len());
          out.copy_from_slice(&factor);
        }
        calls = calls.strict_add(1);
        Ok::<(), ()>(())
      })
      .expect("every supported PSS profile must sign with caller entropy");
    assert_eq!(calls, 3);
    key
      .public_key()
      .verify_pss(profile, message, &signature)
      .expect("every supported caller-random PSS signature must verify");
    assert_private_scratch_is_clear(&scratch);
  }

  for scheme in [0x0804, 0x0805, 0x0806, 0x0809, 0x080a, 0x080b] {
    let profile = RsaSignatureProfile::from_tls13_signature_scheme(scheme)
      .expect("the TLS PSS scheme fixture must map to a profile");
    let mut calls = 0usize;
    key
      .sign_tls13_signature_scheme_with_random_fill_and_scratch(scheme, message, &mut signature, &mut scratch, |out| {
        if calls == 0 {
          out.fill(0x7d);
        } else {
          out.copy_from_slice(&factor);
        }
        calls = calls.strict_add(1);
        Ok::<(), ()>(())
      })
      .expect("TLS RSAE-PSS and PSS-PSS schemes must sign with caller entropy");
    assert_eq!(calls, 3);
    key
      .public_key()
      .verify_signature(profile, message, &signature)
      .expect("the caller-random TLS PSS signature must verify");
    assert_private_scratch_is_clear(&scratch);
  }

  for scheme in [0x0401, 0x0501, 0x0601] {
    let profile = RsaSignatureProfile::from_tls_certificate_signature_scheme(scheme)
      .expect("the TLS certificate RSA scheme fixture must map to a profile");
    let mut calls = 0usize;
    key
      .sign_tls_certificate_signature_scheme_with_random_fill_and_scratch(
        scheme,
        message,
        &mut signature,
        &mut scratch,
        |out| {
          assert_eq!(out.len(), factor.len());
          out.copy_from_slice(&factor);
          calls = calls.strict_add(1);
          Ok::<(), ()>(())
        },
      )
      .expect("TLS certificate PKCS#1 v1.5 schemes must sign with caller entropy");
    assert_eq!(calls, 2);
    key
      .public_key()
      .verify_signature(profile, message, &signature)
      .expect("the caller-random TLS certificate signature must verify");
    assert_private_scratch_is_clear(&scratch);
  }

  let explicit_salt = RsaSignatureProfile::pss_with_salt_len(RsaPssProfile::Sha512, 24);
  let mut calls = 0usize;
  key
    .sign_signature_with_random_fill(explicit_salt, message, &mut signature, |out| {
      if calls == 0 {
        assert_eq!(out.len(), 24);
        out.fill(0x91);
      } else {
        out.copy_from_slice(&factor);
      }
      calls = calls.strict_add(1);
      Ok::<(), ()>(())
    })
    .expect("the general profile boundary must preserve an explicit PSS salt length");
  assert_eq!(calls, 3);
  key
    .public_key()
    .verify_signature(explicit_salt, message, &signature)
    .expect("the explicit-salt caller-random signature must verify");
}

#[test]
fn private_key_caller_random_failures_are_bounded_opaque_and_clear_state() {
  #[derive(Debug)]
  struct CallbackError;

  let key = prevalidated_test_private_key();
  let message = b"rscrypto caller-random failure schedule";
  let profile = RsaSignatureProfile::pkcs1v15(RsaPkcs1v15Profile::Sha256);
  let mut signature = vec![0xa5; key.signature_len()];
  let mut scratch = key.private_scratch();

  for fail_at in 0usize..128 {
    signature.fill(0xa5);
    let mut calls = 0usize;
    assert_eq!(
      key.sign_signature_with_random_fill_and_scratch(profile, message, &mut signature, &mut scratch, |out| {
        let call = calls;
        calls = calls.strict_add(1);
        out.fill(0xa5);
        if call == fail_at {
          Err(CallbackError)
        } else {
          out.fill(0);
          Ok(())
        }
      },),
      Err(RsaPrivateOpError::EntropyUnavailable)
    );
    assert_eq!(calls, fail_at.strict_add(1));
    assert!(is_zero_unsigned_be(&signature));
    assert_private_scratch_is_clear(&scratch);
  }

  signature.fill(0xa5);
  let mut calls = 0usize;
  assert_eq!(
    key.sign_signature_with_random_fill_and_scratch(profile, message, &mut signature, &mut scratch, |out| {
      calls = calls.strict_add(1);
      out.fill(0);
      Ok::<(), CallbackError>(())
    }),
    Err(RsaPrivateOpError::InvalidBlindingFactor)
  );
  assert_eq!(calls, 128);
  assert!(is_zero_unsigned_be(&signature));
  assert_private_scratch_is_clear(&scratch);

  let (factor, _) = factor_two_and_inverse(key.public_key().modulus());
  signature.fill(0xa5);
  let mut calls = 0usize;
  assert_eq!(
    key.sign_signature_with_random_fill_and_scratch(profile, message, &mut signature, &mut scratch, |out| {
      calls = calls.strict_add(1);
      if calls == 1 {
        out.copy_from_slice(&factor);
        Ok(())
      } else {
        out.fill(0xa5);
        Err(CallbackError)
      }
    }),
    Err(RsaPrivateOpError::EntropyUnavailable)
  );
  assert_eq!(calls, 2, "a failed inverse-mask draw must end signing");
  assert!(is_zero_unsigned_be(&signature));
  assert_private_scratch_is_clear(&scratch);

  // Each attempt draws a valid candidate, then 128 zero inverse masks.
  signature.fill(0xa5);
  let mut calls = 0usize;
  assert_eq!(
    key.sign_signature_with_random_fill_and_scratch(profile, message, &mut signature, &mut scratch, |out| {
      if calls.is_multiple_of(129) {
        out.copy_from_slice(&factor);
      } else {
        out.fill(0);
      }
      calls = calls.strict_add(1);
      Ok::<(), CallbackError>(())
    }),
    Err(RsaPrivateOpError::InvalidBlindingFactor)
  );
  assert_eq!(calls, 128 * 129);
  assert!(is_zero_unsigned_be(&signature));
  assert_private_scratch_is_clear(&scratch);

  key
    .sign_signature_with_random_fill_and_scratch(profile, message, &mut signature, &mut scratch, |out| {
      out.copy_from_slice(&factor);
      Ok::<(), CallbackError>(())
    })
    .expect("scratch must be reusable after callback failure and retry exhaustion");
  key
    .public_key()
    .verify_signature(profile, message, &signature)
    .expect("a signature after scratch reuse must verify");
  assert_private_scratch_is_clear(&scratch);
}

#[test]
fn private_key_caller_random_rejects_public_shape_before_entropy() {
  let key = prevalidated_test_private_key();
  let message = b"rscrypto caller-random public rejection";
  let mut signature = vec![0xa5; key.signature_len()];
  let mut scratch = key.private_scratch();

  let mut calls = 0usize;
  let impossible = RsaSignatureProfile::pss_with_salt_len(RsaPssProfile::Sha512, usize::MAX);
  assert_eq!(
    key.sign_signature_with_random_fill_and_scratch(impossible, message, &mut signature, &mut scratch, |_| {
      calls = calls.strict_add(1);
      Ok::<(), ()>(())
    },),
    Err(RsaPrivateOpError::MessageTooLong)
  );
  assert_eq!(calls, 0);
  assert!(is_zero_unsigned_be(&signature));
  assert_private_scratch_is_clear(&scratch);

  signature.fill(0xa5);
  assert_eq!(
    key.sign_tls13_signature_scheme_with_random_fill(0x0401, message, &mut signature, |_| {
      calls = calls.strict_add(1);
      Ok::<(), ()>(())
    }),
    Err(RsaPrivateOpError::UnsupportedAlgorithm)
  );
  assert_eq!(calls, 0);
  assert!(is_zero_unsigned_be(&signature));

  signature.fill(0xa5);
  assert_eq!(
    key.sign_tls_certificate_signature_scheme_with_random_fill_and_scratch(
      0x0201,
      message,
      &mut signature,
      &mut scratch,
      |_| {
        calls = calls.strict_add(1);
        Ok::<(), ()>(())
      },
    ),
    Err(RsaPrivateOpError::UnsupportedAlgorithm)
  );
  assert_eq!(calls, 0);
  assert!(is_zero_unsigned_be(&signature));
  assert_private_scratch_is_clear(&scratch);

  let mut short_signature = vec![0xa5; key.signature_len().strict_sub(1)];
  assert_eq!(
    key.sign_pss_with_random_fill_and_scratch(
      RsaPssProfile::Sha256,
      message,
      &mut short_signature,
      &mut scratch,
      |_| {
        calls = calls.strict_add(1);
        Ok::<(), ()>(())
      },
    ),
    Err(RsaPrivateOpError::InvalidLength)
  );
  assert_eq!(calls, 0);
  assert!(is_zero_unsigned_be(&short_signature));
  assert_private_scratch_is_clear(&scratch);

  signature.fill(0xa5);
  let mut wrong_scratch = wrong_width_private_scratch(&key);
  assert_eq!(
    key.sign_pss_with_random_fill_and_scratch(
      RsaPssProfile::Sha256,
      message,
      &mut signature,
      &mut wrong_scratch,
      |_| {
        calls = calls.strict_add(1);
        Ok::<(), ()>(())
      },
    ),
    Err(RsaPrivateOpError::InvalidScratch)
  );
  assert_eq!(calls, 0);
  assert!(is_zero_unsigned_be(&signature));
  assert_private_scratch_is_clear(&wrong_scratch);

  signature.fill(0xa5);
  assert_eq!(
    key.sign_pss_with_random_fill_and_scratch(RsaPssProfile::Sha256, message, &mut signature, &mut scratch, |out| {
      out.fill(0xa5);
      Err::<(), ()>(())
    },),
    Err(RsaPrivateOpError::EntropyUnavailable)
  );
  assert!(is_zero_unsigned_be(&signature));
  assert_private_scratch_is_clear(&scratch);
}

#[test]
fn public_encryption_random_fill_failures_clear_output() {
  let key = prevalidated_test_private_key();
  let public_key = key.public_key();
  let label = b"rsa random-fill failure label";
  let plaintext = b"rsa random-fill failure plaintext";
  let mut ciphertext = vec![0xa5; key.signature_len()];

  assert_eq!(
    public_key.encrypt_oaep_with_random_fill(RsaOaepProfile::Sha256, label, plaintext, &mut ciphertext, |_| {
      Err(RsaEncryptionError::EntropyUnavailable)
    }),
    Err(RsaEncryptionError::EntropyUnavailable)
  );
  assert!(is_zero_unsigned_be(&ciphertext));

  let mut public_scratch = public_key.public_scratch();
  ciphertext.fill(0xa5);
  assert_eq!(
    public_key.encrypt_pkcs1v15_with_random_fill_and_scratch(plaintext, &mut ciphertext, &mut public_scratch, |out| {
      out.fill(0);
      Ok(())
    },),
    Err(RsaEncryptionError::EntropyUnavailable)
  );
  assert!(is_zero_unsigned_be(&ciphertext));
}

#[test]
fn private_key_invalid_blinding_clears_signing_and_decryption_outputs() {
  let key = prevalidated_test_private_key();
  let message = b"rscrypto RSA invalid blinding output clearing";
  let (blinding_factor, mut bad_blinding_inverse) = factor_two_and_inverse(key.public_key().modulus());
  *bad_blinding_inverse
    .last_mut()
    .expect("the fixed-width blinding inverse must contain at least one byte") ^= 1;
  let mut scratch = key.private_scratch();

  let mut pkcs1v15_signature = vec![0xa5; key.signature_len()];
  assert_eq!(
    key.sign_pkcs1v15_with_blinding_factor(
      RsaPkcs1v15Profile::Sha256,
      message,
      RsaBlindingPair::new(&blinding_factor, &bad_blinding_inverse),
      &mut pkcs1v15_signature,
    ),
    Err(RsaPrivateOpError::InvalidBlindingFactor)
  );
  assert!(is_zero_unsigned_be(&pkcs1v15_signature));

  pkcs1v15_signature.fill(0xa5);
  assert_eq!(
    key.sign_pkcs1v15_with_blinding_factor_and_scratch(
      RsaPkcs1v15Profile::Sha256,
      message,
      RsaBlindingPair::new(&blinding_factor, &bad_blinding_inverse),
      &mut pkcs1v15_signature,
      &mut scratch,
    ),
    Err(RsaPrivateOpError::InvalidBlindingFactor)
  );
  assert!(is_zero_unsigned_be(&pkcs1v15_signature));

  let salt = [0x5c; Sha256::OUTPUT_SIZE];
  let mut pss_signature = vec![0xa5; key.signature_len()];
  assert_eq!(
    key.sign_pss_with_salt_and_blinding_factor(
      RsaPssProfile::Sha256,
      message,
      &salt,
      RsaBlindingPair::new(&blinding_factor, &bad_blinding_inverse),
      &mut pss_signature,
    ),
    Err(RsaPrivateOpError::InvalidBlindingFactor)
  );
  assert!(is_zero_unsigned_be(&pss_signature));

  pss_signature.fill(0xa5);
  assert_eq!(
    key.sign_pss_with_salt_and_blinding_factor_and_scratch(
      RsaPssProfile::Sha256,
      message,
      &salt,
      RsaBlindingPair::new(&blinding_factor, &bad_blinding_inverse),
      &mut pss_signature,
      &mut scratch,
    ),
    Err(RsaPrivateOpError::InvalidBlindingFactor)
  );
  assert!(is_zero_unsigned_be(&pss_signature));

  let label = b"invalid-blinding-clears-oaep";
  let plaintext = b"rsa oaep invalid blinding clear";
  let seed = [0x3d; Sha256::OUTPUT_SIZE];
  let mut ciphertext = vec![0u8; key.signature_len()];
  key
    .public_key()
    .encrypt_oaep_with_random_fill(
      RsaOaepProfile::Sha256,
      label,
      plaintext,
      &mut ciphertext,
      seed_fill(&seed),
    )
    .expect("the OAEP invalid-blinding fixture must encrypt successfully");

  let mut decrypted = vec![0xa5; key.signature_len()];
  assert_eq!(
    key.decrypt_oaep_with_blinding_factor(
      RsaOaepProfile::Sha256,
      label,
      &ciphertext,
      RsaBlindingPair::new(&blinding_factor, &bad_blinding_inverse),
      &mut decrypted,
    ),
    Err(RsaPrivateOpError::InvalidBlindingFactor)
  );
  assert!(is_zero_unsigned_be(&decrypted));

  decrypted.fill(0xa5);
  assert_eq!(
    key.decrypt_oaep_with_blinding_factor_and_scratch(
      RsaOaepProfile::Sha256,
      label,
      &ciphertext,
      RsaBlindingPair::new(&blinding_factor, &bad_blinding_inverse),
      &mut decrypted,
      &mut scratch,
    ),
    Err(RsaPrivateOpError::InvalidBlindingFactor)
  );
  assert!(is_zero_unsigned_be(&decrypted));

  let pkcs1v15_plaintext = b"rsaes pkcs1v15 invalid blinding clear";
  let pkcs1v15_seed = vec![0x6d; key.signature_len().strict_sub(pkcs1v15_plaintext.len()).strict_sub(3)];
  let mut pkcs1v15_ciphertext = vec![0u8; key.signature_len()];
  key
    .public_key()
    .encrypt_pkcs1v15_with_random_fill(pkcs1v15_plaintext, &mut pkcs1v15_ciphertext, seed_fill(&pkcs1v15_seed))
    .expect("the PKCS#1 v1.5 invalid-blinding fixture must encrypt successfully");

  let mut pkcs1v15_decrypted = vec![0xa5; key.signature_len()];
  assert_eq!(
    key.decrypt_pkcs1v15_with_blinding_factor(
      &pkcs1v15_ciphertext,
      RsaBlindingPair::new(&blinding_factor, &bad_blinding_inverse),
      &mut pkcs1v15_decrypted,
    ),
    Err(RsaPrivateOpError::InvalidBlindingFactor)
  );
  assert!(is_zero_unsigned_be(&pkcs1v15_decrypted));

  pkcs1v15_decrypted.fill(0xa5);
  assert_eq!(
    key.decrypt_pkcs1v15_with_blinding_factor_and_scratch(
      &pkcs1v15_ciphertext,
      RsaBlindingPair::new(&blinding_factor, &bad_blinding_inverse),
      &mut pkcs1v15_decrypted,
      &mut scratch,
    ),
    Err(RsaPrivateOpError::InvalidBlindingFactor)
  );
  assert!(is_zero_unsigned_be(&pkcs1v15_decrypted));
}

#[cfg(feature = "getrandom")]
#[test]
fn private_key_getrandom_signs_and_decrypts_end_to_end() {
  let key = prevalidated_test_private_key();
  let message = b"rscrypto RSA getrandom signing roundtrip";
  let mut scratch = key.private_scratch();

  let mut pkcs1v15_signature = vec![0u8; key.signature_len()];
  key
    .sign_pkcs1v15(RsaPkcs1v15Profile::Sha256, message, &mut pkcs1v15_signature)
    .expect("PKCS#1 v1.5 signing with random blinding must succeed");
  key
    .public_key()
    .verify_pkcs1v15(RsaPkcs1v15Profile::Sha256, message, &pkcs1v15_signature)
    .expect("PKCS#1 v1.5 signature must verify");
  key
    .sign_pkcs1v15_with_scratch(
      RsaPkcs1v15Profile::Sha384,
      message,
      &mut pkcs1v15_signature,
      &mut scratch,
    )
    .expect("PKCS#1 v1.5 signing with caller scratch must succeed");
  key
    .public_key()
    .verify_pkcs1v15(RsaPkcs1v15Profile::Sha384, message, &pkcs1v15_signature)
    .expect("caller-scratch PKCS#1 v1.5 signature must verify");

  let mut pss_signature = vec![0u8; key.signature_len()];
  key
    .sign_pss(RsaPssProfile::Sha256, message, &mut pss_signature)
    .expect("PSS signing with random salt and blinding must succeed");
  key
    .public_key()
    .verify_pss(RsaPssProfile::Sha256, message, &pss_signature)
    .expect("PSS signature must verify");
  key
    .sign_pss_with_scratch(RsaPssProfile::Sha384, message, &mut pss_signature, &mut scratch)
    .expect("PSS signing with caller scratch must succeed");
  key
    .public_key()
    .verify_pss(RsaPssProfile::Sha384, message, &pss_signature)
    .expect("caller-scratch PSS signature must verify");
  let pss_zero_salt = RsaSignatureProfile::pss_with_salt_len(RsaPssProfile::Sha256, 0);
  key
    .sign_signature_with_scratch(pss_zero_salt, message, &mut pss_signature, &mut scratch)
    .expect("zero-salt PSS signing with caller scratch must succeed");
  key
    .public_key()
    .verify_signature(pss_zero_salt, message, &pss_signature)
    .expect("zero-salt PSS signature must verify");

  let label = b"rscrypto-getrandom-oaep";
  let plaintext = b"normal oaep api";
  let mut ciphertext = vec![0u8; key.signature_len()];
  key
    .public_key()
    .encrypt_oaep(RsaOaepProfile::Sha256, label, plaintext, &mut ciphertext)
    .expect("OAEP encryption with random seed must succeed");
  let mut decrypted = vec![0u8; key.signature_len()];
  let decrypted_len = key
    .decrypt_oaep(RsaOaepProfile::Sha256, label, &ciphertext, &mut decrypted)
    .expect("OAEP ciphertext must decrypt with random blinding");
  assert_eq!(&decrypted[..decrypted_len], plaintext);
  let decrypted_len = key
    .decrypt_oaep_with_scratch(RsaOaepProfile::Sha256, label, &ciphertext, &mut decrypted, &mut scratch)
    .expect("OAEP ciphertext must decrypt with caller scratch");
  assert_eq!(&decrypted[..decrypted_len], plaintext);

  let pkcs1v15_plaintext = b"normal pkcs1v15 encryption api";
  key
    .public_key()
    .encrypt_pkcs1v15(pkcs1v15_plaintext, &mut ciphertext)
    .expect("PKCS#1 v1.5 encryption with random padding must succeed");
  let decrypted_len = key
    .decrypt_pkcs1v15(&ciphertext, &mut decrypted)
    .expect("PKCS#1 v1.5 ciphertext must decrypt with random blinding");
  assert_eq!(&decrypted[..decrypted_len], pkcs1v15_plaintext);
  let decrypted_len = key
    .decrypt_pkcs1v15_with_scratch(&ciphertext, &mut decrypted, &mut scratch)
    .expect("PKCS#1 v1.5 ciphertext must decrypt with caller scratch");
  assert_eq!(&decrypted[..decrypted_len], pkcs1v15_plaintext);
}

#[cfg(feature = "getrandom")]
#[test]
fn signing_workspace_squares_its_pair_and_refreshes_from_its_blinding_source_every_32_signatures() {
  let key = prevalidated_test_private_key();
  let mut workspace = key.signing_workspace();
  let mut signature = vec![0u8; key.signature_len()];
  assert_workspace_pair_is_clear(&workspace);

  // Signatures 1, 33, and 65 refresh; PSS and PKCS #1 v1.5 share the epoch.
  for index in 0usize..70 {
    let use_in_epoch = index.strict_rem(32).strict_add(1);
    let profile = if index.is_multiple_of(3) {
      RsaSignatureProfile::pss(RsaPssProfile::Sha256)
    } else {
      RsaSignatureProfile::pkcs1v15(RsaPkcs1v15Profile::Sha256)
    };
    let message = format!("rscrypto signing workspace signature {index}");
    let (result, requests) =
      sign_with_workspace_and_factor_two(&key, profile, message.as_bytes(), &mut signature, &mut workspace);
    result.expect("signing through the workspace must succeed");

    // The salt source gets only the PSS salt. The blinding source gets only a
    // refresh: one candidate and one inverse mask.
    let expected = Requests {
      salt: usize::from(matches!(profile, RsaSignatureProfile::Pss { .. })),
      blinding: if use_in_epoch == 1 { 2 } else { 0 },
    };
    assert_eq!(requests, expected, "signature {index}");
    assert_eq!(usize::from(workspace.remaining), 32usize.strict_sub(use_in_epoch));
    let (power, inverse) = factor_two_epoch_pair(&key, use_in_epoch);
    assert_eq!(workspace.blinding_power.as_slice(), power, "signature {index}");
    assert_eq!(workspace.blinding_inverse.as_slice(), inverse, "signature {index}");
    key
      .public_key()
      .verify_signature(profile, message.as_bytes(), &signature)
      .expect("every signature through the workspace must verify");
    assert_private_scratch_is_clear(&workspace.scratch);
  }
}

#[cfg(feature = "getrandom")]
#[test]
fn signing_workspace_failures_clear_its_pair() {
  let key = prevalidated_test_private_key();
  let message = b"rscrypto signing workspace failures";
  let pkcs1v15 = RsaSignatureProfile::pkcs1v15(RsaPkcs1v15Profile::Sha256);
  let pss = RsaSignatureProfile::pss(RsaPssProfile::Sha256);
  let (factor, _) = factor_two_and_inverse(key.public_key().modulus());
  let (first_power, first_inverse) = factor_two_epoch_pair(&key, 1);
  let mut workspace = key.signing_workspace();
  let mut signature = vec![0u8; key.signature_len()];
  let sign = |profile, workspace: &mut RsaSigningWorkspace, out: &mut [u8]| {
    sign_with_workspace_and_factor_two(&key, profile, message, out, workspace)
  };
  let refresh = Requests { salt: 0, blinding: 2 };
  let reuse = Requests { salt: 0, blinding: 0 };
  let assert_refreshed = |workspace: &RsaSigningWorkspace| {
    assert_eq!(workspace.remaining, 31);
    assert_eq!(workspace.blinding_power.as_slice(), first_power);
    assert_eq!(workspace.blinding_inverse.as_slice(), first_inverse);
  };

  for expected in [&refresh, &reuse, &reuse] {
    let (result, requests) = sign(pkcs1v15, &mut workspace, &mut signature);
    result.expect("warm-up signing through the workspace must succeed");
    assert_eq!(&requests, expected);
  }
  assert_eq!(workspace.remaining, 29);

  // A public length error clears the pair before any entropy request.
  let mut short = vec![0xa5; key.signature_len().strict_sub(1)];
  assert_eq!(
    sign(pkcs1v15, &mut workspace, &mut short),
    (Err(RsaPrivateOpError::InvalidLength), reuse)
  );
  assert!(is_zero_unsigned_be(&short));
  assert_workspace_pair_is_clear(&workspace);
  assert_private_scratch_is_clear(&workspace.scratch);
  let (result, requests) = sign(pkcs1v15, &mut workspace, &mut signature);
  result.expect("signing after a length error must refresh the pair");
  assert_eq!(requests, refresh);
  assert_refreshed(&workspace);

  // A failed salt in a warm epoch, and a failed refresh, each clear the pair
  // after one request to the failing source.
  for warm in [true, false] {
    // The first case fails inside the epoch that the refresh above started;
    // the second fails the refresh of the pair that the first case cleared.
    assert_eq!(workspace.remaining > 0, warm, "warm: {warm}");
    let mut requests = Requests { salt: 0, blinding: 0 };
    signature.fill(0xa5);
    assert_eq!(
      key.sign_signature_with_random_fill_and_workspace(
        if warm { pss } else { pkcs1v15 },
        message,
        &mut signature,
        &mut workspace,
        |salt| {
          requests.salt = requests.salt.strict_add(1);
          salt.fill(0xa5);
          Err(())
        },
        |blinding| {
          requests.blinding = requests.blinding.strict_add(1);
          blinding.fill(0xa5);
          Err(())
        },
      ),
      Err(RsaPrivateOpError::EntropyUnavailable)
    );
    let expected = if warm {
      Requests { salt: 1, blinding: 0 }
    } else {
      Requests { salt: 0, blinding: 1 }
    };
    assert_eq!(requests, expected, "warm: {warm}");
    assert!(is_zero_unsigned_be(&signature));
    assert_workspace_pair_is_clear(&workspace);
    assert_private_scratch_is_clear(&workspace.scratch);
  }

  // An inconsistent cached pair fails the fault check, which clears the pair.
  sign(pkcs1v15, &mut workspace, &mut signature)
    .0
    .expect("signing must refresh the cleared pair");
  workspace.blinding_inverse.as_mut_slice().copy_from_slice(&factor);
  signature.fill(0xa5);
  assert_eq!(
    sign(pkcs1v15, &mut workspace, &mut signature),
    (Err(RsaPrivateOpError::FaultCheckFailed), reuse)
  );
  assert!(is_zero_unsigned_be(&signature));
  assert_workspace_pair_is_clear(&workspace);
  assert_private_scratch_is_clear(&workspace.scratch);
}

#[test]
fn private_key_caller_scratch_covers_signing_and_decryption() {
  let key = prevalidated_test_private_key();
  let (blinding_factor, blinding_factor_inverse) = factor_two_and_inverse(key.public_key().modulus());
  let mut scratch = key.private_scratch();
  let message = b"rscrypto RSA private scratch path";

  let mut pkcs1v15_signature = vec![0u8; key.signature_len()];
  key
    .sign_pkcs1v15_with_blinding_factor_and_scratch(
      RsaPkcs1v15Profile::Sha256,
      message,
      RsaBlindingPair::new(&blinding_factor, &blinding_factor_inverse),
      &mut pkcs1v15_signature,
      &mut scratch,
    )
    .expect("PKCS#1 v1.5 signing with explicit blinding and caller scratch must succeed");
  key
    .public_key()
    .verify_pkcs1v15(RsaPkcs1v15Profile::Sha256, message, &pkcs1v15_signature)
    .expect("caller-scratch PKCS#1 v1.5 signature must verify");

  let salt = [0x42; Sha384::OUTPUT_SIZE];
  let mut pss_signature = vec![0u8; key.signature_len()];
  key
    .sign_pss_with_salt_and_blinding_factor_and_scratch(
      RsaPssProfile::Sha384,
      message,
      &salt,
      RsaBlindingPair::new(&blinding_factor, &blinding_factor_inverse),
      &mut pss_signature,
      &mut scratch,
    )
    .expect("PSS signing with explicit salt, blinding, and caller scratch must succeed");
  key
    .public_key()
    .verify_pss(RsaPssProfile::Sha384, message, &pss_signature)
    .expect("caller-scratch PSS signature must verify");

  let label = b"rscrypto-private-scratch-oaep";
  let plaintext = b"private scratch OAEP roundtrip";
  let seed = [0x24; Sha256::OUTPUT_SIZE];
  let mut ciphertext = vec![0u8; key.signature_len()];
  key
    .public_key()
    .encrypt_oaep_with_random_fill(
      RsaOaepProfile::Sha256,
      label,
      plaintext,
      &mut ciphertext,
      seed_fill(&seed),
    )
    .expect("valid OAEP fixture inputs must encrypt");
  let mut decrypted = vec![0u8; key.signature_len()];
  let decrypted_len = key
    .decrypt_oaep_with_blinding_factor_and_scratch(
      RsaOaepProfile::Sha256,
      label,
      &ciphertext,
      RsaBlindingPair::new(&blinding_factor, &blinding_factor_inverse),
      &mut decrypted,
      &mut scratch,
    )
    .expect("OAEP ciphertext must decrypt with explicit blinding and caller scratch");
  assert_eq!(&decrypted[..decrypted_len], plaintext);

  let pkcs1v15_plaintext = b"private scratch RSAES-PKCS1-v1_5 roundtrip";
  let pkcs1v15_seed = vec![0x5a; key.signature_len().strict_sub(pkcs1v15_plaintext.len()).strict_sub(3)];
  key
    .public_key()
    .encrypt_pkcs1v15_with_random_fill(pkcs1v15_plaintext, &mut ciphertext, seed_fill(&pkcs1v15_seed))
    .expect("valid PKCS#1 v1.5 fixture inputs must encrypt");
  let decrypted_len = key
    .decrypt_pkcs1v15_with_blinding_factor_and_scratch(
      &ciphertext,
      RsaBlindingPair::new(&blinding_factor, &blinding_factor_inverse),
      &mut decrypted,
      &mut scratch,
    )
    .expect("PKCS#1 v1.5 ciphertext must decrypt with explicit blinding and caller scratch");
  assert_eq!(&decrypted[..decrypted_len], pkcs1v15_plaintext);

  let mut smaller_scratch = wrong_width_private_scratch(&key);
  assert_eq!(
    key.sign_pkcs1v15_with_blinding_factor_and_scratch(
      RsaPkcs1v15Profile::Sha256,
      message,
      RsaBlindingPair::new(&blinding_factor, &blinding_factor_inverse),
      &mut pkcs1v15_signature,
      &mut smaller_scratch,
    ),
    Err(RsaPrivateOpError::InvalidScratch)
  );
}

#[test]
fn private_key_caller_scratch_rejects_wrong_width_before_operation() {
  let key = prevalidated_test_private_key();
  let (blinding_factor, blinding_factor_inverse) = factor_two_and_inverse(key.public_key().modulus());
  let message = b"rscrypto RSA private scratch shape";
  let mut signature = vec![0xa5; key.signature_len()];
  let mut scratch = wrong_width_private_scratch(&key);

  assert_eq!(
    key.sign_pkcs1v15_with_blinding_factor_and_scratch(
      RsaPkcs1v15Profile::Sha256,
      message,
      RsaBlindingPair::new(&blinding_factor, &blinding_factor_inverse),
      &mut signature,
      &mut scratch,
    ),
    Err(RsaPrivateOpError::InvalidScratch)
  );
  assert!(is_zero_unsigned_be(&signature));
}
