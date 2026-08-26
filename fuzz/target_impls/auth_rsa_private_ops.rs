#[cfg(any(fuzzing, rscrypto_internal_fuzzing))]
use rscrypto::{
  RsaBlindingPair, RsaEncryptionError, RsaOaepProfile, RsaPkcs1v15Profile, RsaPrivateKey, RsaPrivateOpError,
  RsaPssProfile, RsaPublicKeyPolicy, RsaSignatureProfile,
};
#[cfg(any(fuzzing, rscrypto_internal_fuzzing))]
use rscrypto_fuzz::{FuzzInput, some_or_return, split_at_ratio};

#[cfg(any(fuzzing, rscrypto_internal_fuzzing))]
#[path = "auth_rsa_import.rs"]
mod rsa_import_fixture;

#[cfg(any(fuzzing, rscrypto_internal_fuzzing))]
const PKCS1_PROFILES: [RsaPkcs1v15Profile; 3] = [
  RsaPkcs1v15Profile::Sha256,
  RsaPkcs1v15Profile::Sha384,
  RsaPkcs1v15Profile::Sha512,
];
#[cfg(any(fuzzing, rscrypto_internal_fuzzing))]
const PSS_PROFILES: [RsaPssProfile; 3] = [RsaPssProfile::Sha256, RsaPssProfile::Sha384, RsaPssProfile::Sha512];
#[cfg(any(fuzzing, rscrypto_internal_fuzzing))]
const OAEP_PROFILES: [RsaOaepProfile; 3] = [RsaOaepProfile::Sha256, RsaOaepProfile::Sha384, RsaOaepProfile::Sha512];

#[cfg(any(fuzzing, rscrypto_internal_fuzzing))]
pub(super) fn run(data: &[u8]) {
  let mut input = FuzzInput::new(data);
  let mode = some_or_return!(input.byte());
  let selector = some_or_return!(input.byte());
  let split = some_or_return!(input.byte());
  let (left, right) = split_at_ratio(input.rest(), split);

  let key = RsaPrivateKey::from_pkcs1_der_with_policy(
    &rsa_import_fixture::valid_pkcs1_private_key_der(),
    &RsaPublicKeyPolicy::legacy_verification(),
  )
  .expect("fuzz RSA private-key fixture must parse");
  let (one, one_inverse) = factor_one(key.signature_len());

  match mode.rem_euclid(13) {
    0 => {
      let profile = PKCS1_PROFILES[usize::from(selector) % PKCS1_PROFILES.len()];
      let mut signature = vec![0u8; key.signature_len()];
      key
        .sign_pkcs1v15_with_blinding_factor(profile, left, RsaBlindingPair::new(&one, &one_inverse), &mut signature)
        .expect("fixture RSA-PKCS1-v1_5 signing must succeed");
      key
        .public_key()
        .verify_pkcs1v15(profile, left, &signature)
        .expect("self-produced RSA-PKCS1-v1_5 signature must verify");
    }
    1 => {
      let profile = PSS_PROFILES[usize::from(selector) % PSS_PROFILES.len()];
      let salt = bounded_slice(right, profile.digest_len());
      let mut signature = vec![0u8; key.signature_len()];
      key
        .sign_pss_with_salt_and_blinding_factor(
          profile,
          left,
          salt,
          RsaBlindingPair::new(&one, &one_inverse),
          &mut signature,
        )
        .expect("fixture RSA-PSS signing must succeed");
      key
        .public_key()
        .verify_pss_with_salt_len(profile, salt.len(), left, &signature)
        .expect("self-produced RSA-PSS signature must verify");
    }
    2 => {
      let profile = OAEP_PROFILES[usize::from(selector) % OAEP_PROFILES.len()];
      let label = left;
      let message = bounded_slice(right, oaep_message_limit(&key, profile));
      let seed = oaep_seed(profile, selector, left, right);
      let mut ciphertext = vec![0u8; key.signature_len()];
      key
        .public_key()
        .encrypt_oaep_with_random_fill(profile, label, message, &mut ciphertext, fill_random_from(&seed))
        .expect("fixture RSA-OAEP encryption must succeed for bounded message");
      let mut plaintext = vec![0u8; key.signature_len()];
      let plaintext_len = key
        .decrypt_oaep_with_blinding_factor(
          profile,
          label,
          &ciphertext,
          RsaBlindingPair::new(&one, &one_inverse),
          &mut plaintext,
        )
        .expect("self-produced RSA-OAEP ciphertext must decrypt");
      assert_eq!(&plaintext[..plaintext_len], message);
    }
    3 => {
      let profile = RsaSignatureProfile::pkcs1v15(PKCS1_PROFILES[usize::from(selector) % PKCS1_PROFILES.len()]);
      let mut signature = vec![0u8; key.signature_len()];
      key
        .sign_pkcs1v15_with_blinding_factor(
          profile.pkcs1v15_profile().expect("profile is PKCS1-v1_5"),
          left,
          RsaBlindingPair::new(&one, &one_inverse),
          &mut signature,
        )
        .expect("typed fixture RSA-PKCS1-v1_5 signing must succeed");
      key
        .public_key()
        .verify_signature(profile, left, &signature)
        .expect("typed self-produced RSA-PKCS1-v1_5 signature must verify");
    }
    4 => {
      let profile = OAEP_PROFILES[usize::from(selector) % OAEP_PROFILES.len()];
      let mut plaintext = vec![0u8; key.signature_len()];
      let ciphertext = full_width_candidate(left, key.signature_len());
      let _decryption_result = key.decrypt_oaep_with_blinding_factor(
        profile,
        right,
        &ciphertext,
        RsaBlindingPair::new(&one, &one_inverse),
        &mut plaintext,
      );
    }
    5 => {
      let profile = PKCS1_PROFILES[usize::from(selector) % PKCS1_PROFILES.len()];
      let mut signature = vec![0u8; key.signature_len()];
      let bad_factor = full_width_candidate(left, key.signature_len());
      let _signing_result = key.sign_pkcs1v15_with_blinding_factor(
        profile,
        right,
        RsaBlindingPair::new(&bad_factor, &one_inverse),
        &mut signature,
      );
    }
    6 => {
      let profile = PSS_PROFILES[usize::from(selector) % PSS_PROFILES.len()];
      let mut short_signature = vec![0u8; key.signature_len().saturating_sub(1)];
      let _signing_error = key
        .sign_pss_with_salt_and_blinding_factor(
          profile,
          left,
          right,
          RsaBlindingPair::new(&one, &one_inverse),
          &mut short_signature,
        )
        .expect_err("RSA-PSS signing must reject a short output buffer");
    }
    7 => {
      let profile = OAEP_PROFILES[usize::from(selector) % OAEP_PROFILES.len()];
      let mut ciphertext = vec![0u8; key.signature_len()];
      let _encryption_error = key
        .public_key()
        .encrypt_oaep_with_random_fill(profile, left, right, &mut ciphertext, |_| {
          Err(RsaEncryptionError::EntropyUnavailable)
        })
        .expect_err("RSA-OAEP encryption must propagate entropy failure");
    }
    8 => {
      let message = bounded_slice(right, pkcs1v15_message_limit(&key));
      let seed = pkcs1v15_seed(&key, message.len(), selector, left, right);
      let mut ciphertext = vec![0u8; key.signature_len()];
      key
        .public_key()
        .encrypt_pkcs1v15_with_random_fill(message, &mut ciphertext, fill_random_from(&seed))
        .expect("fixture RSAES-PKCS1-v1_5 encryption must succeed for bounded message");
      let mut plaintext = vec![0u8; key.signature_len()];
      let plaintext_len = key
        .decrypt_pkcs1v15_with_blinding_factor(&ciphertext, RsaBlindingPair::new(&one, &one_inverse), &mut plaintext)
        .expect("self-produced RSAES-PKCS1-v1_5 ciphertext must decrypt");
      assert_eq!(&plaintext[..plaintext_len], message);
    }
    9 => {
      let mut plaintext = vec![0u8; key.signature_len()];
      let ciphertext = full_width_candidate(left, key.signature_len());
      let _decryption_result = key.decrypt_pkcs1v15_with_blinding_factor(
        &ciphertext,
        RsaBlindingPair::new(&one, &one_inverse),
        &mut plaintext,
      );
    }
    10 => {
      let profile = PSS_PROFILES[usize::from(selector) % PSS_PROFILES.len()];
      let salt = caller_pss_salt(profile, selector, left, right);
      let mut calls = 0usize;
      let mut signature = vec![0u8; key.signature_len()];
      key
        .sign_pss_with_random_fill(profile, left, &mut signature, |out| {
          if calls == 0 {
            out.copy_from_slice(&salt);
          } else {
            out.copy_from_slice(&one);
          }
          calls = calls.strict_add(1);
          Ok::<(), ()>(())
        })
        .expect("fixture caller-random RSA-PSS signing must succeed");
      assert_eq!(calls, 2);
      key
        .public_key()
        .verify_pss(profile, left, &signature)
        .expect("self-produced caller-random RSA-PSS signature must verify");
    }
    11 => {
      let profile = PSS_PROFILES[usize::from(selector) % PSS_PROFILES.len()];
      let fail_at = usize::from(split) % 130;
      let mut calls = 0usize;
      let mut signature = vec![0xa5; key.signature_len()];
      let mut scratch = key.private_scratch();
      let result = key.sign_pss_with_random_fill_and_scratch(profile, left, &mut signature, &mut scratch, |out| {
        let call = calls;
        calls = calls.strict_add(1);
        out.fill(0);
        if call == fail_at { Err(()) } else { Ok(()) }
      });
      assert!(matches!(
        result,
        Err(RsaPrivateOpError::EntropyUnavailable | RsaPrivateOpError::InvalidBlindingFactor)
      ));
      assert!(signature.iter().all(|&byte| byte == 0));

      key
        .sign_pss_with_random_fill_and_scratch(profile, right, &mut signature, &mut scratch, |out| {
          if out.len() == key.signature_len() {
            out.copy_from_slice(&one);
          } else {
            out.fill(selector);
          }
          Ok::<(), ()>(())
        })
        .expect("caller-random RSA scratch must be reusable after a scheduled entropy failure");
      key
        .public_key()
        .verify_pss(profile, right, &signature)
        .expect("caller-random RSA-PSS signature after scratch reuse must verify");
    }
    12 => {
      const TLS_SCHEMES: [u16; 9] = [0x0401, 0x0501, 0x0601, 0x0804, 0x0805, 0x0806, 0x0809, 0x080a, 0x080b];
      let scheme = TLS_SCHEMES[usize::from(selector) % TLS_SCHEMES.len()];
      let profile = RsaSignatureProfile::from_tls_certificate_signature_scheme(scheme)
        .expect("fixture TLS scheme must map to an RSA profile");
      let mut signature = vec![0u8; key.signature_len()];
      key
        .sign_tls_certificate_signature_scheme_with_random_fill(scheme, left, &mut signature, |out| {
          if out.len() == key.signature_len() {
            out.copy_from_slice(&one);
          } else {
            out.fill(split);
          }
          Ok::<(), ()>(())
        })
        .expect("fixture TLS caller-random RSA signing must succeed");
      key
        .public_key()
        .verify_signature(profile, left, &signature)
        .expect("fixture TLS caller-random RSA signature must verify");
    }
    _ => {}
  }
}

#[cfg(not(any(fuzzing, rscrypto_internal_fuzzing)))]
pub(super) fn run(_data: &[u8]) {}

#[cfg(any(fuzzing, rscrypto_internal_fuzzing))]
fn factor_one(len: usize) -> (Vec<u8>, Vec<u8>) {
  let mut one = vec![0u8; len];
  if let Some(last) = one.last_mut() {
    *last = 1;
  }
  (one.clone(), one)
}

#[cfg(any(fuzzing, rscrypto_internal_fuzzing))]
fn bounded_slice(input: &[u8], max_len: usize) -> &[u8] {
  &input[..input.len().min(max_len)]
}

#[cfg(any(fuzzing, rscrypto_internal_fuzzing))]
fn full_width_candidate(input: &[u8], len: usize) -> Vec<u8> {
  let mut out = vec![0u8; len];
  if input.is_empty() {
    return out;
  }
  for (index, byte) in out.iter_mut().enumerate() {
    *byte = input[index % input.len()];
  }
  out
}

#[cfg(any(fuzzing, rscrypto_internal_fuzzing))]
fn fill_random_from(bytes: &[u8]) -> impl FnMut(&mut [u8]) -> Result<(), RsaEncryptionError> + '_ {
  let mut offset = 0usize;
  move |out| {
    let end = offset.strict_add(out.len());
    if end > bytes.len() {
      return Err(RsaEncryptionError::EntropyUnavailable);
    }
    out.copy_from_slice(&bytes[offset..end]);
    offset = end;
    Ok(())
  }
}

#[cfg(any(fuzzing, rscrypto_internal_fuzzing))]
fn oaep_message_limit(key: &RsaPrivateKey, profile: RsaOaepProfile) -> usize {
  key
    .signature_len()
    .saturating_sub(profile.digest_len().saturating_mul(2))
    .saturating_sub(2)
}

#[cfg(any(fuzzing, rscrypto_internal_fuzzing))]
fn oaep_seed(profile: RsaOaepProfile, selector: u8, left: &[u8], right: &[u8]) -> Vec<u8> {
  let mut seed = vec![selector; profile.digest_len()];
  for (index, byte) in left.iter().chain(right.iter()).copied().enumerate() {
    let seed_len = seed.len();
    seed[index % seed_len] ^= byte;
  }
  seed
}

#[cfg(any(fuzzing, rscrypto_internal_fuzzing))]
fn caller_pss_salt(profile: RsaPssProfile, selector: u8, left: &[u8], right: &[u8]) -> Vec<u8> {
  let mut salt = vec![selector; profile.digest_len()];
  for (index, byte) in left.iter().chain(right.iter()).copied().enumerate() {
    let salt_len = salt.len();
    salt[index % salt_len] ^= byte;
  }
  salt
}

#[cfg(any(fuzzing, rscrypto_internal_fuzzing))]
fn pkcs1v15_message_limit(key: &RsaPrivateKey) -> usize {
  key.signature_len().saturating_sub(11)
}

#[cfg(any(fuzzing, rscrypto_internal_fuzzing))]
fn pkcs1v15_seed(key: &RsaPrivateKey, message_len: usize, selector: u8, left: &[u8], right: &[u8]) -> Vec<u8> {
  let len = key.signature_len().saturating_sub(message_len).saturating_sub(3);
  let mut seed = vec![selector.wrapping_add(1).max(1); len];
  for (index, byte) in left.iter().chain(right.iter()).copied().enumerate() {
    let seed_len = seed.len();
    let value = seed[index % seed_len] ^ byte;
    seed[index % seed_len] = if value == 0 { 1 } else { value };
  }
  seed
}
