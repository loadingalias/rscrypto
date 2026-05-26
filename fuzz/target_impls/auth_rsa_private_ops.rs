#[cfg(any(fuzzing, rscrypto_internal_fuzzing))]
use rscrypto::{
  RsaOaepProfile, RsaPkcs1v15Profile, RsaPrivateKey, RsaPssProfile, RsaPublicKeyPolicy, RsaSignatureProfile,
};
#[cfg(any(fuzzing, rscrypto_internal_fuzzing))]
use rscrypto_fuzz::{FuzzInput, some_or_return, split_at_ratio};

#[cfg(any(fuzzing, rscrypto_internal_fuzzing))]
#[allow(dead_code)]
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
pub fn run(data: &[u8]) {
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

  match mode % 10 {
    0 => {
      let profile = PKCS1_PROFILES[usize::from(selector) % PKCS1_PROFILES.len()];
      let mut signature = vec![0u8; key.signature_len()];
      key
        .sign_pkcs1v15_with_blinding_factor(profile, left, &one, &one_inverse, &mut signature)
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
        .sign_pss_with_salt_and_blinding_factor(profile, left, salt, &one, &one_inverse, &mut signature)
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
        .encrypt_oaep_with_seed(profile, label, message, &seed, &mut ciphertext)
        .expect("fixture RSA-OAEP encryption must succeed for bounded message");
      let mut plaintext = vec![0u8; key.signature_len()];
      let plaintext_len = key
        .decrypt_oaep_with_blinding_factor(profile, label, &ciphertext, &one, &one_inverse, &mut plaintext)
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
          &one,
          &one_inverse,
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
      let _ = key.decrypt_oaep_with_blinding_factor(profile, right, &ciphertext, &one, &one_inverse, &mut plaintext);
    }
    5 => {
      let profile = PKCS1_PROFILES[usize::from(selector) % PKCS1_PROFILES.len()];
      let mut signature = vec![0u8; key.signature_len()];
      let bad_factor = full_width_candidate(left, key.signature_len());
      let _ = key.sign_pkcs1v15_with_blinding_factor(profile, right, &bad_factor, &one_inverse, &mut signature);
    }
    6 => {
      let profile = PSS_PROFILES[usize::from(selector) % PSS_PROFILES.len()];
      let mut short_signature = vec![0u8; key.signature_len().saturating_sub(1)];
      assert!(key
        .sign_pss_with_salt_and_blinding_factor(profile, left, right, &one, &one_inverse, &mut short_signature)
        .is_err());
    }
    7 => {
      let profile = OAEP_PROFILES[usize::from(selector) % OAEP_PROFILES.len()];
      let seed = bounded_slice(right, profile.digest_len().saturating_sub(1));
      let mut ciphertext = vec![0u8; key.signature_len()];
      assert!(key.public_key().encrypt_oaep_with_seed(profile, left, right, seed, &mut ciphertext).is_err());
    }
    8 => {
      let message = bounded_slice(right, pkcs1v15_message_limit(&key));
      let seed = pkcs1v15_seed(&key, message.len(), selector, left, right);
      let mut ciphertext = vec![0u8; key.signature_len()];
      key
        .public_key()
        .encrypt_pkcs1v15_with_seed(message, &seed, &mut ciphertext)
        .expect("fixture RSAES-PKCS1-v1_5 encryption must succeed for bounded message");
      let mut plaintext = vec![0u8; key.signature_len()];
      let plaintext_len = key
        .decrypt_pkcs1v15_with_blinding_factor(&ciphertext, &one, &one_inverse, &mut plaintext)
        .expect("self-produced RSAES-PKCS1-v1_5 ciphertext must decrypt");
      assert_eq!(&plaintext[..plaintext_len], message);
    }
    9 => {
      let mut plaintext = vec![0u8; key.signature_len()];
      let ciphertext = full_width_candidate(left, key.signature_len());
      let _ = key.decrypt_pkcs1v15_with_blinding_factor(&ciphertext, &one, &one_inverse, &mut plaintext);
    }
    _ => unreachable!("mode modulo 10 is always in 0..10"),
  }
}

#[cfg(not(any(fuzzing, rscrypto_internal_fuzzing)))]
pub fn run(_data: &[u8]) {}

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
fn oaep_message_limit(key: &RsaPrivateKey, profile: RsaOaepProfile) -> usize {
  key.signature_len()
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
fn pkcs1v15_message_limit(key: &RsaPrivateKey) -> usize {
  key.signature_len().saturating_sub(11)
}

#[cfg(any(fuzzing, rscrypto_internal_fuzzing))]
fn pkcs1v15_seed(key: &RsaPrivateKey, message_len: usize, selector: u8, left: &[u8], right: &[u8]) -> Vec<u8> {
  let len = key
    .signature_len()
    .saturating_sub(message_len)
    .saturating_sub(3);
  let mut seed = vec![selector.wrapping_add(1).max(1); len];
  for (index, byte) in left.iter().chain(right.iter()).copied().enumerate() {
    let seed_len = seed.len();
    let value = seed[index % seed_len] ^ byte;
    seed[index % seed_len] = if value == 0 { 1 } else { value };
  }
  seed
}
