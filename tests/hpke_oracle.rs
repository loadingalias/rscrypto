//! HPKE against rust-hpke (`hpke` 0.15.1), an independent implementation of
//! RFC 9180 and draft-ietf-hpke-pq on RustCrypto components.
//!
//! Each direction runs: an rscrypto sender to an rust-hpke receiver, and an
//! rust-hpke sender to an rscrypto receiver, over several messages and exports.
#![cfg(all(
  feature = "hpke",
  feature = "x25519",
  feature = "aes-gcm",
  feature = "x-wing",
  feature = "chacha20poly1305",
  feature = "rand_core"
))]

use hpke::{Deserializable as _, Kem as _, OpModeR, OpModeS, Serializable as _};
use proptest::{prelude::*, test_runner::Config as ProptestConfig};
use rscrypto::{
  ChaCha20Rng, HpkeError, HpkeReceiver, HpkeSender, HpkeSuite, HpkeX25519HkdfSha256Aes128Gcm as X25519Suite,
  HpkeXWingHkdfSha256ChaCha20Poly1305 as XWingSuite, XWingCiphertext, XWingEncapsulationKey,
};

type OracleX25519 = hpke::kem::X25519HkdfSha256;
type OracleXWing = hpke::kem::XWing;
type OracleKdf = hpke::kdf::HkdfSha256;

const PROPTEST_CASES: u32 = if cfg!(debug_assertions) { 16 } else { 64 };
const MESSAGES: usize = 3;

fn bytes(max: usize) -> impl Strategy<Value = Vec<u8>> {
  proptest::collection::vec(any::<u8>(), 0..max)
}

fn filled(bytes: &[u8]) -> impl FnMut(&mut [u8]) -> Result<(), HpkeError> + '_ {
  |out| {
    out.copy_from_slice(bytes);
    Ok(())
  }
}

proptest! {
  #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]

  /// DeriveKeyPair agrees, and each side opens and exports what the other seals.
  #[test]
  fn x25519_suite_interoperates(
    ikm in bytes(64),
    random in proptest::array::uniform32(any::<u8>()),
    info in bytes(80),
    aad in bytes(40),
    plaintext in bytes(200),
    exporter_context in bytes(40),
  ) {
    let ours = X25519Suite::derive_keypair(&ikm);
    let (oracle_secret, oracle_public) = OracleX25519::derive_keypair(&ikm);
    // rust-hpke serializes the clamped scalar; RFC 9180 vectors keep the
    // unclamped bytes, which rscrypto matches. Both name one X25519 key.
    let mut clamped = *ours.secret_key().as_bytes();
    clamped[0] &= 248;
    clamped[31] = (clamped[31] & 127) | 64;
    prop_assert_eq!(clamped.to_vec(), oracle_secret.to_bytes().to_vec());
    prop_assert_eq!(ours.public_key().as_bytes().to_vec(), oracle_public.to_bytes().to_vec());

    let (encapsulation, mut sender) =
      HpkeSender::<X25519Suite>::setup(&ours.public_key(), &info, filled(&random)).expect("setup");
    let oracle_encapsulation =
      <OracleX25519 as hpke::Kem>::EncappedKey::from_bytes(encapsulation.as_bytes()).expect("oracle import");
    let mut oracle_receiver = hpke::setup_receiver::<hpke::aead::AesGcm128, OracleKdf, OracleX25519>(
      &OpModeR::Base, &oracle_secret, &oracle_encapsulation, &info,
    ).expect("oracle setup");
    for _ in 0..MESSAGES {
      let ciphertext = sender.seal_to_vec(&aad, &plaintext).expect("seal");
      prop_assert_eq!(oracle_receiver.open(&ciphertext, &aad).expect("oracle open"), plaintext.clone());
    }
    let (mut ours_export, mut oracle_export) = ([0u8; 48], [0u8; 48]);
    sender.export(&exporter_context, &mut ours_export).expect("export");
    oracle_receiver.export(&exporter_context, &mut oracle_export).expect("oracle export");
    prop_assert_eq!(ours_export, oracle_export);

    let mut rng = ChaCha20Rng::from_seed(random);
    let (oracle_encapsulation, mut oracle_sender) =
      hpke::setup_sender_with_rng::<hpke::aead::AesGcm128, OracleKdf, OracleX25519>(
        &OpModeS::Base, &oracle_public, &info, &mut rng,
      ).expect("oracle setup");
    let encapsulation = rscrypto::X25519PublicKey::from_bytes(
      oracle_encapsulation.to_bytes().as_slice().try_into().expect("32-byte encapsulation"),
    );
    let mut receiver = HpkeReceiver::<X25519Suite>::setup(&encapsulation, &ours, &info).expect("setup");
    for _ in 0..MESSAGES {
      let ciphertext = oracle_sender.seal(&plaintext, &aad).expect("oracle seal");
      prop_assert_eq!(receiver.open_to_vec(&aad, &ciphertext).expect("open"), plaintext.clone());
    }
  }

  /// The same for the X-Wing suite, through the prepared X-Wing keys.
  #[test]
  fn xwing_suite_interoperates(
    ikm in bytes(64),
    random in proptest::array::uniform32(any::<u8>()),
    info in bytes(80),
    aad in bytes(40),
    plaintext in bytes(200),
  ) {
    let ours = XWingSuite::derive_keypair(&ikm);
    let (oracle_secret, oracle_public) = OracleXWing::derive_keypair(&ikm);
    prop_assert_eq!(ours.decapsulation_key().as_bytes().to_vec(), oracle_secret.to_bytes().to_vec());
    prop_assert_eq!(ours.encapsulation_key().as_bytes().to_vec(), oracle_public.to_bytes().to_vec());

    let public_key = XWingEncapsulationKey::try_from_slice(&oracle_public.to_bytes())
      .expect("oracle key")
      .prepare();
    let mut randomness = [0u8; 64];
    randomness[..32].copy_from_slice(&random);
    randomness[32..].copy_from_slice(&random);
    randomness[32] ^= 1;
    let (encapsulation, mut sender) =
      HpkeSender::<XWingSuite>::setup(&public_key, &info, filled(&randomness)).expect("setup");
    let oracle_encapsulation =
      <OracleXWing as hpke::Kem>::EncappedKey::from_bytes(encapsulation.as_bytes()).expect("oracle import");
    let mut oracle_receiver = hpke::setup_receiver::<hpke::aead::ChaCha20Poly1305, OracleKdf, OracleXWing>(
      &OpModeR::Base, &oracle_secret, &oracle_encapsulation, &info,
    ).expect("oracle setup");
    for _ in 0..MESSAGES {
      let ciphertext = sender.seal_to_vec(&aad, &plaintext).expect("seal");
      prop_assert_eq!(oracle_receiver.open(&ciphertext, &aad).expect("oracle open"), plaintext.clone());
    }

    let mut rng = ChaCha20Rng::from_seed(random);
    let (oracle_encapsulation, mut oracle_sender) =
      hpke::setup_sender_with_rng::<hpke::aead::ChaCha20Poly1305, OracleKdf, OracleXWing>(
        &OpModeS::Base, &oracle_public, &info, &mut rng,
      ).expect("oracle setup");
    let encapsulation =
      XWingCiphertext::try_from_slice(&oracle_encapsulation.to_bytes()).expect("1120-byte encapsulation");
    let mut receiver = HpkeReceiver::<XWingSuite>::setup(&encapsulation, &ours, &info).expect("setup");
    for _ in 0..MESSAGES {
      let ciphertext = oracle_sender.seal(&plaintext, &aad).expect("oracle seal");
      prop_assert_eq!(receiver.open_to_vec(&aad, &ciphertext).expect("open"), plaintext.clone());
    }
  }
}
