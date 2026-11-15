use hpke::{Deserializable as _, Kem as _, OpModeR, Serializable as _};
use rscrypto::{
  HpkeError, HpkeReceiver, HpkeSender, HpkeSuite, HpkeX25519HkdfSha256Aes128Gcm as X25519Suite,
  HpkeXWingHkdfSha256ChaCha20Poly1305 as XWingSuite, X25519PublicKey, XWingCiphertext,
};
use rscrypto_fuzz::{FuzzInput, some_or_return};

/// HPKE against rust-hpke: an rscrypto sender to both receivers, a changed
/// ciphertext that the rscrypto receiver rejects without a state change, and
/// receiver setup from arbitrary encapsulations.
pub(super) fn run(data: &[u8]) {
  let mut input = FuzzInput::new(data);
  let xwing = some_or_return!(input.byte()) & 1 == 1;
  let ikm: [u8; 32] = some_or_return!(input.bytes());
  let random: [u8; 64] = some_or_return!(input.bytes());
  let mutation = some_or_return!(input.bit_mutation());
  let (info, rest) = some_or_return!(input.split_rest());
  let (aad, plaintext) = rest.split_at(rest.len() / 2);
  let fill = |out: &mut [u8]| {
    out.copy_from_slice(&random[..out.len()]);
    Ok::<(), HpkeError>(())
  };

  if xwing {
    let recipient = XWingSuite::derive_keypair(&ikm);
    let (oracle_secret, oracle_public) = hpke::kem::XWing::derive_keypair(&ikm);
    assert_eq!(
      recipient.encapsulation_key().as_bytes().as_slice(),
      oracle_public.to_bytes().as_slice()
    );
    let public_key = recipient.encapsulation_key().prepare().expect("a derived key is valid");
    let (encapsulation, mut sender) =
      HpkeSender::<XWingSuite>::setup(&public_key, info, fill).expect("a derived key encapsulates");
    let ciphertext = sender.seal_to_vec(aad, plaintext).expect("seal");
    let oracle_encapsulation =
      <hpke::kem::XWing as hpke::Kem>::EncappedKey::from_bytes(encapsulation.as_bytes()).expect("oracle import");
    let mut oracle = hpke::setup_receiver::<hpke::aead::ChaCha20Poly1305, hpke::kdf::HkdfSha256, hpke::kem::XWing>(
      &OpModeR::Base,
      &oracle_secret,
      &oracle_encapsulation,
      info,
    )
    .expect("oracle setup");
    assert_eq!(oracle.open(&ciphertext, aad).expect("oracle open"), plaintext);
    let receiver = HpkeReceiver::<XWingSuite>::setup(&encapsulation, &recipient, info).expect("setup");
    rejects_changed_ciphertext(receiver, aad, plaintext, ciphertext, &mutation);
    if let Ok(arbitrary) = XWingCiphertext::try_from_slice(rest) {
      receiver_setup_is_total(HpkeReceiver::<XWingSuite>::setup(&arbitrary, &recipient, info));
    }
  } else {
    let recipient = X25519Suite::derive_keypair(&ikm);
    let (oracle_secret, oracle_public) = hpke::kem::X25519HkdfSha256::derive_keypair(&ikm);
    assert_eq!(
      recipient.public_key().as_bytes().as_slice(),
      oracle_public.to_bytes().as_slice()
    );
    let (encapsulation, mut sender) =
      HpkeSender::<X25519Suite>::setup(&recipient.public_key(), info, fill).expect("a derived key encapsulates");
    let ciphertext = sender.seal_to_vec(aad, plaintext).expect("seal");
    let oracle_encapsulation =
      <hpke::kem::X25519HkdfSha256 as hpke::Kem>::EncappedKey::from_bytes(encapsulation.as_bytes())
        .expect("oracle import");
    let mut oracle = hpke::setup_receiver::<hpke::aead::AesGcm128, hpke::kdf::HkdfSha256, hpke::kem::X25519HkdfSha256>(
      &OpModeR::Base,
      &oracle_secret,
      &oracle_encapsulation,
      info,
    )
    .expect("oracle setup");
    assert_eq!(oracle.open(&ciphertext, aad).expect("oracle open"), plaintext);
    let receiver = HpkeReceiver::<X25519Suite>::setup(&encapsulation, &recipient, info).expect("setup");
    rejects_changed_ciphertext(receiver, aad, plaintext, ciphertext, &mutation);
    if let Some(arbitrary) = rest.first_chunk::<32>() {
      receiver_setup_is_total(HpkeReceiver::<X25519Suite>::setup(
        &X25519PublicKey::from_bytes(*arbitrary),
        &recipient,
        info,
      ));
    }
  }
}

fn rejects_changed_ciphertext<S: HpkeSuite>(
  mut receiver: HpkeReceiver<S>,
  aad: &[u8],
  plaintext: &[u8],
  ciphertext: Vec<u8>,
  mutation: &rscrypto_fuzz::BitMutation,
) {
  let mut changed = ciphertext.clone();
  mutation.apply(&mut changed);
  assert_eq!(receiver.open_to_vec(aad, &changed).err(), Some(HpkeError::OpenFailed));
  assert_eq!(receiver.sequence(), 0);
  assert_eq!(
    receiver
      .open_to_vec(aad, &ciphertext)
      .expect("the genuine message opens"),
    plaintext
  );
}

/// A receiver setup either succeeds or rejects a small-order X25519 part.
fn receiver_setup_is_total<S: HpkeSuite>(setup: Result<HpkeReceiver<S>, HpkeError>) {
  if let Err(error) = setup {
    assert_eq!(error, HpkeError::DecapsulationFailed);
  }
}
