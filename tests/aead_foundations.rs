#[cfg(feature = "aead")]
use rscrypto::{
  Aead, VerificationError,
  aead::{AeadBufferError, Nonce96, OpenError},
};

#[cfg(feature = "aead")]
#[derive(Clone, PartialEq, Eq)]
struct ToyKey([u8; 32]);

#[cfg(feature = "aead")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ToyTag([u8; 16]);

#[cfg(feature = "aead")]
impl AsRef<[u8]> for ToyTag {
  fn as_ref(&self) -> &[u8] {
    &self.0
  }
}

#[cfg(feature = "aead")]
#[derive(Clone)]
struct ToyAead {
  key: ToyKey,
}

#[cfg(feature = "aead")]
impl ToyAead {
  fn keystream_byte(&self, nonce: &Nonce96, aad: &[u8]) -> u8 {
    self.key.0[0] ^ nonce.as_bytes()[0] ^ (aad.len() as u8)
  }

  fn compute_tag(&self, nonce: &Nonce96, aad: &[u8], buffer: &[u8]) -> ToyTag {
    let mut tag = [0u8; 16];
    tag[0] = self.key.0[0];
    tag[1] = nonce.as_bytes()[0];
    tag[2] = aad.len() as u8;
    tag[3] = buffer.len() as u8;

    for (index, &byte) in buffer.iter().enumerate() {
      let lane = index % 12;
      tag[lane.strict_add(4)] ^= byte;
    }

    ToyTag(tag)
  }
}

#[cfg(feature = "aead")]
impl Aead for ToyAead {
  const KEY_SIZE: usize = 32;
  const NONCE_SIZE: usize = Nonce96::LENGTH;
  const TAG_SIZE: usize = 16;

  type Key = ToyKey;
  type Nonce = Nonce96;
  type Tag = ToyTag;

  fn new(key: &Self::Key) -> Self {
    Self { key: key.clone() }
  }

  fn tag_from_slice(bytes: &[u8]) -> Result<Self::Tag, AeadBufferError> {
    if bytes.len() != Self::TAG_SIZE {
      return Err(AeadBufferError::new());
    }

    let mut tag = [0u8; Self::TAG_SIZE];
    tag.copy_from_slice(bytes);
    Ok(ToyTag(tag))
  }

  fn encrypt_in_place(&self, nonce: &Self::Nonce, aad: &[u8], buffer: &mut [u8]) -> Self::Tag {
    let mask = self.keystream_byte(nonce, aad);
    for byte in buffer.iter_mut() {
      *byte ^= mask;
    }
    self.compute_tag(nonce, aad, buffer)
  }

  fn decrypt_in_place(
    &self,
    nonce: &Self::Nonce,
    aad: &[u8],
    buffer: &mut [u8],
    tag: &Self::Tag,
  ) -> Result<(), VerificationError> {
    if self.compute_tag(nonce, aad, buffer) != *tag {
      return Err(VerificationError::new());
    }

    let mask = self.keystream_byte(nonce, aad);
    for byte in buffer.iter_mut() {
      *byte ^= mask;
    }
    Ok(())
  }
}

#[test]
#[cfg(feature = "aead")]
fn nonce96_round_trips() {
  let nonce = Nonce96::from_bytes([0xA5; Nonce96::LENGTH]);

  assert_eq!(nonce.to_bytes(), [0xA5; Nonce96::LENGTH]);
  assert_eq!(nonce.as_bytes(), &[0xA5; Nonce96::LENGTH]);
}

#[test]
#[cfg(feature = "aead")]
fn aead_encrypt_and_decrypt_helpers_round_trip() {
  let key = ToyKey([0x11; 32]);
  let nonce = Nonce96::from_bytes([0x22; Nonce96::LENGTH]);
  let aad = b"header";
  let plaintext = *b"hello world!";
  let aead = ToyAead::new(&key);

  let mut sealed = [0u8; 28];
  aead.encrypt(&nonce, aad, &plaintext, &mut sealed).unwrap();

  let mut opened = [0u8; 12];
  aead.decrypt(&nonce, aad, &sealed, &mut opened).unwrap();

  assert_eq!(opened, plaintext);
}

#[test]
#[cfg(feature = "aead")]
fn detached_aliases_match_core_behavior() {
  let key = ToyKey([0x44; 32]);
  let nonce = Nonce96::from_bytes([0x55; Nonce96::LENGTH]);
  let aad = b"aad";
  let aead = ToyAead::new(&key);

  let mut left = *b"detached";
  let mut right = left;

  let tag_left = aead.encrypt_in_place(&nonce, aad, &mut left);
  let tag_right = aead.encrypt_in_place_detached(&nonce, aad, &mut right);

  assert_eq!(left, right);
  assert_eq!(tag_left, tag_right);

  aead
    .decrypt_in_place_detached(&nonce, aad, &mut right, &tag_right)
    .unwrap();
  assert_eq!(right, *b"detached");
}

#[test]
#[cfg(feature = "aead")]
fn aead_open_reports_buffer_and_verification_failures() {
  let key = ToyKey([0x77; 32]);
  let nonce = Nonce96::from_bytes([0x88; Nonce96::LENGTH]);
  let aead = ToyAead::new(&key);

  let mut sealed = [0u8; 20];
  aead.encrypt(&nonce, b"aad", b"data", &mut sealed).unwrap();

  let mut short_out = [0u8; 3];
  assert_eq!(
    aead.decrypt(&nonce, b"aad", &sealed, &mut short_out),
    Err(OpenError::buffer())
  );

  sealed[0] ^= 1;
  let mut opened = [0u8; 4];
  assert_eq!(
    aead.decrypt(&nonce, b"aad", &sealed, &mut opened),
    Err(OpenError::verification())
  );
}

#[test]
#[cfg(feature = "aead")]
fn aead_length_helpers_reject_invalid_sizes() {
  assert_eq!(ToyAead::ciphertext_len(5).unwrap(), 21);
  assert_eq!(ToyAead::plaintext_len(21).unwrap(), 5);
  assert_eq!(ToyAead::plaintext_len(15), Err(AeadBufferError::new()));
}
