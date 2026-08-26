#![cfg(any(feature = "aes-gcm", feature = "chacha20poly1305"))]

#[cfg(feature = "aes-gcm")]
use aes::cipher::{Array, BlockCipherEncrypt as _, KeyInit as _};
#[cfg(feature = "chacha20poly1305")]
use chacha20::cipher::{KeyIvInit as _, StreamCipherCore as _};
#[cfg(feature = "aes-gcm")]
use rscrypto::aead::expert::header_protection::{
  Aes128HeaderProtection, Aes128HeaderProtectionKey, Aes256HeaderProtection, Aes256HeaderProtectionKey,
};
#[cfg(feature = "chacha20poly1305")]
use rscrypto::aead::expert::header_protection::{ChaCha20HeaderProtection, ChaCha20HeaderProtectionKey};

#[cfg(miri)]
const GENERATED_CASES: usize = 8;
#[cfg(not(miri))]
const GENERATED_CASES: usize = 512;

fn decode_hex<const N: usize>(hex: &str) -> [u8; N] {
  fn nibble(byte: u8) -> Option<u8> {
    match byte {
      b'0'..=b'9' => Some(byte.strict_sub(b'0')),
      b'a'..=b'f' => Some(byte.strict_sub(b'a').strict_add(10)),
      b'A'..=b'F' => Some(byte.strict_sub(b'A').strict_add(10)),
      _ => None,
    }
  }

  assert_eq!(hex.len(), N.strict_mul(2), "test fixture must have exact length");
  let mut out = [0u8; N];
  for (dst, pair) in out.iter_mut().zip(hex.as_bytes().as_chunks::<2>().0) {
    let high = nibble(pair[0]).expect("test fixture must contain hexadecimal bytes");
    let low = nibble(pair[1]).expect("test fixture must contain hexadecimal bytes");
    *dst = high.strict_shl(4) | low;
  }
  out
}

fn generated_bytes<const N: usize>(state: &mut u64) -> [u8; N] {
  let mut out = [0u8; N];
  for byte in &mut out {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *byte = state.to_le_bytes()[0];
  }
  out
}

#[cfg(feature = "aes-gcm")]
fn aes128_oracle(key: &[u8; 16], sample: &[u8; 16]) -> [u8; 5] {
  let cipher = aes::Aes128::new(&Array::from(*key));
  let mut block = Array::from(*sample);
  cipher.encrypt_block(&mut block);
  block[..5].try_into().expect("five-byte prefix has fixed length")
}

#[cfg(feature = "aes-gcm")]
fn aes256_oracle(key: &[u8; 32], sample: &[u8; 16]) -> [u8; 5] {
  let cipher = aes::Aes256::new(&Array::from(*key));
  let mut block = Array::from(*sample);
  cipher.encrypt_block(&mut block);
  block[..5].try_into().expect("five-byte prefix has fixed length")
}

#[cfg(feature = "chacha20poly1305")]
fn chacha20_oracle(key: &[u8; 32], sample: &[u8; 16]) -> [u8; 5] {
  let counter = u32::from_le_bytes(
    sample[..4]
      .try_into()
      .expect("four-byte counter prefix has fixed length"),
  );
  let nonce: [u8; 12] = sample[4..]
    .try_into()
    .expect("twelve-byte nonce suffix has fixed length");
  let mut core = chacha20::ChaChaCore::<chacha20::R20, chacha20::variants::Ietf>::new(key.into(), (&nonce).into());
  core.set_block_pos(counter);
  let mut block = chacha20::cipher::array::Array::<u8, chacha20::cipher::consts::U64>::default();
  core.write_keystream_block(&mut block);
  block[..5].try_into().expect("five-byte prefix has fixed length")
}

#[cfg(feature = "aes-gcm")]
#[test]
fn rfc9001_aes128_client_and_server_initial_masks() {
  let client_key = decode_hex("9f50449e04a0e810283a1e9933adedd2");
  let client_sample = decode_hex("d1b1c98dd7689fb8ec11d242b123dc9b");
  let client = Aes128HeaderProtection::new(&Aes128HeaderProtectionKey::from_bytes(client_key));
  assert_eq!(client.mask(&client_sample), decode_hex("437b9aec36"));

  let server_key = decode_hex("c206b8d9b9f0f37644430b490eeaa314");
  let server_sample = decode_hex("2cd0991cd25b0aac406a5816b6394100");
  let server = Aes128HeaderProtection::new(&Aes128HeaderProtectionKey::from_bytes(server_key));
  assert_eq!(server.mask(&server_sample), decode_hex("2ec0d8356a"));
}

#[cfg(feature = "chacha20poly1305")]
#[test]
fn rfc9001_chacha20_short_header_mask() {
  let key = decode_hex("25a282b9e82f06f21f488917a4fc8f1b73573685608597d0efcb076b0ab7a7a4");
  let sample = decode_hex("5e5cd55c41f69080575d7999c25a5bfb");
  let context = ChaCha20HeaderProtection::new(&ChaCha20HeaderProtectionKey::from_bytes(key));
  assert_eq!(context.mask(&sample), decode_hex("aefefe7d03"));

  let counter = u32::from_le_bytes(sample[..4].try_into().expect("counter prefix is four bytes"));
  assert_eq!(
    counter, 0x5cd5_5c5e,
    "the sample counter must be interpreted little-endian"
  );
}

#[cfg(feature = "aes-gcm")]
#[test]
fn aes_masks_match_independent_oracles_for_generated_inputs() {
  let mut state = 0x7273_6372_7970_746fu64;
  for _ in 0..GENERATED_CASES {
    let key128 = generated_bytes(&mut state);
    let key256 = generated_bytes(&mut state);
    let sample = generated_bytes(&mut state);
    let hp128 = Aes128HeaderProtection::new(&Aes128HeaderProtectionKey::from_bytes(key128));
    let hp256 = Aes256HeaderProtection::new(&Aes256HeaderProtectionKey::from_bytes(key256));
    assert_eq!(hp128.mask(&sample), aes128_oracle(&key128, &sample));
    assert_eq!(hp256.mask(&sample), aes256_oracle(&key256, &sample));
  }
}

#[cfg(feature = "chacha20poly1305")]
#[test]
fn chacha20_masks_match_independent_oracle_for_generated_inputs() {
  let mut state = 0x6865_6164_6572_2d70u64;
  for _ in 0..GENERATED_CASES {
    let key = generated_bytes(&mut state);
    let sample = generated_bytes(&mut state);
    let hp = ChaCha20HeaderProtection::new(&ChaCha20HeaderProtectionKey::from_bytes(key));
    assert_eq!(hp.mask(&sample), chacha20_oracle(&key, &sample));
  }
}

#[cfg(feature = "aes-gcm")]
#[test]
fn aes_zero_one_and_each_sample_byte_match_oracles() {
  for value in [0u8, u8::MAX] {
    let key128 = [value; 16];
    let key256 = [value; 32];
    let sample = [value; 16];
    let hp128 = Aes128HeaderProtection::new(&Aes128HeaderProtectionKey::from_bytes(key128));
    let hp256 = Aes256HeaderProtection::new(&Aes256HeaderProtectionKey::from_bytes(key256));
    assert_eq!(hp128.mask(&sample), aes128_oracle(&key128, &sample));
    assert_eq!(hp256.mask(&sample), aes256_oracle(&key256, &sample));
  }

  let key128 = [0x53; 16];
  let key256 = [0xa7; 32];
  let hp128 = Aes128HeaderProtection::new(&Aes128HeaderProtectionKey::from_bytes(key128));
  let hp256 = Aes256HeaderProtection::new(&Aes256HeaderProtectionKey::from_bytes(key256));
  for index in 0..16 {
    let mut sample = [0x39; 16];
    sample[index] ^= 0x80;
    assert_eq!(hp128.mask(&sample), aes128_oracle(&key128, &sample));
    assert_eq!(hp256.mask(&sample), aes256_oracle(&key256, &sample));
  }
}

#[cfg(feature = "chacha20poly1305")]
#[test]
fn chacha20_zero_one_and_each_sample_byte_match_oracle() {
  for value in [0u8, u8::MAX] {
    let key = [value; 32];
    let sample = [value; 16];
    let hp = ChaCha20HeaderProtection::new(&ChaCha20HeaderProtectionKey::from_bytes(key));
    assert_eq!(hp.mask(&sample), chacha20_oracle(&key, &sample));
  }

  let key = [0x53; 32];
  let hp = ChaCha20HeaderProtection::new(&ChaCha20HeaderProtectionKey::from_bytes(key));
  for index in 0..16 {
    let mut sample = [0x39; 16];
    sample[index] ^= 0x80;
    assert_eq!(hp.mask(&sample), chacha20_oracle(&key, &sample));
  }
}

#[test]
fn key_and_context_debug_output_is_redacted() {
  #[cfg(feature = "aes-gcm")]
  {
    let key128 = Aes128HeaderProtectionKey::from_bytes([0x53; 16]);
    assert_eq!(format!("{key128:?}"), "Aes128HeaderProtectionKey(****)");
    assert_eq!(
      format!("{:?}", Aes128HeaderProtection::new(&key128)),
      "Aes128HeaderProtection { .. }"
    );

    let key256 = Aes256HeaderProtectionKey::from_bytes([0x53; 32]);
    assert_eq!(format!("{key256:?}"), "Aes256HeaderProtectionKey(****)");
    assert_eq!(
      format!("{:?}", Aes256HeaderProtection::new(&key256)),
      "Aes256HeaderProtection { .. }"
    );
  }

  #[cfg(feature = "chacha20poly1305")]
  {
    let key = ChaCha20HeaderProtectionKey::from_bytes([0x53; 32]);
    assert_eq!(format!("{key:?}"), "ChaCha20HeaderProtectionKey(****)");
    assert_eq!(
      format!("{:?}", ChaCha20HeaderProtection::new(&key)),
      "ChaCha20HeaderProtection { .. }"
    );
  }
}
