use aes::cipher::{Array, BlockCipherEncrypt as _, KeyInit as _};
use chacha20::cipher::{KeyIvInit as _, StreamCipherCore as _};
use rscrypto::aead::expert::header_protection::{
  Aes128HeaderProtection, Aes128HeaderProtectionKey, Aes256HeaderProtection, Aes256HeaderProtectionKey,
  ChaCha20HeaderProtection, ChaCha20HeaderProtectionKey,
};
use rscrypto_fuzz::{FuzzInput, some_or_return};

pub(super) fn run(data: &[u8]) {
  let mut input = FuzzInput::new(data);
  let key: [u8; 32] = some_or_return!(input.bytes());
  let sample: [u8; 16] = some_or_return!(input.bytes());

  let key128: [u8; 16] = key[..16].try_into().expect("AES-128 key prefix has fixed length");
  let hp128 = Aes128HeaderProtection::new(&Aes128HeaderProtectionKey::from_bytes(key128));
  let hp256 = Aes256HeaderProtection::new(&Aes256HeaderProtectionKey::from_bytes(key));
  let chacha = ChaCha20HeaderProtection::new(&ChaCha20HeaderProtectionKey::from_bytes(key));

  let aes128_oracle = aes::Aes128::new(&Array::from(key128));
  let mut block128 = Array::from(sample);
  aes128_oracle.encrypt_block(&mut block128);
  assert_eq!(hp128.mask(&sample), block128[..5]);

  let aes256_oracle = aes::Aes256::new(&Array::from(key));
  let mut block256 = Array::from(sample);
  aes256_oracle.encrypt_block(&mut block256);
  assert_eq!(hp256.mask(&sample), block256[..5]);

  let counter = u32::from_le_bytes(sample[..4].try_into().expect("counter prefix has fixed length"));
  let nonce: [u8; 12] = sample[4..].try_into().expect("nonce suffix has fixed length");
  let mut core = chacha20::ChaChaCore::<chacha20::R20, chacha20::variants::Ietf>::new((&key).into(), (&nonce).into());
  core.set_block_pos(counter);
  let mut chacha_block = chacha20::cipher::array::Array::<u8, chacha20::cipher::consts::U64>::default();
  core.write_keystream_block(&mut chacha_block);
  assert_eq!(chacha.mask(&sample), chacha_block[..5]);
}
