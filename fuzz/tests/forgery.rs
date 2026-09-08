#[path = "../target_impls/aead_aes128gcm.rs"]
mod aead_aes128gcm;

#[path = "../target_impls/aead_aes256gcm.rs"]
mod aead_aes256gcm;

#[path = "../target_impls/aead_aes128gcmsiv.rs"]
mod aead_aes128gcmsiv;

#[path = "../target_impls/aead_aes256gcmsiv.rs"]
mod aead_aes256gcmsiv;

#[path = "../target_impls/aead_chacha20poly1305.rs"]
mod aead_chacha20poly1305;

#[path = "../target_impls/aead_xchacha20poly1305.rs"]
mod aead_xchacha20poly1305;

#[path = "../target_impls/aead_ascon128.rs"]
mod aead_ascon128;

#[path = "../target_impls/aead_aegis256.rs"]
mod aead_aegis256;

#[path = "../target_impls/auth_mlkem512.rs"]
mod auth_mlkem512;

#[path = "../target_impls/auth_mlkem768.rs"]
mod auth_mlkem768;

#[path = "../target_impls/auth_mlkem1024.rs"]
mod auth_mlkem1024;

type Target = fn(&[u8]);

#[test]
fn aead_long_input_forgeries() {
  let targets: &[(Target, usize, usize)] = &[
    (aead_aes128gcm::run, 16, 12),
    (aead_aes256gcm::run, 32, 12),
    (aead_aes128gcmsiv::run, 16, 12),
    (aead_aes256gcmsiv::run, 32, 12),
    (aead_chacha20poly1305::run, 32, 12),
    (aead_xchacha20poly1305::run, 32, 24),
    (aead_ascon128::run, 16, 16),
    (aead_aegis256::run, 32, 32),
  ];
  for &(run, key_len, nonce_len) in targets {
    for target in 0..3 {
      for position in [15u64, 84, 85, 86, 255, 256, 511, 1015, 1023] {
        for bit in 0..8 {
          let mut data = vec![0; key_len + nonce_len];
          data.push(target);
          data.extend_from_slice(&position.to_le_bytes());
          data.extend_from_slice(&[bit, 128]);
          // 2040 bytes split at ratio 128 gives 1024 AAD and 1016 plaintext bytes.
          data.extend_from_slice(&[42; 2040]);
          run(&data);
        }
      }
    }
    // Empty ciphertext/AAD selections must still forge the tag.
    for target in [0, 2] {
      let mut data = vec![0; key_len + nonce_len];
      data.push(target);
      data.extend_from_slice(&u64::MAX.to_le_bytes());
      data.extend_from_slice(&[255, 0]);
      run(&data);
    }
  }
}

#[test]
fn mlkem_suffix_forgeries() {
  use rscrypto::{MlKem512, MlKem768, MlKem1024};
  let targets: &[(Target, usize)] = &[
    (auth_mlkem512::run, MlKem512::CIPHERTEXT_SIZE),
    (auth_mlkem768::run, MlKem768::CIPHERTEXT_SIZE),
    (auth_mlkem1024::run, MlKem1024::CIPHERTEXT_SIZE),
  ];
  for &(run, len) in targets {
    for position in [255, 256, len / 2, len - 2, len - 1] {
      for bit in 0..8 {
        let mut data = vec![0; 96];
        data.extend_from_slice(&(position as u64).to_le_bytes());
        data.push(bit);
        run(&data);
      }
    }
  }
}
