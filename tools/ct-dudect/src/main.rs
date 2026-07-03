use dudect_bencher::{BenchRng, Class, CtRunner, ctbench_main_with_seeds, rand::RngExt};
use rscrypto::{
  Aegis256, Aegis256Key, Aes128Gcm, Aes128GcmKey, Aes128GcmSiv, Aes128GcmSivKey, Aes256Gcm, Aes256GcmKey,
  Aes256GcmSiv, Aes256GcmSivKey, Argon2Params, Argon2i, AsconAead128, AsconAead128Key, Blake2b256, Blake2b512,
  Blake2s128, Blake2s256, Blake3, Blake3KeyedHash, ChaCha20Poly1305, ChaCha20Poly1305Key, EcdsaP256Keypair,
  EcdsaP256SecretKey, EcdsaP384Keypair, EcdsaP384SecretKey, Ed25519Keypair, Ed25519SecretKey, HkdfSha256,
  HkdfSha384, HmacSha256, HmacSha256Tag, HmacSha384, HmacSha384Tag, HmacSha512, HmacSha512Tag, Kmac256, MlKem512,
  MlKem512Ciphertext, MlKem512DecapsulationKey, MlKem768, MlKem768Ciphertext, MlKem768DecapsulationKey, MlKem1024,
  MlKem1024Ciphertext, MlKem1024DecapsulationKey, MlKemError, Pbkdf2Sha256, Pbkdf2Sha512, RsaOaepProfile,
  RsaPkcs1v15Profile, RsaPrivateKey, RsaPssProfile, SecretBytes, Sha512, X25519SecretKey, XChaCha20Poly1305,
  XChaCha20Poly1305Key,
  aead::{
    Nonce96, Nonce128, Nonce192, Nonce256, diag_aes128gcm_ctr32_be, diag_aes128gcm_ghash,
    diag_aes128gcm_tag_aes, diag_aes128gcmsiv_ctr32, diag_aes128gcmsiv_derive_keys,
    diag_aes128gcmsiv_polyval_digest, diag_aes128gcmsiv_raw_tag_aes, diag_aes256gcm_ctr32_be,
    diag_aes256gcm_ghash, diag_aes256gcm_tag_aes, diag_aes256gcmsiv_ctr32, diag_aes256gcmsiv_derive_keys,
    diag_aes256gcmsiv_raw_tag_aes,
  },
  auth::diag_rsa_private_component_validation_32,
  diag_ecdsa_p256_basepoint_blinded_limb_digest, diag_ecdsa_p256_nonce_reduce_limb_digest,
  diag_ecdsa_p256_nonce_inverse_limb_digest, diag_ecdsa_p256_order_mul_fixed_r_limb_digest,
  diag_ecdsa_p256_reduce_wide_order_limb_digest, diag_ecdsa_p256_scalar_finish_limb_digest,
  diag_ecdsa_p256_final_multiply_limb_digest, diag_ecdsa_p384_basepoint_blinded_limb_digest,
  diag_ecdsa_p384_nonce_reduce_limb_digest, diag_ecdsa_p384_nonce_inverse_limb_digest,
  diag_ecdsa_p384_order_mul_fixed_r_limb_digest, diag_ecdsa_p384_reduce_wide_order_limb_digest,
  diag_ecdsa_p384_scalar_finish_limb_digest, diag_ecdsa_p384_final_multiply_limb_digest,
  diag_mlkem512_keygen_secret_noise_digest, diag_mlkem768_keygen_secret_noise_digest,
  diag_mlkem1024_keygen_secret_noise_digest, diag_mlkem1024_multiply_ntts_accumulate_input_digest,
  diag_mlkem_from_montgomery_product_domain_input_digest, diag_mlkem_inverse_ntt_montgomery_product_input_digest,
  diag_mlkem_multiply_ntts_add_assign_input_digest, diag_mlkem_ntt_input_digest,
  diag_mlkem_to_montgomery_product_domain_input_digest, diag_rsa_import_pkcs8_private_key_der_stage,
  diag_rsa_validate_pkcs8_private_key_der, diag_rsa_validate_pkcs8_private_key_der_stage,
  traits::{Kem as _, ct},
  RsaEncryptionError, RsaPublicKeyPolicy,
};

const DEFAULT_SAMPLES: usize = 20_000;
const MESSAGE: &[u8] = b"rscrypto constant-time dudect timing lane input";
const BLAKE3_PARALLEL_MESSAGE_LEN: usize = 1024 * 1024;
const MLKEM_Q: u16 = 3329;
const AAD: &[u8] = b"rscrypto-ct";
const AEAD_PLAINTEXT: [u8; 44] = *b"constant-time seal buffer for key validation";
const RSA_PKCS1_2048: &str = include_str!("../../../testdata/rsa/wycheproof/rsa_pkcs1_2048_test.json");
const RSA_CT_KEY_A_INDEX: usize = 0;
const RSA_CT_KEY_B_SAME_SHAPE_INDEX: usize = 2;

fn samples() -> usize {
  std::env::var("RSCRYPTO_CT_DUDECT_SAMPLES")
    .ok()
    .and_then(|value| value.parse::<usize>().ok())
    .filter(|value| *value > 0)
    .unwrap_or(DEFAULT_SAMPLES)
}

fn rand_array<const N: usize>(rng: &mut BenchRng) -> [u8; N] {
  let mut out = [0u8; N];
  rng.fill(&mut out);
  out
}

fn random_class(rng: &mut BenchRng) -> Class {
  if rng.random::<bool>() {
    Class::Left
  } else {
    Class::Right
  }
}

fn mlkem_poly_from_seed(seed: u16) -> [u16; 256] {
  let mut out = [0u16; 256];
  for (i, coeff) in out.iter_mut().enumerate() {
    *coeff = seed.wrapping_add((i as u16).wrapping_mul(73)) % MLKEM_Q;
  }
  out
}

fn random_mlkem_poly(rng: &mut BenchRng) -> [u16; 256] {
  let mut out = [0u16; 256];
  for coeff in &mut out {
    *coeff = rng.random::<u16>() % MLKEM_Q;
  }
  out
}

fn mlkem_polyvec4_from_seed(seed: u16) -> [[u16; 256]; 4] {
  [
    mlkem_poly_from_seed(seed),
    mlkem_poly_from_seed(seed.wrapping_add(0x101)),
    mlkem_poly_from_seed(seed.wrapping_add(0x202)),
    mlkem_poly_from_seed(seed.wrapping_add(0x303)),
  ]
}

fn random_mlkem_polyvec4(rng: &mut BenchRng) -> [[u16; 256]; 4] {
  [
    random_mlkem_poly(rng),
    random_mlkem_poly(rng),
    random_mlkem_poly(rng),
    random_mlkem_poly(rng),
  ]
}

fn json_string_value_n<'a>(json: &'a str, key: &str, n: usize) -> &'a str {
  let needle = format!("\"{key}\": \"");
  let mut rest = json;
  for index in 0..=n {
    let start = rest.find(&needle).unwrap_or_else(|| panic!("missing JSON string key {key} at index {index}"));
    let value = &rest[start + needle.len()..];
    let end = value.find('"').unwrap_or_else(|| panic!("unterminated JSON string key {key} at index {index}"));
    if index == n {
      return &value[..end];
    }
    rest = &value[end + 1..];
  }
  unreachable!()
}

fn hex_nibble(byte: u8) -> u8 {
  match byte {
    b'0'..=b'9' => byte - b'0',
    b'a'..=b'f' => byte - b'a' + 10,
    b'A'..=b'F' => byte - b'A' + 10,
    _ => panic!("invalid hex byte"),
  }
}

fn hex_to_vec(hex: &str) -> Vec<u8> {
  let bytes = hex.as_bytes();
  assert!(bytes.len().is_multiple_of(2), "hex string must have even length");
  bytes
    .chunks_exact(2)
    .map(|pair| (hex_nibble(pair[0]) << 4) | hex_nibble(pair[1]))
    .collect()
}

fn rsa_pkcs8_der(index: usize) -> Vec<u8> {
  hex_to_vec(json_string_value_n(RSA_PKCS1_2048, "privateKeyPkcs8", index))
}

fn rsa_ct_fixture_key(index: usize) -> RsaPrivateKey {
  let der = rsa_pkcs8_der(index);
  RsaPrivateKey::from_pkcs8_der_with_policy(&der, &RsaPublicKeyPolicy::legacy_verification()).unwrap()
}

fn rsa_blinding_pair(key: &RsaPrivateKey) -> (Vec<u8>, Vec<u8>) {
  let modulus = key.public_key().modulus();
  let mut factor = vec![0u8; modulus.len()];
  factor[modulus.len() - 1] = 2;

  let mut plus_one = modulus.to_vec();
  let mut carry = 1u16;
  for byte in plus_one.iter_mut().rev() {
    let sum = u16::from(*byte) + carry;
    *byte = sum as u8;
    carry = sum >> 8;
    if carry == 0 {
      break;
    }
  }
  if carry != 0 {
    plus_one.insert(0, carry as u8);
  }

  let mut quotient = Vec::with_capacity(plus_one.len());
  let mut remainder = 0u16;
  for byte in plus_one {
    let value = (remainder << 8) | u16::from(byte);
    quotient.push((value / 2) as u8);
    remainder = value % 2;
  }

  let first_nonzero = quotient.iter().position(|&byte| byte != 0).unwrap_or(quotient.len() - 1);
  let inverse = &quotient[first_nonzero..];
  let mut inverse_fixed = vec![0u8; modulus.len()];
  inverse_fixed[modulus.len() - inverse.len()..].copy_from_slice(inverse);
  (factor, inverse_fixed)
}

fn argon2i_params() -> Argon2Params {
  Argon2Params::new()
    .memory_cost_kib(32)
    .time_cost(1)
    .parallelism(1)
    .output_len(16)
    .build()
    .unwrap()
}

fn argon2i_parallel_params() -> Argon2Params {
  Argon2Params::new()
    .memory_cost_kib(512)
    .time_cost(1)
    .parallelism(4)
    .output_len(16)
    .build()
    .unwrap()
}

fn constant_time_eq_equal_vs_first_diff(runner: &mut CtRunner, rng: &mut BenchRng) {
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let a = rand_array::<64>(rng);
    let mut b = a;
    if matches!(class, Class::Right) {
      b[0] ^= 1;
    }
    inputs.push((class, a, b));
  }

  for (class, a, b) in inputs {
    runner.run_one(class, || ct::constant_time_eq(&a, &b));
  }
}

fn secret_wrappers_eq_and_debug_fixed_vs_random(runner: &mut CtRunner, rng: &mut BenchRng) {
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let bytes = if matches!(class, Class::Left) {
      [0x33; SecretBytes::<32>::LENGTH]
    } else {
      rand_array::<{ SecretBytes::<32>::LENGTH }>(rng)
    };
    inputs.push((class, bytes));
  }

  for (class, bytes) in inputs {
    runner.run_one(class, || {
      let left = SecretBytes::<32>::new(bytes);
      let right = SecretBytes::<32>::new(bytes);
      let masked = format!("{left:?}");
      left == right && masked.as_str() == "SecretBytes(****)"
    });
  }
}

macro_rules! hmac_valid_vs_invalid_tag {
  ($name:ident, $ty:ty, $tag_ty:ty, $tag_len:expr) => {
    fn $name(runner: &mut CtRunner, rng: &mut BenchRng) {
      let mut inputs = Vec::with_capacity(samples());
      for _ in 0..samples() {
        let class = random_class(rng);
        let key = rand_array::<32>(rng);
        let mut expected = <$ty>::mac(&key, MESSAGE).to_bytes();
        if matches!(class, Class::Right) {
          expected[0] ^= 1;
        }
        inputs.push((class, key, <$tag_ty>::from_bytes(expected)));
      }

      for (class, key, expected) in inputs {
        let _ = $tag_len;
        runner.run_one(class, || <$ty>::verify_tag(&key, MESSAGE, &expected).is_ok());
      }
    }
  };
}

hmac_valid_vs_invalid_tag!(hmac_sha256_valid_vs_invalid_tag, HmacSha256, HmacSha256Tag, 32);
hmac_valid_vs_invalid_tag!(hmac_sha384_valid_vs_invalid_tag, HmacSha384, HmacSha384Tag, 48);
hmac_valid_vs_invalid_tag!(hmac_sha512_valid_vs_invalid_tag, HmacSha512, HmacSha512Tag, 64);

fn kmac256_valid_vs_invalid_tag(runner: &mut CtRunner, rng: &mut BenchRng) {
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let key = rand_array::<32>(rng);
    let mut expected = Kmac256::mac_array::<32>(&key, b"ct", MESSAGE);
    if matches!(class, Class::Right) {
      expected[0] ^= 1;
    }
    inputs.push((class, key, expected));
  }

  for (class, key, expected) in inputs {
    runner.run_one(class, || Kmac256::verify_tag(&key, b"ct", MESSAGE, &expected).is_ok());
  }
}

fn blake3_keyed_valid_vs_invalid_tag(runner: &mut CtRunner, rng: &mut BenchRng) {
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let key = rand_array::<32>(rng);
    let mut expected = Blake3::keyed_digest(&key, MESSAGE).to_bytes();
    if matches!(class, Class::Right) {
      expected[0] ^= 1;
    }
    inputs.push((class, key, expected));
  }

  for (class, key, expected) in inputs {
    let expected = Blake3KeyedHash::from_bytes(expected);
    runner.run_one(class, || Blake3::verify_keyed(&key, MESSAGE, &expected).is_ok());
  }
}

macro_rules! aead_fixed_vs_random_key_open {
  ($name:ident, $cipher:ty, $key:ty, $nonce:ty, $fixed_key:expr, $key_len:expr, $nonce_byte:expr) => {
    fn $name(runner: &mut CtRunner, rng: &mut BenchRng) {
      let nonce = <$nonce>::from_bytes([$nonce_byte; <$nonce>::LENGTH]);
      let mut inputs = Vec::with_capacity(samples());
      for _ in 0..samples() {
        let class = random_class(rng);
        let key = if matches!(class, Class::Left) {
          $fixed_key
        } else {
          rand_array::<$key_len>(rng)
        };
        let key = <$key>::from_bytes(key);
        let cipher = <$cipher>::new(&key);
        let mut ciphertext = AEAD_PLAINTEXT;
        let tag = cipher.encrypt_in_place(&nonce, AAD, &mut ciphertext).unwrap();
        inputs.push((class, key, nonce, ciphertext, tag));
      }

      for (class, key, nonce, ciphertext, tag) in inputs {
        let cipher = <$cipher>::new(&key);
        runner.run_one(class, || {
          let mut buffer = ciphertext;
          cipher.decrypt_in_place(&nonce, AAD, &mut buffer, &tag).is_ok()
        });
      }
    }
  };
}

aead_fixed_vs_random_key_open!(
  aes128gcm_fixed_vs_random_key_open,
  Aes128Gcm,
  Aes128GcmKey,
  Nonce96,
  [0x13; Aes128Gcm::KEY_SIZE],
  16,
  0x13
);
aead_fixed_vs_random_key_open!(
  aes256gcm_fixed_vs_random_key_open,
  Aes256Gcm,
  Aes256GcmKey,
  Nonce96,
  [0x14; Aes256Gcm::KEY_SIZE],
  32,
  0x14
);
aead_fixed_vs_random_key_open!(
  aes128gcmsiv_fixed_vs_random_key_open,
  Aes128GcmSiv,
  Aes128GcmSivKey,
  Nonce96,
  [0x15; Aes128GcmSiv::KEY_SIZE],
  16,
  0x15
);
aead_fixed_vs_random_key_open!(
  aes256gcmsiv_fixed_vs_random_key_open,
  Aes256GcmSiv,
  Aes256GcmSivKey,
  Nonce96,
  [0x16; Aes256GcmSiv::KEY_SIZE],
  32,
  0x16
);
aead_fixed_vs_random_key_open!(
  ietf_chacha20poly1305_fixed_vs_random_key_open,
  ChaCha20Poly1305,
  ChaCha20Poly1305Key,
  Nonce96,
  [0x17; ChaCha20Poly1305::KEY_SIZE],
  32,
  0x17
);
aead_fixed_vs_random_key_open!(
  xchacha20poly1305_fixed_vs_random_key_open,
  XChaCha20Poly1305,
  XChaCha20Poly1305Key,
  Nonce192,
  [0x18; XChaCha20Poly1305::KEY_SIZE],
  32,
  0x18
);
aead_fixed_vs_random_key_open!(
  aegis256_fixed_vs_random_key_open,
  Aegis256,
  Aegis256Key,
  Nonce256,
  [0x19; Aegis256::KEY_SIZE],
  32,
  0x19
);
aead_fixed_vs_random_key_open!(
  ascon_aead128_fixed_vs_random_key_open,
  AsconAead128,
  AsconAead128Key,
  Nonce128,
  [0x1A; AsconAead128::KEY_SIZE],
  16,
  0x1A
);

fn aes128_gcm_siv_diag_derive_fixed_vs_random_key(runner: &mut CtRunner, rng: &mut BenchRng) {
  let nonce = Nonce96::from_bytes([0x51; Nonce96::LENGTH]);
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let key = if matches!(class, Class::Left) {
      [0x51; Aes128GcmSiv::KEY_SIZE]
    } else {
      rand_array::<{ Aes128GcmSiv::KEY_SIZE }>(rng)
    };
    inputs.push((class, Aes128GcmSiv::new(&Aes128GcmSivKey::from_bytes(key)), nonce));
  }

  for (class, cipher, nonce) in inputs {
    runner.run_one(class, || diag_aes128gcmsiv_derive_keys(&cipher, &nonce));
  }
}

fn aes256_gcm_siv_diag_derive_fixed_vs_random_key(runner: &mut CtRunner, rng: &mut BenchRng) {
  let nonce = Nonce96::from_bytes([0x52; Nonce96::LENGTH]);
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let key = if matches!(class, Class::Left) {
      [0x52; Aes256GcmSiv::KEY_SIZE]
    } else {
      rand_array::<{ Aes256GcmSiv::KEY_SIZE }>(rng)
    };
    inputs.push((class, Aes256GcmSiv::new(&Aes256GcmSivKey::from_bytes(key)), nonce));
  }

  for (class, cipher, nonce) in inputs {
    runner.run_one(class, || diag_aes256gcmsiv_derive_keys(&cipher, &nonce));
  }
}

fn gcm_siv_diag_polyval_fixed_vs_random_auth_key(runner: &mut CtRunner, rng: &mut BenchRng) {
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let auth_key = if matches!(class, Class::Left) {
      [0x53; 16]
    } else {
      rand_array::<16>(rng)
    };
    inputs.push((class, auth_key));
  }

  for (class, auth_key) in inputs {
    runner.run_one(class, || diag_aes128gcmsiv_polyval_digest(&auth_key, AAD, &AEAD_PLAINTEXT));
  }
}

fn aes128_gcm_siv_diag_raw_tag_aes_fixed_vs_random_key(runner: &mut CtRunner, rng: &mut BenchRng) {
  let block = [0x54; 16];
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let enc_key = if matches!(class, Class::Left) {
      [0x54; Aes128GcmSiv::KEY_SIZE]
    } else {
      rand_array::<{ Aes128GcmSiv::KEY_SIZE }>(rng)
    };
    inputs.push((class, enc_key, block));
  }

  for (class, enc_key, block) in inputs {
    runner.run_one(class, || diag_aes128gcmsiv_raw_tag_aes(&enc_key, &block));
  }
}

fn aes256_gcm_siv_diag_raw_tag_aes_fixed_vs_random_key(runner: &mut CtRunner, rng: &mut BenchRng) {
  let block = [0x55; 16];
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let enc_key = if matches!(class, Class::Left) {
      [0x55; Aes256GcmSiv::KEY_SIZE]
    } else {
      rand_array::<{ Aes256GcmSiv::KEY_SIZE }>(rng)
    };
    inputs.push((class, enc_key, block));
  }

  for (class, enc_key, block) in inputs {
    runner.run_one(class, || diag_aes256gcmsiv_raw_tag_aes(&enc_key, &block));
  }
}

fn aes128_gcm_diag_ctr32_be_fixed_vs_random_key(runner: &mut CtRunner, rng: &mut BenchRng) {
  let nonce = Nonce96::from_bytes([0x56; Nonce96::LENGTH]);
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let key = if matches!(class, Class::Left) {
      [0x56; Aes128Gcm::KEY_SIZE]
    } else {
      rand_array::<{ Aes128Gcm::KEY_SIZE }>(rng)
    };
    inputs.push((class, Aes128Gcm::new(&Aes128GcmKey::from_bytes(key)), nonce));
  }

  for (class, cipher, nonce) in inputs {
    runner.run_one(class, || diag_aes128gcm_ctr32_be(&cipher, &nonce, &AEAD_PLAINTEXT));
  }
}

fn aes256_gcm_diag_ctr32_be_fixed_vs_random_key(runner: &mut CtRunner, rng: &mut BenchRng) {
  let nonce = Nonce96::from_bytes([0x57; Nonce96::LENGTH]);
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let key = if matches!(class, Class::Left) {
      [0x57; Aes256Gcm::KEY_SIZE]
    } else {
      rand_array::<{ Aes256Gcm::KEY_SIZE }>(rng)
    };
    inputs.push((class, Aes256Gcm::new(&Aes256GcmKey::from_bytes(key)), nonce));
  }

  for (class, cipher, nonce) in inputs {
    runner.run_one(class, || diag_aes256gcm_ctr32_be(&cipher, &nonce, &AEAD_PLAINTEXT));
  }
}

fn aes128_gcm_diag_ghash_fixed_vs_random_h(runner: &mut CtRunner, rng: &mut BenchRng) {
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let key = if matches!(class, Class::Left) {
      [0x58; Aes128Gcm::KEY_SIZE]
    } else {
      rand_array::<{ Aes128Gcm::KEY_SIZE }>(rng)
    };
    inputs.push((class, Aes128Gcm::new(&Aes128GcmKey::from_bytes(key))));
  }

  for (class, cipher) in inputs {
    runner.run_one(class, || diag_aes128gcm_ghash(&cipher, AAD, &AEAD_PLAINTEXT));
  }
}

fn aes256_gcm_diag_ghash_fixed_vs_random_h(runner: &mut CtRunner, rng: &mut BenchRng) {
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let key = if matches!(class, Class::Left) {
      [0x59; Aes256Gcm::KEY_SIZE]
    } else {
      rand_array::<{ Aes256Gcm::KEY_SIZE }>(rng)
    };
    inputs.push((class, Aes256Gcm::new(&Aes256GcmKey::from_bytes(key))));
  }

  for (class, cipher) in inputs {
    runner.run_one(class, || diag_aes256gcm_ghash(&cipher, AAD, &AEAD_PLAINTEXT));
  }
}

fn aes128_gcm_diag_tag_aes_fixed_vs_random_key(runner: &mut CtRunner, rng: &mut BenchRng) {
  let nonce = Nonce96::from_bytes([0x5A; Nonce96::LENGTH]);
  let acc = [0x5A; 16];
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let key = if matches!(class, Class::Left) {
      [0x5A; Aes128Gcm::KEY_SIZE]
    } else {
      rand_array::<{ Aes128Gcm::KEY_SIZE }>(rng)
    };
    inputs.push((class, Aes128Gcm::new(&Aes128GcmKey::from_bytes(key)), nonce, acc));
  }

  for (class, cipher, nonce, acc) in inputs {
    runner.run_one(class, || diag_aes128gcm_tag_aes(&cipher, &nonce, &acc));
  }
}

fn aes256_gcm_diag_tag_aes_fixed_vs_random_key(runner: &mut CtRunner, rng: &mut BenchRng) {
  let nonce = Nonce96::from_bytes([0x5B; Nonce96::LENGTH]);
  let acc = [0x5B; 16];
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let key = if matches!(class, Class::Left) {
      [0x5B; Aes256Gcm::KEY_SIZE]
    } else {
      rand_array::<{ Aes256Gcm::KEY_SIZE }>(rng)
    };
    inputs.push((class, Aes256Gcm::new(&Aes256GcmKey::from_bytes(key)), nonce, acc));
  }

  for (class, cipher, nonce, acc) in inputs {
    runner.run_one(class, || diag_aes256gcm_tag_aes(&cipher, &nonce, &acc));
  }
}

fn aes128_gcm_siv_diag_ctr32_fixed_vs_random_key(runner: &mut CtRunner, rng: &mut BenchRng) {
  let tag = [0x5E; 16];
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let enc_key = if matches!(class, Class::Left) {
      [0x5E; Aes128GcmSiv::KEY_SIZE]
    } else {
      rand_array::<{ Aes128GcmSiv::KEY_SIZE }>(rng)
    };
    inputs.push((class, enc_key, tag));
  }

  for (class, enc_key, tag) in inputs {
    runner.run_one(class, || diag_aes128gcmsiv_ctr32(&enc_key, &tag, &AEAD_PLAINTEXT));
  }
}

fn aes256_gcm_siv_diag_ctr32_fixed_vs_random_key(runner: &mut CtRunner, rng: &mut BenchRng) {
  let tag = [0x5F; 16];
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let enc_key = if matches!(class, Class::Left) {
      [0x5F; Aes256GcmSiv::KEY_SIZE]
    } else {
      rand_array::<{ Aes256GcmSiv::KEY_SIZE }>(rng)
    };
    inputs.push((class, enc_key, tag));
  }

  for (class, enc_key, tag) in inputs {
    runner.run_one(class, || diag_aes256gcmsiv_ctr32(&enc_key, &tag, &AEAD_PLAINTEXT));
  }
}

fn aes128_gcm_siv_diag_raw_tag_aes_varying_block(runner: &mut CtRunner, rng: &mut BenchRng) {
  let enc_key = [0x60; Aes128GcmSiv::KEY_SIZE];
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let block = if matches!(class, Class::Left) {
      [0x60; 16]
    } else {
      rand_array::<16>(rng)
    };
    inputs.push((class, block));
  }

  for (class, block) in inputs {
    runner.run_one(class, || diag_aes128gcmsiv_raw_tag_aes(&enc_key, &block));
  }
}

fn aes256_gcm_siv_diag_raw_tag_aes_varying_block(runner: &mut CtRunner, rng: &mut BenchRng) {
  let enc_key = [0x61; Aes256GcmSiv::KEY_SIZE];
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let block = if matches!(class, Class::Left) {
      [0x61; 16]
    } else {
      rand_array::<16>(rng)
    };
    inputs.push((class, block));
  }

  for (class, block) in inputs {
    runner.run_one(class, || diag_aes256gcmsiv_raw_tag_aes(&enc_key, &block));
  }
}

fn x25519_fixed_vs_random_scalar(runner: &mut CtRunner, rng: &mut BenchRng) {
  let peer = X25519SecretKey::from_bytes([9u8; X25519SecretKey::LENGTH]).public_key();
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let scalar = if matches!(class, Class::Left) {
      [7u8; X25519SecretKey::LENGTH]
    } else {
      rand_array::<{ X25519SecretKey::LENGTH }>(rng)
    };
    inputs.push((class, scalar, peer));
  }

  for (class, scalar, peer) in inputs {
    let secret = X25519SecretKey::from_bytes(scalar);
    runner.run_one(class, || secret.diffie_hellman(&peer).is_ok());
  }
}

macro_rules! mlkem_dudect_profile {
  (
    $keygen_secret_noise:ident,
    $encapsulate_coins:ident,
    $decapsulate_secret_key:ident,
    $decapsulate_rejection_seed:ident,
    $profile:ty,
    $ciphertext:ty,
    $decapsulation_key:ty,
    $diag:ident
  ) => {
    fn $keygen_secret_noise(runner: &mut CtRunner, rng: &mut BenchRng) {
      let rho = [0x41; <$profile>::ENCAPSULATION_RANDOM_SIZE];
      let mut inputs = Vec::with_capacity(samples());
      for _ in 0..samples() {
        let class = random_class(rng);
        let sigma = if matches!(class, Class::Left) {
          [0x42; <$profile>::ENCAPSULATION_RANDOM_SIZE]
        } else {
          rand_array::<{ <$profile>::ENCAPSULATION_RANDOM_SIZE }>(rng)
        };
        inputs.push((class, sigma));
      }

      for (class, sigma) in inputs {
        runner.run_one(class, || std::hint::black_box($diag(rho, sigma))[0]);
      }
    }

    fn $encapsulate_coins(runner: &mut CtRunner, rng: &mut BenchRng) {
      let (encapsulation_key, _) = <$profile>::generate_keypair(|out| {
        out.copy_from_slice(&[0x51; <$profile>::KEY_GENERATION_RANDOM_SIZE]);
        Ok::<(), MlKemError>(())
      })
      .unwrap();
      let mut inputs = Vec::with_capacity(samples());
      for _ in 0..samples() {
        let class = random_class(rng);
        let random = if matches!(class, Class::Left) {
          [0x52; <$profile>::ENCAPSULATION_RANDOM_SIZE]
        } else {
          rand_array::<{ <$profile>::ENCAPSULATION_RANDOM_SIZE }>(rng)
        };
        inputs.push((class, random));
      }

      for (class, random) in inputs {
        runner.run_one(class, || {
          let (ciphertext, shared_secret) = <$profile>::encapsulate(&encapsulation_key, |out| {
            out.copy_from_slice(&random);
            Ok::<(), MlKemError>(())
          })
          .unwrap();
          std::hint::black_box(ciphertext.as_bytes()[0] ^ shared_secret.as_bytes()[0])
        });
      }
    }

    fn $decapsulate_secret_key(runner: &mut CtRunner, rng: &mut BenchRng) {
      let (fixed_encapsulation_key, fixed_decapsulation_key) = <$profile>::generate_keypair(|out| {
        out.copy_from_slice(&[0x61; <$profile>::KEY_GENERATION_RANDOM_SIZE]);
        Ok::<(), MlKemError>(())
      })
      .unwrap();
      let (ciphertext, _) = <$profile>::encapsulate(&fixed_encapsulation_key, |out| {
        out.copy_from_slice(&[0x62; <$profile>::ENCAPSULATION_RANDOM_SIZE]);
        Ok::<(), MlKemError>(())
      })
      .unwrap();
      let secret_key_len = <$profile>::DECAPSULATION_KEY_SIZE - <$profile>::ENCAPSULATION_KEY_SIZE - 64;
      let mut inputs = Vec::with_capacity(samples());
      for _ in 0..samples() {
        let class = random_class(rng);
        let mut decapsulation_key_bytes = *fixed_decapsulation_key.as_bytes();
        if matches!(class, Class::Right) {
          let random_decapsulation_key = <$profile>::generate_keypair(|out| {
            let random = rand_array::<{ <$profile>::KEY_GENERATION_RANDOM_SIZE }>(rng);
            out.copy_from_slice(&random);
            Ok::<(), MlKemError>(())
          })
          .unwrap()
          .1;
          decapsulation_key_bytes[..secret_key_len]
            .copy_from_slice(&random_decapsulation_key.as_bytes()[..secret_key_len]);
        }
        inputs.push((class, <$decapsulation_key>::from_bytes(decapsulation_key_bytes)));
      }

      for (class, decapsulation_key) in inputs {
        runner.run_one(class, || {
          let shared_secret = <$profile>::decapsulate(&decapsulation_key, &ciphertext).unwrap();
          std::hint::black_box(shared_secret.as_bytes()[0])
        });
      }
    }

    fn $decapsulate_rejection_seed(runner: &mut CtRunner, rng: &mut BenchRng) {
      let (encapsulation_key, decapsulation_key) = <$profile>::generate_keypair(|out| {
        out.copy_from_slice(&[0x71; <$profile>::KEY_GENERATION_RANDOM_SIZE]);
        Ok::<(), MlKemError>(())
      })
      .unwrap();
      let (ciphertext, _) = <$profile>::encapsulate(&encapsulation_key, |out| {
        out.copy_from_slice(&[0x72; <$profile>::ENCAPSULATION_RANDOM_SIZE]);
        Ok::<(), MlKemError>(())
      })
      .unwrap();
      let mut rejected_ciphertext = ciphertext.to_bytes();
      rejected_ciphertext[0] ^= 1;
      let rejected_ciphertext = <$ciphertext>::from_bytes(rejected_ciphertext);
      let z_start = <$profile>::DECAPSULATION_KEY_SIZE - <$profile>::SHARED_SECRET_SIZE;

      let mut inputs = Vec::with_capacity(samples());
      for _ in 0..samples() {
        let class = random_class(rng);
        let mut decapsulation_key_bytes = *decapsulation_key.as_bytes();
        if matches!(class, Class::Left) {
          decapsulation_key_bytes[z_start..].copy_from_slice(&[0x73; <$profile>::SHARED_SECRET_SIZE]);
        } else {
          let rejection_seed = rand_array::<{ <$profile>::SHARED_SECRET_SIZE }>(rng);
          decapsulation_key_bytes[z_start..].copy_from_slice(&rejection_seed);
        }
        inputs.push((class, <$decapsulation_key>::from_bytes(decapsulation_key_bytes)));
      }

      for (class, decapsulation_key) in inputs {
        runner.run_one(class, || {
          let shared_secret = <$profile>::decapsulate(&decapsulation_key, &rejected_ciphertext).unwrap();
          std::hint::black_box(shared_secret.as_bytes()[0])
        });
      }
    }
  };
}

mlkem_dudect_profile!(
  mlkem512_keygen_secret_noise_fixed_vs_random,
  mlkem512_encapsulate_fixed_vs_random_coins,
  mlkem512_decapsulate_fixed_vs_random_secret_key,
  mlkem512_decapsulate_rejection_seed_fixed_vs_random,
  MlKem512,
  MlKem512Ciphertext,
  MlKem512DecapsulationKey,
  diag_mlkem512_keygen_secret_noise_digest
);
mlkem_dudect_profile!(
  mlkem768_keygen_secret_noise_fixed_vs_random,
  mlkem768_encapsulate_fixed_vs_random_coins,
  mlkem768_decapsulate_fixed_vs_random_secret_key,
  mlkem768_decapsulate_rejection_seed_fixed_vs_random,
  MlKem768,
  MlKem768Ciphertext,
  MlKem768DecapsulationKey,
  diag_mlkem768_keygen_secret_noise_digest
);
mlkem_dudect_profile!(
  mlkem1024_keygen_secret_noise_fixed_vs_random,
  mlkem1024_encapsulate_fixed_vs_random_coins,
  mlkem1024_decapsulate_fixed_vs_random_secret_key,
  mlkem1024_decapsulate_rejection_seed_fixed_vs_random,
  MlKem1024,
  MlKem1024Ciphertext,
  MlKem1024DecapsulationKey,
  diag_mlkem1024_keygen_secret_noise_digest
);

fn mlkem_arithmetic_ntt_fixed_vs_random_poly(runner: &mut CtRunner, rng: &mut BenchRng) {
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let poly = if matches!(class, Class::Left) {
      mlkem_poly_from_seed(0x301)
    } else {
      random_mlkem_poly(rng)
    };
    inputs.push((class, poly));
  }

  for (class, poly) in inputs {
    runner.run_one(class, || std::hint::black_box(diag_mlkem_ntt_input_digest(poly)));
  }
}

fn mlkem_arithmetic_inverse_ntt_fixed_vs_random_poly(runner: &mut CtRunner, rng: &mut BenchRng) {
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let poly = if matches!(class, Class::Left) {
      mlkem_poly_from_seed(0x401)
    } else {
      random_mlkem_poly(rng)
    };
    inputs.push((class, poly));
  }

  for (class, poly) in inputs {
    runner.run_one(class, || std::hint::black_box(diag_mlkem_inverse_ntt_montgomery_product_input_digest(poly)));
  }
}

fn mlkem_arithmetic_to_product_domain_fixed_vs_random_poly(runner: &mut CtRunner, rng: &mut BenchRng) {
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let poly = if matches!(class, Class::Left) {
      mlkem_poly_from_seed(0x451)
    } else {
      random_mlkem_poly(rng)
    };
    inputs.push((class, poly));
  }

  for (class, poly) in inputs {
    runner.run_one(class, || std::hint::black_box(diag_mlkem_to_montgomery_product_domain_input_digest(poly)));
  }
}

fn mlkem_arithmetic_from_product_domain_fixed_vs_random_poly(runner: &mut CtRunner, rng: &mut BenchRng) {
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let poly = if matches!(class, Class::Left) {
      mlkem_poly_from_seed(0x471)
    } else {
      random_mlkem_poly(rng)
    };
    inputs.push((class, poly));
  }

  for (class, poly) in inputs {
    runner.run_one(class, || std::hint::black_box(diag_mlkem_from_montgomery_product_domain_input_digest(poly)));
  }
}

fn mlkem_arithmetic_basemul_fixed_vs_random_operands(runner: &mut CtRunner, rng: &mut BenchRng) {
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let (a, b, acc) = if matches!(class, Class::Left) {
      (
        mlkem_poly_from_seed(0x501),
        mlkem_poly_from_seed(0x601),
        mlkem_poly_from_seed(0x701),
      )
    } else {
      (random_mlkem_poly(rng), random_mlkem_poly(rng), random_mlkem_poly(rng))
    };
    inputs.push((class, a, b, acc));
  }

  for (class, a, b, acc) in inputs {
    runner.run_one(class, || std::hint::black_box(diag_mlkem_multiply_ntts_add_assign_input_digest(a, b, acc)));
  }
}

fn mlkem1024_arithmetic_dot_fixed_vs_random_operands(runner: &mut CtRunner, rng: &mut BenchRng) {
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let (a, b, acc) = if matches!(class, Class::Left) {
      (
        mlkem_polyvec4_from_seed(0x801),
        mlkem_polyvec4_from_seed(0x901),
        mlkem_poly_from_seed(0xA01),
      )
    } else {
      (random_mlkem_polyvec4(rng), random_mlkem_polyvec4(rng), random_mlkem_poly(rng))
    };
    inputs.push((class, a, b, acc));
  }

  for (class, a, b, acc) in inputs {
    runner.run_one(class, || std::hint::black_box(diag_mlkem1024_multiply_ntts_accumulate_input_digest(a, b, acc)));
  }
}

fn ed25519_sign_fixed_vs_random_secret(runner: &mut CtRunner, rng: &mut BenchRng) {
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let secret = if matches!(class, Class::Left) {
      [0x42; Ed25519SecretKey::LENGTH]
    } else {
      rand_array::<{ Ed25519SecretKey::LENGTH }>(rng)
    };
    inputs.push((class, secret));
  }

  for (class, secret) in inputs {
    let key = Ed25519SecretKey::from_bytes(secret);
    runner.run_one(class, || key.sign(MESSAGE).to_bytes()[0]);
  }
}

fn ed25519_public_key_fixed_vs_random_secret(runner: &mut CtRunner, rng: &mut BenchRng) {
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let secret = if matches!(class, Class::Left) {
      [0x42; Ed25519SecretKey::LENGTH]
    } else {
      rand_array::<{ Ed25519SecretKey::LENGTH }>(rng)
    };
    inputs.push((class, secret));
  }

  for (class, secret) in inputs {
    let key = Ed25519SecretKey::from_bytes(secret);
    runner.run_one(class, || key.public_key().to_bytes()[0]);
  }
}

fn ed25519_sha512_secret_expand_fixed_vs_random_secret(runner: &mut CtRunner, rng: &mut BenchRng) {
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let secret = if matches!(class, Class::Left) {
      [0x42; Ed25519SecretKey::LENGTH]
    } else {
      rand_array::<{ Ed25519SecretKey::LENGTH }>(rng)
    };
    inputs.push((class, secret));
  }

  for (class, secret) in inputs {
    runner.run_one(class, || Sha512::digest(&secret)[0]);
  }
}

fn ed25519_keypair_sign_fixed_vs_random_secret(runner: &mut CtRunner, rng: &mut BenchRng) {
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let secret = if matches!(class, Class::Left) {
      [0x42; Ed25519SecretKey::LENGTH]
    } else {
      rand_array::<{ Ed25519SecretKey::LENGTH }>(rng)
    };
    inputs.push((class, Ed25519Keypair::from_secret_key(Ed25519SecretKey::from_bytes(secret))));
  }

  for (class, keypair) in inputs {
    runner.run_one(class, || keypair.sign(MESSAGE).to_bytes()[0]);
  }
}

fn valid_p256_secret(rng: &mut BenchRng) -> [u8; EcdsaP256SecretKey::LENGTH] {
  let mut secret = rand_array::<{ EcdsaP256SecretKey::LENGTH }>(rng);
  secret[0] &= 0x7f;
  secret[EcdsaP256SecretKey::LENGTH - 1] |= 1;
  secret
}

fn valid_p384_secret(rng: &mut BenchRng) -> [u8; EcdsaP384SecretKey::LENGTH] {
  let mut secret = rand_array::<{ EcdsaP384SecretKey::LENGTH }>(rng);
  secret[0] &= 0x7f;
  secret[EcdsaP384SecretKey::LENGTH - 1] |= 1;
  secret
}

fn ecdsa_p256_sign_fixed_vs_random_secret(runner: &mut CtRunner, rng: &mut BenchRng) {
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let secret = if matches!(class, Class::Left) {
      [0x42; EcdsaP256SecretKey::LENGTH]
    } else {
      valid_p256_secret(rng)
    };
    inputs.push((class, secret, rand_array::<64>(rng)));
  }

  for (class, secret, blind) in inputs {
    let key = EcdsaP256SecretKey::from_bytes(secret).unwrap();
    runner.run_one(class, || {
      std::hint::black_box(key.try_sign_blinded(MESSAGE, |out| out.copy_from_slice(&blind)).unwrap().to_bytes());
    });
  }
}

fn ecdsa_p256_keypair_sign_fixed_vs_random_secret(runner: &mut CtRunner, rng: &mut BenchRng) {
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let secret = if matches!(class, Class::Left) {
      [0x42; EcdsaP256SecretKey::LENGTH]
    } else {
      valid_p256_secret(rng)
    };
    inputs.push((
      class,
      EcdsaP256Keypair::from_secret_key(EcdsaP256SecretKey::from_bytes(secret).unwrap()),
      rand_array::<64>(rng),
    ));
  }

  for (class, keypair, blind) in inputs {
    runner.run_one(class, || {
      std::hint::black_box(
        keypair
          .try_sign_blinded(MESSAGE, |out| out.copy_from_slice(&blind))
          .unwrap()
          .to_bytes(),
      );
    });
  }
}

fn ecdsa_p384_sign_fixed_vs_random_secret(runner: &mut CtRunner, rng: &mut BenchRng) {
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let secret = if matches!(class, Class::Left) {
      [0x42; EcdsaP384SecretKey::LENGTH]
    } else {
      valid_p384_secret(rng)
    };
    inputs.push((class, secret, rand_array::<96>(rng)));
  }

  for (class, secret, blind) in inputs {
    let key = EcdsaP384SecretKey::from_bytes(secret).unwrap();
    runner.run_one(class, || {
      std::hint::black_box(key.try_sign_blinded(MESSAGE, |out| out.copy_from_slice(&blind)).unwrap().to_bytes());
    });
  }
}

fn ecdsa_p384_keypair_sign_fixed_vs_random_secret(runner: &mut CtRunner, rng: &mut BenchRng) {
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let secret = if matches!(class, Class::Left) {
      [0x42; EcdsaP384SecretKey::LENGTH]
    } else {
      valid_p384_secret(rng)
    };
    inputs.push((
      class,
      EcdsaP384Keypair::from_secret_key(EcdsaP384SecretKey::from_bytes(secret).unwrap()),
      rand_array::<96>(rng),
    ));
  }

  for (class, keypair, blind) in inputs {
    runner.run_one(class, || {
      std::hint::black_box(
        keypair
          .try_sign_blinded(MESSAGE, |out| out.copy_from_slice(&blind))
          .unwrap()
          .to_bytes(),
      );
    });
  }
}

fn ecdsa_p256_diag_nonce_reduce_fixed_vs_random_secret(runner: &mut CtRunner, rng: &mut BenchRng) {
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let secret = if matches!(class, Class::Left) {
      [0x42; EcdsaP256SecretKey::LENGTH]
    } else {
      valid_p256_secret(rng)
    };
    inputs.push((class, secret));
  }

  for (class, secret) in inputs {
    runner.run_one(class, || {
      std::hint::black_box(diag_ecdsa_p256_nonce_reduce_limb_digest(secret, MESSAGE))[0]
    });
  }
}

fn ecdsa_p256_diag_reduce_wide_fixed_vs_random_input(runner: &mut CtRunner, rng: &mut BenchRng) {
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let wide = if matches!(class, Class::Left) {
      [0x42; 64]
    } else {
      rand_array::<64>(rng)
    };
    inputs.push((class, wide));
  }

  for (class, wide) in inputs {
    runner.run_one(class, || {
      std::hint::black_box(diag_ecdsa_p256_reduce_wide_order_limb_digest(wide))[0]
    });
  }
}

fn ecdsa_p256_diag_basepoint_blinded_fixed_vs_random_secret(runner: &mut CtRunner, rng: &mut BenchRng) {
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let secret = if matches!(class, Class::Left) {
      [0x42; EcdsaP256SecretKey::LENGTH]
    } else {
      valid_p256_secret(rng)
    };
    inputs.push((class, secret, rand_array::<64>(rng)));
  }

  for (class, secret, blind) in inputs {
    runner.run_one(class, || {
      std::hint::black_box(diag_ecdsa_p256_basepoint_blinded_limb_digest(secret, blind, MESSAGE))[0]
    });
  }
}

fn ecdsa_p256_diag_scalar_finish_fixed_vs_random_secret(runner: &mut CtRunner, rng: &mut BenchRng) {
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let secret = if matches!(class, Class::Left) {
      [0x42; EcdsaP256SecretKey::LENGTH]
    } else {
      valid_p256_secret(rng)
    };
    inputs.push((class, secret, rand_array::<64>(rng)));
  }

  for (class, secret, nonce_wide) in inputs {
    runner.run_one(class, || {
      std::hint::black_box(diag_ecdsa_p256_scalar_finish_limb_digest(secret, nonce_wide, MESSAGE))[0]
    });
  }
}

fn ecdsa_p256_diag_order_mul_fixed_r_fixed_vs_random_secret(runner: &mut CtRunner, rng: &mut BenchRng) {
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let secret = if matches!(class, Class::Left) {
      [0x42; EcdsaP256SecretKey::LENGTH]
    } else {
      valid_p256_secret(rng)
    };
    inputs.push((class, secret));
  }

  for (class, secret) in inputs {
    runner.run_one(class, || {
      std::hint::black_box(diag_ecdsa_p256_order_mul_fixed_r_limb_digest(secret))[0]
    });
  }
}

fn ecdsa_p256_diag_nonce_inverse_fixed_vs_random_secret(runner: &mut CtRunner, rng: &mut BenchRng) {
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let secret = if matches!(class, Class::Left) {
      [0x42; EcdsaP256SecretKey::LENGTH]
    } else {
      valid_p256_secret(rng)
    };
    inputs.push((class, secret));
  }

  for (class, secret) in inputs {
    runner.run_one(class, || {
      std::hint::black_box(diag_ecdsa_p256_nonce_inverse_limb_digest(secret, MESSAGE))[0]
    });
  }
}

fn ecdsa_p256_diag_final_multiply_fixed_vs_random_secret(runner: &mut CtRunner, rng: &mut BenchRng) {
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let secret = if matches!(class, Class::Left) {
      [0x42; EcdsaP256SecretKey::LENGTH]
    } else {
      valid_p256_secret(rng)
    };
    inputs.push((class, secret, rand_array::<64>(rng)));
  }

  for (class, secret, nonce_wide) in inputs {
    runner.run_one(class, || {
      std::hint::black_box(diag_ecdsa_p256_final_multiply_limb_digest(secret, nonce_wide, MESSAGE))[0]
    });
  }
}

fn ecdsa_p384_diag_nonce_reduce_fixed_vs_random_secret(runner: &mut CtRunner, rng: &mut BenchRng) {
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let secret = if matches!(class, Class::Left) {
      [0x42; EcdsaP384SecretKey::LENGTH]
    } else {
      valid_p384_secret(rng)
    };
    inputs.push((class, secret));
  }

  for (class, secret) in inputs {
    runner.run_one(class, || {
      std::hint::black_box(diag_ecdsa_p384_nonce_reduce_limb_digest(secret, MESSAGE))[0]
    });
  }
}

fn ecdsa_p384_diag_reduce_wide_fixed_vs_random_input(runner: &mut CtRunner, rng: &mut BenchRng) {
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let wide = if matches!(class, Class::Left) {
      [0x42; 96]
    } else {
      rand_array::<96>(rng)
    };
    inputs.push((class, wide));
  }

  for (class, wide) in inputs {
    runner.run_one(class, || {
      std::hint::black_box(diag_ecdsa_p384_reduce_wide_order_limb_digest(wide))[0]
    });
  }
}

fn ecdsa_p384_diag_basepoint_blinded_fixed_vs_random_secret(runner: &mut CtRunner, rng: &mut BenchRng) {
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let secret = if matches!(class, Class::Left) {
      [0x42; EcdsaP384SecretKey::LENGTH]
    } else {
      valid_p384_secret(rng)
    };
    inputs.push((class, secret, rand_array::<96>(rng)));
  }

  for (class, secret, blind) in inputs {
    runner.run_one(class, || {
      std::hint::black_box(diag_ecdsa_p384_basepoint_blinded_limb_digest(secret, blind, MESSAGE))[0]
    });
  }
}

fn ecdsa_p384_diag_scalar_finish_fixed_vs_random_secret(runner: &mut CtRunner, rng: &mut BenchRng) {
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let secret = if matches!(class, Class::Left) {
      [0x42; EcdsaP384SecretKey::LENGTH]
    } else {
      valid_p384_secret(rng)
    };
    inputs.push((class, secret, rand_array::<96>(rng)));
  }

  for (class, secret, nonce_wide) in inputs {
    runner.run_one(class, || {
      std::hint::black_box(diag_ecdsa_p384_scalar_finish_limb_digest(secret, nonce_wide, MESSAGE))[0]
    });
  }
}

fn ecdsa_p384_diag_order_mul_fixed_r_fixed_vs_random_secret(runner: &mut CtRunner, rng: &mut BenchRng) {
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let secret = if matches!(class, Class::Left) {
      [0x42; EcdsaP384SecretKey::LENGTH]
    } else {
      valid_p384_secret(rng)
    };
    inputs.push((class, secret));
  }

  for (class, secret) in inputs {
    runner.run_one(class, || {
      std::hint::black_box(diag_ecdsa_p384_order_mul_fixed_r_limb_digest(secret))[0]
    });
  }
}

fn ecdsa_p384_diag_nonce_inverse_fixed_vs_random_secret(runner: &mut CtRunner, rng: &mut BenchRng) {
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let secret = if matches!(class, Class::Left) {
      [0x42; EcdsaP384SecretKey::LENGTH]
    } else {
      valid_p384_secret(rng)
    };
    inputs.push((class, secret));
  }

  for (class, secret) in inputs {
    runner.run_one(class, || {
      std::hint::black_box(diag_ecdsa_p384_nonce_inverse_limb_digest(secret, MESSAGE))[0]
    });
  }
}

fn ecdsa_p384_diag_final_multiply_fixed_vs_random_secret(runner: &mut CtRunner, rng: &mut BenchRng) {
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let secret = if matches!(class, Class::Left) {
      [0x42; EcdsaP384SecretKey::LENGTH]
    } else {
      valid_p384_secret(rng)
    };
    inputs.push((class, secret, rand_array::<96>(rng)));
  }

  for (class, secret, nonce_wide) in inputs {
    runner.run_one(class, || {
      std::hint::black_box(diag_ecdsa_p384_final_multiply_limb_digest(secret, nonce_wide, MESSAGE))[0]
    });
  }
}

fn rsa_pkcs1v15_fixed_vs_random_message(runner: &mut CtRunner, rng: &mut BenchRng) {
  let key = rsa_ct_fixture_key(RSA_CT_KEY_A_INDEX);
  let sig_len = key.signature_len();
  let (blinding_factor, blinding_inverse) = rsa_blinding_pair(&key);

  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let message = if matches!(class, Class::Left) {
      [0x42; 32]
    } else {
      rand_array::<32>(rng)
    };
    inputs.push((class, message));
  }

  for (class, message) in inputs {
    runner.run_one(class, || {
      let mut out = vec![0u8; sig_len];
      key
        .sign_pkcs1v15_with_blinding_factor(
          RsaPkcs1v15Profile::Sha256,
          &message,
          &blinding_factor,
          &blinding_inverse,
          &mut out,
        )
        .is_ok()
    });
  }
}

fn rsa_pss_fixed_vs_random_message(runner: &mut CtRunner, rng: &mut BenchRng) {
  let key = rsa_ct_fixture_key(RSA_CT_KEY_A_INDEX);
  let sig_len = key.signature_len();
  let (blinding_factor, blinding_inverse) = rsa_blinding_pair(&key);
  let salt = [0x7a; 32];

  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let message = if matches!(class, Class::Left) {
      [0x42; 32]
    } else {
      rand_array::<32>(rng)
    };
    inputs.push((class, message));
  }

  for (class, message) in inputs {
    runner.run_one(class, || {
      let mut out = vec![0u8; sig_len];
      key
        .sign_pss_with_salt_and_blinding_factor(
          RsaPssProfile::Sha256,
          &message,
          &salt,
          &blinding_factor,
          &blinding_inverse,
          &mut out,
        )
        .is_ok()
    });
  }
}

fn rsa_oaep_decrypt_fixed_vs_random_plaintext(runner: &mut CtRunner, rng: &mut BenchRng) {
  let key = rsa_ct_fixture_key(RSA_CT_KEY_A_INDEX);
  let sig_len = key.signature_len();
  let (blinding_factor, blinding_inverse) = rsa_blinding_pair(&key);
  let label = b"rscrypto-ct-oaep";
  let seed = [0x52; 32];

  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let plaintext = if matches!(class, Class::Left) {
      [0x42; 32]
    } else {
      rand_array::<32>(rng)
    };
    let mut ciphertext = vec![0u8; sig_len];
    key
      .public_key()
      .encrypt_oaep_with_random_fill(RsaOaepProfile::Sha256, label, &plaintext, &mut ciphertext, |out| {
        if out.len() != seed.len() {
          return Err(RsaEncryptionError::InvalidLength);
        }
        out.copy_from_slice(&seed);
        Ok(())
      })
      .unwrap();
    inputs.push((class, ciphertext));
  }

  for (class, ciphertext) in inputs {
    runner.run_one(class, || {
      let mut out = vec![0u8; sig_len];
      key
        .decrypt_oaep_with_blinding_factor(
          RsaOaepProfile::Sha256,
          label,
          &ciphertext,
          &blinding_factor,
          &blinding_inverse,
          &mut out,
        )
        .is_ok()
    });
  }
}

fn rsa_pkcs1v15_decrypt_fixed_vs_random_plaintext(runner: &mut CtRunner, rng: &mut BenchRng) {
  let key = rsa_ct_fixture_key(RSA_CT_KEY_A_INDEX);
  let sig_len = key.signature_len();
  let (blinding_factor, blinding_inverse) = rsa_blinding_pair(&key);

  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let plaintext = if matches!(class, Class::Left) {
      [0x42; 32]
    } else {
      rand_array::<32>(rng)
    };
    let mut ciphertext = vec![0u8; sig_len];
    key
      .public_key()
      .encrypt_pkcs1v15_with_random_fill(&plaintext, &mut ciphertext, |out| {
        out.fill(0x5d);
        Ok(())
      })
      .unwrap();
    inputs.push((class, ciphertext));
  }

  for (class, ciphertext) in inputs {
    runner.run_one(class, || {
      let mut out = vec![0u8; sig_len];
      key
        .decrypt_pkcs1v15_with_blinding_factor(&ciphertext, &blinding_factor, &blinding_inverse, &mut out)
        .is_ok()
    });
  }
}

fn rsa_private_component_validation_fixed_vs_random_component(runner: &mut CtRunner, rng: &mut BenchRng) {
  let upper_bound = [0xff; 32];
  let other = [0xa5; 32];
  let fixed = [0x43; 32];

  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let mut component = if matches!(class, Class::Left) {
      fixed
    } else {
      let mut random = rand_array::<32>(rng);
      random[0] |= 1;
      random[31] |= 1;
      if random == other {
        random[1] ^= 1;
      }
      random
    };
    component[0] |= 1;
    component[31] |= 1;
    inputs.push((class, component));
  }

  for (class, component) in inputs {
    runner.run_one(class, || diag_rsa_private_component_validation_32(&component, &upper_bound, &other));
  }
}

fn rsa_private_key_pkcs8_import_key_a_vs_key_b(runner: &mut CtRunner, rng: &mut BenchRng) {
  let der_a = rsa_pkcs8_der(RSA_CT_KEY_A_INDEX);
  let der_b = rsa_pkcs8_der(RSA_CT_KEY_B_SAME_SHAPE_INDEX);
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    inputs.push(random_class(rng));
  }

  let mut der = vec![0u8; der_a.len()];
  for class in inputs {
    let selected = if matches!(class, Class::Left) { &der_a } else { &der_b };
    der.copy_from_slice(selected);
    runner.run_one(class, || {
      let key = RsaPrivateKey::from_pkcs8_der_with_policy(&der, &RsaPublicKeyPolicy::legacy_verification()).unwrap();
      key.signature_len()
    });
  }
}

fn rsa_private_key_pkcs8_validate_key_a_vs_key_b(runner: &mut CtRunner, rng: &mut BenchRng) {
  rsa_private_key_pkcs8_validate_stage_key_a_vs_key_b(runner, rng, u8::MAX);
}

fn rsa_private_key_pkcs8_validate_stage0_key_a_vs_key_b(runner: &mut CtRunner, rng: &mut BenchRng) {
  rsa_private_key_pkcs8_validate_stage_key_a_vs_key_b(runner, rng, 0);
}

fn rsa_private_key_pkcs8_validate_stage1_key_a_vs_key_b(runner: &mut CtRunner, rng: &mut BenchRng) {
  rsa_private_key_pkcs8_validate_stage_key_a_vs_key_b(runner, rng, 1);
}

fn rsa_private_key_pkcs8_validate_stage2_key_a_vs_key_b(runner: &mut CtRunner, rng: &mut BenchRng) {
  rsa_private_key_pkcs8_validate_stage_key_a_vs_key_b(runner, rng, 2);
}

fn rsa_private_key_pkcs8_validate_stage3_key_a_vs_key_b(runner: &mut CtRunner, rng: &mut BenchRng) {
  rsa_private_key_pkcs8_validate_stage_key_a_vs_key_b(runner, rng, 3);
}

fn rsa_private_key_pkcs8_validate_stage4_key_a_vs_key_b(runner: &mut CtRunner, rng: &mut BenchRng) {
  rsa_private_key_pkcs8_validate_stage_key_a_vs_key_b(runner, rng, 4);
}

fn rsa_private_key_pkcs8_validate_stage30_key_a_vs_key_b(runner: &mut CtRunner, rng: &mut BenchRng) {
  rsa_private_key_pkcs8_validate_stage_key_a_vs_key_b(runner, rng, 30);
}

fn rsa_private_key_pkcs8_validate_stage31_key_a_vs_key_b(runner: &mut CtRunner, rng: &mut BenchRng) {
  rsa_private_key_pkcs8_validate_stage_key_a_vs_key_b(runner, rng, 31);
}

fn rsa_private_key_pkcs8_validate_stage32_key_a_vs_key_b(runner: &mut CtRunner, rng: &mut BenchRng) {
  rsa_private_key_pkcs8_validate_stage_key_a_vs_key_b(runner, rng, 32);
}

fn rsa_private_key_pkcs8_validate_stage40_key_a_vs_key_b(runner: &mut CtRunner, rng: &mut BenchRng) {
  rsa_private_key_pkcs8_validate_stage_key_a_vs_key_b(runner, rng, 40);
}

fn rsa_private_key_pkcs8_validate_stage41_key_a_vs_key_b(runner: &mut CtRunner, rng: &mut BenchRng) {
  rsa_private_key_pkcs8_validate_stage_key_a_vs_key_b(runner, rng, 41);
}

fn rsa_private_key_pkcs8_validate_stage42_key_a_vs_key_b(runner: &mut CtRunner, rng: &mut BenchRng) {
  rsa_private_key_pkcs8_validate_stage_key_a_vs_key_b(runner, rng, 42);
}

fn rsa_private_key_pkcs8_import_stage50_key_a_vs_key_b(runner: &mut CtRunner, rng: &mut BenchRng) {
  rsa_private_key_pkcs8_import_stage_key_a_vs_key_b(runner, rng, 50);
}

fn rsa_private_key_pkcs8_import_stage51_key_a_vs_key_b(runner: &mut CtRunner, rng: &mut BenchRng) {
  rsa_private_key_pkcs8_import_stage_key_a_vs_key_b(runner, rng, 51);
}

fn rsa_private_key_pkcs8_import_stage52_key_a_vs_key_b(runner: &mut CtRunner, rng: &mut BenchRng) {
  rsa_private_key_pkcs8_import_stage_key_a_vs_key_b(runner, rng, 52);
}

fn rsa_private_key_pkcs8_import_stage53_key_a_vs_key_b(runner: &mut CtRunner, rng: &mut BenchRng) {
  rsa_private_key_pkcs8_import_stage_key_a_vs_key_b(runner, rng, 53);
}

fn rsa_private_key_pkcs8_import_stage54_key_a_vs_key_b(runner: &mut CtRunner, rng: &mut BenchRng) {
  rsa_private_key_pkcs8_import_stage_key_a_vs_key_b(runner, rng, 54);
}

fn rsa_private_key_pkcs8_import_stage_key_a_vs_key_b(runner: &mut CtRunner, rng: &mut BenchRng, stage: u8) {
  let der_a = rsa_pkcs8_der(RSA_CT_KEY_A_INDEX);
  let der_b = rsa_pkcs8_der(RSA_CT_KEY_B_SAME_SHAPE_INDEX);
  let policy = RsaPublicKeyPolicy::legacy_verification();
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    inputs.push(random_class(rng));
  }

  let mut der = vec![0u8; der_a.len()];
  for class in inputs {
    let selected = if matches!(class, Class::Left) { &der_a } else { &der_b };
    der.copy_from_slice(selected);
    runner.run_one(class, || diag_rsa_import_pkcs8_private_key_der_stage(&der, &policy, stage).unwrap());
  }
}

fn rsa_private_key_pkcs8_validate_stage_key_a_vs_key_b(runner: &mut CtRunner, rng: &mut BenchRng, stage: u8) {
  let der_a = rsa_pkcs8_der(RSA_CT_KEY_A_INDEX);
  let der_b = rsa_pkcs8_der(RSA_CT_KEY_B_SAME_SHAPE_INDEX);
  let policy = RsaPublicKeyPolicy::legacy_verification();
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    inputs.push(random_class(rng));
  }

  let mut der = vec![0u8; der_a.len()];
  for class in inputs {
    let selected = if matches!(class, Class::Left) { &der_a } else { &der_b };
    der.copy_from_slice(selected);
    runner.run_one(class, || {
      if stage == u8::MAX {
        diag_rsa_validate_pkcs8_private_key_der(&der, &policy).unwrap()
      } else {
        diag_rsa_validate_pkcs8_private_key_der_stage(&der, &policy, stage).unwrap()
      }
    });
  }
}

fn rsa_private_key_pkcs8_export_key_a_vs_key_b(runner: &mut CtRunner, rng: &mut BenchRng) {
  let key_a = rsa_ct_fixture_key(RSA_CT_KEY_A_INDEX);
  let key_b = rsa_ct_fixture_key(RSA_CT_KEY_B_SAME_SHAPE_INDEX);
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    inputs.push(random_class(rng));
  }

  for class in inputs {
    let key = if matches!(class, Class::Left) { &key_a } else { &key_b };
    runner.run_one(class, || {
      let der = key.to_pkcs8_der();
      der.len()
    });
  }
}

fn hkdf_sha2_fixed_vs_random_ikm(runner: &mut CtRunner, rng: &mut BenchRng) {
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let ikm = if matches!(class, Class::Left) {
      [0x88; 32]
    } else {
      rand_array::<32>(rng)
    };
    inputs.push((class, ikm));
  }

  for (class, ikm) in inputs {
    runner.run_one(class, || {
      let mut out256 = [0u8; 32];
      let mut out384 = [0u8; 32];
      HkdfSha256::derive(b"rscrypto salt", &ikm, b"rscrypto info", &mut out256).is_ok()
        && HkdfSha384::derive(b"rscrypto salt", &ikm, b"rscrypto info", &mut out384).is_ok()
    });
  }
}

fn pbkdf2_sha2_fixed_vs_random_password(runner: &mut CtRunner, rng: &mut BenchRng) {
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let password = if matches!(class, Class::Left) {
      [0x50; 32]
    } else {
      rand_array::<32>(rng)
    };
    inputs.push((class, password));
  }

  for (class, password) in inputs {
    runner.run_one(class, || {
      let mut out256 = [0u8; 32];
      let mut out512 = [0u8; 32];
      Pbkdf2Sha256::derive_key(&password, b"rscrypto salt", 8, &mut out256).is_ok()
        && Pbkdf2Sha512::derive_key(&password, b"rscrypto salt", 8, &mut out512).is_ok()
    });
  }
}

fn argon2i_fixed_vs_random_password(runner: &mut CtRunner, rng: &mut BenchRng) {
  let params = argon2i_params();
  let salt = [0xA5; 16];
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let password = if matches!(class, Class::Left) {
      [0x70; 32]
    } else {
      rand_array::<32>(rng)
    };
    inputs.push((class, password));
  }

  for (class, password) in inputs {
    runner.run_one(class, || {
      let mut out = [0u8; 16];
      Argon2i::hash(&params, &password, &salt, &mut out).is_ok()
    });
  }
}

fn argon2i_parallel_fixed_vs_random_password(runner: &mut CtRunner, rng: &mut BenchRng) {
  let params = argon2i_parallel_params();
  let salt = [0xA5; 16];
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let password = if matches!(class, Class::Left) {
      [0x71; 32]
    } else {
      rand_array::<32>(rng)
    };
    inputs.push((class, password));
  }

  for (class, password) in inputs {
    runner.run_one(class, || {
      let mut out = [0u8; 16];
      Argon2i::hash(&params, &password, &salt, &mut out).is_ok()
    });
  }
}

macro_rules! aead_fixed_vs_random_key_seal {
  ($name:ident, $cipher:ty, $key:ty, $nonce:ty, $fixed_key:expr, $key_len:expr, $nonce_byte:expr) => {
    fn $name(runner: &mut CtRunner, rng: &mut BenchRng) {
      let nonce = <$nonce>::from_bytes([$nonce_byte; <$nonce>::LENGTH]);
      let mut inputs = Vec::with_capacity(samples());
      for _ in 0..samples() {
        let class = random_class(rng);
        let key = if matches!(class, Class::Left) {
          $fixed_key
        } else {
          rand_array::<$key_len>(rng)
        };
        inputs.push((class, <$key>::from_bytes(key), nonce));
      }

      for (class, key, nonce) in inputs {
        let cipher = <$cipher>::new(&key);
        runner.run_one(class, || {
          let mut buffer = AEAD_PLAINTEXT;
          cipher.encrypt_in_place(&nonce, AAD, &mut buffer).is_ok()
        });
      }
    }
  };
}

aead_fixed_vs_random_key_seal!(
  aes128gcm_fixed_vs_random_key_seal,
  Aes128Gcm,
  Aes128GcmKey,
  Nonce96,
  [0x33; Aes128Gcm::KEY_SIZE],
  16,
  0x33
);
aead_fixed_vs_random_key_seal!(
  aes256gcm_fixed_vs_random_key_seal,
  Aes256Gcm,
  Aes256GcmKey,
  Nonce96,
  [0x34; Aes256Gcm::KEY_SIZE],
  32,
  0x34
);
aead_fixed_vs_random_key_seal!(
  aes128gcmsiv_fixed_vs_random_key_seal,
  Aes128GcmSiv,
  Aes128GcmSivKey,
  Nonce96,
  [0x35; Aes128GcmSiv::KEY_SIZE],
  16,
  0x35
);
aead_fixed_vs_random_key_seal!(
  aes256gcmsiv_fixed_vs_random_key_seal,
  Aes256GcmSiv,
  Aes256GcmSivKey,
  Nonce96,
  [0x36; Aes256GcmSiv::KEY_SIZE],
  32,
  0x36
);
aead_fixed_vs_random_key_seal!(
  ietf_chacha20poly1305_fixed_vs_random_key_seal,
  ChaCha20Poly1305,
  ChaCha20Poly1305Key,
  Nonce96,
  [0x37; ChaCha20Poly1305::KEY_SIZE],
  32,
  0x37
);
aead_fixed_vs_random_key_seal!(
  xchacha20poly1305_fixed_vs_random_key_seal,
  XChaCha20Poly1305,
  XChaCha20Poly1305Key,
  Nonce192,
  [0x38; XChaCha20Poly1305::KEY_SIZE],
  32,
  0x38
);
aead_fixed_vs_random_key_seal!(
  aegis256_fixed_vs_random_key_seal,
  Aegis256,
  Aegis256Key,
  Nonce256,
  [0x39; Aegis256::KEY_SIZE],
  32,
  0x39
);
aead_fixed_vs_random_key_seal!(
  ascon_aead128_fixed_vs_random_key_seal,
  AsconAead128,
  AsconAead128Key,
  Nonce128,
  [0x3A; AsconAead128::KEY_SIZE],
  16,
  0x3A
);

macro_rules! blake2_keyed_fixed_vs_random {
  ($name:ident, $ty:ty) => {
    fn $name(runner: &mut CtRunner, rng: &mut BenchRng) {
      let mut inputs = Vec::with_capacity(samples());
      for _ in 0..samples() {
        let class = random_class(rng);
        let key = if matches!(class, Class::Left) {
          [0xA3; 32]
        } else {
          rand_array::<32>(rng)
        };
        inputs.push((class, key));
      }

      for (class, key) in inputs {
        runner.run_one(class, || <$ty>::keyed_digest(&key, MESSAGE)[0]);
      }
    }
  };
}

blake2_keyed_fixed_vs_random!(blake2b256_keyed_fixed_vs_random_key, Blake2b256);
blake2_keyed_fixed_vs_random!(blake2b512_keyed_fixed_vs_random_key, Blake2b512);
blake2_keyed_fixed_vs_random!(blake2s128_keyed_fixed_vs_random_key, Blake2s128);
blake2_keyed_fixed_vs_random!(blake2s256_keyed_fixed_vs_random_key, Blake2s256);

fn blake3_keyed_fixed_vs_random_key(runner: &mut CtRunner, rng: &mut BenchRng) {
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let key = if matches!(class, Class::Left) {
      [0xA3; 32]
    } else {
      rand_array::<32>(rng)
    };
    inputs.push((class, key));
  }

  for (class, key) in inputs {
    runner.run_one(class, || Blake3::keyed_digest(&key, MESSAGE).to_bytes()[0]);
  }
}

fn blake3_keyed_parallel_fixed_vs_random_key(runner: &mut CtRunner, rng: &mut BenchRng) {
  let message = vec![0xA5; BLAKE3_PARALLEL_MESSAGE_LEN];
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let key = if matches!(class, Class::Left) {
      [0xA4; 32]
    } else {
      rand_array::<32>(rng)
    };
    inputs.push((class, key));
  }

  for (class, key) in inputs {
    runner.run_one(class, || Blake3::keyed_digest(&key, &message).to_bytes()[0]);
  }
}

ctbench_main_with_seeds!(
  (constant_time_eq_equal_vs_first_diff, Some(0x727363727970746f)),
  (secret_wrappers_eq_and_debug_fixed_vs_random, Some(0x7365637265745f77)),
  (hmac_sha256_valid_vs_invalid_tag, Some(0x686d61635f736861)),
  (hmac_sha384_valid_vs_invalid_tag, Some(0x686d61633338345f)),
  (hmac_sha512_valid_vs_invalid_tag, Some(0x686d61633531325f)),
  (kmac256_valid_vs_invalid_tag, Some(0x6b6d61633235365f)),
  (blake3_keyed_valid_vs_invalid_tag, Some(0x626c616b65335f63)),
  (aes128gcm_fixed_vs_random_key_open, Some(0x616573313238676f)),
  (aes256gcm_fixed_vs_random_key_open, Some(0x616573323536676f)),
  (aes128gcmsiv_fixed_vs_random_key_open, Some(0x6131323867736f70)),
  (aes256gcmsiv_fixed_vs_random_key_open, Some(0x6132353667736f70)),
  (aes128_gcm_siv_diag_derive_fixed_vs_random_key, Some(0x6131323867646572)),
  (aes256_gcm_siv_diag_derive_fixed_vs_random_key, Some(0x6132353667646572)),
  (gcm_siv_diag_polyval_fixed_vs_random_auth_key, Some(0x6763736976706f6c)),
  (aes128_gcm_siv_diag_raw_tag_aes_fixed_vs_random_key, Some(0x6131323867746167)),
  (aes256_gcm_siv_diag_raw_tag_aes_fixed_vs_random_key, Some(0x6132353667746167)),
  (aes128_gcm_diag_ctr32_be_fixed_vs_random_key, Some(0x6731323863747262)),
  (aes256_gcm_diag_ctr32_be_fixed_vs_random_key, Some(0x6732353663747262)),
  (aes128_gcm_diag_ghash_fixed_vs_random_h, Some(0x6731323867686173)),
  (aes256_gcm_diag_ghash_fixed_vs_random_h, Some(0x6732353667686173)),
  (aes128_gcm_diag_tag_aes_fixed_vs_random_key, Some(0x673132387461676b)),
  (aes256_gcm_diag_tag_aes_fixed_vs_random_key, Some(0x673235367461676b)),
  (aes128_gcm_siv_diag_ctr32_fixed_vs_random_key, Some(0x733132386374726b)),
  (aes256_gcm_siv_diag_ctr32_fixed_vs_random_key, Some(0x733235366374726b)),
  (aes128_gcm_siv_diag_raw_tag_aes_varying_block, Some(0x7331323874616762)),
  (aes256_gcm_siv_diag_raw_tag_aes_varying_block, Some(0x7332353674616762)),
  (ietf_chacha20poly1305_fixed_vs_random_key_open, Some(0x6368613230706f70)),
  (xchacha20poly1305_fixed_vs_random_key_open, Some(0x7863686163686132)),
  (aegis256_fixed_vs_random_key_open, Some(0x61656769736f706e)),
  (ascon_aead128_fixed_vs_random_key_open, Some(0x6173636f6e6f706e)),
  (x25519_fixed_vs_random_scalar, Some(0x7832353531395f63)),
  (mlkem512_keygen_secret_noise_fixed_vs_random, Some(0x6d6b3531326b676e)),
  (mlkem512_encapsulate_fixed_vs_random_coins, Some(0x6d6b353132656e63)),
  (mlkem512_decapsulate_fixed_vs_random_secret_key, Some(0x6d6b353132646563)),
  (mlkem512_decapsulate_rejection_seed_fixed_vs_random, Some(0x6d6b35313272656a)),
  (mlkem768_keygen_secret_noise_fixed_vs_random, Some(0x6d6c6b656d6b676e)),
  (mlkem768_encapsulate_fixed_vs_random_coins, Some(0x6d6c6b656d656e63)),
  (mlkem768_decapsulate_fixed_vs_random_secret_key, Some(0x6d6c6b656d646563)),
  (mlkem768_decapsulate_rejection_seed_fixed_vs_random, Some(0x6d6c6b656d72656a)),
  (mlkem1024_keygen_secret_noise_fixed_vs_random, Some(0x6d6b313032346b67)),
  (mlkem1024_encapsulate_fixed_vs_random_coins, Some(0x6d6b31303234656e)),
  (mlkem1024_decapsulate_fixed_vs_random_secret_key, Some(0x6d6b313032346465)),
  (mlkem1024_decapsulate_rejection_seed_fixed_vs_random, Some(0x6d6b31303234726a)),
  (mlkem_arithmetic_ntt_fixed_vs_random_poly, Some(0x6d6c6b6e7474706f)),
  (mlkem_arithmetic_inverse_ntt_fixed_vs_random_poly, Some(0x6d6c6b696e747470)),
  (mlkem_arithmetic_to_product_domain_fixed_vs_random_poly, Some(0x6d6c6b746f70726f)),
  (mlkem_arithmetic_from_product_domain_fixed_vs_random_poly, Some(0x6d6c6b667270726f)),
  (mlkem_arithmetic_basemul_fixed_vs_random_operands, Some(0x6d6c6b626173656d)),
  (mlkem1024_arithmetic_dot_fixed_vs_random_operands, Some(0x6d6c6b313032646f)),
  (ed25519_sign_fixed_vs_random_secret, Some(0x656432353531395f)),
  (ed25519_public_key_fixed_vs_random_secret, Some(0x6564323535313950)),
  (ed25519_sha512_secret_expand_fixed_vs_random_secret, Some(0x6564323535314853)),
  (ed25519_keypair_sign_fixed_vs_random_secret, Some(0x656432353531394b)),
  (ecdsa_p256_sign_fixed_vs_random_secret, Some(0x703235365f736967)),
  (ecdsa_p256_keypair_sign_fixed_vs_random_secret, Some(0x703235365f6b6579)),
  (ecdsa_p384_sign_fixed_vs_random_secret, Some(0x703338345f736967)),
  (ecdsa_p384_keypair_sign_fixed_vs_random_secret, Some(0x703338345f6b6579)),
  (ecdsa_p256_diag_nonce_reduce_fixed_vs_random_secret, Some(0x703235366e6f6e63)),
  (ecdsa_p256_diag_reduce_wide_fixed_vs_random_input, Some(0x7032353672656475)),
  (ecdsa_p256_diag_basepoint_blinded_fixed_vs_random_secret, Some(0x7032353662617365)),
  (ecdsa_p256_diag_scalar_finish_fixed_vs_random_secret, Some(0x7032353666696e73)),
  (ecdsa_p256_diag_order_mul_fixed_r_fixed_vs_random_secret, Some(0x703235366d756c72)),
  (ecdsa_p256_diag_nonce_inverse_fixed_vs_random_secret, Some(0x70323536696e766b)),
  (ecdsa_p256_diag_final_multiply_fixed_vs_random_secret, Some(0x703235366d756c73)),
  (ecdsa_p384_diag_nonce_reduce_fixed_vs_random_secret, Some(0x703338346e6f6e63)),
  (ecdsa_p384_diag_reduce_wide_fixed_vs_random_input, Some(0x7033383472656475)),
  (ecdsa_p384_diag_basepoint_blinded_fixed_vs_random_secret, Some(0x7033383462617365)),
  (ecdsa_p384_diag_scalar_finish_fixed_vs_random_secret, Some(0x7033383466696e73)),
  (ecdsa_p384_diag_order_mul_fixed_r_fixed_vs_random_secret, Some(0x703338346d756c72)),
  (ecdsa_p384_diag_nonce_inverse_fixed_vs_random_secret, Some(0x70333834696e766b)),
  (ecdsa_p384_diag_final_multiply_fixed_vs_random_secret, Some(0x703338346d756c73)),
  (rsa_pkcs1v15_fixed_vs_random_message, Some(0x7273615f7369676e)),
  (rsa_pss_fixed_vs_random_message, Some(0x7273615f70737373)),
  (rsa_oaep_decrypt_fixed_vs_random_plaintext, Some(0x7273615f6f616570)),
  (rsa_pkcs1v15_decrypt_fixed_vs_random_plaintext, Some(0x7273615f64656331)),
  (rsa_private_component_validation_fixed_vs_random_component, Some(0x7273615f636f6d70)),
  (rsa_private_key_pkcs8_import_key_a_vs_key_b, Some(0x7273615f6b657969)),
  (rsa_private_key_pkcs8_validate_key_a_vs_key_b, Some(0x7273615f76616c69)),
  (rsa_private_key_pkcs8_validate_stage0_key_a_vs_key_b, Some(0x7273615f76733030)),
  (rsa_private_key_pkcs8_validate_stage1_key_a_vs_key_b, Some(0x7273615f76733031)),
  (rsa_private_key_pkcs8_validate_stage2_key_a_vs_key_b, Some(0x7273615f76733032)),
  (rsa_private_key_pkcs8_validate_stage3_key_a_vs_key_b, Some(0x7273615f76733033)),
  (rsa_private_key_pkcs8_validate_stage4_key_a_vs_key_b, Some(0x7273615f76733034)),
  (rsa_private_key_pkcs8_validate_stage30_key_a_vs_key_b, Some(0x7273615f76333030)),
  (rsa_private_key_pkcs8_validate_stage31_key_a_vs_key_b, Some(0x7273615f76333031)),
  (rsa_private_key_pkcs8_validate_stage32_key_a_vs_key_b, Some(0x7273615f76333032)),
  (rsa_private_key_pkcs8_validate_stage40_key_a_vs_key_b, Some(0x7273615f76343030)),
  (rsa_private_key_pkcs8_validate_stage41_key_a_vs_key_b, Some(0x7273615f76343031)),
  (rsa_private_key_pkcs8_validate_stage42_key_a_vs_key_b, Some(0x7273615f76343032)),
  (rsa_private_key_pkcs8_import_stage50_key_a_vs_key_b, Some(0x7273615f69353030)),
  (rsa_private_key_pkcs8_import_stage51_key_a_vs_key_b, Some(0x7273615f69353031)),
  (rsa_private_key_pkcs8_import_stage52_key_a_vs_key_b, Some(0x7273615f69353032)),
  (rsa_private_key_pkcs8_import_stage53_key_a_vs_key_b, Some(0x7273615f69353033)),
  (rsa_private_key_pkcs8_import_stage54_key_a_vs_key_b, Some(0x7273615f69353034)),
  (rsa_private_key_pkcs8_export_key_a_vs_key_b, Some(0x7273615f6b657978)),
  (hkdf_sha2_fixed_vs_random_ikm, Some(0x686b64665f736861)),
  (pbkdf2_sha2_fixed_vs_random_password, Some(0x70626b6466325f73)),
  (argon2i_fixed_vs_random_password, Some(0x6172676f6e32695f)),
  (argon2i_parallel_fixed_vs_random_password, Some(0x6172673269706172)),
  (aes128gcm_fixed_vs_random_key_seal, Some(0x6165733132386773)),
  (aes256gcm_fixed_vs_random_key_seal, Some(0x6165733235366773)),
  (aes128gcmsiv_fixed_vs_random_key_seal, Some(0x6131323867737365)),
  (aes256gcmsiv_fixed_vs_random_key_seal, Some(0x6132353667737365)),
  (ietf_chacha20poly1305_fixed_vs_random_key_seal, Some(0x6368616368613230)),
  (xchacha20poly1305_fixed_vs_random_key_seal, Some(0x7863686132307365)),
  (aegis256_fixed_vs_random_key_seal, Some(0x6165676973736561)),
  (ascon_aead128_fixed_vs_random_key_seal, Some(0x6173636f6e736561)),
  (blake2b256_keyed_fixed_vs_random_key, Some(0x6232623235365f6b)),
  (blake2b512_keyed_fixed_vs_random_key, Some(0x6232623531325f6b)),
  (blake2s128_keyed_fixed_vs_random_key, Some(0x6232733132385f6b)),
  (blake2s256_keyed_fixed_vs_random_key, Some(0x6232733235365f6b)),
  (blake3_keyed_fixed_vs_random_key, Some(0x626c616b65335f6b)),
  (blake3_keyed_parallel_fixed_vs_random_key, Some(0x62336b6579706172))
);
