use dudect_bencher::{BenchRng, Class, CtRunner, ctbench_main_with_seeds, rand::RngExt};
use rscrypto::{
  Argon2Params, Argon2i, Blake2b256, Blake2b512, Blake2s128, Blake2s256, Blake3, Blake3KeyedHash,
  ChaCha20Poly1305, ChaCha20Poly1305Key, Ed25519Keypair, Ed25519SecretKey, HkdfSha256, HkdfSha384, HmacSha256, Kmac256,
  Pbkdf2Sha256, Pbkdf2Sha512, RsaPkcs1v15Profile, RsaPrivateKey, SecretBytes, Sha512, X25519SecretKey, XChaCha20Poly1305,
  XChaCha20Poly1305Key,
  aead::{Nonce96, Nonce192},
  traits::ct,
};

const DEFAULT_SAMPLES: usize = 20_000;
const MESSAGE: &[u8] = b"rscrypto constant-time dudect timing lane input";
const AAD: &[u8] = b"rscrypto-ct";
const AEAD_PLAINTEXT: [u8; 44] = *b"constant-time seal buffer for key validation";
const RSA_PKCS1_2048: &str = include_str!("../../../testdata/rsa/wycheproof/rsa_pkcs1_2048_test.json");

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

fn hmac_sha256_valid_vs_invalid_tag(runner: &mut CtRunner, rng: &mut BenchRng) {
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let key = rand_array::<32>(rng);
    let mut expected = HmacSha256::mac(&key, MESSAGE);
    if matches!(class, Class::Right) {
      expected[0] ^= 1;
    }
    inputs.push((class, key, expected));
  }

  for (class, key, expected) in inputs {
    runner.run_one(class, || HmacSha256::verify_tag(&key, MESSAGE, &expected).is_ok());
  }
}

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

fn xchacha20poly1305_fixed_vs_random_key_open(runner: &mut CtRunner, rng: &mut BenchRng) {
  let nonce = Nonce192::from_bytes([0x19; Nonce192::LENGTH]);
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let key = if matches!(class, Class::Left) {
      [0x17; XChaCha20Poly1305::KEY_SIZE]
    } else {
      rand_array::<{ XChaCha20Poly1305::KEY_SIZE }>(rng)
    };
    let key = XChaCha20Poly1305Key::from_bytes(key);
    let cipher = XChaCha20Poly1305::new(&key);
    let mut ciphertext = AEAD_PLAINTEXT;
    let tag = cipher.encrypt_in_place(&nonce, AAD, &mut ciphertext).unwrap();
    inputs.push((class, key, nonce, ciphertext, tag));
  }

  for (class, key, nonce, ciphertext, tag) in inputs {
    let cipher = XChaCha20Poly1305::new(&key);
    runner.run_one(class, || {
      let mut buffer = ciphertext;
      cipher.decrypt_in_place(&nonce, AAD, &mut buffer, &tag).is_ok()
    });
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

fn rsa_pkcs1v15_fixed_vs_random_message(runner: &mut CtRunner, rng: &mut BenchRng) {
  let der = rsa_pkcs8_der(0);
  let key = RsaPrivateKey::from_pkcs8_der(&der).unwrap();
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

fn rsa_private_key_pkcs8_roundtrip_key_a_vs_key_b(runner: &mut CtRunner, rng: &mut BenchRng) {
  let der_a = rsa_pkcs8_der(0);
  let der_b = rsa_pkcs8_der(1);
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    inputs.push(random_class(rng));
  }

  for class in inputs {
    let der = if matches!(class, Class::Left) { &der_a } else { &der_b };
    runner.run_one(class, || {
      let key = RsaPrivateKey::from_pkcs8_der(der).unwrap();
      key.to_pkcs8_der().len()
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

fn chacha20poly1305_fixed_vs_random_key_seal(runner: &mut CtRunner, rng: &mut BenchRng) {
  let nonce = Nonce96::from_bytes([0x24; Nonce96::LENGTH]);
  let mut inputs = Vec::with_capacity(samples());
  for _ in 0..samples() {
    let class = random_class(rng);
    let key = if matches!(class, Class::Left) {
      [0x11; ChaCha20Poly1305::KEY_SIZE]
    } else {
      rand_array::<{ ChaCha20Poly1305::KEY_SIZE }>(rng)
    };
    inputs.push((class, ChaCha20Poly1305Key::from_bytes(key), nonce));
  }

  for (class, key, nonce) in inputs {
    let cipher = ChaCha20Poly1305::new(&key);
    runner.run_one(class, || {
      let mut buffer = AEAD_PLAINTEXT;
      cipher.encrypt_in_place(&nonce, AAD, &mut buffer).is_ok()
    });
  }
}

fn blake2_blake3_keyed_fixed_vs_random_key(runner: &mut CtRunner, rng: &mut BenchRng) {
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
    runner.run_one(class, || {
      let mut acc = Blake2b256::keyed_digest(&key, MESSAGE)[0];
      acc ^= Blake2b512::keyed_digest(&key, MESSAGE)[0];
      acc ^= Blake2s128::keyed_digest(&key, MESSAGE)[0];
      acc ^= Blake2s256::keyed_digest(&key, MESSAGE)[0];
      acc ^= Blake3::keyed_digest(&key, MESSAGE).to_bytes()[0];
      acc
    });
  }
}

ctbench_main_with_seeds!(
  (constant_time_eq_equal_vs_first_diff, Some(0x727363727970746f)),
  (secret_wrappers_eq_and_debug_fixed_vs_random, Some(0x7365637265745f77)),
  (hmac_sha256_valid_vs_invalid_tag, Some(0x686d61635f736861)),
  (kmac256_valid_vs_invalid_tag, Some(0x6b6d61633235365f)),
  (blake3_keyed_valid_vs_invalid_tag, Some(0x626c616b65335f63)),
  (xchacha20poly1305_fixed_vs_random_key_open, Some(0x7863686163686132)),
  (x25519_fixed_vs_random_scalar, Some(0x7832353531395f63)),
  (ed25519_sign_fixed_vs_random_secret, Some(0x656432353531395f)),
  (ed25519_public_key_fixed_vs_random_secret, Some(0x6564323535313950)),
  (ed25519_sha512_secret_expand_fixed_vs_random_secret, Some(0x6564323535314853)),
  (ed25519_keypair_sign_fixed_vs_random_secret, Some(0x656432353531394b)),
  (rsa_pkcs1v15_fixed_vs_random_message, Some(0x7273615f7369676e)),
  (rsa_private_key_pkcs8_roundtrip_key_a_vs_key_b, Some(0x7273615f6b657973)),
  (hkdf_sha2_fixed_vs_random_ikm, Some(0x686b64665f736861)),
  (pbkdf2_sha2_fixed_vs_random_password, Some(0x70626b6466325f73)),
  (argon2i_fixed_vs_random_password, Some(0x6172676f6e32695f)),
  (chacha20poly1305_fixed_vs_random_key_seal, Some(0x6368616368613230)),
  (blake2_blake3_keyed_fixed_vs_random_key, Some(0x626c616b65325f33))
);
