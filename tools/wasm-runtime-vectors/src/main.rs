use rscrypto::{
  AesSivCmac256, AesSivCmac256Key, AesSivCmac256Nonce, Blake2b512, Blake3, Digest, EcdsaP256SecretKey,
  EcdsaP384SecretKey, RsaPrivateKey, RsaPrivateOpError, RsaPssProfile, RsaPublicKeyPolicy, Sha256, Sha512,
};
use rscrypto::aead::expert::header_protection::{
  Aes128HeaderProtection, Aes128HeaderProtectionKey, Aes256HeaderProtection, Aes256HeaderProtectionKey,
  ChaCha20HeaderProtection, ChaCha20HeaderProtectionKey,
};
use rscrypto::hashes::legacy::WebSocketAcceptDigest;

const RSA_PRIVATE_KEY_PEM: &str = include_str!("../fixtures/rsa2048_private_pkcs1.txt");

fn hex_value(byte: u8) -> Option<u8> {
  match byte {
    b'0'..=b'9' => Some(byte.strict_sub(b'0')),
    b'a'..=b'f' => Some(byte.strict_sub(b'a').strict_add(10)),
    b'A'..=b'F' => Some(byte.strict_sub(b'A').strict_add(10)),
    _ => None,
  }
}

fn assert_hex(actual: &[u8], expected: &str) {
  assert_eq!(actual.len().strict_mul(2), expected.len());
  for (i, chunk) in expected.as_bytes().as_chunks::<2>().0.iter().enumerate() {
    let high = hex_value(chunk[0]).expect("known hash vector must contain hexadecimal digits");
    let low = hex_value(chunk[1]).expect("known hash vector must contain hexadecimal digits");
    let byte = high.strict_shl(4) | low;
    assert_eq!(actual[i], byte, "hex mismatch at byte {i}");
  }
}

fn base64_value(byte: u8) -> Option<u8> {
  match byte {
    b'A'..=b'Z' => Some(byte.strict_sub(b'A')),
    b'a'..=b'z' => Some(byte.strict_sub(b'a').strict_add(26)),
    b'0'..=b'9' => Some(byte.strict_sub(b'0').strict_add(52)),
    b'+' => Some(62),
    b'/' => Some(63),
    b'=' => Some(0),
    _ => None,
  }
}

fn decode_base64(input: &str) -> Vec<u8> {
  let mut encoded = Vec::with_capacity(input.len());
  for line in input.lines() {
    if !line.starts_with("-----") {
      encoded.extend_from_slice(line.as_bytes());
    }
  }
  assert_eq!(encoded.len() % 4, 0, "PEM body must contain complete base64 quanta");

  let mut out = Vec::with_capacity(encoded.len().strict_div(4).strict_mul(3));
  for quantum in encoded.as_chunks::<4>().0 {
    let a = base64_value(quantum[0]).expect("RSA fixture must contain valid base64");
    let b = base64_value(quantum[1]).expect("RSA fixture must contain valid base64");
    let c = base64_value(quantum[2]).expect("RSA fixture must contain valid base64");
    let d = base64_value(quantum[3]).expect("RSA fixture must contain valid base64");
    out.push(a.strict_shl(2) | b.strict_shr(4));
    if quantum[2] != b'=' {
      out.push((b & 0x0f).strict_shl(4) | c.strict_shr(2));
    }
    if quantum[3] != b'=' {
      out.push((c & 0x03).strict_shl(6) | d);
    }
  }
  out
}

fn decode_hex<const N: usize>(encoded: &str) -> [u8; N] {
  assert_eq!(encoded.len(), N.strict_mul(2));
  let mut decoded = [0u8; N];
  for (byte, chunk) in decoded.iter_mut().zip(encoded.as_bytes().as_chunks::<2>().0) {
    let high = hex_value(chunk[0]).expect("known vector must contain hexadecimal digits");
    let low = hex_value(chunk[1]).expect("known vector must contain hexadecimal digits");
    *byte = high.strict_shl(4) | low;
  }
  decoded
}

fn patterned_bytes(len: usize) -> Vec<u8> {
  (0..len)
    .map(|i| {
      i.to_le_bytes()[0]
        .wrapping_mul(37)
        .wrapping_add(i.strict_shr(8).to_le_bytes()[0])
    })
    .collect()
}

fn assert_core_hash_vectors_match_known_outputs() {
  assert_hex(
    &Sha256::digest(b""),
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  );
  assert_hex(
    &Sha512::digest(b"abc"),
    "\
ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a\
2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f",
  );
  assert_hex(
    &Blake2b512::digest(b""),
    "\
786a02f742015903c6c6fd852552d272912f4740e15847618a86e217f71f5419\
d25e1031afee585313896444934eb04b903a685b1448b755d56f701afe9be2ce",
  );
  assert_hex(
    &Blake3::digest(b""),
    "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262",
  );
}

fn assert_streaming_hashes_match_oneshot_across_block_boundaries() {
  let data = patterned_bytes(4097);

  let sha256_oneshot = Sha256::digest(&data);
  let mut sha256 = Sha256::new();
  for chunk in data.chunks(17) {
    sha256.update(chunk);
  }
  assert_eq!(sha256.finalize(), sha256_oneshot);

  let sha512_oneshot = Sha512::digest(&data);
  let mut sha512 = Sha512::new();
  for chunk in data.chunks(31) {
    sha512.update(chunk);
  }
  assert_eq!(sha512.finalize(), sha512_oneshot);

  let blake2b_oneshot = Blake2b512::digest(&data);
  let mut blake2b = Blake2b512::new();
  for chunk in data.chunks(127) {
    blake2b.update(chunk);
  }
  assert_eq!(blake2b.finalize(), blake2b_oneshot);

  let blake3_oneshot = Blake3::digest(&data);
  let mut blake3 = Blake3::new();
  for chunk in data.chunks(1025) {
    blake3.update(chunk);
  }
  assert_eq!(blake3.finalize(), blake3_oneshot);
}

fn assert_rsa_caller_random_signing_roundtrips() {
  // The fixture is the first RSA-2048 key from Wycheproof's
  // `rsa_pkcs1_2048_sig_gen_test.json`; only the key is copied here so this
  // runtime executable does not embed the complete vector corpus.
  let private_key_der = decode_base64(RSA_PRIVATE_KEY_PEM);
  let key = RsaPrivateKey::from_pkcs1_der_with_policy(&private_key_der, &RsaPublicKeyPolicy::legacy_verification())
    .expect("Wycheproof RSA-2048 private key must import under the explicit legacy policy");
  let profile = RsaPssProfile::Sha256;
  let message = b"rscrypto caller-random RSA signing on wasm32-wasip1";
  let modulus_len = key.signature_len();
  let mut signature = vec![0u8; modulus_len];
  let mut scratch = key.private_scratch();

  let mut requests = 0usize;
  key
    .sign_pss_with_random_fill_and_scratch(profile, message, &mut signature, &mut scratch, |random| {
      if requests == 0 {
        assert_eq!(random.len(), Sha256::OUTPUT_SIZE);
        random.fill(0x5a);
      } else {
        assert_eq!(random.len(), modulus_len);
        random.fill(0);
        random[modulus_len.strict_sub(1)] = 2;
      }
      requests = requests.strict_add(1);
      Ok::<(), ()>(())
    })
    .expect("caller-random RSA-PSS signing must succeed without OS entropy");
  assert_eq!(requests, 2, "factor two must be accepted on the first bounded attempt");
  key
    .public_key()
    .verify_pss(profile, message, &signature)
    .expect("caller-random RSA-PSS signature must verify");

  let error = key
    .sign_pss_with_random_fill_and_scratch(profile, message, &mut signature, &mut scratch, |random| {
      random.fill(0xa5);
      Err::<(), ()>(())
    })
    .expect_err("caller entropy failure must fail closed");
  assert_eq!(error, RsaPrivateOpError::EntropyUnavailable);
  assert!(signature.iter().all(|&byte| byte == 0));
}

fn assert_websocket_accept_digest_matches_rfc_6455() {
  let digest = WebSocketAcceptDigest::compute(b"dGhlIHNhbXBsZSBub25jZQ==");
  assert_eq!(
    digest.as_ref(),
    [
      0xb3, 0x7a, 0x4f, 0x2c, 0xc0, 0x62, 0x4f, 0x16, 0x90, 0xf6, 0x46, 0x06, 0xcf, 0x38, 0x59, 0x45, 0xb2,
      0xbe, 0xc4, 0xea,
    ]
  );
}

fn assert_header_protection_vectors_match_known_outputs() {
  // RFC 9001 Appendix A.2 client Initial header-protection vector.
  let aes128 = Aes128HeaderProtection::new(&Aes128HeaderProtectionKey::from_bytes(decode_hex(
    "9f50449e04a0e810283a1e9933adedd2",
  )));
  assert_hex(
    &aes128.mask(&decode_hex("d1b1c98dd7689fb8ec11d242b123dc9b")),
    "437b9aec36",
  );

  // FIPS 197 Appendix C.3 AES-256 encryption vector, truncated only after encryption.
  let aes256 = Aes256HeaderProtection::new(&Aes256HeaderProtectionKey::from_bytes(decode_hex(
    "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f",
  )));
  assert_hex(
    &aes256.mask(&decode_hex("00112233445566778899aabbccddeeff")),
    "8ea2b7ca51",
  );

  // RFC 9001 Appendix A.5 ChaCha20 short-header protection vector.
  let chacha20 = ChaCha20HeaderProtection::new(&ChaCha20HeaderProtectionKey::from_bytes(decode_hex(
    "25a282b9e82f06f21f488917a4fc8f1b73573685608597d0efcb076b0ab7a7a4",
  )));
  assert_hex(
    &chacha20.mask(&decode_hex("5e5cd55c41f69080575d7999c25a5bfb")),
    "aefefe7d03",
  );
}

fn assert_aes_siv_runtime_vector_and_failed_open_cleanup() {
  let mut key_bytes = [0u8; 32];
  for (index, byte) in key_bytes.iter_mut().enumerate() {
    *byte = u8::try_from(index).expect("key index fits in one byte");
  }
  let cipher = AesSivCmac256::new(&AesSivCmac256Key::from_bytes(key_bytes));
  let nonce = AesSivCmac256Nonce::try_from(&b"wasm nonce"[..]).expect("nonce is non-empty");
  let plaintext = b"wasm AES-SIV runtime vector input";
  let mut combined = [0u8; 49];
  cipher
    .seal(nonce, b"wasm associated data", plaintext, &mut combined)
    .expect("fixed output shape is exact");
  assert_hex(
    &combined,
    "c83dab6674aac8b6ba89d5d4e714eb988d9352d177d26e424465796ce9d4199aba1694731dbeab6c045dfebac553d266af",
  );

  let mut opened = [0u8; 33];
  cipher
    .open(nonce, b"wasm associated data", &combined, &mut opened)
    .expect("known ciphertext authenticates");
  assert_eq!(&opened, plaintext);

  combined[15] ^= 1;
  opened.fill(0xA5);
  assert!(
    cipher
      .open(nonce, b"wasm associated data", &combined, &mut opened)
      .is_err()
  );
  assert_eq!(opened, [0u8; 33]);
}

fn assert_ecdsa_portable_signing_roundtrips() {
  let message = b"rscrypto portable ECDSA signing on wasm32-wasip1";

  let p256 = EcdsaP256SecretKey::from_bytes([0x42; EcdsaP256SecretKey::LENGTH])
    .expect("fixed P-256 scalar is valid");
  let p256_signature = p256.try_sign(message).expect("portable P-256 signing must succeed");
  let p256_blinded = p256
    .try_sign_blinded(message, |blind| blind.fill(0xa6))
    .expect("portable blinded P-256 signing must succeed");
  assert_eq!(p256_signature, p256_blinded);
  p256
    .public_key()
    .verify(message, &p256_signature)
    .expect("portable P-256 signature must verify");

  let p384 = EcdsaP384SecretKey::from_bytes([0x24; EcdsaP384SecretKey::LENGTH])
    .expect("fixed P-384 scalar is valid");
  let p384_signature = p384.try_sign(message).expect("portable P-384 signing must succeed");
  let p384_blinded = p384
    .try_sign_blinded(message, |blind| blind.fill(0x5c))
    .expect("portable blinded P-384 signing must succeed");
  assert_eq!(p384_signature, p384_blinded);
  p384
    .public_key()
    .verify(message, &p384_signature)
    .expect("portable P-384 signature must verify");
}

#[cfg(target_feature = "simd128")]
fn assert_simd128_runtime_caps_are_detected() {
  assert!(rscrypto::platform::caps().has(rscrypto::platform::caps::wasm::SIMD128));
}

#[cfg(not(target_feature = "simd128"))]
fn assert_simd128_runtime_caps_are_detected() {}

fn main() {
  assert_core_hash_vectors_match_known_outputs();
  assert_streaming_hashes_match_oneshot_across_block_boundaries();
  assert_rsa_caller_random_signing_roundtrips();
  assert_websocket_accept_digest_matches_rfc_6455();
  assert_header_protection_vectors_match_known_outputs();
  assert_aes_siv_runtime_vector_and_failed_open_cleanup();
  assert_ecdsa_portable_signing_roundtrips();
  assert_simd128_runtime_caps_are_detected();
}
