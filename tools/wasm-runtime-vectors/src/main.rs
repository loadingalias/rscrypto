use rscrypto::{Blake2b512, Blake3, Digest, Sha256, Sha512};

fn hex_value(byte: u8) -> u8 {
  match byte {
    b'0'..=b'9' => byte - b'0',
    b'a'..=b'f' => byte - b'a' + 10,
    b'A'..=b'F' => byte - b'A' + 10,
    _ => panic!("invalid hex digit"),
  }
}

fn assert_hex(actual: &[u8], expected: &str) {
  assert_eq!(actual.len().strict_mul(2), expected.len());
  for (i, chunk) in expected.as_bytes().chunks_exact(2).enumerate() {
    let byte = (hex_value(chunk[0]) << 4) | hex_value(chunk[1]);
    assert_eq!(actual[i], byte, "hex mismatch at byte {i}");
  }
}

fn patterned_bytes(len: usize) -> Vec<u8> {
  (0..len)
    .map(|i| (i as u8).wrapping_mul(37).wrapping_add((i >> 8) as u8))
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

#[cfg(target_feature = "simd128")]
fn assert_simd128_runtime_caps_are_detected() {
  assert!(rscrypto::platform::caps().has(rscrypto::platform::caps::wasm::SIMD128));
}

#[cfg(not(target_feature = "simd128"))]
fn assert_simd128_runtime_caps_are_detected() {}

fn main() {
  assert_core_hash_vectors_match_known_outputs();
  assert_streaming_hashes_match_oneshot_across_block_boundaries();
  assert_simd128_runtime_caps_are_detected();
}
