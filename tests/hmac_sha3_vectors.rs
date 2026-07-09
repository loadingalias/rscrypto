#![cfg(feature = "hmac-sha3")]

use rscrypto::{
  HmacSha3_224, HmacSha3_224Tag, HmacSha3_256, HmacSha3_256Tag, HmacSha3_384, HmacSha3_384Tag, HmacSha3_512,
  HmacSha3_512Tag, Mac,
};
use sha3::Digest as _;

fn pattern(len: usize, mul: u8, add: u8) -> Vec<u8> {
  (0..len)
    .map(|i| {
      (i as u8)
        .wrapping_mul(mul)
        .wrapping_add(((i >> 3) as u8).wrapping_add(add))
    })
    .collect()
}

macro_rules! hmac_sha3_oracle {
  ($digest:ty, $key:expr, $data:expr, $tag_len:expr, $rate:expr) => {{
    let mut key_block = [0u8; $rate];
    if $key.len() > $rate {
      let digest = <$digest>::digest($key);
      key_block[..$tag_len].copy_from_slice(&digest);
    } else {
      key_block[..$key.len()].copy_from_slice($key);
    }

    let mut ipad = [0x36u8; $rate];
    let mut opad = [0x5cu8; $rate];
    for ((ipad_byte, opad_byte), key_byte) in ipad.iter_mut().zip(opad.iter_mut()).zip(key_block.iter().copied()) {
      *ipad_byte ^= key_byte;
      *opad_byte ^= key_byte;
    }

    let mut inner = <$digest>::new();
    inner.update(&ipad);
    inner.update($data);
    let inner_hash = inner.finalize();

    let mut outer = <$digest>::new();
    outer.update(&opad);
    outer.update(&inner_hash);
    let bytes = outer.finalize();

    let mut tag = [0u8; $tag_len];
    tag.copy_from_slice(&bytes);
    tag
  }};
}

macro_rules! assert_hmac_sha3 {
  ($name:literal, $ours:ty, $tag:ty, $oracle:ty, $tag_len:expr, $rate:expr) => {{
    for &(key_len, data_len, chunk_len) in &[
      (0usize, 0usize, 1usize),
      (1, 1, 1),
      ($tag_len, $rate - 1, 7),
      ($rate, $rate, 31),
      ($rate + 1, $rate + 1, 33),
      ($rate * 2 + 13, 1024, 97),
    ] {
      let key = pattern(key_len, 17, 3);
      let data = pattern(data_len, 29, 11);

      let expected = <$tag>::from_bytes(hmac_sha3_oracle!($oracle, &key, &data, $tag_len, $rate));

      assert_eq!(
        <$ours>::mac(&key, &data),
        expected,
        "{} one-shot mismatch key_len={} data_len={}",
        $name,
        key_len,
        data_len
      );
      assert!(<$ours>::verify_tag(&key, &data, &expected).is_ok());

      let mut streaming = <$ours>::new(&key);
      for chunk in data.chunks(chunk_len) {
        streaming.update(chunk);
      }
      assert_eq!(
        streaming.finalize(),
        expected,
        "{} streaming mismatch key_len={} data_len={} chunk_len={}",
        $name,
        key_len,
        data_len,
        chunk_len
      );

      streaming.reset();
      streaming.update(&data);
      assert_eq!(streaming.finalize(), expected, "{} reset mismatch", $name);

      let mut corrupted = expected.to_bytes();
      if !corrupted.is_empty() {
        corrupted[corrupted.len() / 2] ^= 0x80;
      }
      assert!(<$ours>::verify_tag(&key, &data, &<$tag>::from_bytes(corrupted)).is_err());
    }
  }};
}

#[test]
fn hmac_sha3_matches_rustcrypto() {
  assert_hmac_sha3!("HMAC-SHA3-224", HmacSha3_224, HmacSha3_224Tag, sha3::Sha3_224, 28, 144);
  assert_hmac_sha3!("HMAC-SHA3-256", HmacSha3_256, HmacSha3_256Tag, sha3::Sha3_256, 32, 136);
  assert_hmac_sha3!("HMAC-SHA3-384", HmacSha3_384, HmacSha3_384Tag, sha3::Sha3_384, 48, 104);
  assert_hmac_sha3!("HMAC-SHA3-512", HmacSha3_512, HmacSha3_512Tag, sha3::Sha3_512, 64, 72);
}
