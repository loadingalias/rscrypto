#![cfg(feature = "hmac")]

use rscrypto::{HmacSha384, HmacSha512, Mac};

fn decode_hex<const N: usize>(hex: &str) -> [u8; N] {
  assert_eq!(hex.len(), N * 2, "hex length must match output size");
  let mut out = [0u8; N];
  let bytes = hex.as_bytes();
  let mut i = 0usize;
  while i < N {
    let hi = char::from(bytes[i * 2]).to_digit(16).unwrap();
    let lo = char::from(bytes[i * 2 + 1]).to_digit(16).unwrap();
    out[i] = ((hi << 4) | lo) as u8;
    i += 1;
  }
  out
}

#[test]
fn hmac_sha384_rfc4231_vectors() {
  let cases: &[(&[u8], &[u8], &str)] = &[
    (
      &[0x0b; 20],
      b"Hi There",
      "afd03944d84895626b0825f4ab46907f15f9dadbe4101ec682aa034c7cebc59cfaea9ea9076ede7f4af152e8b2fa9cb6",
    ),
    (
      b"Jefe",
      b"what do ya want for nothing?",
      "af45d2e376484031617f78d2b58a6b1b9c7ef464f5a01b47e42ec3736322445e8e2240ca5e69e2c78b3239ecfab21649",
    ),
    (
      &[0xaa; 20],
      &[0xdd; 50],
      "88062608d3e6ad8a0aa2ace014c8a86f0aa635d947ac9febe83ef4e55966144b2a5ab39dc13814b94e3ab6e101a34f27",
    ),
    (
      &[
        0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11,
        0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19,
      ],
      &[0xcd; 50],
      "3e8a69b7783c25851933ab6290af6ca77a9981480850009cc5577c6e1f573b4e6801dd23c4a7d679ccf8a386c674cffb",
    ),
    (
      &[0xaa; 131],
      b"Test Using Larger Than Block-Size Key - Hash Key First",
      "4ece084485813e9088d2c63a041bc5b44f9ef1012a2b588f3cd11f05033ac4c60c2ef6ab4030fe8296248df163f44952",
    ),
    (
      &[0xaa; 131],
      b"This is a test using a larger than block-size key and a larger than block-size data. The key needs to be hashed before being used by the HMAC algorithm.",
      "6617178e941f020d351e2f254e8fd32c602420feb0b8fb9adccebb82461e99c5a678cc31e799176d3860e6110c46523e",
    ),
  ];

  for (i, (key, data, expected_hex)) in cases.iter().enumerate() {
    let expected = decode_hex::<48>(expected_hex);
    let actual = HmacSha384::mac(key, data);
    assert_eq!(actual, expected, "HMAC-SHA384 RFC 4231 vector {i} mismatch");
    assert!(HmacSha384::verify_tag(key, data, &expected).is_ok());
  }

  let truncated = HmacSha384::mac(&[0x0c; 20], b"Test With Truncation");
  assert_eq!(&truncated[..16], &decode_hex::<16>("3abf34c3503b2a23a46efc619baef897"));
}

#[test]
fn hmac_sha512_rfc4231_vectors() {
  let cases: &[(&[u8], &[u8], &str)] = &[
    (
      &[0x0b; 20],
      b"Hi There",
      "87aa7cdea5ef619d4ff0b4241a1d6cb02379f4e2ce4ec2787ad0b30545e17cdedaa833b7d6b8a702038b274eaea3f4e4be9d914eeb61f1702e696c203a126854",
    ),
    (
      b"Jefe",
      b"what do ya want for nothing?",
      "164b7a7bfcf819e2e395fbe73b56e0a387bd64222e831fd610270cd7ea2505549758bf75c05a994a6d034f65f8f0e6fdcaeab1a34d4a6b4b636e070a38bce737",
    ),
    (
      &[0xaa; 20],
      &[0xdd; 50],
      "fa73b0089d56a284efb0f0756c890be9b1b5dbdd8ee81a3655f83e33b2279d39bf3e848279a722c806b485a47e67c807b946a337bee8942674278859e13292fb",
    ),
    (
      &[
        0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11,
        0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19,
      ],
      &[0xcd; 50],
      "b0ba465637458c6990e5a8c5f61d4af7e576d97ff94b872de76f8050361ee3dba91ca5c11aa25eb4d679275cc5788063a5f19741120c4f2de2adebeb10a298dd",
    ),
    (
      &[0xaa; 131],
      b"Test Using Larger Than Block-Size Key - Hash Key First",
      "80b24263c7c1a3ebb71493c1dd7be8b49b46d1f41b4aeec1121b013783f8f3526b56d037e05f2598bd0fd2215d6a1e5295e64f73f63f0aec8b915a985d786598",
    ),
    (
      &[0xaa; 131],
      b"This is a test using a larger than block-size key and a larger than block-size data. The key needs to be hashed before being used by the HMAC algorithm.",
      "e37b6a775dc87dbaa4dfa9f96e5e3ffddebd71f8867289865df5a32d20cdc944b6022cac3c4982b10d5eeb55c3e4de15134676fb6de0446065c97440fa8c6a58",
    ),
  ];

  for (i, (key, data, expected_hex)) in cases.iter().enumerate() {
    let expected = decode_hex::<64>(expected_hex);
    let actual = HmacSha512::mac(key, data);
    assert_eq!(actual, expected, "HMAC-SHA512 RFC 4231 vector {i} mismatch");
    assert!(HmacSha512::verify_tag(key, data, &expected).is_ok());
  }

  let truncated = HmacSha512::mac(&[0x0c; 20], b"Test With Truncation");
  assert_eq!(&truncated[..16], &decode_hex::<16>("415fad6271580a531d4179bc891d87a6"));
}

#[test]
fn hmac_sha2_64bit_family_oneshot_matches_streaming() {
  let key = &[0x42u8; 64];

  for len in [
    0, 1, 32, 111, 112, 113, 127, 128, 129, 239, 240, 241, 255, 256, 257, 1024,
  ] {
    let data = vec![0xABu8; len];

    let oneshot384 = HmacSha384::mac(key, &data);
    let mut streaming384 = HmacSha384::new(key);
    streaming384.update(&data);
    assert_eq!(
      oneshot384,
      streaming384.finalize(),
      "HMAC-SHA384 oneshot != streaming at len {len}"
    );

    let oneshot512 = HmacSha512::mac(key, &data);
    let mut streaming512 = HmacSha512::new(key);
    streaming512.update(&data);
    assert_eq!(
      oneshot512,
      streaming512.finalize(),
      "HMAC-SHA512 oneshot != streaming at len {len}"
    );
  }
}
