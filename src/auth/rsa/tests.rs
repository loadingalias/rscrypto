pub(super) use super::expert::RsaPkcs1v15Decrypt;
use super::*;
pub(super) use alloc::format;
pub(super) use num_bigint::BigUint;
use proptest::prelude::*;
pub(super) use rsa::traits::{PrivateKeyParts, PublicKeyParts};
#[cfg(feature = "getrandom")]
pub(super) use serde_json::Value;

impl SecretI31 {
  #[inline]
  pub(super) fn as_slice(&self) -> &[u32] {
    &self.words
  }
}

impl RsaPrivateI31InverseScratch {
  pub(super) fn is_cleared(&self) -> bool {
    [&self.a, &self.b, &self.u, &self.v, &self.modulus]
      .iter()
      .all(|words| words.as_slice().iter().all(|&value| value == 0))
  }
}

#[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64", target_arch = "s390x")))]
impl RsaPrivateS62InverseScratch {
  pub(super) fn is_cleared(&self) -> bool {
    [&self.f, &self.g, &self.d, &self.e, &self.modulus]
      .iter()
      .all(|limbs| limbs.as_slice().iter().all(|&value| value == 0))
  }
}

pub(super) fn verify_pss_encoded<D>(message: &[u8], encoded: &[u8], em_bits: usize) -> Result<(), VerificationError>
where
  D: Digest,
{
  let mut db = vec![0u8; encoded.len()];
  let mut db_mask = vec![0u8; encoded.len()];
  pss::verify_pss_encoded_with_scratch::<D>(message, encoded, em_bits, D::OUTPUT_SIZE, &mut db, &mut db_mask)
}

pub(super) fn seed_fill(seed: &[u8]) -> impl FnMut(&mut [u8]) -> Result<(), RsaEncryptionError> {
  move |buffer| {
    buffer.copy_from_slice(seed);
    Ok(())
  }
}

pub(super) fn hex_to_vec(hex: &str) -> Vec<u8> {
  let (chunks, remainder) = hex.as_bytes().as_chunks::<2>();
  assert!(remainder.is_empty(), "hex input must contain complete byte pairs");
  let mut out = Vec::with_capacity(chunks.len());
  for &[hi, lo] in chunks {
    let hi = hex_value(hi).expect("hex fixtures must contain only ASCII hexadecimal digits");
    let lo = hex_value(lo).expect("hex fixtures must contain only ASCII hexadecimal digits");
    out.push((hi << 4) | lo);
  }
  out
}

const fn hex_value(byte: u8) -> Option<u8> {
  match byte {
    b'0'..=b'9' => Some(byte.strict_sub(b'0')),
    b'a'..=b'f' => Some(byte.strict_sub(b'a').strict_add(10)),
    b'A'..=b'F' => Some(byte.strict_sub(b'A').strict_add(10)),
    _ => None,
  }
}

#[cfg(feature = "getrandom")]
pub(super) fn json_field<'a>(value: &'a Value, name: &'static str) -> &'a str {
  value
    .get(name)
    .and_then(Value::as_str)
    .expect("the CAVP fixture field must be a string")
}

#[cfg(feature = "getrandom")]
pub(super) fn test_keygen_drbg(label: &'static [u8]) -> RsaKeygenDrbg {
  let mut seed = [0u8; RSA_KEYGEN_DRBG_ENTROPY_BYTES.strict_add(RSA_KEYGEN_DRBG_NONCE_BYTES)];
  for (index, byte) in seed.iter_mut().enumerate() {
    *byte = u8::try_from(index)
      .expect("the RSA keygen DRBG fixture seed length must fit in one byte")
      .wrapping_mul(17)
      .wrapping_add(0xa5);
  }
  RsaKeygenDrbg::new(&seed, label)
}

pub(super) fn rsa_miri_private_limb_widths() -> core::ops::RangeInclusive<usize> {
  let min_factor_bits = RsaPublicKeyPolicy::MODULUS_2048_TO_8192.min_modulus_bits.div_ceil(2);
  let max_factor_bits = RsaPublicKeyPolicy::MODULUS_2048_TO_8192.max_modulus_bits.div_ceil(2);
  let bits_per_limb = u64::BITS as usize;
  let min_limbs = min_factor_bits.div_ceil(bits_per_limb);
  let max_limbs = max_factor_bits.div_ceil(bits_per_limb);
  min_limbs..=max_limbs
}

pub(super) fn rsa_miri_table_limb(limbs: usize, window: usize, limb: usize) -> u64 {
  0xd6e8_feb8_6659_fd93
    ^ (limbs as u64).rotate_left(11)
    ^ (window as u64).rotate_left(29)
    ^ (limb as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
}

pub(super) fn rsa_miri_fill_exponent_scratch(scratch: &mut RsaPrivateExponentScratch, value: u64) {
  scratch.t.as_mut_slice().fill(value);
  scratch.representative.as_mut_slice().fill(value);
  scratch.one.as_mut_slice().fill(value);
  scratch.base.as_mut_slice().fill(value);
  scratch.acc.as_mut_slice().fill(value);
  scratch.squared.as_mut_slice().fill(value);
  scratch.multiplied.as_mut_slice().fill(value);
  scratch.selected.as_mut_slice().fill(value);
  scratch.reduced.as_mut_slice().fill(value);
  scratch.table.as_mut_slice().fill(value);
}

pub(super) fn rsa_miri_assert_exponent_scratch_zeroed(scratch: &RsaPrivateExponentScratch) {
  for (name, limbs) in [
    ("t", scratch.t.as_slice()),
    ("representative", scratch.representative.as_slice()),
    ("one", scratch.one.as_slice()),
    ("base", scratch.base.as_slice()),
    ("acc", scratch.acc.as_slice()),
    ("squared", scratch.squared.as_slice()),
    ("multiplied", scratch.multiplied.as_slice()),
    ("selected", scratch.selected.as_slice()),
    ("reduced", scratch.reduced.as_slice()),
    ("table", scratch.table.as_slice()),
  ] {
    assert!(limbs.iter().all(|&limb| limb == 0), "{name} was not cleared");
  }
}

pub(super) fn integer_unsigned(value: &[u8]) -> Vec<u8> {
  let first_nonzero = value.iter().position(|&byte| byte != 0);
  let value = first_nonzero.map_or(&[0u8][..], |index| &value[index..]);
  let mut encoded = Vec::with_capacity(value.len().strict_add(usize::from(value[0] & 0x80 != 0)));
  if value[0] & 0x80 != 0 {
    encoded.push(0);
  }
  encoded.extend_from_slice(value);
  tlv(TAG_INTEGER, &encoded)
}

pub(super) fn integer_der_value(value: &[u8]) -> Vec<u8> {
  tlv(TAG_INTEGER, value)
}

pub(super) fn test_pkcs1_public_key(modulus: &[u8], exponent: &[u8]) -> Vec<u8> {
  let modulus = integer_unsigned(modulus);
  let exponent = integer_unsigned(exponent);
  let mut body = Vec::with_capacity(modulus.len().strict_add(exponent.len()));
  body.extend_from_slice(&modulus);
  body.extend_from_slice(&exponent);
  tlv(TAG_SEQUENCE, &body)
}

pub(super) fn algorithm_identifier(algorithm_oid: &[u8], params: Option<&[u8]>) -> Vec<u8> {
  let mut body = Vec::new();
  body.extend_from_slice(&tlv(TAG_OBJECT_IDENTIFIER, algorithm_oid));
  if let Some(params) = params {
    body.extend_from_slice(params);
  }
  tlv(TAG_SEQUENCE, &body)
}

pub(super) fn null() -> Vec<u8> {
  tlv(TAG_NULL, &[])
}

pub(super) fn context_constructed(index: u8, value: &[u8]) -> Vec<u8> {
  tlv(0xa0 | index, value)
}

fn x509_hash_algorithm(profile: RsaPssProfile) -> Vec<u8> {
  let oid = match profile {
    RsaPssProfile::Sha256 => ID_SHA256_OID,
    RsaPssProfile::Sha384 => ID_SHA384_OID,
    RsaPssProfile::Sha512 => ID_SHA512_OID,
  };
  algorithm_identifier(oid, Some(&null()))
}

fn x509_mgf1_algorithm(profile: RsaPssProfile) -> Vec<u8> {
  let hash = x509_hash_algorithm(profile);
  algorithm_identifier(ID_MGF1_OID, Some(&hash))
}

pub(super) fn x509_pss_algorithm(profile: RsaPssProfile, salt_len: usize) -> Vec<u8> {
  let mut params = Vec::new();
  params.extend_from_slice(&context_constructed(0, &x509_hash_algorithm(profile)));
  params.extend_from_slice(&context_constructed(1, &x509_mgf1_algorithm(profile)));
  params.extend_from_slice(&context_constructed(
    2,
    &integer_unsigned(
      &u64::try_from(salt_len)
        .expect("an RSA-PSS test salt length must fit in an unsigned DER integer")
        .to_be_bytes(),
    ),
  ));
  algorithm_identifier(ID_RSASSA_PSS_OID, Some(&tlv(TAG_SEQUENCE, &params)))
}

pub(super) fn rsa_private_modulus() -> Vec<u8> {
  hex_to_vec(
    "\
d397b84d98a4c26138ed1b695a8106ead91d553bf06041b62d3fdc50a041e222b8f4529689c1b82c5e71554f5d\
d69fa2f4b6158cf0dbeb57811a0fc327e1f28e74fe74d3bc166c1eabdc1b8b57b934ca8be5b00b4f29975bcc\
99acaf415b59bb28a6782bb41a2c3c2976b3c18dbadef62f00c6bb226640095096c0cc60d22fe7ef987d75c\
6a81b10d96bf292028af110dc7cc1bbc43d22adab379a0cd5d8078cc780ff5cd6209dea34c922cf784f7717e\
428d75b5aec8ff30e5f0141510766e2e0ab8d473c84e8710b2b98227c3db095337ad3452f19e2b9bfbccdd8\
148abf6776fa552775e6e75956e45229ae5a9c46949bab1e622f0e48f56524a84ed3483b",
  )
}

pub(super) fn rsa_private_exponent() -> Vec<u8> {
  hex_to_vec(
    "\
c4e70c689162c94c660828191b52b4d8392115df486a9adbe831e458d73958320dc1b755456e93701e9702d76\
fb0b92f90e01d1fe248153281fe79aa9763a92fae69d8d7ecd144de29fa135bd14f9573e349e45031e3b76982\
f583003826c552e89a397c1a06bd2163488630d92e8c2bb643d7abef700da95d685c941489a46f54b5316f62\
b5d2c3a7f1bbd134cb37353a44683fdc9d95d36458de22f6c44057fe74a0a436c4308f73f4da42f35c47ac1\
6a7138d483afc91e41dc3a1127382e0c0f5119b0221b4fc639d6b9c38177a6de9b526ebd88c38d7982c07f98\
a0efd877d508aae275b946915c02e2e1106d175d74ec6777f5e80d12c053d9c7be1e341",
  )
}

pub(super) fn rsa_private_prime_p() -> Vec<u8> {
  hex_to_vec(
    "\
f827bbf3a41877c7cc59aebf42ed4b29c32defcb8ed96863d5b090a05a8930dd624a21c9dcf9838568fdfa0d\
f65b8462a5f2ac913d6c56f975532bd8e78fb07bd405ca99a484bcf59f019bbddcb3933f2bce706300b4f7b\
110120c5df9018159067c35da3061a56c8635a52b54273b31271b4311f0795df6021e6355e1a42e61",
  )
}

pub(super) fn rsa_private_prime_q() -> Vec<u8> {
  hex_to_vec(
    "\
da4817ce0089dd36f2ade6a3ff410c73ec34bf1b4f6bda38431bfede11cef1f7f6efa70e5f8063a3b1f6e172\
96ffb15feefa0912a0325b8d1fd65a559e717b5b961ec345072e0ec5203d03441d29af4d64054a04507410cf\
1da78e7b6119d909ec66e6ad625bf995b279a4b3c5be7d895cd7c5b9c4c497fde730916fcdb4e41b",
  )
}

pub(super) fn rsa_private_exponent_p() -> Vec<u8> {
  hex_to_vec(
    "\
1da6e9cf80212856e87522eb59bcef094b7836ba1514a7639e8a1d8dfba37f0245176498315e6337d2c6de554\
2c5c6b8dee973735b6a91adf735fbfc4c1720587b8a419e40495826e55c14d70803312a103af7b4ecc5b2ff26\
5371c4dcd730348a10d7827ddb7d1fcd9da561db09610a4b88f767b25b5e3de21ced73baa59aa1",
  )
}

pub(super) fn rsa_private_exponent_q() -> Vec<u8> {
  hex_to_vec(
    "\
d737a7c8e43d0a10c85bf0011886a16996a6371b0d46b0c5325de3003f9cc47491539f6a0b7d82407f12851c\
bf86e1f34da3d7d8367d104967efa7e7ad2e04cbbb8b1f4aeb165d57bd3e8afed8a62602ef304bd74f1ff106\
d51d44dd9f52a5ed23da1d6d2c82b4e6052fecd5978e0726ad94cd8e295510eb35cc6c49491026ab",
  )
}

pub(super) fn rsa_private_coefficient() -> Vec<u8> {
  hex_to_vec(
    "\
5268d7cf073479aebb2d2ed4dd66b8c89915b52d141e0c4932f56b0c0ed0936141894ec4d27d53bc86453cd8\
ca5b455045218c7e196209c1c651702ece090a15e3cbcc265971300023a86fe9d34ad527e9ef03b7adfe736e\
0680747abfd49839b82f2ffdec43bd0343ca30e13961b32af6cdeddd195672c76b53b76fc3ea76f8",
  )
}

pub(super) fn test_pkcs1_private_key() -> Vec<u8> {
  test_pkcs1_private_key_with_public_exponent(&[0x01, 0x00, 0x01])
}

pub(super) fn prevalidated_test_private_key() -> RsaPrivateKey {
  let modulus = rsa_private_modulus();
  let private_exponent = rsa_private_exponent();
  let prime_p = rsa_private_prime_p();
  let prime_q = rsa_private_prime_q();
  let exponent_p = rsa_private_exponent_p();
  let exponent_q = rsa_private_exponent_q();
  let coefficient = rsa_private_coefficient();
  let components = RsaPrivateKeyDerComponents {
    modulus: &modulus,
    public_exponent: RsaPublicExponent(65_537),
    private_exponent: &private_exponent,
    prime_p: &prime_p,
    prime_q: &prime_q,
    exponent_p: &exponent_p,
    exponent_q: &exponent_q,
    coefficient: &coefficient,
  };

  RsaPrivateKey {
    components: private_key_components_from_prevalidated_parts(&components, unsigned_be_bit_len(&modulus))
      .expect("the prevalidated RSA private-key fixture must satisfy component invariants"),
  }
}

pub(super) fn test_pkcs1_private_key_with_public_exponent(public_exponent: &[u8]) -> Vec<u8> {
  test_pkcs1_private_key_with_crt(
    public_exponent,
    &rsa_private_exponent_p(),
    &rsa_private_exponent_q(),
    &rsa_private_coefficient(),
  )
}

pub(super) fn test_pkcs1_private_key_with_crt(
  public_exponent: &[u8],
  exponent_p: &[u8],
  exponent_q: &[u8],
  coefficient: &[u8],
) -> Vec<u8> {
  test_pkcs1_private_key_with_components(
    public_exponent,
    &rsa_private_exponent(),
    &rsa_private_prime_p(),
    &rsa_private_prime_q(),
    exponent_p,
    exponent_q,
    coefficient,
  )
}

pub(super) fn test_pkcs1_private_key_with_components(
  public_exponent: &[u8],
  private_exponent: &[u8],
  prime_p: &[u8],
  prime_q: &[u8],
  exponent_p: &[u8],
  exponent_q: &[u8],
  coefficient: &[u8],
) -> Vec<u8> {
  let fields = [
    integer_unsigned(&[0]),
    integer_unsigned(&rsa_private_modulus()),
    integer_unsigned(public_exponent),
    integer_unsigned(private_exponent),
    integer_unsigned(prime_p),
    integer_unsigned(prime_q),
    integer_unsigned(exponent_p),
    integer_unsigned(exponent_q),
    integer_unsigned(coefficient),
  ];
  let len = fields.iter().map(Vec::len).sum();
  let mut body = Vec::with_capacity(len);
  for field in fields {
    body.extend_from_slice(&field);
  }
  tlv(TAG_SEQUENCE, &body)
}

pub(super) fn test_pkcs8_private_key(pkcs1: &[u8], algorithm: &[u8]) -> Vec<u8> {
  let mut body = Vec::new();
  body.extend_from_slice(&integer_unsigned(&[0]));
  body.extend_from_slice(algorithm);
  body.extend_from_slice(&tlv(TAG_OCTET_STRING, pkcs1));
  tlv(TAG_SEQUENCE, &body)
}

pub(super) fn masked_oaep_sha256_from_decoded_db(seed: &[u8; Sha256::OUTPUT_SIZE], decoded_db: &[u8]) -> Vec<u8> {
  let mut masked_db = decoded_db.to_vec();
  let mut db_mask = vec![0u8; masked_db.len()];
  mgf1::<Sha256>(seed, &mut db_mask);
  for (byte, mask) in masked_db.iter_mut().zip(db_mask.iter().copied()) {
    *byte ^= mask;
  }

  let mut masked_seed = seed.to_vec();
  let mut seed_mask = vec![0u8; Sha256::OUTPUT_SIZE];
  mgf1::<Sha256>(&masked_db, &mut seed_mask);
  for (byte, mask) in masked_seed.iter_mut().zip(seed_mask.iter().copied()) {
    *byte ^= mask;
  }

  let mut encoded = Vec::with_capacity(1usize.strict_add(masked_seed.len()).strict_add(masked_db.len()));
  encoded.push(0);
  encoded.extend_from_slice(&masked_seed);
  encoded.extend_from_slice(&masked_db);
  encoded
}

pub(super) fn decoded_oaep_sha256_db(label: &[u8], message: &[u8], ps_len: usize) -> Vec<u8> {
  let label_hash = Sha256::digest(label);
  let mut db = Vec::with_capacity(
    Sha256::OUTPUT_SIZE
      .strict_add(ps_len)
      .strict_add(1)
      .strict_add(message.len()),
  );
  db.extend_from_slice(label_hash.as_ref());
  db.resize(Sha256::OUTPUT_SIZE.strict_add(ps_len), 0);
  db.push(1);
  db.extend_from_slice(message);
  db
}

pub(super) fn x509_certificate_fixture_public_key() -> Vec<u8> {
  hex_to_vec(
    "\
30820122300d06092a864886f70d01010105000382010f003082010a0282010100a246ccf6bd59720287837151de9fa5\
5d4a811e456643f7fd0ced5a9ffa8fe52a89d52a8f6bd96246c9f0d23cd4f215609bfd0fd09dfcf13305440cae6e1b9a\
3c48e8e360438ca9993c1cd8ec03363cc3d79edbc4df7764c7f8ddb75f1148037847b356d2697f7d0158072a2e4f38f9\
40c8db08b70305dedb6fe97aeb530dccc009274f7864442f6f02cf6191b5a32268234bcbd7827bf3e570206c0cddf147\
df5169ceda6883b2169768878fd5b107a092ab7482d8ba7f46364b566aaa72153068b6a0174f2f5e0e5f9bcd0213dd4e\
8689d56ffa0be918a16fffcbe4830157eb8535c1a2a50636f8fc8a57f9ae0488b91159456ca94d7e64a1286babad3e92\
f70203010001",
  )
}

pub(super) fn x509_pkcs1v15_certificate_fixture() -> Vec<u8> {
  hex_to_vec(
    "\
3082031b30820203a00302010202144abda3eea77ef52d888f7ab507cd9016cacc900f300d06092a864886f70d01010b\
0500301d311b301906035504030c12727363727970746f2d7273612d706b637331301e170d3236303532323232333332\
305a170d3236303632313232333332305a301d311b301906035504030c12727363727970746f2d7273612d706b637331\
30820122300d06092a864886f70d01010105000382010f003082010a0282010100a246ccf6bd59720287837151de9fa5\
5d4a811e456643f7fd0ced5a9ffa8fe52a89d52a8f6bd96246c9f0d23cd4f215609bfd0fd09dfcf13305440cae6e1b9a\
3c48e8e360438ca9993c1cd8ec03363cc3d79edbc4df7764c7f8ddb75f1148037847b356d2697f7d0158072a2e4f38f9\
40c8db08b70305dedb6fe97aeb530dccc009274f7864442f6f02cf6191b5a32268234bcbd7827bf3e570206c0cddf147\
df5169ceda6883b2169768878fd5b107a092ab7482d8ba7f46364b566aaa72153068b6a0174f2f5e0e5f9bcd0213dd4e\
8689d56ffa0be918a16fffcbe4830157eb8535c1a2a50636f8fc8a57f9ae0488b91159456ca94d7e64a1286babad3e92\
f70203010001a3533051301d0603551d0e04160414fd0e576ce3f05b08884ad67ef3e8b4d39039c65d301f0603551d23\
041830168014fd0e576ce3f05b08884ad67ef3e8b4d39039c65d300f0603551d130101ff040530030101ff300d06092a\
864886f70d01010b050003820101008ed399c2f78e0325f9ec4ae7cd0b978b5cd03d30af8fb61a91925213b6388adb00\
0a59f657dc6a3e983706ff8053b0a4049d5f532c00640e2b67aac3ce30518ad7c4b762078c3816c6f325e9a841920b80\
6d0c5ac16450e8b385c6e50434bf1dc575816a263696b9de661a8bf2ae143853951745d25fa1bbc49d66270197b572aa\
052b0d23d35243cd9087fdc2d79bd2b1d27ad2c67fc7c1960b370b77edc038aee1ee653bec34782bbca87c5aefb3dc92\
5eba1d3c83019a37696d52ea1f14366a13ec6c3f74bda1941c745771bd33ea117f81e6c0968b9692d4dd349743acc149\
73eb5c2a1fcd85691f6dcbd6937b03fe525cbe51610a1f5be86a189de2d2d3",
  )
}

pub(super) fn tlv(tag: u8, value: &[u8]) -> Vec<u8> {
  let encoded_len = der_len(value.len());
  let capacity = 1usize.strict_add(encoded_len.len()).strict_add(value.len());
  let mut out = Vec::with_capacity(capacity);
  out.push(tag);
  out.extend_from_slice(&encoded_len);
  out.extend_from_slice(value);
  out
}

pub(super) fn tlv_with_noncanonical_short_len(tag: u8, value: &[u8]) -> Vec<u8> {
  let mut out = Vec::with_capacity(3usize.strict_add(value.len()));
  out.push(tag);
  out.push(0x81);
  out.push(u8::try_from(value.len()).expect("a noncanonical short DER fixture length must fit in one byte"));
  out.extend_from_slice(value);
  out
}

pub(super) fn tlv_with_leading_zero_long_len(der: &[u8]) -> Vec<u8> {
  let tag = der[0];
  let len_len = usize::from(der[1] & 0x7f);
  assert_ne!(der[1] & 0x80, 0);
  assert_ne!(len_len, 0);

  let mut out = Vec::with_capacity(der.len().strict_add(1));
  out.push(tag);
  out.push(0x80 | u8::try_from(len_len.strict_add(1)).expect("a DER length-of-length must fit in one byte"));
  out.push(0);
  out.extend_from_slice(&der[2..]);
  out
}

fn der_len(len: usize) -> Vec<u8> {
  if len < 128 {
    return vec![u8::try_from(len).expect("a short DER length must fit in one byte")];
  }

  let bytes = len.to_be_bytes();
  let first_nonzero = bytes
    .iter()
    .position(|&byte| byte != 0)
    .expect("a DER length of at least 128 must contain a nonzero byte");
  let len_bytes = &bytes[first_nonzero..];
  let mut out = Vec::with_capacity(1usize.strict_add(len_bytes.len()));
  out.push(0x80 | u8::try_from(len_bytes.len()).expect("a usize DER length prefix must fit in one byte"));
  out.extend_from_slice(len_bytes);
  out
}

pub(super) fn pkcs1v15_encoded_sha256(message: &[u8], em_len: usize) -> Vec<u8> {
  let digest = Sha256::digest(message);
  let digest_info_len = SHA256_DIGEST_INFO_PREFIX.len().strict_add(Sha256::OUTPUT_SIZE);
  let ps_len = em_len.strict_sub(digest_info_len).strict_sub(3);
  let mut encoded = Vec::with_capacity(em_len);
  encoded.push(0);
  encoded.push(1);
  encoded.extend(core::iter::repeat_n(0xff, ps_len));
  encoded.push(0);
  encoded.extend_from_slice(SHA256_DIGEST_INFO_PREFIX);
  encoded.extend_from_slice(&digest);
  encoded
}

pub(super) fn pss_encoded_sha256(message: &[u8], em_bits: usize, salt_len: usize) -> Vec<u8> {
  let em_len = em_bits.div_ceil(8);
  let h_len = Sha256::OUTPUT_SIZE;
  let db_len = em_len.strict_sub(h_len).strict_sub(1);
  let ps_len = db_len.strict_sub(salt_len).strict_sub(1);

  let mut salt = Vec::with_capacity(salt_len);
  for index in 0..salt_len {
    salt.push(
      u8::try_from(index)
        .expect("the RSA-PSS test salt length must fit in one byte")
        .wrapping_mul(17)
        .wrapping_add(0xa5),
    );
  }

  let m_hash = Sha256::digest(message);
  let mut h = Sha256::new();
  h.update(&[0u8; 8]);
  h.update(&m_hash);
  h.update(&salt);
  let h = h.finalize();

  let mut db = Vec::with_capacity(db_len);
  db.extend(core::iter::repeat_n(0, ps_len));
  db.push(1);
  db.extend_from_slice(&salt);

  let mut db_mask = vec![0u8; db_len];
  mgf1::<Sha256>(&h, &mut db_mask);
  for (byte, mask) in db.iter_mut().zip(db_mask.iter().copied()) {
    *byte ^= mask;
  }

  let unused_bits = em_len.strict_mul(8).strict_sub(em_bits);
  if unused_bits > 0
    && let Some(first) = db.first_mut()
  {
    *first &= 0xffu8 >> unused_bits;
  }

  let mut encoded = Vec::with_capacity(em_len);
  encoded.extend_from_slice(&db);
  encoded.extend_from_slice(&h);
  encoded.push(0xbc);
  encoded
}

pub(super) fn factor_two_and_inverse(modulus: &[u8]) -> (Vec<u8>, Vec<u8>) {
  let mut factor = vec![0u8; modulus.len()];
  factor[modulus.len().strict_sub(1)] = 2;

  let mut inverse = vec![0u8; modulus.len()];
  let mut carry = 0u8;
  for (dst, &byte) in inverse.iter_mut().zip(modulus) {
    *dst = (byte >> 1) | carry;
    carry = (byte & 1) << 7;
  }
  for byte in inverse.iter_mut().rev() {
    let (sum, overflow) = byte.overflowing_add(1);
    *byte = sum;
    if !overflow {
      break;
    }
  }

  (factor, inverse)
}

pub(super) fn rsa_private_key_parts<'a>(
  modulus: &'a [u8],
  private_exponent: &'a [u8],
  prime_p: &'a [u8],
  prime_q: &'a [u8],
  exponent_p: &'a [u8],
  exponent_q: &'a [u8],
  coefficient: &'a [u8],
) -> RsaPrivateKeyParts<'a> {
  RsaPrivateKeyParts {
    modulus,
    public_exponent: 65_537,
    private_exponent,
    prime_p,
    prime_q,
    exponent_p,
    exponent_q,
    coefficient,
  }
}

pub(super) fn wrong_width_private_scratch(key: &RsaPrivateKey) -> RsaPrivateScratch {
  let wrong_len = key.signature_len().strict_sub(1);
  RsaPrivateScratch {
    encoded: SecretBigEndianBuffer::zeroed(wrong_len),
    salt: SecretBigEndianBuffer::zeroed(wrong_len),
    blinding_factor: SecretBigEndianBuffer::zeroed(wrong_len),
    blinding_inverse: SecretBigEndianBuffer::zeroed(wrong_len),
    inverse_mask: SecretBigEndianBuffer::zeroed(wrong_len),
    blinding_power: SecretBigEndianBuffer::zeroed(wrong_len),
    blinded: SecretBigEndianBuffer::zeroed(wrong_len),
    blinded_private_result: SecretBigEndianBuffer::zeroed(wrong_len),
    checked: SecretBigEndianBuffer::zeroed(wrong_len),
    one: SecretBigEndianBuffer::zeroed(wrong_len),
    public_scratch: key.public_key().public_scratch(),
    mul_scratch: RsaPrivateMulScratch::new(key.components.public.modulus.limbs.len()),
    inverse_scratch: RsaPrivateInverseScratch::new(key.components.public.modulus.bits),
    exponent_scratch: RsaPrivateExponentScratch::new(key.components.public.modulus.limbs.len()),
  }
}

pub(super) fn assert_private_scratch_is_clear(scratch: &RsaPrivateScratch) {
  for bytes in [
    scratch.encoded.as_slice(),
    scratch.salt.as_slice(),
    scratch.blinding_factor.as_slice(),
    scratch.blinding_inverse.as_slice(),
    scratch.blinding_power.as_slice(),
    scratch.blinded.as_slice(),
    scratch.blinded_private_result.as_slice(),
    scratch.checked.as_slice(),
    scratch.one.as_slice(),
    &scratch.public_scratch.bytes,
  ] {
    assert!(bytes.iter().all(|&byte| byte == 0));
  }
  for words in [
    &scratch.public_scratch.limbs[..],
    scratch.mul_scratch.t.as_slice(),
    scratch.mul_scratch.left_limbs.as_slice(),
    scratch.mul_scratch.right_limbs.as_slice(),
    scratch.mul_scratch.left_mont.as_slice(),
    scratch.mul_scratch.right_mont.as_slice(),
    scratch.mul_scratch.product_mont.as_slice(),
    scratch.mul_scratch.product.as_slice(),
    scratch.exponent_scratch.t.as_slice(),
    scratch.exponent_scratch.representative.as_slice(),
    scratch.exponent_scratch.one.as_slice(),
    scratch.exponent_scratch.base.as_slice(),
    scratch.exponent_scratch.acc.as_slice(),
    scratch.exponent_scratch.squared.as_slice(),
    scratch.exponent_scratch.multiplied.as_slice(),
    scratch.exponent_scratch.selected.as_slice(),
    scratch.exponent_scratch.reduced.as_slice(),
    scratch.exponent_scratch.table.as_slice(),
  ] {
    assert!(words.iter().all(|&word| word == 0));
  }
  assert!(scratch.inverse_scratch.is_cleared());
}

/// The minimal big-endian bytes of a RustCrypto integer.
pub(super) fn rustcrypto_be(value: &rsa::BoxedUint) -> Vec<u8> {
  value.to_be_bytes_trimmed_vartime().into_vec()
}

/// Decode `encoded` against the alternative message `0x80, 0x81, ...` with
/// `alternative_len`; return the reported length and the written window.
pub(super) fn implicit_rejection_decode(encoded: &[u8], alternative_len: usize) -> (usize, Vec<u8>) {
  let mut alternative: Vec<u8> = (0..encoded.len())
    .map(|index| 0x80u8.wrapping_add(u8::try_from(index).expect("test block fits one octet")))
    .collect();
  let mut out = vec![0xa5u8; encoded.len()];
  let len =
    implicit_rejection::decode_pkcs1v15_implicit_rejection(encoded, &mut alternative, alternative_len, &mut out)
      .expect("a modulus-sized encoded message must decode");
  assert!(
    alternative.iter().all(|&byte| byte == 0),
    "the alternative message must be cleared"
  );
  let window = encoded.len().strict_sub(RSAES_PKCS1V15_OVERHEAD);
  assert!(
    out[window..].iter().all(|&byte| byte == 0xa5),
    "bytes after the k - 11 window must stay untouched"
  );
  out.truncate(window);
  (len, out)
}

/// Run one decryption into `out` prefilled with a canary. Pair the result
/// with whether `out` was zeroed on failure.
pub(super) fn decrypt_into_canary(
  out: &mut [u8],
  decrypt: impl FnOnce(&mut [u8]) -> Result<usize, RsaPrivateOpError>,
) -> (Result<usize, RsaPrivateOpError>, bool) {
  out.fill(0xa5);
  let result = decrypt(out);
  let cleared_on_error = result.is_ok() || is_zero_unsigned_be(out);
  (result, cleared_on_error)
}

#[cfg(all(
  any(
    all(target_arch = "aarch64", any(target_os = "linux", target_os = "macos")),
    all(
      target_arch = "x86_64",
      any(feature = "std", all(target_feature = "bmi2", target_feature = "adx"))
    )
  ),
  not(feature = "portable-only"),
  not(miri)
))]
pub(super) fn rsa_montgomery_test_modulus(words: usize) -> RsaPublicModulus {
  let mut bytes = vec![0u8; words.strict_mul(8)];
  for (i, byte) in bytes.iter_mut().enumerate() {
    *byte = 0xa5u8
      .wrapping_add(i.to_le_bytes()[0].wrapping_mul(0x3d))
      .wrapping_add(words.to_le_bytes()[0].wrapping_mul(0x17));
  }
  bytes[0] = 0xff;
  *bytes.last_mut().expect("RSA Montgomery test width must be nonzero") |= 1;
  RsaPublicModulus::new(&bytes, words.strict_mul(64))
}

#[cfg(all(
  any(
    all(target_arch = "aarch64", any(target_os = "linux", target_os = "macos")),
    all(
      target_arch = "x86_64",
      any(feature = "std", all(target_feature = "bmi2", target_feature = "adx"))
    )
  ),
  not(feature = "portable-only"),
  not(miri)
))]
pub(super) fn rsa_montgomery_test_limbs(words: usize, seed: u64) -> Vec<u64> {
  let mut state = seed ^ (words as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
  let mut limbs = vec![0u64; words];
  for limb in &mut limbs {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    *limb = state;
  }
  *limbs.last_mut().expect("RSA Montgomery test width must be nonzero") &= 0x7fff_ffff_ffff_ffff;
  limbs[0] |= 1;
  limbs
}

#[cfg(all(
  any(
    all(target_arch = "aarch64", any(target_os = "linux", target_os = "macos")),
    all(
      target_arch = "x86_64",
      any(feature = "std", all(target_feature = "bmi2", target_feature = "adx"))
    )
  ),
  not(feature = "portable-only"),
  not(miri)
))]
pub(super) fn assert_limbs_match_as_bytes(label: &str, words: usize, asm: &[u64], portable: &[u64]) {
  let mut asm_bytes = vec![0u8; words.strict_mul(8)];
  let mut portable_bytes = vec![0u8; words.strict_mul(8)];
  limbs_to_be(asm, &mut asm_bytes);
  limbs_to_be(portable, &mut portable_bytes);
  assert_eq!(asm_bytes, portable_bytes, "{label} mismatch for {words} limbs");
}

#[cfg(all(
  target_arch = "x86_64",
  any(feature = "std", all(target_feature = "bmi2", target_feature = "adx")),
  not(feature = "portable-only"),
  not(miri)
))]
pub(super) fn public_e65537_cios_portable(
  out: &mut [u64],
  input: &[u64],
  r2: &[u64],
  acc: &mut [u64],
  modulus: &RsaPublicModulus,
  t: &mut [u64],
) {
  let mut tmp = vec![0u64; input.len()];

  mont_mul_cios_portable(out, input, r2, modulus, t);
  copy_limbs(acc, out);
  for _ in 0..16 {
    copy_limbs(&mut tmp, acc);
    mont_mul_cios_portable(acc, &tmp, &tmp, modulus, t);
  }
  copy_limbs(&mut tmp, acc);
  mont_mul_cios_portable(acc, &tmp, out, modulus, t);
  mont_reduce_cios_portable(out, acc, modulus, t);
}

pub(super) fn reference_double_mod_in_place(value: &mut [u64], modulus: &[u64]) {
  let mut carry = 0u64;
  for limb in value.iter_mut() {
    let next = *limb >> 63;
    *limb = (*limb << 1) | carry;
    carry = next;
  }
  if carry != 0 {
    let _ = sub_modulus_in_place(value, modulus);
  } else if cmp_limbs(value, modulus) != core::cmp::Ordering::Less {
    let borrow = sub_modulus_in_place(value, modulus);
    debug_assert_eq!(borrow, 0);
  }
}

proptest! {
  #[test]
  fn representative_limb_conversion_round_trips(bytes in proptest::collection::vec(any::<u8>(), 0..600)) {
    let limbs = limbs_from_be(&bytes);
    let mut encoded = vec![0u8; bytes.len()];
    limbs_to_be(&limbs, &mut encoded);

    prop_assert_eq!(encoded, bytes);
  }

  #[test]
  fn representative_range_comparison_matches_big_endian_byte_order(
    mut left in proptest::collection::vec(any::<u8>(), 1..300),
    mut right in proptest::collection::vec(any::<u8>(), 1..300),
  ) {
    let len = core::cmp::max(left.len(), right.len());
    let mut left_padded = vec![0u8; len];
    let mut right_padded = vec![0u8; len];
    let left_offset = len.strict_sub(left.len());
    let right_offset = len.strict_sub(right.len());
    left_padded[left_offset..].copy_from_slice(&left);
    right_padded[right_offset..].copy_from_slice(&right);

    let left_limbs = limbs_from_be(&left_padded);
    let right_limbs = limbs_from_be(&right_padded);

    prop_assert_eq!(cmp_limbs(&left_limbs, &right_limbs), left_padded.cmp(&right_padded));
    prop_assert_eq!(ct_limbs_lt(&left_limbs, &right_limbs), left_padded < right_padded);

    left.clear();
    right.clear();
  }

  #[test]
  fn private_mod_subtraction_matches_u32_arithmetic(
    modulus in 1u32..=u32::MAX,
    left in any::<u32>(),
    right in any::<u32>(),
  ) {
    let left = left.rem_euclid(modulus);
    let right = right.rem_euclid(modulus);
    let expected = u32::try_from(
      u64::from(left)
        .strict_add(u64::from(modulus))
        .strict_sub(u64::from(right))
        .rem_euclid(u64::from(modulus)),
    )
    .expect("reduced u32 modular difference must fit u32");
    let mut out = [0u8; 4];

    private_sub_mod_unsigned_be_to_fixed(
      &left.to_be_bytes(),
      &right.to_be_bytes(),
      &modulus.to_be_bytes(),
      &mut out,
    )
    .expect("valid u32 modular subtraction inputs must succeed");

    prop_assert_eq!(out, expected.to_be_bytes());
  }

  #[test]
  fn public_operation_rejects_equal_modulus_and_accepts_modulus_minus_one(mut modulus in proptest::collection::vec(any::<u8>(), 256)) {
    modulus[0] |= 0x80;
    modulus[255] |= 0x01;
    let exponent = RsaPublicExponent(65_537);
    let bits = validate_modulus(&modulus, &RsaPublicKeyPolicy::modulus_2048_to_8192())
      .expect("generated full-width odd modulus must satisfy the legacy test policy");
    let key = RsaPublicKey {
      modulus: RsaPublicModulus::new(&modulus, bits),
      exponent,
    };
    let mut out = vec![0u8; key.modulus().len()];
    let mut scratch = key.public_scratch();

    prop_assert_eq!(
      key.public_operation_with_scratch(&modulus, &mut out, &mut scratch),
      Err(RsaPublicOpError::RepresentativeOutOfRange)
    );

    let mut less = modulus.clone();
    for byte in less.iter_mut().rev() {
      if *byte != 0 {
        *byte = byte.strict_sub(1);
        break;
      }
      *byte = 0xff;
    }

    prop_assert!(key.public_operation_with_scratch(&less, &mut out, &mut scratch).is_ok());
  }
}
