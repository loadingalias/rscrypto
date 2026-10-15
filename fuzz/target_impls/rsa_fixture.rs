//! Shared RSA key and DER helpers for the import and private-operation fuzz targets.

pub(super) fn hex_value(byte: u8) -> Option<u8> {
  match byte {
    b'0'..=b'9' => Some(byte.strict_sub(b'0')),
    b'a'..=b'f' => Some(byte.strict_sub(b'a').strict_add(10)),
    b'A'..=b'F' => Some(byte.strict_sub(b'A').strict_add(10)),
    _ => None,
  }
}

pub(super) fn valid_pkcs1_private_key_der() -> Vec<u8> {
  pkcs1_private_key_der_with_crt(
    &hex_to_vec(RSA_PRIVATE_EXPONENT_P_HEX),
    &hex_to_vec(RSA_PRIVATE_EXPONENT_Q_HEX),
    &hex_to_vec(RSA_PRIVATE_COEFFICIENT_HEX),
  )
}

pub(super) fn pkcs1_private_key_der_with_crt(exponent_p: &[u8], exponent_q: &[u8], coefficient: &[u8]) -> Vec<u8> {
  sequence(&[
    integer_unsigned(&[0]),
    integer_unsigned(&hex_to_vec(RSA_PRIVATE_MODULUS_HEX)),
    integer_unsigned(&[0x01, 0x00, 0x01]),
    integer_unsigned(&hex_to_vec(RSA_PRIVATE_EXPONENT_HEX)),
    integer_unsigned(&hex_to_vec(RSA_PRIVATE_PRIME_P_HEX)),
    integer_unsigned(&hex_to_vec(RSA_PRIVATE_PRIME_Q_HEX)),
    integer_unsigned(exponent_p),
    integer_unsigned(exponent_q),
    integer_unsigned(coefficient),
  ])
}

pub(super) fn sequence(fields: &[Vec<u8>]) -> Vec<u8> {
  let len = fields.iter().map(Vec::len).sum();
  let mut body = Vec::with_capacity(len);
  for field in fields {
    body.extend_from_slice(field);
  }
  tlv(0x30, &body)
}

pub(super) fn integer_unsigned(value: &[u8]) -> Vec<u8> {
  let first_nonzero = value.iter().position(|&byte| byte != 0);
  let value = first_nonzero.map_or(&[0u8][..], |index| &value[index..]);
  let mut encoded = Vec::with_capacity(value.len().strict_add(usize::from(value[0] & 0x80 != 0)));
  if value[0] & 0x80 != 0 {
    encoded.push(0);
  }
  encoded.extend_from_slice(value);
  tlv(0x02, &encoded)
}

pub(super) fn tlv(tag: u8, value: &[u8]) -> Vec<u8> {
  let mut out = Vec::with_capacity(1usize.strict_add(der_len(value.len()).len()).strict_add(value.len()));
  out.push(tag);
  out.extend_from_slice(&der_len(value.len()));
  out.extend_from_slice(value);
  out
}

fn der_len(len: usize) -> Vec<u8> {
  if len < 128 {
    return vec![u8::try_from(len).expect("short DER length fits one octet")];
  }

  let bytes = len.to_be_bytes();
  let first_nonzero = bytes
    .iter()
    .position(|&byte| byte != 0)
    .unwrap_or_else(|| bytes.len().strict_sub(1));
  let len_bytes = &bytes[first_nonzero..];
  let mut out = Vec::with_capacity(1usize.strict_add(len_bytes.len()));
  out.push(0x80 | u8::try_from(len_bytes.len()).expect("usize byte width fits one octet"));
  out.extend_from_slice(len_bytes);
  out
}

pub(super) fn hex_to_vec(hex: &str) -> Vec<u8> {
  let mut out = Vec::with_capacity(hex.len() / 2);
  for chunk in hex.as_bytes().as_chunks::<2>().0 {
    let high = hex_value(chunk[0]).expect("fixture must contain hexadecimal digits");
    let low = hex_value(chunk[1]).expect("fixture must contain hexadecimal digits");
    out.push((high << 4) | low);
  }
  out
}

pub(super) const RSA_PRIVATE_MODULUS_HEX: &str = "\
d397b84d98a4c26138ed1b695a8106ead91d553bf06041b62d3fdc50a041e222b8f4529689c1b82c5e71554f5d\
d69fa2f4b6158cf0dbeb57811a0fc327e1f28e74fe74d3bc166c1eabdc1b8b57b934ca8be5b00b4f29975bcc\
99acaf415b59bb28a6782bb41a2c3c2976b3c18dbadef62f00c6bb226640095096c0cc60d22fe7ef987d75c\
6a81b10d96bf292028af110dc7cc1bbc43d22adab379a0cd5d8078cc780ff5cd6209dea34c922cf784f7717e\
428d75b5aec8ff30e5f0141510766e2e0ab8d473c84e8710b2b98227c3db095337ad3452f19e2b9bfbccdd8\
148abf6776fa552775e6e75956e45229ae5a9c46949bab1e622f0e48f56524a84ed3483b";

pub(super) const RSA_PRIVATE_EXPONENT_HEX: &str = "\
c4e70c689162c94c660828191b52b4d8392115df486a9adbe831e458d73958320dc1b755456e93701e9702d76\
fb0b92f90e01d1fe248153281fe79aa9763a92fae69d8d7ecd144de29fa135bd14f9573e349e45031e3b76982\
f583003826c552e89a397c1a06bd2163488630d92e8c2bb643d7abef700da95d685c941489a46f54b5316f62\
b5d2c3a7f1bbd134cb37353a44683fdc9d95d36458de22f6c44057fe74a0a436c4308f73f4da42f35c47ac1\
6a7138d483afc91e41dc3a1127382e0c0f5119b0221b4fc639d6b9c38177a6de9b526ebd88c38d7982c07f98\
a0efd877d508aae275b946915c02e2e1106d175d74ec6777f5e80d12c053d9c7be1e341";

pub(super) const RSA_PRIVATE_PRIME_P_HEX: &str = "\
f827bbf3a41877c7cc59aebf42ed4b29c32defcb8ed96863d5b090a05a8930dd624a21c9dcf9838568fdfa0d\
f65b8462a5f2ac913d6c56f975532bd8e78fb07bd405ca99a484bcf59f019bbddcb3933f2bce706300b4f7b\
110120c5df9018159067c35da3061a56c8635a52b54273b31271b4311f0795df6021e6355e1a42e61";

pub(super) const RSA_PRIVATE_PRIME_Q_HEX: &str = "\
da4817ce0089dd36f2ade6a3ff410c73ec34bf1b4f6bda38431bfede11cef1f7f6efa70e5f8063a3b1f6e172\
96ffb15feefa0912a0325b8d1fd65a559e717b5b961ec345072e0ec5203d03441d29af4d64054a04507410cf\
1da78e7b6119d909ec66e6ad625bf995b279a4b3c5be7d895cd7c5b9c4c497fde730916fcdb4e41b";

pub(super) const RSA_PRIVATE_EXPONENT_P_HEX: &str = "\
1da6e9cf80212856e87522eb59bcef094b7836ba1514a7639e8a1d8dfba37f0245176498315e6337d2c6de554\
2c5c6b8dee973735b6a91adf735fbfc4c1720587b8a419e40495826e55c14d70803312a103af7b4ecc5b2ff26\
5371c4dcd730348a10d7827ddb7d1fcd9da561db09610a4b88f767b25b5e3de21ced73baa59aa1";

pub(super) const RSA_PRIVATE_EXPONENT_Q_HEX: &str = "\
d737a7c8e43d0a10c85bf0011886a16996a6371b0d46b0c5325de3003f9cc47491539f6a0b7d82407f12851c\
bf86e1f34da3d7d8367d104967efa7e7ad2e04cbbb8b1f4aeb165d57bd3e8afed8a62602ef304bd74f1ff106\
d51d44dd9f52a5ed23da1d6d2c82b4e6052fecd5978e0726ad94cd8e295510eb35cc6c49491026ab";

pub(super) const RSA_PRIVATE_COEFFICIENT_HEX: &str = "\
5268d7cf073479aebb2d2ed4dd66b8c89915b52d141e0c4932f56b0c0ed0936141894ec4d27d53bc86453cd8\
ca5b455045218c7e196209c1c651702ece090a15e3cbcc265971300023a86fe9d34ad527e9ef03b7adfe736e\
0680747abfd49839b82f2ffdec43bd0343ca30e13961b32af6cdeddd195672c76b53b76fc3ea76f8";
