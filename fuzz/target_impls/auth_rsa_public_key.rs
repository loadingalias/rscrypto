use rscrypto::{RsaPublicKey, RsaPublicKeyPolicy};
use rscrypto_fuzz::{FuzzInput, some_or_return};

pub fn run(data: &[u8]) {
  let mut input = FuzzInput::new(data);
  let mode = some_or_return!(input.byte());
  let raw_der = input.rest();
  let generated_der;
  let der = if let Some(der) = generated_public_key_der(mode, raw_der) {
    generated_der = der;
    generated_der.as_slice()
  } else {
    raw_der
  };

  let parsed = match mode % 4 {
    0 => RsaPublicKey::from_pkcs1_der(der),
    1 => RsaPublicKey::from_spki_der(der),
    2 => RsaPublicKey::from_pkcs1_der_with_policy(der, &RsaPublicKeyPolicy::modern_verification()),
    _ => RsaPublicKey::from_spki_der_with_policy(
      der,
      &RsaPublicKeyPolicy::legacy_verification().allow_legacy_small_exponents(),
    ),
  };

  if let Ok(key) = parsed {
    let mut representative = key.modulus().to_vec();
    for byte in representative.iter_mut().rev() {
      if *byte != 0 {
        *byte = byte.strict_sub(1);
        break;
      }
      *byte = 0xff;
    }

    let mut out = vec![0u8; key.modulus().len()];
    let mut scratch = key.public_scratch();
    key
      .public_operation_with_scratch(&representative, &mut out, &mut scratch)
      .expect("modulus - 1 representative must be accepted");
  }
}

fn generated_public_key_der(mode: u8, control: &[u8]) -> Option<Vec<u8>> {
  let pkcs1 = valid_pkcs1_public_key_der();
  match control.first().copied() {
    Some(b'V') => Some(public_key_der_for_mode(mode, &pkcs1)),
    Some(b'T') => {
      let mut der = public_key_der_for_mode(mode, &pkcs1);
      der.extend_from_slice(&der_null());
      Some(der)
    }
    Some(b'L') => Some(tlv_with_leading_zero_long_len(&public_key_der_for_mode(mode, &pkcs1))),
    Some(b'U') if mode % 2 == 1 => Some(spki_public_key_der_with_unused_bits(&pkcs1)),
    Some(b'N') if mode.is_multiple_of(2) => Some(pkcs1_public_key_der_with_noncanonical_exponent()),
    _ => None,
  }
}

fn public_key_der_for_mode(mode: u8, pkcs1: &[u8]) -> Vec<u8> {
  if mode.is_multiple_of(2) {
    pkcs1.to_vec()
  } else {
    spki_public_key_der(pkcs1)
  }
}

fn valid_pkcs1_public_key_der() -> Vec<u8> {
  sequence(&[
    integer_unsigned(&hex_to_vec(RSA_PUBLIC_MODULUS_HEX)),
    integer_unsigned(&[0x01, 0x00, 0x01]),
  ])
}

fn pkcs1_public_key_der_with_noncanonical_exponent() -> Vec<u8> {
  sequence(&[
    integer_unsigned(&hex_to_vec(RSA_PUBLIC_MODULUS_HEX)),
    tlv(0x02, &[0x00, 0x01, 0x00, 0x01]),
  ])
}

fn spki_public_key_der(pkcs1: &[u8]) -> Vec<u8> {
  spki_public_key_der_with_subject_public_key(&subject_public_key_bit_string(0, pkcs1))
}

fn spki_public_key_der_with_unused_bits(pkcs1: &[u8]) -> Vec<u8> {
  spki_public_key_der_with_subject_public_key(&subject_public_key_bit_string(1, pkcs1))
}

fn spki_public_key_der_with_subject_public_key(subject_public_key: &[u8]) -> Vec<u8> {
  sequence(&[
    algorithm_identifier(RSA_ENCRYPTION_OID, Some(&der_null())),
    tlv(0x03, subject_public_key),
  ])
}

fn subject_public_key_bit_string(unused_bits: u8, pkcs1: &[u8]) -> Vec<u8> {
  let mut out = Vec::with_capacity(pkcs1.len().strict_add(1));
  out.push(unused_bits);
  out.extend_from_slice(pkcs1);
  out
}

fn sequence(fields: &[Vec<u8>]) -> Vec<u8> {
  let len = fields.iter().map(Vec::len).sum();
  let mut body = Vec::with_capacity(len);
  for field in fields {
    body.extend_from_slice(field);
  }
  tlv(0x30, &body)
}

fn algorithm_identifier(algorithm_oid: &[u8], params: Option<&[u8]>) -> Vec<u8> {
  let mut body = Vec::new();
  body.extend_from_slice(&tlv(0x06, algorithm_oid));
  if let Some(params) = params {
    body.extend_from_slice(params);
  }
  tlv(0x30, &body)
}

fn integer_unsigned(value: &[u8]) -> Vec<u8> {
  let first_nonzero = value.iter().position(|&byte| byte != 0);
  let value = first_nonzero.map_or(&[0u8][..], |index| &value[index..]);
  let mut encoded = Vec::with_capacity(value.len() + usize::from(value[0] & 0x80 != 0));
  if value[0] & 0x80 != 0 {
    encoded.push(0);
  }
  encoded.extend_from_slice(value);
  tlv(0x02, &encoded)
}

fn tlv(tag: u8, value: &[u8]) -> Vec<u8> {
  let mut out = Vec::with_capacity(1 + der_len(value.len()).len() + value.len());
  out.push(tag);
  out.extend_from_slice(&der_len(value.len()));
  out.extend_from_slice(value);
  out
}

fn tlv_with_leading_zero_long_len(der: &[u8]) -> Vec<u8> {
  let tag = der[0];
  let len_len = usize::from(der[1] & 0x7f);
  debug_assert_ne!(der[1] & 0x80, 0);
  debug_assert_ne!(len_len, 0);

  let mut out = Vec::with_capacity(der.len().strict_add(1));
  out.push(tag);
  out.push(0x80 | (len_len.strict_add(1) as u8));
  out.push(0);
  out.extend_from_slice(&der[2..]);
  out
}

fn der_len(len: usize) -> Vec<u8> {
  if len < 128 {
    return vec![len as u8];
  }

  let bytes = len.to_be_bytes();
  let first_nonzero = bytes.iter().position(|&byte| byte != 0).unwrap();
  let len_bytes = &bytes[first_nonzero..];
  let mut out = Vec::with_capacity(1 + len_bytes.len());
  out.push(0x80 | len_bytes.len() as u8);
  out.extend_from_slice(len_bytes);
  out
}

fn der_null() -> Vec<u8> {
  tlv(0x05, &[])
}

fn hex_to_vec(hex: &str) -> Vec<u8> {
  let mut out = Vec::with_capacity(hex.len() / 2);
  for chunk in hex.as_bytes().chunks_exact(2) {
    let hi = hex_value(chunk[0]);
    let lo = hex_value(chunk[1]);
    out.push((hi << 4) | lo);
  }
  out
}

fn hex_value(byte: u8) -> u8 {
  match byte {
    b'0'..=b'9' => byte - b'0',
    b'a'..=b'f' => byte - b'a' + 10,
    b'A'..=b'F' => byte - b'A' + 10,
    _ => 0,
  }
}

const RSA_ENCRYPTION_OID: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x01];

const RSA_PUBLIC_MODULUS_HEX: &str = "\
a246ccf6bd59720287837151de9fa55d4a811e456643f7fd0ced5a9ffa8fe52a89d52a8f6bd96246c9f0d\
23cd4f215609bfd0fd09dfcf13305440cae6e1b9a3c48e8e360438ca9993c1cd8ec03363cc3d79edbc4df776\
4c7f8ddb75f1148037847b356d2697f7d0158072a2e4f38f940c8db08b70305dedb6fe97aeb530dccc009274\
f7864442f6f02cf6191b5a32268234bcbd7827bf3e570206c0cddf147df5169ceda6883b2169768878fd5b107\
a092ab7482d8ba7f46364b566aaa72153068b6a0174f2f5e0e5f9bcd0213dd4e8689d56ffa0be918a16fffc\
be4830157eb8535c1a2a50636f8fc8a57f9ae0488b91159456ca94d7e64a1286babad3e92f702";
