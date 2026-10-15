#[cfg(any(fuzzing, rscrypto_internal_fuzzing))]
#[path = "rsa_fixture.rs"]
mod rsa_fixture;
#[cfg(any(fuzzing, rscrypto_internal_fuzzing))]
use rsa_fixture::{
  RSA_PRIVATE_COEFFICIENT_HEX, RSA_PRIVATE_EXPONENT_HEX, RSA_PRIVATE_EXPONENT_P_HEX, RSA_PRIVATE_EXPONENT_Q_HEX,
  RSA_PRIVATE_MODULUS_HEX, RSA_PRIVATE_PRIME_P_HEX, RSA_PRIVATE_PRIME_Q_HEX, hex_to_vec, hex_value, integer_unsigned,
  pkcs1_private_key_der_with_crt, sequence, tlv, valid_pkcs1_private_key_der,
};

#[cfg(any(fuzzing, rscrypto_internal_fuzzing))]
use rscrypto::auth::rsa::fuzz_rsa_import_der;
#[cfg(any(fuzzing, rscrypto_internal_fuzzing))]
use rscrypto_fuzz::{FuzzInput, some_or_return};

#[cfg(any(fuzzing, rscrypto_internal_fuzzing))]
pub(super) fn run(data: &[u8]) {
  let mut input = FuzzInput::new(data);
  let format = some_or_return!(input.byte());
  let (der, expected) = if input.rest().first().copied() == Some(b'V') {
    valid_private_key_der(format, &input.rest()[1..])
  } else {
    (decoded_der(input.rest()), None)
  };

  let accepted = fuzz_rsa_import_der(format, &der);
  if let Some(expected) = expected {
    assert_eq!(
      accepted, expected,
      "generated RSA private-key import fixture expectation drifted"
    );
  }
}

#[cfg(not(any(fuzzing, rscrypto_internal_fuzzing)))]
pub(super) fn run(_data: &[u8]) {}

#[cfg(any(fuzzing, rscrypto_internal_fuzzing))]
fn decoded_der(input: &[u8]) -> Vec<u8> {
  if input.first().copied() != Some(b'H') {
    return input.to_vec();
  }

  let mut out = Vec::with_capacity(input.len() / 2);
  let mut high = None;
  for &byte in &input[1..] {
    let Some(value) = hex_value(byte) else {
      continue;
    };
    if let Some(high_value) = high.take() {
      out.push((high_value << 4) | value);
    } else {
      high = Some(value);
    }
  }
  out
}

#[cfg(any(fuzzing, rscrypto_internal_fuzzing))]
fn valid_private_key_der(format: u8, control: &[u8]) -> (Vec<u8>, Option<bool>) {
  let control = control.strip_suffix(b"\n").unwrap_or(control);
  let pkcs1 = match control.first().copied() {
    Some(b'P') => pkcs1_private_key_der_with_crt(
      &[1],
      &hex_to_vec(RSA_PRIVATE_EXPONENT_Q_HEX),
      &hex_to_vec(RSA_PRIVATE_COEFFICIENT_HEX),
    ),
    Some(b'Q') => pkcs1_private_key_der_with_crt(
      &hex_to_vec(RSA_PRIVATE_EXPONENT_P_HEX),
      &[1],
      &hex_to_vec(RSA_PRIVATE_COEFFICIENT_HEX),
    ),
    Some(b'C') => pkcs1_private_key_der_with_crt(
      &hex_to_vec(RSA_PRIVATE_EXPONENT_P_HEX),
      &hex_to_vec(RSA_PRIVATE_EXPONENT_Q_HEX),
      &[1],
    ),
    Some(b'N') => pkcs1_private_key_der_with_noncanonical_version(),
    _ => valid_pkcs1_private_key_der(),
  };
  let mut der = private_key_der_for_format(format, &pkcs1);

  if control.is_empty() {
    return (der, Some(true));
  }

  match control[0] {
    b'P' | b'Q' | b'C' | b'N' => (der, Some(false)),
    b'A' if format & 1 == 1 => (pkcs8_private_key_der_with_attributes(&pkcs1), Some(false)),
    b'L' => (tlv_with_leading_zero_long_len(&der), Some(false)),
    b'T' => {
      der.push(0);
      (der, Some(false))
    }
    b'R' => {
      der.pop();
      (der, Some(false))
    }
    b'X' => {
      if !der.is_empty() {
        let index = control.get(1).copied().map_or(0, usize::from).rem_euclid(der.len());
        let mask = control.get(2).copied().unwrap_or(0x01);
        der[index] ^= mask;
      }
      (der, None)
    }
    _ => (der, None),
  }
}

#[cfg(any(fuzzing, rscrypto_internal_fuzzing))]
fn pkcs1_private_key_der_with_noncanonical_version() -> Vec<u8> {
  sequence(&[
    tlv_with_noncanonical_short_len(0x02, &[0]),
    integer_unsigned(&hex_to_vec(RSA_PRIVATE_MODULUS_HEX)),
    integer_unsigned(&[0x01, 0x00, 0x01]),
    integer_unsigned(&hex_to_vec(RSA_PRIVATE_EXPONENT_HEX)),
    integer_unsigned(&hex_to_vec(RSA_PRIVATE_PRIME_P_HEX)),
    integer_unsigned(&hex_to_vec(RSA_PRIVATE_PRIME_Q_HEX)),
    integer_unsigned(&hex_to_vec(RSA_PRIVATE_EXPONENT_P_HEX)),
    integer_unsigned(&hex_to_vec(RSA_PRIVATE_EXPONENT_Q_HEX)),
    integer_unsigned(&hex_to_vec(RSA_PRIVATE_COEFFICIENT_HEX)),
  ])
}

#[cfg(any(fuzzing, rscrypto_internal_fuzzing))]
fn private_key_der_for_format(format: u8, pkcs1: &[u8]) -> Vec<u8> {
  if format & 1 == 0 {
    pkcs1.to_vec()
  } else {
    valid_pkcs8_private_key_der(pkcs1)
  }
}

#[cfg(any(fuzzing, rscrypto_internal_fuzzing))]
fn valid_pkcs8_private_key_der(pkcs1: &[u8]) -> Vec<u8> {
  sequence(&[
    integer_unsigned(&[0]),
    algorithm_identifier(RSA_ENCRYPTION_OID, Some(&der_null())),
    tlv(0x04, pkcs1),
  ])
}

#[cfg(any(fuzzing, rscrypto_internal_fuzzing))]
fn pkcs8_private_key_der_with_attributes(pkcs1: &[u8]) -> Vec<u8> {
  sequence(&[
    integer_unsigned(&[0]),
    algorithm_identifier(RSA_ENCRYPTION_OID, Some(&der_null())),
    tlv(0x04, pkcs1),
    tlv(0xa0, &[]),
  ])
}

#[cfg(any(fuzzing, rscrypto_internal_fuzzing))]
fn algorithm_identifier(oid: &[u8], params: Option<&[u8]>) -> Vec<u8> {
  let mut body = Vec::new();
  body.extend_from_slice(&tlv(0x06, oid));
  if let Some(params) = params {
    body.extend_from_slice(params);
  }
  tlv(0x30, &body)
}

#[cfg(any(fuzzing, rscrypto_internal_fuzzing))]
fn der_null() -> Vec<u8> {
  tlv(0x05, &[])
}

#[cfg(any(fuzzing, rscrypto_internal_fuzzing))]
fn tlv_with_noncanonical_short_len(tag: u8, value: &[u8]) -> Vec<u8> {
  let mut out = Vec::with_capacity(3usize.strict_add(value.len()));
  out.push(tag);
  out.push(0x81);
  out.push(u8::try_from(value.len()).expect("short DER fixture length fits one octet"));
  out.extend_from_slice(value);
  out
}

#[cfg(any(fuzzing, rscrypto_internal_fuzzing))]
fn tlv_with_leading_zero_long_len(der: &[u8]) -> Vec<u8> {
  let len_len = usize::from(der[1] & 0x7f);
  let mut out = Vec::with_capacity(der.len().strict_add(1));
  out.push(der[0]);
  out.push(0x80 | u8::try_from(len_len.strict_add(1)).expect("DER length header fits one octet"));
  out.push(0);
  out.extend_from_slice(&der[2..]);
  out
}

#[cfg(any(fuzzing, rscrypto_internal_fuzzing))]
const RSA_ENCRYPTION_OID: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x01];
