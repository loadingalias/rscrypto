//! RFC 9881 SubjectPublicKeyInfo encoding of ML-DSA public keys.
//!
//! Each parameter set has one DER encoding: a fixed header followed by the raw
//! FIPS 204 public key. Import accepts exactly that encoding. A structural
//! parse runs only to explain a rejection and never accepts input.

use super::MlDsaKeyError;
use crate::backend::der::{self, MalformedDer, TAG_BIT_STRING, TAG_OBJECT_IDENTIFIER, TAG_SEQUENCE};

/// Header bytes before the raw public key.
pub(super) const HEADER_LENGTH: usize = 22;

/// DER contents of the NIST `sigAlgs` arc, 2.16.840.1.101.3.4.3. Each
/// `id-ml-dsa-*` OID appends one final arc.
const SIG_ALGS: [u8; 8] = [0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x03];

/// Position of the 9-byte algorithm OID contents within the header.
const OID: core::ops::Range<usize> = 8..17;

/// Long-form DER length with two length octets.
const LONG_LENGTH_2: u8 = 0x82;

type DerReader<'a> = der::DerReader<'a, MlDsaKeyError>;

impl MalformedDer for MlDsaKeyError {
  const MALFORMED_DER: Self = Self::MalformedDer;
}

/// Header for the parameter set whose OID ends in `arc` and whose raw public
/// key has `key_len` bytes. RFC 9881 section 2 requires absent parameters.
pub(super) const fn header(arc: u8, key_len: usize) -> [u8; HEADER_LENGTH] {
  let [outer_high, outer_low] = two_byte_length(HEADER_LENGTH.strict_sub(4).strict_add(key_len));
  let [key_high, key_low] = two_byte_length(key_len.strict_add(1));
  let [a, b, c, d, e, f, g, h] = SIG_ALGS;
  [
    TAG_SEQUENCE,
    LONG_LENGTH_2,
    outer_high,
    outer_low,
    // AlgorithmIdentifier: one 11-byte OID element and no parameters.
    TAG_SEQUENCE,
    0x0b,
    TAG_OBJECT_IDENTIFIER,
    0x09,
    a,
    b,
    c,
    d,
    e,
    f,
    g,
    h,
    arc,
    TAG_BIT_STRING,
    LONG_LENGTH_2,
    key_high,
    key_low,
    // No unused bits.
    0x00,
  ]
}

const fn two_byte_length(len: usize) -> [u8; 2] {
  assert!(
    len >= 0x100 && len <= 0xffff,
    "ML-DSA SPKI lengths use two length octets"
  );
  let [.., high, low] = len.to_be_bytes();
  [high, low]
}

/// Return the raw public key if `der` is exactly `header` followed by
/// `key_len` bytes.
pub(super) fn decode<'a>(
  der: &'a [u8],
  header: &[u8; HEADER_LENGTH],
  key_len: usize,
) -> Result<&'a [u8], MlDsaKeyError> {
  match der.split_first_chunk::<HEADER_LENGTH>() {
    Some((prefix, key)) if prefix == header && key.len() == key_len => Ok(key),
    _ => Err(match classify(der, &header[OID], key_len) {
      Err(error) => error,
      // A well-formed encoding of this algorithm with the right key length is
      // exactly the header and key, which the exact match above accepts.
      Ok(()) => MlDsaKeyError::MalformedDer,
    }),
  }
}

/// Find the first failing component in order: DER structure, algorithm,
/// parameters, then key length.
fn classify(der: &[u8], oid: &[u8], key_len: usize) -> Result<(), MlDsaKeyError> {
  let mut root = DerReader::new(der);
  let spki = root.read_constructed(TAG_SEQUENCE)?;
  root.finish()?;

  let mut spki = DerReader::new(spki);
  let algorithm = spki.read_constructed(TAG_SEQUENCE)?;
  let subject_public_key = spki.read_primitive(TAG_BIT_STRING)?;
  spki.finish()?;

  let mut algorithm = DerReader::new(algorithm);
  if algorithm.read_primitive(TAG_OBJECT_IDENTIFIER)? != oid {
    return Err(MlDsaKeyError::UnsupportedAlgorithm);
  }
  // Parameters must be absent.
  algorithm.finish()?;

  let (&unused_bits, key) = subject_public_key.split_first().ok_or(MlDsaKeyError::MalformedDer)?;
  if unused_bits != 0 {
    return Err(MlDsaKeyError::MalformedDer);
  }
  if key.len() != key_len {
    return Err(MlDsaKeyError::InvalidPublicKey);
  }
  Ok(())
}

/// Write `header` followed by `key`.
pub(super) const fn encode<const K: usize, const N: usize>(header: &[u8; HEADER_LENGTH], key: &[u8; K]) -> [u8; N] {
  const { assert!(N == HEADER_LENGTH.strict_add(K), "an SPKI is its header and key") };
  let mut out = [0; N];
  let (prefix, rest) = out.split_at_mut(HEADER_LENGTH);
  prefix.copy_from_slice(header);
  rest.copy_from_slice(key);
  out
}
