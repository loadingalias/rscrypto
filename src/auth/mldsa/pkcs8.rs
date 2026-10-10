//! RFC 9881 private keys in RFC 5958 OneAsymmetricKey (PKCS #8) DER.
//!
//! Import accepts each standard private-key form: the 32-byte seed, the
//! expanded key, or both. The container is version 1, or version 2 with a
//! public key; attributes are not accepted. Export writes version 1 without a
//! public key, so each exported form is a fixed header followed by its payload.

use super::{
  MlDsaKeyError,
  spki::{self, LONG_LENGTH_2, two_byte_length},
};
use crate::backend::der::{self, TAG_OBJECT_IDENTIFIER, TAG_SEQUENCE};

const TAG_INTEGER: u8 = 0x02;
const TAG_OCTET_STRING: u8 = 0x04;
/// Private-key CHOICE `seed [0] IMPLICIT OCTET STRING`.
const TAG_SEED: u8 = 0x80;
/// OneAsymmetricKey `attributes [0] IMPLICIT SET OF`.
const TAG_ATTRIBUTES: u8 = 0xa0;
/// OneAsymmetricKey `publicKey [1] IMPLICIT BIT STRING`.
const TAG_PUBLIC_KEY: u8 = 0x81;

/// FIPS 204 key-generation seed length.
pub(super) const SEED_LENGTH: usize = 32;
/// Header bytes before the seed in the version 1 seed form.
pub(super) const SEED_HEADER_LENGTH: usize = 22;
/// Header bytes before the key in the version 1 expanded form.
pub(super) const EXPANDED_HEADER_LENGTH: usize = 28;

type DerReader<'a> = der::DerReader<'a, MlDsaKeyError>;

/// Borrowed private-key material from one OneAsymmetricKey.
pub(super) enum PrivateKey<'a, const SK: usize> {
  Seed(&'a [u8; SEED_LENGTH]),
  Expanded(&'a [u8; SK]),
  Both {
    seed: &'a [u8; SEED_LENGTH],
    expanded: &'a [u8; SK],
  },
}

/// Fields of one OneAsymmetricKey, borrowed from the input.
pub(super) struct Decoded<'a, const SK: usize, const PK: usize> {
  pub(super) private_key: PrivateKey<'a, SK>,
  /// The version 2 public key; its consistency is the caller's check.
  pub(super) public_key: Option<&'a [u8; PK]>,
}

/// Parse a OneAsymmetricKey for the parameter set whose OID ends in `arc`.
///
/// Rejections follow the structure in order. Only lengths are checked here;
/// the caller validates the key material and its redundant fields.
pub(super) fn decode<const SK: usize, const PK: usize>(
  der: &[u8],
  arc: u8,
) -> Result<Decoded<'_, SK, PK>, MlDsaKeyError> {
  let mut root = DerReader::new(der);
  let info = root.read_constructed(TAG_SEQUENCE)?;
  root.finish()?;

  let mut info = DerReader::new(info);
  // RFC 5958: v2 (1) exactly when a public key is present, otherwise v1 (0).
  let version_2 = match info.read_primitive(TAG_INTEGER)? {
    [0] => false,
    [1] => true,
    _ => return Err(MlDsaKeyError::MalformedDer),
  };

  let mut algorithm = DerReader::new(info.read_constructed(TAG_SEQUENCE)?);
  if algorithm.read_primitive(TAG_OBJECT_IDENTIFIER)? != spki::oid(arc) {
    return Err(MlDsaKeyError::UnsupportedAlgorithm);
  }
  // Parameters must be absent.
  algorithm.finish()?;

  let private_key = private_key(info.read_primitive(TAG_OCTET_STRING)?)?;
  if info.peek_byte() == Some(TAG_ATTRIBUTES) {
    return Err(MlDsaKeyError::UnsupportedEncoding);
  }
  let public_key = if version_2 {
    Some(public_key(info.read_primitive(TAG_PUBLIC_KEY)?)?)
  } else {
    None
  };
  info.finish()?;

  Ok(Decoded {
    private_key,
    public_key,
  })
}

/// Select the CHOICE variant by its tag, never by its length (RFC 9881 section 6).
fn private_key<const SK: usize>(contents: &[u8]) -> Result<PrivateKey<'_, SK>, MlDsaKeyError> {
  let mut choice = DerReader::new(contents);
  let key = match choice.peek_byte() {
    Some(TAG_SEED) => PrivateKey::Seed(fixed(choice.read_primitive(TAG_SEED)?)?),
    Some(TAG_OCTET_STRING) => PrivateKey::Expanded(fixed(choice.read_primitive(TAG_OCTET_STRING)?)?),
    Some(TAG_SEQUENCE) => {
      let mut both = DerReader::new(choice.read_constructed(TAG_SEQUENCE)?);
      let seed = fixed(both.read_primitive(TAG_OCTET_STRING)?)?;
      let expanded = fixed(both.read_primitive(TAG_OCTET_STRING)?)?;
      both.finish()?;
      PrivateKey::Both { seed, expanded }
    }
    _ => return Err(MlDsaKeyError::MalformedDer),
  };
  choice.finish()?;
  Ok(key)
}

fn fixed<const N: usize>(contents: &[u8]) -> Result<&[u8; N], MlDsaKeyError> {
  contents.try_into().map_err(|_| MlDsaKeyError::InvalidSecretKey)
}

fn public_key<const PK: usize>(bits: &[u8]) -> Result<&[u8; PK], MlDsaKeyError> {
  let (&unused_bits, key) = bits.split_first().ok_or(MlDsaKeyError::MalformedDer)?;
  if unused_bits != 0 {
    return Err(MlDsaKeyError::MalformedDer);
  }
  key.try_into().map_err(|_| MlDsaKeyError::InvalidPublicKey)
}

/// Version 1 seed-form header for the OID that ends in `arc`.
pub(super) const fn seed_header(arc: u8) -> [u8; SEED_HEADER_LENGTH] {
  const {
    assert!(
      SEED_HEADER_LENGTH.strict_add(SEED_LENGTH).strict_sub(2) == 0x34
        && SEED_LENGTH.strict_add(2) == 0x22
        && SEED_LENGTH == 0x20,
      "the seed form's short lengths match its layout"
    )
  };
  let [s, s_len, o, o_len, a, b, c, d, e, f, g, h, arc] = spki::algorithm_identifier(arc);
  [
    TAG_SEQUENCE,
    // Version, AlgorithmIdentifier, and the 36-byte private key.
    0x34,
    TAG_INTEGER,
    0x01,
    0x00,
    s,
    s_len,
    o,
    o_len,
    a,
    b,
    c,
    d,
    e,
    f,
    g,
    h,
    arc,
    TAG_OCTET_STRING,
    0x22,
    TAG_SEED,
    0x20,
  ]
}

/// Version 1 expanded-form header for the OID that ends in `arc` and a
/// `key_len`-byte expanded key.
pub(super) const fn expanded_header(arc: u8, key_len: usize) -> [u8; EXPANDED_HEADER_LENGTH] {
  let [outer_high, outer_low] = two_byte_length(EXPANDED_HEADER_LENGTH.strict_sub(4).strict_add(key_len));
  let [wrapper_high, wrapper_low] = two_byte_length(key_len.strict_add(4));
  let [key_high, key_low] = two_byte_length(key_len);
  let [s, s_len, o, o_len, a, b, c, d, e, f, g, h, arc] = spki::algorithm_identifier(arc);
  [
    TAG_SEQUENCE,
    LONG_LENGTH_2,
    outer_high,
    outer_low,
    TAG_INTEGER,
    0x01,
    0x00,
    s,
    s_len,
    o,
    o_len,
    a,
    b,
    c,
    d,
    e,
    f,
    g,
    h,
    arc,
    TAG_OCTET_STRING,
    LONG_LENGTH_2,
    wrapper_high,
    wrapper_low,
    TAG_OCTET_STRING,
    LONG_LENGTH_2,
    key_high,
    key_low,
  ]
}

/// Write `header` followed by `payload` into the caller's buffer.
pub(super) fn write<const H: usize, const P: usize, const N: usize>(
  header: &[u8; H],
  payload: &[u8; P],
  out: &mut [u8; N],
) {
  const { assert!(N == H.strict_add(P), "an encoding is its header and payload") };
  let (prefix, rest) = out.split_at_mut(H);
  prefix.copy_from_slice(header);
  rest.copy_from_slice(payload);
}
