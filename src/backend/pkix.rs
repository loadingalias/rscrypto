//! NIST post-quantum key encodings from RFC 9881 (ML-DSA) and RFC 9935
//! (ML-KEM): SubjectPublicKeyInfo and RFC 5958 OneAsymmetricKey (PKCS #8).
//!
//! Both RFCs name a parameter set by the OID 2.16.840.1.101.3.4.`family`.`arc`
//! with absent parameters, carry the raw public key in a BIT STRING, and encode
//! a private key as a CHOICE of seed, expanded key, or both. Only the arcs, the
//! seed length, and the key lengths differ.
//!
//! SPKI import accepts exactly the unique encoding; a structural parse only
//! classifies a rejection. PKCS #8 import accepts each private-key form in a
//! version 1 container, or a version 2 container with a public key, and
//! rejects attributes. Export writes version 1 without a public key, so each
//! exported form is a fixed header followed by its payload.

use crate::backend::der::{self, MalformedDer, TAG_BIT_STRING, TAG_OBJECT_IDENTIFIER, TAG_SEQUENCE};

const TAG_INTEGER: u8 = 0x02;
const TAG_OCTET_STRING: u8 = 0x04;
/// Private-key CHOICE `seed [0] IMPLICIT OCTET STRING`.
const TAG_SEED: u8 = 0x80;
/// OneAsymmetricKey `attributes [0] IMPLICIT SET OF`.
const TAG_ATTRIBUTES: u8 = 0xa0;
/// OneAsymmetricKey `publicKey [1] IMPLICIT BIT STRING`.
const TAG_PUBLIC_KEY: u8 = 0x81;
/// Long-form DER length with two length octets.
const LONG_LENGTH_2: u8 = 0x82;

/// DER contents of the NIST algorithms arc, 2.16.840.1.101.3.4.
const NIST_ALGORITHMS: [u8; 7] = [0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04];

/// Header bytes before the raw public key in an SPKI.
pub(crate) const SPKI_HEADER_LENGTH: usize = 22;
/// Header bytes before the seed in the version 1 seed form.
pub(crate) const SEED_HEADER_LENGTH: usize = 22;
/// Header bytes before the key in the version 1 expanded form.
pub(crate) const EXPANDED_HEADER_LENGTH: usize = 28;

/// Position of the 9-byte OID contents within an SPKI header.
const SPKI_OID: core::ops::Range<usize> = 8..17;

/// Key-import failures that each family maps onto its own public error.
pub(crate) trait KeyError: MalformedDer {
  /// Another algorithm or parameter set.
  const UNSUPPORTED_ALGORITHM: Self;
  /// A well-formed key in a form this import does not accept.
  const UNSUPPORTED_ENCODING: Self;
  /// A public key of the wrong length.
  const INVALID_PUBLIC_KEY: Self;
  /// A private key of the wrong length.
  const INVALID_SECRET_KEY: Self;
}

/// The parameter set named by OID 2.16.840.1.101.3.4.`family`.`arc`.
#[derive(Clone, Copy)]
pub(crate) struct Algorithm {
  /// `3` for signature algorithms, `4` for KEMs.
  pub(crate) family: u8,
  pub(crate) arc: u8,
}

impl Algorithm {
  const fn oid(self) -> [u8; 9] {
    let [a, b, c, d, e, f, g] = NIST_ALGORITHMS;
    [a, b, c, d, e, f, g, self.family, self.arc]
  }

  /// AlgorithmIdentifier: one 11-byte OID element and no parameters.
  const fn identifier(self) -> [u8; 13] {
    let [a, b, c, d, e, f, g, h, i] = self.oid();
    [
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
      i,
    ]
  }
}

/// SPKI header for `algorithm` and a `key_len`-byte raw public key.
pub(crate) const fn spki_header(algorithm: Algorithm, key_len: usize) -> [u8; SPKI_HEADER_LENGTH] {
  let [outer_high, outer_low] = two_byte_length(SPKI_HEADER_LENGTH.strict_sub(4).strict_add(key_len));
  let [key_high, key_low] = two_byte_length(key_len.strict_add(1));
  let [s, s_len, o, o_len, a, b, c, d, e, f, g, h, i] = algorithm.identifier();
  [
    TAG_SEQUENCE,
    LONG_LENGTH_2,
    outer_high,
    outer_low,
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
    i,
    TAG_BIT_STRING,
    LONG_LENGTH_2,
    key_high,
    key_low,
    // No unused bits.
    0x00,
  ]
}

/// Return the raw public key if `der` is exactly `header` followed by `K` bytes.
pub(crate) fn decode_spki<'a, E: KeyError, const K: usize>(
  der: &'a [u8],
  header: &[u8; SPKI_HEADER_LENGTH],
) -> Result<&'a [u8; K], E> {
  match der.split_first_chunk::<SPKI_HEADER_LENGTH>() {
    Some((prefix, key)) if prefix == header => key
      .try_into()
      .map_err(|_| classify_spki::<E, K>(der, &header[SPKI_OID])),
    _ => Err(classify_spki::<E, K>(der, &header[SPKI_OID])),
  }
}

/// Name the first failing component in order: DER structure, algorithm,
/// parameters, then key length.
fn classify_spki<E: KeyError, const K: usize>(der: &[u8], oid: &[u8]) -> E {
  let classified = (|| {
    let mut root = der::DerReader::<E>::new(der);
    let spki = root.read_constructed(TAG_SEQUENCE)?;
    root.finish()?;

    let mut spki = der::DerReader::<E>::new(spki);
    let algorithm = spki.read_constructed(TAG_SEQUENCE)?;
    let subject_public_key = spki.read_primitive(TAG_BIT_STRING)?;
    spki.finish()?;

    let mut algorithm = der::DerReader::<E>::new(algorithm);
    if algorithm.read_primitive(TAG_OBJECT_IDENTIFIER)? != oid {
      return Err(E::UNSUPPORTED_ALGORITHM);
    }
    // Parameters must be absent.
    algorithm.finish()?;

    let (&unused_bits, key) = subject_public_key.split_first().ok_or(E::MALFORMED_DER)?;
    if unused_bits != 0 {
      return Err(E::MALFORMED_DER);
    }
    if key.len() != K {
      return Err(E::INVALID_PUBLIC_KEY);
    }
    Ok(())
  })();
  // A well-formed encoding of this algorithm with the right key length is
  // exactly the header and key, which the caller's exact match accepts.
  classified.err().unwrap_or(E::MALFORMED_DER)
}

/// Borrowed private-key material from one OneAsymmetricKey.
pub(crate) enum PrivateKey<'a, const S: usize, const SK: usize> {
  Seed(&'a [u8; S]),
  Expanded(&'a [u8; SK]),
  Both { seed: &'a [u8; S], expanded: &'a [u8; SK] },
}

/// Fields of one OneAsymmetricKey, borrowed from the input.
pub(crate) struct Pkcs8<'a, const S: usize, const SK: usize, const PK: usize> {
  pub(crate) private_key: PrivateKey<'a, S, SK>,
  /// The version 2 public key; its consistency is the caller's check.
  pub(crate) public_key: Option<&'a [u8; PK]>,
}

/// Parse a OneAsymmetricKey for `algorithm` with an `S`-byte seed, an
/// `SK`-byte expanded key, and a `PK`-byte public key.
///
/// Rejections follow the structure in order. Only lengths are checked here;
/// the caller validates the key material and its redundant fields.
pub(crate) fn decode_pkcs8<E: KeyError, const S: usize, const SK: usize, const PK: usize>(
  der: &[u8],
  algorithm: Algorithm,
) -> Result<Pkcs8<'_, S, SK, PK>, E> {
  let mut root = der::DerReader::<E>::new(der);
  let info = root.read_constructed(TAG_SEQUENCE)?;
  root.finish()?;

  let mut info = der::DerReader::<E>::new(info);
  // RFC 5958: v2 (1) exactly when a public key is present, otherwise v1 (0).
  let version_2 = match info.read_primitive(TAG_INTEGER)? {
    [0] => false,
    [1] => true,
    _ => return Err(E::MALFORMED_DER),
  };

  let mut identifier = der::DerReader::<E>::new(info.read_constructed(TAG_SEQUENCE)?);
  if identifier.read_primitive(TAG_OBJECT_IDENTIFIER)? != algorithm.oid() {
    return Err(E::UNSUPPORTED_ALGORITHM);
  }
  // Parameters must be absent.
  identifier.finish()?;

  let private_key = private_key(info.read_primitive(TAG_OCTET_STRING)?)?;
  if info.peek_byte() == Some(TAG_ATTRIBUTES) {
    return Err(E::UNSUPPORTED_ENCODING);
  }
  let public_key = if version_2 {
    Some(public_key(info.read_primitive(TAG_PUBLIC_KEY)?)?)
  } else {
    None
  };
  info.finish()?;

  Ok(Pkcs8 {
    private_key,
    public_key,
  })
}

/// Select the CHOICE variant by its tag, never by its length.
fn private_key<E: KeyError, const S: usize, const SK: usize>(contents: &[u8]) -> Result<PrivateKey<'_, S, SK>, E> {
  let mut choice = der::DerReader::<E>::new(contents);
  let key = match choice.peek_byte() {
    Some(TAG_SEED) => PrivateKey::Seed(fixed(choice.read_primitive(TAG_SEED)?)?),
    Some(TAG_OCTET_STRING) => PrivateKey::Expanded(fixed(choice.read_primitive(TAG_OCTET_STRING)?)?),
    Some(TAG_SEQUENCE) => {
      let mut both = der::DerReader::<E>::new(choice.read_constructed(TAG_SEQUENCE)?);
      let seed = fixed(both.read_primitive(TAG_OCTET_STRING)?)?;
      let expanded = fixed(both.read_primitive(TAG_OCTET_STRING)?)?;
      both.finish()?;
      PrivateKey::Both { seed, expanded }
    }
    _ => return Err(E::MALFORMED_DER),
  };
  choice.finish()?;
  Ok(key)
}

fn fixed<E: KeyError, const N: usize>(contents: &[u8]) -> Result<&[u8; N], E> {
  contents.try_into().map_err(|_| E::INVALID_SECRET_KEY)
}

fn public_key<E: KeyError, const PK: usize>(bits: &[u8]) -> Result<&[u8; PK], E> {
  let (&unused_bits, key) = bits.split_first().ok_or(E::MALFORMED_DER)?;
  if unused_bits != 0 {
    return Err(E::MALFORMED_DER);
  }
  key.try_into().map_err(|_| E::INVALID_PUBLIC_KEY)
}

/// Version 1 seed-form header for `algorithm` and an `S`-byte seed.
pub(crate) const fn seed_header<const S: usize>(algorithm: Algorithm) -> [u8; SEED_HEADER_LENGTH] {
  let [s, s_len, o, o_len, a, b, c, d, e, f, g, h, i] = algorithm.identifier();
  [
    TAG_SEQUENCE,
    // Version, AlgorithmIdentifier, and the private-key OCTET STRING.
    short_length(SEED_HEADER_LENGTH.strict_sub(2).strict_add(S)),
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
    i,
    TAG_OCTET_STRING,
    short_length(S.strict_add(2)),
    TAG_SEED,
    short_length(S),
  ]
}

/// Version 1 expanded-form header for `algorithm` and a `key_len`-byte key.
pub(crate) const fn expanded_header(algorithm: Algorithm, key_len: usize) -> [u8; EXPANDED_HEADER_LENGTH] {
  let [outer_high, outer_low] = two_byte_length(EXPANDED_HEADER_LENGTH.strict_sub(4).strict_add(key_len));
  let [wrapper_high, wrapper_low] = two_byte_length(key_len.strict_add(4));
  let [key_high, key_low] = two_byte_length(key_len);
  let [s, s_len, o, o_len, a, b, c, d, e, f, g, h, i] = algorithm.identifier();
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
    i,
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

/// `header` followed by `payload`, as an array.
pub(crate) const fn concat<const H: usize, const P: usize, const N: usize>(
  header: &[u8; H],
  payload: &[u8; P],
) -> [u8; N] {
  const { assert!(N == H.strict_add(P), "an encoding is its header and payload") };
  let mut out = [0; N];
  let (prefix, rest) = out.split_at_mut(H);
  prefix.copy_from_slice(header);
  rest.copy_from_slice(payload);
  out
}

/// Write `header` followed by `payload` into the caller's buffer.
pub(crate) fn write<const H: usize, const P: usize, const N: usize>(
  header: &[u8; H],
  payload: &[u8; P],
  out: &mut [u8; N],
) {
  const { assert!(N == H.strict_add(P), "an encoding is its header and payload") };
  let (prefix, rest) = out.split_at_mut(H);
  prefix.copy_from_slice(header);
  rest.copy_from_slice(payload);
}

const fn short_length(len: usize) -> u8 {
  assert!(len < 0x80, "a short DER length");
  let [.., low] = len.to_be_bytes();
  low
}

const fn two_byte_length(len: usize) -> [u8; 2] {
  assert!(len >= 0x100 && len <= 0xffff, "a DER length with two length octets");
  let [.., high, low] = len.to_be_bytes();
  [high, low]
}
