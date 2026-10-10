//! SubjectPublicKeyInfo and RFC 5958 OneAsymmetricKey (PKCS #8) key encodings.
//!
//! The NIST post-quantum encodings are RFC 9881 (ML-DSA), RFC 9935 (ML-KEM),
//! and RFC 9909 (SLH-DSA). All three name a parameter set by the OID
//! 2.16.840.1.101.3.4.`family`.`arc` with absent parameters and carry the raw
//! public key in a BIT STRING. RFC 9881 and RFC 9935 encode a private key as a
//! CHOICE of seed, expanded key, or both; RFC 9909 puts the raw private key
//! directly in `privateKey`. Only the arcs, the private-key form, and the key
//! lengths differ.
//!
//! RFC 8410 (Ed25519) also has absent parameters, and wraps the private key
//! in an inner OCTET STRING. RFC 5915 (ECDSA) names the curve as the
//! parameters of `id-ecPublicKey` and puts an `ECPrivateKey` in
//! `privateKey`.
//!
//! SPKI import accepts exactly the unique encoding; a structural parse only
//! classifies a rejection. PKCS #8 import accepts each private-key form in a
//! version 1 container, or a version 2 container with a public key, and
//! rejects attributes. Export writes version 1 without a container public key,
//! so each exported form is a fixed layout around its payload.

use crate::backend::der::{self, MalformedDer, TAG_BIT_STRING, TAG_OBJECT_IDENTIFIER, TAG_SEQUENCE};

const TAG_INTEGER: u8 = 0x02;
const TAG_OCTET_STRING: u8 = 0x04;
/// Private-key CHOICE `seed [0] IMPLICIT OCTET STRING`.
#[cfg(any(feature = "ml-dsa", feature = "ml-kem"))]
const TAG_SEED: u8 = 0x80;
/// OneAsymmetricKey `attributes [0] IMPLICIT SET OF`.
const TAG_ATTRIBUTES: u8 = 0xa0;
/// OneAsymmetricKey `publicKey [1] IMPLICIT BIT STRING`.
const TAG_PUBLIC_KEY: u8 = 0x81;
/// ECPrivateKey `parameters [0] EXPLICIT ECParameters`.
#[cfg(any(feature = "ecdsa-p256", feature = "ecdsa-p384"))]
const TAG_EC_PARAMETERS: u8 = 0xa0;
/// ECPrivateKey `publicKey [1] EXPLICIT BIT STRING`.
#[cfg(any(feature = "ecdsa-p256", feature = "ecdsa-p384"))]
const TAG_EC_PUBLIC_KEY: u8 = 0xa1;
/// Long-form DER length with two length octets.
#[cfg(any(feature = "ml-dsa", feature = "ml-kem", feature = "slh-dsa"))]
const LONG_LENGTH_2: u8 = 0x82;

/// DER contents of the NIST algorithms arc, 2.16.840.1.101.3.4.
#[cfg(any(feature = "ml-dsa", feature = "ml-kem", feature = "slh-dsa"))]
const NIST_ALGORITHMS: [u8; 7] = [0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04];

/// Header bytes before a raw public key of 255 bytes or more in an SPKI,
/// where both lengths take three octets.
#[cfg(any(feature = "ml-dsa", feature = "ml-kem"))]
pub(crate) const SPKI_HEADER_LENGTH: usize = 22;
/// Header bytes before the seed in the version 1 seed form.
#[cfg(any(feature = "ml-dsa", feature = "ml-kem"))]
pub(crate) const SEED_HEADER_LENGTH: usize = 22;
/// Header bytes before the key in the version 1 expanded form.
#[cfg(any(feature = "ml-dsa", feature = "ml-kem"))]
pub(crate) const EXPANDED_HEADER_LENGTH: usize = 28;
/// Bytes in an AlgorithmIdentifier with one 9-byte OID and no parameters.
#[cfg(any(feature = "ml-dsa", feature = "ml-kem", feature = "slh-dsa"))]
const IDENTIFIER_LENGTH: usize = 13;

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
#[cfg(any(feature = "ml-dsa", feature = "ml-kem", feature = "slh-dsa"))]
#[derive(Clone, Copy)]
pub(crate) struct Algorithm {
  /// `3` for signature algorithms, `4` for KEMs.
  pub(crate) family: u8,
  pub(crate) arc: u8,
}

#[cfg(any(feature = "ml-dsa", feature = "ml-kem", feature = "slh-dsa"))]
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

/// SPKI header length for a `key_len`-byte raw public key.
#[cfg(any(feature = "ml-dsa", feature = "ml-kem", feature = "slh-dsa"))]
pub(crate) const fn spki_header_length(key_len: usize) -> usize {
  let bit_string = key_len.strict_add(1);
  1usize
    .strict_add(length_octets(spki_contents_length(key_len)))
    .strict_add(IDENTIFIER_LENGTH)
    .strict_add(1)
    .strict_add(length_octets(bit_string))
    // No unused bits.
    .strict_add(1)
}

/// SPKI SEQUENCE contents: the AlgorithmIdentifier and the BIT STRING element.
#[cfg(any(feature = "ml-dsa", feature = "ml-kem", feature = "slh-dsa"))]
const fn spki_contents_length(key_len: usize) -> usize {
  let bit_string = key_len.strict_add(1);
  IDENTIFIER_LENGTH
    .strict_add(1)
    .strict_add(length_octets(bit_string))
    .strict_add(bit_string)
}

/// SPKI header for `algorithm` and a `key_len`-byte raw public key.
#[cfg(any(feature = "ml-dsa", feature = "ml-kem", feature = "slh-dsa"))]
pub(crate) const fn spki_header<const H: usize>(algorithm: Algorithm, key_len: usize) -> [u8; H] {
  assert!(H == spki_header_length(key_len), "an SPKI header for this key length");
  let mut out = [0; H];
  out[0] = TAG_SEQUENCE;
  let mut at = put_length(&mut out, 1, spki_contents_length(key_len));
  at = put(&mut out, at, &algorithm.identifier());
  out[at] = TAG_BIT_STRING;
  at = put_length(&mut out, at.strict_add(1), key_len.strict_add(1));
  // No unused bits.
  out[at] = 0x00;
  out
}

/// Return the raw public key if `der` is exactly `header` followed by `K` bytes.
///
/// `header` names one algorithm with absent parameters.
#[cfg(any(feature = "ed25519", feature = "ml-dsa", feature = "ml-kem", feature = "slh-dsa"))]
pub(crate) fn decode_spki<'a, E: KeyError, const H: usize, const K: usize>(
  der: &'a [u8],
  header: &[u8; H],
) -> Result<&'a [u8; K], E> {
  match der.split_first_chunk::<H>() {
    Some((prefix, key)) if prefix == header => key.try_into().map_err(|_| classify_spki::<E, K>(der, spki_oid(header))),
    _ => Err(classify_spki::<E, K>(der, spki_oid(header))),
  }
}

/// The OID contents of an SPKI header: after the outer tag and length, the
/// identifier's SEQUENCE header, and the OID tag, whose length octet follows.
#[cfg(any(feature = "ed25519", feature = "ml-dsa", feature = "ml-kem", feature = "slh-dsa"))]
fn spki_oid(header: &[u8]) -> &[u8] {
  let outer_length = match header.get(1) {
    Some(&first) if first >= 0x80 => usize::from(first & 0x7f).strict_add(1),
    _ => 1,
  };
  let length_at = outer_length.strict_add(4);
  let start = length_at.strict_add(1);
  let oid_length = header.get(length_at).map_or(0, |&length| usize::from(length));
  header.get(start..start.strict_add(oid_length)).unwrap_or_default()
}

/// Name the first failing component in order: DER structure, algorithm,
/// parameters, then key length.
#[cfg(any(feature = "ed25519", feature = "ml-dsa", feature = "ml-kem", feature = "slh-dsa"))]
fn classify_spki<E: KeyError, const K: usize>(der: &[u8], oid: &[u8]) -> E {
  let classified = (|| {
    let mut root = der::DerReader::<E>::new(der);
    let spki = root.read_constructed(TAG_SEQUENCE)?;
    root.finish()?;

    let mut spki = der::DerReader::<E>::new(spki);
    let algorithm = spki.read_constructed(TAG_SEQUENCE)?;
    let subject_public_key = spki.read_primitive(TAG_BIT_STRING)?;
    spki.finish()?;

    oid_without_parameters::<E>(algorithm, oid)?;

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

/// Check AlgorithmIdentifier contents that name `oid` with absent parameters.
#[cfg(any(feature = "ed25519", feature = "ml-dsa", feature = "ml-kem", feature = "slh-dsa"))]
fn oid_without_parameters<E: KeyError>(identifier: &[u8], oid: &[u8]) -> Result<(), E> {
  let mut identifier = der::DerReader::<E>::new(identifier);
  if identifier.read_primitive(TAG_OBJECT_IDENTIFIER)? != oid {
    return Err(E::UNSUPPORTED_ALGORITHM);
  }
  // Parameters must be absent.
  identifier.finish()
}

/// Borrowed private-key material from one OneAsymmetricKey.
#[cfg(any(feature = "ml-dsa", feature = "ml-kem"))]
pub(crate) enum PrivateKey<'a, const S: usize, const SK: usize> {
  Seed(&'a [u8; S]),
  Expanded(&'a [u8; SK]),
  Both { seed: &'a [u8; S], expanded: &'a [u8; SK] },
}

/// Fields of one OneAsymmetricKey, borrowed from the input.
#[cfg(any(feature = "ml-dsa", feature = "ml-kem"))]
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
#[cfg(any(feature = "ml-dsa", feature = "ml-kem"))]
pub(crate) fn decode_pkcs8<E: KeyError, const S: usize, const SK: usize, const PK: usize>(
  der: &[u8],
  algorithm: Algorithm,
) -> Result<Pkcs8<'_, S, SK, PK>, E> {
  let (private_key, public_key) = decode_one_asymmetric_key(
    der,
    |identifier| oid_without_parameters(identifier, &algorithm.oid()),
    private_key,
  )?;
  Ok(Pkcs8 {
    private_key,
    public_key,
  })
}

/// Parse a OneAsymmetricKey for `algorithm` whose `privateKey` holds exactly
/// the raw `SK`-byte private key, with no inner element (RFC 9909).
///
/// Returns the private key and the version 2 public key, whose consistency is
/// the caller's check. Rejections follow the structure in order.
#[cfg(feature = "slh-dsa")]
pub(crate) fn decode_pkcs8_raw<E: KeyError, const SK: usize, const PK: usize>(
  der: &[u8],
  algorithm: Algorithm,
) -> Result<(&[u8; SK], Option<&[u8; PK]>), E> {
  decode_one_asymmetric_key(
    der,
    |identifier| oid_without_parameters(identifier, &algorithm.oid()),
    fixed,
  )
}

/// Parse a OneAsymmetricKey for `oid`, with absent parameters, whose
/// `privateKey` wraps the raw `SK`-byte key in an inner OCTET STRING: RFC 8410
/// `CurvePrivateKey`.
///
/// Returns the private key and the version 2 public key, whose consistency is
/// the caller's check. Rejections follow the structure in order.
#[cfg(feature = "ed25519")]
pub(crate) fn decode_pkcs8_wrapped<'a, E: KeyError, const SK: usize, const PK: usize>(
  der: &'a [u8],
  oid: &[u8],
) -> Result<(&'a [u8; SK], Option<&'a [u8; PK]>), E> {
  decode_one_asymmetric_key(
    der,
    |identifier| oid_without_parameters(identifier, oid),
    |contents| {
      let mut wrapper = der::DerReader::<E>::new(contents);
      let key = fixed(wrapper.read_primitive(TAG_OCTET_STRING)?)?;
      wrapper.finish()?;
      Ok(key)
    },
  )
}

/// RFC 5915 `ECPrivateKey` fields, borrowed from the input.
#[cfg(any(feature = "ecdsa-p256", feature = "ecdsa-p384"))]
pub(crate) struct EcPrivateKey<'a, const N: usize, const PK: usize> {
  /// The big-endian private scalar; its range is the caller's check.
  pub(crate) scalar: &'a [u8; N],
  /// The optional public key; its consistency is the caller's check.
  pub(crate) public_key: Option<&'a [u8; PK]>,
}

/// Parse a OneAsymmetricKey for `algorithm` with the RFC 5480 `namedCurve`
/// parameters `curve`, whose `privateKey` holds an RFC 5915 `ECPrivateKey`
/// for the same curve with an `N`-byte scalar and a `PK`-byte public key.
///
/// Returns the `ECPrivateKey` and the version 2 container public key. The
/// consistency of each public key is the caller's check. Rejections follow
/// the structure in order.
#[cfg(any(feature = "ecdsa-p256", feature = "ecdsa-p384"))]
pub(crate) fn decode_pkcs8_ec<'a, E: KeyError, const N: usize, const PK: usize>(
  der: &'a [u8],
  algorithm: &[u8],
  curve: &[u8],
) -> Result<(EcPrivateKey<'a, N, PK>, Option<&'a [u8; PK]>), E> {
  decode_one_asymmetric_key(
    der,
    |identifier| {
      let mut identifier = der::DerReader::<E>::new(identifier);
      if identifier.read_primitive(TAG_OBJECT_IDENTIFIER)? != algorithm {
        return Err(E::UNSUPPORTED_ALGORITHM);
      }
      // RFC 5480 permits only the namedCurve choice.
      if identifier.read_primitive(TAG_OBJECT_IDENTIFIER)? != curve {
        return Err(E::UNSUPPORTED_ALGORITHM);
      }
      identifier.finish()
    },
    |contents| decode_ec_private_key(contents, curve),
  )
}

/// Parse an RFC 5915 `ECPrivateKey`: version 1, an `N`-byte private key,
/// optional parameters that must name `curve`, and an optional `PK`-byte
/// public key.
///
/// RFC 5915 requires the parameters, but PKCS #8 writers omit them because the
/// algorithm identifier already names the curve, so absence is accepted.
#[cfg(any(feature = "ecdsa-p256", feature = "ecdsa-p384"))]
pub(crate) fn decode_ec_private_key<'a, E: KeyError, const N: usize, const PK: usize>(
  der: &'a [u8],
  curve: &[u8],
) -> Result<EcPrivateKey<'a, N, PK>, E> {
  let mut root = der::DerReader::<E>::new(der);
  let fields = root.read_constructed(TAG_SEQUENCE)?;
  root.finish()?;

  let mut fields = der::DerReader::<E>::new(fields);
  // ecPrivkeyVer1.
  if fields.read_primitive(TAG_INTEGER)? != [1] {
    return Err(E::MALFORMED_DER);
  }
  let scalar = fixed(fields.read_primitive(TAG_OCTET_STRING)?)?;
  if fields.peek_byte() == Some(TAG_EC_PARAMETERS) {
    let mut parameters = der::DerReader::<E>::new(fields.read_constructed(TAG_EC_PARAMETERS)?);
    if parameters.read_primitive(TAG_OBJECT_IDENTIFIER)? != curve {
      return Err(E::UNSUPPORTED_ALGORITHM);
    }
    parameters.finish()?;
  }
  let public_key = if fields.peek_byte() == Some(TAG_EC_PUBLIC_KEY) {
    let mut wrapper = der::DerReader::<E>::new(fields.read_constructed(TAG_EC_PUBLIC_KEY)?);
    let key = public_key(wrapper.read_primitive(TAG_BIT_STRING)?)?;
    wrapper.finish()?;
    Some(key)
  } else {
    None
  };
  fields.finish()?;

  Ok(EcPrivateKey { scalar, public_key })
}

/// Parse the OneAsymmetricKey container: check the AlgorithmIdentifier
/// contents with `identifier`, then read the `privateKey` contents with
/// `private_key` at their position in the structure.
fn decode_one_asymmetric_key<'a, E: KeyError, P, const PK: usize>(
  der: &'a [u8],
  identifier: impl FnOnce(&'a [u8]) -> Result<(), E>,
  private_key: impl FnOnce(&'a [u8]) -> Result<P, E>,
) -> Result<(P, Option<&'a [u8; PK]>), E> {
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

  identifier(info.read_constructed(TAG_SEQUENCE)?)?;

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

  Ok((private_key, public_key))
}

/// Select the CHOICE variant by its tag, never by its length.
#[cfg(any(feature = "ml-dsa", feature = "ml-kem"))]
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
#[cfg(any(feature = "ml-dsa", feature = "ml-kem"))]
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
#[cfg(any(feature = "ml-dsa", feature = "ml-kem"))]
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

/// Version 1 header length before a `key_len`-byte raw private key.
#[cfg(feature = "slh-dsa")]
pub(crate) const fn raw_pkcs8_header_length(key_len: usize) -> usize {
  1usize
    .strict_add(length_octets(raw_pkcs8_contents_length(key_len)))
    .strict_add(3)
    .strict_add(IDENTIFIER_LENGTH)
    .strict_add(1)
    .strict_add(length_octets(key_len))
}

/// OneAsymmetricKey contents: the version, the AlgorithmIdentifier, and the
/// `privateKey` element.
#[cfg(feature = "slh-dsa")]
const fn raw_pkcs8_contents_length(key_len: usize) -> usize {
  3usize
    .strict_add(IDENTIFIER_LENGTH)
    .strict_add(1)
    .strict_add(length_octets(key_len))
    .strict_add(key_len)
}

/// Version 1 header for `algorithm` and a `key_len`-byte raw private key.
#[cfg(feature = "slh-dsa")]
pub(crate) const fn raw_pkcs8_header<const H: usize>(algorithm: Algorithm, key_len: usize) -> [u8; H] {
  assert!(
    H == raw_pkcs8_header_length(key_len),
    "a PKCS #8 header for this key length"
  );
  let mut out = [0; H];
  out[0] = TAG_SEQUENCE;
  let mut at = put_length(&mut out, 1, raw_pkcs8_contents_length(key_len));
  at = put(&mut out, at, &[TAG_INTEGER, 0x01, 0x00]);
  at = put(&mut out, at, &algorithm.identifier());
  out[at] = TAG_OCTET_STRING;
  at = put_length(&mut out, at.strict_add(1), key_len);
  assert!(at == H, "the header ends before the key");
  out
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
#[cfg(any(feature = "ed25519", feature = "ml-dsa", feature = "ml-kem"))]
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

/// Octets in the DER length of `len`.
#[cfg(any(feature = "ml-dsa", feature = "ml-kem", feature = "slh-dsa"))]
const fn length_octets(len: usize) -> usize {
  if len < 0x80 {
    1
  } else if len <= 0xff {
    2
  } else {
    assert!(len <= 0xffff, "a DER length with at most two length octets");
    3
  }
}

/// Write the DER length of `len` at `at`; return the next position.
#[cfg(any(feature = "ml-dsa", feature = "ml-kem", feature = "slh-dsa"))]
const fn put_length<const H: usize>(out: &mut [u8; H], at: usize, len: usize) -> usize {
  let [.., high, low] = len.to_be_bytes();
  match length_octets(len) {
    1 => put(out, at, &[low]),
    2 => put(out, at, &[0x81, low]),
    _ => put(out, at, &[LONG_LENGTH_2, high, low]),
  }
}

/// Copy `bytes` to `at`; return the next position.
#[cfg(any(feature = "ml-dsa", feature = "ml-kem", feature = "slh-dsa"))]
const fn put<const H: usize>(out: &mut [u8; H], at: usize, bytes: &[u8]) -> usize {
  let mut i = 0;
  while i < bytes.len() {
    out[at.strict_add(i)] = bytes[i];
    i = i.strict_add(1);
  }
  at.strict_add(bytes.len())
}

#[cfg(any(feature = "ml-dsa", feature = "ml-kem"))]
const fn short_length(len: usize) -> u8 {
  assert!(len < 0x80, "a short DER length");
  let [.., low] = len.to_be_bytes();
  low
}

#[cfg(any(feature = "ml-dsa", feature = "ml-kem"))]
const fn two_byte_length(len: usize) -> [u8; 2] {
  assert!(len >= 0x100 && len <= 0xffff, "a DER length with two length octets");
  let [.., high, low] = len.to_be_bytes();
  [high, low]
}
