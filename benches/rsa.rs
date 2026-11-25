//! RSA benchmarks for rscrypto public APIs: verification, private signing,
//! RSAES-OAEP encryption and decryption, and legacy RSAES-PKCS1-v1_5 decryption.
//!
//! Comparison case IDs have the shape `<operation path>/<implementation>`, and
//! each group measures one task. The `rscrypto` row calls rscrypto's best
//! public API for that task. `rscrypto-<variant>` rows call other rscrypto APIs
//! for the same task. Other rows are named for the library they call. Each
//! group's boundary comment states what is timed and any remaining API-cost
//! difference.
//!
//! The operation path of a randomized task names its randomness:
//!
//! - `fixed-entropy`: a caller blinding pair (factor 2) and, for PSS, a fixed
//!   salt; no random fill.
//! - `caller-entropy`: a caller fill closure, or a RustCrypto RNG, writes
//!   fixed bytes.
//! - `fresh-entropy`: every operation draws a new salt, seed, or blinding
//!   factor. rscrypto calls `getrandom`, and RustCrypto rows receive a
//!   `getrandom` RNG. aws-lc-rs takes no RNG and draws from AWS-LC's OS-seeded
//!   DRBG.
//! - `cached-blinding`: rscrypto signs through a reused `RsaSigningWorkspace`,
//!   which draws a new blinding factor from `getrandom` every 32 signatures
//!   and squares its pair in between. PSS still draws a new salt in every
//!   call.
//!
//! rscrypto private operations always blind. AWS-LC blinds by default, and
//! aws-lc-rs does not turn blinding off. RustCrypto private rows are named
//! `rustcrypto-rsa-blinded` because RustCrypto also offers unblinded
//! operations.
//!
//! Before timing, every row's output must match an independent oracle: the
//! output or verdict of another library, a Wycheproof vector, or RustCrypto
//! big-integer arithmetic. Every verifier must also reject one flipped bit.

#[path = "common/criterion.rs"]
mod bench_config;

#[path = "common/aws_lc.rs"]
mod aws_lc_support;
#[path = "../tests/common/mod.rs"]
mod fixture_hex;
use fixture_hex::decode_hex_vec;

use aws_lc_support::aws_lc_bench;

use core::hint::black_box;

aws_lc_bench! {
  use aws_lc_rs::{rsa as aws_rsa, signature as aws_signature};
}
use bench_config::{checked, checked_routine, checked_with};
use criterion::{Bencher, Criterion};
use ring::signature as ring_signature;
#[cfg(feature = "getrandom")]
use rsa::{
  Pkcs1v15Encrypt as RustCryptoPkcs1v15Encrypt,
  oaep::DecryptingKey as RustCryptoOaepDecryptingKey,
  traits::{Decryptor as _, RandomizedDecryptor as _},
};
use rsa::{
  PrivateKey as RustCryptoRsaPrivateKey, PublicKey as RustCryptoRsaPublicKey,
  oaep::EncryptingKey as RustCryptoOaepEncryptingKey,
  pkcs1::{DecodeRsaPrivateKey, DecodeRsaPublicKey, EncodeRsaPublicKey as _},
  pkcs1v15::{
    Signature as RustCryptoPkcs1v15Signature, SigningKey as RustCryptoPkcs1v15SigningKey,
    VerifyingKey as RustCryptoPkcs1v15VerifyingKey,
  },
  pkcs8::{DecodePrivateKey, DecodePublicKey},
  pss::{
    Signature as RustCryptoPssSignature, SigningKey as RustCryptoPssSigningKey,
    VerifyingKey as RustCryptoPssVerifyingKey,
  },
  rand_core::{TryCryptoRng as RustCryptoTryCryptoRng, TryRng as RustCryptoTryRng, utils::next_word_via_fill},
  signature::{RandomizedSigner as _, SignatureEncoding as _, Signer as _, Verifier as _},
  traits::RandomizedEncryptor as _,
};
#[cfg(feature = "getrandom")]
use rscrypto::auth::rsa::expert::RsaPkcs1v15Decrypt as _;
#[cfg(all(rscrypto_internal, feature = "diag"))]
use rscrypto::auth::rsa::{
  diag_rsa_blinding_factor_inverse_with_scratch, diag_rsa_public_operation_bitserial, diag_rsa_public_operation_cios,
  diag_rsa_public_operation_cios_portable, diag_rsa_public_operation_generic_exponent,
  diag_rsa_public_operation_product, diag_rsa_verify_pkcs1v15_encoded, diag_rsa_verify_pss_encoded_with_scratch,
};
use rscrypto::{
  Digest, RsaBlindingPair, RsaOaepProfile, RsaPkcs1v15Profile, RsaPrivateKey, RsaPssProfile, RsaPublicKey,
  RsaPublicKeyPolicy, RsaPublicScratch, RsaSignatureProfile, Sha256, Sha384, Sha512, VerificationError,
};

const MESSAGE_PSS: &[u8] = b"rscrypto RSA-PSS verification fixture";
const MESSAGE_PKCS1V15: &[u8] = b"rscrypto RSA-PKCS1-v1_5 verification fixture";

const RSA3072_SPKI: &[u8] = include_bytes!("../testdata/rsa/fixtures/rsa3072_spki.der");
const RSA3072_PSS_SHA256: &[u8] = include_bytes!("../testdata/rsa/fixtures/rsa3072_pss_sha256.sig");
const RSA3072_PKCS1V15_SHA256: &[u8] = include_bytes!("../testdata/rsa/fixtures/rsa3072_pkcs1v15_sha256.sig");
const RSA4096_SPKI: &[u8] = include_bytes!("../testdata/rsa/fixtures/rsa4096_spki.der");
const RSA4096_PSS_SHA256: &[u8] = include_bytes!("../testdata/rsa/fixtures/rsa4096_pss_sha256.sig");
const RSA4096_PKCS1V15_SHA256: &[u8] = include_bytes!("../testdata/rsa/fixtures/rsa4096_pkcs1v15_sha256.sig");
const RSA8192_SPKI: &[u8] = include_bytes!("../testdata/rsa/fixtures/rsa8192_spki.der");
const RSA8192_PSS_SHA256: &[u8] = include_bytes!("../testdata/rsa/fixtures/rsa8192_pss_sha256.sig");
const RSA8192_PKCS1V15_SHA256: &[u8] = include_bytes!("../testdata/rsa/fixtures/rsa8192_pkcs1v15_sha256.sig");

const PRIVATE_SIGNING_MESSAGE: &[u8] = b"rscrypto TLS-shaped RSA private signing benchmark";
/// Every byte of the fixed PSS salt.
const FIXED_SALT_BYTE: u8 = 0x5a;
/// Every byte of the caller OAEP seed.
const OAEP_SEED_BYTE: u8 = 0xa5;

// Wycheproof RSAES-OAEP-SHA256 (MGF1-SHA256) documents. Each holds one private
// key, which the OAEP rows use at every size and the 3072- and 4096-bit
// private-signing rows reuse.
const RSA2048_OAEP_SHA256: &str = include_str!("../testdata/rsa/wycheproof/rsa_oaep_2048_sha256_mgf1sha256_test.json");
const RSA3072_OAEP_SHA256: &str = include_str!("../testdata/rsa/wycheproof/rsa_oaep_3072_sha256_mgf1sha256_test.json");
const RSA4096_OAEP_SHA256: &str = include_str!("../testdata/rsa/wycheproof/rsa_oaep_4096_sha256_mgf1sha256_test.json");
// Wycheproof RSAES-PKCS1-v1_5 decryption documents.
const RSA2048_PKCS1V15_DECRYPT: &str = include_str!("../testdata/rsa/wycheproof/rsa_pkcs1_2048_test.json");
const RSA3072_PKCS1V15_DECRYPT: &str = include_str!("../testdata/rsa/wycheproof/rsa_pkcs1_3072_test.json");
const RSA4096_PKCS1V15_DECRYPT: &str = include_str!("../testdata/rsa/wycheproof/rsa_pkcs1_4096_test.json");
/// Wycheproof test with a 32-byte message and a valid ciphertext in each
/// OAEP-SHA256 and PKCS #1 v1.5 decryption document: a symmetric-key transport
/// shape. The OAEP vectors also have an empty label.
const DECRYPTION_VECTOR_TC_ID: u64 = 7;
const DECRYPTION_MESSAGE_LEN: usize = 32;
const OAEP_LABEL: &[u8] = b"";

#[cfg(feature = "getrandom")]
struct GetrandomRng;

#[cfg(feature = "getrandom")]
impl RustCryptoTryRng for GetrandomRng {
  type Error = getrandom::Error;

  fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
    next_word_via_fill(self)
  }

  fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
    next_word_via_fill(self)
  }

  fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Self::Error> {
    getrandom::fill(dest)
  }
}

#[cfg(feature = "getrandom")]
impl RustCryptoTryCryptoRng for GetrandomRng {}

/// RustCrypto RNG that repeats one byte. RustCrypto PSS draws its salt and
/// its blinding factor from the RNG, and RustCrypto OAEP draws only its seed.
/// Blinding does not change a signature, so this reproduces a fixed-salt
/// signature or a fixed-seed ciphertext.
struct RepeatedByteRng(u8);

impl RustCryptoTryRng for RepeatedByteRng {
  type Error = core::convert::Infallible;

  fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
    next_word_via_fill(self)
  }

  fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
    next_word_via_fill(self)
  }

  fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Self::Error> {
    dest.fill(self.0);
    Ok(())
  }
}

impl RustCryptoTryCryptoRng for RepeatedByteRng {}

fn der_len(len: usize) -> Vec<u8> {
  if len < 128 {
    return vec![u8::try_from(len).expect("short DER length must fit in one byte")];
  }

  let bytes = len.to_be_bytes();
  let first_nonzero = bytes
    .iter()
    .position(|&byte| byte != 0)
    .expect("long DER length must contain a non-zero byte");
  let len_bytes = &bytes[first_nonzero..];
  let mut out = Vec::with_capacity(1usize.strict_add(len_bytes.len()));
  out.push(0x80 | u8::try_from(len_bytes.len()).expect("DER length-of-length must fit in one byte"));
  out.extend_from_slice(len_bytes);
  out
}

fn tlv(tag: u8, value: &[u8]) -> Vec<u8> {
  let encoded_len = der_len(value.len());
  let capacity = 1usize.strict_add(encoded_len.len()).strict_add(value.len());
  let mut out = Vec::with_capacity(capacity);
  out.push(tag);
  out.extend_from_slice(&encoded_len);
  out.extend_from_slice(value);
  out
}

fn sequence(value: &[u8]) -> Vec<u8> {
  tlv(0x30, value)
}

fn integer_unsigned(value: &[u8]) -> Vec<u8> {
  let first_nonzero = value.iter().position(|&byte| byte != 0);
  let value = first_nonzero.map_or(&[0u8][..], |index| &value[index..]);
  let mut encoded = Vec::with_capacity(value.len().strict_add(usize::from(value[0] & 0x80 != 0)));
  if value[0] & 0x80 != 0 {
    encoded.push(0);
  }
  encoded.extend_from_slice(value);
  tlv(0x02, &encoded)
}

fn exponent_bytes(exponent: u64) -> Vec<u8> {
  let bytes = exponent.to_be_bytes();
  let first_nonzero = bytes
    .iter()
    .position(|&byte| byte != 0)
    .unwrap_or_else(|| bytes.len().strict_sub(1));
  bytes[first_nonzero..].to_vec()
}

/// `key` as a PKCS #1 `RSAPublicKey` whose exponent INTEGER has one extra
/// leading zero byte. DER requires minimal INTEGER encodings, so a strict
/// parser rejects it although it encodes the same key.
fn noncanonical_exponent_pkcs1_der(key: &RsaPublicKey) -> Vec<u8> {
  let exponent = exponent_bytes(key.public_exponent().as_u64());
  let mut noncanonical = vec![0u8];
  noncanonical.extend_from_slice(&exponent);
  let mut body = integer_unsigned(key.modulus());
  body.extend_from_slice(&tlv(0x02, &noncanonical));
  sequence(&body)
}

/// An `rsaEncryption` `SubjectPublicKeyInfo` around `pkcs1`.
fn spki_der_from_pkcs1(pkcs1: &[u8]) -> Vec<u8> {
  // AlgorithmIdentifier { rsaEncryption, NULL }.
  const RSA_ENCRYPTION: &[u8] = &[
    0x30, 0x0d, 0x06, 0x09, 0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x01, 0x05, 0x00,
  ];
  let mut bit_string = vec![0u8];
  bit_string.extend_from_slice(pkcs1);
  let mut body = RSA_ENCRYPTION.to_vec();
  body.extend_from_slice(&tlv(0x03, &bit_string));
  sequence(&body)
}

/// The PKCS #1 `RSAPublicKey` inside an SPKI, found by RustCrypto's DER decoder.
fn spki_subject_public_key(spki: &[u8]) -> Vec<u8> {
  rsa::pkcs8::SubjectPublicKeyInfoRef::try_from(spki)
    .expect("RustCrypto must decode the RSA SPKI fixture")
    .subject_public_key
    .raw_bytes()
    .to_vec()
}

fn pkcs1_der_from_modulus_exponent(modulus: &[u8], exponent: &[u8]) -> Vec<u8> {
  let mut body = Vec::new();
  body.extend_from_slice(&integer_unsigned(modulus));
  body.extend_from_slice(&integer_unsigned(exponent));
  sequence(&body)
}

#[cfg(all(rscrypto_internal, feature = "diag"))]
fn synthetic_pkcs1_der(modulus_len: usize) -> Vec<u8> {
  let mut modulus = vec![0xff; modulus_len];
  modulus[0] = 0x80;
  pkcs1_der_from_modulus_exponent(&modulus, &[0x01, 0x00, 0x01])
}

fn pss_spki() -> Vec<u8> {
  include_bytes!("../tests/vectors/rsa/rsa2048_pss_spki.der").to_vec()
}

fn pss_signature_sha256() -> Vec<u8> {
  include_bytes!("../tests/vectors/rsa/rsa2048_pss_signature_sha256.sig").to_vec()
}

fn pkcs1v15_spki() -> Vec<u8> {
  include_bytes!("../tests/vectors/rsa/rsa2048_pkcs1v15_spki.der").to_vec()
}

fn pkcs1v15_signature_sha256() -> Vec<u8> {
  include_bytes!("../tests/vectors/rsa/rsa2048_pkcs1v15_signature_sha256.sig").to_vec()
}

fn legacy_exponent_modulus() -> Vec<u8> {
  include_bytes!("../tests/vectors/rsa/rsa2048_legacy_exponent_modulus.bin").to_vec()
}

fn legacy_exponent_input() -> Vec<u8> {
  include_bytes!("../tests/vectors/rsa/rsa2048_legacy_exponent_input.bin").to_vec()
}

fn modulus_minus_one(key: &RsaPublicKey) -> Vec<u8> {
  let mut value = key.modulus().to_vec();
  for byte in value.iter_mut().rev() {
    if *byte != 0 {
      *byte = byte.strict_sub(1);
      break;
    }
    *byte = 0xff;
  }
  value
}

/// `input^e mod n` with num-bigint, left-padded to the modulus length: the
/// oracle for every public-operation row.
fn reference_public_operation(key: &RsaPublicKey, input: &[u8]) -> Vec<u8> {
  let modulus = key.modulus();
  let value = num_bigint::BigUint::from_bytes_be(input)
    .modpow(
      &num_bigint::BigUint::from(key.public_exponent().as_u64()),
      &num_bigint::BigUint::from_bytes_be(modulus),
    )
    .to_bytes_be();
  let mut out = vec![0u8; modulus.len().strict_sub(value.len())];
  out.extend_from_slice(&value);
  out
}

/// `bytes` with the low bit of its last byte flipped, for rejection checks.
fn tampered(bytes: &[u8]) -> Vec<u8> {
  let mut tampered = bytes.to_vec();
  let last = tampered.last_mut().expect("RSA benchmark fixtures are non-empty");
  *last ^= 1;
  tampered
}

fn rsa2048_private_key() -> RsaPrivateKey {
  RsaPrivateKey::from_pkcs1_der_with_policy(
    include_bytes!("../tests/vectors/rsa/rsa2048_private_pkcs1.der"),
    &RsaPublicKeyPolicy::modulus_2048_to_8192(),
  )
  .expect("the RSA-2048 benchmark key must parse")
}

/// RSA-2048 keys need the legacy policy; larger keys use the modern default.
fn key_policy(bits: usize) -> RsaPublicKeyPolicy {
  if bits < 3072 {
    RsaPublicKeyPolicy::modulus_2048_to_8192()
  } else {
    RsaPublicKeyPolicy::modulus_3072_to_8192()
  }
}

/// A Wycheproof RSA decryption private key and its valid vector
/// [`DECRYPTION_VECTOR_TC_ID`].
struct DecryptionFixture {
  private_key_pkcs8: Vec<u8>,
  message: Vec<u8>,
  ciphertext: Vec<u8>,
}

impl DecryptionFixture {
  /// The vector of an RSAES-OAEP document with SHA-256 for the label hash and MGF1.
  fn oaep(document: &str, bits: usize) -> Self {
    Self::parse(document, bits, "RsaesOaepDecrypt", |group, test| {
      assert_eq!(
        group["sha"], "SHA-256",
        "Wycheproof RSA-OAEP fixture has the wrong hash"
      );
      assert_eq!(
        group["mgfSha"], "SHA-256",
        "Wycheproof RSA-OAEP fixture has the wrong MGF1 hash"
      );
      assert_eq!(
        test["label"], "",
        "Wycheproof RSA-OAEP benchmark vector must have an empty label"
      );
    })
  }

  /// The vector of an RSAES-PKCS1-v1_5 decryption document.
  fn pkcs1v15(document: &str, bits: usize) -> Self {
    Self::parse(document, bits, "RsaesPkcs1Decrypt", |_, _| {})
  }

  fn parse(
    document: &str,
    bits: usize,
    group_type: &str,
    check: impl FnOnce(&serde_json::Value, &serde_json::Value),
  ) -> Self {
    let document: serde_json::Value =
      serde_json::from_str(document).expect("Wycheproof RSA decryption fixture must be valid JSON");
    let group = &document["testGroups"][0];
    assert_eq!(
      group["type"], group_type,
      "Wycheproof RSA decryption fixture has the wrong type"
    );
    assert_eq!(
      group["keySize"], bits,
      "Wycheproof RSA decryption fixture has the wrong size"
    );
    let test = group["tests"]
      .as_array()
      .expect("Wycheproof RSA decryption fixture must list tests")
      .iter()
      .find(|test| test["tcId"] == DECRYPTION_VECTOR_TC_ID)
      .expect("Wycheproof RSA decryption fixture must contain the benchmark vector");
    assert_eq!(
      test["result"], "valid",
      "Wycheproof RSA decryption benchmark vector must be valid"
    );
    check(group, test);
    let hex = |value: &serde_json::Value| {
      decode_hex_vec(
        value
          .as_str()
          .expect("Wycheproof RSA decryption fixture fields must be hex strings"),
      )
    };
    let message = hex(&test["msg"]);
    assert_eq!(
      message.len(),
      DECRYPTION_MESSAGE_LEN,
      "Wycheproof RSA decryption benchmark message length"
    );
    Self {
      private_key_pkcs8: hex(&group["privateKeyPkcs8"]),
      message,
      ciphertext: hex(&test["ct"]),
    }
  }

  fn private_key(&self, bits: usize) -> RsaPrivateKey {
    let key = RsaPrivateKey::from_pkcs8_der_with_policy(&self.private_key_pkcs8, &key_policy(bits))
      .expect("Wycheproof RSA private key must import");
    assert_eq!(key.public_key().modulus_bits(), bits);
    key
  }
}

fn factor_two_and_inverse(modulus: &[u8]) -> (Vec<u8>, Vec<u8>) {
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

fn rsa_private_signing(c: &mut Criterion) {
  rsa_private_signing_for_size(c, 2048, rsa2048_private_key);
  rsa_private_signing_for_size(c, 3072, || {
    DecryptionFixture::oaep(RSA3072_OAEP_SHA256, 3072).private_key(3072)
  });
  rsa_private_signing_for_size(c, 4096, || {
    DecryptionFixture::oaep(RSA4096_OAEP_SHA256, 4096).private_key(4096)
  });
}

/// Verify a PSS-SHA256 signature over [`PRIVATE_SIGNING_MESSAGE`] with rscrypto
/// and RustCrypto.
#[cfg(feature = "getrandom")]
fn assert_private_pss_signature(key: &RsaPrivateKey, rustcrypto: &RustCryptoRsaPublicKey, signature: &[u8]) {
  key
    .public_key()
    .verify_pss(RsaPssProfile::Sha256, PRIVATE_SIGNING_MESSAGE, signature)
    .expect("rscrypto must verify the RSA-PSS benchmark signature");
  RustCryptoPssVerifyingKey::<sha2::Sha256>::new(rustcrypto.clone())
    .verify(
      PRIVATE_SIGNING_MESSAGE,
      &RustCryptoPssSignature::try_from(signature).expect("RSA-PSS benchmark signature must have the modulus length"),
    )
    .expect("RustCrypto must verify the RSA-PSS benchmark signature");
}

fn rsa_private_signing_for_size(c: &mut Criterion, bits: usize, load_key: impl FnOnce() -> RsaPrivateKey) {
  let name = format!("rsa-{bits}-private-signing");
  if !bench_config::selected(&format!("{name}/")) {
    return;
  }
  let key = load_key();
  let salt = [FIXED_SALT_BYTE; Sha256::OUTPUT_SIZE];
  let (factor, inverse) = factor_two_and_inverse(key.public_key().modulus());
  // PKCS #1 `RSAPrivateKey` DER that each comparison library imports.
  let private_key_der = key.to_pkcs1_der();
  let rustcrypto_private_key = {
    let mut private_key = RustCryptoRsaPrivateKey::from_pkcs1_der(&private_key_der)
      .expect("RustCrypto must import the RSA private benchmark fixture");
    private_key
      .precompute()
      .expect("RustCrypto must precompute the RSA private benchmark fixture");
    private_key
  };
  #[cfg(feature = "getrandom")]
  let rustcrypto_public_key = rustcrypto_private_key.to_public_key();
  let rustcrypto_pkcs1v15_key = RustCryptoPkcs1v15SigningKey::<sha2::Sha256>::new(rustcrypto_private_key.clone());
  let rustcrypto_pss_key = RustCryptoPssSigningKey::<sha2::Sha256>::new(rustcrypto_private_key);

  // Untimed oracles. RustCrypto PSS draws its salt and its blinding factor
  // from the repeated-byte RNG, so it reproduces the signature of all four
  // fixed-salt rscrypto rows; blinding does not change a signature. PKCS #1
  // v1.5 is deterministic. Randomized signatures must verify with rscrypto and
  // RustCrypto.
  let pss_with_fixed_salt = || {
    rustcrypto_pss_key
      .try_sign_with_rng(&mut RepeatedByteRng(FIXED_SALT_BYTE), PRIVATE_SIGNING_MESSAGE)
      .expect("RustCrypto RSA-PSS signing must succeed")
      .to_vec()
  };
  let pkcs1v15 = || {
    rustcrypto_pkcs1v15_key
      .try_sign(PRIVATE_SIGNING_MESSAGE)
      .expect("RustCrypto RSA-PKCS1-v1_5 signing must succeed")
      .to_vec()
  };
  // Each row owns its signature buffer and scratch, allocated outside timing.
  let buffers = || (vec![0u8; key.signature_len()], key.private_scratch());

  // rscrypto-only rows: blinding-factor inversion (diagnostic) and private
  // scratch allocation, which the `rscrypto` signing rows reuse.
  let mut group = c.benchmark_group(&name);
  #[cfg(all(rscrypto_internal, feature = "diag"))]
  group.bench_function(
    "blinding-inverse-scratch-rscrypto",
    checked_with(
      buffers(),
      |(computed_inverse, inverse_scratch)| {
        // The factor doubles as a valid inverse mask; production draws a fresh one.
        diag_rsa_blinding_factor_inverse_with_scratch(
          black_box(&key),
          black_box(&factor),
          black_box(&factor),
          black_box(computed_inverse),
          black_box(inverse_scratch),
        )
        .expect("valid scratch-backed RSA blinding-factor inversion must succeed")
      },
      // `factor_two_and_inverse` computes the inverse of 2 as (n + 1) / 2.
      |(), (computed_inverse, _)| assert_eq!(*computed_inverse, inverse),
    ),
  );
  group.bench_function(
    "scratch-setup-rscrypto",
    checked(
      || black_box(key.private_scratch()),
      |mut scratch| {
        let mut signature = vec![0u8; key.signature_len()];
        key
          .sign_pkcs1v15_with_blinding_factor_and_scratch(
            RsaPkcs1v15Profile::Sha256,
            PRIVATE_SIGNING_MESSAGE,
            RsaBlindingPair::new(&factor, &inverse),
            &mut signature,
            &mut scratch,
          )
          .expect("a fresh RSA private scratch must serve signing");
        assert_eq!(signature, pkcs1v15());
      },
    ),
  );
  group.finish();

  // Signing groups time message hashing, padding, blinding, the CRT private
  // operation, and the output write. `rscrypto` reuses a private scratch;
  // `rscrypto-oneshot` allocates one per call.
  //
  // PSS-SHA256 with a caller salt and blinding pair. rscrypto-only: no
  // comparison library accepts a caller blinding pair. The signature must
  // match RustCrypto PSS with the same salt.
  let mut group = c.benchmark_group(format!("{name}/sign-pss-sha256/fixed-entropy"));
  group.bench_function(
    "rscrypto",
    checked_with(
      buffers(),
      |(signature, scratch)| {
        key
          .sign_pss_with_salt_and_blinding_factor_and_scratch(
            RsaPssProfile::Sha256,
            black_box(PRIVATE_SIGNING_MESSAGE),
            black_box(&salt),
            RsaBlindingPair::new(black_box(&factor), black_box(&inverse)),
            black_box(signature),
            black_box(scratch),
          )
          .expect("valid scratch-backed RSA-PSS benchmark signing must succeed")
      },
      |(), (signature, _)| assert_eq!(*signature, pss_with_fixed_salt()),
    ),
  );
  group.bench_function(
    "rscrypto-oneshot",
    checked_with(
      vec![0u8; key.signature_len()],
      |signature| {
        key
          .sign_pss_with_salt_and_blinding_factor(
            RsaPssProfile::Sha256,
            black_box(PRIVATE_SIGNING_MESSAGE),
            black_box(&salt),
            RsaBlindingPair::new(black_box(&factor), black_box(&inverse)),
            black_box(signature),
          )
          .expect("valid one-shot RSA-PSS benchmark signing must succeed")
      },
      |(), signature| assert_eq!(*signature, pss_with_fixed_salt()),
    ),
  );
  group.finish();

  // PSS-SHA256 whose fill closure copies the fixed salt and blinding factor
  // 2; rscrypto then computes the inverse inside timing. rscrypto-only:
  // RustCrypto draws its blinding factor from an RNG and computes different
  // blinding work, and aws-lc-rs and ring take no caller entropy. The
  // signature must match RustCrypto PSS with the same salt.
  let mut group = c.benchmark_group(format!("{name}/sign-pss-sha256/caller-entropy"));
  group.bench_function(
    "rscrypto",
    checked_with(
      buffers(),
      |(signature, scratch)| {
        key
          .sign_pss_with_random_fill_and_scratch(
            RsaPssProfile::Sha256,
            black_box(PRIVATE_SIGNING_MESSAGE),
            black_box(signature),
            black_box(scratch),
            |out| {
              if out.len() == salt.len() {
                out.copy_from_slice(black_box(&salt));
              } else {
                out.copy_from_slice(black_box(&factor));
              }
              Ok::<(), ()>(())
            },
          )
          .expect("valid caller-random scratch-backed RSA-PSS benchmark signing must succeed")
      },
      |(), (signature, _)| assert_eq!(*signature, pss_with_fixed_salt()),
    ),
  );
  group.bench_function(
    "rscrypto-oneshot",
    checked_with(
      vec![0u8; key.signature_len()],
      |signature| {
        key
          .sign_pss_with_random_fill(
            RsaPssProfile::Sha256,
            black_box(PRIVATE_SIGNING_MESSAGE),
            black_box(signature),
            |out| {
              if out.len() == salt.len() {
                out.copy_from_slice(black_box(&salt));
              } else {
                out.copy_from_slice(black_box(&factor));
              }
              Ok::<(), ()>(())
            },
          )
          .expect("valid caller-random one-shot RSA-PSS benchmark signing must succeed")
      },
      |(), signature| assert_eq!(*signature, pss_with_fixed_salt()),
    ),
  );
  group.finish();

  // PKCS #1 v1.5-SHA256 with a caller blinding pair. rscrypto-only for the
  // same reason as PSS `fixed-entropy`. The signature must match RustCrypto's
  // deterministic signature.
  let mut group = c.benchmark_group(format!("{name}/sign-pkcs1v15-sha256/fixed-entropy"));
  group.bench_function(
    "rscrypto",
    checked_with(
      buffers(),
      |(signature, scratch)| {
        key
          .sign_pkcs1v15_with_blinding_factor_and_scratch(
            RsaPkcs1v15Profile::Sha256,
            black_box(PRIVATE_SIGNING_MESSAGE),
            RsaBlindingPair::new(black_box(&factor), black_box(&inverse)),
            black_box(signature),
            black_box(scratch),
          )
          .expect("valid scratch-backed RSA-PKCS1-v1_5 benchmark signing must succeed")
      },
      |(), (signature, _)| assert_eq!(*signature, pkcs1v15()),
    ),
  );
  group.finish();

  // PSS-SHA256 with a fresh salt per call. Every library imports its key pair
  // outside timing.
  // - rscrypto draws the salt and blinding factor from `getrandom`.
  // - RustCrypto's `SigningKey` draws both from a `getrandom` RNG,
  //   uses its precomputed CRT values, and returns a new signature allocation
  //   inside timing.
  // - aws-lc-rs `RsaKeyPair::sign` ignores its RNG argument: AWS-LC draws the
  //   salt from its DRBG and blinds with its cached blinding pair (see the
  //   OAEP decryption groups).
  // - ring `RsaKeyPair::sign` draws the salt from `SystemRandom` and does not
  //   blind; ring relies on its constant-time arithmetic instead, so it does
  //   less work per call.
  // rscrypto and RustCrypto must both verify every signature.
  #[cfg(feature = "getrandom")]
  {
    aws_lc_bench! {
      let aws_key_pair = aws_signature::RsaKeyPair::from_der(&private_key_der)
        .expect("aws-lc-rs must import the RSA private benchmark fixture");
      let aws_rng = aws_lc_rs::rand::SystemRandom::new();
    }
    let ring_key_pair = ring_signature::RsaKeyPair::from_der(&private_key_der)
      .expect("ring must import the RSA private benchmark fixture");
    let ring_rng = ring::rand::SystemRandom::new();

    let mut group = c.benchmark_group(format!("{name}/sign-pss-sha256/fresh-entropy"));
    group.bench_function(
      "rscrypto",
      checked_with(
        buffers(),
        |(signature, scratch)| {
          key
            .sign_pss_with_scratch(
              RsaPssProfile::Sha256,
              black_box(PRIVATE_SIGNING_MESSAGE),
              black_box(signature),
              black_box(scratch),
            )
            .expect("valid OS-random scratch-backed RSA-PSS benchmark signing must succeed")
        },
        |(), (signature, _)| assert_private_pss_signature(&key, &rustcrypto_public_key, signature),
      ),
    );
    group.bench_function(
      "rscrypto-oneshot",
      checked_with(
        vec![0u8; key.signature_len()],
        |signature| {
          key
            .sign_pss(
              RsaPssProfile::Sha256,
              black_box(PRIVATE_SIGNING_MESSAGE),
              black_box(signature),
            )
            .expect("valid OS-random one-shot RSA-PSS benchmark signing must succeed")
        },
        |(), signature| assert_private_pss_signature(&key, &rustcrypto_public_key, signature),
      ),
    );
    group.bench_function(
      "rustcrypto-rsa-blinded",
      checked(
        || {
          black_box(&rustcrypto_pss_key)
            .try_sign_with_rng(&mut GetrandomRng, black_box(PRIVATE_SIGNING_MESSAGE))
            .expect("valid RustCrypto RSA-PSS benchmark signing must succeed")
        },
        |signature| assert_private_pss_signature(&key, &rustcrypto_public_key, &signature.to_vec()),
      ),
    );
    aws_lc_bench! {
      group.bench_function(
        "aws-lc-rs",
        checked_with(
          vec![0u8; key.signature_len()],
          |signature| {
            aws_key_pair
              .sign(
                &aws_signature::RSA_PSS_SHA256,
                &aws_rng,
                black_box(PRIVATE_SIGNING_MESSAGE),
                black_box(signature),
              )
              .expect("valid aws-lc-rs RSA-PSS benchmark signing must succeed")
          },
          |(), signature| assert_private_pss_signature(&key, &rustcrypto_public_key, signature),
        ),
      );
    }
    group.bench_function(
      "ring",
      checked_with(
        vec![0u8; key.signature_len()],
        |signature| {
          ring_key_pair
            .sign(
              &ring_signature::RSA_PSS_SHA256,
              &ring_rng,
              black_box(PRIVATE_SIGNING_MESSAGE),
              black_box(signature),
            )
            .expect("valid ring RSA-PSS benchmark signing must succeed")
        },
        |(), signature| assert_private_pss_signature(&key, &rustcrypto_public_key, signature),
      ),
    );
    group.finish();

    // PKCS #1 v1.5-SHA256 with fresh blinding per call. Every library imports
    // its key pair outside timing. rscrypto draws and inverts a blinding factor
    // from `getrandom`; RustCrypto's `RandomizedSigner` blinds with a
    // `getrandom` RNG and returns a new signature allocation inside timing.
    // AWS-LC blinds with its cached blinding pair, and ring does not blind. The
    // padding is deterministic, so every signature must equal RustCrypto's
    // deterministic signature; the RustCrypto row must equal ring's.
    let mut group = c.benchmark_group(format!("{name}/sign-pkcs1v15-sha256/fresh-entropy"));
    group.bench_function(
      "rscrypto",
      checked_with(
        buffers(),
        |(signature, scratch)| {
          key
            .sign_pkcs1v15_with_scratch(
              RsaPkcs1v15Profile::Sha256,
              black_box(PRIVATE_SIGNING_MESSAGE),
              black_box(signature),
              black_box(scratch),
            )
            .expect("valid OS-random scratch-backed RSA-PKCS1-v1_5 benchmark signing must succeed")
        },
        |(), (signature, _)| assert_eq!(*signature, pkcs1v15()),
      ),
    );
    group.bench_function(
      "rscrypto-oneshot",
      checked_with(
        vec![0u8; key.signature_len()],
        |signature| {
          key
            .sign_pkcs1v15(
              RsaPkcs1v15Profile::Sha256,
              black_box(PRIVATE_SIGNING_MESSAGE),
              black_box(signature),
            )
            .expect("valid OS-random one-shot RSA-PKCS1-v1_5 benchmark signing must succeed")
        },
        |(), signature| assert_eq!(*signature, pkcs1v15()),
      ),
    );
    let ring_pkcs1v15 = {
      let mut signature = vec![0u8; key.signature_len()];
      ring_key_pair
        .sign(
          &ring_signature::RSA_PKCS1_SHA256,
          &ring_rng,
          PRIVATE_SIGNING_MESSAGE,
          &mut signature,
        )
        .expect("ring RSA-PKCS1-v1_5 signing must succeed");
      signature
    };
    group.bench_function(
      "rustcrypto-rsa-blinded",
      checked(
        || {
          black_box(&rustcrypto_pkcs1v15_key)
            .try_sign_with_rng(&mut GetrandomRng, black_box(PRIVATE_SIGNING_MESSAGE))
            .expect("valid RustCrypto RSA-PKCS1-v1_5 benchmark signing must succeed")
        },
        |signature| assert_eq!(signature.to_vec(), ring_pkcs1v15),
      ),
    );
    aws_lc_bench! {
      group.bench_function(
        "aws-lc-rs",
        checked_with(
          vec![0u8; key.signature_len()],
          |signature| {
            aws_key_pair
              .sign(
                &aws_signature::RSA_PKCS1_SHA256,
                &aws_rng,
                black_box(PRIVATE_SIGNING_MESSAGE),
                black_box(signature),
              )
              .expect("valid aws-lc-rs RSA-PKCS1-v1_5 benchmark signing must succeed")
          },
          |(), signature| assert_eq!(*signature, pkcs1v15()),
        ),
      );
    }
    group.bench_function(
      "ring",
      checked_with(
        vec![0u8; key.signature_len()],
        |signature| {
          ring_key_pair
            .sign(
              &ring_signature::RSA_PKCS1_SHA256,
              &ring_rng,
              black_box(PRIVATE_SIGNING_MESSAGE),
              black_box(signature),
            )
            .expect("valid ring RSA-PKCS1-v1_5 benchmark signing must succeed")
        },
        |(), signature| assert_eq!(*signature, pkcs1v15()),
      ),
    );
    group.finish();

    // PSS-SHA256 and PKCS #1 v1.5-SHA256 through one reused
    // `RsaSigningWorkspace` per row. PSS draws a fresh salt from `getrandom` in
    // every call. The workspace draws a new blinding factor from `getrandom` every
    // 32 signatures and squares its pair in between, so the time per call
    // includes one refresh amortized over each epoch. AWS-LC caches and
    // squares its blinding pair on the same schedule in the `fresh-entropy`
    // groups. rscrypto-only. PSS signatures must verify with rscrypto and
    // RustCrypto; PKCS #1 v1.5 signatures must equal RustCrypto's
    // deterministic signature.
    let cached_buffers = || (vec![0u8; key.signature_len()], key.signing_workspace());
    let mut group = c.benchmark_group(format!("{name}/sign-pss-sha256/cached-blinding"));
    group.bench_function(
      "rscrypto",
      checked_with(
        cached_buffers(),
        |(signature, workspace)| {
          key
            .sign_signature_with_workspace(
              RsaSignatureProfile::pss(RsaPssProfile::Sha256),
              black_box(PRIVATE_SIGNING_MESSAGE),
              black_box(signature),
              black_box(workspace),
            )
            .expect("valid cached-blinding RSA-PSS benchmark signing must succeed")
        },
        |(), (signature, _)| assert_private_pss_signature(&key, &rustcrypto_public_key, signature),
      ),
    );
    group.finish();

    let mut group = c.benchmark_group(format!("{name}/sign-pkcs1v15-sha256/cached-blinding"));
    group.bench_function(
      "rscrypto",
      checked_with(
        cached_buffers(),
        |(signature, workspace)| {
          key
            .sign_signature_with_workspace(
              RsaSignatureProfile::pkcs1v15(RsaPkcs1v15Profile::Sha256),
              black_box(PRIVATE_SIGNING_MESSAGE),
              black_box(signature),
              black_box(workspace),
            )
            .expect("valid cached-blinding RSA-PKCS1-v1_5 benchmark signing must succeed")
        },
        |(), (signature, _)| assert_eq!(*signature, pkcs1v15()),
      ),
    );
    group.finish();
  }
}

fn rsa_oaep(c: &mut Criterion) {
  rsa_oaep_for_size(c, 2048, RSA2048_OAEP_SHA256);
  rsa_oaep_for_size(c, 3072, RSA3072_OAEP_SHA256);
  rsa_oaep_for_size(c, 4096, RSA4096_OAEP_SHA256);
}

/// rscrypto RSA-OAEP-SHA256 decryption with the fixed blinding pair, for
/// untimed checks of other libraries' ciphertexts.
fn rscrypto_oaep_plaintext(key: &RsaPrivateKey, ciphertext: &[u8]) -> Vec<u8> {
  let (factor, inverse) = factor_two_and_inverse(key.public_key().modulus());
  let mut out = vec![0u8; key.signature_len()];
  let len = key
    .decrypt_oaep_with_blinding_factor(
      RsaOaepProfile::Sha256,
      OAEP_LABEL,
      ciphertext,
      RsaBlindingPair::new(&factor, &inverse),
      &mut out,
    )
    .expect("rscrypto must decrypt the RSA-OAEP benchmark ciphertext");
  out.truncate(len);
  out
}

/// RSAES-OAEP with SHA-256 for the label hash and MGF1, an empty label, and
/// the 32-byte Wycheproof message. Keys are imported outside timing.
///
/// Untimed checks: every decryption recovers the Wycheproof message. Each
/// caller-seed ciphertext matches the other library's ciphertext with the same
/// seed, and every fresh-entropy ciphertext decrypts to the message with
/// another library.
fn rsa_oaep_for_size(c: &mut Criterion, bits: usize, document: &str) {
  let name = format!("rsa-{bits}-oaep-sha256");
  if !bench_config::selected(&format!("{name}/")) {
    return;
  }
  let fixture = DecryptionFixture::oaep(document, bits);
  let message = fixture.message.as_slice();
  let key = fixture.private_key(bits);
  let public_key = key.public_key();
  let modulus_len = public_key.modulus().len();
  let (factor, inverse) = factor_two_and_inverse(public_key.modulus());
  let seed = [OAEP_SEED_BYTE; Sha256::OUTPUT_SIZE];

  let rustcrypto_private_key = RustCryptoRsaPrivateKey::from_pkcs8_der(&fixture.private_key_pkcs8)
    .expect("RustCrypto must import the Wycheproof RSA private key");
  let rustcrypto_encrypting_key =
    RustCryptoOaepEncryptingKey::<sha2::Sha256>::new(rustcrypto_private_key.to_public_key());
  #[cfg(feature = "getrandom")]
  let rustcrypto_decrypting_key = RustCryptoOaepDecryptingKey::<sha2::Sha256>::new(rustcrypto_private_key);
  #[cfg(feature = "getrandom")]
  let rustcrypto_decrypts = |ciphertext: &[u8]| {
    assert_eq!(
      rustcrypto_decrypting_key
        .decrypt(ciphertext)
        .expect("RustCrypto must decrypt the rscrypto RSA-OAEP ciphertext"),
      message,
      "RustCrypto decryption of an rscrypto RSA-OAEP ciphertext"
    );
  };
  aws_lc_bench! {
    let aws_encrypting_key = aws_rsa::OaepPublicEncryptingKey::new(
      aws_rsa::PublicEncryptingKey::from_der(&public_key.to_spki_der())
        .expect("aws-lc-rs must import the RSA public key"),
    )
    .expect("aws-lc-rs must build the RSA-OAEP public key");
    let aws_decrypting_key = aws_rsa::OaepPrivateDecryptingKey::new(
      aws_rsa::PrivateDecryptingKey::from_pkcs8(&fixture.private_key_pkcs8)
        .expect("aws-lc-rs must import the Wycheproof RSA private key"),
    )
    .expect("aws-lc-rs must build the RSA-OAEP private key");
  }

  // Encryption with a caller seed: OAEP encoding, the seed write, and the
  // public operation. rscrypto's fill closure copies the seed; rscrypto reuses
  // scratch and writes the ciphertext into a caller buffer. RustCrypto
  // receives an RNG that writes the same seed bytes, and it returns a new
  // `Vec` inside timing. aws-lc-rs takes no caller seed, and ring has no RSA
  // encryption.
  let mut group = c.benchmark_group(format!("{name}/encrypt/caller-entropy"));
  group.bench_function(
    "rscrypto",
    checked_with(
      (vec![0u8; modulus_len], public_key.public_scratch()),
      |(ciphertext, scratch)| {
        public_key
          .encrypt_oaep_with_random_fill_and_scratch(
            RsaOaepProfile::Sha256,
            black_box(OAEP_LABEL),
            black_box(message),
            black_box(ciphertext),
            black_box(scratch),
            |out| {
              out.copy_from_slice(black_box(&seed));
              Ok(())
            },
          )
          .expect("valid caller-random RSA-OAEP benchmark encryption must succeed")
      },
      |(), (ciphertext, _)| {
        let replayed = rustcrypto_encrypting_key
          .encrypt_with_rng(&mut RepeatedByteRng(OAEP_SEED_BYTE), message)
          .expect("RustCrypto RSA-OAEP encryption must succeed");
        assert_eq!(*ciphertext, replayed, "RSA-OAEP ciphertexts from the same seed");
      },
    ),
  );
  group.bench_function(
    "rustcrypto-rsa",
    checked(
      || {
        black_box(&rustcrypto_encrypting_key)
          .encrypt_with_rng(&mut RepeatedByteRng(black_box(OAEP_SEED_BYTE)), black_box(message))
          .expect("valid RustCrypto RSA-OAEP benchmark encryption must succeed")
      },
      |ciphertext| {
        let mut replayed = vec![0u8; modulus_len];
        public_key
          .encrypt_oaep_with_random_fill(RsaOaepProfile::Sha256, OAEP_LABEL, message, &mut replayed, |out| {
            out.copy_from_slice(&seed);
            Ok(())
          })
          .expect("rscrypto RSA-OAEP encryption must succeed");
        assert_eq!(ciphertext, replayed, "RSA-OAEP ciphertexts from the same seed");
      },
    ),
  );
  group.finish();

  // Encryption with a fresh seed per call: OAEP encoding, the seed draw, and
  // the public operation. See the module documentation for each library's
  // entropy source. aws-lc-rs builds an `EVP_PKEY_CTX` and RustCrypto returns
  // a new `Vec` on every call; both stay timed because they are part of those
  // APIs. `rscrypto` reuses scratch; `rscrypto-oneshot` allocates it per call.
  // ring has no RSA encryption.
  let mut group = c.benchmark_group(format!("{name}/encrypt/fresh-entropy"));
  #[cfg(feature = "getrandom")]
  group.bench_function(
    "rscrypto",
    checked_with(
      (vec![0u8; modulus_len], public_key.public_scratch()),
      |(ciphertext, scratch)| {
        public_key
          .encrypt_oaep_with_scratch(
            RsaOaepProfile::Sha256,
            black_box(OAEP_LABEL),
            black_box(message),
            black_box(ciphertext),
            black_box(scratch),
          )
          .expect("valid OS-random scratch-backed RSA-OAEP benchmark encryption must succeed")
      },
      |(), (ciphertext, _)| rustcrypto_decrypts(ciphertext),
    ),
  );
  #[cfg(feature = "getrandom")]
  group.bench_function(
    "rscrypto-oneshot",
    checked_with(
      vec![0u8; modulus_len],
      |ciphertext| {
        public_key
          .encrypt_oaep(
            RsaOaepProfile::Sha256,
            black_box(OAEP_LABEL),
            black_box(message),
            black_box(ciphertext),
          )
          .expect("valid OS-random one-shot RSA-OAEP benchmark encryption must succeed")
      },
      |(), ciphertext| rustcrypto_decrypts(ciphertext),
    ),
  );
  #[cfg(feature = "getrandom")]
  group.bench_function(
    "rustcrypto-rsa",
    checked(
      || {
        black_box(&rustcrypto_encrypting_key)
          .encrypt_with_rng(&mut GetrandomRng, black_box(message))
          .expect("valid RustCrypto RSA-OAEP benchmark encryption must succeed")
      },
      |ciphertext| {
        assert_eq!(
          rscrypto_oaep_plaintext(&key, &ciphertext),
          message,
          "rscrypto decryption of a RustCrypto RSA-OAEP ciphertext"
        )
      },
    ),
  );
  aws_lc_bench! {
    group.bench_function(
      "aws-lc-rs",
      checked_with(
        vec![0u8; aws_encrypting_key.ciphertext_size()],
        |ciphertext| {
          aws_encrypting_key
            .encrypt(
              &aws_rsa::OAEP_SHA256_MGF1SHA256,
              black_box(message),
              black_box(ciphertext),
              None,
            )
            .expect("valid aws-lc-rs RSA-OAEP benchmark encryption must succeed")
            .len()
        },
        |len, ciphertext| {
          assert_eq!(len, modulus_len, "aws-lc-rs RSA-OAEP ciphertext length");
          assert_eq!(
            rscrypto_oaep_plaintext(&key, ciphertext),
            message,
            "rscrypto decryption of an aws-lc-rs RSA-OAEP ciphertext"
          )
        },
      ),
    );
  }
  group.finish();

  // Decryption of the Wycheproof ciphertext with the caller blinding pair:
  // the blinded private operation and OAEP decoding, with reused scratch and
  // output. rscrypto-only: no comparison library accepts a blinding pair.
  let mut group = c.benchmark_group(format!("{name}/decrypt/fixed-entropy"));
  group.bench_function(
    "rscrypto",
    checked_with(
      (vec![0u8; modulus_len], key.private_scratch()),
      |(plaintext, scratch)| {
        key
          .decrypt_oaep_with_blinding_factor_and_scratch(
            RsaOaepProfile::Sha256,
            black_box(OAEP_LABEL),
            black_box(&fixture.ciphertext),
            RsaBlindingPair::new(black_box(&factor), black_box(&inverse)),
            black_box(plaintext),
            black_box(scratch),
          )
          .expect("valid fixed-blinding RSA-OAEP benchmark decryption must succeed")
      },
      |len, (plaintext, _)| assert_eq!(&plaintext[..len], message),
    ),
  );
  group.finish();

  // Decryption of the Wycheproof ciphertext with fresh blinding: blinding, the
  // private operation, and OAEP decoding. rscrypto and RustCrypto draw and
  // invert a new blinding factor on every call. AWS-LC caches a blinding pair
  // per key, squares it between uses, and draws a new one every 32 uses: a
  // real cost difference of that design. RustCrypto returns a new `Vec`, and
  // aws-lc-rs builds an `EVP_PKEY_CTX` on every call; both stay timed.
  // `rscrypto` reuses scratch; `rscrypto-oneshot` allocates it per call. ring
  // has no RSA decryption.
  let mut group = c.benchmark_group(format!("{name}/decrypt/fresh-entropy"));
  #[cfg(feature = "getrandom")]
  group.bench_function(
    "rscrypto",
    checked_with(
      (vec![0u8; modulus_len], key.private_scratch()),
      |(plaintext, scratch)| {
        key
          .decrypt_oaep_with_scratch(
            RsaOaepProfile::Sha256,
            black_box(OAEP_LABEL),
            black_box(&fixture.ciphertext),
            black_box(plaintext),
            black_box(scratch),
          )
          .expect("valid OS-random scratch-backed RSA-OAEP benchmark decryption must succeed")
      },
      |len, (plaintext, _)| assert_eq!(&plaintext[..len], message),
    ),
  );
  #[cfg(feature = "getrandom")]
  group.bench_function(
    "rscrypto-oneshot",
    checked_with(
      vec![0u8; modulus_len],
      |plaintext| {
        key
          .decrypt_oaep(
            RsaOaepProfile::Sha256,
            black_box(OAEP_LABEL),
            black_box(&fixture.ciphertext),
            black_box(plaintext),
          )
          .expect("valid OS-random one-shot RSA-OAEP benchmark decryption must succeed")
      },
      |len, plaintext| assert_eq!(&plaintext[..len], message),
    ),
  );
  #[cfg(feature = "getrandom")]
  group.bench_function(
    "rustcrypto-rsa-blinded",
    checked(
      || {
        black_box(&rustcrypto_decrypting_key)
          .decrypt_with_rng(&mut GetrandomRng, black_box(&fixture.ciphertext))
          .expect("valid RustCrypto RSA-OAEP benchmark decryption must succeed")
      },
      |plaintext| assert_eq!(plaintext, message),
    ),
  );
  aws_lc_bench! {
    group.bench_function(
      "aws-lc-rs",
      checked_with(
        vec![0u8; aws_decrypting_key.min_output_size()],
        |plaintext| {
          aws_decrypting_key
            .decrypt(
              &aws_rsa::OAEP_SHA256_MGF1SHA256,
              black_box(&fixture.ciphertext),
              black_box(plaintext),
              None,
            )
            .expect("valid aws-lc-rs RSA-OAEP benchmark decryption must succeed")
            .len()
        },
        |len, plaintext| assert_eq!(&plaintext[..len], message),
      ),
    );
  }
  group.finish();
}

#[cfg(feature = "getrandom")]
fn rsa_pkcs1v15_decryption(c: &mut Criterion) {
  rsa_pkcs1v15_decryption_for_size(c, 2048, RSA2048_PKCS1V15_DECRYPT);
  rsa_pkcs1v15_decryption_for_size(c, 3072, RSA3072_PKCS1V15_DECRYPT);
  rsa_pkcs1v15_decryption_for_size(c, 4096, RSA4096_PKCS1V15_DECRYPT);
}

/// Legacy RSAES-PKCS1-v1_5 decryption of the 32-byte Wycheproof message with
/// fresh blinding: blinding, the private operation, and padding decoding. Keys
/// are imported outside timing. rscrypto and RustCrypto draw and invert a new
/// blinding factor on every call; AWS-LC reuses its cached blinding pair (see
/// the OAEP decryption groups). RustCrypto returns a new `Vec`, and aws-lc-rs
/// builds an `EVP_PKEY_CTX` on every call; both stay timed. `rscrypto` reuses
/// scratch; `rscrypto-oneshot` allocates it per call. ring has no RSA
/// decryption.
///
/// The libraries handle invalid padding differently, and that design costs
/// time on valid input too. rscrypto applies the implicit rejection of
/// draft-irtf-cfrg-rsa-guidance: every call hashes the private exponent, keys
/// HMAC-SHA256 with the ciphertext, and expands a modulus-length synthetic
/// message, which it returns only when the padding is invalid. RustCrypto
/// 0.9 and AWS-LC return an error instead.
///
/// Untimed check: every row recovers the Wycheproof message.
#[cfg(feature = "getrandom")]
fn rsa_pkcs1v15_decryption_for_size(c: &mut Criterion, bits: usize, document: &str) {
  let name = format!("rsa-{bits}-pkcs1v15-encryption");
  if !bench_config::selected(&format!("{name}/")) {
    return;
  }
  let fixture = DecryptionFixture::pkcs1v15(document, bits);
  let message = fixture.message.as_slice();
  let key = fixture.private_key(bits);
  let modulus_len = key.public_key().modulus().len();
  let rustcrypto_private_key = RustCryptoRsaPrivateKey::from_pkcs8_der(&fixture.private_key_pkcs8)
    .expect("RustCrypto must import the Wycheproof RSA private key");

  let mut group = c.benchmark_group(format!("{name}/decrypt/fresh-entropy"));
  group.bench_function(
    "rscrypto",
    checked_with(
      (vec![0u8; modulus_len], key.private_scratch()),
      |(plaintext, scratch)| {
        key
          .decrypt_pkcs1v15_with_scratch(black_box(&fixture.ciphertext), black_box(plaintext), black_box(scratch))
          .expect("valid OS-random scratch-backed RSA-PKCS1-v1_5 benchmark decryption must succeed")
      },
      |len, (plaintext, _)| assert_eq!(&plaintext[..len], message),
    ),
  );
  group.bench_function(
    "rscrypto-oneshot",
    checked_with(
      vec![0u8; modulus_len],
      |plaintext| {
        key
          .decrypt_pkcs1v15(black_box(&fixture.ciphertext), black_box(plaintext))
          .expect("valid OS-random one-shot RSA-PKCS1-v1_5 benchmark decryption must succeed")
      },
      |len, plaintext| assert_eq!(&plaintext[..len], message),
    ),
  );
  group.bench_function(
    "rustcrypto-rsa-blinded",
    checked(
      || {
        black_box(&rustcrypto_private_key)
          .decrypt_blinded(
            &mut GetrandomRng,
            RustCryptoPkcs1v15Encrypt,
            black_box(&fixture.ciphertext),
          )
          .expect("valid RustCrypto RSA-PKCS1-v1_5 benchmark decryption must succeed")
      },
      |plaintext| assert_eq!(plaintext, message),
    ),
  );
  aws_lc_bench! {
    let aws_decrypting_key = aws_rsa::Pkcs1PrivateDecryptingKey::new(
      aws_rsa::PrivateDecryptingKey::from_pkcs8(&fixture.private_key_pkcs8)
        .expect("aws-lc-rs must import the Wycheproof RSA private key"),
    )
    .expect("aws-lc-rs must build the RSA-PKCS1-v1_5 private key");
    group.bench_function(
      "aws-lc-rs",
      checked_with(
        vec![0u8; aws_decrypting_key.min_output_size()],
        |plaintext| {
          aws_decrypting_key
            .decrypt(black_box(&fixture.ciphertext), black_box(plaintext))
            .expect("valid aws-lc-rs RSA-PKCS1-v1_5 benchmark decryption must succeed")
            .len()
        },
        |len, plaintext| assert_eq!(&plaintext[..len], message),
      ),
    );
  }
  group.finish();
}

/// One RSA signature scheme, as each library names it.
trait VerifyScheme {
  /// The operation segment of the case ID.
  const OPERATION: &'static str;
  /// The message that the fixture signature covers.
  const MESSAGE: &'static [u8];
  type RustCryptoKey: rsa::signature::Verifier<Self::RustCryptoSignature>;
  type RustCryptoSignature: for<'s> TryFrom<&'s [u8], Error = rsa::signature::Error>;

  fn rustcrypto_key(key: RustCryptoRsaPublicKey) -> Self::RustCryptoKey;
  fn ring_algorithm() -> &'static ring_signature::RsaParameters;
  aws_lc_bench! {
    fn aws_algorithm() -> &'static aws_signature::RsaParameters;
  }
  fn verify(key: &RsaPublicKey, message: &[u8], signature: &[u8]) -> Result<(), VerificationError>;
  fn verify_with_scratch(
    key: &RsaPublicKey,
    message: &[u8],
    signature: &[u8],
    scratch: &mut RsaPublicScratch,
  ) -> Result<(), VerificationError>;
}

/// RSASSA-PSS with SHA-256, MGF1-SHA256, and a 32-byte salt.
struct PssSha256;

impl VerifyScheme for PssSha256 {
  const OPERATION: &'static str = "verify-pss-sha256";
  const MESSAGE: &'static [u8] = MESSAGE_PSS;
  type RustCryptoKey = RustCryptoPssVerifyingKey<sha2::Sha256>;
  type RustCryptoSignature = RustCryptoPssSignature;

  fn rustcrypto_key(key: RustCryptoRsaPublicKey) -> Self::RustCryptoKey {
    RustCryptoPssVerifyingKey::new(key)
  }

  fn ring_algorithm() -> &'static ring_signature::RsaParameters {
    &ring_signature::RSA_PSS_2048_8192_SHA256
  }

  aws_lc_bench! {
    fn aws_algorithm() -> &'static aws_signature::RsaParameters {
      &aws_signature::RSA_PSS_2048_8192_SHA256
    }
  }

  fn verify(key: &RsaPublicKey, message: &[u8], signature: &[u8]) -> Result<(), VerificationError> {
    key.verify_pss(RsaPssProfile::Sha256, message, signature)
  }

  fn verify_with_scratch(
    key: &RsaPublicKey,
    message: &[u8],
    signature: &[u8],
    scratch: &mut RsaPublicScratch,
  ) -> Result<(), VerificationError> {
    key.verify_pss_with_scratch(RsaPssProfile::Sha256, message, signature, scratch)
  }
}

/// RSASSA-PKCS1-v1_5 with SHA-256.
struct Pkcs1v15Sha256;

impl VerifyScheme for Pkcs1v15Sha256 {
  const OPERATION: &'static str = "verify-pkcs1v15-sha256";
  const MESSAGE: &'static [u8] = MESSAGE_PKCS1V15;
  type RustCryptoKey = RustCryptoPkcs1v15VerifyingKey<sha2::Sha256>;
  type RustCryptoSignature = RustCryptoPkcs1v15Signature;

  fn rustcrypto_key(key: RustCryptoRsaPublicKey) -> Self::RustCryptoKey {
    RustCryptoPkcs1v15VerifyingKey::new(key)
  }

  fn ring_algorithm() -> &'static ring_signature::RsaParameters {
    &ring_signature::RSA_PKCS1_2048_8192_SHA256
  }

  aws_lc_bench! {
    fn aws_algorithm() -> &'static aws_signature::RsaParameters {
      &aws_signature::RSA_PKCS1_2048_8192_SHA256
    }
  }

  fn verify(key: &RsaPublicKey, message: &[u8], signature: &[u8]) -> Result<(), VerificationError> {
    key.verify_pkcs1v15(RsaPkcs1v15Profile::Sha256, message, signature)
  }

  fn verify_with_scratch(
    key: &RsaPublicKey,
    message: &[u8],
    signature: &[u8],
    scratch: &mut RsaPublicScratch,
  ) -> Result<(), VerificationError> {
    key.verify_pkcs1v15_with_scratch(RsaPkcs1v15Profile::Sha256, message, signature, scratch)
  }
}

/// One verification fixture and the inputs that every row must reject.
struct VerifyFixture<'a> {
  /// The PKCS #1 `RSAPublicKey` DER inside the SPKI fixture: the bytes that
  /// every import row receives.
  pkcs1: Vec<u8>,
  /// The same key with a non-minimal exponent encoding.
  noncanonical_pkcs1: Vec<u8>,
  signature: &'a [u8],
  tampered_signature: Vec<u8>,
}

impl<'a> VerifyFixture<'a> {
  fn new(key: &RsaPublicKey, spki: &[u8], signature: &'a [u8]) -> Self {
    let pkcs1 = spki_subject_public_key(spki);
    // The SPKI builder that wraps the non-canonical key reproduces the fixture.
    assert_eq!(spki_der_from_pkcs1(&pkcs1), spki, "rebuilt RSA SPKI fixture");
    Self {
      pkcs1,
      noncanonical_pkcs1: noncanonical_exponent_pkcs1_der(key),
      signature,
      tampered_signature: tampered(signature),
    }
  }
}

/// Which verification task a row measures.
#[derive(Clone, Copy, PartialEq, Eq)]
enum VerifyTask {
  /// The caller holds a key prepared outside timing; `verify` ignores the key
  /// bytes.
  ReuseKey,
  /// Every call imports the key bytes.
  ImportKey,
}

/// Registers one verification row. `verify(state, key, signature)` is the
/// timed operation on the fixture key bytes and signature.
///
/// The untimed check runs `verify` with the row's own state: the fixture
/// signature verifies twice, a signature with one flipped bit is rejected, and
/// the fixture signature verifies again. An import row must also reject the
/// key with a non-minimal exponent encoding. That key is mathematically the
/// fixture key, so only the import's DER validation can reject it. `oracle`
/// then confirms both signature verdicts with a library other than the row's.
fn verify_row<S, E: core::fmt::Debug>(
  fixture: &VerifyFixture<'_>,
  task: VerifyTask,
  state: S,
  verify: impl FnMut(&mut S, &[u8], &[u8]) -> Result<(), E>,
  oracle: impl FnOnce(),
) -> impl FnMut(&mut Bencher<'_>) {
  checked_routine(
    (state, verify),
    move |(state, verify)| {
      for _ in 0..2 {
        verify(state, &fixture.pkcs1, fixture.signature).expect("the RSA benchmark signature must verify");
      }
      verify(state, &fixture.pkcs1, &fixture.tampered_signature).expect_err("a tampered RSA signature must not verify");
      verify(state, &fixture.pkcs1, fixture.signature)
        .expect("the RSA benchmark signature must verify after a rejection");
      if task == VerifyTask::ImportKey {
        verify(state, &fixture.noncanonical_pkcs1, fixture.signature)
          .expect_err("RSA key import must reject a non-minimal exponent encoding");
      }
      oracle();
    },
    move |b, (state, verify)| {
      b.iter(|| {
        verify(
          black_box(&mut *state),
          black_box(&fixture.pkcs1),
          black_box(fixture.signature),
        )
        .expect("valid RSA benchmark fixture must succeed")
      })
    },
  )
}

/// Untimed oracle for rscrypto rows: ring accepts the fixture signature and
/// rejects the tampered one.
fn ring_confirms<V: VerifyScheme>(fixture: &VerifyFixture<'_>) {
  let key = ring_signature::UnparsedPublicKey::new(V::ring_algorithm(), &fixture.pkcs1);
  key
    .verify(V::MESSAGE, fixture.signature)
    .expect("ring must verify the RSA benchmark signature");
  key
    .verify(V::MESSAGE, &fixture.tampered_signature)
    .expect_err("ring must reject a tampered RSA signature");
}

/// Untimed oracle for comparison rows: rscrypto accepts the fixture signature
/// and rejects the tampered one.
fn rscrypto_confirms<V: VerifyScheme>(key: &RsaPublicKey, fixture: &VerifyFixture<'_>) {
  V::verify(key, V::MESSAGE, fixture.signature).expect("rscrypto must verify the RSA benchmark signature");
  V::verify(key, V::MESSAGE, &fixture.tampered_signature).expect_err("rscrypto must reject a tampered RSA signature");
}

/// Registers the reuse-key and import-key verification groups of one scheme.
fn rsa_verify_for_scheme<V: VerifyScheme>(
  c: &mut Criterion,
  name: &str,
  key: &RsaPublicKey,
  fixture: &VerifyFixture<'_>,
  import_policy: &RsaPublicKeyPolicy,
) {
  let ring_oracle = || ring_confirms::<V>(fixture);
  let rscrypto_oracle = || rscrypto_confirms::<V>(key, fixture);
  let rustcrypto_key =
    RustCryptoRsaPublicKey::from_pkcs1_der(&fixture.pkcs1).expect("RustCrypto imports every benchmark modulus");

  // Verification with a key that the caller already holds: each call verifies
  // the fixture signature over the fixed message. Each library's prepared key
  // is built outside timing. Message hashing, signature decoding, the public
  // operation, and padding checks are timed.
  // - `rscrypto` reuses the key and a public scratch, which caches
  //   R^2 mod n for Montgomery multiplication.
  // - `rscrypto-oneshot` reuses the key, but allocates scratch on every call
  //   and enters the Montgomery domain by one division by n instead of R^2.
  // - aws-lc-rs reuses a `ParsedPublicKey`. AWS-LC caches its Montgomery
  //   context in the key after the first call. aws-lc-rs builds an
  //   `EVP_PKEY_CTX` on every call.
  // - RustCrypto reuses a `VerifyingKey`, which holds only n and e. It
  //   converts the signature bytes to an integer on every call.
  // - ring has no prepared-key verifier. Its best API for this task,
  //   `RsaPublicKeyComponents`, skips DER parsing but rebuilds the modulus and
  //   its Montgomery constant on every call.
  let mut group = c.benchmark_group(format!("{name}/{}/reuse-key", V::OPERATION));
  group.bench_function(
    "rscrypto",
    verify_row(
      fixture,
      VerifyTask::ReuseKey,
      key.public_scratch(),
      |scratch, _, signature| V::verify_with_scratch(key, black_box(V::MESSAGE), signature, scratch),
      &ring_oracle,
    ),
  );
  group.bench_function(
    "rscrypto-oneshot",
    verify_row(
      fixture,
      VerifyTask::ReuseKey,
      (),
      |_, _, signature| V::verify(key, black_box(V::MESSAGE), signature),
      &ring_oracle,
    ),
  );
  aws_lc_bench! {
    let aws_key = aws_signature::ParsedPublicKey::new(V::aws_algorithm(), &fixture.pkcs1)
      .expect("aws-lc-rs must parse the RSA benchmark key");
    group.bench_function(
      "aws-lc-rs",
      verify_row(
        fixture,
        VerifyTask::ReuseKey,
        (),
        |_, _, signature| aws_key.verify_sig(black_box(V::MESSAGE), signature),
        &rscrypto_oracle,
      ),
    );
  }
  let verifying_key = V::rustcrypto_key(rustcrypto_key);
  group.bench_function(
    "rustcrypto-rsa",
    verify_row(
      fixture,
      VerifyTask::ReuseKey,
      (),
      |_, _, signature| {
        let signature = V::RustCryptoSignature::try_from(signature)?;
        verifying_key.verify(black_box(V::MESSAGE), &signature)
      },
      &rscrypto_oracle,
    ),
  );
  let exponent = exponent_bytes(key.public_exponent().as_u64());
  let ring_key = ring_signature::RsaPublicKeyComponents {
    n: key.modulus(),
    e: exponent.as_slice(),
  };
  group.bench_function(
    "ring",
    verify_row(
      fixture,
      VerifyTask::ReuseKey,
      (),
      |_, _, signature| ring_key.verify(V::ring_algorithm(), black_box(V::MESSAGE), signature),
      &rscrypto_oracle,
    ),
  );
  group.finish();

  // Import plus verification: each call parses the same PKCS #1
  // `RSAPublicKey` DER bytes, validates the key, and verifies the fixture
  // signature, as a one-time certificate or handshake check does. All per-key
  // setup is timed. rscrypto calls `from_pkcs1_der_with_policy` and its
  // one-shot verifier; aws-lc-rs and ring call `UnparsedPublicKey::verify`;
  // RustCrypto calls `from_pkcs1_der` and builds a `VerifyingKey`. Every
  // import rejects non-minimal DER integers and an even modulus. The other
  // checks differ:
  // - rscrypto: the policy's modulus range (2048-8192 bits for RSA-2048,
  //   3072-8192 bits otherwise) and e = 65537.
  // - ring: a 2048-8192-bit modulus and odd e from 3 to 2^33 - 1.
  // - aws-lc-rs: AWS-LC accepts a modulus of at most 16384 bits and odd e
  //   from 3 to 2^33 - 1; aws-lc-rs then requires 2048-8192 bits.
  // - RustCrypto: a modulus of at most 8192 bits, and odd e from 3 to
  //   2^33 - 1 and below n.
  let mut group = c.benchmark_group(format!("{name}/{}/import-key", V::OPERATION));
  group.bench_function(
    "rscrypto",
    verify_row(
      fixture,
      VerifyTask::ImportKey,
      (),
      |_, key_der, signature| {
        let key =
          RsaPublicKey::from_pkcs1_der_with_policy(key_der, import_policy).map_err(|_| "rscrypto rejected the key")?;
        V::verify(&key, black_box(V::MESSAGE), signature).map_err(|_| "rscrypto rejected the signature")
      },
      &ring_oracle,
    ),
  );
  aws_lc_bench! {
    group.bench_function(
      "aws-lc-rs",
      verify_row(
        fixture,
        VerifyTask::ImportKey,
        (),
        |_, key_der, signature| {
          aws_signature::UnparsedPublicKey::new(V::aws_algorithm(), key_der).verify(black_box(V::MESSAGE), signature)
        },
        &rscrypto_oracle,
      ),
    );
  }
  group.bench_function(
    "rustcrypto-rsa",
    verify_row(
      fixture,
      VerifyTask::ImportKey,
      (),
      |_, key_der, signature| {
        let key = RustCryptoRsaPublicKey::from_pkcs1_der(key_der).map_err(|_| "RustCrypto rejected the key")?;
        let signature = V::RustCryptoSignature::try_from(signature).map_err(|_| "RustCrypto rejected the signature")?;
        V::rustcrypto_key(key)
          .verify(black_box(V::MESSAGE), &signature)
          .map_err(|_| "RustCrypto rejected the signature")
      },
      &rscrypto_oracle,
    ),
  );
  group.bench_function(
    "ring",
    verify_row(
      fixture,
      VerifyTask::ImportKey,
      (),
      |_, key_der, signature| {
        ring_signature::UnparsedPublicKey::new(V::ring_algorithm(), key_der).verify(black_box(V::MESSAGE), signature)
      },
      &rscrypto_oracle,
    ),
  );
  group.finish();
}

fn rsa_components_for_size(
  c: &mut Criterion,
  name: &str,
  pss_spki: &[u8],
  pss_sig: &[u8],
  pkcs1_spki: &[u8],
  pkcs1_sig: &[u8],
  import_policy: &RsaPublicKeyPolicy,
) {
  if !bench_config::selected(&format!("{name}/")) {
    return;
  }
  let pss_key =
    RsaPublicKey::from_spki_der_with_policy(pss_spki, import_policy).expect("valid RSA benchmark fixture must succeed");
  let pkcs1_key = RsaPublicKey::from_spki_der_with_policy(pkcs1_spki, import_policy)
    .expect("valid RSA benchmark fixture must succeed");
  let pss_fixture = VerifyFixture::new(&pss_key, pss_spki, pss_sig);
  let pkcs1_fixture = VerifyFixture::new(&pkcs1_key, pkcs1_spki, pkcs1_sig);
  let representative = modulus_minus_one(&pss_key);
  let modulus_len = pss_key.modulus().len();

  #[cfg(all(rscrypto_internal, feature = "diag"))]
  let (pss_encoded, pss_em_bits, mut pss_db, mut pss_db_mask, pkcs1_encoded) = {
    let mut pss_encoded = vec![0u8; pss_key.modulus().len()];
    pss_key
      .public_operation_with_scratch(pss_sig, &mut pss_encoded, &mut pss_key.public_scratch())
      .expect("valid RSA benchmark fixture must succeed");
    let pss_em_bits = pss_key.modulus_bits().strict_sub(1);
    let pss_em_len = pss_em_bits.strict_add(7) / 8;
    let leading = pss_encoded.len().strict_sub(pss_em_len);
    pss_encoded.drain(..leading);

    let mut pkcs1_encoded = vec![0u8; pkcs1_key.modulus().len()];
    pkcs1_key
      .public_operation_with_scratch(pkcs1_sig, &mut pkcs1_encoded, &mut pkcs1_key.public_scratch())
      .expect("valid RSA benchmark fixture must succeed");

    (
      pss_encoded,
      pss_em_bits,
      vec![0u8; pss_em_len],
      vec![0u8; pss_em_len],
      pkcs1_encoded,
    )
  };

  // Untimed oracle: RustCrypto's modpow for the public-operation rows.
  let public_operation_reference = || reference_public_operation(&pss_key, &representative);

  // rscrypto-only rows: public scratch allocation, the raw public operation,
  // and diagnostic padding and Montgomery-variant rows.
  let mut group = c.benchmark_group(name);
  group.bench_function(
    "scratch-setup-rscrypto",
    checked(
      || black_box(pss_key.public_scratch()),
      |mut scratch| {
        let mut out = vec![0u8; modulus_len];
        pss_key
          .public_operation_with_scratch(&representative, &mut out, &mut scratch)
          .expect("a fresh RSA public scratch must serve the public operation");
        assert_eq!(out, public_operation_reference());
      },
    ),
  );
  group.bench_function(
    "public-op-e65537",
    checked_with(
      (vec![0u8; modulus_len], pss_key.public_scratch()),
      |(out, scratch)| {
        pss_key
          .public_operation_with_scratch(black_box(&representative), black_box(out), black_box(scratch))
          .expect("valid RSA benchmark fixture must succeed")
      },
      |(), (out, _)| assert_eq!(*out, public_operation_reference()),
    ),
  );
  #[cfg(all(rscrypto_internal, feature = "diag"))]
  {
    group.bench_function(
      "public-op-e65537-product-montgomery",
      checked_with(
        (vec![0u8; modulus_len], pss_key.public_scratch()),
        |(out, scratch)| {
          diag_rsa_public_operation_product(
            black_box(&pss_key),
            black_box(&representative),
            black_box(out),
            black_box(scratch),
          )
          .expect("valid RSA benchmark fixture must succeed")
        },
        |(), (out, _)| assert_eq!(*out, public_operation_reference()),
      ),
    );
    group.bench_function(
      "public-op-e65537-cios-candidate",
      checked_with(
        (vec![0u8; modulus_len], pss_key.public_scratch()),
        |(out, scratch)| {
          diag_rsa_public_operation_cios(
            black_box(&pss_key),
            black_box(&representative),
            black_box(out),
            black_box(scratch),
          )
          .expect("valid RSA benchmark fixture must succeed")
        },
        |(), (out, _)| assert_eq!(*out, public_operation_reference()),
      ),
    );
  }
  // The padding rows decode the encoded messages that the production public
  // operation recovers from the fixture signatures. ring confirms that each
  // fixture signature is valid.
  #[cfg(all(rscrypto_internal, feature = "diag"))]
  group.bench_function(
    "padding-pss-sha256-rscrypto",
    checked(
      || {
        diag_rsa_verify_pss_encoded_with_scratch(
          RsaPssProfile::Sha256,
          black_box(MESSAGE_PSS),
          black_box(&pss_encoded),
          black_box(pss_em_bits),
          black_box(&mut pss_db),
          black_box(&mut pss_db_mask),
        )
        .expect("valid RSA benchmark fixture must succeed")
      },
      |()| {
        diag_rsa_verify_pss_encoded_with_scratch(
          RsaPssProfile::Sha256,
          MESSAGE_PSS,
          &tampered(&pss_encoded),
          pss_em_bits,
          &mut vec![0u8; pss_encoded.len()],
          &mut vec![0u8; pss_encoded.len()],
        )
        .expect_err("a tampered RSA-PSS encoded message must not verify");
        ring_confirms::<PssSha256>(&pss_fixture);
      },
    ),
  );
  #[cfg(all(rscrypto_internal, feature = "diag"))]
  group.bench_function(
    "padding-pkcs1v15-sha256-rscrypto",
    checked(
      || {
        diag_rsa_verify_pkcs1v15_encoded(
          RsaPkcs1v15Profile::Sha256,
          black_box(MESSAGE_PKCS1V15),
          black_box(&pkcs1_encoded),
        )
        .expect("valid RSA benchmark fixture must succeed")
      },
      |()| {
        diag_rsa_verify_pkcs1v15_encoded(RsaPkcs1v15Profile::Sha256, MESSAGE_PKCS1V15, &tampered(&pkcs1_encoded))
          .expect_err("a tampered RSA-PKCS1-v1_5 encoded message must not verify");
        ring_confirms::<Pkcs1v15Sha256>(&pkcs1_fixture);
      },
    ),
  );
  group.finish();

  // SPKI parsing: each library parses the same `rsaEncryption` SPKI bytes into
  // a reusable public key. Timed: DER parsing, key validation, and the key's
  // allocation and destruction. Key validation matches the import-key groups,
  // except that aws-lc-rs checks its 2048-8192-bit range only when it
  // verifies. aws-lc-rs `ParsedPublicKey` first tries PKCS #1, then SPKI, and
  // copies the input bytes. ring has no public SPKI parser for RSA keys.
  // Untimed checks: each parsed key matches another library's result,
  // and each parser rejects the SPKI around the key with a non-minimal
  // exponent encoding.
  let malformed_spki = spki_der_from_pkcs1(&pss_fixture.noncanonical_pkcs1);
  let mut group = c.benchmark_group(format!("{name}/parse-spki"));
  group.bench_function(
    "rscrypto",
    checked(
      || {
        black_box(
          RsaPublicKey::from_spki_der_with_policy(black_box(pss_spki), import_policy)
            .expect("valid RSA benchmark fixture must succeed"),
        )
      },
      |key| {
        assert_eq!(key.to_pkcs1_der(), pss_fixture.pkcs1, "rscrypto parse of the RSA SPKI");
        assert!(
          RsaPublicKey::from_spki_der_with_policy(&malformed_spki, import_policy).is_err(),
          "rscrypto must reject a non-minimal RSA exponent encoding"
        );
      },
    ),
  );
  aws_lc_bench! {
    group.bench_function(
      "aws-lc-rs",
      checked(
        || {
          black_box(
            aws_signature::ParsedPublicKey::new(PssSha256::aws_algorithm(), black_box(pss_spki))
              .expect("valid RSA benchmark fixture must succeed"),
          )
        },
        |key| {
          key
            .verify_sig(MESSAGE_PSS, pss_sig)
            .expect("the parsed aws-lc-rs key must verify the RSA-PSS benchmark signature");
          key
            .verify_sig(MESSAGE_PSS, &pss_fixture.tampered_signature)
            .expect_err("the parsed aws-lc-rs key must reject a tampered RSA-PSS signature");
          rscrypto_confirms::<PssSha256>(&pss_key, &pss_fixture);
          assert!(
            aws_signature::ParsedPublicKey::new(PssSha256::aws_algorithm(), &malformed_spki).is_err(),
            "aws-lc-rs must reject a non-minimal RSA exponent encoding"
          );
        },
      ),
    );
  }
  group.bench_function(
    "rustcrypto-rsa",
    checked(
      || {
        black_box(
          RustCryptoRsaPublicKey::from_public_key_der(black_box(pss_spki))
            .expect("valid RSA benchmark fixture must succeed"),
        )
      },
      |key| {
        let encoded = key
          .to_pkcs1_der()
          .expect("RustCrypto must encode the parsed RSA public key");
        assert_eq!(
          encoded.as_bytes(),
          pss_key.to_pkcs1_der(),
          "RustCrypto parse of the RSA SPKI"
        );
        assert!(
          RustCryptoRsaPublicKey::from_public_key_der(&malformed_spki).is_err(),
          "RustCrypto must reject a non-minimal RSA exponent encoding"
        );
      },
    ),
  );
  group.finish();

  rsa_verify_for_scheme::<PssSha256>(c, name, &pss_key, &pss_fixture, import_policy);
  rsa_verify_for_scheme::<Pkcs1v15Sha256>(c, name, &pkcs1_key, &pkcs1_fixture, import_policy);
}

fn rsa_public_exponents(c: &mut Criterion) {
  if !bench_config::selected("rsa-2048-public-exponents") {
    return;
  }
  let modulus = legacy_exponent_modulus();
  let input = legacy_exponent_input();
  let policy = RsaPublicKeyPolicy::modulus_2048_to_8192().allow_legacy_small_exponents();

  let key_e3 = RsaPublicKey::from_pkcs1_der_with_policy(&pkcs1_der_from_modulus_exponent(&modulus, &[0x03]), &policy)
    .expect("valid RSA benchmark fixture must succeed");
  let key_e17 = RsaPublicKey::from_pkcs1_der_with_policy(&pkcs1_der_from_modulus_exponent(&modulus, &[0x11]), &policy)
    .expect("valid RSA benchmark fixture must succeed");
  let key_e65537 =
    RsaPublicKey::from_pkcs1_der_with_policy(&pkcs1_der_from_modulus_exponent(&modulus, &[0x01, 0x00, 0x01]), &policy)
      .expect("valid RSA benchmark fixture must succeed");
  let key_generic = RsaPublicKey::from_pkcs1_der_with_policy(
    &pkcs1_der_from_modulus_exponent(&modulus, &[0x49, 0xd2, 0xa1]),
    &policy.allow_legacy_odd_exponents(),
  )
  .expect("valid RSA benchmark fixture must succeed");
  // Each row owns its output and scratch. RustCrypto's modpow is the oracle.
  let buffers = |key: &RsaPublicKey| (vec![0u8; key.modulus().len()], key.public_scratch());
  let matches_reference = |key: &RsaPublicKey, out: &[u8]| assert_eq!(out, reference_public_operation(key, &input));

  let mut group = c.benchmark_group("rsa-2048-public-exponents");
  group.bench_function(
    "public-op-e3",
    checked_with(
      buffers(&key_e3),
      |(out, scratch)| {
        key_e3
          .public_operation_with_scratch(black_box(&input), black_box(out), black_box(scratch))
          .expect("valid RSA benchmark fixture must succeed")
      },
      |(), (out, _)| matches_reference(&key_e3, out),
    ),
  );
  #[cfg(all(rscrypto_internal, feature = "diag"))]
  group.bench_function(
    "public-op-e3-bitserial-baseline",
    checked_with(
      vec![0u8; modulus.len()],
      |out| {
        diag_rsa_public_operation_bitserial(black_box(&key_e3), black_box(&input), black_box(out))
          .expect("valid RSA benchmark fixture must succeed")
      },
      |(), out| matches_reference(&key_e3, out),
    ),
  );
  #[cfg(all(rscrypto_internal, feature = "diag"))]
  group.bench_function(
    "public-op-e3-product-montgomery",
    checked_with(
      buffers(&key_e3),
      |(out, scratch)| {
        diag_rsa_public_operation_product(
          black_box(&key_e3),
          black_box(&input),
          black_box(out),
          black_box(scratch),
        )
        .expect("valid RSA benchmark fixture must succeed")
      },
      |(), (out, _)| matches_reference(&key_e3, out),
    ),
  );
  #[cfg(all(rscrypto_internal, feature = "diag"))]
  group.bench_function(
    "public-op-e3-generic-exponent",
    checked_with(
      buffers(&key_e3),
      |(out, scratch)| {
        diag_rsa_public_operation_generic_exponent(
          black_box(&key_e3),
          black_box(&input),
          black_box(out),
          black_box(scratch),
        )
        .expect("valid RSA benchmark fixture must succeed")
      },
      |(), (out, _)| matches_reference(&key_e3, out),
    ),
  );
  #[cfg(all(rscrypto_internal, feature = "diag"))]
  group.bench_function(
    "public-op-e3-cios-candidate",
    checked_with(
      buffers(&key_e3),
      |(out, scratch)| {
        diag_rsa_public_operation_cios(
          black_box(&key_e3),
          black_box(&input),
          black_box(out),
          black_box(scratch),
        )
        .expect("valid RSA benchmark fixture must succeed")
      },
      |(), (out, _)| matches_reference(&key_e3, out),
    ),
  );
  group.bench_function(
    "public-op-e17",
    checked_with(
      buffers(&key_e17),
      |(out, scratch)| {
        key_e17
          .public_operation_with_scratch(black_box(&input), black_box(out), black_box(scratch))
          .expect("valid RSA benchmark fixture must succeed")
      },
      |(), (out, _)| matches_reference(&key_e17, out),
    ),
  );
  #[cfg(all(rscrypto_internal, feature = "diag"))]
  group.bench_function(
    "public-op-e17-bitserial-baseline",
    checked_with(
      vec![0u8; modulus.len()],
      |out| {
        diag_rsa_public_operation_bitserial(black_box(&key_e17), black_box(&input), black_box(out))
          .expect("valid RSA benchmark fixture must succeed")
      },
      |(), out| matches_reference(&key_e17, out),
    ),
  );
  #[cfg(all(rscrypto_internal, feature = "diag"))]
  group.bench_function(
    "public-op-e17-product-montgomery",
    checked_with(
      buffers(&key_e17),
      |(out, scratch)| {
        diag_rsa_public_operation_product(
          black_box(&key_e17),
          black_box(&input),
          black_box(out),
          black_box(scratch),
        )
        .expect("valid RSA benchmark fixture must succeed")
      },
      |(), (out, _)| matches_reference(&key_e17, out),
    ),
  );
  #[cfg(all(rscrypto_internal, feature = "diag"))]
  group.bench_function(
    "public-op-e17-generic-exponent",
    checked_with(
      buffers(&key_e17),
      |(out, scratch)| {
        diag_rsa_public_operation_generic_exponent(
          black_box(&key_e17),
          black_box(&input),
          black_box(out),
          black_box(scratch),
        )
        .expect("valid RSA benchmark fixture must succeed")
      },
      |(), (out, _)| matches_reference(&key_e17, out),
    ),
  );
  #[cfg(all(rscrypto_internal, feature = "diag"))]
  group.bench_function(
    "public-op-e17-cios-candidate",
    checked_with(
      buffers(&key_e17),
      |(out, scratch)| {
        diag_rsa_public_operation_cios(
          black_box(&key_e17),
          black_box(&input),
          black_box(out),
          black_box(scratch),
        )
        .expect("valid RSA benchmark fixture must succeed")
      },
      |(), (out, _)| matches_reference(&key_e17, out),
    ),
  );
  group.bench_function(
    "public-op-e65537",
    checked_with(
      buffers(&key_e65537),
      |(out, scratch)| {
        key_e65537
          .public_operation_with_scratch(black_box(&input), black_box(out), black_box(scratch))
          .expect("valid RSA benchmark fixture must succeed")
      },
      |(), (out, _)| matches_reference(&key_e65537, out),
    ),
  );
  #[cfg(all(rscrypto_internal, feature = "diag"))]
  group.bench_function(
    "public-op-e65537-bitserial-baseline",
    checked_with(
      vec![0u8; modulus.len()],
      |out| {
        diag_rsa_public_operation_bitserial(black_box(&key_e65537), black_box(&input), black_box(out))
          .expect("valid RSA benchmark fixture must succeed")
      },
      |(), out| matches_reference(&key_e65537, out),
    ),
  );
  #[cfg(all(rscrypto_internal, feature = "diag"))]
  group.bench_function(
    "public-op-e65537-product-montgomery",
    checked_with(
      buffers(&key_e65537),
      |(out, scratch)| {
        diag_rsa_public_operation_product(
          black_box(&key_e65537),
          black_box(&input),
          black_box(out),
          black_box(scratch),
        )
        .expect("valid RSA benchmark fixture must succeed")
      },
      |(), (out, _)| matches_reference(&key_e65537, out),
    ),
  );
  #[cfg(all(rscrypto_internal, feature = "diag"))]
  group.bench_function(
    "public-op-e65537-generic-exponent",
    checked_with(
      buffers(&key_e65537),
      |(out, scratch)| {
        diag_rsa_public_operation_generic_exponent(
          black_box(&key_e65537),
          black_box(&input),
          black_box(out),
          black_box(scratch),
        )
        .expect("valid RSA benchmark fixture must succeed")
      },
      |(), (out, _)| matches_reference(&key_e65537, out),
    ),
  );
  #[cfg(all(rscrypto_internal, feature = "diag"))]
  group.bench_function(
    "public-op-e65537-cios-candidate",
    checked_with(
      buffers(&key_e65537),
      |(out, scratch)| {
        diag_rsa_public_operation_cios(
          black_box(&key_e65537),
          black_box(&input),
          black_box(out),
          black_box(scratch),
        )
        .expect("valid RSA benchmark fixture must succeed")
      },
      |(), (out, _)| matches_reference(&key_e65537, out),
    ),
  );
  group.bench_function(
    "public-op-e0x49d2a1-generic",
    checked_with(
      buffers(&key_generic),
      |(out, scratch)| {
        key_generic
          .public_operation_with_scratch(black_box(&input), black_box(out), black_box(scratch))
          .expect("valid RSA benchmark fixture must succeed")
      },
      |(), (out, _)| matches_reference(&key_generic, out),
    ),
  );
  #[cfg(all(rscrypto_internal, feature = "diag"))]
  group.bench_function(
    "public-op-e0x49d2a1-bitserial-baseline",
    checked_with(
      vec![0u8; modulus.len()],
      |out| {
        diag_rsa_public_operation_bitserial(black_box(&key_generic), black_box(&input), black_box(out))
          .expect("valid RSA benchmark fixture must succeed")
      },
      |(), out| matches_reference(&key_generic, out),
    ),
  );
  #[cfg(all(rscrypto_internal, feature = "diag"))]
  group.bench_function(
    "public-op-e0x49d2a1-product-montgomery",
    checked_with(
      buffers(&key_generic),
      |(out, scratch)| {
        diag_rsa_public_operation_product(
          black_box(&key_generic),
          black_box(&input),
          black_box(out),
          black_box(scratch),
        )
        .expect("valid RSA benchmark fixture must succeed")
      },
      |(), (out, _)| matches_reference(&key_generic, out),
    ),
  );
  #[cfg(all(rscrypto_internal, feature = "diag"))]
  group.bench_function(
    "public-op-e0x49d2a1-cios-candidate",
    checked_with(
      buffers(&key_generic),
      |(out, scratch)| {
        diag_rsa_public_operation_cios(
          black_box(&key_generic),
          black_box(&input),
          black_box(out),
          black_box(scratch),
        )
        .expect("valid RSA benchmark fixture must succeed")
      },
      |(), (out, _)| matches_reference(&key_generic, out),
    ),
  );
  group.finish();
}

fn rsa_hash_components(c: &mut Criterion) {
  if !bench_config::selected("rsa-hash-components") {
    return;
  }
  let mut group = c.benchmark_group("rsa-hash-components");

  // RustCrypto `sha2` is the oracle for every digest.
  group.bench_function(
    "sha256-message-pss",
    checked(
      || black_box(Sha256::digest(black_box(MESSAGE_PSS))),
      |digest| {
        assert_eq!(
          digest.as_ref(),
          <sha2::Sha256 as sha2::Digest>::digest(MESSAGE_PSS).as_slice()
        )
      },
    ),
  );
  group.bench_function(
    "sha384-message-pss",
    checked(
      || black_box(Sha384::digest(black_box(MESSAGE_PSS))),
      |digest| {
        assert_eq!(
          digest.as_ref(),
          <sha2::Sha384 as sha2::Digest>::digest(MESSAGE_PSS).as_slice()
        )
      },
    ),
  );
  group.bench_function(
    "sha512-message-pss",
    checked(
      || black_box(Sha512::digest(black_box(MESSAGE_PSS))),
      |digest| {
        assert_eq!(
          digest.as_ref(),
          <sha2::Sha512 as sha2::Digest>::digest(MESSAGE_PSS).as_slice()
        )
      },
    ),
  );
  group.bench_function(
    "sha256-message-pkcs1v15",
    checked(
      || black_box(Sha256::digest(black_box(MESSAGE_PKCS1V15))),
      |digest| {
        assert_eq!(
          digest.as_ref(),
          <sha2::Sha256 as sha2::Digest>::digest(MESSAGE_PKCS1V15).as_slice()
        )
      },
    ),
  );
  group.bench_function(
    "sha384-message-pkcs1v15",
    checked(
      || black_box(Sha384::digest(black_box(MESSAGE_PKCS1V15))),
      |digest| {
        assert_eq!(
          digest.as_ref(),
          <sha2::Sha384 as sha2::Digest>::digest(MESSAGE_PKCS1V15).as_slice()
        )
      },
    ),
  );
  group.bench_function(
    "sha512-message-pkcs1v15",
    checked(
      || black_box(Sha512::digest(black_box(MESSAGE_PKCS1V15))),
      |digest| {
        assert_eq!(
          digest.as_ref(),
          <sha2::Sha512 as sha2::Digest>::digest(MESSAGE_PKCS1V15).as_slice()
        )
      },
    ),
  );

  group.finish();
}

#[cfg(all(rscrypto_internal, feature = "diag"))]
fn rsa_montgomery_thresholds(c: &mut Criterion) {
  if !bench_config::selected("rsa-montgomery-thresholds") {
    return;
  }
  let mut group = c.benchmark_group("rsa-montgomery-thresholds");

  for (name, modulus_len) in [
    ("rsa-2048-32-limbs", 256usize),
    ("rsa-3072-48-limbs", 384usize),
    ("rsa-4096-64-limbs", 512usize),
    ("rsa-4160-65-limbs", 520usize),
    ("rsa-6144-96-limbs", 768usize),
    ("rsa-8128-127-limbs", 1016usize),
    ("rsa-8192-128-limbs", 1024usize),
  ] {
    let key = RsaPublicKey::from_pkcs1_der_with_policy(
      &synthetic_pkcs1_der(modulus_len),
      &RsaPublicKeyPolicy::modulus_2048_to_8192(),
    )
    .expect("valid RSA benchmark fixture must succeed");
    let input = modulus_minus_one(&key);
    // Each row owns its output and scratch. RustCrypto's modpow is the oracle.
    let buffers = || (vec![0u8; key.modulus().len()], key.public_scratch());
    let matches_reference = |out: &[u8]| assert_eq!(out, reference_public_operation(&key, &input));

    group.bench_function(
      format!("{name}/auto"),
      checked_with(
        buffers(),
        |(out, scratch)| {
          key
            .public_operation_with_scratch(black_box(&input), black_box(out), black_box(scratch))
            .expect("valid RSA benchmark fixture must succeed")
        },
        |(), (out, _)| matches_reference(out),
      ),
    );
    group.bench_function(
      format!("{name}/product-montgomery"),
      checked_with(
        buffers(),
        |(out, scratch)| {
          diag_rsa_public_operation_product(black_box(&key), black_box(&input), black_box(out), black_box(scratch))
            .expect("valid RSA benchmark fixture must succeed")
        },
        |(), (out, _)| matches_reference(out),
      ),
    );
    group.bench_function(
      format!("{name}/cios-candidate"),
      checked_with(
        buffers(),
        |(out, scratch)| {
          diag_rsa_public_operation_cios(black_box(&key), black_box(&input), black_box(out), black_box(scratch))
            .expect("valid RSA benchmark fixture must succeed")
        },
        |(), (out, _)| matches_reference(out),
      ),
    );
    group.bench_function(
      format!("{name}/cios-portable"),
      checked_with(
        buffers(),
        |(out, scratch)| {
          diag_rsa_public_operation_cios_portable(
            black_box(&key),
            black_box(&input),
            black_box(out),
            black_box(scratch),
          )
          .expect("valid RSA benchmark fixture must succeed")
        },
        |(), (out, _)| matches_reference(out),
      ),
    );
  }

  group.finish();
}

fn rsa_components(c: &mut Criterion) {
  let pss2048_spki = pss_spki();
  let pss2048_sig = pss_signature_sha256();
  let pkcs1v15_2048_spki = pkcs1v15_spki();
  let pkcs1v15_2048_sig = pkcs1v15_signature_sha256();
  let legacy_policy = RsaPublicKeyPolicy::modulus_2048_to_8192();
  let modern_policy = RsaPublicKeyPolicy::modulus_3072_to_8192();

  rsa_components_for_size(
    c,
    "rsa-2048",
    &pss2048_spki,
    &pss2048_sig,
    &pkcs1v15_2048_spki,
    &pkcs1v15_2048_sig,
    &legacy_policy,
  );
  rsa_components_for_size(
    c,
    "rsa-3072",
    RSA3072_SPKI,
    RSA3072_PSS_SHA256,
    RSA3072_SPKI,
    RSA3072_PKCS1V15_SHA256,
    &modern_policy,
  );
  rsa_components_for_size(
    c,
    "rsa-4096",
    RSA4096_SPKI,
    RSA4096_PSS_SHA256,
    RSA4096_SPKI,
    RSA4096_PKCS1V15_SHA256,
    &modern_policy,
  );
  rsa_components_for_size(
    c,
    "rsa-8192",
    RSA8192_SPKI,
    RSA8192_PSS_SHA256,
    RSA8192_SPKI,
    RSA8192_PKCS1V15_SHA256,
    &modern_policy,
  );
}

fn main() {
  // Qualify the production BMI2/ADX path on an IFMA host before fixtures initialize detection.
  // The benchmark evidence records this same runtime restriction for the auth target.
  if let Some(value) = std::env::var_os("RSCRYPTO_BENCH_DISABLE_IFMA") {
    assert_eq!(value, "1", "RSCRYPTO_BENCH_DISABLE_IFMA must be 1");
    #[cfg(not(target_arch = "x86_64"))]
    {
      eprintln!("RSCRYPTO_BENCH_DISABLE_IFMA requires x86_64");
      std::process::exit(2);
    }
    #[cfg(target_arch = "x86_64")]
    {
      use rscrypto::platform::{caps::x86, expert};

      let mut detected = expert::detect_uncached();
      assert!(detected.caps.has(x86::AVX2), "AVX2 benchmark requires an AVX2 host");
      detected.caps = detected.caps.difference(x86::AVX512IFMA);
      expert::try_set_override(Some(detected)).expect("restrict capabilities before initialization");
      eprintln!("rscrypto-bench IFMA disabled; runtime_caps={}", detected.caps);
    }
  }
  bench_config::run(&[
    rsa_components,
    rsa_private_signing,
    rsa_public_exponents,
    rsa_hash_components,
    #[cfg(all(rscrypto_internal, feature = "diag"))]
    rsa_montgomery_thresholds,
    rsa_oaep,
    #[cfg(feature = "getrandom")]
    rsa_pkcs1v15_decryption,
  ]);
}
