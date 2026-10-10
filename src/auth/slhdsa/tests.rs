use super::*;
use alloc::{vec, vec::Vec};
use serde_json::Value;

fn hex(value: &Value) -> Vec<u8> {
  value
    .as_str()
    .expect("vector hex string")
    .as_bytes()
    .as_chunks::<2>()
    .0
    .iter()
    .map(|pair| u8::from_str_radix(core::str::from_utf8(pair).expect("ASCII hex"), 16).expect("hex byte"))
    .collect()
}

#[expect(clippy::panic, reason = "unknown pinned ACVP metadata is a broken fixture")]
fn vectors(name: &str) -> Value {
  let source = match name {
    "keyGen-prompt" => include_str!("../../../testdata/slhdsa/acvp/keyGen-prompt.json"),
    "keyGen-expected" => include_str!("../../../testdata/slhdsa/acvp/keyGen-expectedResults.json"),
    _ => panic!("unknown vector file"),
  };
  serde_json::from_str(source).expect("official vector JSON")
}

/// The cases of one derived sigGen or sigVer fixture: `SLHACVP1`, the field
/// and case counts, then each case's fields as a little-endian `u32` length
/// and its bytes (testdata/slhdsa/acvp/README.md).
fn derived_cases(data: &'static [u8], fields: usize) -> Vec<Vec<&'static [u8]>> {
  fn take<'a>(input: &mut &'a [u8], len: usize) -> &'a [u8] {
    let (head, tail) = input.split_at(len);
    *input = tail;
    head
  }
  fn length(input: &mut &[u8]) -> usize {
    let bytes = take(input, 4).try_into().expect("a length");
    usize::try_from(u32::from_le_bytes(bytes)).expect("a length fits usize")
  }
  let mut input = data;
  assert_eq!(take(&mut input, 8), b"SLHACVP1");
  assert_eq!(length(&mut input), fields);
  let count = length(&mut input);
  let cases = (0..count)
    .map(|_| {
      (0..fields)
        .map(|_| {
          let len = length(&mut input);
          take(&mut input, len)
        })
        .collect()
    })
    .collect();
  assert!(input.is_empty(), "trailing fixture bytes");
  cases
}

fn ascii(field: &[u8]) -> &str {
  core::str::from_utf8(field).expect("ASCII fixture field")
}

/// One derived sigGen case.
struct SigGenCase {
  parameter_set: &'static str,
  internal: bool,
  prehash: bool,
  deterministic: bool,
  hash: &'static str,
  id: (&'static str, &'static str),
  sk: &'static [u8],
  message: &'static [u8],
  context: &'static [u8],
  addrnd: &'static [u8],
  signature_sha256: &'static [u8],
}

fn signing_cases() -> Vec<SigGenCase> {
  derived_cases(include_bytes!("../../../testdata/slhdsa/acvp/sigGen.bin"), 12)
    .into_iter()
    .map(|fields| SigGenCase {
      parameter_set: ascii(fields[0]),
      internal: fields[1] == b"internal",
      prehash: fields[2] == b"preHash",
      deterministic: fields[3] == b"true",
      hash: ascii(fields[4]),
      id: (ascii(fields[5]), ascii(fields[6])),
      sk: fields[7],
      message: fields[8],
      context: fields[9],
      addrnd: fields[10],
      signature_sha256: fields[11],
    })
    .collect()
}

/// One derived sigVer case.
struct SigVerCase {
  parameter_set: &'static str,
  internal: bool,
  prehash: bool,
  hash: &'static str,
  id: (&'static str, &'static str),
  passed: bool,
  pk: &'static [u8],
  message: &'static [u8],
  context: &'static [u8],
  signature: &'static [u8],
}

fn verification_cases() -> Vec<SigVerCase> {
  derived_cases(include_bytes!("../../../testdata/slhdsa/acvp/sigVer.bin"), 11)
    .into_iter()
    .map(|fields| SigVerCase {
      parameter_set: ascii(fields[0]),
      internal: fields[1] == b"internal",
      prehash: fields[2] == b"preHash",
      hash: ascii(fields[3]),
      id: (ascii(fields[4]), ascii(fields[5])),
      passed: fields[6] == b"true",
      pk: fields[7],
      message: fields[8],
      context: fields[9],
      signature: fields[10],
    })
    .collect()
}

fn groups(value: &Value) -> &[Value] {
  value["testGroups"].as_array().expect("vector groups")
}

fn cases(value: &Value) -> &[Value] {
  value["tests"].as_array().expect("vector cases")
}

/// The expected-results case with the same group and case identifiers.
fn expected<'a>(expected: &'a Value, group: &Value, case: &Value) -> &'a Value {
  let group = groups(expected)
    .iter()
    .find(|candidate| candidate["tgId"] == group["tgId"])
    .expect("expected group");
  cases(group)
    .iter()
    .find(|candidate| candidate["tcId"] == case["tcId"])
    .expect("expected case")
}

/// The internal functions of one parameter set (FIPS 205 Algorithms 18-20).
#[derive(Clone, Copy)]
struct Internal {
  n: usize,
  signature_len: usize,
  /// `(SK.seed, PK.seed) -> PK.root`.
  keygen: fn(&[u8], &[u8]) -> Vec<u8>,
  /// `(SK, M, opt_rand) -> SIG`.
  sign: InternalSign,
  /// `(PK, M, SIG) -> accepted`.
  verify: InternalVerify,
}

type InternalSign = fn(&[u8], &[&[u8]], &[u8]) -> Vec<u8>;
type InternalVerify = fn(&[u8], &[&[u8]], &[u8]) -> bool;

macro_rules! internal {
  ($p:path, $suite:ty, $n:literal) => {
    Internal {
      n: $n,
      signature_len: $p.signature_len,
      keygen: |sk_seed, pk_seed| {
        let mut root = [0; $n];
        scheme::keygen::<$suite, $n>(
          &$p,
          sk_seed.try_into().expect("SK.seed"),
          pk_seed.try_into().expect("PK.seed"),
          &mut root,
        );
        root.to_vec()
      },
      sign: |sk, message, opt_rand| {
        let (fields, _) = sk.as_chunks::<$n>();
        let mut signature = vec![0; $p.signature_len];
        let key = scheme::PrivateKey {
          sk_seed: &fields[0],
          sk_prf: &fields[1],
          pk_seed: &fields[2],
          pk_root: &fields[3],
        };
        scheme::sign::<$suite, $n>(
          &$p,
          &key,
          message,
          opt_rand.try_into().expect("opt_rand"),
          signature.as_chunks_mut::<$n>().0,
        );
        signature
      },
      verify: |pk, message, signature| {
        let (pk_seed, pk_root) = halves::<$n>(pk);
        scheme::verify::<$suite, $n>(&$p, pk_seed, pk_root, message, signature)
      },
    }
  };
}

#[expect(clippy::panic, reason = "unknown pinned ACVP metadata is a broken fixture")]
fn internal(parameter_set: &str) -> Internal {
  match parameter_set {
    "SLH-DSA-SHA2-128s" => internal!(params::P128S, hash::Sha2Category1<16>, 16),
    "SLH-DSA-SHA2-128f" => internal!(params::P128F, hash::Sha2Category1<16>, 16),
    "SLH-DSA-SHA2-192s" => internal!(params::P192S, hash::Sha2Category3And5<24>, 24),
    "SLH-DSA-SHA2-192f" => internal!(params::P192F, hash::Sha2Category3And5<24>, 24),
    "SLH-DSA-SHA2-256s" => internal!(params::P256S, hash::Sha2Category3And5<32>, 32),
    "SLH-DSA-SHA2-256f" => internal!(params::P256F, hash::Sha2Category3And5<32>, 32),
    "SLH-DSA-SHAKE-128s" => internal!(params::P128S, hash::Shake<16>, 16),
    "SLH-DSA-SHAKE-128f" => internal!(params::P128F, hash::Shake<16>, 16),
    "SLH-DSA-SHAKE-192s" => internal!(params::P192S, hash::Shake<24>, 24),
    "SLH-DSA-SHAKE-192f" => internal!(params::P192F, hash::Shake<24>, 24),
    "SLH-DSA-SHAKE-256s" => internal!(params::P256S, hash::Shake<32>, 32),
    "SLH-DSA-SHAKE-256f" => internal!(params::P256F, hash::Shake<32>, 32),
    _ => panic!("unrecognized parameter set"),
  }
}

/// The public API of one parameter set's pure and HashSLH-DSA profiles.
#[derive(Clone, Copy)]
struct Public {
  /// ACVP name of the RFC 9909 pre-hash pairing.
  prehash: &'static str,
  /// `seeds -> (PK, SK)`.
  generate: PublicGenerate,
  /// `(mode, SK, message, context, addrnd) -> SIG`.
  sign: PublicSign,
  /// `(mode, PK, message, context, SIG) -> accepted`.
  verify: PublicVerify,
}

type PublicGenerate = fn(&[u8]) -> (Vec<u8>, Vec<u8>);
type PublicSign = fn(Mode, &[u8], &[u8], &[u8], Option<&[u8]>) -> Vec<u8>;
type PublicVerify = fn(Mode, &[u8], &[u8], &[u8], &[u8]) -> bool;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mode {
  Pure,
  Prehash,
}

macro_rules! public {
  ($profile:ident, $public:ident, $secret:ident, $hash_public:ident, $hash_secret:ident, $prehash:literal) => {
    Public {
      prehash: $prehash,
      generate: |seeds| {
        let (public, secret) = $profile::generate_keypair(|out| {
          out.copy_from_slice(seeds);
          Ok(())
        })
        .expect("key generation");
        (public.to_bytes().to_vec(), secret.expose_secret().as_bytes().to_vec())
      },
      sign: |mode, sk, message, context, addrnd| {
        let mut signature = vec![0; $profile::SIGNATURE_LENGTH];
        let out = <&mut [u8; $profile::SIGNATURE_LENGTH]>::try_from(signature.as_mut_slice()).expect("buffer");
        let fill = |random: &mut [u8]| {
          random.copy_from_slice(addrnd.expect("hedged randomness"));
          Ok(())
        };
        let result = match (mode, addrnd) {
          (Mode::Pure, None) => $secret::try_from_slice(sk)
            .expect("secret key")
            .sign_deterministic(message, context, out),
          (Mode::Pure, Some(_)) => $secret::try_from_slice(sk)
            .expect("secret key")
            .sign_with(message, context, fill, out),
          (Mode::Prehash, None) => $hash_secret::try_from_slice(sk)
            .expect("secret key")
            .sign_deterministic(message, context, out),
          (Mode::Prehash, Some(_)) => $hash_secret::try_from_slice(sk)
            .expect("secret key")
            .sign_with(message, context, fill, out),
        };
        result.expect("signing");
        signature
      },
      verify: |mode, pk, message, context, signature| match mode {
        Mode::Pure => $public::try_from_slice(pk)
          .expect("public key")
          .verify_with_context(message, context, signature)
          .is_ok(),
        Mode::Prehash => $hash_public::try_from_slice(pk)
          .expect("public key")
          .verify_with_context(message, context, signature)
          .is_ok(),
      },
    }
  };
}

#[expect(clippy::panic, reason = "unknown pinned ACVP metadata is a broken fixture")]
fn public(parameter_set: &str) -> Public {
  match parameter_set {
    "SLH-DSA-SHA2-128s" => public!(
      SlhDsaSha2_128s,
      SlhDsaSha2_128sPublicKey,
      SlhDsaSha2_128sSecretKey,
      HashSlhDsaSha2_128sWithSha256PublicKey,
      HashSlhDsaSha2_128sWithSha256SecretKey,
      "SHA2-256"
    ),
    "SLH-DSA-SHA2-128f" => public!(
      SlhDsaSha2_128f,
      SlhDsaSha2_128fPublicKey,
      SlhDsaSha2_128fSecretKey,
      HashSlhDsaSha2_128fWithSha256PublicKey,
      HashSlhDsaSha2_128fWithSha256SecretKey,
      "SHA2-256"
    ),
    "SLH-DSA-SHA2-192s" => public!(
      SlhDsaSha2_192s,
      SlhDsaSha2_192sPublicKey,
      SlhDsaSha2_192sSecretKey,
      HashSlhDsaSha2_192sWithSha512PublicKey,
      HashSlhDsaSha2_192sWithSha512SecretKey,
      "SHA2-512"
    ),
    "SLH-DSA-SHA2-192f" => public!(
      SlhDsaSha2_192f,
      SlhDsaSha2_192fPublicKey,
      SlhDsaSha2_192fSecretKey,
      HashSlhDsaSha2_192fWithSha512PublicKey,
      HashSlhDsaSha2_192fWithSha512SecretKey,
      "SHA2-512"
    ),
    "SLH-DSA-SHA2-256s" => public!(
      SlhDsaSha2_256s,
      SlhDsaSha2_256sPublicKey,
      SlhDsaSha2_256sSecretKey,
      HashSlhDsaSha2_256sWithSha512PublicKey,
      HashSlhDsaSha2_256sWithSha512SecretKey,
      "SHA2-512"
    ),
    "SLH-DSA-SHA2-256f" => public!(
      SlhDsaSha2_256f,
      SlhDsaSha2_256fPublicKey,
      SlhDsaSha2_256fSecretKey,
      HashSlhDsaSha2_256fWithSha512PublicKey,
      HashSlhDsaSha2_256fWithSha512SecretKey,
      "SHA2-512"
    ),
    "SLH-DSA-SHAKE-128s" => public!(
      SlhDsaShake128s,
      SlhDsaShake128sPublicKey,
      SlhDsaShake128sSecretKey,
      HashSlhDsaShake128sWithShake128PublicKey,
      HashSlhDsaShake128sWithShake128SecretKey,
      "SHAKE-128"
    ),
    "SLH-DSA-SHAKE-128f" => public!(
      SlhDsaShake128f,
      SlhDsaShake128fPublicKey,
      SlhDsaShake128fSecretKey,
      HashSlhDsaShake128fWithShake128PublicKey,
      HashSlhDsaShake128fWithShake128SecretKey,
      "SHAKE-128"
    ),
    "SLH-DSA-SHAKE-192s" => public!(
      SlhDsaShake192s,
      SlhDsaShake192sPublicKey,
      SlhDsaShake192sSecretKey,
      HashSlhDsaShake192sWithShake256PublicKey,
      HashSlhDsaShake192sWithShake256SecretKey,
      "SHAKE-256"
    ),
    "SLH-DSA-SHAKE-192f" => public!(
      SlhDsaShake192f,
      SlhDsaShake192fPublicKey,
      SlhDsaShake192fSecretKey,
      HashSlhDsaShake192fWithShake256PublicKey,
      HashSlhDsaShake192fWithShake256SecretKey,
      "SHAKE-256"
    ),
    "SLH-DSA-SHAKE-256s" => public!(
      SlhDsaShake256s,
      SlhDsaShake256sPublicKey,
      SlhDsaShake256sSecretKey,
      HashSlhDsaShake256sWithShake256PublicKey,
      HashSlhDsaShake256sWithShake256SecretKey,
      "SHAKE-256"
    ),
    "SLH-DSA-SHAKE-256f" => public!(
      SlhDsaShake256f,
      SlhDsaShake256fPublicKey,
      SlhDsaShake256fSecretKey,
      HashSlhDsaShake256fWithShake256PublicKey,
      HashSlhDsaShake256fWithShake256SecretKey,
      "SHAKE-256"
    ),
    _ => panic!("unrecognized parameter set"),
  }
}

/// The public profile an external ACVP case exercises: pure, or the RFC 9909
/// pairing. Other pre-hash functions reach only the internal interface.
fn public_mode(internal: bool, prehash: bool, hash: &str, public: Public) -> Option<Mode> {
  match (internal, prehash) {
    (true, _) => None,
    (false, false) => Some(Mode::Pure),
    (false, true) => (hash == public.prehash).then_some(Mode::Prehash),
  }
}

/// The DER OID and digest of an ACVP pre-hash function, from independent
/// hash implementations.
#[expect(clippy::panic, reason = "unknown pinned ACVP metadata is a broken fixture")]
fn prehash_digest(name: &str, message: &[u8]) -> ([u8; 11], Vec<u8>) {
  use digest::Digest as _;
  use tiny_keccak::Hasher as _;
  let (arc, digest) = match name {
    "SHA2-224" => (4, sha2::Sha224::digest(message).to_vec()),
    "SHA2-256" => (1, sha2::Sha256::digest(message).to_vec()),
    "SHA2-384" => (2, sha2::Sha384::digest(message).to_vec()),
    "SHA2-512" => (3, sha2::Sha512::digest(message).to_vec()),
    "SHA2-512/224" => (5, sha2::Sha512_224::digest(message).to_vec()),
    "SHA2-512/256" => (6, sha2::Sha512_256::digest(message).to_vec()),
    "SHA3-224" => (7, sha3::Sha3_224::digest(message).to_vec()),
    "SHA3-256" => (8, sha3::Sha3_256::digest(message).to_vec()),
    "SHA3-384" => (9, sha3::Sha3_384::digest(message).to_vec()),
    "SHA3-512" => (10, sha3::Sha3_512::digest(message).to_vec()),
    "SHAKE-128" => {
      let mut hash = tiny_keccak::Shake::v128();
      hash.update(message);
      let mut out = vec![0; 32];
      hash.finalize(&mut out);
      (11, out)
    }
    "SHAKE-256" => {
      let mut hash = tiny_keccak::Shake::v256();
      hash.update(message);
      let mut out = vec![0; 64];
      hash.finalize(&mut out);
      (12, out)
    }
    _ => panic!("unrecognized pre-hash function"),
  };
  (
    [0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, arc],
    digest,
  )
}

/// M' for one external ACVP case (FIPS 205 Algorithms 22-25), or the raw
/// message for the internal interface.
fn formatted_message(internal: bool, prehash: bool, hash: &str, message: &[u8], context: &[u8]) -> Vec<u8> {
  if internal {
    return message.to_vec();
  }
  let mut formatted = vec![u8::from(prehash), u8::try_from(context.len()).expect("ACVP context")];
  formatted.extend_from_slice(context);
  if prehash {
    let (oid, digest) = prehash_digest(hash, message);
    formatted.extend_from_slice(&oid);
    formatted.extend_from_slice(&digest);
  } else {
    formatted.extend_from_slice(message);
  }
  formatted
}

#[test]
fn acvp_key_generation_matches_all_cases() {
  let prompt = vectors("keyGen-prompt");
  let results = vectors("keyGen-expected");
  let mut count = 0usize;
  for group in groups(&prompt) {
    let internal = internal(group["parameterSet"].as_str().expect("parameter set"));
    for case in cases(group) {
      let want = expected(&results, group, case);
      let (sk_seed, sk_prf, pk_seed) = (hex(&case["skSeed"]), hex(&case["skPrf"]), hex(&case["pkSeed"]));
      let pk_root = (internal.keygen)(&sk_seed, &pk_seed);
      assert_eq!(
        [pk_seed.as_slice(), &pk_root].concat(),
        hex(&want["pk"]),
        "tcId {}",
        case["tcId"]
      );
      assert_eq!(
        [sk_seed.as_slice(), &sk_prf, &pk_seed, &pk_root].concat(),
        hex(&want["sk"]),
        "tcId {}",
        case["tcId"]
      );
      let (public_key, secret_key) = (public(group["parameterSet"].as_str().expect("parameter set")).generate)(
        &[sk_seed.as_slice(), &sk_prf, &pk_seed].concat(),
      );
      assert_eq!(public_key, hex(&want["pk"]), "tcId {}", case["tcId"]);
      assert_eq!(secret_key, hex(&want["sk"]), "tcId {}", case["tcId"]);
      assert_eq!(internal.n, pk_root.len());
      count = count.strict_add(1);
    }
  }
  assert_eq!(count, 120);
}

/// Run every sigGen case of `parameter_set` and variant through the internal
/// interface, comparing an independent SHA-256 of each signature with the
/// fixture's, and the first case of each external group that a public profile
/// covers through that profile. Returns the internal and public case counts.
fn acvp_signing(parameter_set: &str, deterministic: bool) -> (usize, usize) {
  use digest::Digest as _;
  let internal = internal(parameter_set);
  let public = public(parameter_set);
  let (mut internal_count, mut public_count) = (0usize, 0usize);
  let mut public_done = [false; 2];
  for case in signing_cases()
    .iter()
    .filter(|case| case.parameter_set == parameter_set && case.deterministic == deterministic)
  {
    let opt_rand = if deterministic {
      &case.sk[internal.n.strict_mul(2)..internal.n.strict_mul(3)]
    } else {
      case.addrnd
    };
    let message = formatted_message(case.internal, case.prehash, case.hash, case.message, case.context);
    let signature = (internal.sign)(case.sk, &[&message], opt_rand);
    assert_eq!(signature.len(), internal.signature_len);
    assert_eq!(
      sha2::Sha256::digest(&signature).as_slice(),
      case.signature_sha256,
      "{parameter_set} tgId {} tcId {}",
      case.id.0,
      case.id.1
    );
    internal_count = internal_count.strict_add(1);

    if let Some(mode) = public_mode(case.internal, case.prehash, case.hash, public)
      && !public_done[usize::from(case.prehash)]
    {
      let addrnd = (!deterministic).then_some(case.addrnd);
      let signature = (public.sign)(mode, case.sk, case.message, case.context, addrnd);
      assert_eq!(
        sha2::Sha256::digest(&signature).as_slice(),
        case.signature_sha256,
        "{parameter_set} public tgId {} tcId {}",
        case.id.0,
        case.id.1
      );
      public_done[usize::from(case.prehash)] = true;
      public_count = public_count.strict_add(1);
    }
  }
  (internal_count, public_count)
}

/// Run every sigVer case of `parameter_set` through the internal interface,
/// and every external case that a public profile covers through that
/// profile. Returns the internal and public case counts.
fn acvp_verification(parameter_set: &str) -> (usize, usize) {
  let internal = internal(parameter_set);
  let public = public(parameter_set);
  let (mut internal_count, mut public_count) = (0usize, 0usize);
  for case in verification_cases()
    .iter()
    .filter(|case| case.parameter_set == parameter_set)
  {
    let message = formatted_message(case.internal, case.prehash, case.hash, case.message, case.context);
    let accepted = (internal.verify)(case.pk, &[&message], case.signature);
    assert_eq!(
      accepted, case.passed,
      "{parameter_set} tgId {} tcId {}",
      case.id.0, case.id.1
    );
    internal_count = internal_count.strict_add(1);

    if let Some(mode) = public_mode(case.internal, case.prehash, case.hash, public) {
      let accepted = (public.verify)(mode, case.pk, case.message, case.context, case.signature);
      assert_eq!(
        accepted, case.passed,
        "{parameter_set} public tgId {} tcId {}",
        case.id.0, case.id.1
      );
      public_count = public_count.strict_add(1);
    }
  }
  (internal_count, public_count)
}

macro_rules! acvp_set_tests {
  ($($deterministic:ident, $hedged:ident, $verification:ident, $parameter_set:literal, $public_verification:literal;)*) => {$(
    /// 26 internal sigGen cases per variant; one public case each for the
    /// pure and RFC 9909 external groups.
    #[test]
    fn $deterministic() {
      assert_eq!(acvp_signing($parameter_set, true), (26, 2));
    }

    #[test]
    fn $hedged() {
      assert_eq!(acvp_signing($parameter_set, false), (26, 2));
    }

    #[test]
    fn $verification() {
      assert_eq!(acvp_verification($parameter_set), (42, $public_verification));
    }
  )*};
}

acvp_set_tests! {
  acvp_signing_sha2_128s_deterministic, acvp_signing_sha2_128s_hedged, acvp_verification_sha2_128s, "SLH-DSA-SHA2-128s", 15;
  acvp_signing_sha2_128f_deterministic, acvp_signing_sha2_128f_hedged, acvp_verification_sha2_128f, "SLH-DSA-SHA2-128f", 15;
  acvp_signing_sha2_192s_deterministic, acvp_signing_sha2_192s_hedged, acvp_verification_sha2_192s, "SLH-DSA-SHA2-192s", 15;
  acvp_signing_sha2_192f_deterministic, acvp_signing_sha2_192f_hedged, acvp_verification_sha2_192f, "SLH-DSA-SHA2-192f", 16;
  acvp_signing_sha2_256s_deterministic, acvp_signing_sha2_256s_hedged, acvp_verification_sha2_256s, "SLH-DSA-SHA2-256s", 15;
  acvp_signing_sha2_256f_deterministic, acvp_signing_sha2_256f_hedged, acvp_verification_sha2_256f, "SLH-DSA-SHA2-256f", 15;
  acvp_signing_shake_128s_deterministic, acvp_signing_shake_128s_hedged, acvp_verification_shake_128s, "SLH-DSA-SHAKE-128s", 15;
  acvp_signing_shake_128f_deterministic, acvp_signing_shake_128f_hedged, acvp_verification_shake_128f, "SLH-DSA-SHAKE-128f", 15;
  acvp_signing_shake_192s_deterministic, acvp_signing_shake_192s_hedged, acvp_verification_shake_192s, "SLH-DSA-SHAKE-192s", 15;
  acvp_signing_shake_192f_deterministic, acvp_signing_shake_192f_hedged, acvp_verification_shake_192f, "SLH-DSA-SHAKE-192f", 15;
  acvp_signing_shake_256s_deterministic, acvp_signing_shake_256s_hedged, acvp_verification_shake_256s, "SLH-DSA-SHAKE-256s", 15;
  acvp_signing_shake_256f_deterministic, acvp_signing_shake_256f_hedged, acvp_verification_shake_256f, "SLH-DSA-SHAKE-256f", 15;
}

/// A fixed key pair of the fast SHA-2 set for contract tests.
fn key_pair() -> (SlhDsaSha2_128fPublicKey, SlhDsaSha2_128fSecretKey) {
  SlhDsaSha2_128f::generate_keypair(|seeds| {
    for (byte, value) in seeds.iter_mut().zip(1u8..) {
      *byte = value;
    }
    Ok(())
  })
  .expect("key generation")
}

const SIGNATURE_LENGTH: usize = SlhDsaSha2_128f::SIGNATURE_LENGTH;

#[test]
fn over_long_context_fails_before_entropy_and_clears_the_signature() {
  let (public, secret) = key_pair();
  let context = [0x5a; 256];
  let mut signature = [0xa5; SIGNATURE_LENGTH];
  let mut requested = false;
  let result = secret.sign_with(
    b"message",
    &context,
    |_| {
      requested = true;
      Ok(())
    },
    &mut signature,
  );
  assert_eq!(result, Err(SlhDsaError::ContextTooLong));
  assert!(!requested, "entropy requested for a rejected context");
  assert_eq!(signature, [0; SIGNATURE_LENGTH]);

  signature.fill(0xa5);
  let hash_secret =
    HashSlhDsaSha2_128fWithSha256SecretKey::try_from_slice(secret.expose_secret().as_bytes()).expect("secret key");
  let result = hash_secret.sign_deterministic(b"message", &context, &mut signature);
  assert_eq!(result, Err(SlhDsaError::ContextTooLong));
  assert_eq!(signature, [0; SIGNATURE_LENGTH]);

  secret
    .sign_deterministic(b"message", &context[..255], &mut signature)
    .expect("a 255-byte context");
  public
    .verify_with_context(b"message", &context[..255], &signature)
    .expect("a 255-byte context verifies");
  assert!(public.verify_with_context(b"message", &context, &signature).is_err());
}

#[test]
fn entropy_failure_clears_the_signature_and_leaves_the_key_usable() {
  let (public, secret) = key_pair();
  let mut signature = [0xa5; SIGNATURE_LENGTH];
  let result = secret.sign_with(
    b"message",
    b"",
    |random| {
      random[0] = 1;
      Err(SlhDsaError::RandomGenerationFailed)
    },
    &mut signature,
  );
  assert_eq!(result, Err(SlhDsaError::RandomGenerationFailed));
  assert_eq!(signature, [0; SIGNATURE_LENGTH]);

  secret
    .sign_with(
      b"message",
      b"",
      |random| {
        random.fill(9);
        Ok(())
      },
      &mut signature,
    )
    .expect("hedged signing");
  public.verify(b"message", &signature).expect("hedged signature");
  let mut deterministic = [0; SIGNATURE_LENGTH];
  secret
    .sign_deterministic(b"message", b"", &mut deterministic)
    .expect("deterministic signing");
  assert_ne!(signature, deterministic, "hedged signing ignored its randomness");

  let failed = SlhDsaSha2_128f::generate_keypair(|seeds| {
    seeds[0] = 1;
    Err(SlhDsaError::RandomGenerationFailed)
  });
  assert!(matches!(failed, Err(SlhDsaError::RandomGenerationFailed)));
}

#[test]
fn pure_prehash_context_and_parameter_set_are_domain_separated() {
  use digest::Digest as _;
  let (public, secret) = key_pair();
  let encoded = secret.expose_secret();
  let hash_secret = HashSlhDsaSha2_128fWithSha256SecretKey::try_from_slice(encoded.as_bytes()).expect("secret key");
  let hash_public = HashSlhDsaSha2_128fWithSha256PublicKey::from_bytes(public.to_bytes());

  let mut pure = [0; SIGNATURE_LENGTH];
  secret
    .sign_deterministic(b"message", b"context", &mut pure)
    .expect("pure signing");
  let mut prehash = [0; SIGNATURE_LENGTH];
  hash_secret
    .sign_deterministic(b"message", b"context", &mut prehash)
    .expect("HashSLH-DSA signing");
  public
    .verify_with_context(b"message", b"context", &pure)
    .expect("pure signature");
  hash_public
    .verify_with_context(b"message", b"context", &prehash)
    .expect("HashSLH-DSA signature");

  assert!(hash_public.verify_with_context(b"message", b"context", &pure).is_err());
  assert!(public.verify_with_context(b"message", b"context", &prehash).is_err());
  assert!(public.verify_with_context(b"message", b"contexT", &pure).is_err());
  assert!(public.verify(b"message", &pure).is_err());

  let digest: [u8; 32] = sha2::Sha256::digest(b"message").into();
  let mut from_digest = [0; SIGNATURE_LENGTH];
  hash_secret
    .sign_prehash_deterministic(&digest, b"context", &mut from_digest)
    .expect("HashSLH-DSA digest signing");
  assert_eq!(from_digest, prehash);
  hash_public
    .verify_prehash(&digest, b"context", &prehash)
    .expect("HashSLH-DSA digest verification");

  // The same key bytes under the other hash family of the same shape.
  let shake = SlhDsaShake128fPublicKey::from_bytes(public.to_bytes());
  assert!(shake.verify_with_context(b"message", b"context", &pure).is_err());
}

#[test]
fn secret_import_checks_the_public_root_against_the_seeds() {
  let (public, secret) = key_pair();
  let encoded = *secret.expose_secret().as_bytes();
  let imported = SlhDsaSha2_128fSecretKey::try_from_slice(&encoded).expect("own encoding");
  assert_eq!(imported.public_key(), &public);
  assert_eq!(imported.expose_secret().as_bytes(), &encoded);

  // SK.seed, PK.seed, and PK.root determine one another; SK.prf is free.
  for (index, accepted) in [
    (0, false),
    (15, false),
    (16, true),
    (31, true),
    (32, false),
    (48, false),
    (63, false),
  ] {
    let mut altered = encoded;
    altered[index] ^= 1;
    assert_eq!(
      SlhDsaSha2_128fSecretKey::try_from_slice(&altered).is_ok(),
      accepted,
      "byte {index}"
    );
  }
  for len in [0, 63, 65] {
    let input = vec![0; len];
    assert_eq!(
      SlhDsaSha2_128fSecretKey::try_from_slice(&input).map(|_| ()),
      Err(SlhDsaKeyError::InvalidSecretKey)
    );
  }
  assert_eq!(
    SlhDsaSha2_128fPublicKey::try_from_slice(&encoded[..31]).map(|_| ()),
    Err(SlhDsaKeyError::InvalidPublicKey)
  );
}
