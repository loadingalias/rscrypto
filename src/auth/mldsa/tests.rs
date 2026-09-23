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
    "keyGen-prompt" => include_str!("../../../testdata/mldsa/acvp/keyGen-prompt.json"),
    "keyGen-expected" => include_str!("../../../testdata/mldsa/acvp/keyGen-expectedResults.json"),
    "sigGen-prompt" => include_str!("../../../testdata/mldsa/acvp/sigGen-prompt.json"),
    "sigGen-expected" => include_str!("../../../testdata/mldsa/acvp/sigGen-expectedResults.json"),
    "sigVer-prompt" => include_str!("../../../testdata/mldsa/acvp/sigVer-prompt.json"),
    "sigVer-expected" => include_str!("../../../testdata/mldsa/acvp/sigVer-expectedResults.json"),
    _ => panic!("unknown vector file"),
  };
  serde_json::from_str(source).expect("official vector JSON")
}

fn groups(value: &Value) -> &[Value] {
  value["testGroups"].as_array().expect("vector groups")
}
fn cases(value: &Value) -> &[Value] {
  value["tests"].as_array().expect("vector cases")
}

#[expect(clippy::panic, reason = "unknown pinned ACVP metadata is a broken fixture")]
fn params(group: &Value) -> Parameters {
  match group["parameterSet"].as_str().expect("parameter set") {
    "ML-DSA-44" => P44,
    "ML-DSA-65" => P65,
    "ML-DSA-87" => P87,
    _ => panic!("unrecognized parameter set"),
  }
}

macro_rules! with_parameters {
  ($p:expr, $function:ident ($($arg:expr),*)) => {
    match $p.k {
      4 => portable::$function::<4, 4>($($arg),*),
      6 => portable::$function::<6, 5>($($arg),*),
      8 => portable::$function::<8, 7>($($arg),*),
      _ => panic!("invalid test dimensions"),
    }
  };
}

#[expect(clippy::panic, reason = "unknown pinned ACVP metadata is a broken fixture")]
fn prehash(name: &str, message: &[u8]) -> (MlDsaPrehashAlgorithm, Vec<u8>) {
  use MlDsaPrehashAlgorithm as A;
  use digest::Digest as _;
  use tiny_keccak::Hasher as _;
  match name {
    "SHA2-224" => (A::Sha224, sha2::Sha224::digest(message).to_vec()),
    "SHA2-256" => (A::Sha256, sha2::Sha256::digest(message).to_vec()),
    "SHA2-384" => (A::Sha384, sha2::Sha384::digest(message).to_vec()),
    "SHA2-512" => (A::Sha512, sha2::Sha512::digest(message).to_vec()),
    "SHA2-512/224" => (A::Sha512_224, sha2::Sha512_224::digest(message).to_vec()),
    "SHA2-512/256" => (A::Sha512_256, sha2::Sha512_256::digest(message).to_vec()),
    "SHA3-224" => (A::Sha3_224, sha3::Sha3_224::digest(message).to_vec()),
    "SHA3-256" => (A::Sha3_256, sha3::Sha3_256::digest(message).to_vec()),
    "SHA3-384" => (A::Sha3_384, sha3::Sha3_384::digest(message).to_vec()),
    "SHA3-512" => (A::Sha3_512, sha3::Sha3_512::digest(message).to_vec()),
    "SHAKE-128" => {
      let mut hash = tiny_keccak::Shake::v128();
      hash.update(message);
      let mut out = vec![0; 32];
      hash.finalize(&mut out);
      (A::Shake128, out)
    }
    "SHAKE-256" => {
      let mut hash = tiny_keccak::Shake::v256();
      hash.update(message);
      let mut out = vec![0; 64];
      hash.finalize(&mut out);
      (A::Shake256, out)
    }
    _ => panic!("unrecognized ACVP prehash"),
  }
}

fn mu(group: &Value, case: &Value, tr: &[u8]) -> [u8; 64] {
  if group["externalMu"] == true {
    return hex(&case["mu"]).try_into().expect("64-byte mu");
  }
  let message = hex(&case["message"]);
  if group["signatureInterface"] == "internal" {
    let mut mu = [0; 64];
    sampling::hash(&[tr, &message], &mut mu);
    return mu;
  }
  let context = hex(&case["context"]);
  if group["preHash"] == "preHash" {
    let (algorithm, digest) = prehash(case["hashAlg"].as_str().expect("hash identifier"), &message);
    let prehash = MlDsaPrehash::new(algorithm, &digest).expect("valid digest");
    let mut mu = [0; 64];
    representative(tr, &[], &context, Some(prehash), &mut mu).expect("valid context");
    mu
  } else {
    let mut mu = [0; 64];
    representative(tr, &message, &context, None, &mut mu).expect("valid context");
    mu
  }
}

#[test]
fn acvp_key_generation_all_parameter_sets() {
  let prompt = vectors("keyGen-prompt");
  let expected = vectors("keyGen-expected");
  let mut count = 0usize;
  for group in groups(&prompt) {
    let p = params(group);
    let answers = groups(&expected)
      .iter()
      .find(|g| g["tgId"] == group["tgId"])
      .expect("matching group");
    for case in cases(group) {
      let answer = cases(answers)
        .iter()
        .find(|t| t["tcId"] == case["tcId"])
        .expect("matching case");
      let seed = hex(&case["seed"]).try_into().expect("32-byte seed");
      let expected_pk = hex(&answer["pk"]);
      let expected_sk = hex(&answer["sk"]);
      let mut pk = vec![0; expected_pk.len()];
      let mut sk = vec![0; expected_sk.len()];
      with_parameters!(p, keygen(&seed, p, &mut pk, &mut sk)).expect("key generation");
      assert!(pk == expected_pk, "public key mismatch: {}", case["tcId"]);
      assert!(sk == expected_sk, "secret key mismatch: {}", case["tcId"]);
      let mut recovered = vec![0; pk.len()];
      with_parameters!(p, validate_secret(&sk, p, &mut recovered)).expect("official key validation");
      assert!(recovered == expected_pk, "import public-key mismatch");
      count = count.strict_add(1);
    }
  }
  assert_eq!(count, 75);
}

#[test]
fn acvp_signature_generation_all_modes() {
  let prompt = vectors("sigGen-prompt");
  let expected = vectors("sigGen-expected");
  let mut count = 0usize;
  for group in groups(&prompt) {
    let p = params(group);
    let answers = groups(&expected)
      .iter()
      .find(|g| g["tgId"] == group["tgId"])
      .expect("matching group");
    for case in cases(group) {
      let answer = cases(answers)
        .iter()
        .find(|t| t["tcId"] == case["tcId"])
        .expect("matching case");
      let sk = hex(&case["sk"]);
      let expected_signature = hex(&answer["signature"]);
      let mu = mu(group, case, &sk[64..128]);
      let random = if group["deterministic"] == true {
        [0; 32]
      } else {
        hex(&case["rnd"]).try_into().expect("32-byte randomness")
      };
      let mut signature = vec![0; p.signature_len()];
      with_parameters!(p, sign(&sk, &mu, &random, p, &mut signature)).expect("official signing case");
      assert!(signature == expected_signature, "signature mismatch: {}", case["tcId"]);
      // Official sigGen successes also cover prehash combinations missing from
      // the upstream sigVer positive cases; expected bytes remain independent.
      let mut pk = vec![0; 32usize.strict_add(p.k.strict_mul(320))];
      with_parameters!(p, validate_secret(&sk, p, &mut pk)).expect("official secret import");
      with_parameters!(p, verify(&pk, &mu, &expected_signature, p)).expect("official positive signature");
      count = count.strict_add(1);
    }
  }
  assert_eq!(count, 360);
}

fn exhaustion_and_reuse<const K: usize, const L: usize>(
  secret: &[u8],
  mu: &[u8; 64],
  random: &[u8; 32],
  p: Parameters,
  expected: &[u8],
) {
  let mut rejecting = p;
  // Fault injection through an existing private input: gamma2 - beta becomes
  // zero, so every candidate fails the production norm check. Sampling, NTTs,
  // and the full production retry loop still execute. Public APIs cannot
  // select these nonstandard parameters.
  rejecting.beta = p.gamma2;
  let mut signature = vec![0xa5; p.signature_len()];
  assert_eq!(
    portable::sign::<K, L>(secret, mu, random, rejecting, &mut signature),
    Err(MlDsaError::RejectionLimit)
  );
  assert!(signature.iter().all(|&byte| byte == 0), "rejected output escaped");
  portable::sign::<K, L>(secret, mu, random, p, &mut signature).expect("compact signing after exhaustion");
  assert_eq!(signature, expected, "compact key reuse must match ACVP");

  let mut state = portable::SigningState::<K, L>::zero();
  state.decode(secret, p).expect("official secret decode");
  let mut matrix = portable::Matrix::<K, L>::zero();
  matrix.expand_into(&secret[..32]).expect("public matrix expansion");
  signature.fill(0xa5);
  assert_eq!(
    portable::sign_with_state(secret, mu, random, rejecting, &mut signature, &state, Some(&matrix)),
    Err(MlDsaError::RejectionLimit)
  );
  assert!(
    signature.iter().all(|&byte| byte == 0),
    "prepared rejected output escaped"
  );
  portable::sign_with_state(secret, mu, random, p, &mut signature, &state, Some(&matrix))
    .expect("prepared signing after exhaustion");
  assert_eq!(signature, expected, "prepared state reuse must match ACVP");
}

#[test]
fn signing_exhaustion_clears_output_and_preserves_reusable_state() {
  let prompt = vectors("sigGen-prompt");
  let expected = vectors("sigGen-expected");
  for p in [P44, P65, P87] {
    let group = groups(&prompt)
      .iter()
      .find(|group| params(group).k == p.k)
      .expect("official parameter-set group");
    let case = &cases(group)[0];
    let answers = groups(&expected)
      .iter()
      .find(|answer| answer["tgId"] == group["tgId"])
      .expect("matching group");
    let answer = cases(answers)
      .iter()
      .find(|answer| answer["tcId"] == case["tcId"])
      .expect("matching case");
    let secret = hex(&case["sk"]);
    let mu = mu(group, case, &secret[64..128]);
    let random = if group["deterministic"] == true {
      [0; 32]
    } else {
      hex(&case["rnd"]).try_into().expect("32-byte randomness")
    };
    let signature = hex(&answer["signature"]);
    match p.k {
      4 => exhaustion_and_reuse::<4, 4>(&secret, &mu, &random, p, &signature),
      6 => exhaustion_and_reuse::<6, 5>(&secret, &mu, &random, p, &signature),
      _ => exhaustion_and_reuse::<8, 7>(&secret, &mu, &random, p, &signature),
    }
  }
}

#[test]
fn acvp_verification_accepts_and_rejects() {
  let prompt = vectors("sigVer-prompt");
  let expected = vectors("sigVer-expected");
  let mut count = 0usize;
  for group in groups(&prompt) {
    let p = params(group);
    let answers = groups(&expected)
      .iter()
      .find(|g| g["tgId"] == group["tgId"])
      .expect("matching group");
    for case in cases(group) {
      let answer = cases(answers)
        .iter()
        .find(|t| t["tcId"] == case["tcId"])
        .expect("matching case");
      let pk = hex(&case["pk"]);
      let signature = hex(&case["signature"]);
      let mut tr = [0; 64];
      sampling::hash(&[&pk], &mut tr);
      let mu = mu(group, case, &tr);
      let result = with_parameters!(p, verify(&pk, &mu, &signature, p));
      assert_eq!(
        result.is_ok(),
        answer["testPassed"].as_bool().expect("expected verdict"),
        "case {}",
        case["tcId"]
      );
      count = count.strict_add(1);
    }
  }
  assert_eq!(count, 180);
}
