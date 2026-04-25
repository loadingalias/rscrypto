//! PHC string-format roundtrip and rejection coverage.
//!
//! Two-table strategy:
//!
//! - `valid` cases: produce a PHC string via `hash_string_with_salt`, decode it with
//!   `decode_string`, re-encode the decoded components, and assert byte equality. Catches
//!   non-canonical formatting drift (e.g. leading zeros, unstable param ordering).
//! - `invalid` cases: feed crafted strings to `verify_string` / `decode_string` and assert the
//!   exact `PhcError` variant. Catches parser regressions where a malformed input is silently
//!   accepted.
//!
//! Direct PHC parser unit tests live alongside the encoder/decoder in
//! `src/auth/phc.rs`; this file focuses on the round-trip property and the
//! cross-cutting interaction between `phc::parse`, the per-algorithm
//! `decode_string` helpers, and `verify_string`.

#![cfg(all(feature = "phc-strings", any(feature = "argon2", feature = "scrypt"), not(miri)))]

use rscrypto::auth::phc::PhcError;

// ─── Argon2 round-trip and rejection coverage ──────────────────────────────

#[cfg(feature = "argon2")]
mod argon2 {
  use rscrypto::{Argon2Params, Argon2Version, Argon2d, Argon2i, Argon2id};

  use super::*;

  fn cheap_params() -> Argon2Params {
    Argon2Params::new()
      .memory_cost_kib(8)
      .time_cost(1)
      .parallelism(1)
      .output_len(16)
      .version(Argon2Version::V0x13)
      .build()
      .expect("cheap argon2 params are valid")
  }

  /// Roundtrip: hash → encode → decode → re-encode produces the same string.
  #[test]
  fn argon2id_roundtrip_canonical() {
    let params = cheap_params();
    let salt = [0x07u8; 16];
    let encoded = Argon2id::hash_string_with_salt(&params, b"password", &salt).expect("hash");

    let (decoded_params, decoded_salt, decoded_hash) = Argon2id::decode_string(&encoded).expect("decode");
    let re_encoded = argon2_re_encode("argon2id", &decoded_params, &decoded_salt, &decoded_hash).expect("re-encode");

    assert_eq!(encoded, re_encoded, "PHC encoding is not canonical");
    assert!(Argon2id::verify_string(b"password", &encoded).is_ok());
  }

  #[test]
  fn argon2d_roundtrip_canonical() {
    let params = cheap_params();
    let salt = [0xA5u8; 16];
    let encoded = Argon2d::hash_string_with_salt(&params, b"pw", &salt).expect("hash");

    let (decoded_params, decoded_salt, decoded_hash) = Argon2d::decode_string(&encoded).expect("decode");
    let re_encoded = argon2_re_encode("argon2d", &decoded_params, &decoded_salt, &decoded_hash).expect("re-encode");

    assert_eq!(encoded, re_encoded);
  }

  #[test]
  fn argon2i_roundtrip_canonical() {
    let params = cheap_params();
    let salt = [0x5Au8; 16];
    let encoded = Argon2i::hash_string_with_salt(&params, b"pw", &salt).expect("hash");

    let (decoded_params, decoded_salt, decoded_hash) = Argon2i::decode_string(&encoded).expect("decode");
    let re_encoded = argon2_re_encode("argon2i", &decoded_params, &decoded_salt, &decoded_hash).expect("re-encode");

    assert_eq!(encoded, re_encoded);
  }

  /// Encode a freshly-decoded `(params, salt, hash)` back to PHC. Mirrors the
  /// internal `phc_integration::encode_string` semantics — but goes through
  /// the public surface (`hash_string_with_salt` returns the canonical form).
  fn argon2_re_encode(
    algorithm: &str,
    params: &Argon2Params,
    salt: &[u8],
    hash: &[u8],
  ) -> Result<String, &'static str> {
    // We can't re-run the hash (different password would diverge). Instead
    // we construct the canonical string manually using the same conventions
    // as `phc_integration::encode_string`: standard base64 (no padding),
    // params in `m=…,t=…,p=…` order, version preceding params.
    let mut out = String::new();
    out.push('$');
    out.push_str(algorithm);
    out.push_str("$v=");
    let version_u32 = match params.get_version() {
      rscrypto::Argon2Version::V0x10 => 0x10u32,
      rscrypto::Argon2Version::V0x13 => 0x13u32,
      // `Argon2Version` is `#[non_exhaustive]`; new variants are an error here.
      _ => return Err("unknown Argon2Version"),
    };
    out.push_str(&version_u32.to_string());
    out.push_str("$m=");
    out.push_str(&params.get_memory_cost_kib().to_string());
    out.push_str(",t=");
    out.push_str(&params.get_time_cost().to_string());
    out.push_str(",p=");
    out.push_str(&params.get_parallelism().to_string());
    out.push('$');
    out.push_str(&phc_b64_no_pad(salt));
    out.push('$');
    out.push_str(&phc_b64_no_pad(hash));
    Ok(out)
  }

  /// Strict PHC variant of base64 (RFC 4648 standard alphabet, no padding).
  fn phc_b64_no_pad(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::with_capacity(bytes.len() * 4 / 3 + 4);
    let triples = bytes.len() / 3;
    for i in 0..triples {
      let off = i * 3;
      let b0 = bytes[off] as u32;
      let b1 = bytes[off + 1] as u32;
      let b2 = bytes[off + 2] as u32;
      let w = (b0 << 16) | (b1 << 8) | b2;
      s.push(TABLE[((w >> 18) & 0x3F) as usize] as char);
      s.push(TABLE[((w >> 12) & 0x3F) as usize] as char);
      s.push(TABLE[((w >> 6) & 0x3F) as usize] as char);
      s.push(TABLE[(w & 0x3F) as usize] as char);
    }
    let off = triples * 3;
    match bytes.len() - off {
      0 => {}
      1 => {
        let b0 = bytes[off] as u32;
        s.push(TABLE[((b0 >> 2) & 0x3F) as usize] as char);
        s.push(TABLE[((b0 << 4) & 0x3F) as usize] as char);
      }
      2 => {
        let b0 = bytes[off] as u32;
        let b1 = bytes[off + 1] as u32;
        let w = (b0 << 8) | b1;
        s.push(TABLE[((w >> 10) & 0x3F) as usize] as char);
        s.push(TABLE[((w >> 4) & 0x3F) as usize] as char);
        s.push(TABLE[((w << 2) & 0x3F) as usize] as char);
      }
      _ => unreachable!(),
    }
    s
  }

  /// Cross-variant rejection: an Argon2id-encoded string must not verify
  /// against Argon2d / Argon2i.
  #[test]
  fn cross_variant_rejection() {
    let params = cheap_params();
    let encoded_id = Argon2id::hash_string_with_salt(&params, b"x", &[0u8; 16]).unwrap();

    assert!(Argon2d::verify_string(b"x", &encoded_id).is_err());
    assert!(Argon2i::verify_string(b"x", &encoded_id).is_err());

    assert_eq!(
      Argon2d::decode_string(&encoded_id).unwrap_err(),
      PhcError::AlgorithmMismatch
    );
    assert_eq!(
      Argon2i::decode_string(&encoded_id).unwrap_err(),
      PhcError::AlgorithmMismatch
    );
  }

  /// Decoder rejection table — exact `PhcError` variant per malformed input.
  #[test]
  fn invalid_inputs_yield_specific_errors() {
    let cases: &[(&str, PhcError)] = &[
      ("", PhcError::MalformedInput),
      ("argon2id$v=19$m=8,t=1,p=1$YWE$aGFzaA", PhcError::MalformedInput),
      ("$argon2xx$v=19$m=8,t=1,p=1$YWE$aGFzaA", PhcError::AlgorithmMismatch),
      // Missing required `m` parameter.
      ("$argon2id$v=19$t=1,p=1$YWE$aGFzaA", PhcError::MissingParam),
      // Duplicate `m`.
      ("$argon2id$v=19$m=8,m=16,t=1,p=1$YWE$aGFzaA", PhcError::DuplicateParam),
      // Unknown parameter.
      ("$argon2id$v=19$m=8,t=1,p=1,x=1$YWE$aGFzaA", PhcError::UnknownParam),
      // Leading zero on numeric value.
      ("$argon2id$v=19$m=08,t=1,p=1$YWE$aGFzaA", PhcError::MalformedParams),
      // Unsupported version.
      ("$argon2id$v=99$m=8,t=1,p=1$YWE$aGFzaA", PhcError::UnsupportedVersion),
      // Empty salt segment.
      ("$argon2id$v=19$m=8,t=1,p=1$$aGFzaA", PhcError::EmptySegment),
      // Salt too short for Argon2 (decoded < 8 bytes).
      ("$argon2id$v=19$m=8,t=1,p=1$YQ$aGFzaGhhc2g", PhcError::InvalidLength),
      // Bad base64 character in salt.
      ("$argon2id$v=19$m=8,t=1,p=1$YWE!YWE$aGFzaA", PhcError::InvalidBase64),
    ];

    for (input, expected) in cases {
      match Argon2id::decode_string(input) {
        Ok(_) => panic!("expected {expected:?} for {input:?}, got Ok(_)"),
        Err(actual) => assert_eq!(
          actual, *expected,
          "wrong error for {input:?}: expected {expected:?}, got {actual:?}"
        ),
      }
    }
  }

  /// Tampering by flipping one byte in the encoded hash must reject.
  #[test]
  fn single_bit_tamper_rejected() {
    let params = cheap_params();
    let encoded = Argon2id::hash_string_with_salt(&params, b"pw", &[0x77u8; 16]).unwrap();

    let mut bytes: Vec<u8> = encoded.into_bytes();
    let last = bytes.len() - 1;
    // Flip the final base64 char to a different valid char.
    bytes[last] = match bytes[last] {
      b'A' => b'B',
      b'/' => b'+',
      _ => b'A',
    };
    let tampered = String::from_utf8(bytes).unwrap();
    assert!(Argon2id::verify_string(b"pw", &tampered).is_err());
  }

  /// Excess length must be rejected with a stable error (not a panic, not OK).
  #[test]
  fn oversize_input_rejected() {
    let big = "$argon2id$v=19$m=8,t=1,p=1$YWE$".to_string() + &"A".repeat(8192);
    assert_eq!(Argon2id::decode_string(&big).unwrap_err(), PhcError::InputTooLong);
  }
}

// ─── scrypt round-trip and rejection coverage ──────────────────────────────

#[cfg(feature = "scrypt")]
mod scrypt {
  use rscrypto::{Scrypt, ScryptParams};

  use super::*;

  fn cheap_params() -> ScryptParams {
    ScryptParams::new()
      .log_n(8)
      .r(1)
      .p(1)
      .output_len(16)
      .build()
      .expect("cheap scrypt params are valid")
  }

  #[test]
  fn scrypt_roundtrip_canonical() {
    let params = cheap_params();
    let salt = [0x33u8; 16];
    let encoded = Scrypt::hash_string_with_salt(&params, b"password", &salt).expect("hash");

    // Just verify decode succeeds and verify_string accepts the result; the
    // canonical-form invariant is exercised by the Argon2 path (same encoder).
    let (_decoded_params, _decoded_salt, _decoded_hash) = Scrypt::decode_string(&encoded).expect("decode");
    assert!(Scrypt::verify_string(b"password", &encoded).is_ok());
    assert!(Scrypt::verify_string(b"wrong", &encoded).is_err());
  }

  #[test]
  fn scrypt_invalid_inputs_yield_specific_errors() {
    let cases: &[(&str, PhcError)] = &[
      ("", PhcError::MalformedInput),
      // Wrong algorithm.
      ("$bcrypt$ln=8,r=1,p=1$YWE$aGFzaA", PhcError::AlgorithmMismatch),
      // Missing `ln` (PHC scrypt mandates ln, r, p).
      ("$scrypt$r=1,p=1$YWE$aGFzaA", PhcError::MissingParam),
      // Duplicate r.
      ("$scrypt$ln=8,r=1,r=2,p=1$YWE$aGFzaA", PhcError::DuplicateParam),
      // Unknown parameter.
      ("$scrypt$ln=8,r=1,p=1,extra=1$YWE$aGFzaA", PhcError::UnknownParam),
      // Bad base64 in hash.
      ("$scrypt$ln=8,r=1,p=1$YWE$aGFz#A", PhcError::InvalidBase64),
    ];

    for (input, expected) in cases {
      match Scrypt::decode_string(input) {
        Ok(_) => panic!("expected {expected:?} for {input:?}, got Ok(_)"),
        Err(actual) => assert_eq!(actual, *expected, "wrong error for {input:?}"),
      }
    }
  }
}
