//! The AES-NI/PCLMUL tier of AES-GCM and AES-GCM-SIV matches Wycheproof and
//! RustCrypto.
//!
//! x86-64 hosts without AVX-512 select this tier: Intel hybrid parts, where
//! detection removes AVX-512, and AMD parts before Zen 4. Every CI x86-64 host
//! has AVX-512 and VAES, so this test removes the capabilities that detection
//! removes on an Intel hybrid part. The override must come before detection,
//! and each integration test file is its own process, so this file holds one
//! test.

#![cfg(all(
  feature = "aes-gcm",
  feature = "aes-gcm-siv",
  feature = "std",
  target_arch = "x86_64",
  not(feature = "portable-only"),
  not(miri)
))]

use aes_gcm::aead::{Aead as Oracle, KeyInit as _, Payload, array::Array, consts::U12};
use rscrypto::{
  AeadGather, Aes128Gcm, Aes128GcmKey, Aes128GcmSiv, Aes128GcmSivKey, Aes256Gcm, Aes256GcmKey, Aes256GcmSiv,
  Aes256GcmSivKey,
  aead::{Nonce96, expert::AeadWithNonce, introspect},
  platform::{Caps, Detected, caps::x86, expert},
};
use serde_json::Value;

mod common;
use common::decode_hex_vec;

const AES_GCM: &str = include_str!("../testdata/aead/wycheproof/aes_gcm_test.json");
const AES_GCM_SIV: &str = include_str!("../testdata/aead/wycheproof/aes_gcm_siv_test.json");

/// Every AVX-512 and AVX10 capability: what detection removes on an Intel hybrid part.
const HYBRID_REMOVED: Caps = x86::AVX512F
  .union(x86::AVX512DQ)
  .union(x86::AVX512IFMA)
  .union(x86::AVX512CD)
  .union(x86::AVX512BW)
  .union(x86::AVX512VL)
  .union(x86::AVX512VBMI)
  .union(x86::AVX512VBMI2)
  .union(x86::AVX512VNNI)
  .union(x86::AVX512BITALG)
  .union(x86::AVX512VPOPCNTDQ)
  .union(x86::AVX512BF16)
  .union(x86::AVX512FP16)
  .union(x86::AVX512VP2INTERSECT)
  .union(x86::AVX10_1)
  .union(x86::AVX10_2);

/// Open each valid case of the 96-bit-nonce, 128-bit-tag groups for one key
/// size, and seal its message back to its ciphertext and tag. Reject each
/// invalid case and clear its buffer. Return the number of cases.
fn check_wycheproof<C: AeadWithNonce<Nonce = Nonce96>>(json: &str, key_bits: u64, new: impl Fn(&[u8]) -> C) -> usize {
  let suite: Value = serde_json::from_str(json).expect("the pinned Wycheproof JSON parses");
  let mut cases = 0usize;
  for group in suite["testGroups"].as_array().expect("Wycheproof test groups") {
    if group["keySize"].as_u64() != Some(key_bits)
      || group["ivSize"].as_u64() != Some(96)
      || group["tagSize"].as_u64() != Some(128)
    {
      continue;
    }
    for test in group["tests"].as_array().expect("Wycheproof tests") {
      let field = |name: &str| decode_hex_vec(test[name].as_str().expect(name));
      let id = test["tcId"].as_u64().expect("numeric tcId");
      let cipher = new(&field("key"));
      let nonce = Nonce96::from_bytes(field("iv").try_into().expect("96-bit nonce"));
      let (aad, msg, ct, tag_bytes) = (field("aad"), field("msg"), field("ct"), field("tag"));
      let tag = C::tag_from_slice(&tag_bytes).expect("128-bit tag");
      let mut buffer = ct.clone();
      let result = test["result"].as_str();
      assert!(
        matches!(result, Some("valid" | "invalid")),
        "tcId {id} has the unsupported result {result:?}"
      );
      if result == Some("valid") {
        let opened = cipher.decrypt_in_place(&nonce, &aad, &mut buffer, &tag);
        assert!(opened.is_ok(), "tcId {id} must open: {opened:?}");
        assert_eq!(buffer, msg, "tcId {id} opened");
        let mut sealed = msg;
        let sealed_tag = cipher.encrypt_in_place(&nonce, &aad, &mut sealed).expect("valid seal");
        assert_eq!(
          (sealed.as_slice(), sealed_tag.as_ref()),
          (ct.as_slice(), tag_bytes.as_slice()),
          "tcId {id} sealed"
        );
      } else {
        assert!(
          cipher.decrypt_in_place(&nonce, &aad, &mut buffer, &tag).is_err(),
          "tcId {id} must not open"
        );
        assert!(buffer.iter().all(|&byte| byte == 0), "tcId {id} cleared its buffer");
      }
      cases = cases.strict_add(1);
    }
  }
  cases
}

fn patterned(len: usize, seed: u8) -> Vec<u8> {
  (0..len)
    .map(|i| i.to_le_bytes()[0].wrapping_mul(0x9d).wrapping_add(seed) ^ (i >> 8).to_le_bytes()[0])
    .collect()
}

/// Plaintext lengths through every loop and tail of the tier, and AAD lengths
/// on both sides of the 64-byte wide-GHASH threshold.
fn shapes() -> impl Iterator<Item = (usize, usize)> {
  let lengths = (0..=129).chain([255, 256, 257, 1023, 1024, 1025, 4095, 4096, 4097, 16_385]);
  lengths.flat_map(|len| [0, 13, 64, 65, 300].map(|aad_len| (len, aad_len)))
}

/// Seal and open each shape, compare with RustCrypto, and reject a changed tag.
fn check_oracle<C: AeadWithNonce<Nonce = Nonce96>>(name: &str, cipher: &C, oracle: &impl Oracle<NonceSize = U12>) {
  let nonce = *b"aes-ni tier!";
  for (len, aad_len) in shapes() {
    let (plaintext, aad) = (patterned(len, 0x31), patterned(aad_len, 0xc4));
    let expected = oracle
      .encrypt(
        &Array(nonce),
        Payload {
          msg: &plaintext,
          aad: &aad,
        },
      )
      .expect("RustCrypto seals the fixture");
    let (expected_text, expected_tag) = expected.split_at(len);

    let mut buffer = plaintext.clone();
    let tag = cipher
      .encrypt_in_place(&Nonce96::from_bytes(nonce), &aad, &mut buffer)
      .expect("valid seal");
    assert_eq!(
      (buffer.as_slice(), tag.as_ref()),
      (expected_text, expected_tag),
      "{name} seal, len={len}, aad={aad_len}"
    );
    cipher
      .decrypt_in_place(&Nonce96::from_bytes(nonce), &aad, &mut buffer, &tag)
      .expect("a fresh seal opens");
    assert_eq!(buffer, plaintext, "{name} open, len={len}, aad={aad_len}");

    let mut changed = expected_tag.to_vec();
    changed[len % 16] ^= 0x04;
    let changed = C::tag_from_slice(&changed).expect("128-bit tag");
    let mut buffer = expected_text.to_vec();
    assert!(
      cipher
        .decrypt_in_place(&Nonce96::from_bytes(nonce), &aad, &mut buffer, &changed)
        .is_err(),
      "{name} must reject a changed tag, len={len}, aad={aad_len}"
    );
    assert!(buffer.iter().all(|&byte| byte == 0), "{name} cleared a rejected open");
  }
}

/// Gather three slices of each shape into one message, as RustCrypto seals it.
fn check_gather<C: AeadGather<Nonce = Nonce96>>(name: &str, cipher: &C, oracle: &impl Oracle<NonceSize = U12>) {
  let nonce = *b"gather tier!";
  for (len, aad_len) in shapes() {
    let (plaintext, aad) = (patterned(len, 0x5e), patterned(aad_len, 0x17));
    let expected = oracle
      .encrypt(
        &Array(nonce),
        Payload {
          msg: &plaintext,
          aad: &aad,
        },
      )
      .expect("RustCrypto seals the fixture");
    let (first, rest) = plaintext.split_at(len / 3);
    let (second, third) = rest.split_at(len / 3);
    let mut output = vec![0; len];
    let tag = cipher
      .encrypt_gather(&Nonce96::from_bytes(nonce), &aad, [first, second, third], &mut output)
      .expect("valid gather");
    assert_eq!(
      (output.as_slice(), tag.as_ref()),
      expected.split_at(len),
      "{name} gather, len={len}, aad={aad_len}"
    );
  }
}

#[test]
fn aes_ni_tier_matches_wycheproof_and_rustcrypto() {
  let host = expert::detect_uncached();
  if !host.caps.has(x86::AESNI.union(x86::PCLMUL_READY)) {
    eprintln!("skipping the AES-NI/PCLMUL tier: this CPU lacks AES-NI, PCLMULQDQ, or SSSE3");
    return;
  }
  expert::try_set_override(Some(Detected {
    caps: host.caps.difference(HYBRID_REMOVED),
    arch: host.arch,
  }))
  .expect("the override precedes detection in this process");
  for (name, backend) in [
    ("AES-128-GCM", introspect::aes128gcm_backend()),
    ("AES-256-GCM", introspect::aes256gcm_backend()),
    ("AES-128-GCM-SIV", introspect::aes128gcmsiv_backend()),
    ("AES-256-GCM-SIV", introspect::aes256gcmsiv_backend()),
  ] {
    assert_eq!(backend, "x86_64/aesni+pclmul", "{name} must run the tier under test");
  }

  let gcm = check_wycheproof(AES_GCM, 128, |key| {
    Aes128Gcm::new(&Aes128GcmKey::from_bytes(key.try_into().expect("AES-128 key")))
  }) + check_wycheproof(AES_GCM, 256, |key| {
    Aes256Gcm::new(&Aes256GcmKey::from_bytes(key.try_into().expect("AES-256 key")))
  });
  let siv = check_wycheproof(AES_GCM_SIV, 128, |key| {
    Aes128GcmSiv::new(&Aes128GcmSivKey::from_bytes(key.try_into().expect("AES-128 key")))
  }) + check_wycheproof(AES_GCM_SIV, 256, |key| {
    Aes256GcmSiv::new(&Aes256GcmSivKey::from_bytes(key.try_into().expect("AES-256 key")))
  });
  // The counts of tests/aead_wycheproof.rs for the same groups.
  assert_eq!((gcm, siv), (79 + 54, 136 + 66), "Wycheproof case counts");

  let (key128, key256) = ([0x6b; 16], [0xd2; 32]);
  let aes128gcm = Aes128Gcm::new(&Aes128GcmKey::from_bytes(key128));
  let aes256gcm = Aes256Gcm::new(&Aes256GcmKey::from_bytes(key256));
  let oracle128 = aes_gcm::Aes128Gcm::new(&Array(key128));
  let oracle256 = aes_gcm::Aes256Gcm::new(&Array(key256));
  check_oracle("AES-128-GCM", &aes128gcm, &oracle128);
  check_oracle("AES-256-GCM", &aes256gcm, &oracle256);
  check_gather("AES-128-GCM", &aes128gcm, &oracle128);
  check_gather("AES-256-GCM", &aes256gcm, &oracle256);
  check_oracle(
    "AES-128-GCM-SIV",
    &Aes128GcmSiv::new(&Aes128GcmSivKey::from_bytes(key128)),
    &aes_gcm_siv::Aes128GcmSiv::new(&Array(key128)),
  );
  check_oracle(
    "AES-256-GCM-SIV",
    &Aes256GcmSiv::new(&Aes256GcmSivKey::from_bytes(key256)),
    &aes_gcm_siv::Aes256GcmSiv::new(&Array(key256)),
  );
}
