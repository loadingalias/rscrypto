//! AES-GCM and AEGIS-256 through the RV64 scalar-crypto backend (Zkne).
//!
//! The test removes Zvkned from the detected capabilities before first use, so that each AEAD
//! selects the scalar backend. It runs only where the CPU reports Zkne. AES-GCM also needs a scalar
//! carry-less multiply (Zbc or Zbkc). This binary holds one test because the override is
//! process-wide.

#![cfg(all(
  target_arch = "riscv64",
  not(feature = "portable-only"),
  feature = "aes-gcm",
  feature = "std"
))]

use rscrypto::aead::{Nonce96, expert::AeadWithNonce};
use rscrypto::platform::{Detected, caps::riscv, expert};
use rscrypto::{Aes128Gcm, Aes128GcmKey, Aes128GcmTag, Aes256Gcm, Aes256GcmKey, Aes256GcmTag};
use serde_json::Value;

mod common;
use common::decode_hex_vec;

const AES_GCM: &str = include_str!("../testdata/aead/wycheproof/aes_gcm_test.json");

fn field(test: &Value, name: &str) -> Vec<u8> {
  decode_hex_vec(test[name].as_str().expect("hex field"))
}

#[test]
fn scalar_crypto_aeads_match_known_answers() {
  let detected = expert::detect_uncached();
  if !detected.caps.has(riscv::ZKNE) {
    return;
  }
  let forced = Detected {
    caps: detected.caps.difference(riscv::ZVKNED),
    arch: detected.arch,
  };
  expert::try_set_override(Some(forced)).expect("a subset of the detected capabilities");
  if detected.caps.has(riscv::ZBC) || detected.caps.has(riscv::ZBKC) {
    aes_gcm_matches_wycheproof();
  }
  #[cfg(feature = "aegis256")]
  aegis256_matches_draft_and_wycheproof();
}

fn aes_gcm_matches_wycheproof() {
  let suite: Value = serde_json::from_str(AES_GCM).expect("Wycheproof AES-GCM JSON");
  let mut checked = 0usize;
  for group in suite["testGroups"].as_array().expect("groups") {
    if group["ivSize"].as_u64() != Some(96) || group["tagSize"].as_u64() != Some(128) {
      continue;
    }
    for test in group["tests"].as_array().expect("tests") {
      let (key, iv, aad, msg, ct, tag) = (
        field(test, "key"),
        field(test, "iv"),
        field(test, "aad"),
        field(test, "msg"),
        field(test, "ct"),
        field(test, "tag"),
      );
      let valid = test["result"].as_str() == Some("valid");
      let nonce = Nonce96::from_bytes(iv.as_slice().try_into().expect("96-bit IV"));
      let tag: [u8; 16] = tag.as_slice().try_into().expect("128-bit tag");
      let mut opened = ct.clone();
      let result = match key.len() {
        16 => {
          let cipher = Aes128Gcm::new(&Aes128GcmKey::from_bytes(key.as_slice().try_into().expect("key")));
          if valid {
            let mut sealed = msg.clone();
            let sealed_tag = cipher.encrypt_in_place(&nonce, &aad, &mut sealed).expect("seal");
            assert_eq!(
              (sealed.as_slice(), sealed_tag.as_bytes()),
              (ct.as_slice(), &tag),
              "tcId {}",
              test["tcId"]
            );
          }
          cipher.decrypt_in_place(&nonce, &aad, &mut opened, &Aes128GcmTag::from_bytes(tag))
        }
        32 => {
          let cipher = Aes256Gcm::new(&Aes256GcmKey::from_bytes(key.as_slice().try_into().expect("key")));
          if valid {
            let mut sealed = msg.clone();
            let sealed_tag = cipher.encrypt_in_place(&nonce, &aad, &mut sealed).expect("seal");
            assert_eq!(
              (sealed.as_slice(), sealed_tag.as_bytes()),
              (ct.as_slice(), &tag),
              "tcId {}",
              test["tcId"]
            );
          }
          cipher.decrypt_in_place(&nonce, &aad, &mut opened, &Aes256GcmTag::from_bytes(tag))
        }
        _ => continue,
      };
      assert_eq!(result.is_ok(), valid, "tcId {}", test["tcId"]);
      if valid {
        assert_eq!(opened, msg, "tcId {}", test["tcId"]);
      }
      checked = checked.strict_add(1);
    }
  }
  assert!(checked > 100, "Wycheproof AES-GCM coverage");
}

#[cfg(feature = "aegis256")]
fn aegis256_matches_draft_and_wycheproof() {
  use rscrypto::aead::Nonce256;
  use rscrypto::{Aegis256, Aegis256Key, Aegis256Tag};

  const AEGIS256: &str = include_str!("../testdata/aead/wycheproof/aegis256_test.json");
  let open = |key: &[u8], nonce: &[u8], aad: &[u8], msg: &[u8], ct: &[u8], tag: &[u8], valid: bool, case: &str| {
    let cipher = Aegis256::new(&Aegis256Key::from_bytes(key.try_into().expect("256-bit key")));
    let nonce = Nonce256::from_bytes(nonce.try_into().expect("256-bit nonce"));
    let tag: [u8; 16] = tag.try_into().expect("128-bit tag");
    if valid {
      let mut sealed = msg.to_vec();
      let sealed_tag = cipher.encrypt_in_place(&nonce, aad, &mut sealed).expect("seal");
      assert_eq!((sealed.as_slice(), sealed_tag.as_bytes()), (ct, &tag), "{case}");
    }
    let mut opened = ct.to_vec();
    let result = cipher.decrypt_in_place(&nonce, aad, &mut opened, &Aegis256Tag::from_bytes(tag));
    assert_eq!(result.is_ok(), valid, "{case}");
    if valid {
      assert_eq!(opened, msg, "{case}");
    }
  };

  // draft-irtf-cfrg-aegis-aead, Appendix A.3: the key and nonce of every AEGIS-256 vector.
  let key = decode_hex_vec("1001000000000000000000000000000000000000000000000000000000000000");
  let nonce = decode_hex_vec("1000020000000000000000000000000000000000000000000000000000000000");
  for (index, (msg, aad, ct, tag)) in [
    (
      "00000000000000000000000000000000",
      "",
      "754fc3d8c973246dcc6d741412a4b236",
      "3fe91994768b332ed7f570a19ec5896e",
    ),
    ("", "", "", "e3def978a0f054afd1e761d7553afba3"),
    (
      "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f",
      "0001020304050607",
      "f373079ed84b2709faee373584585d60accd191db310ef5d8b11833df9dec711",
      "8d86f91ee606e9ff26a01b64ccbdd91d",
    ),
    (
      "000102030405060708090a0b0c0d",
      "0001020304050607",
      "f373079ed84b2709faee37358458",
      "c60b9c2d33ceb058f96e6dd03c215652",
    ),
    (
      "101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f3031323334353637",
      "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20212223242526272829",
      "57754a7d09963e7c787583a2e7b859bb24fa1e04d49fd550b2511a358e3bca252a9b1b8b30cc4a67",
      "ab8a7d53fd0e98d727accca94925e128",
    ),
  ]
  .into_iter()
  .enumerate()
  {
    let (msg, aad, ct, tag) = (
      decode_hex_vec(msg),
      decode_hex_vec(aad),
      decode_hex_vec(ct),
      decode_hex_vec(tag),
    );
    open(
      &key,
      &nonce,
      &aad,
      &msg,
      &ct,
      &tag,
      true,
      &format!("A.3 vector {}", index.strict_add(1)),
    );
  }

  let suite: Value = serde_json::from_str(AEGIS256).expect("Wycheproof AEGIS-256 JSON");
  let mut checked = 0usize;
  for group in suite["testGroups"].as_array().expect("groups") {
    for test in group["tests"].as_array().expect("tests") {
      let valid = test["result"].as_str() == Some("valid");
      let case = format!("tcId {}", test["tcId"]);
      open(
        &field(test, "key"),
        &field(test, "iv"),
        &field(test, "aad"),
        &field(test, "msg"),
        &field(test, "ct"),
        &field(test, "tag"),
        valid,
        &case,
      );
      checked = checked.strict_add(1);
    }
  }
  assert!(checked > 400, "Wycheproof AEGIS-256 coverage");
}
