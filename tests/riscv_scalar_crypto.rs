//! AES-GCM through the RV64 scalar-crypto backend (Zkne).
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
