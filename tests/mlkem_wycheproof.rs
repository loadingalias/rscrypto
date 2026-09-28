#![cfg(feature = "ml-kem")]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::alloc::Global;

use rscrypto::{
  Kem, MlKem512, MlKem512Ciphertext, MlKem512DecapsulationKey, MlKem512EncapsulationKey, MlKem768, MlKem768Ciphertext,
  MlKem768DecapsulationKey, MlKem768EncapsulationKey, MlKem1024, MlKem1024Ciphertext, MlKem1024DecapsulationKey,
  MlKem1024EncapsulationKey, MlKemError,
};
use serde_json::Value;

mod common;
use common::decode_hex_vec;

// C2SP Wycheproof ML-KEM corpora; provenance in testdata/mlkem/wycheproof/README.md.

#[derive(Debug, Default, PartialEq, Eq)]
struct Tally {
  valid: usize,
  invalid: usize,
}

impl Tally {
  fn record(&mut self, valid: bool) {
    if valid {
      self.valid = self.valid.strict_add(1);
    } else {
      self.invalid = self.invalid.strict_add(1);
    }
  }
}

fn field<'a>(value: &'a Value, name: &str) -> &'a str {
  value[name].as_str().expect("Wycheproof field must be a string")
}

fn bytes(value: &Value, name: &str) -> Vec<u8> {
  decode_hex_vec(field(value, name))
}

fn is_valid(case: &Value) -> bool {
  let result = field(case, "result");
  assert!(
    result == "valid" || result == "invalid",
    "unexpected Wycheproof result {result}"
  );
  result == "valid"
}

/// Parse a corpus, check its header, and yield every case with its group.
fn cases<'a>(suite: &'a Value, schema: &str, parameter_set: &str) -> impl Iterator<Item = (&'a Value, &'a Value)> {
  assert_eq!(field(suite, "algorithm"), "ML-KEM");
  assert_eq!(field(suite, "schema"), schema);
  suite["testGroups"]
    .as_array()
    .expect("Wycheproof testGroups must be an array")
    .iter()
    .inspect(move |group| assert_eq!(field(group, "parameterSet"), parameter_set))
    .flat_map(|group| {
      group["tests"]
        .as_array()
        .expect("Wycheproof tests must be an array")
        .iter()
        .map(move |case| (group, case))
    })
}

fn total(suite: &Value) -> usize {
  usize::try_from(suite["numberOfTests"].as_u64().expect("numberOfTests")).expect("count fits usize")
}

macro_rules! wycheproof_suite {
  (
    $keygen:ident, $combined:ident, $encaps:ident, $decaps:ident,
    $profile:ty, $ek:ty, $dk:ty, $ct:ty, $parameter_set:literal, $size:literal,
    keygen = $keygen_counts:expr, combined = $combined_counts:expr,
    encaps = $encaps_counts:expr, decaps = $decaps_counts:expr $(,)?
  ) => {
    fn keypair_from_seed(seed: &[u8]) -> Option<($ek, $dk)> {
      let seed: [u8; <$profile>::KEY_GENERATION_RANDOM_SIZE] = seed.try_into().ok()?;
      Some(
        <$profile>::generate_keypair(|out| {
          out.copy_from_slice(&seed);
          Ok::<(), MlKemError>(())
        })
        .expect("key generation from a full seed must succeed"),
      )
    }

    #[test]
    fn $keygen() {
      let suite: Value = serde_json::from_str(include_str!(concat!(
        "../testdata/mlkem/wycheproof/mlkem_",
        $size,
        "_keygen_seed_test.json"
      )))
      .unwrap();
      let mut tally = Tally::default();
      for (_, case) in cases(&suite, "mlkem_keygen_seed_test_schema.json", $parameter_set) {
        let tc_id = &case["tcId"];
        assert!(is_valid(case), "keygen tcId {tc_id} has no negative form");
        let (ek, dk) = keypair_from_seed(&bytes(case, "seed")).expect("keygen seed length");
        assert!(ek.as_bytes() == bytes(case, "ek").as_slice(), "keygen tcId {tc_id} ek");
        assert!(
          dk.expose_secret().as_bytes() == bytes(case, "dk").as_slice(),
          "keygen tcId {tc_id} dk"
        );
        #[cfg(feature = "alloc")]
        {
          let seed = bytes(case, "seed");
          let (ek, dk) = <$profile>::generate_keypair_in(
            |out| {
              out.copy_from_slice(&seed);
              Ok(())
            },
            Global,
          )
          .expect("allocated key generation from a full seed must succeed");
          assert!(
            ek.as_bytes() == bytes(case, "ek").as_slice(),
            "keygen_in tcId {tc_id} ek"
          );
          assert!(
            dk.expose_secret().as_bytes() == bytes(case, "dk").as_slice(),
            "keygen_in tcId {tc_id} dk"
          );
        }
        tally.record(true);
      }
      assert_eq!(tally.valid.strict_add(tally.invalid), total(&suite));
      assert_eq!(tally, $keygen_counts);
    }

    /// Key generation from `d || z`, then decapsulation of the given ciphertext.
    /// Invalid cases use a seed or ciphertext of the wrong length, which the typed
    /// API must refuse. Valid cases include implicit rejection of modified ciphertexts.
    #[test]
    fn $combined() {
      let suite: Value = serde_json::from_str(include_str!(concat!(
        "../testdata/mlkem/wycheproof/mlkem_",
        $size,
        "_test.json"
      )))
      .unwrap();
      let mut tally = Tally::default();
      for (_, case) in cases(&suite, "mlkem_test_schema.json", $parameter_set) {
        let tc_id = &case["tcId"];
        let keys = keypair_from_seed(&bytes(case, "seed"));
        let ciphertext = <$ct>::try_from_slice(&bytes(case, "c")).ok();
        match (is_valid(case), keys, ciphertext) {
          (true, Some((ek, dk)), Some(ciphertext)) => {
            assert!(
              ek.as_bytes() == bytes(case, "ek").as_slice(),
              "combined tcId {tc_id} ek"
            );
            let shared = <$profile>::decapsulate(&dk, &ciphertext).expect("decapsulation of a well-formed input");
            assert!(
              shared.expose_secret().as_bytes() == bytes(case, "K").as_slice(),
              "combined tcId {tc_id} K"
            );
            #[cfg(feature = "alloc")]
            {
              let prepared = dk.prepare_in(Global).expect("a generated key must prepare");
              let shared = prepared
                .decapsulate(&ciphertext)
                .expect("decapsulation of a well-formed input");
              assert!(
                shared.expose_secret().as_bytes() == bytes(case, "K").as_slice(),
                "combined prepare_in tcId {tc_id} K"
              );
            }
          }
          (valid, keys, ciphertext) => assert!(
            !valid && (keys.is_none() || ciphertext.is_none()),
            "combined tcId {tc_id}: valid={valid} but seed accepted={} ciphertext accepted={}",
            keys.is_some(),
            ciphertext.is_some()
          ),
        }
        tally.record(is_valid(case));
      }
      assert_eq!(tally.valid.strict_add(tally.invalid), total(&suite));
      assert_eq!(tally, $combined_counts);
    }

    /// Encapsulation with a fixed message. Invalid cases are unreduced
    /// coefficients (the FIPS 203 modulus check) and wrong key lengths.
    #[test]
    fn $encaps() {
      let suite: Value = serde_json::from_str(include_str!(concat!(
        "../testdata/mlkem/wycheproof/mlkem_",
        $size,
        "_encaps_test.json"
      )))
      .unwrap();
      let mut tally = Tally::default();
      for (_, case) in cases(&suite, "mlkem_encaps_test_schema.json", $parameter_set) {
        let tc_id = &case["tcId"];
        let key = <$ek>::try_from_slice(&bytes(case, "ek"));
        if !is_valid(case) {
          assert!(key.is_err(), "encaps tcId {tc_id}: invalid encapsulation key accepted");
          tally.record(false);
          continue;
        }
        let key = key.expect("valid encapsulation key must import");
        let message: [u8; <$profile>::ENCAPSULATION_RANDOM_SIZE] = bytes(case, "m").try_into().expect("message length");
        let (ciphertext, shared) = <$profile>::encapsulate(&key, |out| {
          out.copy_from_slice(&message);
          Ok::<(), MlKemError>(())
        })
        .expect("encapsulation with a valid key must succeed");
        assert!(
          ciphertext.as_bytes() == bytes(case, "c").as_slice(),
          "encaps tcId {tc_id} c"
        );
        assert!(
          shared.expose_secret().as_bytes() == bytes(case, "K").as_slice(),
          "encaps tcId {tc_id} K"
        );
        tally.record(true);
      }
      assert_eq!(tally.valid.strict_add(tally.invalid), total(&suite));
      assert_eq!(tally, $encaps_counts);
    }

    /// Decapsulation with an expanded decapsulation key. Invalid cases are wrong
    /// lengths and keys whose embedded encapsulation key or hash is corrupted.
    #[test]
    fn $decaps() {
      let suite: Value = serde_json::from_str(include_str!(concat!(
        "../testdata/mlkem/wycheproof/mlkem_",
        $size,
        "_semi_expanded_decaps_test.json"
      )))
      .unwrap();
      let mut tally = Tally::default();
      for (_, case) in cases(&suite, "mlkem_semi_expanded_decaps_test_schema.json", $parameter_set) {
        let tc_id = &case["tcId"];
        let key = <$dk>::try_from_slice(&bytes(case, "dk")).ok();
        let ciphertext = <$ct>::try_from_slice(&bytes(case, "c")).ok();
        #[cfg(feature = "alloc")]
        {
          let boxed = <$dk>::try_from_slice_in(&bytes(case, "dk"), Global).ok();
          assert_eq!(
            boxed.as_deref().map(|key| *key.expose_secret().as_bytes()),
            key.as_ref().map(|key| *key.expose_secret().as_bytes()),
            "decaps tcId {tc_id}: allocated import disagrees with import"
          );
          if let (true, Some(boxed), Some(ciphertext)) = (is_valid(case), &boxed, &ciphertext) {
            let prepared = boxed.prepare_in(Global).expect("a valid key must prepare");
            let shared = prepared
              .decapsulate(ciphertext)
              .expect("decapsulation of a well-formed input");
            assert!(
              shared.expose_secret().as_bytes() == bytes(case, "K").as_slice(),
              "decaps try_from_slice_in tcId {tc_id} K"
            );
          }
        }
        let shared = match (&key, &ciphertext) {
          (Some(key), Some(ciphertext)) => <$profile>::decapsulate(key, ciphertext).ok(),
          _ => None,
        };
        if is_valid(case) {
          let shared = shared.expect("valid decapsulation input must succeed");
          assert!(
            shared.expose_secret().as_bytes() == bytes(case, "K").as_slice(),
            "decaps tcId {tc_id} K"
          );
        } else {
          assert!(
            shared.is_none(),
            "decaps tcId {tc_id}: invalid input produced a shared secret"
          );
          // Preparation must reject a full-length corrupted key it is handed unvalidated.
          #[cfg(feature = "alloc")]
          if let Ok(raw) = <[u8; <$dk>::LENGTH]>::try_from(bytes(case, "dk").as_slice()) {
            if key.is_none() {
              assert_eq!(
                <$dk>::from_bytes(raw).prepare_in(Global).err(),
                Some(MlKemError::InvalidDecapsulationKey),
                "decaps tcId {tc_id}: prepare_in accepted a corrupted key"
              );
            }
          }
        }
        tally.record(is_valid(case));
      }
      assert_eq!(tally.valid.strict_add(tally.invalid), total(&suite));
      assert_eq!(tally, $decaps_counts);
    }
  };
}

mod mlkem512 {
  use super::*;
  wycheproof_suite!(
    keygen_seed,
    combined,
    encaps,
    semi_expanded_decaps,
    MlKem512,
    MlKem512EncapsulationKey,
    MlKem512DecapsulationKey,
    MlKem512Ciphertext,
    "ML-KEM-512",
    "512",
    keygen = Tally { valid: 100, invalid: 0 },
    combined = Tally {
      valid: 160,
      invalid: 40
    },
    encaps = Tally {
      valid: 133,
      invalid: 128
    },
    decaps = Tally { valid: 3, invalid: 6 },
  );
}

mod mlkem768 {
  use super::*;
  wycheproof_suite!(
    keygen_seed,
    combined,
    encaps,
    semi_expanded_decaps,
    MlKem768,
    MlKem768EncapsulationKey,
    MlKem768DecapsulationKey,
    MlKem768Ciphertext,
    "ML-KEM-768",
    "768",
    keygen = Tally { valid: 100, invalid: 0 },
    combined = Tally {
      valid: 161,
      invalid: 40
    },
    encaps = Tally {
      valid: 133,
      invalid: 132
    },
    decaps = Tally { valid: 3, invalid: 6 },
  );
}

mod mlkem1024 {
  use super::*;
  wycheproof_suite!(
    keygen_seed,
    combined,
    encaps,
    semi_expanded_decaps,
    MlKem1024,
    MlKem1024EncapsulationKey,
    MlKem1024DecapsulationKey,
    MlKem1024Ciphertext,
    "ML-KEM-1024",
    "1024",
    keygen = Tally { valid: 100, invalid: 0 },
    combined = Tally {
      valid: 162,
      invalid: 40
    },
    encaps = Tally {
      valid: 133,
      invalid: 136
    },
    decaps = Tally { valid: 3, invalid: 6 },
  );
}
