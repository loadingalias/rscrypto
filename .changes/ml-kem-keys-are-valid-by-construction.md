---
"rscrypto" = "minor"
---

ML-KEM keys are valid by construction. `MlKem*EncapsulationKey` and `MlKem*DecapsulationKey` no longer have `from_bytes`, `validate`, or `FromStr`. Build them with `try_from_slice`, DER import, serde, or key generation; each runs the FIPS 203 input check once (sections 7.2 and 7.3). Deserialization of these keys now rejects an invalid key; before, it accepted any value of the right length. `MlKem*Ciphertext::validate` is removed, because the type check is the length. Encapsulation and decapsulation no longer repeat the key check, so each decapsulation skips one SHA3-256 of the embedded encapsulation key. Preparation cannot fail: `prepare`, `prepare_in`, `MlKem*::prepare_encapsulation_key`, `MlKem*::prepare_decapsulation_key`, and the conversions to prepared keys (now `From`, before `TryFrom`) return the prepared key. Prepared ML-KEM decapsulation and `MlKem*::decapsulate_prepared` return the shared secret.
