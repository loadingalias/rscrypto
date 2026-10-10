---
"rscrypto" = "minor"
---

Add RFC 9935 key encodings for ML-KEM. `EncapsulationKey::from_spki_der` and `to_spki_der` handle SubjectPublicKeyInfo with the FIPS 203 modulus check. `DecapsulationKey::from_pkcs8_der` and `from_pkcs8_der_in` accept the seed, expanded, and both forms in a version 1 container or a version 2 container with a public key; an expanded key also passes a deterministic pairwise consistency check, which rejects a secret vector that FIPS 203's hash check accepts. Raw `try_from_slice` keeps exactly the FIPS 203 checks. The new `MlKem512Seed`, `MlKem768Seed`, and `MlKem1024Seed` owners keep the recommended `d || z` seed form, and `DecapsulationKey::to_pkcs8_der_into` writes the expanded form. Import failures use the new `MlKemKeyError`; KEM operations keep `MlKemError`. Nothing allocates except the `_in` constructors.
