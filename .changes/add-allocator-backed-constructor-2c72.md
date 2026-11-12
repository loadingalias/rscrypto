---
"rscrypto" = "minor"
---

Add allocator-backed constructors for composite ML-DSA secret keys: `MlDsa65Ed25519::generate_keypair_in` and `MlDsa65EcdsaP256::generate_keypair_in` (and `try_generate_keypair_in` with `getrandom`), and `try_from_slice_in` and `from_pkcs8_der_in` on `MlDsa65Ed25519SecretKey` and `MlDsa65EcdsaP256SecretKey`. They write the ML-DSA-65 seed and expanded key and the Ed25519 or ECDSA P-256 key directly into memory from a caller-selected `Allocator` and return `Box<_, A>`, so moving the key moves only a pointer and leaves no by-value copy, and signing through the box reads the key in place. Each returns the same errors as its by-value form, and a rejected key is cleared and freed before the call returns.
