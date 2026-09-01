# Test evidence

This map identifies the independent evidence behind each primitive family and
the important boundaries that remain outside it. Test filenames are the stable
entry points; individual corpus files remain owned by `testdata/` and the test
readers.

## Coverage map

| Family | Independent evidence | Boundary or gap |
| --- | --- | --- |
| CRC-16/24/32/64 | Property tests plus `crc`, `crc-fast`, `crc32fast`, `crc32c`, and `crc64fast` oracles where applicable | Checksums have no Wycheproof suites. |
| SHA-2, SHA-3, SHAKE, cSHAKE, KMAC | NIST vectors, vendored `.blb` corpora, and RustCrypto differentials | KMAC128 has no mapped Wycheproof suite. |
| BLAKE2, BLAKE3, Ascon hash/XOF | Upstream or NIST corpora plus independent differentials | Hash functions have no invalid ciphertext or signature class. |
| XXH3 and RapidHash | Upstream-crate differentials, streaming tests, and fuzzing | Non-cryptographic; no Wycheproof suite applies. |
| HMAC, HKDF, PBKDF2, Poly1305 | Official vectors, Wycheproof where the public profile maps, properties, and differentials | HMAC-SHA3 has no mapped Wycheproof suite. Only explicitly exposed tag widths map. |
| Argon2 and scrypt | Published vectors, RustCrypto differentials, kernel/parallel tests, and Miri | No Wycheproof PHC-string suite exists. |
| AEADs | Wycheproof where variants and nonce widths map, official vectors, RustCrypto oracles, corruption tests, and backend equivalence | Unsupported key or nonce sizes are filtered at the typed API boundary. Wycheproof's older Ascon variant does not match NIST Ascon-AEAD128. |
| ECDSA, Ed25519, X25519 | RFC or official vectors, Wycheproof, RustCrypto/dalek oracles, properties, and fuzzing | ASN.1, JWK, or variable-length profiles are excluded where the public API accepts fixed arrays only. |
| ML-KEM-512/768/1024 | NIST ACVP key-generation, encapsulation, decapsulation, and key-check vectors plus `fips203` differentials | No vendored Wycheproof ML-KEM suite is mapped. |
| RSA signatures, encryption, and parsing | NIST CAVP, Wycheproof, RustCrypto oracles, profile-confusion, allocation, and leakage tests | Public APIs expose fixed SHA-2 profiles rather than every Wycheproof parameter combination. |
| Dispatch and fallback | Portable-versus-accelerated differential tests across lengths, tails, and vectored input | Cross-compilation alone is not runtime evidence. |

The WebSocket accept digest has the RFC 6455 example, private SHA-1 known-answer
tests, RustCrypto differential tests, and fuzzing. It is compatibility-only and
makes no collision-resistance or authentication claim.

## Run the evidence

```sh
just test --all
just feature-contracts runtime
just test-fuzz
```

Specialized Miri, target, constant-time, and leakage recipes are listed by
`just --list`. `just check` validates test-vector provenance and the feature,
target, benchmark, and constant-time manifests.

A passing vector proves behavior for that vector. Stronger assurance comes from
combining published vectors, a separate implementation, properties, hostile
inputs, fuzzing, portable-versus-accelerated equivalence, and target execution.
