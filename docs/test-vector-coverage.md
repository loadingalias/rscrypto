# Test Vector Coverage Ledger

This ledger makes negative-vector coverage visible per primitive instead of
relying on broad claims like "covered by oracles". It is built from the actual
`tests/` and `testdata/` files.

Wycheproof suites are vendored from `C2SP/wycheproof` `testvectors_v1/`. The
upstream project describes the JSON suites as implementation-agnostic test
vectors for known attacks and edge cases, and recommends mapping the vectors to
the concrete inputs and outputs of each cryptography API.

## Negative Coverage Policy

- Prefer Wycheproof when its current JSON suite maps directly to a public
  rscrypto API.
- Use standards vectors for positive KATs.
- Use differential oracles for broad valid-input equivalence.
- Use explicit negative behavior tests for parsing, authentication failure,
  invalid encodings, low-order points, and protocol/profile confusion.
- Hashes and checksums do not have an "invalid ciphertext" style API. Their
  negative coverage is limited to parser/format boundaries and dispatch
  equivalence.

## Coverage Table

| Primitive | Positive vector / oracle coverage | Negative behavior coverage | Gaps / notes |
| --- | --- | --- | --- |
| CRC-16 family | `tests/crc16_properties.rs`; oracle crates in dev-deps | Property tests cover combine/reset/streaming boundaries | No Wycheproof suite applies |
| CRC-24 family | `tests/crc24_properties.rs` | Property tests cover combine/reset/streaming boundaries | No Wycheproof suite applies |
| CRC-32 family | `tests/crc32_properties.rs`; `crc`, `crc32fast`, `crc32c` oracles | Property tests cover combine/reset/streaming boundaries | No Wycheproof suite applies |
| CRC-64 family | `tests/crc64_properties.rs`; `crc64fast` and `crc-fast` oracles | Property tests cover combine/reset/streaming boundaries | No Wycheproof suite applies |
| SHA-224 | `tests/sha2_official_vectors.rs`, `tests/sha256_official_vectors.rs`, `testdata/sha2/sha224.blb` | Streaming/oneshot and dispatch equivalence | No invalid input class |
| SHA-256 | `tests/sha2_official_vectors.rs`, `tests/sha256_official_vectors.rs`, `tests/sha256_differential.rs`, `testdata/sha2/sha256.blb` | Streaming/oneshot and dispatch equivalence | Wycheproof has HMAC/HKDF/PBKDF2 SHA-256 suites, not raw SHA-256 |
| SHA-384 | `tests/sha2_official_vectors.rs`, `testdata/sha2/sha384.blb` | Streaming/oneshot and dispatch equivalence | No invalid input class |
| SHA-512 | `tests/sha2_official_vectors.rs`, `tests/sha512_differential.rs`, `testdata/sha2/sha512.blb` | Streaming/oneshot and dispatch equivalence | No invalid input class |
| SHA-512/256 | `tests/sha2_official_vectors.rs`, `testdata/sha2/sha512_256.blb` | Streaming/oneshot and dispatch equivalence | No invalid input class |
| SHA3-224/256/384/512 | `tests/sha3_official_vectors.rs`, `tests/sha3_differential.rs`, `testdata/sha3/sha3_*.blb` | Streaming/oneshot and dispatch equivalence | No invalid input class |
| SHAKE128 | `tests/sha3_official_vectors.rs`, `tests/shake128_differential.rs`, `testdata/sha3/shake128.blb` | XOF output-length boundaries | No invalid input class |
| SHAKE256 | `tests/sha3_official_vectors.rs`, `tests/shake256_differential.rs`, `testdata/sha3/shake256.blb` | XOF output-length boundaries | No invalid input class |
| cSHAKE128 / cSHAKE256 | `tests/cshake256_nist_vectors.rs`, `tests/cshake256_differential.rs`, `src/hashes/crypto/cshake.rs` unit tests | Customization/function-name boundary coverage through oracle tests; cSHAKE128 also covers SHAKE128 equivalence for empty function/custom strings | No Wycheproof suite currently mapped |
| KMAC128 | `tests/kmac128_nist_vectors.rs`, `tests/kmac128_differential.rs` | NIST samples, streaming/one-shot equivalence, empty/corrupted tag rejection, and `tiny-keccak` differential coverage | No Wycheproof KMAC128 suite currently mapped |
| KMAC256 | `tests/kmac256_nist_vectors.rs`, `tests/kmac256_differential.rs` | `tests/kmac_wycheproof.rs` covers Wycheproof KMAC256 no-customization valid/invalid tags across 128/256/512-bit outputs | Customization coverage stays in NIST/differential tests |
| BLAKE2b | `tests/blake2_official_vectors.rs`, `tests/blake2_differential.rs`, `testdata/blake2/blake2b.blb` | Keyed/unkeyed differential coverage | No invalid input class |
| BLAKE2s | `tests/blake2_official_vectors.rs`, `tests/blake2_differential.rs`, `testdata/blake2/blake2s.blb` | Keyed/unkeyed differential coverage | No invalid input class |
| BLAKE3 | `tests/blake3_official_vectors.rs`, `tests/blake3_differential.rs`, `testdata/blake3/test_vectors.*` | XOF/keyed/derive-key differential coverage | No invalid input class |
| Ascon hash/XOF/CXOF | `tests/ascon_official_vectors.rs`, `tests/ascon_hash_oracle.rs`, `tests/ascon_cxof_vectors.rs`, `tests/ascon_differential.rs`, `testdata/ascon/*.blb` | XOF and customization boundary coverage through oracle tests | No invalid input class |
| XXH3 | `tests/xxh3_differential.rs` | Seeded/streaming/property-style differential coverage | Non-cryptographic; no Wycheproof suite applies |
| RapidHash | `tests/rapidhash_differential.rs` | Seeded/streaming/property-style differential coverage | Non-cryptographic; no Wycheproof suite applies |
| HMAC-SHA-256 | `tests/hmac_sha256_vectors.rs`, `tests/hmac_sha256_proptest.rs`, `tests/hmac_sha2_family_vectors.rs` | `tests/hmac_wycheproof.rs` covers full-tag Wycheproof valid/invalid tags; proptests/family vectors cover mismatch behavior | Truncated-tag Wycheproof groups are out of API scope: `verify_tag` accepts only `[u8; 32]` full tags |
| HMAC-SHA-384 | `tests/hmac_sha384_proptest.rs`, `tests/hmac_sha2_family_vectors.rs` | `tests/hmac_wycheproof.rs` covers full-tag Wycheproof valid/invalid tags; proptests/family vectors cover mismatch behavior | Truncated-tag Wycheproof groups are out of API scope: `verify_tag` accepts only `[u8; 48]` full tags |
| HMAC-SHA-512 | `tests/hmac_sha512_proptest.rs`, `tests/hmac_sha2_family_vectors.rs` | `tests/hmac_wycheproof.rs` covers full-tag Wycheproof valid/invalid tags; proptests/family vectors cover mismatch behavior | Truncated-tag Wycheproof groups are out of API scope: `verify_tag` accepts only `[u8; 64]` full tags |
| HMAC-SHA3-224/256/384/512 | `tests/hmac_sha3_vectors.rs` | Streaming/one-shot/reset coverage against direct RFC 2104 oracles built over RustCrypto SHA-3 digests | No Wycheproof HMAC-SHA3 suite currently mapped |
| HKDF-SHA-256 | `tests/hkdf_sha256_vectors.rs`, `tests/hkdf_sha256_proptest.rs` | `tests/hkdf_wycheproof.rs` covers Wycheproof valid OKM vectors and oversized-output rejection | Current suite maps directly |
| HKDF-SHA-384 | `tests/hkdf_sha384_vectors.rs`, `tests/hkdf_sha384_proptest.rs` | `tests/hkdf_wycheproof.rs` covers Wycheproof valid OKM vectors and oversized-output rejection | Current suite maps directly |
| HKDF-SHA-512 | `tests/hkdf_sha512_vectors.rs` | RFC 5869 case 1, RustCrypto differential coverage, derive-vs-expand equivalence, and oversized-output rejection | Current suite maps directly |
| Poly1305 | `tests/poly1305_vectors.rs` | RFC 8439 §2.5.2 vector, streaming/one-shot equivalence, corrupted tag rejection, and fallible key-generation hook coverage | No Wycheproof standalone Poly1305 suite currently mapped |
| PBKDF2-SHA-256/SHA-512 | `tests/pbkdf2_kat_vectors.rs`, `tests/pbkdf2_differential.rs` | `tests/pbkdf2_wycheproof.rs` covers Wycheproof valid derived-key vectors plus explicit wrong-password/wrong-output rejection | Wycheproof PBKDF2 suites contain valid KATs only for the mapped SHA-2 profiles |
| Argon2d/i/id | `tests/argon2_vectors.rs`, `tests/argon2_differential.rs`, `tests/argon2_kernels.rs`, `tests/argon2_parallel.rs`, `tests/argon2_miri.rs` | `tests/phc_roundtrip.rs` covers generated records, wrong passwords, canonical-only parsing, rehash status, and zero-allocation resource rejection; fuzz corpus replay covers the bounded public verifier | No Wycheproof PHC string suite exists |
| scrypt | `tests/scrypt_vectors.rs`, `tests/scrypt_differential.rs` | `tests/phc_roundtrip.rs` covers generated records, wrong passwords, canonical-only parsing, rehash status, and zero-allocation resource rejection; fuzz corpus replay covers the bounded public verifier | Wycheproof PBKDF2 exists, but not scrypt PHC strings |
| AES-128-GCM | `tests/aes128gcm_oracle.rs` | `tests/aead_wycheproof.rs` covers Wycheproof AES-GCM 128-bit key, 96-bit nonce open failure; oracle tamper tests cover modified tag/ciphertext/AAD | AES-192 vectors are unsupported by API and skipped |
| AES-256-GCM | `tests/aes256gcm_oracle.rs` | `tests/aead_wycheproof.rs` covers Wycheproof AES-GCM 256-bit key, 96-bit nonce open failure; oracle tamper tests cover modified tag/ciphertext/AAD | Non-96-bit Wycheproof nonce cases are unsupported by API and skipped |
| AES-128-GCM-SIV | `tests/aes128gcmsiv_oracle.rs` | `tests/aead_wycheproof.rs` covers Wycheproof AES-GCM-SIV 128-bit key open failure; oracle tamper tests cover modified tag/ciphertext/AAD | Current suite maps directly |
| AES-256-GCM-SIV | `tests/aes256gcmsiv_oracle.rs` | `tests/aead_wycheproof.rs` covers Wycheproof AES-GCM-SIV 256-bit key open failure; oracle tamper tests cover modified tag/ciphertext/AAD | Current suite maps directly |
| ChaCha20-Poly1305 | `tests/chacha20poly1305.rs` | `tests/aead_wycheproof.rs` covers Wycheproof 96-bit nonce open failure; unit/integration tests cover wrong nonce/tag/AAD | Non-96-bit Wycheproof nonce cases are unsupported by API and skipped |
| XChaCha20-Poly1305 | `tests/xchacha20poly1305.rs` | `tests/aead_wycheproof.rs` covers Wycheproof 192-bit nonce open failure; unit/integration tests cover wrong nonce/tag/AAD | Current suite maps directly after nonce-size filtering |
| AEGIS-256 | `tests/aegis256_oracle.rs` | `tests/aead_wycheproof.rs` covers Wycheproof AEGIS-256 open failure; unit/integration tests cover wrong nonce/tag/AAD | Current suite maps directly |
| Ascon-AEAD128 | `tests/ascon_aead_oracle.rs` | Unit/integration tests cover wrong nonce/tag/AAD and oracle decrypt failure | Current Wycheproof `ASCON128` vectors do not match this crate's NIST Ascon-AEAD128 variant, so they are not vendored |
| ECDSA P-256/P-384 signing and verification | `tests/ecdsa_oracle.rs`; `src/auth/ecdsa.rs` unit tests; RustCrypto `p256 0.14.0` / `p384 0.13.1` oracles; `fuzz/target_impls/auth_ecdsa_verify.rs`; `fuzz/target_impls/auth_ecdsa_sign.rs` | Invalid SEC1 public keys, malformed SPKI, malformed DER signatures, zero/out-of-range scalars, tampered signatures, wrong messages, deterministic signing, blinded signing, low-S normalization, public-key derivation, and fuzz parser/differential coverage | CT evidence covers blinded signing. Public verification remains public-input work unless promoted by the CT manifest. Wycheproof ECDSA vectors are not mapped yet. |
| Ed25519 | `tests/ed25519_rfc8032_vectors.rs`, `tests/ed25519_oracle.rs` | `tests/ed25519_wycheproof.rs` covers Wycheproof valid/invalid signatures and invalid public/signature encodings; unit tests cover small-order and non-canonical signatures | Current suite maps directly |
| X25519 | `tests/x25519_vectors.rs`, `tests/x25519_oracle.rs` | `tests/x25519_wycheproof.rs` covers Wycheproof valid/acceptable XDH vectors and rejects all-zero shared secrets; RFC low-order and non-canonical public cases remain in `tests/x25519_vectors.rs` | ASN/JWK/PEM suites do not apply to byte-array API |
| ML-KEM-512/768/1024 | `tests/mlkem_acvp.rs` covers NIST ACVP FIPS 203 keyGen, encapsulation, decapsulation, decapsulationKeyCheck, and encapsulationKeyCheck vectors for all parameter sets; `tests/mlkem_properties.rs` differentials arbitrary seeds against the `fips203` crate; `tests/mlkem_types.rs` checks FIPS 203 sizes, randomness, security categories, byte wrappers, secret redaction, and constant-time equality | `tests/mlkem_ops.rs` covers non-canonical public-key rejection before randomness, prepared-key parity, prepared-key invalid material, wrong-length parsers, decapsulation-key hash mismatch, and modified-ciphertext implicit rejection; `fuzz/target_impls/auth_mlkem512.rs`, `auth_mlkem768.rs`, and `auth_mlkem1024.rs` cover round trips, parser inputs, and modified ciphertexts | No vendored Wycheproof ML-KEM suite is currently mapped; official ACVP vectors plus all-profile FIPS 203 differential/property coverage are the primary oracle set |
| RSA signatures | `tests/rsa_wycheproof.rs`, `tests/rsa_nist_cavp.rs`, `tests/rsa_public_key.rs` | Wycheproof invalid PKCS#1 v1.5/PSS signatures; `tests/rsa_profile_confusion.rs` rejects PKCS#1/PSS and protocol-scheme confusion | RSA-PSS parameter Wycheproof suites are partly not mapped because the public profile supports SHA-2 fixed profiles |
| RSA OAEP / RSAES-PKCS1-v1_5 | `tests/rsa_wycheproof.rs`, `tests/rsa_public_key.rs` | Wycheproof invalid ciphertexts; scratch decrypt failure clears plaintext; unsupported MGF1-SHA1 vectors reject | Current SHA-2 OAEP suites map directly |
| RSA key parsing / X.509 / TLS / COSE adapters | `tests/rsa_public_key.rs`, `tests/rsa_allocations.rs`, `tests/rsa_leakage.rs` | DER non-canonical forms, unsupported algorithms, policy boundaries, profile confusion, and leakage gate | Keep these tests explicit because the attack surface is protocol/profile confusion, not only raw RSA math |
| Hex/serde public formats | `src/hex.rs`, `tests/serde_roundtrip.rs` | Invalid hex length/character tests; serde byte roundtrips | Negative coverage is format-boundary only |
| Dispatch/fallback surface | `tests/aead_kernel_equivalence.rs`, `tests/aead_foundations.rs`, `tests/portable_fallback.rs`, `tests/vectored_dispatch.rs` | Backend equivalence and fallback dispatch checks | Not a primitive vector suite, but required for SIMD correctness |
