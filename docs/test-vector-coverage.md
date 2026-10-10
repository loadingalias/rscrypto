# Test evidence

This map shows the independent evidence for each primitive family,
and the important limits outside that evidence.
Test filenames are the stable entry points.
`testdata/` and the test readers own the individual corpus files.

## Coverage map

| Family | Independent evidence | Limit or gap |
| --- | --- | --- |
| CRC-16/24/32/64 | Property tests, and the `crc`, `crc-fast`, `crc32fast`, `crc32c`, and `crc64fast` oracles where they apply. | No Wycheproof suites exist for checksums. |
| SHA-2, SHA-3, SHAKE, cSHAKE, KMAC | NIST vectors, vendored `.blb` corpora, and RustCrypto differentials. | No Wycheproof suite maps to KMAC128. |
| BLAKE2, BLAKE3, Ascon-Hash/XOF | Upstream or NIST corpora, and independent differentials. | Hash functions have no invalid-ciphertext or invalid-signature class. |
| Bao combined streams | Pinned Bao 0.13.1 vectors and encoder/decoder oracles, exhaustive corruption/truncation tests, maximum-depth proofs, and decoder fuzzing. | Covers sequential unkeyed combined decoding; outboard encodings, slices, and seeking are not implemented. |
| XXH3 and RapidHash | Differentials against the upstream crates, streaming tests, and fuzzing. | Not cryptographic. No Wycheproof suite applies. |
| HMAC, HKDF, PBKDF2, Poly1305 | Official vectors, Wycheproof where the public profile maps, properties, and differentials. | No Wycheproof suite maps to HMAC-SHA3. Only the tag widths that the API exposes map. |
| Argon2 and scrypt | Published vectors, RustCrypto differentials, kernel and parallel tests, and Miri. | No Wycheproof suite exists for PHC strings. |
| AEADs | Wycheproof where the variant and nonce width map, official vectors, RustCrypto oracles, corruption tests, and backend equivalence. | The typed API rejects unsupported key and nonce sizes, so those cases are filtered out. Wycheproof's older Ascon variant does not match NIST Ascon-AEAD128. |
| ECDSA, Ed25519, X25519 | RFC or official vectors, Wycheproof, RustCrypto and dalek oracles, properties, and fuzzing. | ASN.1, JWK, and variable-length profiles are excluded where the public API accepts only fixed arrays. |
| P-256 ECDH | All 25 NIST CAVP P-256 ECC CDH component records, all 355 pinned Wycheproof `ecpoint` cases, RustCrypto differentials, a ring cross-agreement, Miri, and fuzzing. | The public API accepts only canonical uncompressed SEC1 points. Wycheproof supplies the full-width leading-zero and all-zero x-coordinate cases. The NIST slice has no full leading-zero byte. |
| P-384 ECDH | All 25 NIST CAVP P-384 ECC CDH component records, all 790 pinned Wycheproof `ecpoint` cases, RustCrypto differentials and properties, a ring cross-agreement, and fuzzing. | The public API accepts only canonical uncompressed SEC1 points. |
| ML-KEM-512/768/1024 | See [ML-KEM evidence](#ml-kem-evidence). | CCTV's unlucky-sampling and accumulated vectors target the FIPS 203 draft and are not imported. Wycheproof's high-rejection matrix seeds cover sampling under the final standard. |
| ML-DSA-44/65/87 | All 615 pinned NIST ACVP cases, all 1,138 pinned Wycheproof signing and verification cases, RustCrypto 0.1.1 differentials, and rejection tests for context, encoding, entropy, and prehash domain. RFC 9881 Appendix C public and private keys, including the inconsistent keys; SPKI and PKCS #8 exchange with RustCrypto 0.1.1, aws-lc-rs 1.18.1, and the OpenSSL CLI. | Successful sigGen vectors cover the positive prehash cases that the sigVer corpus lacks. Timing and resource qualification are still open. The OpenSSL exchange runs only where `openssl` supports ML-DSA. Provenance: [ACVP](../testdata/mldsa/acvp/README.md), [Wycheproof](../testdata/mldsa/wycheproof/README.md), [RFC 9881](../testdata/mldsa/rfc9881/README.md). |
| SLH-DSA, all 12 sets | All 1,248 pinned NIST ACVP cases through the internal-interface test hooks, including pre-hash cases for all 12 FIPS 205 hash functions; the pure and RFC 9909 cases also through the public profiles. Differentials against RustCrypto `slh-dsa` 0.2.0-rc.5 (key generation, deterministic and hedged signing) and `fips205` 0.4.1 (pure and HashSLH-DSA signing and verification). RFC 9909 Appendix C keys and certificate signature; SPKI and PKCS #8 for all 24 profiles against DER built from the RFC's OIDs, a rejection test per rule, framing sweeps, and exchange with RustCrypto and the OpenSSL CLI, including signatures. Context, entropy-failure, domain-separation, and import-check tests. | The signing fixture keeps the SHA-256 of each expected signature instead of the signature. OpenSSL and RustCrypto have no HashSLH-DSA codecs, so those encodings rest on the RFC-derived DER. No Wycheproof suite exists. Timing, cleanup, and resource qualification are open. The OpenSSL exchange runs only where `openssl` supports SLH-DSA. Provenance: [ACVP](../testdata/slhdsa/acvp/README.md), [RFC 9909](../testdata/slhdsa/rfc9909/README.md). |
| RSA signatures, encryption, and parsing | NIST CAVP, Wycheproof, RustCrypto and system OpenSSL/LibreSSL oracles, and profile-confusion, allocation, and leakage tests. | The public API exposes fixed SHA-2 profiles, not every Wycheproof parameter combination. The system-library oracles run only where the test host has them. |
| Dispatch and fallback | Differential tests, portable against accelerated, across lengths, tails, and vectored input. | Cross-compilation alone is not runtime evidence. |

### ML-KEM evidence

- NIST ACVP key-generation, encapsulation, decapsulation, and key-check vectors.
- All 1,725 pinned Wycheproof cases ([provenance](../testdata/mlkem/wycheproof/README.md)).
- All 2,595 CCTV modulus-check keys, and the CCTV `strcmp` rejection vectors
  ([provenance](../testdata/mlkem/cctv/README.md)).
- The accumulated ML-KEM-768 hashes from the Go standard library:
  10,000 iterations run by default; the 1,000,000-iteration test is ignored by default.
- `fips203` differentials.
- RFC 9935 Appendix C public and private keys, including the four bad keys
  ([provenance](../testdata/mlkem/rfc9935/README.md)).
  SPKI and PKCS #8 exchange with RustCrypto ML-KEM 0.3.2 (seed form) and the OpenSSL CLI,
  which runs only where `openssl` supports ML-KEM.

### Other cases

The WebSocket accept digest has the RFC 6455 example, private SHA-1 known-answer tests,
RustCrypto differential tests, and fuzzing.
It exists only for compatibility.
It makes no collision-resistance or authentication claim.

ECDSA P-256/SHA-384 and P-384/SHA-256 verification use the RFC 6979 Appendix A.2.5 and A.2.6 vectors
and RustCrypto differentials in `tests/ecdsa_oracle.rs`.
They cover digest truncation, hash selection, message boundaries, and rejection of changed messages,
keys, and signatures.
The vendored ECDSA Wycheproof suites cover P-256/SHA-256 and P-384/SHA-384.

## Run the evidence

```bash
just test --all
just test-fuzz --all
```

`just --list` shows the specialized Miri, target, constant-time, and leakage recipes.

The [Fuzz workflow](../.github/workflows/fuzz.yml) runs x86-64 fuzzing and focused Miri checks for pull requests.
Release qualification selects both x86-64 and ARM64 fuzzing.
Each fuzz job first replays the committed corpus under AddressSanitizer,
then runs a bounded live campaign.
Runner profiles can change independently of target selection, concurrency, and sampling budgets.
The same elapsed time does not mean the same fuzzing throughput.

A passing vector proves the behavior for that vector only.
Stronger assurance comes from a combination: published vectors, a separate implementation,
properties, hostile inputs, fuzzing, portable-versus-accelerated equivalence, and target execution.
