# Test evidence

This map shows the independent evidence for each primitive family,
and the important limits outside that evidence.
Test filenames are the stable entry points.
`testdata/` and the test readers own the individual corpus files.
An OpenSSL differential skips where `openssl` is missing or lacks the algorithm.
The native Linux CI rows put the pinned OpenSSL 4.0.3 peer on `PATH` and set `RSCRYPTO_REQUIRE_OPENSSL`,
so there each such skip fails the test.

## Coverage map

| Family | Independent evidence | Limit or gap |
| --- | --- | --- |
| CRC-16/24/32/64 | Property tests, and the `crc`, `crc-fast`, `crc32fast`, `crc32c`, and `crc64fast` oracles where they apply. Wasmtime scalar and SIMD128 builds check CRC-32, CRC-32C, and CRC-64/XZ against the bit-serial catalogue definition, which the catalogue check values anchor. They check every length to 100 bytes at offsets 0 to 15, 4,096 bytes, and chunked writes. | No Wycheproof suites exist for checksums. |
| Adler-32 | CPython 3.14.8 `zlib.adler32` (zlib 1.2.12) values for short input and for all-zero and all-0xff input at the 16-byte step and at one, two, and three times the 5,552-byte reduction distance up to 1 MiB, one-shot and streamed, and all-0xff input from both sums at 65520; `simd-adler32` differentials for every two-part and uniform split, resume point, and combination up to 300 bytes; combinations whose second length is 0, a multiple of 65521, or 2^32 + 7 against `zlib.adler32` of the concatenation; reset. CPython zlib fixtures check the big-endian trailer and the preset dictionary identifier, which CRC-32 does not reproduce. Wasmtime scalar and SIMD128 builds, which run the portable and SIMD128 kernels, check a 139,269-byte all-0xff vector from the initial value and from both sums at 65520, one-shot and streamed. They also check `zlib.adler32` values for every length to 640 bytes at every offset from 0 to 63 and for chunked writes of 20,000 bytes. The AArch64 NEON and dot-product kernels and the riscv64 RVV kernel match the portable kernel for every length to 640 bytes at every alignment from 0 to 63, around 5,552 bytes, 8 KiB, 16 KiB, and 128 KiB and their multiples, from three starting values, and in chunked writes across them. The x86-64 SSSE3, AVX2, and AVX-512 VNNI kernels and the POWER and IBM Z kernels match the portable kernel in their own kernel tests, and each forced tier runs through the public API. Provenance: [fixtures](../testdata/adler32/README.md). | Not cryptographic. No Wycheproof suite applies. AArch64 runtime evidence comes from Apple Silicon and the native Linux AArch64 CI rows. RVV runtime evidence comes only from the native RISC-V CI row. A host without V skips the RVV kernel tests, and the selection test fails if a host reports V and does not run RVV. |
| SHA-2, SHA-3, SHAKE, cSHAKE, KMAC | NIST vectors, vendored `.blb` corpora, and RustCrypto differentials. The `hash_cshake128`, `hash_cshake256`, `auth_kmac128`, and `auth_kmac256` fuzz targets repeat the RustCrypto `cshake` and `tiny-keccak` differentials over arbitrary inputs and splits. | No Wycheproof suite maps to KMAC128. |
| BLAKE2, BLAKE3, Ascon-Hash/XOF | Upstream or NIST corpora, and independent differentials. | Hash functions have no invalid-ciphertext or invalid-signature class. |
| Bao combined streams | Pinned Bao 0.13.1 vectors and encoder/decoder oracles, exhaustive corruption/truncation tests, maximum-depth proofs, and decoder fuzzing. | Covers sequential unkeyed combined decoding; outboard encodings, slices, and seeking are not implemented. |
| XXH3 and RapidHash | Differentials against the upstream crates, streaming tests, and fuzzing. Wasmtime scalar and SIMD128 builds check xxhsum 0.8.4 XXH3-64 values for one long input at seed zero and a nonzero seed. | Not cryptographic. No Wycheproof suite applies. |
| XXH32 and XXH64 | xxhsum 0.8.4 values for the xxHash sanity buffer at seed zero and a nonzero seed, at every byte offset from 0 to 7, one-shot and streamed; `xxhash-rust` differentials for every two-part and uniform split up to 130 bytes; XXH32 past 4 GiB, where the length counter passes 2^32. LZ4 1.11.0 and Zstandard 1.5.7 frame fixtures check the LZ4 descriptor, block, and content checksums and the Zstandard content checksum, which XXH3 does not reproduce. Wasmtime scalar and SIMD128 builds, on `wasm32-unknown-unknown` and `wasm32-wasip1`, check a 4,109-byte xxhsum vector for each algorithm at both seeds, one-shot and streamed. The `fast_xxh32` and `fast_xxh64` fuzz targets compare one-shot and arbitrary incremental partitions, including after a reset, with `xxhash-rust`. Provenance: [fixtures](../testdata/xxhash/README.md). | Not cryptographic. No Wycheproof suite applies. |
| SipHash-1-3 and SipHash-2-4 | The reference `vectors.h` SipHash-2-4 values for every length to 63 bytes; Rust std SipHash-2-4 and `siphasher` 1.0.4 differentials for three keys, every length to 64 bytes, and three-part splits; integer writes at every pending offset; `finish` without consuming the state; fail-closed `SipRandomState` construction and redacted `Debug`. The `auth_siphash` fuzz target repeats the `siphasher` differential over arbitrary write schedules. Wasmtime scalar and SIMD128 builds check one reference value and the 4-byte `write_usize` of wasm32. | No Wycheproof suite exists for SipHash. |
| `ChaCha20Rng` | The RFC 8439 Appendix A.1 keystream as the first output. The documented stream against RustCrypto `chacha20` 0.10.2 across key refills, request splits, and integer reads, as a property over random seeds and in the `auth_chacha20_rng` fuzz target. Child generators against RustCrypto `chacha20` for fixed paths, independent of parent draws and of the constructor. `rand_core` 0.10 trait outputs and `fork`, refusal of `seed_from_u64`, fail-closed caller entropy, and the portable diagnostic fill. Wasmtime scalar and SIMD128 builds check the RFC 8439 stream and one child. | No standard test suite exists for this construction. The generator never reseeds itself. |
| HMAC, HKDF, PBKDF2, Poly1305 | Official vectors, Wycheproof where the public profile maps (HMAC-SHA-2, the four HMAC-SHA3 suites, HKDF-SHA-256/384/512, and PBKDF2), properties, and differentials. Wasmtime scalar and SIMD128 builds check one RFC 4231, one RFC 5869, and one PBKDF2-HMAC-SHA256 value. The `auth_hmac_sha3`, `auth_hkdf_sha512`, and `auth_poly1305` fuzz targets compare arbitrary inputs with RustCrypto `hmac` over `sha3`, `hkdf` over `sha2`, and `poly1305`. | Only the tag widths that the API exposes map. |
| Argon2 and scrypt | Published vectors, RustCrypto differentials, kernel and parallel tests, and Miri. | No Wycheproof suite exists for PHC strings. |
| AEADs | Wycheproof where the variant and nonce width map, official vectors, RustCrypto oracles, corruption tests, and backend equivalence. `tests/aes_gcm_aesni_tier.rs` masks the capabilities that detection removes on an Intel hybrid part. It requires the AES-NI/PCLMUL tier of AES-GCM and AES-GCM-SIV, and runs the Wycheproof suites and RustCrypto oracles on that tier. Unit tests run FIPS 197 C.1 and C.3 on the forced RV64 Zkne functions and check its round-key packing against the Sail rule of the scalar cryptography specification. `tests/riscv_scalar_crypto.rs` removes Zvkned from the detected capabilities, so AES-GCM and AEGIS-256 select the Zkne backend, and runs the Wycheproof AES-GCM and AEGIS-256 suites and the five draft-irtf-cfrg-aegis-aead A.3 vectors. Wasmtime scalar and SIMD128 builds run the Wycheproof ChaCha20-Poly1305, XChaCha20-Poly1305, and AES-GCM suites. | The typed API rejects unsupported key and nonce sizes, so those cases are filtered out. Wycheproof's older Ascon variant does not match NIST Ascon-AEAD128. No CI host has Zkne, so the Zkne tests run only under QEMU and skip elsewhere. |
| QUIC-LB connection-ID cipher | The draft-ietf-quic-load-balancers-22 Appendix B.2 vectors in both directions. Independent connection IDs for all 120 server-ID and nonce length pairs: 351 from google/quiche for 117 pairs, and 9 from hopf-quic 0.7.0 for the 3 pairs with a nonce over 16 octets, which quiche refuses. The single-pass form against RustCrypto `aes` in both directions. FIPS 197 C.1 and RustCrypto `aes` for the AES-128 inverse cipher of every backend that the host runs, and the bitsliced and portable inverse S-boxes against FIPS 197 Figure 14. A server-ID round trip and the buffer contract for every length pair, and construction for exactly the draft lengths. The `aead_quic_lb` fuzz target checks construction, the buffer contract, the round trip, and the single-pass form against an independent AES-128. Provenance: [vectors](../testdata/quic_lb/README.md). | No Wycheproof suite exists. The draft is not final. |
| ECDSA, Ed25519, X25519 | RFC or official vectors, Wycheproof, RustCrypto and dalek oracles, properties, and fuzzing. The 12 Ed25519 speccheck cases run through the public API and each verify backend, with the verdicts of ed25519-dalek `verify_strict` ([provenance](../testdata/ed25519/speccheck/README.md)). Wasmtime scalar and SIMD128 builds check RFC 8032 test 2 and the RFC 6979 A.2.5 P-256/SHA-256 signature. Key encodings: see [ECDSA and Ed25519 key encodings](#ecdsa-and-ed25519-key-encodings). | JWK and variable-length profiles are excluded where the public API accepts only fixed arrays. |
| P-256 ECDH | All 25 NIST CAVP P-256 ECC CDH component records, all 355 pinned Wycheproof `ecpoint` cases, RustCrypto differentials, a ring cross-agreement, Miri, and fuzzing. | The public API accepts only canonical uncompressed SEC1 points. Wycheproof supplies the full-width leading-zero and all-zero x-coordinate cases. The NIST slice has no full leading-zero byte. |
| P-384 ECDH | All 25 NIST CAVP P-384 ECC CDH component records, all 790 pinned Wycheproof `ecpoint` cases, RustCrypto differentials and properties, a ring cross-agreement, and fuzzing. | The public API accepts only canonical uncompressed SEC1 points. |
| ML-KEM-512/768/1024 | See [ML-KEM evidence](#ml-kem-evidence). | CCTV's unlucky-sampling and accumulated vectors target the FIPS 203 draft and are not imported. Wycheproof's high-rejection matrix seeds cover sampling under the final standard. |
| X-Wing (ML-KEM-768 + X25519) | The 3 draft-connolly-cfrg-xwing-kem-11 Appendix C vectors and the 10 MLKEM768-X25519 vectors of draft-irtf-cfrg-concrete-hybrid-kems-04 Appendix B.2, through the `Kem` functions, the prepared keys, and `prepare_in`; the concrete vectors also check the expanded ML-KEM seed and X25519 key. All 14 small-order X25519 encodings of X-Wing pull request 39, in the ciphertext and in the encapsulation key, with the key unchanged after each rejection. Differentials against RustCrypto `x-wing` 0.1.1 (its rejecting `DecapsulationKeyRejectNonContrib`) for key generation, deterministic encapsulation, and decapsulation of valid, single-bit-changed, and arbitrary ciphertexts. Rejection of an unreduced ML-KEM coefficient before randomness, wrong lengths, and random-source failure. The `auth_xwing` fuzz target repeats the differential, including key validation. Wasmtime scalar and SIMD128 builds run the draft-11 vectors. | The draft is not final: -11 does not specify the small-order rejection that rscrypto follows. Neither draft has a Wycheproof suite. Timing and resource qualification are open. Provenance: [vectors](../testdata/xwing/README.md). |
| HPKE Base mode, both suites | RFC 9180 Appendix A.1.1 (257 encryptions, 3 exports) and the draft-ietf-hpke-hpke-06 Appendix D.2 to D.4 edge cases for DHKEM(X25519) + AES-128-GCM; draft-ietf-hpke-pq-05 A.5 for X-Wing + ChaCha20Poly1305, and the KEM values of A.12. Each vector runs `DeriveKeyPair`, the sender with `ikmE`, both roles in sequence order, and every export. All 73 pinned Wycheproof cases of the X25519 suite: 19 encapsulation, 47 decryption, and 7 sequence cases, with the sender reproducing each honest ciphertext. The 14 small-order X25519 encodings in both directions of both suites. Differentials against `hpke` 0.15.1 in both directions for both suites. A failed open keeps the sequence number; length errors, an oversized export, a failed random source, and the last sequence number change no state. The `auth_hpke` fuzz target repeats the differential. Wasmtime scalar and SIMD128 builds run A.1.1 and A.5. | No Wycheproof or CCTV suite covers X-Wing HPKE, and Wycheproof does not cover `DeriveKeyPair` or `Export`. Erratum 7121 is not applied (see the `hpke` module documentation). Timing and resource qualification are open. Provenance: [vectors](../testdata/hpke/README.md). |
| ML-DSA-44/65/87 | All 615 pinned NIST ACVP cases, all 1,138 pinned Wycheproof signing and verification cases, RustCrypto 0.1.1 differentials, and rejection tests for context, encoding, entropy, and prehash domain. RFC 9881 Appendix C public and private keys, including the inconsistent keys; SPKI and PKCS #8 exchange with RustCrypto 0.1.1, aws-lc-rs 1.18.2, and the OpenSSL CLI. | Successful sigGen vectors cover the positive prehash cases that the sigVer corpus lacks. Timing and resource qualification are still open. The OpenSSL exchange runs only where `openssl` supports ML-DSA. Provenance: [ACVP](../testdata/mldsa/acvp/README.md), [Wycheproof](../testdata/mldsa/wycheproof/README.md), [RFC 9881](../testdata/mldsa/rfc9881/README.md). |
| Composite ML-DSA-65 with Ed25519 or ECDSA P-256 | Both draft-ietf-lamps-pq-composite-sigs-19 Appendix E cases: the empty-context and context signatures, the self-signed certificate signatures, and the raw, SubjectPublicKeyInfo, and PKCS #8 keys, imported and re-exported byte for byte. Bidirectional differentials over seven message lengths and six keys each: `id-MLDSA65-Ed25519-SHA512` against `darkbio-crypto` 0.20.1 (`xdsa`), including key expansion and SPKI; `id-MLDSA65-ECDSA-P256-SHA512` against `kryptering` 0.7.1, which derives and checks the public key. Hostile SPKI, PKCS #8, `ECPrivateKey`, point, DER signature, length, and hint encodings; context, entropy-failure, prehash, preparation, and stripping tests. The `auth_composite_mldsa` fuzz target checks signing, single-bit forgery rejection, and canonical encodings for both composites. Wasmtime scalar and SIMD128 builds verify both draft cases with and without the context. | The draft's generator signs with fresh randomness, so its signatures are verification vectors only. Neither comparator takes an application context; the draft's context vectors cover it. `kryptering` uses JOSE encodings for the P-256 component, which the test converts. Timing and resource qualification are open. Provenance: [draft-19](../testdata/composite_mldsa/draft19/README.md). |
| SLH-DSA, all 12 sets | All 1,248 pinned NIST ACVP cases through the internal-interface test hooks, including pre-hash cases for all 12 FIPS 205 hash functions; the pure and RFC 9909 cases also through the public profiles. Differentials against RustCrypto `slh-dsa` 0.2.0-rc.5 (key generation, deterministic and hedged signing) and `fips205` 0.5.0 (pure and HashSLH-DSA signing and verification). The precomputed SHA-2 states after the PK.seed block match RustCrypto `sha2` at every data length through two blocks, split around each block edge, including lengths that no parameter set reaches. RFC 9909 Appendix C keys and certificate signature; SPKI and PKCS #8 for all 24 profiles against DER built from the RFC's OIDs, a rejection test per rule, framing sweeps, and exchange with RustCrypto and the OpenSSL CLI, including signatures. Context, entropy-failure, domain-separation, and import-check tests. The `auth_slhdsa` fuzz target checks verification and canonical key encodings for all 24 profiles. Wasmtime scalar and SIMD128 builds generate each set's ACVP key, re-sign one deterministic case, and verify pure and pre-hash signatures. | The signing fixture keeps the SHA-256 of each expected signature instead of the signature. OpenSSL and RustCrypto have no HashSLH-DSA codecs, so those encodings rest on the RFC-derived DER. No Wycheproof suite exists. Timing results wait for the CT campaign, and no native residue run covers Windows. The OpenSSL exchange runs only where `openssl` supports SLH-DSA. Provenance: [ACVP](../testdata/slhdsa/acvp/README.md), [RFC 9909](../testdata/slhdsa/rfc9909/README.md). |
| RSA signatures, encryption, and parsing | NIST CAVP, Wycheproof, RustCrypto and system OpenSSL/LibreSSL oracles, and profile-confusion, allocation, and leakage tests. Wasmtime scalar and SIMD128 builds check a Wycheproof RSA-2048 PKCS #1 v1.5 SHA-256 signature. | The public API exposes fixed SHA-2 profiles, not every Wycheproof parameter combination. The system-library oracles run only where the test host has them. |
| Dispatch and fallback | Differential tests, portable against accelerated, across lengths, tails, and vectored input. The AArch64 Linux CI row runs on Graviton4, and it fails unless detection reports SVE2, SVE2-AES, SVE2-PMULL, and SVE2-SHA3. So the SVE2 kernel tests run in every CI run. The RISC-V QEMU workflow runs the riscv64 suites under QEMU user mode in three CPU configurations: every extension at VLEN 128, VLEN 1024 without Zbc, and the scalar extensions without V. It fails when detection reports no extension. | Cross-compilation alone is not runtime evidence. QEMU proves function, not timing or speed. |
| Process fork detection | `tests/process_origin.rs` forks once in each test, with the kernel's fork as the oracle. It requires that a fork child and a grandchild each see a foreign `ProcessOrigin`, that a `NonceCounter` and an HPKE sender refuse in the child, and that a `ChaCha20Rng` child keeps its root's origin. A child that allocates fails the test. | It runs only where the target can fork. A restored virtual-machine snapshot is not detected. |

### RSA implicit rejection

`tests/rsa_implicit_rejection.rs` checks all four legacy decryption entry points
against 36 Appendix B vectors from draft-irtf-cfrg-rsa-guidance-10 and 96 seeded
OpenSSL 4.0.3 decryptions, for 2048-, 3072-, and 4096-bit keys. It also checks
59 Wycheproof invalid-padding results against OpenSSL's synthetic messages and
requires public ciphertext-format errors to remain errors. Unit tests isolate
each padding rule, candidate masking and last-fit selection, empty and maximum
messages, fixed-window output, and scratch equivalence.

An additional unit test executes production derivation and depadding for all
48 Appendix B cases, including the 2049-bit key, using the stored exponent and
OpenSSL raw RSA outputs as inputs and the draft messages as expectations.
Appendix B.2's 2049-bit key fails the existing half-width-factor import policy.
Its 12 decryption vectors and 32 seeded ciphertexts remain in the fixture and
are cross-checked with OpenSSL during generation; they do not qualify rscrypto's
public decryption API. See [provenance and reproduction](../testdata/rsa/implicit_rejection/README.md).

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

### ECDSA and Ed25519 key encodings

`tests/ecdsa_pkix.rs` exchanges P-256 and P-384 PKCS #8, SEC1, and SPKI keys with RustCrypto `p256`/`p384`
0.14.0, ring 0.17.14, and the OpenSSL CLI, in both directions.
The PKCS #8 export equals the RustCrypto and OpenSSL encodings byte for byte.
DER signatures equal RustCrypto's encoding, round-trip through `from_der`, and verify under ring and
`openssl pkeyutl`.
`tests/ed25519_pkix.rs` round-trips the RFC 8410 section 10 public and private keys and exchanges keys with
ring 0.17.14 and the OpenSSL CLI.
Both files have a rejection test for each container, identifier, scalar, and public-key rule.
The `auth_ecdsa_sign` and `auth_ed25519` fuzz targets check export round trips, canonical acceptance,
and parser robustness.
The OpenSSL exchanges run only where `openssl` is installed.

## Run the evidence

```bash
just test --all
just test-fuzz --all
```

`just --list` shows the specialized Miri, target, constant-time, and leakage recipes.

The [Fuzz workflow](../.github/workflows/fuzz.yml) runs only when explicitly dispatched.
Manual runs select x86-64, ARM64, or both.
Each run also includes focused x86-64 Miri checks.
Each fuzz job first replays the committed corpus and the restored corpus checkpoint
under AddressSanitizer, then runs a bounded live campaign.
Release requires a passing manual run on the exact source commit, covering both architectures,
every target, and at least 120 seconds per target in a smoke campaign.
It checks that run's `Release qualification` job without starting Fuzz.
Runner profiles can change independently of target selection, concurrency, and sampling budgets.
The same elapsed time does not mean the same fuzzing throughput.

A passing vector proves the behavior for that vector only.
Stronger assurance comes from a combination: published vectors, a separate implementation,
properties, hostile inputs, fuzzing, portable-versus-accelerated equivalence, and target execution.
