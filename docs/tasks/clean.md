# Code cleanup tasks

Start with T001–T006.
Together they should remove about 1,300–1,600 lines while preserving the public APIs and algorithms.
The largest exact duplicate accounts for 36,288 assembly table lines.
The RSA public SPKI encoder constructs 14 heap buffers to return one.

The audit covered 2,160 tracked files at `d613fe00a52c`.
Source comparisons establish exact counts.
Ranges estimate net line reductions.
Some later tasks overlap.
Runtime, stack use, and machine-code equivalence still need separate evidence.

Preserve feature independence, portable fallbacks, public behavior, and secret cleanup.
For kernel or dispatch changes, compare backend selection and outputs across lengths, alignments,
tails, and state transitions.
Request approval before benchmarks, profiles, or heavy qualification runs.

## First six changes

- [ ] **T001 Share SHA-512 kernel selection** (about 348 lines).
  Four files repeat kernel IDs, capability mappings, diagnostic names,
  and selection order for SHA-384 and SHA-512/256.
  Follow the SHA-224 re-export pattern.
  Retain each digest's initial state, output width, and backend coverage.
  [SHA-384 kernels](../../src/hashes/crypto/sha384/kernels.rs#L9) · [SHA-512/256 policy](../../src/hashes/crypto/sha512_256/dispatch_policy.rs#L11) · [SHA-224 precedent](../../src/hashes/crypto/sha224/kernels.rs#L3)

- [ ] **T002 Declare fuzz corpus replay tests once** (500–680 lines).
  All 76 bodies match after substitution of the target name.
  Generate one test per entry in a single list.
  Preserve each test name, target entry point, and empty-corpus assertion.
  Compare the names with the existing ASan inventory.
  [Replay tests](../../fuzz/tests/corpus_replay.rs#L238) · [Inventory check](../../scripts/test/test-fuzz-asan.sh#L72)

- [ ] **T003 Share the allocation counter across five test binaries** (200–300 lines).
  Move the thread-local counter and measurement guard into test support for P-256, P-384, RSA,
  fast hashes, and WebSocket SHA-1.
  Preserve allocation, zeroed-allocation, and reallocation counts.
  Keep measurements isolated across threads and restore the guard on unwind.
  [P-256](../../tests/p256_ecdh_allocations.rs#L15) · [P-384](../../tests/p384_ecdh_allocations.rs#L15) · [RSA](../../tests/rsa_allocations.rs#L16) · [Fast hashes](../../tests/fast_hash_allocations.rs#L28) · [WebSocket SHA-1](../../tests/websocket_sha1_allocations.rs#L15)

- [ ] **T004 Share AES-GCM nonce issuance** (120–145 lines).
  The two `NonceCounter` implementations repeat construction, fork checks, exhaustion, nonce encoding,
  and sealing order.
  Share these operations privately.
  Keep both typed public APIs.
  A failed seal must still consume its nonce.
  [AES-256 counter](../../src/aead/nonce_counter.rs#L110) · [AES-128 counter](../../src/aead/nonce_counter.rs#L261)

- [ ] **T005 Move the target-catalog check out of CPU detection** (about 70 lines).
  A 74-line module scans embedded JSON for two prefixes.
  Only a debug assertion calls it in production.
  Check both `x86_64-` and `aarch64-` in catalog validation.
  Remove the runtime scanner and its caller.
  [Scanner](../../src/platform/target_matrix.rs#L32) · [Caller](../../src/platform/detect.rs#L127) · [Catalog validation](../../scripts/check/check.sh#L46)

- [ ] **T006 Combine Linux and macOS ML-KEM dispatch** (about 44 lines).
  Both AArch64 blocks call the same three Rust entry points for polynomial-vector widths 2, 3,
  and 4.
  Use one OS condition.
  Retain the platform ABI bindings, pointer proofs,
  and exclusions for Miri and portable-only builds.
  [Linux block](../../src/auth/mlkem/operations.rs#L5462) · [macOS block](../../src/auth/mlkem/operations.rs#L5506) · [AArch64 bindings](../../src/auth/mlkem/operations/aarch64.rs#L288)

## Largest source reductions

- [ ] **T007 Store each Ed25519 and X25519 assembly table once** (36,288 duplicate data lines).
  Each curve has a distinct 6,048-line table.
  Each table occurs four times.
  All four copies match byte for byte, including line endings.

  Insert each shared fragment at its original location.
  Reconstruct each complete assembly input and verify its existing provenance digest.
  Compare emitted sections, alignment, symbols, and relocations for each variant.
  [Ed25519 table](../../src/auth/ed25519/asm/rscrypto_ed25519_scalarmulbase_aarch64_unknown_linux.s#L1405) · [First x86 X25519 copy](../../src/auth/x25519/asm/rscrypto_x25519_x86_64_sysv.s#L3278) · [Second x86 copy](../../src/auth/x25519/asm/rscrypto_x25519_x86_64_sysv.s#L12598) · [Pinned inputs](../../src/auth/asm/rscrypto_signature_assembly_provenance.tsv#L49) ·
  [Provenance checker](../../scripts/asm/provenance.py#L34)

- [ ] **T008 Share identical Darwin and Linux assembly bodies** (27,192 candidate instruction lines).
  The audit found matching runs of at least eight instruction lines across 22 file pairs.
  This count excludes labels, directives, and T007's data.
  The x86 BLAKE3 AVX2 and AVX-512 pairs contain 6,345 matching instruction lines.

  Start with those BLAKE3 pairs.
  Extend the comparison to P-256 scalar multiplication and ChaCha20-Poly1305.
  Keep target symbol spelling, sections, unwind directives, cleanup, and licenses.
  Compare each target's output and preserve its source provenance.
  [BLAKE3 bindings](../../src/hashes/crypto/blake3/x86_64/asm.rs#L14) · [Darwin P-256](../../src/auth/asm/rscrypto_p256_scalarmul_alt_aarch64_apple_darwin.s#L1879) · [Linux P-256](../../src/auth/asm/rscrypto_p256_scalarmul_alt_aarch64_unknown_linux.s#L1926) · [ChaCha20-Poly1305](../../src/aead/chacha20poly1305/asm/rscrypto_chacha20_poly1305_aarch64_linux.s#L1833)

- [ ] **T009 Derive the Ed25519 SIMD tables at compile time** (20,295 existing source lines).
  Derive the AVX2 and IFMA radix-16 representations from the portable table.
  Keep conversion outside signing.
  Existing tests compare every entry with portable field arithmetic
  and the exact vector representation.
  Compare emitted bytes, including partially reduced limbs.
  Check compile time and binary size under the affected features before adopting the change.
  [AVX2 table](../../src/auth/ed25519/basepoint_radix16_avx2.rs) · [IFMA table](../../src/auth/ed25519/basepoint_radix16_ifma.rs) · [Portable table](../../src/auth/ed25519/basepoint_tables.rs) · [Representation checks](../../src/auth/ed25519/point_avx2.rs#L1102)

## Changes that remove runtime work

- [ ] **T010 Encode RSA keys directly into the final DER buffer.**
  Public PKCS #1 constructs six heap buffers.
  SubjectPublicKeyInfo (SPKI) constructs fourteen.
  Both counts include the returned buffer.
  Private PKCS #8 also materializes a complete private PKCS #1 encoding and copies it.

  Size the output once.
  Reuse the direct integer writer and write each nested object into its destination.
  Keep private output in a zeroizing owner.
  Check sign padding, long-form lengths, canonical bytes, and independent DER round trips.
  [Public encoders](../../src/auth/rsa.rs#L5946) · [Direct writer](../../src/auth/rsa.rs#L7539) · [Private encoder](../../src/auth/rsa.rs#L1683) · [Private oracle](../../tests/rsa_public_key.rs#L1461) ·
  [Public encoding test](../../tests/rsa_public_key.rs#L3426)

- [ ] **T011 Remove temporary allocations and a secret copy from the RSA key-generation DRBG.**
  Every key update allocates a vector of slice references, including the update after each fill.
  Its HMAC helper copies a 32-byte key into a temporary 64-byte block.

  Stream the value, domain byte, and seed slices into HMAC.
  XOR the key directly into the initialized pads.
  Fill output by chunks.
  Preserve domain order, the post-fill update for empty output, feature independence,
  and secret cleanup.
  [Fill and update](../../src/auth/rsa.rs#L8043) · [Slice list and HMAC](../../src/auth/rsa.rs#L8081)

- [ ] **T012 Calculate the vectored HKDF info length once.**
  SHA-256 and SHA-384 rescan the slice lengths for every output block.
  A maximum-length expansion repeats the sum 255 times.
  Calculate it after the output checks and pass it to each block.
  Retain SHA-256's `u64` and SHA-384's `u128` overflow semantics.
  [SHA-256 scan](../../src/auth/hkdf.rs#L1005) · [SHA-384 scan](../../src/auth/hkdf.rs#L1082)

- [ ] **T013 Remove two temporary vectors from prepared TLS certificate verification.**
  Each prepared X.509 path builds one vector of algorithms and one of trait-object references.
  Cargo features fix the inventory.
  Build bounded arrays from that inventory.
  Preserve algorithm order, client and server verification, errors, and fallback behavior.
  Check stack use with the largest supported feature set.
  [Server path](../../rscrypto-rustls/src/verify.rs#L164) · [Client path](../../rscrypto-rustls/src/verify.rs#L267) · [References](../../rscrypto-rustls/src/verify.rs#L318) · [Algorithm vector](../../rscrypto-rustls/src/verify.rs#L682) · [Inventory](../../rscrypto-rustls/src/verify.rs#L819)

- [ ] **T014 Enforce single-use QUIC address-token sealing through ownership.**
  Every seal creates a fresh boxed cipher and performs an atomic swap on its only call.
  Make the private sealer consume `self: Box<Self>`.
  Remove the atomic guard.
  Preserve boxed construction, stack cleanup, and length rejection before entropy consumption.
  Update the lifecycle description.
  [Seal caller](../../rscrypto-rustls/src/quic/token.rs#L108) · [Guard](../../rscrypto-rustls/src/quic/token.rs#L191) · [Cleanup contract](../secret-lifecycle.md#rustls-quic-integration)

- [ ] **T015 Share the resolved SHA-2 compression entry within each compression family.**
  The dispatch macro constructs one `OnceCache` per digest.
  T001 identifies the shared SHA-512 policy.
  Keep the initial state, finalization, diagnostic names, compile-time selection,
  and capability checks for each digest.
  [Cache construction](../../src/macros.rs#L172) · [SHA-384 instance](../../src/hashes/crypto/sha384/dispatch.rs#L6)

## Further production simplifications

- [ ] **T016 Share SHA-2 streaming buffers within the SHA-512 family** (60–120 lines).
  SHA-384, SHA-512, and SHA-512/256 repeat block buffering and tail handling.
  Share the private update operation.
  Retain secret finalization and each digest's output handling.
  [SHA-384](../../src/hashes/crypto/sha384.rs#L142) · [SHA-512/256](../../src/hashes/crypto/sha512_256.rs#L105) · [SHA-512](../../src/hashes/crypto/sha512/mod.rs#L309)

- [ ] **T017 Share fixed-output SHA-3 wrappers** (140–220 lines).
  Four implementations repeat the same `Digest` operations.
  Reuse the existing family pattern and retain each rate and output width.
  Preserve SHAKE's specialized batch paths and the public and secret Keccak state boundaries.
  [SHA-3 implementations](../../src/hashes/crypto/sha3.rs#L65) · [cSHAKE precedent](../../src/hashes/crypto/cshake.rs#L9)

- [ ] **T018 Share BLAKE2 parameter-builder and fixed-output forwarding** (100–200 lines).
  Share matching BLAKE2b and BLAKE2s methods.
  Keep each variant's key, salt, personalization, and output limits explicit.
  Retain the existing compression backends.
  [BLAKE2b](../../src/hashes/crypto/blake2b/mod.rs#L658) · [BLAKE2s](../../src/hashes/crypto/blake2s/mod.rs#L582)

- [ ] **T019 Share the HMAC-SHA-384 and HMAC-SHA-512 lifecycle methods.**
  Both types repeat `Mac`, reset, verification, and cleanup methods.
  Retain tag width, secret-state clearing, and the SHA-256 specializations.
  [HMAC-SHA-384](../../src/auth/hmac.rs#L737) · [HMAC-SHA-512](../../src/auth/hmac.rs#L1014)

- [ ] **T020 Use the existing word-wipe primitive at six HKDF sites.**
  Replace the inline volatile word loops in expansion and drop.
  Keep each wipe at its current point and retain each compiler fence.
  [HKDF wipe](../../src/auth/hkdf.rs#L426) · [Word wipe and fence](../../src/traits/ct.rs#L372)

- [ ] **T021 Share matching big-endian word writers across HKDF and PBKDF2.**
  The SHA-2 paths repeat the 32-bit and 64-bit conversions.
  Keep the destination lengths and SHA-384 truncation explicit.
  [HKDF writers](../../src/auth/hkdf.rs#L28) · [PBKDF2 writers](../../src/auth/pbkdf2.rs#L63)

- [ ] **T022 Remove two unsafe blocks from HMAC pad access.**
  Replace raw-pointer word reads, writes,
  and repeated offsets with fixed eight-byte chunks and native-endian conversions.
  Compare generated instructions on the affected hot paths before accepting the replacement.
  [HMAC pad construction](../../src/auth/hmac.rs#L76)

- [ ] **T023 Share matching AES key forwarding methods** (180–320 lines).
  The AES-128 and AES-256 `AesKey` implementations repeat backend forwarding.
  Share the methods at compile time.
  Retain key schedule sizes, target features, and secret cleanup.
  [AES-128](../../src/aead/aes.rs#L1026) · [AES-256](../../src/aead/aes.rs#L1426)

- [ ] **T024 Share ECDSA keypair and public-key forwarding** (90–200 lines).
  P-256 and P-384 repeat these wrappers.
  Keep entropy widths, key validation, signing kernels, and public types specific to each curve.
  [P-256](../../src/auth/ecdsa.rs#L1062) · [P-384](../../src/auth/ecdsa.rs#L1165)

- [ ] **T025 Share ML-DSA and composite TLS signer adapters** (35–50 lines).
  Both adapters repeat scheme selection, `Arc` ownership, and SPKI access.
  Keep each family's compact and prepared key forms, signing operations, and error translation.
  [ML-DSA signer](../../rscrypto-rustls/src/sign/mldsa.rs#L96) · [Composite signer](../../rscrypto-rustls/src/sign/composite.rs#L105)

- [ ] **T026 Share checksum and digest I/O adapter implementations** (85–160 lines).
  Four reader and writer types repeat construction, accessors, finalization, and forwarding.
  Keep the four public types.
  Preserve short reads and writes, vectored I/O, and the bytes included in each result.
  [Checksum reader](../../src/traits/io.rs#L168) · [Checksum writer](../../src/traits/io.rs#L304) · [Digest reader](../../src/traits/io.rs#L437) · [Digest writer](../../src/traits/io.rs#L562)

- [ ] **T027 Share AArch64 RSA Montgomery-reduction bindings** (70–85 lines).
  Linux and Darwin repeat scratch calculations, reduction wrappers, and pointer proofs.
  Select the assembly source and external link names by target.
  Keep one Rust wrapper with the existing scratch and ABI requirements.
  [Darwin bindings](../../src/auth/rsa_aarch64_asm.rs#L10) · [Linux bindings](../../src/auth/rsa_aarch64_linux_asm.rs#L10)

- [ ] **T028 Share RSA public exponent schedules** (100–200 lines).
  Several multiplication backends repeat schedules for public exponents 3, 17, and 65,537.
  Share those schedules at compile time.
  Retain the independent generic and diagnostic algorithms.
  Compare generated code for each affected backend.
  [Production schedule](../../src/auth/rsa.rs#L6970) · [Alternate backend](../../src/auth/rsa.rs#L7249)

- [ ] **T029 Read the CRC configuration once per diagnostic constructor.**
  Four constructors repeat field assembly and call `config()` twice.
  Share matching construction and preserve polynomial, reason, and selected kernel.
  [CRC diagnostic constructors](../../src/checksum/diag.rs#L106)

- [ ] **T030 Reuse QUIC Retry tag calculation during verification.**
  Generation and verification construct the same authenticator for empty plaintext.
  Share the calculation and use the library's constant-time tag comparison.
  Preserve rejection of short packets, oversized connection IDs, and incorrect tags.
  [Tag generation](../../rscrypto-rustls/src/quic/initial.rs#L44) · [Verification](../../rscrypto-rustls/src/quic/initial.rs#L62)

## Kernel changes that need target evidence

These four tasks have the highest cost for establishing throughput and constant-time equivalence.
Compare target instructions and portable results before accepting a shared implementation.

- [ ] **T031 Share the x86 SHA-512 round schedule** (about 205 matching lines).
  Share the schedule expansion between AVX2 and AVX-512VL.
  Retain the intrinsics and CPU feature requirements of each kernel.
  [AVX2 schedule](../../src/hashes/crypto/sha512/x86_64_avx2.rs#L302) · [AVX-512VL schedule](../../src/hashes/crypto/sha512/x86_64_avx512vl.rs#L589)

- [ ] **T032 Share RISC-V AEGIS-256 encryption and decryption framing** (130–160 lines).
  The scalar and vector crypto backends repeat the surrounding operations.
  Keep the AES round instructions and target features in each backend.
  [Scalar backend](../../src/aead/aegis256/riscv64_zkne.rs#L68) · [Vector backend](../../src/aead/aegis256/riscv64_zvkned.rs#L69)

- [ ] **T033 Share matching portable P-256 and P-384 limb helpers.**
  Both curves repeat carry, borrow, and mask operations.
  Retain the deliberate s390x arithmetic and check its emitted code.
  [P-256 helpers](../../src/auth/p256_portable.rs#L1352) · [P-384 helpers](../../src/auth/p384_portable.rs#L1022)

- [ ] **T034 Share Poly1305 limb scaling and narrowing within the AEAD implementation.**
  Two backend bodies repeat these helpers.
  Retain the architecture-specific multiply and reduction operations and the partial-block rule.
  [First implementation](../../src/aead/poly1305.rs#L495) · [Second implementation](../../src/aead/poly1305.rs#L777)

## Tests, tooling, and documentation

- [ ] **T035 Share independent-oracle test drivers** (450–750 lines).
  AES-GCM, AES-GCM-SIV, KMAC, and SHAKE repeat boundary sweeps across widths.
  Parameterize the drivers.
  Keep each independent oracle, seed, vector, and rejection case.
  [AES-GCM](../../tests/aes128gcm_oracle.rs) · [AES-128-GCM-SIV](../../tests/aes128gcmsiv_oracle.rs#L23) · [AES-256-GCM-SIV](../../tests/aes256gcmsiv_oracle.rs#L22) · [KMAC](../../tests/kmac128_differential.rs#L31) · [SHAKE](../../tests/shake128_differential.rs)

- [ ] **T036 Share fuzz target implementations within families** (200–350 lines beyond T002).
  ML-KEM-512/1024 and HKDF-SHA-256/384 repeat input handling and assertions.
  Keep each fuzz target, corpus, and algorithm-specific case.
  [ML-KEM-512](../../fuzz/target_impls/auth_mlkem512.rs#L6) · [ML-KEM-1024](../../fuzz/target_impls/auth_mlkem1024.rs) · [HKDF-SHA-256](../../fuzz/target_impls/auth_hkdf_sha256.rs#L4) · [HKDF-SHA-384](../../fuzz/target_impls/auth_hkdf_sha384.rs)

- [ ] **T037 Replace tests that cannot detect the claimed defect.**
  The `ToyKem` test exercises its own algorithm.
  Exercise the `Kem` trait through production ML-KEM and its existing vectors.

  The `DispatchInfo` copy test uses `let _ = info`, which does not require `Copy`.
  Remove that test.
  Remove nonempty-string assertions where concrete backend-selection checks cover the behavior.
  Retain checks of the actual selected backend.
  [Toy implementation](../../tests/api_consistency.rs#L61) · [Toy test](../../tests/api_consistency.rs#L566) · [Copy test](../../src/checksum/introspect.rs#L65) · [Diagnostic tests](../../src/hashes/introspect.rs#L193) · [ML-KEM operations](../../tests/mlkem_ops.rs#L35) ·
  [ML-KEM vectors](../../tests/mlkem_acvp.rs#L134)

- [ ] **T038 Reuse fixture decoders and identical RSA inputs** (80–200 lines).
  Several tests duplicate strict hex decoding.
  RSA tests and benchmarks also repeat large key literals.
  Use the shared test decoder and independent fixture bytes.
  Keep the CAVP decoder's support for odd-length hexadecimal integers.
  [Shared decoder](../../tests/common/mod.rs#L6) · [RSA decoders](../../tests/rsa_public_key.rs#L205) · [RSA fixture](../../tests/rsa_public_key.rs#L521) · [CAVP decoder](../../tests/rsa_nist_cavp.rs#L79)

- [ ] **T039 Share benchmark setup within algorithm families** (200–450 lines).
  ECDSA P-256/P-384 signing and AES-128/256-GCM decryption repeat setup, peer selection,
  and validation.
  Preserve benchmark IDs, input preparation, blinding policy,
  and the exact work inside each timed closure.
  [P-256 signing](../../benches/auth.rs#L1275) · [P-384 signing](../../benches/auth.rs#L1618) · [AES-256-GCM decryption](../../benches/aead.rs#L1104) · [AES-128-GCM decryption](../../benches/aead.rs#L1342)

- [ ] **T040 Share constant-time harness entry-point bodies.**
  P-256 and P-384 repeat public-key and signing wrappers for default
  and explicitly blinded operations.
  Generate matching bodies.
  Retain exported symbols, input widths, blinding inputs, and measurement boundaries.
  [P-256 entry points](../../tools/ct-harness/src/lib.rs#L1372) · [P-384 entry points](../../tools/ct-harness/src/lib.rs#L1476)

- [ ] **T041 Group identical assembly waiver records** (about 780 gross lines).
  Of 562 records, 497 are distinct when the comparison excludes only `primitive`.
  Explicit primitive lists could replace 65 extra records.
  Compare expanded records field for field, including the reviewed slice digest, classification,
  rationale, reviewer, and date.
  The renewal tool edits by index.
  Preserve the ability to renew one primitive independently and reject stale or duplicate waivers.
  [Waiver records](../../ct.toml#L7198) · [Renewal tool](../../scripts/ct/renew_waivers.py#L202)

- [ ] **T042 Share assembly-generator maintenance code** (40–80 lines).
  The P-256, P-384, and paired-RSA scripts repeat mode handling, stale-output reports,
  and rewrite logic.
  Share the check/write loop.
  Keep the arithmetic generators and simulations independent.
  [P-256 script](../../scripts/asm/p256.py#L74) · [P-384 script](../../scripts/asm/p384.py#L78) · [Paired-RSA script](../../scripts/asm/rsa_pair.py#L34)

- [ ] **T043 Correct the provider's private-key cleanup documentation.**
  The ownership document describes the `rustls-pki-types` 1.15.1 guard.
  The provider pins 1.16.0.
  Identify the build covered by the existing evidence.
  Cite or verify the current pin before extending the cleanup claim.
  [Ownership statement](../secret-ownership.md#rustls-authentication) · [Provider pin](../../rscrypto-rustls/Cargo.toml#L63)

## Intentional copies

- Keep the provider's QUIC vectors in its crate.
  Its tests consume those files from the independently packaged archive.
  [Vector provenance](../../rscrypto-rustls/tests/vectors/README.md) · [Package check](../../scripts/check/package.py#L85)
- Keep the generated paired RSA kernel source.
  Its generator checks that the committed output matches.
  T042 covers the maintenance code around that generator.
  Both exclusions fall outside the savings estimates.
  [Generated kernel](../../src/auth/rsa_aarch64_pair.rs) · [Generator](../../scripts/asm/rsa_pair.py#L8)
