# Secret lifecycle

`rscrypto` clears its named secret owners and its explicit secret temporaries on success, failure,
early return, reuse, and drop.
Cleanup uses volatile writes and a compiler fence.

## Scope

The claim covers:

- arrays that the crate owns;
- initialized heap storage;
- reusable scratch;
- parser and generator staging;
- finalized keyed-hash snapshots;
- expanded key state.

The claim does not cover:

- input that the caller owns;
- ordinary bytes after an explicit export;
- register or spill copies that the compiler makes;
- swapped pages and crash dumps;
- hardware-backed storage.

## Cleanup boundaries

Keyed and derive-key BLAKE3 batches clear their named key and fallback digest
owners. `merge_level` clears its bounded
child/parent scratch in every mode. RAII covers normal return and ordinary
unwinding after population; no whole-route erasure follows. The
delivery evidence (`benchmark_results/blake3-delivery-20261008T185226Z/report.md`, local)
records the reviewed compiler/target scope. Existing Portable state, outer
by-value keys, native fallbacks, plain-array unwind, returned outputs and
compiler/register/spill copies remain separate limitations.

The Bao decoder clears its fixed input buffer on returned errors and on drop.
A caught reader panic permanently disables decoding; the buffer is cleared
when the decoder drops. No claim covers the underlying reader or bytes already
returned to the caller. Abort and double-panic exits do not guarantee cleanup.

The BLAKE3 buffered reader methods clear every initialized byte of their owned
stack and heap input storage before release on success, I/O error, and panic
unwinding. Storage is zeroed only as reads need it, so the clear covers the bytes
passed to readers and the heap alignment padding. The heap allocation is
reserved before it holds input and never reallocates; alignment uses a borrowed
slice within it. An abort does not run destructors. This buffer contract does not
extend cleanup claims to the caller's reader, operating-system caches, or
compiler-created copies.

The Linux AArch64 plain parallel reader clears its named 64-byte `ParentBlock`
on normal return and unwinding after construction. The
ordinary native review (`benchmark_results/blake3-linux-reader-20261008T074000Z/cleanup-review/arm64-review/report.md`, local)
retains emitted clears and fences for this owner and for the fixed input
allocation that preceded the current reader storage. The current storage clears
through the same `ct::zeroize` and has no separate machine-code review.
The recursive return copy, caller-side CV argument and recursive plain arrays
remain outside that scope. This does not close the separate Portable-state,
outer-key, native-fallback or compiler-spill findings.

The BLAKE3 SIMD128 backend clears its named vector, tail, parent, and output scratch in keyed
and derive-key modes. Its plain tiny-input specialization adds no secret route. The
[WASM backend record](../benchmark_results/OVERVIEW.md#2026-10-06-blake3-wasm-four-lane-backend)
retains source-bound cleanup evidence and the separate pre-existing Portable state-owner gap.
The parent caller review (`benchmark_results/blake3-wasm-parent-cost-20261006T063400Z/secret-review/key-borrow-follow-up.md`, local)
also records existing outer by-value key copies without explicit cleanup. The parent SIMD
borrow introduces no key-copy owner and preserves explicit clears; differing stack placement is not proof
of identical post-return residue. These outer copies remain a separate cleanup follow-up.
The [state-owner reduction](../benchmark_results/OVERVIEW.md#2026-10-06-blake3-wasm-state-ownership)
removes the separate vector CV array while preserving full round-state, message,
padding, and output cleanup. Its smaller guest frame does not imply smaller native
frames or equal physical residue.
The [padding-clear guard](../benchmark_results/OVERVIEW.md#2026-10-06-blake3-wasm-unused-padding-cleanup)
skips wiping padding that received no input bytes. A secret partial block still
receives complete padding cleanup; all state, message, output clears and fences remain.
That outcome left the normal generic frame and padding initialization unchanged.
The later [root-mode specialization](../benchmark_results/OVERVIEW.md#2026-10-06-blake3-wasm-root-mode-specialization)
lets full chunks omit unused padding storage and initialization while preserving
all populated secret-owner clears and fences. Plain partial batches retain padding;
OR-ing ROOT preserves the secret-mode bits for the generic source contract.
Native batch frames grow, and observed compiler-created key spills lie outside the
named guest clears. This is not a guarantee of erased native slots or equal residue.

The BLAKE3 partial-chunk forest clears its populated leaf, parent and frontier
scratch on normal return and unwinding in keyed and derive-key modes. Its NEON lane
worker clears named vector working storage in every mode. Abort exits, the AVX-512
lane workers' register and spill copies, and by-value frontier copies before the
existing stack merge remain outside this claim.

The shared BLAKE3 root-output fallback clears its decoded words and compression output
after copying them in keyed and derive-key modes. Optimized WASM evidence and its limits are retained in
the [WASM SIMD128 record](../benchmark_results/OVERVIEW.md#2026-10-06-blake3-wasm-simd128).

| Owner or operation | When cleanup happens |
| --- | --- |
| Typed keys, private keys, and shared secrets | On concrete or nested `Drop`. A consuming export clears the source, or moves responsibility to the caller explicitly. |
| AEAD and header protection | Context drop clears the retained keys. Operation-local schedules, authentication state, and materialized cipher output are cleared after use. A failed open clears the unauthenticated plaintext. |
| Poly1305 | The shared arithmetic owner clears its clamped key, accumulator, and additive pad on drop. The standalone authenticator also clears its partial-block buffer. AEAD framing and accelerated scratch retain their own cleanup; the AVX-512 IFMA AEAD authenticator clears its stack copies of the key powers and its scalar state. No vector backend clears key powers held only in vector registers. |
| HMAC, HKDF, KMAC, PBKDF2, and keyed BLAKE2/BLAKE3 | Finalization copies, keyed prefixes, work buffers, emitted blocks, and replaced state are cleared after their last use. HKDF state clears its PRK and HMAC prefixes on drop, whether `new`, `extract`, or `from_prk` built it. Extract writes the HMAC tag into a cleared PRK copy, and the compression function sees only cleared scratch; the state is built from plain values. |
| ECDSA, P-256 ECDH, P-384 ECDH, ML-KEM, and RSA private work | Secret scalars, digests, limbs, encoded messages, inverse state, and initialized scratch are cleared on every return path. See the notes below. |
| Ed25519 and X25519 private work | Secret owners and staging are cleared; arithmetic temporaries have a narrower boundary. See [Ed25519 and X25519](#ed25519-and-x25519). |
| ECDSA and Ed25519 key encodings | See [ECDSA and Ed25519 key encodings](#ecdsa-and-ed25519-key-encodings). |
| ML-DSA private work | See [ML-DSA](#ml-dsa). |
| SLH-DSA private work | See [SLH-DSA](#slh-dsa). |
| Argon2 and scrypt | See [Argon2 and scrypt](#argon2-and-scrypt). |
| Secret parsing and generation | RAII owners cover success, parse failure, entropy failure, and early return. |
| Caller-filled secret owners, P-256 and P-384 ECDH generation, and ECDSA blinding | See [Caller-filled owners](#caller-filled-owners). |

### RSA private arithmetic

- The CRT exponentiation keeps its limbs, window tables, and engine workspaces in
  `RsaPrivateScratch`, which every private operation clears before it returns.
  This includes the radix-2^52 IFMA workspace and the paired radix-2^28 NEON workspace.
- Each blinding attempt keeps its candidate factor, the factor's inverse, and a fresh inverse
  mask in `RsaPrivateScratch`. The masked inverse clears the mask and the masked product on
  every return, and the inverse on failure. A rejected candidate clears the factor and the
  inverse before the next draw. The allocating path inverts through a temporary scratch and
  clears it before it returns.
- **Not claimed:** the general-purpose and vector registers that the AArch64 and x86-64
  Montgomery, IFMA, and paired NEON kernels and the AArch64 NEON window selection use.

### RSA implicit rejection

Legacy PKCS#1 v1.5 decryption hashes the stored private exponent at the public
modulus width. Import and generation pad its encoding once and retain the CRT
coefficient at the factor width; key destruction clears both owners. Decryption
clears the exponent hash after deriving the per-ciphertext key derivation key
(KDK). SHA-256 uses its secret finalization path, which clears
the copied state and final block; the live hash states clear on drop.

Each internal HMAC call clears its padded key block, ipad, opad, and inner
digest. The implicit-rejection PRF (IRPRF) clears every returned HMAC block after
copying its required bytes. The decoder clears the KDK, the 256 candidate-length
bytes, and the alternative message (AM) on every ordinary return, including
public-length errors. AM is either an operation-local heap buffer or
`RsaPrivateScratch::blinding_power`; scratch cleanup also clears
the latter after the operation. The allocating caller clears the encoded
message and AM before freeing them, including private-operation failure.

Output selection reads both encoded and alternative messages in full and moves
the chosen suffix through a fixed sequence of shifts. It clears the remainder
of `out[..k - 11]`; bytes beyond that window remain untouched on success.
The caller owns the returned message. Public-input or private-operation errors
clear the entire output. These source-level cleanup boundaries do not establish
erasure of every compiler-created copy, register, or spill.

### ML-KEM

- A prepared decapsulation key keeps a decryption mask derived from the implicit-rejection secret.
  Both of its polynomials are cleared when the prepared key drops.
- The secret SHA-3 and SHAKE calls (G, J, and the PRF) run in an out-of-line worker.
  After the worker returns, a fixed volatile scrub clears its dead stack: 2 KiB,
  or 4 KiB after the four-lane PRF.
  `just stack-frames` checks that each linked worker fits inside its scrub on the reviewed targets.
- By-value key generation, import, and preparation leave moved-from copies in the caller's frame.
  The `*_in` constructors, including PKCS #8 import, fill a box that the caller allocated, in place.
  In the measured RV32 and Cortex-M runs, `generate_keypair_in`, `try_from_slice_in`, and
  `from_pkcs8_der_in` leave no copies. Residue of `prepare_in` and `generate_prepared_keypair_in`
  is not measured.
- The pairwise check of PKCS #8 expanded-key import encapsulates with public coins and
  decapsulates under data-independent timing. It clears both shared secrets before it returns.
  A rejected import clears every owner it built.

### P-256 ECDH

- Portable P-256 ECDH uses a separate projective owner type.
  Its coordinates are cleared each time an intermediate is replaced, and when the operation returns.
- The selected AArch64 and x86-64 assembly clears secret-derived frames, saved-register spill slots,
  and volatile integer registers before it returns.
- The Windows public-point batch wrapper handles only public coordinates.

### P-384 ECDH

- P-384 ECDH has no register-clearing assembly.
- Each intermediate overwrites the Jacobian accumulator owner in place.
  The owner is cleared on return.
- The scalar limbs, the recoded window digits,
  and the affine shared x-coordinate are cleared before return.
- **Not claimed:** field-arithmetic and table-selection temporaries outside those owners.
  This includes the general-purpose and vector registers that the AArch64 inline assembly uses,
  and the stack frames and registers of the x86-64 fused point doubling and addition.

### Ed25519 and X25519

- Ed25519 clears its expanded secret and signing scalar/digest staging.
  X25519 clears its secret-key and clamped-scalar owners.
- **Not claimed:** recoded digits, cached selections,
  and field/point arithmetic temporaries in the Rust Edwards fixed-base workers.
  This includes the AVX2 and IFMA selectors' stack frames and register-save slots.
- The separate AVX2 selector boundary supports binary constant-time analysis.
  It does not establish whole-operation stack or register cleanup.

### ECDSA and Ed25519 key encodings

- PKCS #8 and SEC1 import borrow the caller's input and never clear it.
  The import copies the scalar or key out of the input into its secret owner.
- A scalar outside the group order, or a key whose carried public key is not its own,
  is cleared before the error returns.
- PKCS #8 export writes into a fixed-size buffer that the caller owns and clears.
  ECDSA export and ECDSA import of a carried public key derive the public key
  through the existing public-key path.
- **Not measured:** moved-copy residue of the by-value copy out of the input.

### ML-DSA

- Expanded keys, transformed polynomials, seeds, randomness, message representatives, SHAKE state,
  rejected candidates, and Serde staging use zeroizing owners.
- Ordinary decode and message hashing fill existing owners in place.
- Private SHAKE256 finalization and squeezing use a local zeroizing reader.
  The reader is initialized before the absorbed state is copied into it.
  It is dropped before control returns to the sampler or the hashing caller.
- The paired mask sampler copies its two locally built SHAKE256 states into zeroizing readers
  and clears the local states.
  Both helpers run in the scrubbed out-of-line worker described under [ML-KEM](#ml-kem),
  and `just stack-frames` checks that worker.
- Secret-noise acceptance masks and counts share the wiped bit-plane owners.
- By-value key constructors leave moved-from copies in the caller's frame.
  The `*_in` constructors, including PKCS #8 import, fill a box that the caller allocated, in place.
  In the measured RV32 and Cortex-M runs, the `*_in` constructors leave no copies.
- PKCS #8 import borrows the caller's DER and never clears it.
  PKCS #8 export writes into a fixed-size buffer that the caller owns and clears.
  A rejected import clears every owner it built before it returns.
- Preparation decodes directly into its storage, without moving the prepared owner:
  storage that the caller owns for `prepare`, or a box allocated first for `prepare_in`.
  It copies rho, `K`, and tr from the key into the storage.
  The prepared key clears the transformed polynomials and `K` on drop and when preparation fails.
  The storage clears them again on drop.
  When `prepare_in` was added (`44dfa51d`), its release frames on five Linux targets were at most
  2,240 bytes, so the storage was not built on the stack; residue of `prepare_in` is not measured.
- The 2026-10-16 macOS AArch64 signing-residue check found no selected secret
  windows after 240 compact/prepared, deterministic/hedged calls across all three
  parameter sets and native/portable dispatch. It includes NTT masks, rejected
  responses and commitments, and hint maps from candidates rejected after hint
  construction. Planted/cleared controls and an isolated missing-polynomial-wipe
  mutation validate the scanner.
  This checks the measured stack snapshots, not registers, heap residue or arbitrary
  smaller fragments. See the [qualification record](../benchmark_results/OVERVIEW.md#2026-10-16-ml-dsa-optimization-qualification).
- Full target qualification is still open.
  See [ML-DSA](mldsa.md).

### SLH-DSA

- Secret keys keep SK.seed and SK.prf in zeroizing owners, apart from the public key.
  Key generation and import fill the owners in place; a failed callback or a rejected key
  clears every owner already built.
- Hedged signing keeps `addrnd` in a zeroizing owner.
  Every PRF and F call, which may hold SK.seed, a WOTS+ chain value, or a FORS leaf secret,
  clears its hash block and state copies before it returns.
  The HMAC and SHAKE state of PRF_msg clears on drop.
  Signing and verification clear the message digest after use.
- WOTS+ chain values advance in place in the caller's signature buffer to their revealed positions.
  Signing fails only before it starts, and then zero-fills the buffer.
- Raw and PKCS #8 import borrow the caller's input and never clear it.
- Signing and verification leave no SK.seed, SK.prf, or `addrnd` bytes in the measured RV32 and
  Cortex-M runs; key generation and both imports return the key by value and leave moved-from copies.
  See [the residue results](#optimized-evidence).
- **Not claimed:** register and spill copies in the hash workers.
  Target qualification is still open.

### Argon2 and scrypt

- Every block that an operation uses is cleared before the operation returns, also on error paths.
  This applies when `rscrypto` allocates the work memory,
  and when the caller supplies it to an Argon2 or scrypt `*_with_memory` method.
- An input error returns before the operation touches caller memory.
- Blocks past the required length are never written.
- PHC `verify_password_with_memory` and Argon2's
  `verify_password_with_context_and_memory` approve the record and its resource
  limits before borrowing the required workspace prefix for derivation.
  Malformed records and short workspaces leave caller memory unchanged;
  password or context mismatches clear the used prefix. The computed PHC digest
  remains in `ZeroizingBytes<32>` and is cleared when that owner drops.

### Caller-filled owners

- The zero-initialized owner exists before the callback runs.
- These cases all reach its `Drop`: success, immediate failure, partial-fill failure,
  and P-256 or P-384 scalar rejection or exhaustion.
- If the ECDSA callback fails, the operation returns before message hashing
  and before private scalar arithmetic.
- If the ECDH callback fails, the operation returns before public derivation and before agreement.

### rustls symmetric adapters

`rscrypto-rustls` holds HMAC keys, HKDF state, and AEAD/header-protection contexts
in the clearing owners named in the [ownership inventory](secret-ownership.md#rustls-symmetric-adapters).
No provider expansion loop duplicates HKDF. The adapter's fixed expansion
scratch clears on return or unwind after rustls copies the output.
Key construction crosses by-value arrays into typed rscrypto owners; compiler-made
copies at that boundary remain unqualified.

An entropy fill error clears the entire destination, including partial writes.
Record encryption reserves the whole record in rustls's output vector before it
copies plaintext, so growth leaves no reallocated plaintext copy; a sealing
failure clears the bytes it appended, then truncates them.
An AEAD authentication failure clears the body before returning an opaque rustls
error. Public length and key-shape rejections occur before decryption; those
paths leave ciphertext unchanged. Successful plaintext belongs to rustls or its
caller. The provider does not clear external record storage after successful use.
These source contracts add no linked-code, residue, or timing qualification.

### rustls key exchange

Key generation uses fallible entropy callbacks into the primitives' clearing
scratch owners. Partial-fill failure propagates without a share. Malformed
peer shares consume and drop the active exchange. Successful agreement copies
the primitive result into rustls's clearing `SharedSecret`; hybrid fallback
also drops the unused prepared ML-KEM decapsulation key. Ownership is listed in the
[key-exchange inventory](secret-ownership.md#rustls-key-exchange).
The adapter adds no raw private-key export or explicit secret duplication.
Compiler-made moves and stack residue have not been separately qualified here.

### rustls authentication

The provider's key loader uses rscrypto's bounded credential codecs and shares
successful clearing key owners with signers. Rejected parsed keys drop before
another format is attempted. Each RSA signature builds and drops its own clearing
scratch; entropy failure returns no signature.
ECDSA and hedged ML-DSA entropy failure also return no signature. ML-DSA keys are
prepared once by default and the compact key is dropped; a compact-mode key prepares
temporary storage per signature. ML-DSA tests cover partial entropy failure and
successful reuse in both forms for all three parameter sets. `rustls::crypto::Credentials::new`
compares exported public SPKI with the certificate before use.

The adapter tests cover partial entropy failure, subsequent RSA signing,
malformed and oversized credential rejection, and independent signature
verification. They do not qualify compiler-made copies or native timing.

Input DER enters a `zeroize::Zeroizing` guard before validation. The pinned
pki-types/zeroize implementation wipes owned vector storage before deallocation
on every return path and during unwinding. Borrowed static bytes remain unchanged.
On `aarch64-apple-darwin` with `nightly-2026-09-30` and zeroize 1.9.0, pre-deallocation observations
cover all 13 accepted key forms, rejection, partial entropy failures, and unwind.
A guard-removal control leaves the input uncleared. A separate, uninstrumented
O3/fat-LTO/panic-abort artifact retains volatile writes through the vector's
full capacity before deallocation. This evidence covers the owned input
allocation, not earlier caller copies, parsed keys, registers, or abort cleanup.
The local record is `benchmark_results/provider-entropy-closure-20261011T152551Z/`.
Key loading and signers retain the selected entropy source; partial failures
produce no signature and never select an OS fallback. Regression tests exercise
ECDSA import, every signing family, and recovery through the same signer.

### rustls session tickets

Ticket generation fails on partial entropy acquisition without exposing a key.
A nonce reservation is never returned to the counter, including after allocation
or sealing failure. The bounded counter cannot wrap. Temporary plaintext uses
`SecretVec::try_fill_with`, which clears initialized bytes on failure or unwind.
The adapter transfers successful decryption output to rustls through the required
`Vec<u8>` interface; this transfer does not promise cleanup of rustls's copies.

The six-hour rotator keeps current and previous clearing key owners. Each key
also enforces a monotonic twelve-hour deadline before either seal or open and
takes/drops its cipher on the first expired operation. Expired keys cannot be
revived by an earlier timestamp. On-use expiration can defer destruction while idle, and rustls configuration sharing can
extend the rotator's lifetime. Tests cover independent AES-GCM, hostile tickets,
concurrent exhaustion, entropy failure, resumption, expiry, and bounded real-clock
rotation. They do not qualify compiler-made copies or linked-code erasure.

### rustls QUIC integration

Endpoint reset and server token-key acquisition use the selected fallible entropy source and
clearing owners. Token derivation clears its intermediate key. The derived
sealer atomically consumes its one-use permission before encryption and rejects
plaintext above 4 KiB. Authentication failure clears the decrypted token body;
length rejection occurs before mutation. Token buffers and authenticated
metadata remain owned by Quinn.

Failed TLS handshake processing is sticky. The adapter suppresses subsequent
handshake writes, traffic-key updates, exporters, identity, and handshake data.
Rustls types own destruction of handshake, next-generation traffic, and exporter
secrets, including owners retained by the adapter's event queue and session.
Packet-key wrappers preserve the primitive's authenticated-decryption failure
contract. Source-level clearing and functional failure tests do not measure
compiler-created copies or establish native timing evidence.

## Export and capacity

- `SecretBytes::expose()` clears its source, then returns an ordinary array.
- `SecretVec::into_unprotected_vec()` moves the existing allocation without clearing it.
- `SecretString::into_unprotected_string()` does the same for a UTF-8 allocation.

This difference is intentional.

`SecretVec` and `SecretString` clear every byte in their initialized length.
They do not claim to clear spare capacity,
because these owners do not expose it as initialized memory.
`SecretBytes<N>` always clears all `N` bytes.

When panic unwinding is enabled,
an owner that already wraps callback storage is dropped during the unwind.
Process abort, termination, and power loss do not run destructors.
They have no cleanup claim.

## Optimized evidence

`just check` and `just ci-check` do not verify optimized zeroization.
Source cleanup and passing tests do not prove that secret stores survive optimization.
Machine-code evidence applies only to the compiler, target, features, and operation inspected.

`just ct-full` also checks a destructor sentinel in the linked release harness: an 8-byte-aligned `SecretBytes<32>`.
The check needs complete volatile clearing and a compiler fence in the emitted release-LTO IR.
It keeps the linked symbol, the disassembly, and the artifact hashes.
Missing or partial cleanup fails the gate.
The sentinel does not qualify other alignments, owners, heap storage, error paths,
or compiler-made copies.
The retained machine code still needs review for each target.

`just stack-residue` measures moved-copy residue under QEMU on the RV32 `virt` board and the Cortex-M3 `mps2-an385` board,
with native and portable backends.
[`scripts/README.md`](../scripts/README.md) describes the method and its controls.
Results on `nightly-2026-09-25` and QEMU 11.1.2:

- By-value ML-KEM key generation leaves every byte of `dk_pke` (768, 1,152, or 1,536 bytes)
  and of `z` in the caller's frame.
- By-value ML-KEM import leaves all of `dk_pke` except one 16-byte window, and all of `z`.
- By-value ML-DSA key generation leaves `K` and the encoded `s1`, `s2`, and `t0`:
  one whole copy on RV32, and two on Cortex-M3.
- The `*_in` paths leave none of these bytes on the stack or in the freed allocation,
  for every parameter set, both boards, and both backends.

On 2026-10-09 the same harness measured ML-DSA PKCS #8 import from a static input buffer,
on `nightly-2026-09-30` and QEMU 11.1.2, before the change was committed:

- `SecretKey::from_pkcs8_der_in` leaves no seed, `K`, `s1`, `s2`, or `t0` bytes,
  from the seed form and from the expanded form,
  for every parameter set, both boards, and both backends.
- By-value expanded import leaves `K` and `s1`, `s2`, and `t0`:
  three whole copies on RV32, and four on Cortex-M3.
- `Seed::from_pkcs8_der` returns its owner by value and leaves one copy of the 32-byte seed.
- Raw results: `benchmark_results/mldsa-pkcs8-residue-20261009/` (local).

The ML-KEM PKCS #8 scenarios ran the same way:

- `DecapsulationKey::from_pkcs8_der_in` leaves no `dk_pke` or `z` bytes,
  from the seed form and from the expanded form with its pairwise check,
  for every parameter set, both boards, and both backends.
- By-value expanded import leaves all of `dk_pke` except one 16-byte window, and two or three copies of `z`,
  as by-value raw import does.
- `Seed::from_pkcs8_der` returns its owner by value and leaves copies of the seed.
- Raw results: `benchmark_results/mlkem-pkix-residue-20261009/` (local).

On 2026-10-10 the harness measured SLH-DSA at `1805bcdd` plus its uncommitted scenarios,
on `nightly-2026-09-30` and QEMU 11.1.2, for SHA2-128s and the six fast sets, which cover every
SLH-DSA code instantiation:

- Deterministic, hedged, and pre-hash signing leave no SK.seed, SK.prf, or `addrnd` bytes,
  for every measured set, both boards, and both backends. The scenarios now require this.
- By-value key generation and PKCS #8 import leave three whole copies of SK.seed and SK.prf
  on RV32, and two or three on Cortex-M3; raw import leaves two on RV32, and one or two on Cortex-M3.
- Raw results: `benchmark_results/slhdsa-residue-20261010-final/` (local; `SHA256SUMS` inside).

On 2026-10-16 the harness measured HKDF-SHA256, HKDF-SHA384, and HKDF-SHA512 extract,
on `nightly-2026-10-14` and QEMU 11.1.2. Each scenario searches for the PRK and both HMAC prefix states:

- At `29c244fe`, extract left one whole SHA-512 PRK copy on both boards and both backends,
  and 16 bytes of the SHA-384 PRK on Cortex-M3 with the portable backend.
- A build with only the controls and HKDF scenarios found one or two whole copies of both
  prefix states for every hash, board, and backend. The prefix states are as sensitive as the PRK.
- After the fix, extract leaves no PRK or prefix-state bytes for any hash, board, or backend.
  The scenarios now require this.
- Raw results: `benchmark_results/hkdf-extract-residue-20261016/` (local).

An earlier campaign, whose harness was removed on 2026-09-28,
also found 4.2–6.4 KiB of prepared ML-KEM state after import, preparation, and decapsulation.
The rebuilt harness does not measure prepared state, decapsulation, or the Ed25519, X25519,
and ECDSA owners yet.

## Redaction

`tests/secret_redaction.rs` pins the public `Debug` and error behavior.
Errors expose only public sizes or opaque verification failures,
unless a documented variant explicitly returns caller data.
`expert::DisplaySecret` and the diagnostic APIs are deliberate declassification boundaries.

See [`secret-ownership.md`](secret-ownership.md) for the inventory of types.
