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
delivery evidence (`benchmark_results/blake3-delivery-20261008T185226Z/report.md`)
records the reviewed compiler/target scope. Existing Portable state, outer
by-value keys, native fallbacks, plain-array unwind, returned outputs and
compiler/register/spill copies remain separate limitations.

The Bao decoder clears its fixed input buffer on returned errors and on drop.
A caught reader panic permanently disables decoding; the buffer is cleared
when the decoder drops. No claim covers the underlying reader or bytes already
returned to the caller. Abort and double-panic exits do not guarantee cleanup.

The BLAKE3 buffered reader methods clear their owned input allocation before
deallocation on success, I/O error, and panic unwinding. The allocation stays
fixed while it contains input; alignment uses a borrowed slice within it.
An abort does not run destructors. This buffer contract does not extend cleanup
claims to the caller's reader, operating-system caches, or compiler-created copies.

The Linux AArch64 plain parallel reader clears its named 64-byte `ParentBlock`
on normal return and unwinding after construction. The
ordinary native review (`benchmark_results/blake3-linux-reader-20261008T074000Z/cleanup-review/arm64-review/report.md`)
retains emitted clears and fences for this owner and the input allocation.
The recursive return copy, caller-side CV argument and recursive plain arrays
remain outside that scope. This does not close the separate Portable-state,
outer-key, native-fallback or compiler-spill findings.

The BLAKE3 SIMD128 backend clears its named vector, tail, parent, and output scratch in keyed
and derive-key modes. Its plain tiny-input specialization adds no secret route. The
[WASM backend record](../benchmark_results/OVERVIEW.md#2026-10-06-blake3-wasm-four-lane-backend)
retains source-bound cleanup evidence and the separate pre-existing Portable state-owner gap.
The parent caller review (`benchmark_results/blake3-wasm-parent-cost-20261006T063400Z/secret-review/key-borrow-follow-up.md`)
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
| Poly1305 | The shared arithmetic owner clears its clamped key, accumulator, and additive pad on drop. The standalone authenticator also clears its partial-block buffer. AEAD framing and accelerated scratch retain their own cleanup. |
| HMAC, HKDF, KMAC, PBKDF2, and keyed BLAKE2/BLAKE3 | Finalization copies, keyed prefixes, work buffers, emitted blocks, and replaced state are cleared after their last use. |
| ECDSA, P-256 ECDH, P-384 ECDH, ML-KEM, and RSA private work | Secret scalars, digests, limbs, encoded messages, inverse state, and initialized scratch are cleared on every return path. See the notes below. |
| Ed25519 and X25519 private work | Secret owners and staging are cleared; arithmetic temporaries have a narrower boundary. See [Ed25519 and X25519](#ed25519-and-x25519). |
| ML-DSA private work | See [ML-DSA](#ml-dsa). |
| Argon2 and scrypt | See [Argon2 and scrypt](#argon2-and-scrypt). |
| Secret parsing and generation | RAII owners cover success, parse failure, entropy failure, and early return. |
| Caller-filled secret owners, P-256 and P-384 ECDH generation, and ECDSA blinding | See [Caller-filled owners](#caller-filled-owners). |

### ML-KEM

- A prepared decapsulation key keeps a decryption mask derived from the implicit-rejection secret.
  Both of its polynomials are cleared when the prepared key drops.
- The secret SHA-3 and SHAKE calls (G, J, and the PRF) run in an out-of-line worker.
  After the worker returns, a fixed volatile scrub clears its dead stack: 2 KiB,
  or 4 KiB after the four-lane PRF.
  `just stack-frames` checks that each linked worker fits inside its scrub on the reviewed targets.
- By-value key generation, import, and preparation leave moved-from copies in the caller's frame.
  The `*_in` constructors fill a box that the caller allocated, in place.
  In the measured RV32 and Cortex-M runs, they leave no copies.

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

### ML-DSA

- Expanded keys, transformed polynomials, seeds, randomness, message representatives, SHAKE state,
  rejected candidates, and Serde staging use zeroizing owners.
- Ordinary decode and message hashing fill existing owners in place.
- Private SHAKE256 finalization and squeezing use a local zeroizing reader.
  The reader is initialized before the absorbed state is copied into it.
  It is dropped before control returns to the sampler or the hashing caller.
- Secret-noise acceptance masks and counts share the wiped bit-plane owners.
- By-value key constructors leave moved-from copies in the caller's frame.
  The `*_in` constructors, including PKCS #8 import, fill a box that the caller allocated, in place.
- PKCS #8 import borrows the caller's DER and never clears it.
  PKCS #8 export writes into a fixed-size buffer that the caller owns and clears.
  A rejected import clears every owner it built before it returns.
  In the measured RV32 and Cortex-M runs, they leave no copies.
- Preparation decodes directly into storage that the caller owns, without moving the prepared owner.
  The handle clears it on drop and when preparation fails.
  The storage clears it again on drop.
- Full target qualification is still open.
  See [ML-DSA](mldsa.md).

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
