# Secret lifecycle

`rscrypto` clears its named secret owners and explicit secret temporaries on
success, failure, early return, reuse, and drop. Cleanup uses volatile writes
and a compiler fence.

This claim covers crate-owned arrays, initialized heap storage, reusable
scratch, parser and generator staging, finalized keyed-hash snapshots, and
expanded key state.

It does not cover caller-owned input, ordinary bytes after explicit export,
compiler-created register or spill copies, swapped pages, crash dumps, or
hardware-backed storage.

## Cleanup boundaries

| Owner or operation | Cleanup boundary |
| --- | --- |
| Typed keys, private keys, and shared secrets | Concrete or nested `Drop`; consuming export clears the source or transfers responsibility explicitly. |
| AEAD and header protection | Context drop clears retained keys; operation-local schedules, authentication state, and materialized cipher output are cleared after use. Failed opens clear unauthenticated plaintext. |
| HMAC, HKDF, KMAC, PBKDF2, and keyed BLAKE2/BLAKE3 | Finalization copies, keyed prefixes, work buffers, emitted blocks, and replaced state are cleared after their last use. |
| ECDSA, Ed25519, X25519, P-256 ECDH, P-384 ECDH, ML-KEM, and RSA private work | Secret scalars, digests, limbs, encoded messages, inverse state, and initialized scratch are cleared on every return path. Prepared ML-KEM decapsulation keys retain a decryption mask derived from the implicit-rejection secret; both of its polynomials are cleared when the prepared key drops. Portable P-256 ECDH uses a type-distinct projective owner whose coordinates are cleared whenever an intermediate is replaced or the operation returns. Its selected AArch64 and x86-64 assembly clears secret-derived frames, saved-register spill slots, and volatile integer registers before return. The Windows public-point batch wrapper handles public coordinates only. P-384 ECDH has no register-clearing assembly: its Jacobian accumulator owner is overwritten in place by each intermediate and cleared on return, and its scalar limbs, recoded window digits, and affine shared x-coordinate are cleared before return. Field-arithmetic and table-selection temporaries outside those owners, including the general-purpose and vector registers used by its AArch64 inline assembly and the stack frames and registers of its x86-64 fused point doubling and addition, are not claimed cleared. |
| ML-DSA private work | Expanded keys, transformed polynomials, seeds, randomness, message representatives, SHAKE state, rejected candidates, and Serde staging use zeroizing owners. Ordinary decode and message hashing fill existing owners in place. Private SHAKE256 finalization and squeezing use a local zeroizing reader, initialized before the absorbed state is copied into it; the reader is dropped before returning to the sampler or hashing caller. Secret-noise acceptance masks and counts share the wiped bit-plane owners. By-value key constructors leave moved-from copies in the caller's frame; the `*_in` constructors fill a caller-allocated box in place and leave none in the measured RV32 and Cortex-M runs. Preparation decodes directly into caller-owned storage without moving the prepared owner; the handle clears it on drop or failed preparation, and the storage clears it again on drop. Full target qualification remains open; see [ML-DSA](mldsa.md). |
| Argon2 and scrypt | Every initialized block in owned work memory is cleared before deallocation, including error paths. |
| Secret parsing and generation | RAII owners cover success, parse failure, entropy failure, and early return. |
| Caller-filled secret owners, P-256 and P-384 ECDH generation, and ECDSA blinding | The zero-initialized owner exists before the callback runs. Success, immediate failure, partial-fill failure, and P-256 or P-384 scalar rejection/exhaustion all reach its `Drop`. ECDSA callback failure returns before message hashing or private scalar arithmetic; ECDH callback failure returns before public derivation or agreement. |

`SecretBytes::expose()` clears its source before returning an ordinary array.
`SecretVec::into_unprotected_vec()` transfers the existing allocation without
clearing it. `SecretString::into_unprotected_string()` does the same for a UTF-8
allocation. That distinction is intentional.

`SecretVec` and `SecretString` clear every byte in their initialized length.
They do not claim to clear spare allocation capacity, which is not an
initialized region exposed by these owners. `SecretBytes<N>` always clears all
`N` bytes.

When panic unwinding is enabled, an owner already constructed around callback
storage is dropped during an unwind. Process abort, termination, and power loss
do not run destructors and carry no cleanup claim.

## Optimized evidence

`just check` and `just ci-check` do not verify optimized zeroization. Source
cleanup and passing tests alone do not establish that secret stores survive
optimization; machine-code evidence must be scoped to the compiler, target,
features, and operation inspected.

`just ct-full` additionally checks an 8-byte-aligned `SecretBytes<32>` destructor
sentinel in the existing linked release harness. It requires complete volatile
clearing and a compiler fence in the emitted release-LTO IR, and retains the
linked symbol, disassembly, and artifact hashes. Missing or partial cleanup
fails this gate. The sentinel does not qualify other alignments, owners, heap
storage, error paths, or compiler-created copies; the retained machine code
still needs target-specific review.

`tests/secret_redaction.rs` pins public `Debug` and error behavior. Errors expose
only public sizes or opaque verification failures unless a documented variant
explicitly returns caller data. `expert::DisplaySecret` and diagnostic APIs are
deliberate declassification boundaries.

See [`secret-ownership.md`](secret-ownership.md) for the capability inventory.
