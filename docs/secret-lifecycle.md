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
| ECDSA, Ed25519, X25519, P-256 ECDH, ML-KEM, and RSA private work | Secret scalars, digests, limbs, encoded messages, inverse state, and initialized scratch are cleared on every return path. Portable P-256 ECDH uses a type-distinct projective owner whose coordinates are cleared whenever an intermediate is replaced or the operation returns. Its selected AArch64 and x86-64 assembly clears secret-derived frames, saved-register spill slots, and volatile integer registers before return. The Windows public-point batch wrapper handles public coordinates only. |
| Argon2 and scrypt | Every initialized block in owned work memory is cleared before deallocation, including error paths. |
| Secret parsing and generation | RAII owners cover success, parse failure, entropy failure, and early return. |
| Caller-filled secret owners, P-256 ECDH generation, and ECDSA blinding | The zero-initialized owner exists before the callback runs. Success, immediate failure, partial-fill failure, and P-256 scalar rejection/exhaustion all reach its `Drop`. ECDSA callback failure returns before message hashing or private scalar arithmetic; P-256 ECDH callback failure returns before public derivation or agreement. |

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

Run:

```sh
just check-zeroize-evidence
```

The check builds optimized diagnostic entry points and verifies their presence
in release MIR, LLVM IR, and assembly. It then checks for volatile LLVM zero
stores and host-architecture zero-store instructions across these shapes:

- Fixed stack and variable heap owners.
- Move, early-return, fallible-fill, parse-success, and parse-error paths.
- UTF-8 owner destruction and ECDSA blinding success/partial-fill failure.
- HMAC, HKDF, ECDSA, P-256 ECDH, keyed BLAKE3, and ML-KEM state.
- AEAD authentication, header protection, and AES-SIV state.
- RSA success, entropy failure, and staged private-key validation.

Scheduled and release Qualification run the same check cache-cold on Linux
x86-64.

This evidence binds the generated host binary. Each target needs its own run;
source review remains the only evidence for an untested target.

`tests/secret_redaction.rs` pins public `Debug` and error behavior. Errors expose
only public sizes or opaque verification failures unless a documented variant
explicitly returns caller data. `expert::DisplaySecret` and diagnostic APIs are
deliberate declassification boundaries.

See [`secret-ownership.md`](secret-ownership.md) for the capability inventory.
