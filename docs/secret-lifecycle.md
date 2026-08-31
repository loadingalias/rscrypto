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
| ECDSA, Ed25519, X25519, ML-KEM, and RSA private work | Secret scalars, digests, limbs, encoded messages, inverse state, and initialized scratch are cleared on every return path. |
| Argon2 and scrypt | Every initialized block in owned work memory is cleared before deallocation, including error paths. |
| Secret parsing and generation | RAII owners cover success, parse failure, entropy failure, and early return. |

`SecretBytes::expose()` clears its source before returning an ordinary array.
`SecretVec::into_unprotected_vec()` transfers the existing allocation without
clearing it. That distinction is intentional.

## Optimized evidence

Run:

```sh
just check-zeroize-evidence
```

The check builds optimized diagnostic entry points and verifies their presence
in release MIR, LLVM IR, and assembly. It then checks for volatile LLVM zero
stores and host-architecture zero-store instructions across these shapes:

- Fixed stack and variable heap owners.
- Move, early-return, parse-success, and parse-error paths.
- HMAC, HKDF, ECDSA, keyed BLAKE3, and ML-KEM state.
- AEAD authentication, header protection, and AES-SIV state.
- RSA success, entropy failure, and staged private-key validation.

This evidence binds the generated host binary. Each target needs its own run;
source review remains the only evidence for an untested target.

`tests/secret_redaction.rs` pins public `Debug` and error behavior. Errors expose
only public sizes or opaque verification failures unless a documented variant
explicitly returns caller data. `expert::DisplaySecret` and diagnostic APIs are
deliberate declassification boundaries.

See [`secret-ownership.md`](secret-ownership.md) for the capability inventory.
