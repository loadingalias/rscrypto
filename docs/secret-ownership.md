# Secret Ownership And Generic Capabilities

This inventory identifies named types that retain secret material or
secret-derived state, then records where generic capabilities can duplicate,
format, serialize, or allocate that state. It is the review baseline for
secret-lifetime work; it is not proof that every compiler-created copy is
erased. See [`constant-time.md`](constant-time.md) for timing claims and the
[`THREAT_MODEL`](../THREAT_MODEL.md) for the system boundary.

The inventory treats keys, private-key state, shared secrets, password/pepper
state, MAC/KDF state, key schedules, and explicitly keyed hash/XOF state as
secret-bearing. Authentication tags and keyed outputs are included because
their concrete owner controls comparison, but they are protocol-visible values
rather than confidential key material.

Ordinary unkeyed digest/XOF states, public keys, signatures, nonces,
ciphertexts, PHC records, password-hash parameters, and public RSA scratch are
not semantic secret owners. They can still receive sensitive caller data.
Operation-local arrays, scalars, limbs, plaintext buffers, and individual
backend transfer records are grouped here only where their type-level
`Clone`/`Copy` capability affects ownership. Their complete data flow belongs
to the temporary-flow review.

In the tables below, “explicit duplicate” means a method named
`duplicate_secret()` rather than `Clone`. “Masked” means `Debug` can identify
the type or show public metadata but does not print the owned secret bytes.
The use column explains why a capability exists; it does not pre-approve that
capability for permanent retention.

## Confidential Public Owners And Views

| Type or family | Clone / Copy | Debug | Serialization or export | Storage | Capability use |
|---|---|---|---|---|---|
| `SecretBytes<N>` | Neither | Masked | Consuming `expose()` returns a plain array | Inline | Fixed-size integration boundary whose owned source is cleared on extraction |
| `SecretVec` | Neither | Masked | Consuming `into_unprotected_vec()` returns an ordinary allocation | Heap `Vec<u8>` | Variable-size RSA private-key export and explicit transfer to APIs that cannot borrow |
| `DisplaySecret<'a>` | Neither | Intentionally prints bytes through both `Display` and `Debug` | Hex formatting only | Borrowed | Explicit opt-in escape hatch for integrations that must render a key; never use it in logs |
| AEAD `*Key` types | Explicit duplicate; no `Clone` or `Copy` | Masked | `SecretBytes` export, hex opt-in, and `serde-secrets` | Inline | Lets an owned cipher context coexist with a caller-retained typed key while making the extra lifetime visible |
| AEAD cipher contexts | Neither | Masked | None | Inline, except boxed RISC-V fixslice AES schedules | Reusable expanded key and authentication subkey state without exposing a generic duplication path |
| ECDSA P-256/P-384 secret keys and keypairs | Explicit duplicate; no `Clone` or `Copy` | Secret keys are masked; keypairs show only the public half | `SecretBytes` export and hex opt-in; no Serde | Inline | Caller-controlled key/keypair duplication for independent signing owners |
| Ed25519 secret key and keypair | Explicit duplicate; no `Clone` or `Copy` | Secret key is masked; keypair shows only the public half | Secret-key `SecretBytes` export, hex opt-in, and `serde-secrets`; no keypair Serde | Inline | Independent signing owners; keypair duplication also copies its expanded secret state deliberately |
| X25519 secret key and shared secret | Explicit duplicate; no `Clone` or `Copy` | Masked | `SecretBytes` export, hex opt-in, and `serde-secrets` | Inline | Explicit transfer of key-agreement material into a separately owned protocol or KDF context |
| ML-KEM decapsulation keys and shared secrets | Explicit duplicate; no `Clone` or `Copy` | Masked | `SecretBytes` export, hex opt-in, and `serde-secrets` | Inline | Explicit transfer of decapsulation or established key material into another owner |
| ML-KEM prepared decapsulation keys | Explicit duplicate; no `Clone` or `Copy` | Masked | `SecretBytes` export; no Serde | Inline | Reuse of validated private arithmetic without making implicit copies |
| `RsaPrivateKey` | Neither | Public key plus a redacted private-components field | PKCS#1/PKCS#8 DER into `SecretVec`; no Serde | Heap-backed big integers and Montgomery state | Standards-compatible private-key storage/export while keeping the returned allocation typed as secret |
| `RsaPrivateScratch` | Neither | Public sizing metadata only | None | Reusable heap buffers and limb vectors | Amortizes private-operation allocation while keeping intermediate ownership with the caller |
| `RsaPrivateKeyParts<'a>` | `Clone + Copy` | No `Debug` | Borrowed import fields; no Serde | Borrowed | Pass-by-value import description; copying duplicates references, not private bytes |
| `RsaSignatureSigner<'a>` | `Clone + Copy` | Signature profile only | None | Borrowed | Reusable profile-bound signing handle; copying duplicates a private-key reference, not the key |
| HMAC-SHA-2 and HMAC-SHA-3 states | Neither | Masked | None | Inline | Reuse through `reset`; independent streamed owners require explicit keyed construction rather than an implicit state copy |
| HKDF-SHA-2 states | Neither | Masked | None | Inline | Repeated expansion borrows one extracted PRK owner, which can also be shared by reference |
| KMAC128/256 states | Neither | Masked | None | Inline | Reuse through `reset`; private cSHAKE snapshots implement non-consuming finalization and reset without exposing a public keyed-state copy |
| PBKDF2-SHA-2 states | Neither | Masked | None | Inline | Repeated derivation borrows one password-prefix owner, which can also be shared by reference |
| `Poly1305OneTimeKey` and `Poly1305` | Neither | Masked | `SecretBytes` key export; no Serde | Inline | Enforces one-time key consumption and consuming finalization |
| `Argon2Context<'a>` | `Clone + Copy` | Secret bytes redacted; secret and associated-data lengths shown | None | Borrowed | Pass-by-value optional pepper/associated-data configuration without duplicating either byte string |
| BLAKE2 keyed parameter builders and variable-output state | Neither | Key bytes masked; builders show key length, salt, and personalization | None | Inline | Reuse by borrowing; independent keyed owners require explicit construction |
| Fixed-output BLAKE2 states | `Clone` | Masked | None | Inline | The same type serves unkeyed `Digest` and keyed modes, so the public `Digest: Clone` contract also permits keyed prefix forks |
| `Blake3` | `Clone` | Masked | None | Inline, plus `Vec` scratch with `parallel` | Fork a streamed common prefix in keyed or derive-key mode; cloning also duplicates initialized parallel scratch |
| `Blake3XofReader` | `Clone` | Masked | None | Inline | Checkpoint or fork an output cursor; the reader is secret-bearing when created from keyed or derive-key state |

## Protocol-Visible Authentication Owners

These owners provide constant-work comparison semantics, but their bytes are
normally transmitted or stored with the protected message. Copying, rendering,
and ordinary serialization therefore do not duplicate a confidential key.

| Type or family | Clone / Copy | Debug | Serialization | Storage | Capability use |
|---|---|---|---|---|---|
| AEAD tags | `Clone + Copy` | Raw hex | `serde` | Inline | Detached-tag wire formats and cheap by-value API use |
| HMAC-SHA-2 and HMAC-SHA-3 tags | `Clone + Copy` | Raw hex | `serde` | Inline | Protocol transport plus sealed `ct_eq` verification |
| `Poly1305Tag` | `Clone + Copy` | Raw hex | None | Inline | Detached one-time authenticator transport and verification |
| `Blake3KeyedHash` | `Clone + Copy` | Raw hex | None | Inline | Protocol-visible keyed output with sealed `ct_eq` verification |

## Internal And Operation-Scoped Owners

| Owner | Clone / Copy | Debug / serialization | Storage | Capability use |
|---|---|---|---|---|
| `ZeroizingBytes<N>` | Neither | Neither | Inline | Generation and parsing scratch that cannot escape as a generic clone |
| AES expanded schedules | Neither | Neither | Inline; boxed only for the large RISC-V fixslice schedule | Retained by an AEAD context and borrowed by block operations; unused private `Clone` derives were removed during this inventory |
| AEAD authentication working state | Private copies only where a backend finalizer consumes a value | Neither | Inline | Bound, intra-operation snapshot needed by consuming backend finalization |
| HMAC-SHA-3 and KMAC Keccak/cSHAKE snapshots | Private use of `Clone` | Neither | Inline | Implement non-consuming finalization and reset inside one public keyed owner |
| Ed25519 `ExpandedSecret` | Private `Clone` | Masked; no serialization | Inline | Implements the public keypair's explicit `duplicate_secret()` operation |
| X25519 clamped scalar and ECDSA secret scalar/word wrappers | No generic duplication on the owning wrappers | Neither | Inline | Bound one-operation arithmetic ownership |
| ML-KEM prepared decapsulation arithmetic | Private `Clone` | Neither | Inline | Implements the prepared key's explicit `duplicate_secret()` operation |
| RSA private components, buffers, limbs, and key-generation DRBG | Neither | Private integers are masked; no serialization | Heap for variable-width integers/scratch; DRBG inline | Variable-width private arithmetic, reusable scratch, and generated-key construction |
| Argon2 matrix | `MemoryBlock` and borrowed `MatrixView` are `Clone + Copy`; `Matrix` is not | Neither | Heap `Vec<MemoryBlock>` | Algorithm-defined whole-block mixing and disjoint parallel lane views; copying the view duplicates only a pointer/length pair |
| scrypt working state | `SalsaBlock` is `Clone + Copy`; owning states are not | Neither | Heap vectors | Algorithm-defined whole-block ROMix copies while one owner retains the complete work area |
| BLAKE3 parallel scratch | Cloned with `Blake3` | Neither | Per-state vectors plus thread-local vectors | Avoid repeated allocation during parallel subtree reduction; keyed modes make stored chaining values secret-derived |
| Private-key and AEAD backend transfer records | Private `Clone`/`Copy` only where passed by value or snapshotted by a consuming kernel | Neither | Inline | Fixed-layout, call-scoped handoff to portable, SIMD, or assembly code; no public capability |

## Review Consequences

- `serde` alone covers protocol-visible values. `serde-secrets` is the explicit
  opt-in for AEAD keys, Ed25519/X25519 secrets, and ML-KEM decapsulation/shared
  secrets. ECDSA and RSA private keys do not implement Serde.
- `DisplaySecret` is the only generic formatting wrapper that intentionally
  makes secret bytes visible through `Debug`; constructing it is the disclosure
  decision.
- Heap-backed secret state is confined to `SecretVec`, RSA private-key import,
  generation, keys, and scratch, Argon2 and scrypt work areas, BLAKE3 parallel
  scratch, and the RISC-V fixslice AES schedule. Caller-owned plaintext and
  derived-output buffers are outside this named-owner inventory.
- HMAC, HKDF, KMAC, PBKDF2, keyed BLAKE2 parameter builders, and variable-output
  BLAKE2 state do not expose `Clone`; `Mac` does not require it. Their reusable
  operations borrow or reset one owner, and independent keyed state must be
  constructed explicitly.
- Fixed-output BLAKE2 and BLAKE3 hash/XOF states retain `Clone` because the same
  types serve ordinary unkeyed `Digest`/`Xof` use. A clone made in keyed or
  derive-key mode is another secret lifetime even though the type cannot encode
  that distinction.
- Private-key owners remain non-`Clone`. Borrowed RSA views copy references
  only, while the private clones behind Ed25519 and ML-KEM
  `duplicate_secret()` make the requested owned duplication explicit at the
  public call site.
- Wiping success, failure, early-return, move, reuse, and drop paths—including
  BLAKE3 XOF and reusable heap scratch—is mapped to source and optimized-binary
  evidence in [`secret-lifecycle.md`](secret-lifecycle.md).

When this surface changes, review the public re-exports and the implementations
under [`src/secret.rs`](../src/secret.rs), [`src/hex.rs`](../src/hex.rs),
[`src/aead`](../src/aead), [`src/auth`](../src/auth), and
[`src/hashes/crypto`](../src/hashes/crypto). Search derives and manual trait
implementations for `Clone`, `Copy`, and `Debug`; search Serde macros and
implementations under both feature gates; and search owned `Vec`/`Box` fields.
Every new capability needs a concrete use in this document or must be removed.
