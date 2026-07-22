# Secret Lifecycle And Diagnostic Redaction

This document records where rscrypto-owned secret storage is cleared and how
that source-level policy is checked after release optimization. The type and
capability boundary is defined in
[`secret-ownership.md`](secret-ownership.md); this document follows those
owners through success, failure, early return, transfer, reuse, and drop.

## Claim Boundary

The lifecycle claim covers named owner storage and explicit source-level
temporaries: inline arrays and words, initialized `MaybeUninit` regions,
heap-backed vectors and boxes, reusable scratch, finalized hash snapshots, and
parser or generator staging buffers. Cleanup uses volatile writes plus a
compiler fence.

The claim does not extend to caller-owned input, a secret after an explicit
export into ordinary bytes, compiler-created register or spill copies, swapped
pages, crash dumps, or hardware-backed storage. Protocol-visible tags, keyed
outputs, signatures, public keys, nonces, and ciphertexts are not confidential
owners merely because they are produced by secret-bearing operations.

## Source Ownership And Cleanup

| Owner or flow | Retained secret state | Cleanup boundary |
|---|---|---|
| `SecretBytes`, `SecretVec`, typed keys, private keys, and shared secrets | Fixed or variable-length key material | Concrete `Drop`; consuming export either clears the source allocation or explicitly transfers responsibility to the returned ordinary bytes |
| AEAD contexts and AES backend schedules | Expanded encryption keys and authentication subkeys | Context and nested schedule `Drop`; operation-local subkeys and authentication state are cleared after use; failed open and private-output paths clear rejected output |
| HMAC-SHA-2 | Live SHA state, keyed inner/outer prefixes, oversized-key digests, and inner-digest finalization snapshots | Secret-specific SHA finalization clears copied state and padding blocks; reset clears the replaced live state; `Drop` clears the live state and both saved prefixes |
| HKDF and PBKDF2 | PRK or password-derived HMAC prefix words and derivation scratch | Prefix-owner `Drop`; oversized-key/password digests and per-block working values are cleared on every return path |
| Ed25519 signing | Expanded scalar, nonce prefix, nonce hash state, digest, and scalar intermediates | Expanded-secret `Drop`; secret-specific SHA-512 digest/finalization clears hash state and padding snapshots; signing clears scalar and digest temporaries before return |
| HMAC-SHA-3 and KMAC | Keyed Keccak state, initial snapshots, and finalized inner state | Secret-mode Keccak owners clear on replacement and `Drop`; fixed-output finalized sponge copies and inner digests are cleared after absorption |
| Keyed BLAKE2 | Stored key, chaining state, block buffer, and finalization copies | Core and parameter-owner `Drop`; finalized chaining words and partial-block copies are cleared before return |
| Keyed and derive-key BLAKE3 | Key words, chunk/output/root state, CV stack, XOF root, and per-state, reduction, or thread-local parallel CV vectors | Conditional nested `Drop` follows the mode flags; emitted output blocks are cleared after copying; reusable and reduction vectors are cleared after their last keyed use, before logical `Vec::clear`, and on owner drop |
| ECDSA, X25519, ML-KEM, and RSA private operations | Secret scalars, decapsulation arithmetic, private components, blinding values, limbs, encoded-message buffers, and reusable private scratch | Typed and nested owner `Drop`; operation wrappers clear local arrays and initialized heap regions; rejected private outputs are cleared |
| Argon2 and scrypt | Password/pepper-derived matrix or ROMix working set | Owning matrix/state `Drop` clears every initialized block; allocation and parameter failures retain the same RAII boundary |

Backend call records and fixed-size array arguments may be copied into registers
or ABI spill slots by the compiler. Their durable source owners and explicit
stack/heap temporaries are covered above; the compiler-created copies remain an
explicit limitation rather than an unprovable erasure claim.

## Path Audit

| Path | Source audit result | Optimized evidence |
|---|---|---|
| Success | Finalized HMAC/Keccak/BLAKE copies, oversized-key digests, emitted BLAKE3 blocks, parser staging, and private-operation scratch are cleared after the last read | Fixed stack, secret hex success, HMAC-SHA-2/SHA-3 finalization, and keyed BLAKE3 wrappers retain volatile zero stores |
| Error | `ZeroizingBytes` and RAII owners cover parser/generator failure; AEAD and RSA clear rejected plaintext/private output | The secret hex error wrapper reaches the same audited parser as success, and its release IR contains both cleanup paths |
| Early return | Scope-owned fixed and heap secrets retain `Drop` cleanup across `return` and `?`; explicit cleanup precedes returns from manual scratch paths | `diag_zeroize_early_return` retains zero stores in release MIR, LLVM IR, and assembly |
| Move or transfer | `SecretBytes::expose` clears its source before returning ordinary bytes; `SecretVec::into_unprotected_vec` transfers the allocation and responsibility; keyed XOF moves transfer one root owner whose destination clears on drop | Fixed-owner move and keyed BLAKE3 XOF move/consume wrappers retain source and destination cleanup |
| Reuse | HMAC clears replaced live SHA state; secret-mode Keccak assignment drops the replaced state; BLAKE3 replacement drops the old owner; BLAKE3 parallel vectors are wiped before reuse | The BLAKE3 reset wrapper contains separate production `Drop` calls for the replaced and final owners |
| Drop | Every confidential public owner in the ownership inventory reaches a concrete or nested cleanup implementation; heap owners traverse initialized storage before deallocation | The gate follows the keyed BLAKE3 wrapper into its production `drop_in_place` and requires retained owner and heap-scratch zero stores |

## Release-Binary Inspection

Run:

```text
just check-zeroize-evidence
```

[`scripts/check/zeroize-evidence.sh`](../scripts/check/zeroize-evidence.sh)
builds an optimized, single-codegen-unit library with the diagnostic entry
points needed to make each lifecycle shape observable. It requires every entry
point in release MIR, LLVM IR, and assembly, then checks volatile LLVM zero
stores and host-architecture zero-store instructions.

The gate maps evidence to production behavior as follows:

| Evidence entry point | Boundary made observable |
|---|---|
| `diag_zeroize_fixed_stack`, `diag_zeroize_variable_heap` | Inline and heap owner drop |
| `diag_zeroize_fixed_move`, `diag_zeroize_early_return` | Ownership transfer and early return |
| `diag_zeroize_hex_success`, `diag_zeroize_hex_error` | Shared secret-parser success and error cleanup |
| `diag_zeroize_hmac_sha256_finalize`, `diag_zeroize_hmac_sha3_finalize` | SHA-2 and Keccak keyed finalization, temporary cleanup, and owner drop |
| `diag_zeroize_blake3_drop`, `diag_zeroize_blake3_reuse` | Production keyed owner drop and replaced-state cleanup |
| `diag_zeroize_blake3_xof_move`, `diag_zeroize_blake3_xof_consume` | Keyed XOF ownership transfer and destination drop |
| `diag_zeroize_blake3_thread_scratch`, `diag_zeroize_blake3_parallel_scratch` | Thread-local and per-state heap CV wipe before reuse or deallocation |

This is host-binary evidence, not a universal machine-code proof. The gate must
run on each target whose generated cleanup is being claimed; unsupported
architectures retain the source audit only.

## Formatting And Error Audit

[`tests/secret_redaction.rs`](../tests/secret_redaction.rs) holds exact `Debug`
and error snapshots for generic secret wrappers, AEAD keys and contexts,
HMAC/HKDF/KMAC/PBKDF2 state, keyed BLAKE2/BLAKE3 state and XOF readers,
ECDSA/Ed25519 keypairs, X25519 and ML-KEM shared secrets, prepared ML-KEM
decapsulation state, Argon2 context, and representative secret-input errors.
RSA private components, the public-key-only private-key view, private scratch,
`SecretVec`, and Ed25519 expanded state have adjacent unit snapshots where
their internal fixtures are available.

Secret-key hex errors retain the public offending-byte field for programmatic
inspection but omit that byte from both `Debug` and `Display`. Generic ECDSA
random-source errors omit the caller's payload from `Debug`, `Display`, and the
standard error-source chain; callers can still recover it by explicitly
matching the public `Random` variant. Other reviewed errors contain only
discriminants, public sizes, or opaque verification failures.

`DisplaySecret` is the deliberate exception: constructing it explicitly opts
into rendering borrowed secret bytes. Feature-gated diagnostic functions may
return their declared result bytes, but they do not implicitly format owning
keys, seeds, nonce material, intermediate state, or unmasked shares through
`Debug` or an error value.
