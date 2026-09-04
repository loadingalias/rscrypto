# Threat Model

`rscrypto` is a cryptographic primitives library. This document defines the
threats the crate addresses, the responsibilities its callers retain, and the
evidence required to evaluate its security claims. Use it to plan an
integration or security review.

Report vulnerabilities through [`SECURITY.md`](SECURITY.md).

## Boundary

The crate accepts caller-provided keys, passwords, messages, nonces, public
keys, signatures, ciphertexts, tags, and encoded records. Treat data received
from a peer as untrusted.

Protected assets are keys, passwords, shared and derived secrets, secret
intermediate state, unauthenticated plaintext, and the integrity of published
artifacts.

With `getrandom`, selected APIs obtain entropy from the operating system. With
`std`, runtime dispatch reads CPU and OS capability data. The `parallel`
feature uses Rayon for BLAKE3 and Argon2. The caller owns protocol composition,
certificate validation, key storage and rotation, nonce policy, transport, and
access control.

`rscrypto` owns:

- Primitive semantics, parameter validation, and documented input bounds.
- Secret-dependent computation inside the selected primitive.
- Runtime capability validation, backend selection, and portable fallback.
- Opaque secret-dependent verification failures after public shape checks.
- Cleanup of named secret owners and rejected private output within the
  documented lifecycle boundary.

## Caller contract

1. Select a primitive and profile that meet the protocol's security
   requirements. The crate does not make protocol composition secure.
2. Protect keys and passwords, provide adequate entropy, and enforce rotation
   and invocation limits.
3. Keep every required nonce unique for its key. Use an algorithm's documented
   misuse bounds; `NonceCounter` covers only its stated AES-GCM profile.
4. Reject failed authentication or verification. Public shape errors may be
   distinct; secret-dependent failures remain opaque.
5. Protect every exported, formatted, or serialized secret. After an explicit
   escape hatch returns ordinary bytes, their lifetime belongs to the caller.

## Threats in scope

### Hostile inputs

A remote peer may supply malformed or adversarial encoded keys, password
records, public keys, signatures, ciphertexts, tags, or protocol selectors.
Relevant failures include incorrect acceptance, memory unsafety, reachable
panics, resource use beyond documented bounds, and authentication
oracles. P-256 ECDH rejects malformed or non-canonical peer points before
private scalar arithmetic. Rejected AEAD plaintext and RSA private-operation
output are cleared by the APIs that receive those buffers.

### Timing observation

A co-located attacker may measure secret-bearing operations. Constant-time
behavior is claimed only for operations and configurations backed by the exact
release evidence required by [`ct.toml`](ct.toml) and
[`docs/constant-time.md`](docs/constant-time.md). `ct_intended` marks work
inside the evidence policy; it is not a release claim.

Within a claimed operation, public values may affect control flow, including
algorithms, lengths, parameters, parsing results, resource use, and backend
selection. One opaque authentication or verification result may also be
observable.

### Backend and dispatch faults

Accelerated Rust, SIMD, and assembly execute only after their compile-time or
runtime requirements are established. Portable Rust defines the result; tests
compare eligible accelerated backends with that implementation. Unsafe code,
intrinsics, assembly, capability detection, and dispatch remain inside the
security review boundary.

### Integration misuse

Typed keys, nonces, tags, policy objects, explicit `expert` modules,
`#[must_use]` results, and bounded helpers reduce common mistakes. They cannot
prevent a caller from choosing the wrong primitive, reusing exported secrets,
ignoring a result, or violating a protocol rule.

### Release substitution

An attacker may target source, dependencies, build hosts, or published
artifacts. The repository currently provides no automated release-integrity
binding between reviewed source and a published artifact. Treat artifact origin
and integrity as unverified unless they are established independently.

## Outside this model

- Physical side channels, including power, electromagnetic, acoustic, and
  fault-injection attacks.
- Speculative-execution attacks beyond the claimed control-flow and
  memory-address discipline.
- A compromised host, operating system, hypervisor, CPU capability report, or
  toolchain.
- The quality of entropy returned successfully by the operating system or a
  caller-supplied source.
- Downstream protocol design, certificate path validation, key custody,
  transport security, and access control.

## Assurance

| Property | Evidence |
| --- | --- |
| Primitive correctness and hostile-input handling | Official vectors, independent implementations, negative tests, fuzzing, Miri, and backend differential tests; see [`docs/test-vector-coverage.md`](docs/test-vector-coverage.md). |
| Constant-time behavior | The operation inventory, target policy, generated-code review, binary checks, and native timing evidence defined by [`ct.toml`](ct.toml) and [`docs/constant-time.md`](docs/constant-time.md). |
| Secret ownership and cleanup | The type inventory, source audit, optimized cleanup checks, and redaction tests in [`docs/secret-ownership.md`](docs/secret-ownership.md) and [`docs/secret-lifecycle.md`](docs/secret-lifecycle.md). |

Published assurance is scoped evidence, not a blanket whole-crate proof.
Source-level cleanup does not cover compiler-created register or spill copies,
swapped pages, or crash dumps. Miri covers portable paths, not native SIMD or
assembly backends. No third-party security audit or whole-crate formal proof is
claimed.

## Review priorities

1. Secret-dependent operations, comparisons, declassification, and failure
   paths listed in `ct.toml`.
2. Untrusted parsers, bounded-resource policies, RSA key import, and private
   padding checks.
3. Unsafe Rust, SIMD, assembly, target-feature gates, dispatch, and portable
   equivalence.
4. Secret construction, duplication, export, serialization, cleanup, and
   error paths.
5. Dependencies, build authority, release identity, and published artifacts.
