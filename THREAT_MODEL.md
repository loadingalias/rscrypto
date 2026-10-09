# Threat Model

`rscrypto` is a library of cryptographic primitives.
This document defines the threats that the crate addresses, the responsibilities that callers keep,
and the evidence needed to evaluate its security claims.
Use it to plan an integration or a security review.

Report vulnerabilities through [`SECURITY.md`](SECURITY.md).

## Boundary

The crate accepts keys, passwords, messages, nonces, public keys, signatures, ciphertexts, tags,
and encoded records from the caller.
Treat data from a peer as untrusted.

The protected assets are:

- keys and passwords;
- shared and derived secrets;
- secret intermediate state;
- unauthenticated plaintext;
- the integrity of published artifacts.

The crate touches the system in these ways only:

- With `getrandom`, some APIs get entropy from the operating system.
- With `std`, runtime dispatch reads CPU and operating-system capability data.
- The `parallel` feature uses Rayon for BLAKE3 and Argon2.

The caller owns protocol composition, certificate validation, key storage and rotation,
nonce policy, transport, and access control.

`rscrypto` owns:

- primitive behavior, parameter validation, and the documented input bounds;
- secret-dependent computation inside the selected primitive;
- runtime capability validation, backend selection, and portable fallback;
- opaque verification failures for secret-dependent checks, after the public shape checks;
- cleanup of named secret owners and of rejected private output,
  within the documented lifecycle boundary.

## Caller contract

1. Select a primitive and a profile that meet the security requirements of your protocol.
   The crate does not make a protocol composition secure.
1. Protect keys and passwords, supply adequate entropy, and enforce rotation and invocation limits.
1. Keep each required nonce unique for its key.
   Stay within each algorithm's documented misuse bounds.
   `NonceCounter` covers only its stated AES-GCM profile.
1. Reject every failed authentication or verification.
   Public shape errors can be distinct.
   Secret-dependent failures stay opaque.
1. Protect every secret that you export, format, or serialize.
   After an explicit escape hatch returns ordinary bytes, you own their lifetime.

## Cryptographic assumptions and migration

Primitive security depends on the assumptions, parameters, and usage bounds of its construction.
Implementation tests, timing evidence, and formal proofs do not establish that these assumptions
will withstand every future mathematical or quantum attack.
The [cryptographic resilience guide](docs/cryptographic-resilience.md) explains how to distinguish
algorithm families, authentication, confidentiality, and implementation evidence.
The caller owns protocol migration, trust-anchor distribution, and downgrade policy.
Planned algorithms and research results do not extend the crate's implemented security boundary.

## Threats in scope

### Hostile inputs

A remote peer can send malformed or adversarial encoded keys, password records, public keys,
signatures, ciphertexts, tags, or protocol selectors.
The relevant failures are: incorrect acceptance, memory unsafety, reachable panics,
resource use beyond the documented bounds, and authentication oracles.

- P-256 and P-384 ECDH reject malformed and non-canonical peer points
  before private scalar arithmetic.
- The APIs that receive rejected AEAD plaintext
  and RSA private-operation output clear those buffers.

### Timing observation

An attacker on the same machine can measure operations that handle secrets.
A constant-time claim exists only for operations and configurations
that have the exact release evidence that [`ct.toml`](ct.toml) and [`docs/constant-time.md`](docs/constant-time.md) require.
`claim = "ct-intended"` in `ct.toml` marks work that is inside the evidence policy.
It is not a release claim.

Within a claimed operation, public values can affect control flow.
These include algorithms, lengths, parameters, parsing results, resource use, and backend selection.
One opaque authentication or verification result can also be visible.

### Backend and dispatch faults

Accelerated Rust, SIMD, and assembly run only after their compile-time
or runtime requirements are met.
Portable Rust defines the result.
Tests compare each eligible accelerated backend with that implementation.
Unsafe code, intrinsics, assembly, capability detection,
and dispatch are inside the security review boundary.

### Integration misuse

Typed keys, nonces, and tags, policy objects, explicit `expert` modules, `#[must_use]` results,
and bounded helpers reduce common mistakes.
They cannot stop a caller from selecting the wrong primitive, reusing exported secrets,
ignoring a result, or breaking a protocol rule.

### Release substitution

An attacker can target the source, the dependencies, the build hosts, or the published artifacts.
Treat the origin and integrity of an artifact as unverified until you confirm them independently.
Assembly derived from upstream code is pinned to its upstream archive and member hashes
in `src/**/*_assembly_provenance.tsv` manifests, which `just check` verifies against the committed files.

## Outside this model

- Physical side channels: power, electromagnetic, acoustic, and fault-injection attacks.
- Speculative-execution attacks beyond the claimed control-flow and memory-address discipline.
- A compromised host, operating system, hypervisor, CPU capability report, or toolchain.
- The quality of entropy that the operating system or a caller-supplied source returns successfully.
- Downstream protocol design, certificate path validation, key custody, transport security,
  and access control.

## Assurance

| Property                                         | Evidence |
| ------------------------------------------------ | -------- |
| Primitive correctness and hostile-input handling | Official vectors, independent implementations, negative tests, fuzzing, Miri, and backend differential tests. See [`docs/test-vector-coverage.md`](docs/test-vector-coverage.md). |
| Constant-time behavior                           | The operation inventory, target policy, generated-code review, binary checks, and native timing evidence that [`ct.toml`](ct.toml) and [`docs/constant-time.md`](docs/constant-time.md) define. |
| Secret ownership and cleanup                     | The type inventory, source audit, optimized cleanup checks, and redaction tests in [`docs/secret-ownership.md`](docs/secret-ownership.md) and [`docs/secret-lifecycle.md`](docs/secret-lifecycle.md). |

Published assurance is evidence with a stated scope.
It is not a proof for the whole crate.

- Source-level cleanup does not cover register or spill copies that the compiler makes,
  swapped pages, or crash dumps.
- Miri covers portable paths, not native SIMD or assembly backends.
- No third-party security audit and no formal proof of the whole crate is claimed.

## Review priorities

1. Secret-dependent operations, comparisons, declassification, and failure paths listed in `ct.toml`.
1. Parsers for untrusted input, bounded-resource policies, RSA key import,
   and private padding checks.
1. Unsafe Rust, SIMD, assembly, target-feature gates, dispatch, and portable equivalence.
1. Secret construction, duplication, export, serialization, cleanup, and error paths.
1. Dependencies, build authority, release identity, and published artifacts.
