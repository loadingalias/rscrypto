# Trust Profile

`rscrypto` is a pure Rust cryptographic primitives crate. The trust argument is
the code, the tests, the constant-time policy, the published benchmark data,
and the explicit limits below.

Start here for the short map. Follow the links for exact boundaries.

## What This Crate Is

`rscrypto` provides primitive cryptography: hashes, MACs, KDFs, password
hashing, AEADs, signatures, key exchange, ML-KEM, RSA, and checksums behind one
feature model.

The portable Rust implementation is always present. SIMD and assembly paths are
accelerators, and they are tested against the portable path.

## What It Tries To Protect

- Cryptographic correctness for the supported primitive APIs.
- Secret-dependent timing behavior for the release-claimed constant-time
  primitive/configuration pairs.
- ML-KEM key generation, encapsulation, decapsulation, and implicit-rejection
  behavior for the FIPS 203 parameter sets exposed by the public API.
- Authentication failure shape for MACs, AEAD open, signatures, password
  checks, and RSA private operations where the API claims an opaque failure.
- Secret material in public formatting and normal drop paths.
- Backend drift between portable and accelerated implementations.

## Constant-Time Claims

Constant-time behavior is a scoped release claim. The exact claim is the set of
primitive/configuration pairs marked `ct_claimed` in [`ct.toml`](../ct.toml).

[`docs/constant-time.md`](constant-time.md) defines the threat model, what may
leak, what must not leak, target scope, and when evidence needs to be renewed.

Unlisted targets, feature sets, compilers, linkers, CPU features, or primitive
paths are not covered by a release constant-time claim.

## Correctness Evidence

The coverage ledger is [`docs/test-vector-coverage.md`](test-vector-coverage.md).
It points to the tests and test data used for:

- Official standards vectors.
- NIST ACVP FIPS 203 ML-KEM keyGen, encapsulation, decapsulation,
  decapsulationKeyCheck, and encapsulationKeyCheck vectors.
- Wycheproof vectors where they map to the public API.
- Differential or oracle tests against established crates.
- Invalid input, malformed encoding, tamper, and failure-shape tests.
- Fuzz corpus replay and Miri coverage where applicable.

## Platform Evidence

The platform and dispatch model is [`docs/platforms.md`](platforms.md).

In short: portable Rust is the correctness reference. Runtime acceleration is
used only when the target and CPU support it. `portable-only` forces runtime
dispatch toward portable backends, but it is not a proof and does not remove
compile-time target-feature code from the binary.

## Release And Benchmark Evidence

Constant-time release evidence is produced by the
[`ct.yaml`](https://github.com/loadingalias/rscrypto/actions/workflows/ct.yaml)
workflow for release commits and checked against [`ct.toml`](../ct.toml).

Current benchmark claims should match
[`benchmark_results/OVERVIEW.md`](../benchmark_results/OVERVIEW.md). The
benchmark method is described in [`docs/benchmarking.md`](benchmarking.md).
ML-KEM has both end-to-end benchmark rows and phase selectors for matrix
sampling, arithmetic, PKE, and decapsulation attribution.

Performance evidence is not security evidence. It only describes measured speed
for the listed platform, commit, feature set, and comparison shape.

## Non-Claims

`rscrypto` is not:

- A TLS stack.
- A PKI toolkit.
- A key store.
- A protocol implementation.
- A FIPS 140-3 validated module.
- Third-party audited today.
- Formally verified today.
- A whole-crate constant-time claim.

For compliance wording, use [`docs/compliance.md`](compliance.md). Say
"FIPS-oriented building blocks" only when the context needs that distinction.
Do not say "FIPS validated" unless a real validation exists.

## Vulnerability Reporting

Report real vulnerabilities through
[GitHub Private Vulnerability Reporting](https://github.com/loadingalias/rscrypto/security/advisories/new)
or the process in [`SECURITY.md`](../SECURITY.md). Do not report real-world
vulnerabilities through public GitHub issues.
