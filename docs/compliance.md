# Compliance

`rscrypto` is not a FIPS 140-3 validated cryptographic module. Depending on it
does not make a product compliant.

NIST validates defined cryptographic modules, not isolated algorithm
implementations. Implementing an approved algorithm or passing algorithm tests
is not a module validation. Confirm required modules in the
[CMVP validated modules database](https://csrc.nist.gov/projects/cryptographic-module-validation-program/validated-modules).

## What rscrypto provides

The crate provides review evidence that may support a separately defined
module or product:

- Standards-based primitive implementations and public test vectors.
- Differential, property, fuzz, Miri, and backend-equivalence tests.
- Scoped constant-time and secret-lifecycle evidence.
- Explicit feature and platform contracts.

Start with [`test-vector-coverage.md`](test-vector-coverage.md),
[`constant-time.md`](constant-time.md), and
[`secret-lifecycle.md`](secret-lifecycle.md).

## What an integrator owns

The product or module owner must define and validate:

- The cryptographic boundary and operational environments.
- Approved algorithms, modes, parameters, and protocol profiles.
- Entropy, keys, nonces, salts, counters, and error-state behavior.
- Required self-tests and known-answer tests.
- Build provenance, binary distribution, and change control.
- The lab, assessor, or customer evidence package.

`portable-only` can make runtime dispatch choose portable backends. It does not
remove accelerated code, override compile-time target features, prove constant
time, or create a validation boundary.

Use accurate downstream wording:

```text
This product uses rscrypto, a pure Rust cryptographic primitives library.
rscrypto is not a FIPS 140-3 validated module. Its public evidence includes
test vectors, differential tests, platform coverage, and scoped constant-time
analysis.
```

Do not describe the crate as FIPS validated, FIPS certified, approved, audited,
or a compliance replacement.

Current program requirements live in the
[FIPS 140-3 standard](https://csrc.nist.gov/pubs/fips/140-3/final) and the
[CMVP FIPS 140-3 program documents](https://csrc.nist.gov/projects/cryptographic-module-validation-program/fips-140-3-standards).
