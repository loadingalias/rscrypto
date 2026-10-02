# Compliance

`rscrypto` is not a FIPS 140-3 validated cryptographic module.
A product that depends on it does not become compliant because of it.

NIST validates defined cryptographic modules, not single algorithm implementations.
An approved algorithm, or a passing algorithm test, is not a module validation.
To confirm that a module is validated, search the [CMVP validated modules database](https://csrc.nist.gov/projects/cryptographic-module-validation-program/validated-modules).

## What rscrypto supplies

`rscrypto` supplies review evidence that can support a module or product that someone else defines:

- Primitive implementations based on published standards, with public test vectors.
- Differential, property, fuzz, Miri, and backend-equivalence tests.
- Constant-time and secret-lifecycle evidence, each with a stated scope.
- Explicit feature and platform contracts.

Start with [`test-vector-coverage.md`](test-vector-coverage.md), [`constant-time.md`](constant-time.md), and [`secret-lifecycle.md`](secret-lifecycle.md).

## What the integrator owns

The owner of the product or module must define and validate:

- The cryptographic boundary and the operational environments.
- The approved algorithms, modes, parameters, and protocol profiles.
- The behavior for entropy, keys, nonces, salts, counters, and error states.
- The required self-tests and known-answer tests.
- Build provenance, binary distribution, and change control.
- The evidence package for the lab, assessor, or customer.

`portable-only` can make runtime dispatch select portable backends.
It does not remove accelerated code, override compile-time target features, prove constant time,
or create a validation boundary.

Use accurate wording downstream:

```text
This product uses rscrypto, a Rust library of cryptographic primitives.
rscrypto is not a FIPS 140-3 validated module. Its public evidence includes
test vectors, differential tests, platform coverage, and scoped constant-time
analysis.
```

Do not describe `rscrypto` as FIPS validated, FIPS certified, approved, audited,
or a replacement for compliance work.

The [FIPS 140-3 standard](https://csrc.nist.gov/pubs/fips/140-3/final) and the [CMVP FIPS 140-3 program documents](https://csrc.nist.gov/projects/cryptographic-module-validation-program/fips-140-3-standards) define the current program
requirements.
