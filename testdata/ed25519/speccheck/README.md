# Ed25519 speccheck cases

`cases.json` is byte for byte the file of the same name from
[`novifinancial/ed25519-speccheck`](https://github.com/novifinancial/ed25519-speccheck)
commit `65519336fda78a3d016e947df6d82848aca0c9da`, fetched on 2026-11-13.
The upstream repository licenses it under Apache-2.0.
The cases come from "Taming the many EdDSAs" (Chalkias, Garillot, and Nikolaenko, SSR 2020;
<https://eprint.iacr.org/2020/1244>).
rscrypto code computed none of these bytes.

The twelve cases separate verifiers by these checks:
small-order A and R, mixed-order A and R,
S of at least L, non-canonical encodings of A and R,
and the cofactored and cofactorless equations.
Under the strict policy that `rscrypto::auth::ed25519::verify` documents, only case 3 verifies.
`ed25519-dalek` `verify_strict` gives the same verdicts.

`tests/ed25519_speccheck.rs` checks all twelve cases through the public API on each host's
production backend, and through the portable backend in the portable run.
The `speccheck_double_scalar_products_agree_on_every_backend` unit test runs cases 3, 4, and 5
through every double-scalar backend that the host can execute.

`SHA256SUMS` is the complete one-file payload inventory.
Verify it from this directory with `shasum -a 256 -c SHA256SUMS`.
