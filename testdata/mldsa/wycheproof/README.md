# ML-DSA Wycheproof vectors

These nine JSON files are byte-for-byte copies of [`C2SP/wycheproof`](https://github.com/C2SP/wycheproof) commit
`3fa63dd0344abb611f1fb1d77e119938603ea230`, directory `testvectors_v1/`, retrieved 2026-09-28.
The upstream repository licenses them under Apache-2.0.

Each parameter set has three corpora:

- `sign_seed`: signing with a key derived from a 32-byte seed.
- `sign_noseed`: signing with an imported expanded secret key, including
  secret vectors outside `[-eta, eta]` that import must reject.
- `verify`: verification, including norm violations, malformed hint encodings,
  modified signatures, zero public keys, oversized contexts, and wrong lengths.

The signing corpora cover deterministic and hedged signing, contexts,
rejection-loop boundary conditions, and cases that supply only the message representative `mu`.
The `wycheproof_mldsa*` unit tests in `src/auth/mldsa/tests.rs` consume every case and check the exact valid and invalid counts.
`SHA256SUMS` is the complete payload inventory; verify it from this directory with `shasum -a 256 -c SHA256SUMS`.

The corpus is test evidence, not a generated rscrypto artifact.
Updating it requires a new full upstream commit and a review of the changed cases.
