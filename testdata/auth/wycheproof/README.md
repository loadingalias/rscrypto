# Authentication Wycheproof vectors

These fourteen JSON files are derived from [`C2SP/wycheproof`](https://github.com/C2SP/wycheproof) commit
`b61843a9a5115bb758134b6a1f5d5e502d445342`, directory `testvectors_v1/`.
They were imported into rscrypto on 2026-06-04 in commit `d422eb47`, except the P-384 ECDH corpus,
which was imported from the same upstream commit on 2026-09-25.
The upstream repository licenses them under Apache-2.0.

The P-256 and P-384 ECDH corpora are byte-for-byte identical to their upstream sources;
older corpora retain their established formatting transforms.

The corpus is test evidence, not a generated rscrypto artifact.
Updating it requires a new full upstream commit, reviewed source changes,
and corresponding test-coverage review.

`SHA256SUMS` is the complete fourteen-file payload inventory and records the checked-in representation.
Verify it from this directory with `shasum -a 256 -c SHA256SUMS`.
The `*_wycheproof.rs` integration tests, `tests/p256_ecdh_oracle.rs`, and `tests/p384_ecdh_oracle.rs` are its consumers.
