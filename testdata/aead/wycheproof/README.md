# AEAD Wycheproof vectors

These five JSON documents come from [`C2SP/wycheproof`](https://github.com/C2SP/wycheproof)
commit `b61843a9a5115bb758134b6a1f5d5e502d445342`, directory `testvectors_v1/`.
They were imported into rscrypto on 2026-06-04 in commit `d422eb47`.
The upstream repository licenses them under Apache-2.0.

The JSON payloads match upstream after key-order and whitespace normalization; no test cases were changed.
`SHA256SUMS` is the complete five-file payload inventory and records the checked-in byte representation.
Verify it from this directory with `shasum -a 256 -c SHA256SUMS`.

`tests/aead_wycheproof.rs` consumes the corpus. `docs/test-vector-coverage.md` owns its coverage claims.
