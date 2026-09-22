# RSA Wycheproof vectors

These 36 JSON documents come from [`C2SP/wycheproof`](https://github.com/C2SP/wycheproof)
commit `b61843a9a5115bb758134b6a1f5d5e502d445342`, directory `testvectors_v1/`.
They were imported into rscrypto beginning on 2026-05-22 in commit `e2996405`.
The upstream repository licenses them under Apache-2.0.

The JSON payloads match upstream after key-order and whitespace normalization;
no test cases were changed. `SHA256SUMS` is the complete 36-file payload
inventory and records the checked-in representation. Verify it from this
directory with `shasum -a 256 -c SHA256SUMS`.

`tests/rsa_wycheproof.rs` consumes the full mapped set and
`tests/rsa_leakage.rs` consumes one OAEP corpus for failure-class checks.
`docs/test-vector-coverage.md` owns the coverage limits.
