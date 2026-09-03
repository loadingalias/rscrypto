# Authentication Wycheproof vectors

These thirteen JSON files are derived from
[`C2SP/wycheproof`](https://github.com/C2SP/wycheproof) commit
`b61843a9a5115bb758134b6a1f5d5e502d445342`, directory
`testvectors_v1/`. The upstream repository licenses them under Apache-2.0.

Run `scripts/lib/python.sh scripts/check/auth-vector-provenance.py` to verify
the committed file set and SHA-256 digests. Pass `--upstream-root PATH` to
additionally require an exact checkout of that commit and compare every local
JSON document with its upstream source semantically. The P-256 ECDH corpus is
byte-for-byte identical; older corpora retain their established formatting
transforms.

The corpus is test evidence, not a generated rscrypto artifact. Updating it
requires a new full upstream commit, reviewed digest changes, and corresponding
test-coverage review.
