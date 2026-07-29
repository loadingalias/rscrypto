# Authentication Wycheproof vectors

These twelve JSON files are copied byte-for-byte from
[`C2SP/wycheproof`](https://github.com/C2SP/wycheproof) commit
`b61843a9a5115bb758134b6a1f5d5e502d445342`, directory
`testvectors_v1/`.

Run `scripts/check/auth-vector-provenance.py` to verify the committed file set
and SHA-256 digests. Pass `--upstream-root PATH` to additionally require an
exact checkout of that commit and compare every local file with its upstream
source bytes.

The corpus is test evidence, not a generated rscrypto artifact. Updating it
requires a new full upstream commit, reviewed digest changes, and corresponding
test-coverage review.
