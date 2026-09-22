# RSA NIST CAVP subsets

These three JSON documents derive from NIST's official
[`186-3rsatestvectors.zip`](https://csrc.nist.gov/CSRC/media/Projects/Cryptographic-Algorithm-Validation-Program/documents/dss/186-3rsatestvectors.zip)
archive, SHA-256 `8405aeb3572a4f98ed4b1a3ccb3f2f49e725462dd28ec4759d6a15d88855d19c`.
They were imported into rscrypto beginning on 2026-05-22 in commit `e2996405`.
The archive contains U.S. government test material and no separate third-party
license file.

Each JSON document identifies its source response files, retained profiles,
transform, and expected record counts. `SHA256SUMS` is the complete three-file
output inventory. Verify it from this directory with
`shasum -a 256 -c SHA256SUMS`. `tests/rsa_nist_cavp.rs`,
`tests/rsa_public_key.rs`, and RSA's internal tests consume the subsets.
