# NIST ECDH vectors

`KAS_ECC_CDH_PrimitiveTest_P-256.rsp` and `KAS_ECC_CDH_PrimitiveTest_P-384.rsp`
are the complete `[P-256]` and `[P-384]` sections from NIST CAVP's
`KAS_ECC_CDH_PrimitiveTest.txt`. The source is the official
[`ecccdhtestvectors.zip`](https://csrc.nist.gov/CSRC/media/Projects/Cryptographic-Algorithm-Validation-Program/documents/components/ecccdhtestvectors.zip)
archive published by NIST. The P-256 section was imported into rscrypto on
2026-09-03 in commit `3bde831e`; the P-384 section was imported on 2026-09-25.
The archive contains U.S. government test material and no separate third-party
license file.

The reproducible transform is deliberately narrow:

1. Require archive SHA-256
   `5fff092551f2d72e89a3d9362711878708f9a14b502f0dfae819649105b0ea39`.
2. Require the archive to contain only `KAS_ECC_CDH_PrimitiveTest.txt`.
3. Normalize CRLF line endings to LF.
4. Retain the bytes starting at the section header (`[P-256]` or `[P-384]`)
   and ending immediately before the next section header (`[P-384]` or
   `[P-521]`).
5. Remove the section-separator blank lines and terminate the extracted file
   with one LF.

The P-256 file has SHA-256
`5a7006d1ae4f7001ba7d6d45c2c2f1f8bc5e5d48e2021eb55c5995cd055eea32`, and the
P-384 file has SHA-256
`ef5d2ce668969493cf986e31f4ab2ec37944c6f70568c515bf03984366762a24`.
Each file contains all 25 component-test records for its curve.

`SHA256SUMS` is the complete payload inventory. Verify it from this directory
with `shasum -a 256 -c SHA256SUMS`.
`tests/p256_ecdh_oracle.rs` and `tests/p384_ecdh_oracle.rs` consume the
extracted sections.
