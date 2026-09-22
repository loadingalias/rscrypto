# ML-KEM ACVP vectors

These four JSON documents are checked-in representations of the ML-KEM FIPS 203
prompt and expected-result files from NIST's
[`usnistgov/ACVP-Server`](https://github.com/usnistgov/ACVP-Server) commit
`15c0f3deeefbfa8cb6cd32a99e1ca3b738c66bf0`. They were imported into rscrypto
across 2026-06-15 and 2026-06-18, with the complete inventory established by
commit `490356f4`. The upstream repository contains U.S. government test
material and does not publish a separate license file.

The JSON payloads match the upstream `gen-val/json-files/ML-KEM-*-FIPS203`
files after key-order and whitespace normalization; no test cases were changed.
`SHA256SUMS` is the complete four-file payload inventory. Verify it from this
directory with `shasum -a 256 -c SHA256SUMS`.
`tests/mlkem_acvp.rs` consumes every file.
