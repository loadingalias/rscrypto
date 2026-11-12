# Composite ML-DSA draft-19 certificates and keys

Each file is one base64-decoded field of an Appendix E test case of
[draft-ietf-lamps-pq-composite-sigs-19](https://datatracker.ietf.org/doc/draft-ietf-lamps-pq-composite-sigs/19/),
named `<tcId>-<field>.der`. The cases are `id-MLDSA65-Ed25519-SHA512` and
`id-MLDSA65-ECDSA-P256-SHA512`. The fields are `x5c` (the self-signed certificate) and
`sk_pkcs8` (the composite PKCS #8 private key). No byte was changed.

- Source: `testdata/composite_mldsa/draft19/vectors.json` in the rscrypto repository, SHA-256
  `90d23a33ac852a8716f9ccc77da95f97382194ae987130ea9aeb412f405151b0`. Its
  [README](../../../../../testdata/composite_mldsa/draft19/README.md) records the upstream file
  (`src/testvectors.json` in [`lamps-wg/draft-composite-sigs`](https://github.com/lamps-wg/draft-composite-sigs),
  commit `b9e2674c0072df66ea09abeabc1c1be5ef9baf29`) and its match with the draft text.
- Extracted 2026-11-11. To check every file, run from the repository root:

  ```sh
  for case in id-MLDSA65-Ed25519-SHA512 id-MLDSA65-ECDSA-P256-SHA512; do
    for field in x5c sk_pkcs8; do
      jq -r --arg case "$case" --arg field "$field" '.tests[] | select(.tcId == $case) | .[$field]' \
        testdata/composite_mldsa/draft19/vectors.json | base64 -d |
        cmp - "rscrypto-rustls/tests/vectors/composite_mldsa/draft19/$case-$field.der"
    done
  done
  ```

`rscrypto-rustls/src/tests/composite.rs` verifies each certificate's signature through the
provider, authenticates a TLS client with each certificate and key through webpki, and checks that
each key signs only its composite scheme. A new draft revision replaces these files and the
repository's `vectors.json` together.

Draft text and test data are subject to the IETF Trust Legal Provisions. These files contain test
data only. The private keys are public and must never protect real traffic.

`SHA256SUMS` pins every payload.
