# ML-KEM Wycheproof vectors

These twelve JSON files are byte-for-byte copies of [`C2SP/wycheproof`](https://github.com/C2SP/wycheproof) commit
`3fa63dd0344abb611f1fb1d77e119938603ea230`, directory `testvectors_v1/`, retrieved 2026-09-28.
The upstream repository licenses them under Apache-2.0.

Each parameter set has four corpora:

- `keygen_seed`: key generation from the 64-byte `d || z` seed.
- `test`: key generation followed by decapsulation, including modified
  ciphertexts that must trigger implicit rejection, and wrong seed or
  ciphertext lengths.
- `encaps`: encapsulation with a fixed message, including unreduced
  encapsulation keys that must fail the FIPS 203 modulus check.
- `semi_expanded_decaps`: decapsulation with an expanded key, including keys
  whose embedded encapsulation key or hash is corrupted.

`tests/mlkem_wycheproof.rs` consumes every case and checks the exact valid and invalid counts.
`SHA256SUMS` is the complete payload inventory; verify it from this directory with `shasum -a 256 -c SHA256SUMS`.

The corpus is test evidence, not a generated rscrypto artifact.
Updating it requires a new full upstream commit and a review of the changed cases.
