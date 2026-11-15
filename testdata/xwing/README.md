# X-Wing vectors

These files pin the X-Wing test material that `tests/xwing.rs` and `tests/xwing_oracle.rs` use.
They were retrieved on 2026-11-13.

| File | Source | Content |
| --- | --- | --- |
| `draft-11-test-vectors.json` | [`dconnolly/draft-connolly-cfrg-xwing-kem`](https://github.com/dconnolly/draft-connolly-cfrg-xwing-kem) `spec/test-vectors.json` at `984c2f7a`, last changed in `5cb311dc` | The 3 vectors of draft-connolly-cfrg-xwing-kem-11 Appendix C |
| `concrete-hybrid-kems-04-mlkem768-x25519.json` | [`cfrg/draft-irtf-cfrg-concrete-hybrid-kems`](https://github.com/cfrg/draft-irtf-cfrg-concrete-hybrid-kems) `test-vectors.json` at tag `draft-irtf-cfrg-concrete-hybrid-kems-04` (`e76e1939`) | The 10 MLKEM768-X25519 vectors of draft-irtf-cfrg-concrete-hybrid-kems-04 Appendix B.2 |
| `low-order-points.txt` | `dconnolly/draft-connolly-cfrg-xwing-kem` `spec/low-order-points.txt` at `782c880e`, the head of pull request 39 | The 14 X25519 encodings for which X25519 gives the all-zero output |

`draft-11-test-vectors.json` is a byte-for-byte copy.
Its values equal the Appendix C text of the draft
(`draft-connolly-cfrg-xwing-kem-11.txt`, SHA-256
`1353f61f0d706199d86b6802eab83cabbb5195b2d99fb35bca20f5e4f83db786`).
Each vector gives the key seed `seed`, which is also the decapsulation key `sk`,
the encapsulation key `pk`, the 64-byte encapsulation randomness `eseed`,
the ciphertext `ct`, and the shared secret `ss`.

`concrete-hybrid-kems-04-mlkem768-x25519.json` is the `mlkem768_x25519` member of the upstream file
(SHA-256 `08b47d9fe5c827f7ceb20dd10a59bc853a949d6eaf0ad835405e86d833f92bad`),
written with `jq -c '.mlkem768_x25519'`. Its values equal the Appendix B.2 text of the draft
(`draft-irtf-cfrg-concrete-hybrid-kems-04.txt`, SHA-256
`b47aa429c2db5b53f74a6e68cdaf4f30bb754e84d72b0dca0849e92a0916353c`).
That draft states that MLKEM768-X25519 is identical to X-Wing.
Its `decapsulation_key_pq` is the 64-byte ML-KEM-768 seed `d || z` and `decapsulation_key_t` is the
X25519 secret key: together they are the SHAKE256 expansion of the decapsulation key.

`low-order-points.txt` is a byte-for-byte copy. Pull request 39 makes the all-zero check mandatory;
rscrypto follows it, as `src/auth/xwing.rs` documents. The list holds the five u-coordinates of small
order on Curve25519 and its twist in their seven encodings below 2^255, which include the
non-canonical `p` and `p + 1`. Each encoding also appears with bit 255 set, which X25519 ignores.

Internet-Draft test data is subject to the IETF Trust Legal Provisions. These files contain test data
only; no upstream implementation source is included.

`SHA256SUMS` is the complete payload inventory; verify it from this directory with `shasum -a 256 -c SHA256SUMS`.
