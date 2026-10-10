# ML-DSA RFC 9881 examples

These DER files come from [RFC 9881](https://www.rfc-editor.org/rfc/rfc9881.html) Appendix C,
decoded from their RFC 7468 textual encoding without other changes. The plain-text RFC was
retrieved from `https://www.rfc-editor.org/rfc/rfc9881.txt` on 2026-10-09 with SHA-256
`20ae3b519bc69e32989fa6e625ab8746453eddf457e20ddbfd79d841a44237a2`.
The RFC derives every consistent example key from the seed `000102…1e1f`.

| Files | Appendix | Content |
| --- | --- | --- |
| `mldsa{44,65,87}_spki.der` | C.2 | SubjectPublicKeyInfo public keys |
| `mldsa{44,65,87}_pkcs8_seed.der` | C.1 | Version 1 OneAsymmetricKey, seed form |
| `mldsa{44,65,87}_pkcs8_expanded.der` | C.1 | Version 1 OneAsymmetricKey, expanded form |
| `mldsa{44,65,87}_pkcs8_both.der` | C.1 | Version 1 OneAsymmetricKey, both form |
| `mldsa44_pkcs8_inconsistent_seed.der` | C.4, first | Both form whose seed and expanded key disagree |
| `mldsa44_pkcs8_inconsistent_tr.der` | C.4, second | Expanded form whose public-key hash `tr` disagrees |
| `mldsa44_pkcs8_inconsistent_t0.der` | C.4, third | Expanded form whose `t0` disagrees with `s1` and `s2` |

The RFC errata page listed two verified errata on 2026-10-09 and no reported ones.
Errata 8699 corrects the Appendix A ASN.1 size of an ML-DSA-87 public key to 2,592 bytes,
which these examples already use; errata 8700 is editorial. Neither changes these bytes.

RFC test data is subject to the IETF Trust Legal Provisions. These files contain
test data only; no upstream implementation source is included.

`SHA256SUMS` records every payload. `tests/mldsa_spki.rs` compares the public keys byte for
byte with rscrypto's export from the same seed and imports them. `tests/mldsa_pkcs8.rs` imports
every private key, compares the seed and expanded forms with rscrypto's export, and requires
each inconsistent key to be rejected.
