# ML-DSA RFC 9881 SubjectPublicKeyInfo examples

These three DER files are the ML-DSA-44, ML-DSA-65, and ML-DSA-87 public keys from
[RFC 9881](https://www.rfc-editor.org/rfc/rfc9881.html) Appendix C.2, decoded from their
RFC 7468 textual encoding without other changes. The plain-text RFC was retrieved from
`https://www.rfc-editor.org/rfc/rfc9881.txt` on 2026-10-09 with SHA-256
`20ae3b519bc69e32989fa6e625ab8746453eddf457e20ddbfd79d841a44237a2`.
The RFC derives every example key from the seed `000102…1e1f`.

The RFC errata page listed two verified errata on retrieval. Errata 8699 corrects
the Appendix A ASN.1 size of an ML-DSA-87 public key to 2,592 bytes, which these
examples already use; errata 8700 is editorial. Neither changes these bytes.

RFC test data is subject to the IETF Trust Legal Provisions. These files contain
test data only; no upstream implementation source is included.

`SHA256SUMS` records all three payloads. `tests/mldsa_spki.rs` compares them byte for
byte with rscrypto's export from the same seed and imports them.
