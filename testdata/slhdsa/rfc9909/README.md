# SLH-DSA RFC 9909 examples

These DER files come from [RFC 9909](https://www.rfc-editor.org/rfc/rfc9909.html) Appendix C,
decoded from their RFC 7468 textual encoding without other changes. The plain-text RFC was
retrieved from `https://www.rfc-editor.org/rfc/rfc9909.txt` on 2026-10-10 with SHA-256
`2c77519a6eeff691a2931a86643b0c9346ce7018303316d864cad1e585a5a50b`.
All three use `id-slh-dsa-sha2-128s` and one key pair.

| File | Appendix | Content |
| --- | --- | --- |
| `slhdsa_sha2_128s_spki.der` | C.1 | SubjectPublicKeyInfo public key |
| `slhdsa_sha2_128s_pkcs8.der` | C.2 | Version 1 OneAsymmetricKey private key |
| `slhdsa_sha2_128s_certificate.der` | C.3 | Self-signed certificate; its signature is pure SLH-DSA over the TBSCertificate with an empty context |

The RFC errata page listed one reported erratum on 2026-10-10. Errata 8891 renames an ASN.1
arc label in Section 3; it changes no OID value and none of these bytes.

RFC test data is subject to the IETF Trust Legal Provisions. These files contain
test data only; no upstream implementation source is included.

`SHA256SUMS` records every payload. `tests/slhdsa_pkix.rs` imports the public and private keys,
checks the private key's public root against its seeds, compares rscrypto's exports byte for
byte, and verifies the certificate signature.
