# ML-KEM RFC 9935 examples

These DER files come from [RFC 9935](https://www.rfc-editor.org/rfc/rfc9935.html) Appendix C,
decoded from their RFC 7468 textual encoding without other changes. The plain-text RFC was
retrieved from `https://www.rfc-editor.org/rfc/rfc9935.txt` on 2026-10-09 with SHA-256
`9a369bca93a613d0b89b8f4ed119ba9ded91c00acd97ae5b9a2f9b3f02e49b4f`.
Every consistent example key derives from the 64-byte seed `000102…3e3f`, `d || z`.
The RFC prose abbreviates it as `000102...1e1f`; the encoded seeds hold all 64 bytes.

| Files | Appendix | Content |
| --- | --- | --- |
| `mlkem{512,768,1024}_spki.der` | C.2 | SubjectPublicKeyInfo encapsulation keys |
| `mlkem{512,768,1024}_pkcs8_seed.der` | C.1 | Version 1 OneAsymmetricKey, seed form |
| `mlkem{512,768,1024}_pkcs8_expanded.der` | C.1 | Version 1 OneAsymmetricKey, expanded form |
| `mlkem{512,768,1024}_pkcs8_both.der` | C.1 | Version 1 OneAsymmetricKey, both form |
| `mlkem512_pkcs8_inconsistent_seed.der` | C.4.1, first | Both form whose seed and expanded key disagree |
| `mlkem512_pkcs8_inconsistent_s.der` | C.4.1, second | Expanded form with a mutated secret vector and a valid `H(ek)` |
| `mlkem512_pkcs8_inconsistent_hash.der` | C.4.1, third | Expanded form with a mutated `H(ek)` |
| `mlkem512_pkcs8_inconsistent_z.der` | C.4.1, fourth | Both form that differs from its seed only in `z` |

The RFC errata page listed no verified errata and two reported ones on 2026-10-09.
Errata 9020 and 9021 propose the same Section 9 note: a certificate's signature should be at
least as strong as its ML-KEM key. They concern issuance policy and change none of these bytes.

RFC test data is subject to the IETF Trust Legal Provisions. These files contain
test data only; no upstream implementation source is included.

`SHA256SUMS` records every payload. `tests/mlkem_pkix.rs` compares the public keys and the
seed and expanded private keys byte for byte with rscrypto's export, imports every key,
and requires each inconsistent key to be rejected.
