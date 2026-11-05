# Composite ML-DSA draft-19 vectors

`vectors.json` holds the `id-MLDSA65-ECDSA-P256-SHA512` and `id-MLDSA65-Ed25519-SHA512`
cases of Appendix E of
[draft-ietf-lamps-pq-composite-sigs-19](https://datatracker.ietf.org/doc/draft-ietf-lamps-pq-composite-sigs/19/),
which the RFC Editor is publishing as RFC 10073.

- Source: `src/testvectors.json` in [`lamps-wg/draft-composite-sigs`](https://github.com/lamps-wg/draft-composite-sigs),
  commit `b9e2674c0072df66ea09abeabc1c1be5ef9baf29` (2026-11-01). The file was last changed in
  commit `f0627ab34acfe1aee0abce4bee91ed2b577eab76` and has SHA-256
  `a60f697f9fd94c3cd5e4501a40c4396d9a9ffc7cf0f8a1ddd87dedbb46a7abf0`.
- Retrieved 2026-11-05. The draft text `draft-ietf-lamps-pq-composite-sigs-19.txt` (SHA-256
  `b4ee04416efc26e7d8de7c740dc3848ffe8b8cc9dcd035ad9d4e33e18844ec83`) carries the same values
  in Appendix E: the unwrapped JSON matches the repository file in every field of all 21 cases.
- The repository's final RFC copy, `rfc10073.xml` (SHA-256
  `65c4d9469594bddd0f7677fed2aef82b7eb0d0271be28565e7662caf90c4b448`), contains every field of
  both cases unchanged, with the same prefix, labels, and OIDs.
- Transform: the two cases above were selected, with the top-level `m` and `ctx`, and written as
  indented JSON. No field value was changed.

Each case holds base64 fields: `pk` (raw public key), `x5c` (self-signed certificate), `sk`
(raw private key), `sk_pkcs8`, `s` (signature over `m` with the empty context), and
`sWithContext` (signature over `m` with `ctx`). The generator signs ML-DSA with fresh
randomness, so the signatures are verification vectors, not signing answers.

Draft text and test data are subject to the IETF Trust Legal Provisions. These files contain
test data only; no upstream implementation source is included.

`SHA256SUMS` records the payload. `tests/composite_mldsa.rs` verifies both signatures and both
certificate signatures, imports and re-exports every key form byte for byte, and builds its hostile
encodings from these keys. The `auth_composite_mldsa` fuzz seeds start from them.
