# SHA-2 vector provenance

The five `.blb` files are copied byte-for-byte from
`RustCrypto/hashes` commit `82c36a428f8d6f05f3bfccdedb243e9d1f85359d`,
under `sha2/tests/data/`. The upstream repository licenses them under
`MIT OR Apache-2.0`. They were imported into rscrypto on 2026-03-14 in
commit `e94d8ce7`; no transformation was applied.

`SHA256SUMS` is the complete five-file payload inventory. Verify it from this
directory with `shasum -a 256 -c SHA256SUMS`.
`tests/sha2_official_vectors.rs` and `tests/sha256_official_vectors.rs` consume
the corpus.
