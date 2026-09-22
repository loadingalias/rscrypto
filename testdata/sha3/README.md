# SHA-3 vector provenance

The six `.blb` files are copied byte-for-byte from
`RustCrypto/hashes` commit `1637e892b5658941d04a4d895165b66780c7d7ab`,
under `sha3/tests/data/`. The upstream repository licenses them under
`MIT OR Apache-2.0`. They were imported into rscrypto on 2026-03-14 in
commit `e94d8ce7`; no transformation was applied.

`SHA256SUMS` is the complete six-file payload inventory. Verify it from this
directory with `shasum -a 256 -c SHA256SUMS`.
`tests/sha3_official_vectors.rs` consumes the corpus.
