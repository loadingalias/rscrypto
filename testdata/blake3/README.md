# BLAKE3 vector provenance

`test_vectors.json` is copied byte-for-byte from `BLAKE3-team/BLAKE3` commit
`8aa5145039b972ba30e98e788752d37d14568824`, file
`test_vectors/test_vectors.json`. The upstream repository provides CC0-1.0
and Apache-2.0 license files. The corpus was imported into rscrypto on
2026-03-14 in commit `e94d8ce7`.

`test_vectors.blb` emits, for each source case, the UTF-8 key, UTF-8 context,
eight-byte little-endian input length, and the three decoded hexadecimal
outputs. It uses the blobby 0.3 VLQ format with an empty deduplication table.

`SHA256SUMS` is the complete source-and-derived payload inventory. Verify it
from this directory with `shasum -a 256 -c SHA256SUMS`.
`tests/blake3_official_vectors.rs` consumes the blobby form.
