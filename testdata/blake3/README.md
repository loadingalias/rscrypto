# BLAKE3 vector provenance

`test_vectors.json` is copied byte-for-byte from `BLAKE3-team/BLAKE3` commit
`8aa5145039b972ba30e98e788752d37d14568824`, file
`test_vectors/test_vectors.json`. The upstream repository provides CC0-1.0
and Apache-2.0 license files.

`test_vectors.blb` emits, for each source case, the UTF-8 key, UTF-8 context,
eight-byte little-endian input length, and the three decoded hexadecimal
outputs. It uses the blobby 0.3 VLQ format with an empty deduplication table.
The committed JSON and binary transform are checked by default:

```bash
scripts/check/hash-vector-provenance.py
```

After checking out the exact upstream commit, also verify the source bytes:

```bash
scripts/check/hash-vector-provenance.py --blake3-root /path/to/BLAKE3
```
