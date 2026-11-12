# XXH32 and XXH64 frame fixtures

These files are independent frame checksum evidence for XXH32 and XXH64.
The reference tools made them on 2026-11-11 on Apple Silicon macOS:
LZ4 CLI 1.11.0, Zstandard CLI 1.5.7, and xxhsum 0.8.4.
rscrypto code computed none of these bytes.

- `content.bin` is the 3,849-byte uncompressed content.
  It holds 32 copies of the 64-byte line
  `rscrypto frame checksum fixture: XXH32 for LZ4, XXH64 for Zstd.\n`,
  then 1,024 bytes of the xxHash sanity-buffer generator,
  then the first 777 bytes of the 32 lines again.
  The generator starts a 64-bit state at 2654435761, emits its top byte,
  then multiplies the state by 11400714785074694797 modulo 2^64.
- `content.lz4` is an LZ4 frame with a content size, block checksums, and a content checksum:
  `lz4 -q -BX --content-size -B1024 content.bin content.lz4`.
  Its four blocks hold at most 1,024 content bytes each.
  The third block holds the generator bytes, which do not compress, so the frame stores that block uncompressed.
- `content.zst` is a single-segment Zstandard frame with a content checksum:
  `zstd -q --check -19 content.bin -o content.zst`.

`lz4 -t` and `zstd -t` accept both frames.
xxhsum 0.8.4 gives XXH32 `0377b3f9` and XXH64 `3c3ababcd693d1ed` for `content.bin`.
The LZ4 frame stores the first value little-endian. The Zstandard frame stores the low 32 bits of the second, little-endian.

`tests/xxhash_frames.rs` reads each checksum field at its documented offset
and checks it against rscrypto's XXH32 or XXH64.
The test does not decompress; rscrypto has no frame parser or codec.

`SHA256SUMS` is the complete three-file payload inventory.
Verify it from this directory with `shasum -a 256 -c SHA256SUMS`.
