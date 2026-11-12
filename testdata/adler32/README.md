# Adler-32 zlib fixtures

These files are independent zlib checksum evidence for Adler-32.
CPython 3.14.8's `zlib` module, linked to zlib 1.2.12, made them on 2026-11-11
on Apple Silicon macOS, through `scripts/lib/python.sh`.
rscrypto code computed none of these bytes.

- `payload.bin` is the 4,072-byte uncompressed payload.
  It holds 48 copies of the 64-byte line
  `rscrypto zlib fixture: Adler-32 trailer and a preset dictionary\n`,
  then 1,000 bytes of the xxHash sanity-buffer generator.
  The generator starts a 64-bit state at 2654435761, emits its top byte,
  then multiplies the state by 11400714785074694797 modulo 2^64.
- `dictionary.bin` is the 113-byte preset dictionary
  `preset dictionary for the rscrypto zlib fixture: ` followed by the same line.
- `payload.zlib` is `zlib.compress(payload, 9)`.
- `payload-dictionary.zlib` is the output of
  `zlib.compressobj(level=9, zdict=dictionary)` for the payload, then `flush()`.

The generator checks that `zlib.decompress` and `zlib.decompressobj(zdict=dictionary)`
return the payload.
`zlib.adler32` gives `bc00580d` for the payload and `889629f6` for the dictionary;
`zlib.crc32` gives `48023396` and `d6d29ad8`.
Both streams end in the payload checksum, big-endian.
The dictionary stream sets FLG.FDICT and stores the dictionary checksum,
big-endian, in its header after CMF and FLG (RFC 1950, section 2.2).

`tests/adler32.rs` reads these fields at their RFC 1950 offsets
and checks them against rscrypto's Adler-32.
The test does not inflate; rscrypto has no zlib parser or codec.

`SHA256SUMS` is the complete four-file payload inventory.
Verify it from this directory with `shasum -a 256 -c SHA256SUMS`.
