# BLAKE2 vector provenance

`blake2b.blb` and `blake2s.blb` contain the 512 corresponding cases from
`BLAKE2/BLAKE2` commit `ed1974ea83433eba7b2d95c5dcd9ac33cb847913`,
file `testvectors/blake2-kat.json`. The upstream corpus is CC0-1.0 and has
SHA-256 `5031ac14800798ae15cee79c04d65e326a575f2c968c7e2846a79bd07a1c0e61`.

The deterministic transform emits each case's `in`, `key`, and `out` hex
fields, in source order, using the blobby 0.3 VLQ format with an empty
deduplication table. After checking out the exact commit, reproduce and verify
both files with:

```bash
scripts/check/hash-vector-provenance.py --blake2-root /path/to/BLAKE2
```
