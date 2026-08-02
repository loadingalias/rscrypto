# SHA-3 vector provenance

The six `.blb` files are copied byte-for-byte from
`RustCrypto/hashes` commit `1637e892b5658941d04a4d895165b66780c7d7ab`,
under `sha3/tests/data/`. The upstream repository licenses them under
`MIT OR Apache-2.0`.

After checking out that exact commit, verify the source bytes with:

```bash
scripts/check/hash-vector-provenance.py --sha3-root /path/to/RustCrypto-hashes
```

The verifier pins every source and destination SHA-256. No conversion is
performed.
