# SHA-2 vector provenance

The five `.blb` files are copied byte-for-byte from
`RustCrypto/hashes` commit `82c36a428f8d6f05f3bfccdedb243e9d1f85359d`,
under `sha2/tests/data/`. The upstream repository licenses them under
`MIT OR Apache-2.0`.

After checking out that exact commit, verify the source bytes with:

```bash
scripts/check/hash-vector-provenance.py --sha2-root /path/to/RustCrypto-hashes
```

The verifier pins every source and destination SHA-256. No conversion is
performed.
