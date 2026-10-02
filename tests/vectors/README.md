# ML-KEM ACVP vectors

These four JSON files are checked-in copies of the ML-KEM FIPS 203 prompt
and expected-result files from NIST's [`usnistgov/ACVP-Server`](https://github.com/usnistgov/ACVP-Server),
commit `15c0f3deeefbfa8cb6cd32a99e1ca3b738c66bf0`.

- They were imported between 2026-06-15 and 2026-06-18.
  Commit `490356f4` completed the inventory.
- The upstream repository contains U.S. government test material.
  It does not publish a separate license file.
- The JSON payloads match the upstream `gen-val/json-files/ML-KEM-*-FIPS203` files after key-order and whitespace normalization.
  No test case was changed.

`SHA256SUMS` lists all four payload files.
To verify them, run this command from this directory:

```bash
shasum -a 256 -c SHA256SUMS
```

`tests/mlkem_acvp.rs` reads every file.
