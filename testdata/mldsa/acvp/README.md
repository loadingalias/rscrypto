# ML-DSA ACVP vectors

These six JSON files are unmodified NIST ACVP inputs and expected outputs from
[`usnistgov/ACVP-Server`](https://github.com/usnistgov/ACVP-Server/tree/975de31eb83d87039ec88934fdc47d8c312b892d/gen-val/json-files),
commit `975de31eb83d87039ec88934fdc47d8c312b892d`, retrieved 2026-09-21.
The source directories are `ML-DSA-keyGen-FIPS204`, `ML-DSA-sigGen-FIPS204`,
and `ML-DSA-sigVer-FIPS204`. Local filenames prefix the operation to the upstream
`prompt.json` and `expectedResults.json` names. No cases or payload bytes were changed.
`SHA256SUMS` records every retained payload.

The National Institute of Standards and Technology (NIST) supplies these vectors.
The upstream licensing notice is retained in `NIST-NOTICE.txt`; trailing whitespace
was removed on 2026-09-22 without changing its wording.
These files contain test data only; no upstream implementation source is included.

The production ML-DSA unit tests consume all 75 key-generation, 360 signing, and
180 verification cases. Signing covers pure and prehash external interfaces,
internal formatted-message and external-mu interfaces, and both randomness modes
for all three parameter sets. The twelve prehash identifiers use independent
hash oracles. Official successful sigGen outputs also supply positive verification
cases for prehash combinations missing from the sigVer sample corpus.

Vector success proves the covered contracts; timing, cleanup, resource bounds,
hostile-input coverage beyond these cases, and independent ML-DSA differentials
require separate evidence.

The [runtime subset](runtime/README.md) records field-level hexadecimal decoding
for the WASM/WASI runner. Its binary files have a separate hash manifest.
