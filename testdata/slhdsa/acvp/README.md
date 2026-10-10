# SLH-DSA ACVP vectors

These fixtures come from NIST ACVP inputs and expected outputs in
[`usnistgov/ACVP-Server`](https://github.com/usnistgov/ACVP-Server/tree/975de31eb83d87039ec88934fdc47d8c312b892d/gen-val/json-files),
commit `975de31eb83d87039ec88934fdc47d8c312b892d`, retrieved 2026-10-10. The source
directories are `SLH-DSA-keyGen-FIPS205`, `SLH-DSA-sigGen-FIPS205`, and `SLH-DSA-sigVer-FIPS205`.

The two key-generation files are unmodified: local names prefix the operation to the upstream
`prompt.json` and `expectedResults.json`.

The signing and verification files total about 68 MB upstream, mostly hexadecimal signatures,
so `sigGen.bin` and `sigVer.bin` keep every case in a compact derived form instead.
[`scripts/test/slhdsa_acvp.py`](../../../scripts/test/slhdsa_acvp.py) `derive` writes them, and
the runtime subset below, after checking each upstream file against these SHA-256 values and
the keyGen files against `SHA256SUMS`:

| Upstream file | SHA-256 |
| --- | --- |
| `SLH-DSA-sigGen-FIPS205/prompt.json` | `afa673eacdf0aec53512a159159b7632684adfcd0d88f8640a7f6f5796aacdc8` |
| `SLH-DSA-sigGen-FIPS205/expectedResults.json` | `71e8e0f7e4b0cfd1747314299204d9d4d50968d200a4ae873921eaa7aabeaad1` |
| `SLH-DSA-sigVer-FIPS205/prompt.json` | `4e7beb1233e47baa0acdd36417c66c45811aa40a4e32ffdb1a35d93b13b289fb` |
| `SLH-DSA-sigVer-FIPS205/expectedResults.json` | `259f5e2a0665de0adc0fefa45b5db3a2a6ed13c3c44d14bdaf64a80aee12c687` |

Each derived file is the 8 bytes `SLHACVP1`, the field count and the case count as
little-endian `u32` values, then every case in upstream order. A case is its fields, each a
little-endian `u32` length followed by that many bytes. Text fields are ASCII; byte fields are
the hexadecimal-decoded upstream values; an absent field is empty.

- `sigGen.bin`, 12 fields: `parameterSet`, `signatureInterface`, `preHash`, `deterministic`
  (`true` or `false`), `hashAlg`, `tgId`, `tcId`, `sk`, `message`, `context`,
  `additionalRandomness`, and the SHA-256 of the expected `signature`.
- `sigVer.bin`, 11 fields: `parameterSet`, `signatureInterface`, `preHash`, `hashAlg`, `tgId`,
  `tcId`, the expected `testPassed` (`true` or `false`), `pk`, `message`, `context`, and
  `signature`.

The hash replaces only the expected signatures, which signing reproduces; comparing their
SHA-256 keeps the full byte-equality check up to SHA-256 collision resistance. No case is
dropped. Regenerating from the pinned files reproduces these bytes exactly.

`runtime.bin` uses the same format, with 12 fields and one record per parameter set, for the
WASM and WASI vector runner (`tools/wasm-runtime-vectors`): the parameter set; the first keyGen
case's `skSeed || skPrf || pkSeed`, `pk`, and `sk`; the deterministic external pure sigGen case
with the shortest message (first on ties), as `sk`, `message`, `context`, and the full expected
`signature`; and the deterministic external pre-hash case whose `hashAlg` is the RFC 9909 pairing,
in the same four fields. The runner generates the key, signs the pure case, verifies both
expected signatures, and requires a changed message and a pure key to reject.

The National Institute of Standards and Technology (NIST) supplies these vectors.
The upstream licensing notice is retained in `NIST-NOTICE.txt`.
These files contain test data only; no upstream implementation source is included.
`SHA256SUMS` records every retained payload.

The production SLH-DSA unit tests consume all 120 key-generation, 624 signing,
and 504 verification cases for all 12 parameter sets.
Signing covers the external pure and pre-hash interfaces and the internal interface,
deterministic and hedged, through the internal-interface test hooks.
The pre-hash cases use all 12 FIPS 205 hash functions; independent hash oracles build M'.
The pure cases and the RFC 9909 pre-hash pairings also run through the public profiles:
every covered verification case, and the first signing case of each covered group.

Vector success proves the covered contracts; timing, cleanup, resource bounds,
hostile-input coverage beyond these cases, and independent SLH-DSA differentials
require separate evidence.
