# ML-DSA target-runtime fixtures

These bytes are hexadecimal decodings of fields from the pinned, unchanged
[ACVP JSON files](../README.md). They exercise key generation, expanded-key
import, deterministic signing, prepared keys, and verification through the
public production API in the WASM/WASI runner. The inherited
[NIST notice](../NIST-NOTICE.txt) applies.

| Parameter set | Mode | tgId | tcId | Decoded fields |
| --- | --- | ---: | ---: | --- |
| ML-DSA-44 | keyGen | 1 | 1 | seed, pk, sk |
| ML-DSA-44 | sigGen | 1 | 7 | sk, context, message, signature |
| ML-DSA-65 | keyGen | 2 | 26 | seed, pk, sk |
| ML-DSA-65 | sigGen | 3 | 41 | sk, context, message, signature |
| ML-DSA-87 | keyGen | 3 | 51 | seed, pk, sk |
| ML-DSA-87 | sigGen | 5 | 72 | sk, context, message, signature |

For each `keyGen` group, select its first test. For each parameter set's pure,
external, deterministic `sigGen` group, select the first test with the shortest
message. Join prompts and expected results by `(tgId, tcId)`. Apply Python's
`bytes.fromhex` to each listed field, with no further transformation. Each file
is named `{parameter}-{mode}-{field}.bin`. Expected bytes come from NIST; the
implementation under test did not generate them. `SHA256SUMS` covers these files.

This is a small runtime sample. The native unit harness retains and exercises
all 615 cases, including HashML-DSA and invalid inputs.
