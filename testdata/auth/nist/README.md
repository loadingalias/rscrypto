# NIST P-256 ECDH vectors

`KAS_ECC_CDH_PrimitiveTest_P-256.rsp` is the complete `[P-256]` section from
NIST CAVP's `KAS_ECC_CDH_PrimitiveTest.txt`. The source is the official
[`ecccdhtestvectors.zip`](https://csrc.nist.gov/CSRC/media/Projects/Cryptographic-Algorithm-Validation-Program/documents/components/ecccdhtestvectors.zip)
archive published by NIST.

The reproducible transform is deliberately narrow:

1. Require archive SHA-256
   `5fff092551f2d72e89a3d9362711878708f9a14b502f0dfae819649105b0ea39`.
2. Require the archive to contain only `KAS_ECC_CDH_PrimitiveTest.txt`.
3. Normalize CRLF line endings to LF.
4. Retain the bytes starting at `[P-256]` and ending immediately before
   `[P-384]`.
5. Remove the section-separator blank lines and terminate the extracted file
   with one LF.

The resulting file has SHA-256
`5a7006d1ae4f7001ba7d6d45c2c2f1f8bc5e5d48e2021eb55c5995cd055eea32`
and contains all 25 P-256 component-test records. Run:

```bash
scripts/lib/python.sh scripts/check/auth-vector-provenance.py \
  --nist-archive PATH/TO/ecccdhtestvectors.zip
```

The default invocation verifies the committed digest without requiring a
network download. The optional archive argument also reproduces and compares
the exact transform.
