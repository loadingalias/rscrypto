# ML-KEM CCTV vectors

These files come from [`C2SP/CCTV`](https://github.com/C2SP/CCTV) commit `50a8ecf2a220f4c8bdc4f085789b8e85c26829e7`, directory `ML-KEM/`,
retrieved 2026-09-28, and are licensed CC0 1.0.

- `modulus-ML-KEM-*.txt`: one invalid encapsulation key per line.
  Every value from q to 2^12 - 1 appears in every coefficient position,
  and each key must fail the FIPS 203 modulus check.
  Upstream ships these gzip-compressed; they are stored here decompressed, with content unchanged.
- `strcmp-ML-KEM-*.txt`: a decapsulation key and a ciphertext whose re-encryption differs only after a zero byte.
  Implicit rejection must fire, so a `strcmp`-style comparison fails these vectors.
  These are byte-for-byte copies.

Upstream's intermediate, unlucky-sampling,
and accumulated vectors target the FIPS 203 initial public draft,
whose key generation differs from the final standard, so they are not imported.
`tests/mlkem_cctv.rs` instead checks the Go standard library's final-standard accumulated ML-KEM-768 hashes:
10,000 iterations in the normal suite and 1,000,000 iterations as an ignored test.

`SHA256SUMS` is the complete payload inventory; verify it from this directory with `shasum -a 256 -c SHA256SUMS`.
