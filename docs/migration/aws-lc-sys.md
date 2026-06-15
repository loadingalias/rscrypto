# Migration: `aws-lc-sys` -> `rscrypto`

There is no direct `aws-lc-sys` migration. `aws-lc-sys` exposes the low-level C
FFI surface for AWS-LC; rscrypto exposes safe Rust primitive APIs.

## What To Do Instead

If your code uses `aws-lc-sys` directly, first identify the safe operation you
need:

| `aws-lc-sys` use | Better migration target |
|---|---|
| Hashing, MAC, KDF, AEAD, signatures, X25519, RSA verify | rscrypto primitive APIs |
| TLS provider integration | keep AWS-LC through `aws-lc-rs` / your TLS stack |
| Raw BIGNUM, ASN.1, EVP, provider, engine, or FIPS-module plumbing | keep AWS-LC/OpenSSL-family bindings |

## Notes

- Do not translate C pointer ownership into rscrypto. Delete that layer and
  rebuild around safe primitive types.
- If your application depends on AWS-LC certification, provider configuration,
  or exact C ABI behavior, rscrypto is not a replacement.
- For safe wrapper migration, start with [`aws-lc-rs.md`](aws-lc-rs.md).
