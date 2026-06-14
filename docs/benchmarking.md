# Benchmarking

`rscrypto` publishes benchmarks so users can understand the shape of the crate:
where it is fast, where it is merely competitive, and where another library may
still be the better choice.

Benchmark numbers are only meaningful with their platform, commit, feature set,
and comparison shape. Treat every headline number as a pointer to the raw
results in [`benchmark_results/`](../benchmark_results/).

## Reading The Numbers

Speedup is reported as:

```text
external_crate_time / rscrypto_time
```

Values above `1.00x` mean `rscrypto` was faster for that row. Values below
`1.00x` mean the comparison crate was faster.

Use the geomean summaries for broad shape. Use individual rows when a specific
primitive or message size matters to your deployment.

## Published Sources

Raw Criterion output lives under:

```text
benchmark_results/<date>/<os>/<arch>/
```

The generated result headers record the date, platform, benchmark mode, and
commit. Public performance claims should match
[`benchmark_results/OVERVIEW.md`](../benchmark_results/OVERVIEW.md) rather than
hand-edited numbers.

Platform-specific claims need platform-specific raw results. A strong x86_64
result does not imply the same result on aarch64, Power, s390x, RISC-V, WASM, or
`no_std`.

## Competitor Set

The current public comparison set is intentionally Rust-focused and
shape-compatible:

| Area | Compared against |
|---|---|
| AEAD | RustCrypto AEADs, `aws-lc-rs`, `ring`, `aegis` |
| SHA-2 / HMAC / HKDF / PBKDF2 | RustCrypto, `aws-lc-rs`, `ring` |
| BLAKE2 / BLAKE3 | RustCrypto, `dryoc`, upstream `blake3` |
| ECDSA P-256/P-384 | RustCrypto `p256`/`p384`, `aws-lc-rs`, `ring` |
| Ed25519 / X25519 | dalek, `aws-lc-rs`, `ring` where API-compatible, `dryoc` |
| RSA import / verification | RustCrypto `rsa`, `ring`, target-available `aws-lc-rs` |
| Password hashing | RustCrypto, `dryoc` where API-compatible |
| XXH3 / RapidHash | upstream crates |
| CRC | `crc`, `crc-fast`, `crc32fast`, `crc32c`, `crc64fast` |

Some common libraries are not primary benchmark baselines:

- `openssl`, `libsodium-sys`, and `sodiumoxide` add C/FFI and system-library
  linkage that do not match the normal pure-Rust deployment shape.
- `boring` currently overlaps with the `aws-lc-rs` comparison.
- Generic trait crates such as `digest` are not algorithms.

## Shape Notes

- ECDSA rows are split by curve and operation. P-256 uses SHA-256; P-384 uses
  SHA-384.
- RSA import rows measure more than raw ASN.1 parsing when the public API also
  validates key material or prepares arithmetic state.
- `ring` X25519 is excluded from static-key Diffie-Hellman rows because its
  public API exposes an ephemeral agreement shape that consumes the private key.
- `dryoc` XChaCha20-Poly1305 is excluded from one-shot AEAD rows because the
  exposed benchmark shape is libsodium secretstream, not detached one-shot AEAD.

## Reproducing Locally

Use the `just bench` recipes when you want local numbers:

```sh
just bench
just bench rsa
just bench crate=rscrypto bench=auth filter='^ecdsa-p256/'
just bench crate=rscrypto bench=auth filter='^ecdsa-p384/'
```

Local runs are useful for capacity planning on your hardware. They should not be
mixed with published claims unless the run metadata and raw results are kept.
