# Benchmarking

Use these benchmarks to compare a specific primitive, operation, input shape,
and target. Do not treat a crate-wide aggregate as a deployment result.

Benchmark numbers are only meaningful with their platform, commit, feature set,
and comparison shape. Treat every headline number as a pointer to the raw
results in [`benchmark_results/`](../benchmark_results/).

The published 2026-07-04 aggregate is archival, not an equivalent-work
performance claim. Its RustCrypto HMAC-SHA-256 rows include key setup inside
the timed loop while the rscrypto, `ring`, and AWS-LC rows reuse keyed state.
The current benchmark source corrects that mismatch; publish a new aggregate
only after regenerating the complete artifact.

## Read the numbers

Speedup is reported as:

```text
external_crate_time / rscrypto_time
```

Values above `1.00x` mean `rscrypto` was faster for that row. Values below
`1.00x` mean the comparison crate was faster.

The published W/T/L summaries classify ratios above `1.05x` as wins, ratios
from `0.95x` through `1.05x` as ties, and ratios below `0.95x` as losses. Use
individual equivalent-work rows—not the archival aggregate—when a primitive or
message size matters to a deployment.

## Published sources

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

## Competitor set

The comparison set in the published snapshot is Rust-focused and
shape-compatible:

| Area | Compared against |
|---|---|
| AEAD | RustCrypto AEADs, `aws-lc-rs`, `ring`, `aegis` |
| SHA-2 / HMAC / HKDF / PBKDF2 | RustCrypto, `aws-lc-rs`, `ring` |
| BLAKE2 / BLAKE3 | RustCrypto, `dryoc`, upstream `blake3` |
| ECDSA P-256/P-384 | RustCrypto `p256`/`p384`, `aws-lc-rs`, `ring` |
| Ed25519 / X25519 | dalek, `aws-lc-rs`, `ring` where API-compatible, `dryoc` |
| ML-KEM-512/768/1024 | `libcrux`, `fips203`, RustCrypto `ml-kem`, and target-available `aws-lc-rs` |
| RSA import / verification | RustCrypto `rsa`, `ring`, target-available `aws-lc-rs` |
| Password hashing | RustCrypto, `dryoc` where API-compatible |
| XXH3 / RapidHash | upstream crates |
| CRC | `crc`, `crc-fast`, `crc32fast`, `crc32c`, `crc64fast` |

Some common libraries are not primary benchmark baselines:

- `openssl`, `libsodium-sys`, and `sodiumoxide` add C/FFI and system-library
  linkage that do not match the normal pure-Rust deployment shape.
- `boring` was not included because its covered primitive cases overlapped the
  `aws-lc-rs` comparison.
- Generic trait crates such as `digest` are not algorithms.

## Shape notes

- ECDSA rows are split by curve and operation. P-256 uses SHA-256; P-384 uses
  SHA-384.
- Ed25519 signing includes both retained-keypair signing and direct
  `Ed25519SecretKey::sign` rows. The latter includes secret expansion and
  public-key derivation on every call.
- ML-KEM end-to-end rows are split by parameter set and operation:
  key generation, encapsulation, and decapsulation for ML-KEM-512, ML-KEM-768,
  and ML-KEM-1024.
- RSA import rows measure more than raw ASN.1 parsing when the public API also
  validates key material or prepares arithmetic state.
- `ring` X25519 is excluded from static-key Diffie-Hellman rows because its
  public API exposes an ephemeral agreement shape that consumes the private key.
- `dryoc` XChaCha20-Poly1305 is excluded from one-shot AEAD rows because the
  exposed benchmark shape is libsodium secretstream, not detached one-shot AEAD.

## Reproduce locally

Use the `just bench` recipes when you want local numbers:

```sh
just bench
just bench rsa
just bench crate=rscrypto bench=auth filter='^ecdsa-p256/'
just bench crate=rscrypto bench=auth filter='^ecdsa-p384/'
just bench mlkem
```

Local runs are useful for capacity planning on your hardware. They should not be
mixed with published claims unless the run metadata and raw results are kept.
