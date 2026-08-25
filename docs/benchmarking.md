# Benchmarking

Use these benchmarks to compare a specific primitive, operation, input shape,
and target. Do not treat a crate-wide aggregate as a deployment result.

Benchmark numbers are only meaningful with their platform, commit, feature set,
and comparison shape. Treat every headline number as a pointer to the raw
results in [`benchmark_results/`](../benchmark_results/).

The published aggregate is the 2026-08-18 eight-runner Linux CI pass at commit
`7eb44e9`. The earlier RustCrypto HMAC-SHA-256 mismatch—key setup inside the
timed loop while the rscrypto, `ring`, and AWS-LC rows reused keyed state—is
corrected in the benchmark source and in this artifact, so the aggregate is an
equivalent-work claim. The RISE RISC-V runner did not execute in that run, so
row counts are not comparable to the nine-runner 2026-07-04 snapshot.

## Read the numbers

Speedup is reported as:

```text
external_crate_time / rscrypto_time
```

Values above `1.00x` mean `rscrypto` was faster for that row. Values below
`1.00x` mean the comparison crate was faster.

The published W/T/L summaries classify ratios above `1.05x` as wins, ratios
from `0.95x` through `1.05x` as ties, and ratios below `0.95x` as losses. Use
individual rows—not the crate-wide aggregate—when a primitive or message size
matters to a deployment. A single regressed runner can move a whole category
aggregate: in the 2026-08-18 pass the s390x ECDSA regression pulls the ECDSA
geomean from above parity to `0.87x`.

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

`.config/benchmark-matrix.json` is the source of truth for benchmark binaries,
features, selectors, aliases, and algorithm filters. Keep it synchronized with
Cargo benchmark targets; `just check` validates that contract.

## Competitor set

The comparison set in the published snapshot is Rust-focused and
shape-compatible:

| Area                         | Compared against                                                            |
| ---------------------------- | --------------------------------------------------------------------------- |
| AEAD                         | RustCrypto AEADs, `aws-lc-rs`, `ring`, `aegis`                              |
| SHA-2 / HMAC / HKDF / PBKDF2 | RustCrypto, `aws-lc-rs`, `ring`                                             |
| BLAKE2 / BLAKE3              | RustCrypto, `dryoc`, upstream `blake3`                                      |
| ECDSA P-256/P-384            | RustCrypto `p256`/`p384`, `aws-lc-rs`, `ring`                               |
| Ed25519 / X25519             | dalek, `aws-lc-rs`, `ring` where API-compatible, `dryoc`                    |
| ML-KEM-512/768/1024          | `libcrux`, `fips203`, RustCrypto `ml-kem`, and target-available `aws-lc-rs` |
| RSA import / verification    | RustCrypto `rsa`, `ring`, target-available `aws-lc-rs`                      |
| Password hashing             | RustCrypto, `dryoc` where API-compatible                                    |
| XXH3 / RapidHash             | upstream crates                                                             |
| CRC                          | `crc`, `crc-fast`, `crc32fast`, `crc32c`, `crc64fast`                       |

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
- RSA private-signing rows separate reusable-scratch setup, fixed test entropy,
  caller-provided entropy, and OS entropy. The RustCrypto comparison uses the
  same OS entropy source with private-operation blinding enabled, but its
  high-level signer returns an owned signature while rscrypto writes into a
  caller buffer; interpret allocation differences as part of those public API
  shapes.
- The diagnostic RSA blinding-inverse row isolates the production fixed-schedule
  batched inverse with reusable scratch and cleanup. Use it with the fixed- and
  caller-entropy signing rows to distinguish inverse cost from CRT
  exponentiation and whole-operation overhead.
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

Criterion is the wall-clock authority. The small `just bench-structural` suite
uses Gungraun and Valgrind to inspect instruction and cache-cost structure on a
supported Linux host. Structural counts are compiler- and model-sensitive; do
not convert them into elapsed-time claims.

Use the diagnostic recipes only after a benchmark establishes a concrete
question:

```sh
just profile sha2 'sha256/64' 10
just perf-codegen --asm <function>
just perf-llvm-lines --filter <pattern>
```

`just profile` saves a Samply capture under `target/profiles/`.
`perf-codegen` and `perf-llvm-lines` explain generated code and IR volume; they
do not prove that a change is faster.

Local runs are useful for capacity planning on your hardware. They should not be
mixed with published claims unless the run metadata and raw results are kept.
On macOS, the local benchmark entry point selects the host CPU unless the caller
provides `RUSTFLAGS` or `CARGO_ENCODED_RUSTFLAGS`; normal builds remain portable.
