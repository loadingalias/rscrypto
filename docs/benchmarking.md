# Benchmarking

rscrypto benchmarks are for defensible engineering decisions, not decoration.
The public claim is only as strong as the competitor set, platform matrix, and
extraction method behind it.

## Methodology

- Compare against established Rust baselines and the fastest relevant external
  implementation per primitive family.
- Keep raw Criterion output under `benchmark_results/<date>/<os>/<arch>/`.
- Rewrite `benchmark_results/OVERVIEW.md` from extracted raw results, not by
  hand-editing headline numbers.
- Use speedup as `external_crate_time / rscrypto_time`; values above `1.00x`
  mean rscrypto is faster.
- Treat platform-specific claims as invalid until that platform has raw results
  in the benchmark tree.
- Do not publish README claims from a known-incomplete run.

## Competitor Set

Current expanded competitors:

| Area | Competitors |
|---|---|
| AEAD | RustCrypto AEADs, `aws-lc-rs`, `ring`, `aegis` |
| SHA-2 / HMAC / HKDF / PBKDF2 | RustCrypto, `aws-lc-rs`, `ring` |
| Ed25519 / X25519 | dalek, `aws-lc-rs`, `ring` where shape-compatible, `dryoc` |
| BLAKE2 / password hashing | RustCrypto, `dryoc` where shape-compatible |
| XXH3 / RapidHash | upstream crates, `gxhash`, `ahash`, `foldhash` |
| CRC | `crc`, `crc-fast`, `crc32fast`, `crc32c`, `crc64fast`, `crc64fast-nvme` |

## Excluded Competitors

These are intentional exclusions, not omissions.

- `libsodium-sys` / `sodiumoxide`: FFI to C libsodium. Adds system-library
  build friction across no_std, wasm, and cross-target CI. `dryoc` covers the
  libsodium-style pure-Rust comparison, while `aws-lc-rs` covers the C/ASM
  speed-floor story.
- `openssl`: FFI and system-OpenSSL link friction. `aws-lc-rs` covers the same
  performance niche without the deployment ambiguity.
- `boring`: redundant with `aws-lc-rs` for this benchmark purpose.
- `twox-hash`: useful for migration coverage, but not a primary performance
  baseline for modern XXH3; `xxhash-rust` already covers that lane.
- generic `crc`: already used as a correctness oracle. The performance bar is
  the specialized CRC crates.
- `digest`: trait utility, not an algorithm.

## Shape Deviations

- `ring` AES-128-GCM is included because ring supports it natively and TLS /
  standards users care about the AES-128 column.
- `ring` X25519 is excluded from static-key DH benches. Its public API exposes
  ephemeral agreement that consumes the private key, so each iteration would
  force a fresh keygen-and-discard path instead of matching rscrypto, dalek,
  `aws-lc-rs`, and `dryoc`.
- `dryoc` XChaCha20-Poly1305 is excluded from one-shot AEAD benches because
  dryoc 0.7 exposes the libsodium secretstream shape, not the IETF one-shot
  detached AEAD shape.

## Publication Gate

Before updating README performance claims:

1. Every claimed primitive has result IDs in the raw run.
2. The overview was generated from raw artifacts.
3. Fastest-external comparisons are available for the claimed family.
4. Known compile-log artifacts are excluded from the overview.
5. The README claim matches the overview exactly.
