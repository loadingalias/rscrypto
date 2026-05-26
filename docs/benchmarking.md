# Benchmarking

rscrypto benchmarks are for defensible engineering decisions, not decoration.
The public claim is only as strong as the competitor set, platform matrix, and
extraction method behind it.

## Methodology

- Compare against established Rust baselines and the fastest relevant external
  implementation per primitive family.
- Keep raw Criterion output under `benchmark_results/<date>/<os>/<arch>/`.
- Keep the generated result header intact. It records date, time, mode,
  platform, and commit.
- Use `just bench rsa` for RSA verifier benchmarks. RSA uses the same
  Criterion result format and extraction path as the other primitive benches.
- RSA import rows are not pure ASN.1 parser rows. The rscrypto
  `parse-validate-spki-rscrypto` row measures SPKI/PKCS#1 DER parsing plus RSA
  public-key validation without constructing Montgomery state. The
  `import-spki-precompute-rscrypto` row measures `RsaPublicKey::from_spki_der`,
  including validation and Montgomery precompute. The diagnostic
  `precompute-r2-rscrypto` row isolates current `R^2 mod n` setup for imported
  public keys. Treat pure DER parsing, key import, and precompute as separate
  optimization targets when reading local results.
- Diagnostic RSA runs with `diag` include `rsa-oracle-failure-timing`, which
  measures valid RSA-2048 PSS verification beside short signatures,
  representatives equal to the modulus, fixed-width padding failures, tampered
  signatures, unsupported and legacy JWT/TLS selectors, legacy/non-RSA COSE
  selectors, COSE padding confusion, and X.509 PSS SHA-1/default rejection
  paths. Use it as timing-review input, not as a standalone proof that an
  oracle class is closed.
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
| RSA verification | RustCrypto `rsa`, `ring`, target-available `aws-lc-rs` |
| BLAKE2 / password hashing | RustCrypto, `dryoc` where shape-compatible |
| XXH3 / RapidHash | upstream crates |
| CRC | `crc`, `crc-fast`, `crc32fast`, `crc32c`, `crc64fast`, `crc64fast-nvme` |

## Excluded Competitors

These are intentional exclusions, not omissions.

- `libsodium-sys` / `sodiumoxide`: FFI to C libsodium. Adds system-library
  build friction across no_std, wasm, and cross-target CI. `dryoc` covers the
  libsodium-style pure-Rust comparison, while `aws-lc-rs` covers the C/ASM
  speed-floor story.
- `openssl`: excluded because FFI and system-OpenSSL linkage make the
  deployment story ambiguous for the normal bench matrix.
- `boring`: excluded because it is redundant with `aws-lc-rs` for the current
  benchmark matrix.
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
