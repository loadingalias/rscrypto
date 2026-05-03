# rscrypto

> Pure Rust cryptography, hardware-accelerated on ten architectures. `no_std` first.

[![Crates.io](https://img.shields.io/crates/v/rscrypto.svg)](https://crates.io/crates/rscrypto)
[![MSRV 1.91.0](https://img.shields.io/badge/MSRV-1.91.0-blue)](Cargo.toml)
[![License: MIT OR Apache-2.0](https://img.shields.io/crates/l/rscrypto)](#license)
[![CI](https://github.com/loadingalias/rscrypto/actions/workflows/ci.yaml/badge.svg)](https://github.com/loadingalias/rscrypto/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/gh/loadingalias/rscrypto/branch/main/graph/badge.svg?token=ILNLPXBW0P)](https://codecov.io/gh/loadingalias/rscrypto)

---

**On 5,796 head-to-head benchmarks against the established Rust crypto baselines, `rscrypto` wins 3,717 — at a 1.75× geometric-mean speedup.** Across nine Linux CI runners (AMD Zen4 / Zen5, Intel Sapphire Rapids / Ice Lake, AWS Graviton3 / 4, IBM Z, IBM POWER10, RISE RISC-V) plus macOS Apple Silicon. Zero default dependencies. No C, no FFI, no OpenSSL, no `libcrypto`.

| What you compute | Against | What you'll see |
|---|---|---|
| **SHA-3 / SHAKE** | RustCrypto `sha3` | **2.18× / 2.60×** geomean — peaks at 25.83× / 22.41× on IBM Z |
| **BLAKE3 ≥ 64 KiB** | `blake3` | **2.37×** geomean — peaks at 7.70× |
| **AEAD** (AES-GCM, AES-GCM-SIV, ChaCha20-Poly1305, AEGIS, Ascon) | RustCrypto AEADs, `aegis` | **1.84×** geomean — AES-GCM decrypt 2.51×, AES-GCM-SIV encrypt 3.79× |
| **Ed25519** signing | `ed25519-dalek` | **1.57×** geomean |
| **CRC family** (CRC-16/24/32/32C/64) | `crc`, `crc32fast`, `crc64fast`, `crc-fast` | **4.41×** geomean — CRC-32C 2.24×, CRC-64/NVMe 2.35× |

Full per-platform results: [`benchmark_results/OVERVIEW.md`](benchmark_results/OVERVIEW.md). Numbers above were recomputed from raw Criterion medians on 2026-04-29.

## Quick Start

```toml
[dependencies]
rscrypto = { version = "0.1", default-features = false, features = ["sha2"] }
```

```rust
use rscrypto::{Digest, Sha256};

let one_shot = Sha256::digest(b"hello world");

let mut h = Sha256::new();
h.update(b"hello ");
h.update(b"world");
assert_eq!(h.finalize(), one_shot);
```

That is the whole pattern. One-shot for convenience, streaming for everything else, byte-identical results regardless of which path the runtime picks.

## Why I Built `rscrypto`

I needed a Rust crypto stack that ran on every arch I deployed to, didn't pull a C toolchain into my supply chain, and didn't quietly fall back to a slow portable path on the CPUs my services actually used. Rust has excellent crates for individual primitives — but stitching them together meant a dependency graph wide enough to make supply-chain audits painful and a performance story that depended entirely on which crate I happened to land on.

So I built one. `rscrypto` is a single-crate stack you can shrink to one feature (`sha2`, `aes-gcm`, anything) or expand to the full primitive set (`full`). The portable Rust path is byte-for-byte authoritative; SIMD and ASM kernels are accelerators, differential-tested against the portable path on every release.

I also got help. **IBM** opened POWER10 and IBM Z runner access for this project — without that, the `ppc64le` and `s390x` backends would not exist. **RISE** opened RISC-V runner access — without that, the `riscv64` backend would not exist. I am genuinely grateful.

The crate is alive: see [`CHANGELOG.md`](CHANGELOG.md) for what shipped and what's next.  Vulnerabilities go to [`SECURITY.md`](SECURITY.md). Feature requests go to [GitHub Issues](https://github.com/loadingalias/rscrypto/issues).

**NOTE** The 'CHANGELOG.md' is built by cargo-rail and my commit style here wasn't following the limitations cargo-rail defines for the CHANGELOG. This is being ironed out this week. Forgive the messiness.

## What's Inside

| Family | Algorithms | Smallest feature |
|---|---|---|
| **Cryptographic hashes** | SHA-2 (224/256/384/512/512-256), SHA-3 (224/256/384/512), SHAKE128/256, cSHAKE256, BLAKE2b/2s, BLAKE3 (hash, keyed, derive-key, XOF), Ascon-Hash256, Ascon-Xof, Ascon-CXof | `sha2`, `sha3`, `blake2b`, `blake2s`, `blake3`, `ascon-hash` |
| **Fast (non-crypto) hashes** | XXH3-64, XXH3-128, RapidHash 64/128 (with `BuildHasher`) | `xxh3`, `rapidhash` |
| **Checksums** | CRC-16 (CCITT, IBM), CRC-24 (OpenPGP), CRC-32 (ISO-HDLC), CRC-32C (Castagnoli), CRC-64/XZ, CRC-64/NVMe | `crc16`, `crc24`, `crc32`, `crc64` |
| **MACs / KDFs** | HMAC-SHA-{256, 384, 512}, KMAC256, HKDF-SHA-{256, 384}, PBKDF2-HMAC-SHA-{256, 512} | `hmac`, `kmac`, `hkdf`, `pbkdf2` |
| **Password hashing** | Argon2 (id / d / i), scrypt, PHC-string encode + bounded-policy verify | `argon2`, `scrypt`, `phc-strings` |
| **Signatures / KEX** | Ed25519 sign / verify, X25519 ECDH | `ed25519`, `x25519` |
| **AEADs** | AES-256-GCM, AES-256-GCM-SIV, ChaCha20-Poly1305, XChaCha20-Poly1305, AEGIS-256, Ascon-AEAD128 | `aes-gcm`, `aes-gcm-siv`, `chacha20poly1305`, `xchacha20poly1305`, `aegis256`, `ascon-aead` |
| **Utilities** | `platform::caps()` runtime detection, hex encode/decode, `SecretBytes` zeroizing wrapper, constant-time equality | always available |

Need just one primitive? Enable one leaf feature. Need the lot? Enable `full`. Either way the dependency graph stays the same: zero default deps; `getrandom`, `serde`, and `rayon` are opt-in.

## Bring the Toolbox

```toml
[dependencies]
rscrypto = { version = "0.1", default-features = false, features = ["full"] }
```

```rust
use rscrypto::{
  ChaCha20Poly1305, ChaCha20Poly1305Key, Checksum, Crc32C,
  Digest, Ed25519Keypair, Ed25519SecretKey, FastHash, HkdfSha256, HmacSha256,
  Sha256, Shake256, X25519SecretKey, Xof, Xxh3, aead::Nonce96,
};

// Checksum
let crc = Crc32C::checksum(b"data");

// Hash (one-shot or streaming)
let hash = Sha256::digest(b"data");
let mut h = Sha256::new();
h.update(b"da"); h.update(b"ta");
assert_eq!(h.finalize(), hash);

// MAC + constant-time verify
let tag = HmacSha256::mac(b"key", b"data");
assert!(HmacSha256::verify_tag(b"key", b"data", &tag).is_ok());

// KDF
let mut okm = [0u8; 32];
HkdfSha256::new(b"salt", b"ikm").expand(b"info", &mut okm)?;

// XOF
let mut xof = Shake256::xof(b"data");
let mut out = [0u8; 64];
xof.squeeze(&mut out);

// Signature
let kp = Ed25519Keypair::from_secret_key(Ed25519SecretKey::from_bytes([7u8; 32]));
let sig = kp.sign(b"data");
assert!(kp.public_key().verify(b"data", &sig).is_ok());

// Key exchange
let alice = X25519SecretKey::from_bytes([7u8; 32]);
let bob = X25519SecretKey::from_bytes([9u8; 32]);
assert_eq!(alice.diffie_hellman(&bob.public_key())?, bob.diffie_hellman(&alice.public_key())?);

// AEAD (in-place, detached tag)
let cipher = ChaCha20Poly1305::new(&ChaCha20Poly1305Key::from_bytes([0x11; 32]));
let nonce = Nonce96::from_bytes([0x22; 12]);
let mut buf = *b"data";
let tag = cipher.encrypt_in_place(&nonce, b"aad", &mut buf)?;
cipher.decrypt_in_place(&nonce, b"aad", &mut buf, &tag)?;
assert_eq!(&buf, b"data");

// Fast hash
let _ = Xxh3::hash(b"data");
# Ok::<(), Box<dyn std::error::Error>>(())
```

Password hashing uses the PHC string format with a bounded-policy verifier:

```toml
[dependencies]
rscrypto = { version = "0.1", default-features = false, features = ["password-hashing", "getrandom"] }
```

```rust
use rscrypto::{Argon2Params, Argon2VerifyPolicy, Argon2id, Scrypt, ScryptParams, ScryptVerifyPolicy};

let password = b"correct horse battery staple";

let argon2 = Argon2Params::new().build()?;
let encoded = Argon2id::hash_string(&argon2, password)?;
assert!(Argon2id::verify_string_with_policy(password, &encoded, &Argon2VerifyPolicy::default()).is_ok());
assert!(Argon2id::verify_string_with_policy(b"wrong", &encoded, &Argon2VerifyPolicy::default()).is_err());

let scrypt = ScryptParams::new().build()?;
let encoded = Scrypt::hash_string(&scrypt, password)?;
assert!(Scrypt::verify_string_with_policy(password, &encoded, &ScryptVerifyPolicy::default()).is_ok());
assert!(Scrypt::verify_string_with_policy(b"wrong", &encoded, &ScryptVerifyPolicy::default()).is_err());
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Why You Can Trust This

Every one of the following is enforced by tests in `tests/`, `fuzz/`, or the CI matrix.

- **Portable Rust is the byte-for-byte authority.** Every SIMD or ASM backend is differential-tested against the portable path on every CI run. If a backend ever produces a different byte, the build fails.
- **Three-tier dispatch.** Compile-time `#[cfg(target_feature)]` → runtime detection (`is_x86_feature_detected!` and equivalents, cached in `platform::caps()`) → portable fallback. The fallback is always present.
- **Constant-time verification.** All MAC / AEAD / signature comparisons run through `ct::constant_time_eq` with a `black_box` barrier — no early exit, no branch on secret bytes.
- **Opaque verification errors.** `VerificationError` is zero-size and leaks no failure detail. AEAD `decrypt` paths wipe the output buffer before returning the error, so plaintext never escapes from a failed open.
- **Zeroize on drop.** All secret-key, shared-secret, and intermediate-secret types perform volatile writes plus a compiler fence on drop.
- **Overflow-safe arithmetic.** Counters, lengths, indices, and offsets use `strict_add` / `strict_sub` / `strict_mul` / shift variants. Release builds keep `overflow-checks = true`.
- **Miri-clean.** The portable backends — the same paths the SIMD kernels are compared to — pass Miri's Stacked Borrows check.
- **Continuous fuzzing.** libFuzzer harnesses cover parsers (PHC strings, encoded params), streaming APIs, and differential oracles across every primitive family. The weekly CI lane replays the corpus and uploads coverage.
- **`portable-only` for regulated deployments.** A single feature flag forces every dispatcher to the constant-time portable backend and bypasses runtime SIMD invocation — useful when an audited code path needs to be the only running code path.
- **Supply chain.** `cargo deny` and `cargo audit` run weekly. Zero default dependencies; opt-in `getrandom`, `serde`, and `rayon`.

## Feature Flags

Default features: `std` (which implies `alloc`).

| Feature | Default | Enables |
|---|---|---|
| `std` | **Yes** | Runtime CPU detection, I/O adapters. Implies `alloc` |
| `alloc` | (via `std`) | Buffered wrappers, `to_vec` methods |
| `checksums` | No | `crc16` + `crc24` + `crc32` + `crc64` |
| `crypto-hashes` | No | `sha2` + `sha3` + `blake2b` + `blake2s` + `blake3` + `ascon-hash` |
| `fast-hashes` | No | `xxh3` + `rapidhash` |
| `hashes` | No | `crypto-hashes` + `fast-hashes` |
| `macs` | No | `hmac` + `kmac` |
| `kdfs` | No | `hkdf` + `pbkdf2` (implies `hmac`) |
| `password-hashing` | No | `argon2` + `scrypt` + `phc-strings` |
| `signatures` | No | `ed25519` |
| `key-exchange` | No | `x25519` |
| `auth` | No | `macs` + `kdfs` + `password-hashing` + `signatures` + `key-exchange` |
| `aead` | No | All six AEAD leaves |
| `full` | No | `checksums` + `hashes` + `auth` + `aead` |
| `parallel` | No | Rayon-backed parallel BLAKE3 and Argon2 lane parallelism. Implies `std` + `blake3` + `argon2` |
| `portable-only` | No | Force portable backends; suppress runtime SIMD invocation. For FIPS / DO-178C / ISO 26262 / IEC 62443 deployments |
| `getrandom` | No | `random()` constructors on key/nonce types and random-salt PHC password hashing |
| `serde` | No | `Serialize` / `Deserialize` on non-secret byte wrappers (nonces, tags, public keys, signatures) |
| `serde-secrets` | No | Explicit raw-byte `Serialize` / `Deserialize` for secret keys and shared secrets. Implies `serde` |
| `diag` | No | Dispatch introspection. Implies `std` |

**Leaf features** (enable in isolation for minimal builds): `crc16`, `crc24`, `crc32`, `crc64`, `sha2`, `sha3`, `blake2b`, `blake2s`, `blake3`, `ascon-hash`, `xxh3`, `rapidhash`, `hmac`, `hkdf`, `pbkdf2`, `kmac`, `ed25519`, `x25519`, `argon2`, `scrypt`, `phc-strings`, `aes-gcm`, `aes-gcm-siv`, `chacha20poly1305`, `xchacha20poly1305`, `aegis256`, `ascon-aead`.

## Platform Support

Three-tier SIMD dispatch: compile-time `#[cfg(target_feature)]` → runtime detection (with `std`) → portable fallback. Without `std`, only compile-time detection runs.

### CI-Tested Architectures

| Architecture | ISA Extensions Used |
|---|---|
| **x86_64** Intel Sapphire Rapids | AVX-512, VPCLMULQDQ, AES-NI, SHA-NI |
| **x86_64** Intel Ice Lake | AVX-512, VPCLMULQDQ, AES-NI |
| **x86_64** AMD Zen4 / Zen5 | AVX-512, VPCLMULQDQ, AES-NI |
| **aarch64** AWS Graviton3 / Graviton4 | NEON, PMULL, AES-CE, SHA2-CE |
| **aarch64** macOS Apple Silicon | NEON, PMULL, AES-CE, SHA2-CE, SHA3-CE |
| **s390x** IBM Z | z/Vector, VGFM |
| **ppc64le** IBM POWER10 | AltiVec, VSX |
| **riscv64** RISE | V, Zbc |

### Targets That Build (no_std)

`thumbv6m-none-eabi`, `riscv32imac-unknown-none-elf`, `aarch64-unknown-none`, `x86_64-unknown-none`, `wasm32-unknown-unknown`, `wasm32-wasip1`.

The portable fallback is always available, so any target Rust supports will run `rscrypto` correctly. Only the SIMD / ASM acceleration is gated to the architectures above.

## Correctness and Testing

| Layer | What it covers | Command |
|---|---|---|
| Unit + integration | Official vectors, differential oracles, API invariants | `just test` |
| Property tests | 256 cases per proptest, run alongside unit + integration | `just test` |
| Feature matrix | Leaf and bundle reduced-feature combinations build cleanly | `just test-feature-matrix` |
| Miri | UB detection on the portable backends | `just test-miri` |
| Fuzz | libFuzzer on parsers + streaming, with differential oracles | `just test-fuzz` |
| Coverage | Nextest + replayed fuzz corpora, single LCOV report to Codecov | `just test-all-coverage` |
| Supply chain | `cargo deny` + `cargo audit` | Weekly CI |

The portable backend is the byte-for-byte authority for every primitive. Every SIMD or ASM kernel is differential-tested against it on every CI run; mismatch fails the build. Verification errors are opaque, secret-key types zeroize on drop, and release builds keep `overflow-checks = true`.

## Deployment Modes

### Embedded `no_std`

```toml
[dependencies]
rscrypto = { version = "0.1", default-features = false, features = ["sha2", "hmac"] }
```

Pick only the leaf features you need. `default-features = false` removes `std` and `alloc`. Add `alloc` back if you need heap-output APIs (most AEAD `_to_vec` helpers and the buffered wrappers).

### WASM (browser, edge)

```toml
[dependencies]
rscrypto = { version = "0.1", default-features = false, features = ["full", "alloc", "getrandom"] }
```

Portable Rust kernels run on `wasm32-unknown-unknown` and `wasm32-wasip1`. SIMD dispatch is not used on WASM today; the portable path is what runs. Use `getrandom` for entropy.

### Server / CLI (parallel-friendly workloads)

```toml
[dependencies]
rscrypto = { version = "0.1", features = ["full", "parallel", "getrandom"] }
```

Enables Rayon-backed parallel BLAKE3 and Argon2 lane parallelism. Best on multi-core x86_64 / aarch64.

### Regulated / FIPS-oriented (`portable-only`)

```toml
[dependencies]
rscrypto = { version = "0.1", features = ["full", "portable-only"] }
```

`portable-only` forces every dispatcher to its constant-time portable backend and bypasses runtime SIMD invocation. Intended for FIPS 140-3 / DO-178C / ISO 26262 / IEC 62443 deployment modes where the running code path must be the audited one regardless of host capabilities. Note: this suppresses *invocation* of SIMD kernels; it does not strip them from the binary. Targets needing binary-level exclusion should additionally restrict `target-feature` via `RUSTFLAGS`.

`rscrypto` provides FIPS-aligned primitives (AES-256-GCM, SHA-2, SHA-3 / SHAKE, HMAC, KMAC, HKDF, PBKDF2) but is not a FIPS 140-3 validated module. Validation requires defining the module boundary, operational environment, self-tests, documentation, and a CMVP lab review — work tracked in [`docs/compliance.md`](docs/compliance.md).

## API Conventions

| Family | One-shot | Streaming | Verify |
|---|---|---|---|
| Checksums | `Type::checksum(data)` | `new` / `update` / `finalize` / `reset` | — |
| Digests | `Type::digest(data)` | `new` / `update` / `finalize` / `reset` | — |
| XOFs | `Type::xof(data)` | `new` / `update` / `finalize_xof` / `squeeze` | — |
| MACs | `Type::mac(key, data)` | `new(key)` / `update` / `finalize` / `reset` | `verify` / `verify_tag` |
| KMAC256 | `mac_into(key, cust, data, out)` | `new(key, cust)` / `update` / `finalize_into(out)` | `verify` / `verify_tag` |
| HKDF | `derive(salt, ikm, info, out)` | `new(salt, ikm)` / `expand(info, out)` | — |
| AEAD (combined) | `encrypt(nonce, aad, pt, out)` | — | `decrypt(nonce, aad, ct, out)` |
| AEAD (detached) | `encrypt_in_place(nonce, aad, buf)` | — | `decrypt_in_place(nonce, aad, buf, tag)` |
| Fast hashes | `Type::hash(data)` | — | — |

Key, nonce, tag, and signature types round-trip through `from_bytes` / `to_bytes` / `as_bytes`. Secret-key types mask `Debug`; explicit secret display / export APIs are opt-in and should not be logged.

Full type inventory: [`docs/types.md`](docs/types.md).

## Roadmap

`rscrypto` is at 0.1.0. The release line is intentionally pre-`1.0` — the API is small enough that breaking changes can still happen, but the algorithms and platform backends are production-quality (this is the same code I run in my own services).

What I'm working on next, in rough order:

- **Post-quantum.** ML-DSA (FIPS 204) and ML-KEM (FIPS 203). Hybrid-mode helpers for transitioning from classical Ed25519 / X25519.
- **`nist-approved` feature bundle.** A curated subset that exposes only FIPS-aligned primitives, paired with `portable-only`, to make FIPS-oriented deployments a single feature flip.
- **More RISC-V hardware.** RISE provides the only RISC-V runners I have access to today. If you can offer additional RISC-V hardware (Zvk, Zvkned, Zvkb), please reach out.
- **Additional AEADs.** AEGIS-128L, Deoxys-II are on the list.
- **Audited release.** A third-party audit before v1.

If a primitive or platform you need isn't listed, open an issue. I'm prioritizing what real users want.

## Examples & Docs

| Path | Purpose |
|---|---|
| [`examples/`](examples/) | Runnable examples with feature-set instructions |
| [`docs/security.md`](docs/security.md) | Nonce lifecycle, verification handling, PHC verification limits, RISC-V backend guidance |
| [`docs/compliance.md`](docs/compliance.md) | Compliance posture across FIPS, DO-178C, ISO 26262, IEC 62443 |
| [`docs/architecture.md`](docs/architecture.md) | Module hierarchy, dispatch model, advanced internals |
| [`docs/types.md`](docs/types.md) | Full public type inventory |
| [`benchmark_results/OVERVIEW.md`](benchmark_results/OVERVIEW.md) | Per-platform Criterion medians and category scorecards |

## Security and Contributing

Vulnerabilities: report via [GitHub's Private Vulnerability Reporting](https://github.com/loadingalias/rscrypto/security/advisories/new). Acknowledgment within 48 hours; details in [`SECURITY.md`](SECURITY.md). AI-assisted reports are welcome — please disclose the assist so I can reproduce.

Issues, ideas, and PRs: [GitHub Issues](https://github.com/loadingalias/rscrypto/issues). For larger changes please open an issue first so we can discuss design.

## MSRV

**Rust 1.91.0** (Edition 2024). Tested on stable and the nightly pinned in `rust-toolchain.toml` (Miri and fuzz lanes only).

## License

Dual-licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
