# rscrypto

> Fast, pure-Rust cryptography for servers, CLIs, embedded targets, and WASM.

[![Crates.io](https://img.shields.io/crates/v/rscrypto.svg)](https://crates.io/crates/rscrypto)
[![Docs.rs](https://docs.rs/rscrypto/badge.svg)](https://docs.rs/rscrypto)
[![CI](https://github.com/loadingalias/rscrypto/actions/workflows/ci.yaml/badge.svg)](https://github.com/loadingalias/rscrypto/actions/workflows/ci.yaml)
[![License: MIT OR Apache-2.0](https://img.shields.io/crates/l/rscrypto)](#license)

`rscrypto` is a single-crate Rust crypto stack: hashes, checksums, MACs, KDFs,
password hashing, signatures, key exchange, and AEAD encryption. Enable one
primitive for a tiny `no_std` build, or enable `full` for the whole toolbox.

Use it when you want:

- **One dependency instead of a pile of primitive crates.**
- **No C, no FFI, no OpenSSL, no `libcrypto`.**
- **Zero default third-party dependencies.** `getrandom`, `serde`, and `rayon`
  are opt-in.
- **Portable Rust first, SIMD second.** Every optimized backend is checked
  against the portable implementation.
- **Real cross-platform coverage.** The support matrix spans x86_64, aarch64,
  s390x, ppc64le, riscv64, Apple Silicon, WASM, and embedded `no_std` targets.

Do not use it as a certified FIPS module today. `rscrypto` exposes FIPS-aligned
primitives and a `portable-only` mode, but it is not a FIPS 140-3 validated
module.

## Install

Pick the primitive you need:

```toml
[dependencies]
rscrypto = { version = "0.1", default-features = false, features = ["sha2"] }
```

```rust
use rscrypto::{Digest, Sha256};

let digest = Sha256::digest(b"hello world");

let mut h = Sha256::new();
h.update(b"hello ");
h.update(b"world");

assert_eq!(h.finalize(), digest);
```

Or bring the full stack:

```toml
[dependencies]
rscrypto = { version = "0.1", features = ["full", "getrandom"] }
```

## Why It Exists

Rust has strong crypto crates, but production systems often end up stitching
together many APIs, feature policies, dependency graphs, and platform-specific
performance stories.

`rscrypto` is the opposite bet: one Rust-first crate, one consistent API shape,
one dispatch model, and one test matrix. The portable implementation is the
authority. SIMD and ASM are accelerators, never alternate sources of truth.

## Performance

Benchmarks are not marketing copy here; they are part of the design loop. The
latest public benchmark pass compares `rscrypto` against established Rust
baselines across nine Linux CI runners plus macOS Apple Silicon.

| Area | Baseline | Result |
|---|---|---|
| Overall Linux CI | 5,796 matched comparisons | **3,717 wins**, **1.75x** geomean |
| SHA-3 / SHAKE | RustCrypto `sha3` | **2.18x / 2.60x** geomean |
| BLAKE3, `>=64 KiB` | `blake3` | **2.37x** geomean |
| AEAD | RustCrypto AEADs, `aegis` | **1.84x** geomean |
| Ed25519 signing | `ed25519-dalek` | **1.57x** geomean |
| Checksums | `crc`, `crc32fast`, `crc64fast`, `crc-fast` | **4.41x** geomean |

Full raw results, platform scorecards, and methodology:
[`benchmark_results/OVERVIEW.md`](benchmark_results/OVERVIEW.md).

## What You Get

| Family | Algorithms | Feature |
|---|---|---|
| Cryptographic hashes | SHA-2, SHA-3, SHAKE, cSHAKE256, BLAKE2, BLAKE3, Ascon-Hash/XOF/CXOF | `crypto-hashes` or leaf features |
| Fast hashes | XXH3-64/128, RapidHash 64/128 | `fast-hashes`, `xxh3`, `rapidhash` |
| Checksums | CRC-16, CRC-24, CRC-32, CRC-32C, CRC-64/XZ, CRC-64/NVMe | `checksums` or leaf features |
| MACs / KDFs | HMAC-SHA-2, KMAC256, HKDF-SHA-2, PBKDF2-HMAC-SHA-2 | `auth`, `macs`, `kdfs` |
| Password hashing | Argon2d/i/id, scrypt, PHC string encode/verify | `password-hashing` |
| Signatures / KEX | Ed25519, X25519 | `signatures`, `key-exchange` |
| AEADs | AES-256-GCM, AES-256-GCM-SIV, ChaCha20-Poly1305, XChaCha20-Poly1305, AEGIS-256, Ascon-AEAD128 | `aead` or leaf features |
| Utilities | Runtime CPU detection, hex parsing/formatting, `SecretBytes`, constant-time equality | always available |

Leaf features exist for size-conscious builds: `sha2`, `sha3`, `blake3`,
`hmac`, `hkdf`, `ed25519`, `x25519`, `aes-gcm`, `chacha20poly1305`, `crc32`,
and the rest.

Migrating from an existing crate? Start with [`docs/migration/`](docs/migration/).
The guides cover RustCrypto crates, `blake3`, CRC crates, fast hashes, AEADs,
signatures, key exchange, and password hashing.

## Common Recipes

### `no_std` Hashing

```toml
[dependencies]
rscrypto = { version = "0.1", default-features = false, features = ["sha2"] }
```

### AEAD Encryption

```toml
[dependencies]
rscrypto = { version = "0.1", default-features = false, features = ["chacha20poly1305"] }
```

```rust
use rscrypto::{ChaCha20Poly1305, ChaCha20Poly1305Key, aead::Nonce96};

let key = ChaCha20Poly1305Key::from_bytes([0x11; 32]);
let nonce = Nonce96::from_bytes([0x22; 12]);
let cipher = ChaCha20Poly1305::new(&key);

let mut msg = *b"pay bob 10";
let tag = cipher.encrypt_in_place(&nonce, b"v1", &mut msg)?;

assert_ne!(&msg, b"pay bob 10");

cipher.decrypt_in_place(&nonce, b"v1", &mut msg, &tag)?;
assert_eq!(&msg, b"pay bob 10");
# Ok::<(), Box<dyn std::error::Error>>(())
```

### Password Hashing

```toml
[dependencies]
rscrypto = { version = "0.1", default-features = false, features = ["password-hashing", "getrandom"] }
```

```rust
use rscrypto::{Argon2Params, Argon2VerifyPolicy, Argon2id};

let params = Argon2Params::new().build()?;
let encoded = Argon2id::hash_string(&params, b"correct horse battery staple")?;

assert!(
  Argon2id::verify_string_with_policy(
    b"correct horse battery staple",
    &encoded,
    &Argon2VerifyPolicy::default(),
  )
  .is_ok()
);

assert!(Argon2id::verify_string_with_policy(b"wrong", &encoded, &Argon2VerifyPolicy::default()).is_err());
# Ok::<(), Box<dyn std::error::Error>>(())
```

### Server / CLI Builds

```toml
[dependencies]
rscrypto = { version = "0.1", features = ["full", "parallel", "getrandom"] }
```

`parallel` enables Rayon-backed BLAKE3 and Argon2 lane parallelism.

### Regulated or Audit-Constrained Builds

```toml
[dependencies]
rscrypto = { version = "0.1", features = ["full", "portable-only"] }
```

`portable-only` bypasses runtime SIMD invocation and forces the audited portable
backend. It does not remove SIMD code from the binary; use target-level
`RUSTFLAGS` as well if your deployment requires binary-level exclusion.

## Trust Model

The core rule is simple: optimized backends must behave exactly like portable
Rust.

- Official vectors cover standards-based primitives.
- Differential tests compare SIMD/ASM backends against portable fallbacks.
- MAC, AEAD, and signature verification use constant-time equality.
- Verification failures are opaque; AEAD failed-open paths wipe output buffers.
- Secret-bearing types zeroize on drop and mask `Debug`.
- Counters, lengths, offsets, and indices use checked or strict arithmetic.
- Miri checks the portable backends; fuzzing covers parsers, streaming APIs, and
  differential oracles.
- `cargo deny` and `cargo audit` run in CI.

Security guidance: [`docs/security.md`](docs/security.md). Vulnerabilities:
[GitHub Private Vulnerability Reporting](https://github.com/loadingalias/rscrypto/security/advisories/new).

## Platform Support

Runtime dispatch uses compile-time target features first, cached runtime CPU
detection second, and portable Rust fallback last. Without `std`, only the
compile-time tier runs.

The acceleration surface is broader than one ISA hook per target. Availability
depends on the CPU and compile/runtime feature detection; the portable fallback
is still present on every target.

| Target family | Acceleration used |
|---|---|
| x86_64 | SSE4.2 CRC32, SSSE3/PCLMULQDQ, AVX2, AES-NI, SHA-NI, AVX-512F/VL/BW/DQ, AVX-512IFMA, VPCLMULQDQ, VAES |
| aarch64 / Apple Silicon | NEON, AES, PMULL, CRC, SHA2, SHA3/EOR3, SHA512, SVE2-PMULL where available |
| s390x IBM Z | z/Vector, vector enhancements, CPACF/MSA, VGFM |
| ppc64le POWER | AltiVec, VSX, POWER8 vector/crypto, POWER9/POWER10 vector, VPMSUMD |
| riscv64 | V/RVV, Zbc, Zvbc, Zbkc, Zkne/Zknd, Zvkned, Zkt/Zvkt |
| wasm32 | SIMD128 where enabled |

Known `no_std` build targets include `thumbv6m-none-eabi`,
`riscv32imac-unknown-none-elf`, `aarch64-unknown-none`,
`x86_64-unknown-none`, `wasm32-unknown-unknown`, and `wasm32-wasip1`.

## API Shape

The APIs are intentionally repetitive:

| Task | Pattern |
|---|---|
| Checksums | `Type::checksum(data)` or `new` / `update` / `finalize` / `reset` |
| Digests | `Type::digest(data)` or `new` / `update` / `finalize` / `reset` |
| XOFs | `Type::xof(data)` or `new` / `update` / `finalize_xof` / `squeeze` |
| MACs | `Type::mac(key, data)` and `Type::verify_tag(key, data, tag)` |
| HKDF | `derive(salt, ikm, info, out)` or `new(salt, ikm)` / `expand(info, out)` |
| AEADs | typed keys/nonces with combined and detached APIs |
| Fast hashes | `Type::hash(data)` |

Full public type inventory: [`docs/types.md`](docs/types.md). Runnable examples:
[`examples/`](examples/). Migration guides: [`docs/migration/`](docs/migration/).
Internal design: [`docs/architecture.md`](docs/architecture.md).

## Project Status

Current release: **0.1.1**. The crate is pre-`1.0`, so API adjustments can still
happen, but the implementation is already tested against official vectors,
external oracles, fuzz corpora, Miri, and a broad hardware matrix.

Near-term priorities:

- ML-KEM and ML-DSA.
- A `nist-approved` feature bundle.
- More RISC-V hardware coverage.
- AEGIS-128L and Deoxys-II.
- Third-party audit before `1.0`.

If this project is useful to you, star the repo so more Rust users can find it.
If another primitive or CPU target would make it useful for you, open an issue.

## MSRV

Rust **1.91.0**. Full local validation uses the pinned nightly in
[`rust-toolchain.toml`](rust-toolchain.toml) for Miri, fuzzing, and exotic
architecture checks.

## License

Dual-licensed under either:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
