# rscrypto

[![Crates.io](https://img.shields.io/crates/v/rscrypto.svg)](https://crates.io/crates/rscrypto)
[![Docs.rs](https://docs.rs/rscrypto/badge.svg)](https://docs.rs/rscrypto)
[![MSRV 1.100.0](https://img.shields.io/badge/MSRV-1.100.0-blue)](Cargo.toml)
[![License: Apache-2.0](https://img.shields.io/crates/l/rscrypto)](#license)

`rscrypto` is a Rust library of cryptographic primitives, hashes, password hashing, random generation, and checksums.
One feature model controls all of them.
Portable Rust is the reference implementation.
SIMD and assembly backends make it faster on supported targets.
Production builds do not depend on C, FFI, OpenSSL, or system libraries.

The `rscrypto` crate supplies primitives.
The companion [`rscrypto-rustls`](#tls-and-quic-provider) supplies TLS and QUIC cryptographic adapters.

## Install

`rscrypto` needs Rust 1.100 or later.
Some accelerated POWER, IBM Z, and RISC-V backends need a nightly compiler;
builds without their features, and `portable-only` builds, support the declared MSRV.
See [Toolchain](docs/platforms.md#toolchain).

Minimal `no_std` build with SHA-2 only:

```toml
[dependencies]
rscrypto = { version = "0.13", default-features = false, features = ["sha2"] }
```

All primitives, with operating-system randomness:

```toml
[dependencies]
rscrypto = { version = "0.13", features = ["full", "getrandom"] }
```

The default feature is `std`.
Set `default-features = false` to remove it.
Enable `getrandom` only for the APIs that get randomness from the operating system,
such as key generation, nonces, salts, hedged signing, ML-KEM and X-Wing encapsulation, and RSA blinding.
The [feature guide](docs/features.md) explains how to select features.
[`Cargo.toml`](Cargo.toml) defines the exact feature graph.

## Quick start

### Encrypt a message and read it back

```toml
[dependencies]
rscrypto = { version = "0.13", features = ["xchacha20poly1305", "getrandom"] }
```

```rust
use rscrypto::{Aead, XChaCha20Poly1305, XChaCha20Poly1305Key};

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let key = XChaCha20Poly1305Key::try_random()?;
  let cipher = XChaCha20Poly1305::new(&key);
  let aad = b"record 42";
  let message = b"meet at noon";
  let (nonce, ciphertext) = cipher.seal_random_to_vec(aad, message)?;
  let plaintext = cipher.decrypt_to_vec(&nonce, aad, &ciphertext)?;
  assert_eq!(plaintext.as_slice(), message);
  Ok(())
}
```

Store the returned nonce with the ciphertext. Keep the key secret.
Use the same key and associated data to decrypt later. The [runnable example](examples/aead_seal_open.rs) uses this workflow.

### Hash a message

```rust
use rscrypto::Sha256;

let one_shot = Sha256::digest(b"hello world");

let mut hasher = Sha256::new();
hasher.update(b"hello ");
hasher.update(b"world");

assert_eq!(hasher.finalize(), one_shot);
```

Hash types support one-shot and streaming use.
[`examples/README.md`](examples/README.md) has runnable examples for AEAD, signatures, RSA, P-256 and P-384 ECDH,
X25519, ML-KEM, password hashing, and backend introspection.

## Primitives and features

| Family | Primitives | Feature |
| --- | --- | --- |
| Checksums | CRC-16, CRC-24, CRC-32, CRC-32C, CRC-64/XZ, CRC-64/NVMe, Adler-32 | `checksums` or a leaf feature |
| Cryptographic hashes | SHA-2, SHA-3, SHAKE, cSHAKE, BLAKE2, BLAKE3, Ascon-Hash/XOF/CXOF | `crypto-hashes` or a leaf feature |
| Fast hashes | XXH3-64/128, XXH32, XXH64, RapidHash V3-64 | `fast-hashes` or a leaf feature |
| MACs and KDFs | HMAC-SHA-2/SHA-3, KMAC128/256, Poly1305, SipHash-1-3/2-4 (flood-resistant hash tables), HKDF-SHA-2, PBKDF2-HMAC-SHA-2 | `macs`, `kdfs`, or a leaf feature |
| Password hashing | Argon2d/i/id, scrypt, bounded PHC password records | `password-hashing` or a leaf feature |
| Signatures and RSA | ECDSA P-256/P-384, Ed25519, [ML-DSA-44/65/87](docs/mldsa.md), [SLH-DSA](docs/slhdsa.md) (all 12 FIPS 205 sets, pure and HashSLH-DSA), composite ML-DSA-65 with Ed25519 or ECDSA P-256 (draft-ietf-lamps-pq-composite-sigs-19), RSA signing, verification, encryption, and key generation | `signatures` or a leaf feature |
| Key exchange and KEMs | P-256 ECDH, P-384 ECDH, X25519, ML-KEM-512/768/1024, X-Wing (ML-KEM-768 + X25519, draft-connolly-cfrg-xwing-kem-11), HPKE Base mode (RFC 9180) with the DHKEM(X25519) + AES-128-GCM and X-Wing + ChaCha20Poly1305 suites | `key-exchange` or a leaf feature |
| AEADs | AES-GCM, AES-GCM-SIV, AES-SIV-CMAC, ChaCha20-Poly1305, XChaCha20-Poly1305, AEGIS-256, Ascon-AEAD128 | `aead` or a leaf feature |
| Random generation | `ChaCha20Rng` | `chacha20-rng` (included in `full`) |

`ChaCha20Rng` accepts a caller-supplied seed and works without `std` or `alloc`.
`rand_core` adds generator traits; `getrandom` adds operating-system seeding.

Signature and KEM keys import and export standard DER (PKCS #8 and SubjectPublicKeyInfo)
for ECDSA, Ed25519, RSA, ML-DSA, composite ML-DSA, ML-KEM, and SLH-DSA.

`AeadGather` encrypts multiple plaintext slices into one ciphertext, and decrypts one ciphertext in place across
ordered segments, for AES-GCM and ChaCha20-Poly1305.
With `aes-gcm`, `aead::expert::quic_lb` encrypts QUIC-LB connection IDs (draft-ietf-quic-load-balancers-22).
BLAKE3 supports reader, file, and batch hashing, and an expert decoder verifies sequential unkeyed Bao combined streams.
`hashes::expert::BLAKE3_PORTABLE_SOURCE` holds the source text of the portable BLAKE3 compression function.
The [feature guide](docs/features.md#capability-features) describes these APIs and their allocation and `std` requirements.

The WebSocket accept digest exists only for protocol compatibility.
It needs the `websocket-sha1` feature.
No umbrella feature, including `full`, enables it.

See [docs.rs](https://docs.rs/rscrypto) for exact types and methods.
The [API compatibility notes](docs/api-compatibility.md) explain output types, collection seeds,
and retained method names.
The [cryptographic resilience guide](docs/cryptographic-resilience.md) explains security assumptions,
migration boundaries, and the evidence to review when choosing stronger cryptography.

## TLS and QUIC provider

The `rscrypto-rustls` companion connects these primitives to rustls:

| Capability | Coverage |
| --- | --- |
| TLS | TLS 1.3; opt-in TLS 1.2 ECDHE suites |
| Client privacy | [Encrypted Client Hello](rscrypto-rustls/README.md#connect-with-encrypted-client-hello) with X25519, HKDF-SHA256, and AES-128-GCM |
| Key exchange | X25519MLKEM768 hybrid (preferred), X25519, P-256, P-384; opt-in SecP256r1MLKEM768, SecP384r1MLKEM1024, MLKEM768, MLKEM1024 |
| Authentication | ECDSA P-256/P-384, Ed25519, RSA; opt-in ML-DSA-44/65/87 (`ml-dsa`) and composite ML-DSA-65 with Ed25519 or ECDSA P-256 at provisional draft codepoints (`mldsa65-ed25519`, `mldsa65-ecdsa-p256`) |
| QUIC | QUIC v1 packet and header protection through rustls's QUIC API; Initial keys, Retry tags, connection IDs, and reset and address tokens for any transport |

The provider supports mutual authentication, session resumption, stateless tickets, and caller-supplied entropy.
rustls owns the TLS protocol and rustls-webpki validates certificate paths. The QUIC transport belongs to the application.

The companion requires `std`.
Its [manifest](rscrypto-rustls/Cargo.toml) pins rustls to the crates.io development release `=0.24.0-dev.3`.
A later companion release moves the pin to rustls 0.24.0. Consumers must depend on the same rustls version.
See the [provider guide](rscrypto-rustls/README.md) for configuration, feature requirements, and limits.

## Platforms and dispatch

The portable Rust implementation defines the correct output, byte for byte.
Every accelerated backend must give the same output.
At compile time, the target sets which backends are available.
With `std`, `rscrypto` also detects CPU features at run time and selects a backend.
If no accelerated backend is available, the portable implementation runs.

The [platform guide](docs/platforms.md) lists the supported targets.
It also explains dispatch, `no_std` support, and the limits of `portable-only`.

## Assurance

A security claim exists only while its evidence passes.
Missing or old evidence removes the claim; it never weakens a gate.

- **Correctness:** NIST, RFC, upstream, and Wycheproof vectors, independent implementations,
  property tests, negative tests, and Miri.
- **Fuzzing:** fuzz targets run the production code across primitives, parsers, state machines,
  and trait boundaries.
  Minimized inputs replay as tests.
  A separate lane runs them with sanitizers.
- **Backends:** differential tests compare each accelerated backend with portable Rust.
  They cover lengths, alignments, tails, state transitions, dispatch,
  and fallback on native targets.
- **TLS and QUIC:** provider tests compare with ring and AWS-LC.
  Separate runners exercise OpenSSL and Go TLS peers, quic-go QUIC peers,
  and a declared BoGo TLS 1.2/1.3 profile.
  The [provider test commands](scripts/README.md#test-entry-points) document campaign scope and exclusions.
- **Constant time:** [`ct.toml`](ct.toml) lists the exact operations under test.
  The evidence combines inspection of optimized linked binaries,
  BINSEC proofs for fixed-shape kernels, and DudeCT timing tests for end-to-end operations.
- **Secrets:** types that own secrets hide their contents in `Debug` output
  and clear their initialized storage on drop.
  Copy rules vary by type; for example, keyed BLAKE2 and BLAKE3 state supports `Clone`.
  The [secret ownership inventory](docs/secret-ownership.md) lists each type.
- **Failures:** verification failures do not tell why they failed.
  A failed AEAD open clears the unauthenticated plaintext.

A constant-time claim applies only to the target, features, compiler, profile,
and operation that its evidence covers.
Source code that looks branchless is not proof.

For details, see the [test evidence](docs/test-vector-coverage.md), the [constant-time model](docs/constant-time.md), the [secret lifecycle](docs/secret-lifecycle.md),
and the [threat model](THREAT_MODEL.md).

`rscrypto` has not had a third-party security audit.
The project cannot pay for one now, and automated evidence does not replace it.
`rscrypto` does not claim to be audited, FIPS 140-3 validated, formally verified,
or constant time as a whole crate.

Report a suspected vulnerability through [GitHub Private Vulnerability Reporting](https://github.com/loadingalias/rscrypto/security/advisories/new).
Follow the [`SECURITY.md`](SECURITY.md) process.
Do not open a public issue.

## Performance

The [benchmark overview](benchmark_results/OVERVIEW.md) measures the v0.13 release source on eight CI lanes and an Apple M1 Pro.
On the CI lanes, [Bench run 44681056190](https://github.com/loadingalias/rscrypto/actions/runs/44681056190) measured every primitive,
and later runs measured again each row that a later change can move.
The results cover checksums, hashes, MACs, KDFs, password hashing, AEADs, signatures, key exchange,
ML-KEM, X-Wing, HPKE, ML-DSA, SLH-DSA, and RSA.

Each point divides the fastest external median time by the `rscrypto` median for [equivalent work](docs/benchmarking.md#comparison-rules).
Above 1.00x favors `rscrypto`; the fixed tie band is 0.95x through 1.05x.
The table counts the headline points, which hold each primitive's primary operations.

| Platform              | Points | Faster | Tie | Slower | Median | Geomean |
| --------------------- | -----: | -----: | --: | -----: | -----: | ------: |
| Intel Linux (`c8i`)   |   1032 |    675 | 251 |    106 |  1.10x |   1.37x |
| AMD Linux (`c8a`)     |   1032 |    676 | 290 |     66 |  1.13x |   1.39x |
| Intel Windows (`c8i`) |   1032 |    728 | 204 |    100 |  1.19x |   1.46x |
| AMD Windows (`c8a`)   |   1032 |    774 | 190 |     68 |  1.15x |   1.48x |
| ARM64 Linux (`c9g`)   |   1032 |    650 | 300 |     82 |  1.10x |   1.35x |
| POWER10 Linux         |   1032 |    530 | 404 |     98 |  1.05x |   1.48x |
| IBM Z Linux           |   1029 |    792 | 130 |    107 |  2.13x |   2.84x |
| RISC-V Linux          |   1032 |    584 | 292 |    156 |  1.07x |   1.14x |
| macOS (Apple M1 Pro)  |   1032 |    486 | 354 |    192 |  1.04x |   1.27x |

Each point comes from one run with 20 samples, 100 ms warm-up, and 400 ms measurement, using `nightly-2026-10-24`,
on a host with no load or thermal checks. These classifications describe point estimates, not statistical significance.
The overview gives the results by family and lists what the record leaves out:
the TLS/QUIC provider targets need OpenSSL, which CI does not provision.

## Project

The guides and examples describe the source in this repository.
For a published version, use the matching [API documentation](https://docs.rs/rscrypto).

[`CHANGELOG.md`](CHANGELOG.md) lists published changes.

## License

Licensed under [Apache-2.0](LICENSE). Three ChaCha20-Poly1305 assembly files adapted from AWS-LC and
BoringSSL keep their [ISC](LICENSE-ISC) license, so the package license is `Apache-2.0 AND ISC`.
