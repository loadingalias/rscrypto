# rscrypto-rustls

A pure-Rust cryptographic provider for [rustls](https://crates.io/crates/rustls), backed by
[rscrypto](https://crates.io/crates/rscrypto). It gives rustls TLS 1.3, opt-in TLS 1.2, and the
QUIC v1 endpoint cryptography, with hybrid post-quantum key exchange by default.

## Why use it

- **No C toolchain.** Cargo builds it with no C compiler or C build tool. Its normal and build
  dependency graph has no ring, AWS-LC, OpenSSL, or BoringSSL.
- **Post-quantum by default.** The default provider prefers X25519MLKEM768. Opt-in features add
  three more ML-KEM groups, ML-DSA certificates, and composite ML-DSA.
- **Measured primitives.** rscrypto selects SIMD and assembly kernels for the CPU, with portable
  Rust as the reference. Its [benchmark record](https://github.com/loadingalias/rscrypto/blob/main/benchmark_results/OVERVIEW.md)
  compares each primitive with the fastest external library on nine platforms.
- **QUIC endpoint keys.** The `quic` module adds what rustls leaves to a QUIC transport: Initial
  keys, Retry tags, QUIC-LB connection IDs, stateless reset tokens, and address tokens.
- **Your own entropy.** Each part can take its randomness from your `SecureRandom`. With
  `chacha20-rng`, one seed replays a whole run, for deterministic simulation tests.
- **Small clients.** `ClientProvider` links no signing, ticket, or OS entropy code.
- **Recorded secret handling.** The provider forbids unsafe code. rscrypto records the owner and
  the cleanup of each secret, and runs its constant-time checks on six CI platforms.

Do not use it if you need FIPS-validated cryptography, the stable rustls 0.23 API, or code with a
third-party security audit. See [Limits](#limits).

## Install

```toml
[dependencies]
rscrypto-rustls = "0.1"
rustls = "=0.24.0-dev.3"
```

Use the same rustls version as the provider, so that the configuration types match. The crate needs
`std` and Rust 1.100.0 or later.

## Quick start

```rust
use std::sync::Arc;

// Add the trust anchors that the client accepts.
let roots = rustls::RootCertStore::empty();
let provider = Arc::new(rscrypto_rustls::default_provider());
let config = rustls::ClientConfig::builder(provider)
  .with_root_certificates(roots)
  .with_no_client_auth()?;
# let _ = config;
# Ok::<(), rustls::Error>(())
```

## Connect with Encrypted Client Hello

The default build includes the [RFC 9849 client suite](https://www.rfc-editor.org/rfc/rfc9849.html#section-9).
To offer ECH, get the server's
`ECHConfigList` from a trusted source, such as its HTTPS DNS record through an encrypted resolver.
Pass the wire bytes to rustls. Decode base64 first if your resolver returns the DNS presentation form.

```rust
use std::sync::Arc;

use rscrypto_rustls::hpke::ALL_SUPPORTED_SUITES;
use rustls::{ClientConfig, RootCertStore, client::EchConfig, pki_types::EchConfigListBytes};

fn ech_client(list: EchConfigListBytes<'_>, roots: RootCertStore) -> Result<ClientConfig, rustls::Error> {
  let ech = EchConfig::new(list, ALL_SUPPORTED_SUITES)?;
  ClientConfig::builder(Arc::new(rscrypto_rustls::default_provider()))
    .with_ech(ech.into())
    .with_root_certificates(roots)
    .with_no_client_auth()
}
```

Connect with the service's real DNS name. Use this TLS 1.3 configuration only for services covered by that ECH list.
After the handshake, `connection.side().ech_status()` reports `EchStatus::Accepted` when the server accepts ECH.
A rejection ends the connection. If rustls returns `Error::RejectedEch(rejection)` with retry configurations,
use `EchConfig::for_retry(rejection, ALL_SUPPORTED_SUITES)` for a fresh connection. Keep ECH required on that connection.

For a build with custom features, enable `ech` for the adapter and `getrandom` for the suite list.
For caller-supplied randomness, construct `hpke::X25519HkdfSha256Aes128Gcm::new(random)` and pass a static reference in your suite list.

## Choose a provider

| Constructor | Use it for |
| --- | --- |
| `default_provider()` | An owned provider for a rustls config builder |
| `DEFAULT_PROVIDER` | An API that borrows a provider, such as `WebPkiServerVerifier::builder` |
| `post_quantum_required_provider()` | Connections that must use X25519MLKEM768 |
| `ClientProvider::new(random)` | A TLS 1.3 client over your own `SecureRandom`, with no OS entropy |

The first three use OS entropy and need the `getrandom` feature. No constructor installs a global
provider.

## Features

The default features enable the TLS 1.3 suites, client ECH support, the default key-exchange groups, classical signatures, and `getrandom`.
The `ech` feature selects rscrypto's `hpke`, `x25519`, and `aes-gcm` features.

| Opt-in feature | Adds |
| --- | --- |
| `ml-dsa` | ML-DSA-44, ML-DSA-65, and ML-DSA-87 authentication |
| `mldsa65-ed25519`, `mldsa65-ecdsa-p256` | Composite ML-DSA authentication, experimental |
| `chacha20-rng` | The `replay` module: one seeded random stream for each connection |
| `portable-only` | rscrypto's portable code only, for validation |

## What it supports

| Area | Default | Opt-in |
| --- | --- | --- |
| TLS 1.3 suites | AES-256-GCM, AES-128-GCM, ChaCha20-Poly1305 | — |
| Client ECH | X25519, HKDF-SHA256, AES-128-GCM | Supply the server's ECH configuration to offer it |
| Key exchange | X25519MLKEM768, X25519, P-256, P-384 | SecP256r1MLKEM768, SecP384r1MLKEM1024, MLKEM768, MLKEM1024 |
| Signatures | ECDSA P-256 and P-384, Ed25519, RSA-PSS, RSA PKCS#1 v1.5 | ML-DSA; composite ML-DSA-65 with Ed25519 or ECDSA P-256 |
| TLS 1.2 | — | Six ECDHE suites with AES-GCM or ChaCha20-Poly1305 |
| QUIC v1 | Initial keys, Retry tags, QUIC-LB IDs, reset and address tokens | — |
| Session tickets | Stateless tickets; their keys rotate every six hours | `ticket::ticketer_with` for your own entropy and clock |

- To offer an opt-in group, add it to the provider's `kx_groups`.
- To enable TLS 1.2, set the provider's `tls12_cipher_suites` to
  `cipher_suite::ALL_TLS12_CIPHER_SUITES`. Each TLS 1.2 connection requires the extended master
  secret.
- `sign::KEY_PROVIDER` loads PKCS#8, SEC1, and PKCS#1 keys. `sign::Provider::new(random).compact()`
  uses less memory and signs more slowly.
- `with_root_certificates` builds rustls's verifier, which suits a short-lived client.
  `verify::PreparedServerVerifier::new` and `verify::PreparedClientVerifier::new` prepare EC and ML-DSA roots once for repeated handshakes.
- `with_algorithms` applies the catalog's certificate algorithms to each chain link and its TLS mapping to handshake signatures.
  Rustls checks these chains directly.
  Use `with_tls_signature_mapping` to restrict only handshake signatures and keep prepared roots.

## Limits

- The crate pins rustls to the development release `=0.24.0-dev.3`. A later release moves the pin to
  rustls 0.24.0.
- Composite ML-DSA is experimental. Its code points are private-use values: 0xFE41 for
  ML-DSA-65 with Ed25519, and 0xFE40 for ML-DSA-65 with ECDSA P-256. They change when IANA assigns
  code points.
- The prepared verifiers support no revocation lists, no optional client authentication, and no
  custom root hints. Use rustls's verifiers for these.
- The provider is not FIPS validated. Neither rscrypto nor this crate has had a third-party security
  audit.

## Documentation

The [API documentation](https://docs.rs/rscrypto-rustls) gives each operation, its errors, and its
limits.

| Module | Covers |
| --- | --- |
| [`cipher_suite`](https://docs.rs/rscrypto-rustls/latest/rscrypto_rustls/cipher_suite/) | TLS 1.3 and TLS 1.2 suites |
| [`hpke`](https://docs.rs/rscrypto-rustls/latest/rscrypto_rustls/hpke/) | ECH suites and their entropy source |
| [`kx_group`](https://docs.rs/rscrypto-rustls/latest/rscrypto_rustls/kx_group/) | Key-exchange groups and their entropy source |
| [`verify`](https://docs.rs/rscrypto-rustls/latest/rscrypto_rustls/verify/) | Signature algorithms, prepared verifiers, and composite code points |
| [`sign`](https://docs.rs/rscrypto-rustls/latest/rscrypto_rustls/sign/) | Key loading, signing, and memory use |
| [`ticket`](https://docs.rs/rscrypto-rustls/latest/rscrypto_rustls/ticket/) | Session tickets and key rotation |
| [`replay`](https://docs.rs/rscrypto-rustls/latest/rscrypto_rustls/replay/) | Replayable entropy for each connection |
| [`quic`](https://docs.rs/rscrypto-rustls/latest/rscrypto_rustls/quic/) | QUIC v1 endpoint cryptography |

The [secret ownership](https://github.com/loadingalias/rscrypto/blob/main/docs/secret-ownership.md#rustls-symmetric-adapters)
and [secret lifecycle](https://github.com/loadingalias/rscrypto/blob/main/docs/secret-lifecycle.md#rustls-symmetric-adapters)
records give the owner and cleanup of each secret.

## Security

Report a suspected vulnerability through the
[security policy](https://github.com/loadingalias/rscrypto/blob/main/SECURITY.md). Do not open a
public issue.

rustls owns the TLS protocol, and rustls-webpki owns certificate path validation. This crate owns
the cryptographic adapters. The rscrypto
[threat model](https://github.com/loadingalias/rscrypto/blob/main/THREAT_MODEL.md) covers the
primitives, not this crate.
