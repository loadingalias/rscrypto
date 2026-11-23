# rscrypto-rustls

A cryptographic provider for [rustls](https://crates.io/crates/rustls), backed by
[rscrypto](https://crates.io/crates/rscrypto). It supplies TLS 1.3, opt-in TLS 1.2, and the QUIC v1
endpoint cryptography. Hybrid post-quantum key exchange is the default.

rustls composes the protocol, rustls-webpki validates certificate paths, and rustls-pki-types
supplies the credential types. This crate does the cryptography for them. It contains no QUIC
transport. Its normal and build dependency graph has no ring, AWS-LC, OpenSSL, or BoringSSL, and no
C build tool.

## Install

```toml
[dependencies]
rscrypto-rustls = "0.1"
rustls = "=0.24.0-dev.3"
```

This release pins rustls to the crates.io development release `=0.24.0-dev.3`, not to the stable
rustls 0.24.0. Use the same rustls version, so that your configuration types match the provider's
types. A later release moves the pin to rustls 0.24.0.

The crate needs `std` and Rust 1.100.0 or later.

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

`default_provider()` needs the default features.

## Choose a provider

| Constructor | Use it for | Needs |
| --- | --- | --- |
| `DEFAULT_PROVIDER` | An API that borrows a provider, such as `WebPkiServerVerifier::builder(roots, &rscrypto_rustls::DEFAULT_PROVIDER)` | `getrandom` |
| `default_provider()` | An owned provider for a rustls config builder, or a provider whose lists you change | `getrandom` |
| `post_quantum_required_provider()` | A provider that offers only X25519MLKEM768 | `getrandom` |
| `ClientProvider::new(random)` | A TLS 1.3 client over your own `SecureRandom`, with no OS entropy | — |

`DEFAULT_PROVIDER` and `default_provider()` prefer hybrid key exchange and allow classical fallback.
`post_quantum_required_provider()` requires hybrid key establishment. It does not require a
post-quantum certificate chain. No constructor installs a global provider.

`DEFAULT_PROVIDER` and the two functions also need at least one signature verification feature. The
Cargo features set the available suites, groups, and algorithms. rustls refuses a configuration with no usable suite or
group.

`ClientProvider` holds its key-exchange groups, so build it in a `static`. Then pass `provider()` to
the rustls client builder. It offers the enabled suites, the groups of `kx_group::ALL_KX_GROUPS`
in the same order, and `verify::SUPPORTED_ALGORITHMS`. It refuses every private key and every
ticketer, so it cannot authenticate as a client or serve. A client built this way links no signing,
ticket, or OS entropy code.

## Features

Each algorithm feature enables the rscrypto feature of the same name.

| Feature | Default | Gives |
| --- | --- | --- |
| `sha2`, `hmac`, `hkdf` | yes | Transcript hashes, HMAC, and HKDF |
| `aes-gcm`, `chacha20poly1305` | yes | Record and QUIC packet protection |
| `x25519`, `p256-ecdh`, `p384-ecdh`, `ml-kem` | yes | Key-exchange groups |
| `ecdsa-p256`, `ecdsa-p384`, `ed25519`, `rsa` | yes | Signing and verification |
| `getrandom` | yes | OS entropy for `DEFAULT_PROVIDER`, the two provider functions, `ALL_KX_GROUPS` and the group constants, `sign::KEY_PROVIDER`, and `ticket::ticketer()` |
| `ml-dsa` | no | ML-DSA-44, ML-DSA-65, and ML-DSA-87 authentication |
| `mldsa65-ed25519`, `mldsa65-ecdsa-p256` | no | [Composite ML-DSA](#composite-ml-dsa) authentication, experimental |
| `chacha20-rng` | no | The `replay` module: one seeded random stream for each connection |
| `portable-only` | no | rscrypto's portable code only, for validation |

## Cipher suites and key exchange

`cipher_suite::ALL_CIPHER_SUITES` lists the TLS 1.3 suites in preference order:
TLS13_AES_256_GCM_SHA384, TLS13_AES_128_GCM_SHA256, and TLS13_CHACHA20_POLY1305_SHA256. The AES
suites need `hkdf` and `aes-gcm`. The ChaCha20 suite needs `hkdf` and `chacha20poly1305`.

`kx_group::ALL_KX_GROUPS` prefers X25519MLKEM768, then X25519, P-256, and P-384. The hybrid needs
`x25519` and `ml-kem`, and each group needs its own primitive feature. The hybrid follows RFC 10024:
the ML-KEM part comes first in both shares and in the shared secret. The server validates the ML-KEM
encapsulation key before it generates its key. rustls can use the classical component for its
fallback optimization.

Four more groups are opt-in. No constructor offers them, so add them to a provider's `kx_groups`:

| Group | Constant / caller-entropy type | Features | Client / server share, secret (bytes) |
| --- | --- | --- | --- |
| SecP256r1MLKEM768 (RFC 10024) | `SECP256R1MLKEM768` / `P256MlKem768Group` | `p256-ecdh`, `ml-kem` | 1,249 / 1,153, 64 |
| SecP384r1MLKEM1024 (RFC 10024) | `SECP384R1MLKEM1024` / `P384MlKem1024Group` | `p384-ecdh`, `ml-kem` | 1,665 / 1,665, 80 |
| MLKEM768 (draft-ietf-tls-mlkem-11) | `MLKEM768` / `MlKem768Group` | `ml-kem` | 1,184 / 1,088, 32 |
| MLKEM1024 (draft-ietf-tls-mlkem-11) | `MLKEM1024` / `MlKem1024Group` | `ml-kem` | 1,568 / 1,568, 32 |

The two NIST-curve hybrids put the uncompressed ECDHE part first, unlike X25519MLKEM768. Each group
checks the exact share lengths, the encapsulation key (FIPS 203 section 7.2), and the classical
point. These failures are `illegal_parameter`. Pure ML-KEM has no classical component and no
fallback. A malformed share or an entropy failure never gives a successful exchange.

## Certificate verification

`verify::SUPPORTED_ALGORITHMS` verifies ECDSA P-256 and P-384, Ed25519, RSA-PSS, and RSA PKCS#1
v1.5, and ML-DSA and composite ML-DSA when their features are on. TLS 1.3 uses RSA PKCS#1 v1.5 only
for certificate signatures. A TLS 1.3 ECDSA scheme must match its curve and its hash. Peer RSA keys
can have 2,048 to 8,192 bits. To build your own TLS signature-scheme mapping with
`WebPkiSupportedAlgorithms::new`, start from `verify::ALL_ALGORITHMS`, which keeps the certificate
algorithm catalog.

`with_root_certificates` builds rustls's `WebPkiServerVerifier` with these algorithms.
`verify::PreparedServerVerifier` and `verify::PreparedClientVerifier` instead prepare the P-256,
P-384, and ML-DSA root keys once, when you build the verifier. Install them with
`ClientConfig::builder(..).dangerous().with_custom_certificate_verifier` and
`ServerConfig::builder(..).with_client_cert_verifier`.

- Results, errors, signature schemes, and configuration hashing match rustls's
  `WebPkiServerVerifier`, and its `WebPkiClientVerifier` with mandatory client authentication.
- They support no revocation lists, no optional client authentication, and no custom root hints.
  Use rustls's verifiers when you need one of these.
- When a chain fails, the client verifier asks rustls's verifier for the exact error. A failure
  therefore costs twice the work.
- RSA and Ed25519 roots, and each key that a peer sends, are imported for each signature. The
  verifiers do not keep them.

Choose the server verifier by the number of handshakes that it verifies. A short-lived client,
such as a command that makes a few connections, uses rustls's verifier. A long-lived client that
verifies many chains to P-256 or P-384 roots uses `PreparedServerVerifier`. In runs of 2026-10-22,
before the ECDSA verification changes of rscrypto 0.13, preparing a Debian or Fedora root bundle
cost 4.1–5.1 ms. It saved 84–90 µs for each handshake to an EC root, and it paid for itself after
46–61 such handshakes. A chain to an RSA root costs the same with either verifier.

`retained_anchor_bytes()` reports the memory that a prepared verifier keeps. On 64-bit targets,
each root keeps the bytes below, plus hash-map overhead:

| Root key | P-256 | P-384 | ML-DSA-44 | ML-DSA-65 | ML-DSA-87 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bytes | 1,425 | 2,001 | 21,864 | 38,888 | 68,200 |

## Keys and signing

`sign::KEY_PROVIDER` loads PKCS#8 keys, SEC1 EC keys, and PKCS#1 RSA keys of at most 16 KiB. It
needs `getrandom`. Each signing key gives its SPKI for `rustls::crypto::Credentials::new`. The
provider clears owned DER input after it accepts or refuses the key, also when the input is too
large. It cannot clear borrowed static DER, so that stays your responsibility.

`sign::Provider::new(random)` is the same key provider over your own `SecureRandom`.

- **ECDSA and RSA.** Signing draws its blinding and the RSA-PSS salt from the key provider's entropy
  source. Each ECDSA key signs with one scheme in every TLS version: P-256 with SHA-256, and P-384
  with SHA-384.
- **RSA.** Keys load with 2,048- to 8,192-bit moduli and exponent 65,537.
  `.rsa_key_sizes(sign::RsaSigningKeySizes::Bits3072To8192)` requires 3,072 bits or more. A key
  signs RSA-PSS when the peer offers it, and otherwise PKCS#1 v1.5. TLS 1.3 never signs PKCS#1 v1.5.
- **RSA workspaces.** Each RSA key keeps idle signing workspaces, at most one for each available CPU
  and at most 64. A signature squares the previous blinding pair instead of making a new factor.
  The pair is drawn again after 32 signatures, after an error, and after a process fork, so the
  factors in one such period are related.
- **Ed25519.** Signing uses no entropy.
- **ML-DSA** (`ml-dsa`). RFC 9881 PKCS#8 keys load in the seed, seed-and-key, and expanded forms.
  Signing is hedged and uses the empty context of draft-ietf-tls-mldsa-06. Pure ML-DSA only: a
  HashML-DSA key or signature does not enter this TLS profile. All three parameter sets support
  all-ML-DSA certificate chains.

By default, an ML-DSA key is prepared once when it loads, and concurrent signers share it through an
`Arc`. `.compact()` keeps no RSA workspace and no prepared ML-DSA key. Each RSA signature then builds
a workspace and draws a new factor. Each ML-DSA signature prepares temporary storage of the prepared
size. This saves memory, and signing is slower. Prepared secrets are cleared before their storage is
freed.

| ML-DSA key memory (bytes) | ML-DSA-44 | ML-DSA-65 | ML-DSA-87 |
| --- | ---: | ---: | ---: |
| Prepared | 28,800 | 48,256 | 81,024 |
| Compact, with the public key | 3,872 | 5,984 | 7,488 |

The [resource evidence](https://github.com/loadingalias/rscrypto/blob/main/docs/provider-resource-evidence.md)
records the allocations, heap use, and cleanup checks.

## Your own entropy source

To choose the entropy source, build static key-exchange groups and a `sign::Provider` over the same
`&'static dyn rustls::crypto::SecureRandom`. The group types are `kx_group::X25519Group`,
`P256Group`, `P384Group`, `X25519MlKem768Group`, and the opt-in group types. Put them in the
`CryptoProvider`'s `kx_groups` and `key_provider`, and set `secure_random` to the same source.

- A change to `secure_random` alone does not change existing groups or signing keys.
- Each type keeps its source and returns its failures. It never falls back to OS entropy.
- The source must be cryptographically secure. A deterministic test source is not safe in
  production.
- These types need only their primitive features. Without `getrandom`, the provider links no OS
  entropy source.

### Replayable runs

With `chacha20-rng`, `replay::ScopedRandom` gives each connection its own `ChaCha20Rng` stream, so
one recorded root seed replays a whole run. Use it as the source of the types above or of
`ClientProvider::new`.

1. Give each connection its own `replay::Stream`, for example a `ChaCha20Rng::child` of the
   endpoint's generator.
2. Run each rustls call of that connection inside its `replay::scope`: the call that builds the
   connection, and each call that processes peer data.
3. In production, seed the root generator once in each process, with 32 secret bytes from the
   operating system.

A replay that rebuilds the same generators and makes the same calls in the same order sends the
same bytes. One connection alone does not replay.

- Outside a scope, `fill` returns `GetRandomFailed`. A missed scope fails the handshake and never
  draws from another connection's stream.
- A shared owner never draws from a connection's stream. `ticket::ticketer_with_rng(rng, time)`
  owns its generator. An RSA key that `sign::Provider::scoped()` loads seeds its own blinding
  generator with 32 bytes from the scope that loads it.
- `ScopedRandom` refuses a draw by a shared owner, so `ticketer_with` and `sign::Provider::new` RSA
  keys over it fail at once.
- In a forked child process, each copied stream refuses to draw.

## Session tickets

`ticket::ticketer_with(random, time)` gives stateless session tickets. It takes its keys from your
`SecureRandom` and its rotation time from your rustls `TimeProvider`. Pass the same clock as the
server configuration. `ticket::ticketer()` is the same ticketer over OS entropy and the monotonic
clock. Both need `aes-gcm`, and `ticketer()` also needs `getrandom`. Share the returned ticketer
across the server's connections.

- Keys rotate every six hours, and the advertised ticket lifetime is six hours.
- Each key issues at most 2^24 tickets of at most 16 KiB of plaintext.
- Rotation occurs on use. The first use after six hours replaces the key and keeps one previous
  key. The first use at or after twelve hours erases a key. Until that use, an inactive key stays
  in memory.
- A clock that falls behind delays rotation. It never brings back an erased key.
- A process restart replaces the ticket keys. In a forked child, the first use replaces a copied
  key, and the copy only opens earlier tickets.
- A missing time reading, an encryption failure, or a failed rotation stops tickets but does not
  fail TLS. After a failed rotation, the previous key still opens its tickets.
- Early data stays off unless you enable it.

## TLS 1.2

TLS 1.2 is opt-in: no constructor enables it. `cipher_suite::ALL_TLS12_CIPHER_SUITES` lists six
ECDHE suites in preference order: ECDSA before RSA, each with AES-256-GCM, AES-128-GCM, and then
ChaCha20-Poly1305. There are no CBC, RSA key-transport, or finite-field Diffie-Hellman suites. The
AES suites need `aes-gcm`, the ChaCha20 suites need `chacha20poly1305`, and each suite needs `hmac`
or `hkdf` for the PRF.

Assign the list to an owned provider's `tls12_cipher_suites`, as the rustdoc example of the list
shows. `ClientProvider::provider()` returns an owned provider, so you opt in the same way. For client
certificates, also set its `key_provider` to a static `sign::Provider::new(random)`.

- The PRF derives a master secret only with the extended master secret (RFC 7627). A connection
  whose peer omits it fails with `handshake_failure`, so `require_ems` needs no setting.
- A client that also supports TLS 1.3 refuses the downgrade marker of a TLS 1.3 server
  (RFC 8446 section 4.1.3).
- AES-GCM records use RFC 5288 nonces, and ChaCha20-Poly1305 records use RFC 7905 nonces. A record
  whose plaintext would be larger than 2^14 bytes is refused before decryption.
- Key exchange uses X25519, P-256, or P-384. The hybrid groups are TLS 1.3 only, so
  `post_quantum_required_provider()` refuses every TLS 1.2 handshake. QUIC refuses a provider with a
  TLS 1.2 suite.
- An ECDSA key signs only its own scheme, so a peer that offers only the other ECDSA hash gets no
  signature from it.
- In TLS 1.2, an ECDSA scheme names a hash, not a curve. For example, an `ecdsa_secp384r1_sha384`
  signature from a P-256 key verifies, as Go servers send it. TLS 1.3 verification stays bound to
  the curve.

## Composite ML-DSA

The composite schemes are experimental. With `mldsa65-ed25519` or `mldsa65-ecdsa-p256`, the provider
signs and verifies the composite ML-DSA schemes of draft-ietf-tls-composite-mldsa-00:
`id-MLDSA65-Ed25519-SHA512` and `id-MLDSA65-ECDSA-P256-SHA512` of
draft-ietf-lamps-pq-composite-sigs-19 (RFC-to-be 10073). They serve CertificateVerify and the
certificate signatures of a webpki chain. Both component signatures must verify. TLS uses the empty
composite context, and ML-DSA signing is hedged.

- IANA has not assigned code points. The provider uses provisional private-use values:
  `verify::MLDSA65_ED25519_SHA512` is 0xFE41, and `verify::MLDSA65_ECDSA_SECP256R1_SHA512` is
  0xFE40. A peer interoperates only if it uses the same values. Both values change when IANA
  assigns code points.
- The provider does not offer or accept 0x090B. The earlier individual draft gave it to the Ed25519
  composite, and IANA lists it as unassigned.
- No default feature enables a composite scheme. Its feature adds it to
  `verify::SUPPORTED_ALGORITHMS`, so a provider built from that catalog offers it.
- Composite schemes are TLS 1.3 only, as section 2 of the draft requires.
- A composite PKCS#8 key loads once and keeps the prepared ML-DSA-65 key (48,256 bytes) and the
  traditional key. `.compact()` works as for ML-DSA.
- A composite key signs only its composite scheme, never a component scheme. To refuse stand-alone
  ML-DSA and traditional signatures, install `verify::COMPOSITE_ALGORITHMS` as the signature
  verification algorithms.

## QUIC v1

With `aes-gcm` and `hkdf`, the `quic` module supplies the QUIC v1 cryptography that rustls's QUIC API
leaves to the transport. It does not need `getrandom`. The QUIC-LB connection IDs need only
`aes-gcm`.

- `quic::initial_keys(client_dst_connection_id, side)` derives the Initial packet and header keys as
  rustls `quic::Keys`.
- `quic::retry_integrity_tag` and `quic::is_valid_retry` compute and check Retry integrity tags.
- `quic::LoadBalancerConfig` issues connection IDs that encode an encrypted server ID, as
  [draft-ietf-quic-load-balancers-22](https://datatracker.ietf.org/doc/draft-ietf-quic-load-balancers-22/)
  specifies. It also decodes the server ID of an ID. The configuration holds the rotation code
  point, the server-ID and nonce lengths, the first-octet form, and a `LoadBalancerKey`. The
  transport supplies each nonce, which must not repeat under one key. Decoding refuses reserved and
  foreign rotation code points and short IDs with one `None`. The no-key form, server-use octets,
  and unroutable IDs are not implemented.
- `quic::StatelessResetKey` derives the stateless-reset token of each connection ID from a static
  HMAC-SHA256 key.
- `quic::AddressTokenKey` seals and opens Retry and NEW_TOKEN address tokens of up to 4 KiB of
  plaintext. Each token has a new 16-byte nonce, and HKDF-SHA256 derives a one-use AES-256-GCM key
  from it. The transport owns the token contents, the expiry, and the replay policy.

Each key and nonce comes from the `SecureRandom` that you pass to the call. A failing source fails
the call, with no OS fallback, and leaves your data unchanged. QUIC callers must enforce the usage
limit that the provider reports for each key.

## Security

Report a suspected vulnerability through the
[security policy](https://github.com/loadingalias/rscrypto/blob/main/SECURITY.md). Do not open a
public issue.

- rustls owns the TLS protocol: the transcript, the key schedule, the HKDF labels, record framing,
  and sequence numbers. rustls-webpki owns certificate path validation. This crate owns the
  cryptographic adapters.
- A record or packet authentication failure gives no primitive error detail.
- The crate forbids unsafe code. The rscrypto
  [threat model](https://github.com/loadingalias/rscrypto/blob/main/THREAT_MODEL.md) covers the
  primitives, not this crate.
- The [ownership](https://github.com/loadingalias/rscrypto/blob/main/docs/secret-ownership.md#rustls-symmetric-adapters)
  and [lifecycle](https://github.com/loadingalias/rscrypto/blob/main/docs/secret-lifecycle.md#rustls-symmetric-adapters)
  inventories record the secret owners and their cleanup. They do not show that the compiler's
  copies are erased.
- The provider is not FIPS validated. Neither rscrypto nor this crate has had a third-party
  security audit.

## Versions

This crate's release tags are `rscrypto-rustls-v<version>`. rscrypto's tags are `v<version>`.
