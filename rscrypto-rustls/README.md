# rscrypto-rustls

This unpublished crate supplies rscrypto-backed TLS 1.3 and QUIC cryptographic
adapters for rustls 0.23.45, including hybrid key exchange, classical authentication, and opt-in ML-DSA
authentication. The `quinn` feature supplies a QUIC v1 transport adapter for
quinn-proto 0.11.19.

The crate requires `std` and the same minimum Rust version as rscrypto.
Each algorithm feature forwards to the corresponding rscrypto leaf feature.
The default set covers the first TLS profile; `ml-dsa` is opt-in.
`portable-only` selects rscrypto's portable dispatch for validation.

rustls provides protocol composition, rustls-webpki provides certificate path
validation, and rustls-pki-types provides credential types. Their built-in crypto
backends are disabled. The provider's normal and build dependency graph must
exclude ring, AWS-LC, OpenSSL, BoringSSL, and C build tools.

The workspace pins a one-line fix to quinn-proto 0.11.19 that makes its
browser-WASM `ring` dependency optional. The upstream fix is
[Quinn PR #2930](https://github.com/quinn-rs/quinn/pull/2930).
Applications that enable `quinn` must copy the `[patch.crates-io]` entry from
the [workspace manifest](../Cargo.toml) into their own workspace manifest.
Cargo reads patches only from the consuming workspace root. Keep this exact
commit pin until a compatible Quinn release includes the fix.

## Implementation inventory

| Operation | rscrypto implementation | Adapter |
| --- | --- | --- |
| Transcript hash | `Sha256`, `Sha384` | [hash](src/hash.rs) |
| HMAC | `HmacSha256`, `HmacSha384`; each call creates a keyed state | [hash](src/hash.rs) |
| HKDF extract and expand | `HkdfSha256`, `HkdfSha384`; imported PRKs and multi-part info | [hash](src/hash.rs) |
| TLS records and QUIC packets | `Aes128Gcm`, `Aes256Gcm`, `ChaCha20Poly1305` | [aead](src/aead.rs) |
| QUIC header protection | `Aes128HeaderProtection`, `Aes256HeaderProtection`, `ChaCha20HeaderProtection` | [quic](src/quic.rs) |
| Stateless tickets | `Aes256Gcm`, fresh process-local keys, atomic counter nonces; rustls `TicketRotator` | [ticket](src/ticket.rs) |
| Entropy | OS entropy through `getrandom::fill`, or caller-selected `SecureRandom`; partial primitive inputs clear on failure | [random](src/random.rs), [kx_group](src/kx_group.rs), [sign](src/sign.rs) |
| Certificate and handshake verification | ECDSA P-256/P-384, Ed25519, RSA-PSS and RSA PKCS#1 v1.5 | [verify](src/verify.rs) |
| Key import and signing | rscrypto PKCS#8/SEC1/PKCS#1 codecs; blinded ECDSA, Ed25519, blinded RSA-PSS | [sign](src/sign.rs) |
| ML-DSA authentication | Pure ML-DSA-44/65/87 with the empty TLS context; compact boxed keys and hedged signing | [sign](src/sign/mldsa.rs), [verify](src/verify.rs) |
| QUIC Retry, reset, and address tokens | RFC 9001 AES-128-GCM Retry; HMAC-SHA256 reset tokens; HKDF-SHA256 and AES-256-GCM token keys | [quinn](src/quinn/) |
| Key exchange | `X25519SecretKey`, `P256EphemeralSecret`, `P384EphemeralSecret`, `MlKem768` | [kx_group](src/kx_group.rs) |

`cipher_suite::ALL_CIPHER_SUITES` lists enabled suites in preference order.
AES suites need `hkdf` and `aes-gcm`; ChaCha20 needs `hkdf` and `chacha20poly1305`.
Hash and primitive adapters remain available with their own leaf features.
The implementation-inventory test checks each suite's wire identifier, hash,
key size, TLS and QUIC usage limits, and available operations. Independent
AWS-LC comparisons and [RFC vectors](tests/vectors/README.md) exercise them.
`just ci-provider` checks feature subsets and the provider's own normal/build
dependency graph; independent crypto peers are dev dependencies.

rustls owns transcript composition, the TLS key schedule, HKDF labels, record
framing, and sequence numbers. These adapters perform the selected primitives
and translate buffers and errors. Record authentication failures and packet
authentication failures expose no primitive error detail. QUIC callers must
enforce the reported per-key usage limits. The provider is not FIPS validated.

Secret owners and cleanup boundaries are recorded in the workspace's
[ownership](../docs/secret-ownership.md#rustls-symmetric-adapters) and
[lifecycle](../docs/secret-lifecycle.md#rustls-symmetric-adapters) inventories.
Those contracts do not establish that compiler-made copies are erased.

`kx_group::ALL_KX_GROUPS` prefers X25519MLKEM768, then X25519, P-256, and
P-384. Every group requires `getrandom` and its primitive feature; the hybrid
requires both `x25519` and `ml-kem`. All four groups accept only TLS 1.3.
The hybrid follows RFC 10024: ML-KEM precedes X25519 in both shares and the
shared secret. It validates the ML-KEM encapsulation key before server key
generation. A well-formed altered ciphertext retains ML-KEM's implicit rejection;
the TLS Finished check rejects the resulting different secret. The classical
component is available to rustls's fallback optimization.

The pinned rustls version maps all key-exchange errors to `illegal_parameter`.
RFC 10024 requires `internal_error` for an internal decapsulation failure.
That upstream limitation remains; malformed public input and entropy failures
never return a successful exchange. No new timing qualification is claimed.

Use `default_provider()` with rustls's explicit `builder_with_provider` or
`builder_with_details`. It prefers hybrid key exchange and permits classical
fallback. `post_quantum_required_provider()` offers only X25519MLKEM768; it
requires hybrid key establishment, not a post-quantum certificate chain. Neither
constructor installs a global provider. Cargo features determine the available
lists; rustls rejects configurations without usable suites or groups.

`sign::KEY_PROVIDER` accepts PKCS#8 keys, EC SEC1 keys, and RSA PKCS#1 keys,
bounded to 16 KiB. Its signing keys expose SPKI for `CertifiedKey::keys_match`.
Owned DER input is zeroized on success and rejection, including oversized input.
Borrowed static DER remains the caller's responsibility; it cannot be cleared
through an immutable borrow.
ECDSA and RSA signing require `getrandom` for blinding and RSA-PSS salt;
Ed25519 signing does not. RSA signing keys require 3,072–8,192-bit moduli and
exponent 65,537. Certificate verification accepts deployed 2,048–8,192-bit RSA
keys under rscrypto's legacy verification policy. TLS 1.3 ECDSA signature
schemes match their exact curve and hash; cross-hash ECDSA and RSA PKCS#1 v1.5
are available for certificate verification only.

With `ml-dsa`, RFC 9881 PKCS#8 seed and combined seed/key forms load into boxed
compact keys. Signing uses fresh randomness and the empty context required
by draft-ietf-tls-mldsa-06. Pure ML-DSA and HashML-DSA remain separate; HashML-DSA
keys and signatures do not enter this TLS profile. All three parameter sets
support all-ML-DSA certificate chains. These keys are shared by `Arc`; the
provider does not retain a prepared matrix per credential.

For scoped entropy, construct caller-owned static `kx_group::X25519Group`,
`P256Group`, `P384Group`, or `X25519MlKem768Group` values and `sign::Provider`
with the same `&'static dyn rustls::crypto::SecureRandom`. Assign them to the
`CryptoProvider`'s `kx_groups` and `key_provider`, and assign that source to
`secure_random`. Changing `secure_random` alone does not change existing groups
or signing keys. Each constructor retains its selected source and propagates
failures without OS fallback. The source must be cryptographically secure;
deterministic test sources are unsuitable for production. These constructors
retain the same feature requirements as their OS-backed constants.

`ticket::ticketer()` supplies stateless session tickets with six-hour key rotation
and a twelve-hour advertised lifetime. It requires `aes-gcm` and `getrandom`.
Share the returned owner across server connections. Each key issues at most
2^24 tickets of at most 16 KiB plaintext. Rotation occurs on use; idle keys can
remain resident until the next operation. Each key enforces a monotonic
twelve-hour deadline and releases its cipher on its first use at or after expiry,
even when the rotator was idle. Process restart replaces ticket keys.
Ticket encryption or rotation failure suppresses issuance without failing TLS.
Early data remains off unless the caller explicitly enables it.

## QUIC v1

Enable `quinn` and select only a runtime feature on the application's Quinn
dependency. Use `quinn::endpoint_config()`, `quinn::client_config(tls)`, and
`quinn::server_config(tls)` from this crate. Pass the resulting configuration
and UDP socket to Quinn's `Endpoint::new`. The TLS configurations use this
provider explicitly, require TLS 1.3, and must supply a nonempty ALPN list.
The adapter accepts version 1 only. QUIC v2 and draft versions are excluded.

`quinn::HandshakeData` reports the negotiated ALPN, cipher suite, key-exchange
group, server name, and completed handshake kind. Quinn enforces the packet
keys' confidentiality and integrity limits. Rustls owns the TLS handshake,
exporters, early-data secrets, and packet-key updates. QUIC KeyUpdate does not
perform another key exchange or supply fresh post-compromise entropy.

Initial keys always use AES-128-GCM and HKDF-SHA256, even when the negotiated
suite is ChaCha20 or AES-256. Retry uses the fixed RFC 9001 key and nonce.
Endpoint configurations retain the reset key and matching connection-ID filter
when cloned. Address-token roots belong to the returned server configuration;
cloning it keeps outstanding tokens valid. Each derived token key seals once,
with at most 4 KiB of plaintext. Quinn supplies the fresh per-token random value.
Quinn also owns connection-ID generation and its Rust random generator.
`quinn::endpoint_config_with_random` selects the reset and CID filter key source
at construction; `quinn::server_config` uses the TLS provider's `secure_random`
for its address-token root. Quinn's transport RNG and the optional stateless
ticket rotator have separate entropy ownership; the ticketer uses OS entropy.

Early data is off by default. Opting in requires a replay-safe application
protocol, a client with `enable_early_data`, and a server with
`max_early_data_size = u32::MAX`. Rustls uses stateful session storage for its
one-use early-data tickets. Rejected early requests must follow Quinn's retry
contract; never assume that an application write has been accepted.

Quinn 0.11 has no client error variant for TLS initialization failure. This
adapter returns `ConnectError::EndpointStopping` for such a failure, including
entropy failure. It returns no usable session. TLS handshake-processing failures are terminal;
failed sessions expose no keys, exporter output, peer identity, or handshake data.

The bounded [loopback peer](examples/quic_interop.rs) shows explicit construction.
`just test-provider-quic` exercises independent quic-go connections, including
resumption, Retry, early data, and observed packet-key changes. The Go-server
fixture uses bounded stateful tickets: quic-go 0.61's default stateless tickets
make an ML-DSA-87 mutual-authentication ClientHello exceed its fixed 16 KiB
CRYPTO-stream limit. Provider-server cases exercise the stateless ticketer.
This is a peer configuration constraint, not a passed default-peer profile.

rscrypto retains its `v<version>` tags. This crate uses
`rscrypto-rustls-v<version>` tags when it becomes publishable, after rscrypto v0.11.
