# rscrypto-rustls

This unpublished crate supplies rscrypto-backed TLS 1.3 and QUIC cryptographic
adapters for rustls 0.24 development, including hybrid key exchange, classical
authentication, and opt-in ML-DSA authentication. The `quinn` feature supplies a QUIC v1 transport adapter for
quinn-proto 0.12 development.

The [manifest](Cargo.toml) pins rustls and Quinn to upstream `main` revisions.
These are development revisions, not released rustls 0.24 or Quinn 0.12 versions.
Consumers must use the same Git sources and revisions so their configuration
types match the provider's types.

The crate requires `std` and the same minimum Rust version as rscrypto.
Each algorithm feature forwards to the corresponding rscrypto leaf feature.
Default features enable every TLS 1.3 suite, key-exchange group, and classical
signature algorithm; `ml-dsa` and `quinn` are opt-in.
`portable-only` selects rscrypto's portable dispatch for validation.

rustls provides protocol composition, rustls-webpki provides certificate path
validation, and rustls-pki-types provides credential types. Their built-in crypto
backends are disabled. The provider's normal and build dependency graph must
exclude ring, AWS-LC, OpenSSL, BoringSSL, and C build tools. The pinned
quinn-proto revision makes its browser-WASM `ring` dependency optional
([Quinn PR #2930](https://github.com/quinn-rs/quinn/pull/2930)).

## Implementation inventory

| Operation | rscrypto implementation | Adapter |
| --- | --- | --- |
| Transcript hash | `Sha256`, `Sha384` | [hash](src/hash.rs) |
| HMAC | `HmacSha256`, `HmacSha384`; each call creates a keyed state | [hash](src/hash.rs) |
| HKDF extract and expand | `HkdfSha256`, `HkdfSha384`; imported PRKs and multi-part info | [hash](src/hash.rs) |
| TLS records and QUIC packets | `Aes128Gcm`, `Aes256Gcm`, `ChaCha20Poly1305` | [aead](src/aead.rs) |
| QUIC header protection | `Aes128HeaderProtection`, `Aes256HeaderProtection`, `ChaCha20HeaderProtection` | [quic](src/quic.rs) |
| Stateless tickets | `Aes256Gcm`, fresh process-local keys, atomic counter nonces; rotation on a monotonic or caller clock | [ticket](src/ticket.rs) |
| Entropy | OS entropy through `getrandom::fill` with the `getrandom` feature, or a caller-selected `SecureRandom` without it; partial primitive inputs clear on failure | [random](src/random.rs), [kx_group](src/kx_group.rs), [sign](src/sign.rs), [client](src/client.rs) |
| Certificate and handshake verification | ECDSA P-256/P-384, Ed25519, RSA-PSS and RSA PKCS#1 v1.5 | [verify](src/verify.rs) |
| Key import and signing | rscrypto PKCS#8/SEC1/PKCS#1 codecs; blinded ECDSA, Ed25519, blinded RSA-PSS | [sign](src/sign.rs) |
| ML-DSA authentication | Pure ML-DSA-44/65/87 with the empty TLS context; keys prepared once (or compact on request) and hedged signing | [sign](src/sign/mldsa.rs), [verify](src/verify.rs) |
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
P-384. Every group requires its primitive feature, and the hybrid requires both
`x25519` and `ml-kem`; `ALL_KX_GROUPS` and the group constants use OS entropy and
also require `getrandom`. The provider supplies no TLS 1.2 suites.
The hybrid follows RFC 10024: ML-KEM precedes X25519 in both shares and the
shared secret. It validates the ML-KEM encapsulation key before server key
generation. A well-formed altered ciphertext retains ML-KEM's implicit rejection;
the TLS Finished check rejects the resulting different secret. The classical
component is available to rustls's fallback optimization.

Malformed public shares and entropy failures never return a successful exchange.
No new timing qualification is claimed.

TLS 1.3 record encryption reads gathered plaintext directly into reserved ciphertext output.
The provider does not stage plaintext in its output vector. Core rscrypto owns output
initialization and error cleanup, so the companion keeps its `forbid(unsafe_code)` boundary.

Use `DEFAULT_PROVIDER` when an API borrows a provider, for example
`WebPkiServerVerifier::builder(roots, &rscrypto_rustls::DEFAULT_PROVIDER)`.
This constant avoids constructing an owned provider for a borrowed argument.
Use `default_provider()` when rustls's explicit `builder` or
`builder_with_details` needs ownership or when customizing its lists.
Both forms prefer hybrid key exchange and permit classical
fallback. `post_quantum_required_provider()` offers only X25519MLKEM768; it
requires hybrid key establishment, not a post-quantum certificate chain. Neither
constructor installs a global provider. Cargo features determine the available
lists. The constant and both constructors require `getrandom`; each also
requires at least one signature verification feature. Primitive-only feature
subsets still compile. Rustls rejects configurations without usable suites or groups.

`ClientProvider::new(random)` is a TLS 1.3 client over a caller-owned
`SecureRandom`, and it does not need `getrandom`. Build it in a `static`, because
its key-exchange groups live in it, and pass `provider()` to rustls's client
builder. The provider offers the enabled suites, the groups in
`ALL_KX_GROUPS`'s order, and `verify::SUPPORTED_ALGORITHMS`. Its key provider
refuses every private key (clearing owned DER) and its ticketer factory refuses,
so it supports no client authentication and cannot serve. A client built this way
links no signing, ticket, or OS-entropy code; `scripts/bench/provider_size.py`
measures the linked size of a minimal client with each constructor on
top-secret's host targets.
`verify::ALL_ALGORITHMS` preserves the certificate algorithm catalog when callers
build a custom TLS signature-scheme mapping with `WebPkiSupportedAlgorithms::new`.

`verify::PreparedServerVerifier` and `verify::PreparedClientVerifier` prepare the
configured roots' P-256, P-384 and ML-DSA keys once, when the verifier is built, instead of
once per issued certificate. Install them with
`ClientConfig::builder(..).dangerous().with_custom_certificate_verifier` and
`ServerConfig::builder(..).with_client_cert_verifier`. Results, errors, signature
schemes and configuration hashing match rustls's `WebPkiServerVerifier` and
mandatory-authentication `WebPkiClientVerifier` for the same roots, with this
provider's algorithms and no revocation lists; use those verifiers when you need
CRLs, optional client authentication or custom root hints. The client verifier
asks rustls's verifier for the exact error when a chain fails, so failures cost
twice the work. RSA and Ed25519 anchors, and every key a peer sends, are
imported per signature and never retained. `retained_anchor_bytes()` reports
the retained key memory on 64-bit targets: 1,425 bytes per P-256 anchor, 2,001 per
P-384 anchor, and 21,864, 38,888 or 68,200 per ML-DSA-44, 65 or 87 anchor, plus
hash-map overhead. rustls's own verifiers with `verify::SUPPORTED_ALGORITHMS`
retain no prepared anchors.

Choose the server verifier by how many handshakes it will verify. On the Intel
and Graviton4 bench hosts, preparing a Debian or Fedora bundle (148 and 146 roots,
42 and 44 of them P-256 or P-384) costs 4.1–5.1 ms. It saves 84–90 µs per handshake
only when the chain ends at a P-256 or P-384 root; chains that end at an RSA root
cost the same with either verifier. The preparation pays for itself after 46–61
such handshakes. A short-lived client, such as a command that makes a few
connections, should use rustls's `WebPkiServerVerifier` with
`verify::SUPPORTED_ALGORITHMS`, which `with_root_certificates` builds: for 1, 3,
and 10 handshakes it costs 2.3–123 times less than the prepared verifier. A
long-lived client that reuses one verifier for many handshakes to EC-rooted origins
should use `PreparedServerVerifier`. Lazy per-anchor preparation would not help the
short-lived case, because preparing one anchor (about 98–117 µs) costs more than one
prepared verification saves. The `tls-server-verifier` benchmark measures these
rows with captured origin chains and the bundles in `tools/provider-bench/fixtures/server-verifier`.

`sign::KEY_PROVIDER`, which uses OS entropy and requires `getrandom`, accepts PKCS#8 keys, EC SEC1 keys, and RSA PKCS#1 keys,
bounded to 16 KiB. Its signing keys expose SPKI for `rustls::crypto::Credentials::new`.
Owned DER input is zeroized on success and rejection, including oversized input.
Borrowed static DER remains the caller's responsibility; it cannot be cleared
through an immutable borrow.
ECDSA and RSA signing draw blinding and RSA-PSS salt from the key provider's
entropy source; Ed25519 signing does not. RSA signing keys require 3,072–8,192-bit moduli and
exponent 65,537. Certificate verification accepts deployed 2,048–8,192-bit RSA
keys under rscrypto's legacy verification policy. TLS 1.3 ECDSA signature
schemes match their exact curve and hash; cross-hash ECDSA and RSA PKCS#1 v1.5
are available for certificate verification only.

With `ml-dsa`, RFC 9881 PKCS#8 keys load and are prepared once. Provider tests
cover the seed and combined seed/key forms; the expanded form also loads through
the same rscrypto import. The prepared key keeps the transformed secrets and
public matrix (28,800, 48,256, or 81,024 bytes for ML-DSA-44, 65, or 87), and
the provider drops the compact key. Signing uses fresh randomness and the empty context required
by draft-ietf-tls-mldsa-06. Pure ML-DSA and HashML-DSA remain separate; HashML-DSA
keys and signatures do not enter this TLS profile. All three parameter sets
support all-ML-DSA certificate chains. Keys are shared by `Arc`, and concurrent
signers read the same prepared key. `sign::Provider::new(random).compact()` keeps
compact keys (2,560, 4,032, or 4,896 bytes) instead; each signature then prepares
temporary heap storage of the sizes above, so it expands the matrix once rather
than in every rejection attempt, and produces the same signatures more slowly.
Prepared secrets are cleared before their storage is freed.

For scoped entropy, construct caller-owned static `kx_group::X25519Group`,
`P256Group`, `P384Group`, or `X25519MlKem768Group` values and `sign::Provider`
with the same `&'static dyn rustls::crypto::SecureRandom`. Assign them to the
`CryptoProvider`'s `kx_groups` and `key_provider`, and assign that source to
`secure_random`. Changing `secure_random` alone does not change existing groups
or signing keys. Each constructor retains its selected source and propagates
failures without OS fallback. The source must be cryptographically secure;
deterministic test sources are unsuitable for production. These constructors
need only their primitive features; without `getrandom`, the provider links no
OS entropy source.

`ticket::ticketer_with(random, time)` supplies stateless session tickets with
keys from a caller-selected `SecureRandom` and rotation timed by a caller's rustls
`TimeProvider`; pass the same clock as the server configuration. It requires
`aes-gcm`. `ticket::ticketer()` is the same ticketer over OS entropy and the
monotonic clock, and also requires `getrandom`. Keys rotate every six hours and
the advertised lifetime is six hours. Share the returned owner across server
connections. Each key issues at most 2^24 tickets of at most 16 KiB plaintext.
Rotation occurs on use: the first use after six hours replaces the key and keeps
one previous key, and the first use at or after twelve hours erases a key, even
after a long idle period. Inactive keys can stay resident until that use. A
caller clock that falls behind delays rotation; it never revives an erased key.
Process restart replaces ticket keys. A missing time reading, encryption failure,
or failed rotation suppresses tickets without failing TLS; after a failed
rotation the previous key still opens its tickets. Early data remains off unless
the caller explicitly enables it.

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

Quinn has no client error variant for TLS initialization failure. This
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

The crate stays unpublished while it depends on unreleased rustls and Quinn revisions.
rscrypto keeps its `v<version>` tags. Once this crate can be published, its tags
will be `rscrypto-rustls-v<version>`.
