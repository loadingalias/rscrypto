# rscrypto-rustls

This unpublished crate supplies rscrypto-backed TLS 1.3 and QUIC cryptographic
adapters for rustls 0.24 development, including hybrid key exchange, classical
authentication, opt-in ML-DSA authentication, and opt-in TLS 1.2 suites. The `quic` module supplies QUIC v1 endpoint
cryptography for any transport; this crate contains no QUIC transport.

The [manifest](Cargo.toml) pins rustls to an upstream `main` revision.
This is a development revision, not a released rustls 0.24 version.
Consumers must use the same Git sources and revisions so their configuration
types match the provider's types.

The crate requires `std` and the same minimum Rust version as rscrypto.
Each algorithm feature forwards to the corresponding rscrypto leaf feature.
Default features enable every TLS 1.3 suite, key-exchange group, and classical
signature algorithm; `ml-dsa` and the composite `mldsa65-ed25519` and `mldsa65-ecdsa-p256` are opt-in.
`portable-only` selects rscrypto's portable dispatch for validation.

rustls provides protocol composition, rustls-webpki provides certificate path
validation, and rustls-pki-types provides credential types. Their built-in crypto
backends are disabled. The provider's normal and build dependency graph must
exclude ring, AWS-LC, OpenSSL, BoringSSL, and C build tools.

## Implementation inventory

| Operation | rscrypto implementation | Adapter |
| --- | --- | --- |
| Transcript hash | `Sha256`, `Sha384` | [hash](src/hash.rs) |
| HMAC | `HmacSha256`, `HmacSha384`; each call creates a keyed state | [hash](src/hash.rs) |
| TLS 1.2 PRF | RFC 5246 P_hash with a local HMAC state | [hash](src/hash.rs) |
| HKDF extract and expand | `HkdfSha256`, `HkdfSha384`; imported PRKs and multi-part info | [hash](src/hash.rs) |
| TLS 1.3 and TLS 1.2 records, QUIC packets | `Aes128Gcm`, `Aes256Gcm`, `ChaCha20Poly1305` | [aead](src/aead.rs) |
| QUIC header protection | `Aes128HeaderProtection`, `Aes256HeaderProtection`, `ChaCha20HeaderProtection` | [quic](src/quic.rs) |
| Stateless tickets | `Aes256Gcm`, fresh process-local keys, atomic counter nonces; rotation on a monotonic or caller clock | [ticket](src/ticket.rs) |
| Entropy | OS entropy through `getrandom::fill` with the `getrandom` feature, or a caller-selected `SecureRandom` without it; partial primitive inputs clear on failure | [random](src/random.rs), [kx_group](src/kx_group.rs), [sign](src/sign.rs), [client](src/client.rs) |
| Certificate and handshake verification | ECDSA P-256/P-384, Ed25519, RSA-PSS and RSA PKCS#1 v1.5 | [verify](src/verify.rs) |
| Key import and signing | rscrypto PKCS#8/SEC1/PKCS#1 codecs; blinded ECDSA, Ed25519, blinded RSA-PSS and RSA PKCS#1 v1.5 | [sign](src/sign.rs) |
| ML-DSA authentication | Pure ML-DSA-44/65/87 with the empty TLS context; keys prepared once (or compact on request) and hedged signing | [sign](src/sign/mldsa.rs), [verify](src/verify.rs) |
| Composite ML-DSA authentication | `mldsa65_ed25519_sha512` and `mldsa65_ecdsa_secp256r1_sha512` (draft-reddy-tls-composite-mldsa-12) at provisional codepoints, TLS 1.3 only; keys prepared once (or compact on request) and hedged signing | [sign](src/sign/composite.rs), [verify](src/verify.rs) |
| QUIC Initial keys and Retry | RFC 9001 AES-128-GCM Initial keys and Retry integrity tags | [quic](src/quic/initial.rs) |
| QUIC reset and address tokens | HMAC-SHA256 reset tokens; HKDF-SHA256 and AES-256-GCM token keys | [quic](src/quic/token.rs) |
| QUIC connection IDs | AES-128 connection-ID tags over caller-drawn nonces | [quic](src/quic/cid.rs) |
| Key exchange | `X25519SecretKey`, `P256EphemeralSecret`, `P384EphemeralSecret`, `MlKem768`, `MlKem1024` | [kx_group](src/kx_group.rs) |

`cipher_suite::ALL_CIPHER_SUITES` lists enabled TLS 1.3 suites in preference order.
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
also require `getrandom`. TLS 1.2 is opt-in; see [TLS 1.2](#tls-12).
The hybrid follows RFC 10024: ML-KEM precedes X25519 in both shares and the
shared secret. It validates the ML-KEM encapsulation key before server key
generation. A well-formed altered ciphertext retains ML-KEM's implicit rejection;
the TLS Finished check rejects the resulting different secret. The classical
component is available to rustls's fallback optimization.

Four more groups are opt-in; no provider constructor offers them, so add them to
a provider's `kx_groups` explicitly:

| Group | Constant / caller-entropy type | Features | Client / server share, secret (bytes) |
| --- | --- | --- | --- |
| SecP256r1MLKEM768 (RFC 10024) | `SECP256R1MLKEM768` / `P256MlKem768Group` | `p256-ecdh`, `ml-kem` | 1,249 / 1,153, 64 |
| SecP384r1MLKEM1024 (RFC 10024) | `SECP384R1MLKEM1024` / `P384MlKem1024Group` | `p384-ecdh`, `ml-kem` | 1,665 / 1,665, 80 |
| MLKEM768 (draft-ietf-tls-mlkem-11) | `MLKEM768` / `MlKem768Group` | `ml-kem` | 1,184 / 1,088, 32 |
| MLKEM1024 (draft-ietf-tls-mlkem-11) | `MLKEM1024` / `MlKem1024Group` | `ml-kem` | 1,568 / 1,568, 32 |

Unlike X25519MLKEM768, the two NIST-curve hybrids put the uncompressed ECDHE
share first in both shares and the ECDHE secret first in the shared secret. Each
group checks exact share lengths, validates the encapsulation key (FIPS 203
section 7.2) before server entropy, and validates the classical point; these
failures are `illegal_parameter`. Pure ML-KEM offers no classical component and
no fallback. Preference order and required-hybrid admission for these groups are
not decided; `post_quantum_required_provider()` still offers only X25519MLKEM768.
In-process tests agree with the pinned aws-lc-rs groups in both roles, with
rustls's hybrid composer over aws-lc-rs and *ring* components (the only
SecP384r1MLKEM1024 peers in the pinned rustls tree), and through full TLS 1.3
handshakes; `just test-provider-interop --group NAME` adds OpenSSL 4.0.3 for all
four and Go 1.27.3 for every group except MLKEM768.

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
entropy source; Ed25519 signing does not. RSA signing keys load with 2,048–8,192-bit moduli
and exponent 65,537, so servers and clients with RSA-2048 certificates work.
`sign::Provider::new(random).rsa_key_sizes(sign::RsaSigningKeySizes::Bits3072To8192)`
requires 3,072 bits or more. Certificate and handshake verification accept 2,048–8,192-bit RSA
peer keys whatever the signing policy. TLS 1.3 ECDSA signature
schemes match their exact curve and hash, and RSA PKCS#1 v1.5 is available for
certificate verification only. TLS 1.2 handshake signatures follow TLS 1.2's
rules; see [TLS 1.2](#tls-12).

Each loaded RSA key keeps idle signing workspaces, at most one per available CPU
and never more than 64. A workspace holds private-operation scratch and an
`RsaBlindingCache`, so a signature squares the previous blinding pair instead of
drawing, inverting, and exponentiating a new factor. The pair is redrawn every 32
signatures and after any error, as in AWS-LC, so factors within that epoch are
correlated. `sign::Provider::new(random).compact()` keeps no workspace; each RSA
signature then builds scratch and draws a new factor.

With `ml-dsa`, RFC 9881 PKCS#8 keys load and are prepared once. Provider tests
cover the seed and combined seed/key forms; the expanded form also loads through
the same rscrypto import. The prepared key keeps the transformed secrets and
public matrix (28,800, 48,256, or 81,024 bytes for ML-DSA-44, 65, or 87), and
the provider drops the compact key. Signing uses fresh randomness and the empty context required
by draft-ietf-tls-mldsa-06. Pure ML-DSA and HashML-DSA remain separate; HashML-DSA
keys and signatures do not enter this TLS profile. All three parameter sets
support all-ML-DSA certificate chains. Keys are shared by `Arc`, and concurrent
signers read the same prepared key. `sign::Provider::new(random).compact()` keeps
compact keys instead. Their secret encodings occupy 2,560, 4,032, or 4,896 bytes;
the compact owner also retains the public key, for 3,872, 5,984, or 7,488 bytes
before allocator rounding and provider owners. Each signature then prepares
temporary heap storage of the sizes above, so it expands the matrix once rather
than in every rejection attempt, and produces the same signatures more slowly.
Prepared secrets are cleared before their storage is freed.
The [resource evidence](../docs/provider-resource-evidence.md) records allocation
counts, shared and concurrent heap observations, cleanup checks, and unresolved
whole-call stack paths. Preparation remains bound to the configuration; peer
keys are not cached.

For scoped entropy, construct caller-owned static `kx_group::X25519Group`,
`P256Group`, `P384Group`, `X25519MlKem768Group`, or opt-in group values and `sign::Provider`
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

## Composite ML-DSA

With `mldsa65-ed25519` or `mldsa65-ecdsa-p256`, the provider signs and verifies the
composite ML-DSA schemes of draft-reddy-tls-composite-mldsa-12: `id-MLDSA65-Ed25519-SHA512`
and `id-MLDSA65-ECDSA-P256-SHA512` from draft-ietf-lamps-pq-composite-sigs-19 (RFC-to-be 10073),
for CertificateVerify and for certificate signatures in a webpki chain. Both component signatures
must verify. TLS uses the empty composite context; ML-DSA signing is hedged.

IANA has not assigned the codepoints, so they are provisional:
`verify::MLDSA65_ED25519_SHA512` is 0x090B, the value of draft revisions -00 through -05 that
deployed draft implementations use, and `verify::MLDSA65_ECDSA_SECP256R1_SHA512` is 0xFE40 in the
private-use range, because no draft assigned that composite a value. Both will change when IANA
assigns codepoints. Neither codepoint is a TLS 1.2 signature algorithm, so TLS 1.2 never
negotiates them, as section 2 of the draft requires.

Composite PKCS#8 keys load and are prepared once, keeping the prepared ML-DSA-65 key (48,256
bytes) and the traditional key; `sign::Provider::new(random).compact()` keeps the compact key and
prepares temporary storage per signature. A composite key signs only its composite scheme, never a
component scheme. A relying party that must not accept stand-alone ML-DSA or a traditional
signature installs `verify::COMPOSITE_ALGORITHMS` as its signature verification algorithms.

Provider tests verify the draft's reference-signed certificates, authenticate a client with one
through webpki, run server and mutual authentication and root → intermediate → leaf chains for
both composites, and check TLS 1.2 refusal, component-scheme refusal, and composite-only relying
parties.

## TLS 1.2

TLS 1.2 is opt-in: no provider constructor enables it. `cipher_suite::ALL_TLS12_CIPHER_SUITES`
lists six ECDHE suites in preference order: ECDSA before RSA, each with AES-256-GCM,
AES-128-GCM, then ChaCha20-Poly1305. There are no CBC, RSA key-transport, or
finite-field Diffie-Hellman suites. AES suites need `aes-gcm`, ChaCha20 needs
`chacha20poly1305`, and each suite needs `hmac` or `hkdf` for the PRF. Assign the
list to an owned provider's `tls12_cipher_suites`, as the suite list's rustdoc example
shows. `ClientProvider::provider()` returns an owned provider, so a caller-entropy
client opts in the same way; for client certificates, also set its `key_provider` to
a static `sign::Provider::new(random)`.

The provider's TLS 1.2 PRF derives a master secret only with the extended master
secret (RFC 7627). A client or server whose peer omits it aborts with
`handshake_failure`, as RFC 7627 §5.2 allows, so no session on this provider uses the
legacy derivation and `require_ems` needs no setting. With `require_ems = true`,
rustls refuses such a peer one message earlier. A client that also supports TLS 1.3
rejects a TLS 1.3 server's downgrade marker (RFC 8446 §4.1.3).

AES-GCM records use RFC 5288 nonces: the 4-byte salt and an 8-byte explicit part,
which is the key block's explicit bytes XOR the sequence number, as in rustls's other
providers. ChaCha20-Poly1305 records use RFC 7905 nonces. Record encryption reads
gathered plaintext directly into the output, as in TLS 1.3, and a record whose
plaintext would exceed 2^14 bytes is rejected before decryption. rustls derives the
extended master secret and key block through that PRF.

Key exchange uses X25519, P-256, or P-384. Hybrid groups are TLS 1.3 only, so
`post_quantum_required_provider()` with TLS 1.2 suites still refuses every TLS 1.2
handshake. QUIC rejects a provider with any TLS 1.2 suite.

RSA keys sign RSA-PSS when the peer offers it and PKCS#1 v1.5 otherwise, for example
for a TLS 1.2 server that accepts only PKCS#1 v1.5 client certificates. TLS 1.3 never
signs PKCS#1 v1.5. Each ECDSA key signs with one scheme in every TLS version: P-256
with SHA-256, and P-384 with SHA-384, as ring and AWS-LC keys do. A TLS 1.2 peer that
offers only the other ECDSA hash gets no signature from such a key. Verification
follows TLS 1.2's rule that an ECDSA scheme names a hash, not a curve: for example, an
`ecdsa_secp384r1_sha384` signature from a P-256 key, which Go servers produce,
verifies in TLS 1.2. TLS 1.3 verification stays curve-bound.

Tests run both roles in memory against ring and AWS-LC: all six suites, the three
groups, session-ID and stateless-ticket resumption, mutual authentication, PKCS#1
v1.5-only peers, the downgrade marker, and a stripped extended-master-secret request
or acknowledgement in either role with or without `require_ems`. Record tests compare
every byte with both peers' TLS 1.2 record layers. `just test-provider-interop
--protocol tls12` runs the suites against the pinned Go and OpenSSL peers.

`just test-provider-bogo` runs the pinned runner's declared TLS 1.2/1.3 profile, including independent EMS refusal cases.
The shim opts into TLS 1.2; provider constructors retain their TLS 1.3 defaults.
See the [campaign scope](../scripts/README.md#test-entry-points) for exclusions and diagnostic runs.

## QUIC v1

With `aes-gcm` and `hkdf`, the `quic` module supplies the QUIC v1 cryptography
that rustls's QUIC API leaves to the transport, without `getrandom`:

- `quic::initial_keys(client_dst_connection_id, side)` derives Initial packet
  and header keys as rustls `quic::Keys`.
- `quic::retry_integrity_tag` and `quic::is_valid_retry` compute and check Retry
  integrity tags.
- `quic::ConnectionIdKey` issues eight-byte connection IDs, a three-byte nonce
  and a five-byte AES-128 tag, and recognizes the IDs it issued. IDs carry no
  counter or endpoint index. The 24-bit nonce repeats after a few thousand IDs,
  so the transport rejects an issued ID that is already in use.
- `quic::StatelessResetKey` derives each connection ID's stateless-reset token
  from a static HMAC-SHA256 key.
- `quic::AddressTokenKey` seals and opens Retry and NEW_TOKEN address tokens of
  up to 4 KiB of plaintext. Each token carries a fresh 16-byte nonce, from which
  HKDF-SHA256 derives a one-use AES-256-GCM key. The transport owns the token
  contents, expiry, and replay policy.

Every key and nonce comes from the `SecureRandom` source passed to the call. A
failing source fails the call, without OS fallback, and leaves caller data
unchanged. Rustls's QUIC API supplies the handshake and the packet and header
protection keys through this provider's `quic` algorithms.

The crate stays unpublished while it depends on an unreleased rustls revision.
rscrypto keeps its `v<version>` tags. Once this crate can be published, its tags
will be `rscrypto-rustls-v<version>`.
