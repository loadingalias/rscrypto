# rscrypto-rustls

This unpublished crate supplies rscrypto-backed TLS 1.3 and QUIC cryptographic
adapters for rustls 0.23.45. Provider constructors follow with key exchange and
authentication.

The crate requires `std` and the same minimum Rust version as rscrypto.
Each algorithm feature forwards to the corresponding rscrypto leaf feature.
The default set covers the first TLS profile; `ml-dsa` is opt-in.
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
| HKDF extract and expand | `HkdfSha256`, `HkdfSha384`; imported PRKs and multi-part info | [hash](src/hash.rs) |
| TLS records and QUIC packets | `Aes128Gcm`, `Aes256Gcm`, `ChaCha20Poly1305` | [aead](src/aead.rs) |
| QUIC header protection | `Aes128HeaderProtection`, `Aes256HeaderProtection`, `ChaCha20HeaderProtection` | [quic](src/quic.rs) |
| Entropy | OS entropy through `getrandom::fill`; partial output clears on failure | [random](src/random.rs) |

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

rscrypto retains its `v<version>` tags. This crate uses
`rscrypto-rustls-v<version>` tags when it becomes publishable, after rscrypto v0.11.
