# rscrypto-rustls

This unpublished crate is the workspace skeleton for an rscrypto-backed rustls
TLS 1.3 provider. It does not yet expose adapters or provider constructors.

The crate requires `std` and the same minimum Rust version as rscrypto.
Each algorithm feature forwards to the corresponding rscrypto leaf feature.
The default set covers the planned first TLS profile; `ml-dsa` is opt-in.
`portable-only` selects rscrypto's portable dispatch for validation.

rustls provides protocol composition, rustls-webpki provides certificate path
validation, and rustls-pki-types provides credential types. Their built-in crypto
backends are disabled. The provider's normal and build dependency graph must
exclude ring, AWS-LC, OpenSSL, BoringSSL, and C build tools.

rscrypto retains its `v<version>` tags. This crate uses
`rscrypto-rustls-v<version>` tags when it becomes publishable, after rscrypto v0.11.
