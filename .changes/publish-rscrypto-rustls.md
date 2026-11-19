---
"rscrypto-rustls" = "minor"
---

Publish `rscrypto-rustls` on crates.io: a rustls cryptographic provider for TLS 1.3, opt-in TLS 1.2,
and QUIC v1 endpoint keys, backed by rscrypto. It pins rustls `=0.24.0-dev.3`, the crates.io
development release, with rustls-webpki `=0.104.0-alpha.8` and rustls-pki-types `=1.16.0`. A later
release moves the pin to rustls 0.24.0. Depend on the same rustls version, so that your configuration
types match the provider's types.
