---
"rscrypto-rustls" = "patch"
---

TLS 1.3 clients can use Encrypted Client Hello with the RFC 9849 X25519, HKDF-SHA256, and AES-128-GCM suite.
Pass the server configuration list and `hpke::ALL_SUPPORTED_SUITES` to rustls.
