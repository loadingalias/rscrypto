---
"rscrypto-rustls" = "patch"
---

Prepared client and server verifiers now enforce the certificate and handshake algorithms passed to `with_algorithms`. Rustls validates these chains directly. Use `new` for prepared roots with the default catalog. Use `with_tls_signature_mapping` to keep prepared roots and restrict only handshake signatures.
