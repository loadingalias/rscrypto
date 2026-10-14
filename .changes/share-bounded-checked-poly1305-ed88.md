---
"rscrypto" = "patch"
---

Standalone Poly1305 and ChaCha20-Poly1305 now share one bounded, checked arithmetic core, and each feature still builds alone. Bulk throughput is unchanged on Graviton4; 15-byte standalone Poly1305 authentication is about 9% slower there.
