---
"rscrypto" = "patch"
---

Speed up ML-DSA signing. The constant-time SampleInBall scans for each challenge candidate and its placement
now run with AVX2 on Linux x86-64 and NEON on AArch64 instead of one element at a time. ML-DSA-65 signing is
about 20% faster on Intel Granite Rapids and AWS Graviton4, and an ML-DSA-65 TLS 1.3 handshake about 12%.
Signatures are unchanged.
