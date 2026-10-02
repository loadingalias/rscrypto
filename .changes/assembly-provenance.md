---
"rscrypto" = "patch"
---

The package now pins the upstream source and SHA-256 of the ChaCha20-Poly1305 and BLAKE3 x86-64 assembly that
derives from AWS-LC and BLAKE3, as it already did for the RSA and signature assembly.
The x86-64 ChaCha20-Poly1305 assembly now carries the CloudFlare copyright notice of its upstream source.
