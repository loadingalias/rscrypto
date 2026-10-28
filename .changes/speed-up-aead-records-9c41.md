---
"rscrypto" = "patch"
---

Speed up AES-128-GCM, AES-256-GCM, and ChaCha20-Poly1305 for short messages, message tails, and
TLS-record-shaped gathers on x86-64. With VAES and AVX-512, one finishing pass encrypts the final
blocks together with the tag block and folds GHASH with one reduction, so short records, the end
of long records, and the short second record of a 16 KiB TLS write no longer take single-block
paths. With AVX-512, ChaCha20-Poly1305 runs owned kernels at every length, and
the ChaCha20 keystream's final blocks no longer fall back to four-way or portable code. Short
opens check the tag before writing plaintext. The new x86-64 workers scrub the stack they spill to.
