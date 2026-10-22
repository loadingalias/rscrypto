---
"rscrypto" = "patch"
---

Restructure AES-GCM key setup. Constructing an AES-128-GCM or AES-256-GCM key computes its GHASH powers once
and takes every window table from them instead of building each table serially. On x86-64 and on macOS and
Linux AArch64, the powers above H^8 come from eight independent chains. Ciphertexts and tags are unchanged.
