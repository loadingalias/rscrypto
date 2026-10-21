---
"rscrypto" = "patch"
---

Restructure AES-GCM key setup. Constructing an AES-128-GCM or AES-256-GCM key computes its largest GHASH power
table once, in eight independent chains, and takes every smaller table from it instead of building each table
serially. Ciphertexts and tags are unchanged.
