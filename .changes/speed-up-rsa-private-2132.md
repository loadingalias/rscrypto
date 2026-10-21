---
"rscrypto" = "patch"
---

Speed up RSA private operations on AArch64. The constant-time fixed-window table selection merges each entry under vector-compare masks in NEON assembly instead of scalar volatile reads; it still reads every entry in a fixed order, and its results are unchanged.
