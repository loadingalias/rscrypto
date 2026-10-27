---
"rscrypto" = "patch"
---

Align the cached wide GHASH tables in AES-128-GCM on Linux x86-64 to improve native record
encryption. The context retains the same authentication data and clears both tables on drop.
