---
"rscrypto" = "patch"
---

Reduce AArch64 AES-GCM setup work and HMAC-SHA256/SHA384 finalization copies, and use a stack-free SHA-512 hardware kernel on little-endian AArch64.
