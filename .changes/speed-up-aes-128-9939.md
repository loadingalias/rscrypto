---
"rscrypto" = "patch"
---

Speed up AES-128-GCM and AES-256-GCM key setup on x86-64. With AVX-512 and VPCLMULQDQ, the GHASH
key powers are computed four at a time; other x86-64 CPUs with PCLMULQDQ use a balanced recurrence
with a shorter dependency chain than the previous eight chains. Each context keeps one
64-byte-aligned power table, 128 powers on Linux and 16 on other x86-64 targets, instead of six
separate tables, and clears it on drop. On Granite Rapids, constructing a key and encrypting 64
bytes takes about 54% less time, and resumed TLS 1.2 AES-GCM handshakes take 9 to 12% less.
