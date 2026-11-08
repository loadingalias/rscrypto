---
"rscrypto" = "patch"
---

Speed up AES-128-GCM-SIV and AES-256-GCM-SIV on x86-64 CPUs with VAES and VPCLMULQDQ. Sealing and opening
now run key derivation, the key schedule, POLYVAL, and CTR in one fused path for every message length, and
derive only the hash powers each length needs. On Intel Granite Rapids and AMD Zen 5, messages from 64 B to
1 KiB take 36–60% less time and longer messages 7–30% less. Opening below 64 B takes 51–64% less time, and
AES-256 sealing below 64 B 52–62% less; AES-128 sealing below 64 B, which already had a fused path, stays
within 9% of its previous time.
