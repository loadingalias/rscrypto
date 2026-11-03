---
"rscrypto" = "patch"
---

Use the AVX2 and AVX-512 ML-DSA backends on Windows x86-64, as on Linux x86-64, after the same cached CPU and OS capability detection. On Granite Rapids under Windows, ML-DSA signing runs about 4x and verification about 3x faster than the portable arithmetic it replaces; key generation runs about 1.35x faster.
