---
"rscrypto" = "patch"
---
Allow `Blake3::digest_const` to hash inputs longer than one chunk using portable compression and a fixed tree stack, without allocation.
