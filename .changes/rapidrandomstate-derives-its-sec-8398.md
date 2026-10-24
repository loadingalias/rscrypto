---
"rscrypto" = "patch"
---

`RapidRandomState` derives its secret schedule once, when the state is created, instead of on every byte-slice
hash, so a hash no longer pays six serial derivation steps before it reads its input. The state grows from 16 to
64 bytes. Hashes are unchanged.
