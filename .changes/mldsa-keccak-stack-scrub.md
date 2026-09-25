---
"rscrypto" = "patch"
---

Clear the dead stack below ML-DSA's private SHAKE256 helper after each call.
This removes compiler-created Keccak lane spill copies that named state wipes
cannot reach. Outputs and public APIs are unchanged.
