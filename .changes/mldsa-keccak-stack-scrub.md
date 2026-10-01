---
"rscrypto" = "patch"
---

Clear the dead stack below ML-DSA's private SHAKE256 helper after each call.
This removes compiler-created Keccak lane spill copies that named state wipes
cannot reach. Capability detection runs before the helper, so the first call in
a process no longer initializes the detection cache below the cleared region.
Outputs and public APIs are unchanged.
