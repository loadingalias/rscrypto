---
"rscrypto" = "patch"
---

Clear the dead stack below ML-DSA's private SHAKE256 helper after each call.
This removes Keccak lane spill copies that the compiler creates and
that named state wipes cannot reach.
Capability detection now runs before the helper.
The first call in a process therefore no longer initializes the detection cache below the cleared
region.
Outputs and public APIs do not change.
