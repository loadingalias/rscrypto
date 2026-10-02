---
"rscrypto" = "patch"
---

Clear the dead stack below ML-KEM's secret SHA-3 and SHAKE calls
(G, J, and the PRF) after each call, as ML-DSA does.
This removes Keccak lane spill copies of seeds, messages,
and the implicit-rejection secret that the compiler creates and that named state wipes cannot reach.
Outputs and public APIs do not change.
Public SHA-3 and SHAKE hashing is not scrubbed.
