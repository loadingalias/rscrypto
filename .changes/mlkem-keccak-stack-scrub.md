---
"rscrypto" = "patch"
---

Clear the dead stack below ML-KEM's secret SHA-3 and SHAKE calls (G, J, and the
PRF) after each call, as ML-DSA does. This removes compiler-created Keccak lane
spill copies of seeds, messages, and the implicit-rejection secret that named
state wipes cannot reach. Outputs and public APIs are unchanged, and public
SHA-3 and SHAKE hashing is not scrubbed.
