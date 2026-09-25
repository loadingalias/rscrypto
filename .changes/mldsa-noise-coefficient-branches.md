---
"rscrypto" = "patch"
---

Remove secret-coefficient branches from ML-DSA secret-noise coefficient mapping
in IBM Z release builds. The eta=2 reduction now uses a comparison-free
multiply-shift, and the final modular correction uses the shared register-barrier
reduction. Outputs, sampling order, and public APIs are unchanged.
