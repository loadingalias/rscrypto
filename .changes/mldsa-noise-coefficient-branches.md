---
"rscrypto" = "patch"
---

Remove secret-coefficient branches from ML-DSA secret-noise coefficient mapping in IBM Z release
builds.
The eta=2 reduction now uses a multiply-shift without comparisons.
The final modular correction uses the shared register-barrier reduction.
Outputs, sampling order, and public APIs do not change.
