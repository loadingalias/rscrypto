---
"rscrypto" = "patch"
---
Build `portable-only` on POWER, IBM Z, RV64, and RV32 with the declared MSRV by excluding nightly-only backends. Preserve portable fallbacks and check these target and feature combinations in the compatibility lane.
