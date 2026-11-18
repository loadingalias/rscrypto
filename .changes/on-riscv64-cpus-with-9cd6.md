---
"rscrypto" = "patch"
---

On riscv64 CPUs with Zbc or Zbkc but without Zvbc, AES-GCM, AES-GCM-SIV, GHASH, and POLYVAL now give correct results. The scalar carry-less multiply let the compiler place its low product in an input register that the high product still read. The Zvbc CRC folds had the same operand constraint, which this build did not miscompile; it is fixed too.
