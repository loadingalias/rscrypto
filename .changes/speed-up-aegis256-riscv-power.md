---
"rscrypto" = "patch"
---

Speed up AEGIS-256 on RISC-V CPUs without AES extensions and on POWER. On RISC-V, each state update now
bitslices its round inputs straight from the state instead of copying blocks through intermediate
arrays: sealing and opening take 54–62% less time at every message size. On POWER, the AES round is
scheduled with the surrounding state updates, which takes 1–4% less time.
