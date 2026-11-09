---
"rscrypto" = "patch"
---

Speed up AES-GCM and AES-GCM-SIV on RISC-V CPUs without AES and carryless-multiply extensions, and on
POWER. On RISC-V, GHASH and long POLYVAL transcripts share one reduction per four blocks, the generic CTR
path clears 3 KiB of stack in one pass instead of 8 KiB in two, and short CTR tails share one AES call. On
a T-Head C910, AES-GCM takes 12–42% less time at every measured size, and AES-GCM-SIV messages of 1 KiB
and more take 2–17% less. On POWER10, opening AES-GCM-SIV messages of 256 B and more takes 33–61% less
time, and AES-GCM messages of one to three blocks take 16–40% less.
