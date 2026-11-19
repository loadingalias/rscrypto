---
"rscrypto" = "patch"
---

Speed up AES-GCM and AES-GCM-SIV on RISC-V CPUs without AES and carryless-multiply extensions, and on
POWER. On RISC-V, GHASH and long POLYVAL transcripts share one reduction per four blocks, and short CTR
tails share one AES call. On a T-Head C910, AES-GCM at 4 KiB and 1 MiB and AES-GCM-SIV at 16 KiB now run
1.03–1.09x as fast as RustCrypto; AES-GCM messages of up to 256 B still take about 1 µs longer. On POWER10,
opening AES-GCM-SIV messages of 256 B and more takes 33–61% less time, and AES-GCM messages of one to three
blocks take 16–40% less.
