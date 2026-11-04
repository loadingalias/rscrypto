---
"rscrypto" = "patch"
---

Speed up XXH3-64 and XXH3-128 for inputs of 4 KiB and more on x86-64: about 1.3–1.7x on Intel
CPUs with AVX-512 and 1.3–2.3x on AMD Zen 5. The AVX-512 and AVX2 kernels keep each stripe's term
off the accumulator, sum each block's stripes into independent partial sums, and step through
blocks without per-stripe checked offset arithmetic.
