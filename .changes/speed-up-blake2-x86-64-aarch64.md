---
"rscrypto" = "patch"
---

Speed up BLAKE2b and BLAKE2s on x86-64 and AArch64. On AMD Zen 5, runtime dispatch now selects the
scalar kernel, which is about twice as fast as the vector kernels there, including builds that
enable AVX2 or AVX-512 at compile time. On Intel CPUs with AVX-512VL, BLAKE2b is 16–20% faster and
BLAKE2s 12–20% faster: the rounds are unrolled, message vectors come from vector loads instead of
gathers, and the diagonal step no longer adds a permute to the dependency chain. On non-Apple
AArch64, the scalar rounds are 7–9% faster: each XOR-and-rotate step puts one shifted-register
`EOR` on the dependency chain instead of two instructions.
