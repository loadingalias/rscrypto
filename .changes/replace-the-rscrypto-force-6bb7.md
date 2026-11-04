---
"rscrypto" = "minor"
---

Replace the `RSCRYPTO_FORCE_AVX512` environment variable with
`platform::expert::allow_hybrid_avx512`. On Intel hybrid CPUs (Alder Lake through Arrow Lake),
detection removes AVX-512 and AVX10 even when CPUID reports them, because the E-cores lack them. A
process whose every core implements AVX-512, for example with the E-cores disabled in firmware, now
keeps them by calling the `unsafe` `allow_hybrid_avx512()` before the first `platform::caps()` call.
Setting `RSCRYPTO_FORCE_AVX512` no longer has any effect, so such systems keep AVX-512 masked until
they make that call. x86-64 capability detection no longer reads the environment or allocates.
