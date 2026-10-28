---
"rscrypto" = "minor"
---

Add the optional rand_core feature: ChaCha20Rng implements rand_core 0.10's TryRng, TryCryptoRng, and SeedableRng, so rand and other rand_core consumers use it directly. The trait methods return the inherent stream, from_rng and fork write the seed straight into the new generator, and seed_from_u64 panics because a 64-bit value is not a 256-bit seed. The feature enables chacha20-rng and adds only the rand_core dependency.
