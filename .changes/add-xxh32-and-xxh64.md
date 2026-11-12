---
"rscrypto" = "minor"
---

Add XXH32 and XXH64, the original xxHash algorithms that LZ4 and Zstandard frames use for checksums, behind the new `xxh32` and `xxh64` leaf features, part of `fast-hashes`. `Xxh32` and `Xxh64` implement `FastHash` with 32-bit and 64-bit seeds and outputs. `Xxh32Hasher` and `Xxh64Hasher` hash incrementally in a fixed-size state without `alloc` or `std`, and `reset` keeps their seed. Both are portable Rust and non-cryptographic: a frame checksum detects accidental corruption, but gives no authentication and no flood resistance. XXH3 is unchanged.
