---
"rscrypto" = "minor"
---

Add Adler-32 from RFC 1950, the checksum of zlib streams and HTTP `deflate`, behind the new `adler32` leaf feature, part of `checksums`. `Adler32` implements `Checksum`: one-shot `checksum`, incremental `update`, `finalize`, `reset` to the standard initial value, and `with_initial` to resume from an earlier checksum. It implements `ChecksumCombine` with zlib's `adler32_combine` semantics for every length. It is portable Rust without `alloc` or `std` and keeps no input. It is non-cryptographic: it detects accidental corruption and gives no authentication. CRC outputs are unchanged.
