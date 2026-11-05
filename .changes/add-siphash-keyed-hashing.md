---
"rscrypto" = "minor"
---

Add SipHash-1-3 and SipHash-2-4 keyed hashing behind the `siphash` feature, part of `macs`. SipHashKey holds the 128-bit key and clears it on drop. SipHash13 and SipHash24 hash one byte string or act as a core::hash::Hasher. SipRandomState is a BuildHasher for HashMap and HashSet that keys SipHash-1-3 from caller entropy (try_new_with) or the operating system (try_new), so tables keyed by attacker-chosen values, such as QUIC connection IDs, resist collision flooding.
