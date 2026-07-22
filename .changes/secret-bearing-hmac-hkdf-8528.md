---
"rscrypto" = "minor"
---

Secret-bearing HMAC, HKDF, KMAC, and PBKDF2 states no longer implement
`Clone`. Keyed BLAKE2 parameter builders and the variable-output Blake2b state
also no longer do, because they can retain a MAC key. The `Mac` trait no longer
requires `Clone`; reuse or share one owner, or construct another keyed state
explicitly.

Secret-bearing SHA/HMAC, Ed25519, Keccak, BLAKE2, and BLAKE3 temporaries now
clear at their actual finalization, reset, transfer, or drop boundary, including
keyed XOF and parallel heap scratch. The optimized zeroization gate covers
success, error, early-return, move, reuse, and drop shapes. Secret-key hex and
generic ECDSA random-source errors no longer echo input or payload bytes through
`Debug`, `Display`, or the standard error-source chain. Their public fields and
variants still allow explicit recovery when callers deliberately inspect them.
