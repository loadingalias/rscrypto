---
"rscrypto" = "minor"
---

Add `Blake3XofReader::position` and `Blake3XofReader::set_position` for random access into BLAKE3 output.
A seek moves to any byte offset of the output stream, forward or backward,
without producing the bytes before it.
It matches `OutputReader::set_position` of the upstream `blake3` crate in hash, keyed, and derive-key modes.
