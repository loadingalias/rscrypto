---
"rscrypto" = "minor"
---

Add `Blake3XofReader::position` and `Blake3XofReader::set_position` for random
access into BLAKE3 output. Seeking moves to any byte offset of the output stream,
forward or backward, without producing the bytes before it, and matches the
upstream `blake3` crate's `OutputReader::set_position` in hash, keyed, and
derive-key modes.
