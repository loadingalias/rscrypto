---
"rscrypto" = "patch"
---

Speed up Blake3::update_reader and Blake3Subtree::update_reader, which no longer allocate, zero, and clear a 1 MiB buffer on every call. Reads start at 1 KiB and grow as the reader fills them, storage is zeroed only as reads need it, and inputs up to 64 KiB use stack storage without allocating. Serial hashing uses 16 KiB buffers at first and grows to 1 MiB reads for long inputs and files; hashers that may use threads still receive whole 1 MiB buffers. Every byte passed to the reader is still cleared before release, including bytes written by a reader that then fails. Outputs, returned counts, and error behavior are unchanged.
