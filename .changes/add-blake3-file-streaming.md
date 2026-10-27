---
"rscrypto" = "minor"
---

Add `Blake3::update_file` to hash from an existing file's current position through EOF with advisory, bounded file buffering. It preserves incremental hash state and read-error behavior, uses stack input storage for small files, and clears owned input storage. Generic `update_reader` buffering remains available for other readers.
