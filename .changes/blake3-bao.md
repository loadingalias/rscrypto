---
"rscrypto" = "minor"
---

Add a bounded Bao combined-stream decoder under `hashes::expert::bao` with
`std`. Verify each chunk before returning it and reject corrupt or truncated
encodings against an independently trusted BLAKE3 root.
