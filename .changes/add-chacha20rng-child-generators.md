---
"rscrypto" = "minor"
---

Add ChaCha20Rng::child, which derives a generator from this generator's seed, an 8-byte label, and a 64-bit index with one RFC 8439 block under a branch key. A child does not depend on the parent's draws, so one recorded root seed and each owner's path replay every stream. The stream for each seed does not change. A party that learns a generator's state learns every child seed. Drop clears the branch key.
