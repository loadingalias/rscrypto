---
"rscrypto" = "patch"
---

Document that a RapidHash seed is not a key. `RapidHash64::hash_with_seed`, `RapidStreamHasher::with_seed`, and
`RapidSeededState` keep RapidHash's public secrets under every seed, so families of keys share one hash under every
seed, even a secret one. Use `RapidRandomState` for keys an attacker can choose. `FastHash::hash_with_seed` now
states that a secret seed does not prevent chosen-input collisions unless an implementation documents otherwise.
Hashes are unchanged.
