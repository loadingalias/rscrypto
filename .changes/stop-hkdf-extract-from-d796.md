---
"rscrypto" = "patch"
---

Stop HKDF-Extract from leaving secret copies on the stack. `HkdfSha512::extract` left a copy of its pseudorandom key (PRK), and extract for every hash left copies of the HMAC prefix states, which are as sensitive as the PRK. Extract now writes the HMAC tag into a PRK copy that it clears, and the prefix setup, which `from_prk` shares, gives the compression function only scratch that it clears. Outputs are unchanged.
