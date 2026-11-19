---
"rscrypto" = "minor"
---

Add `HmacSha256::finalize_into` and `HmacSha384::finalize_into`. Each writes the current tag into a caller-owned
array without consuming or resetting the state, so secret-derived output, such as key material, never becomes a
copyable tag value. The caller owns the array and its cleanup.

Initialize keyed outer hash states in place during SHA-2 HMAC finalization.
