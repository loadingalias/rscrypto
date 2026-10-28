---
"rscrypto" = "minor"
---

Breaking: RsaPublicKeyPolicy's constructors now name the moduli they accept, because the policy also gates private-key import and key generation, not only verification. legacy_verification() and LEGACY_VERIFICATION become modulus_2048_to_8192() and MODULUS_2048_TO_8192; modern_verification() and MODERN_VERIFICATION become modulus_3072_to_8192() and MODULUS_3072_TO_8192. The accepted moduli, the exponent rule, and Default (3,072 to 8,192 bits) are unchanged; rename the calls.
