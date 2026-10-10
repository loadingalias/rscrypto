---
"rscrypto" = "minor"
---

Add SLH-DSA (FIPS 205) behind the new `slh-dsa` leaf feature, also enabled by `signatures`. All 12 parameter sets ship, each as a pure profile such as `SlhDsaSha2_128s` and an RFC 9909 HashSLH-DSA profile such as `HashSlhDsaSha2_128sWithSha256`, whose keys are distinct types that never sign or verify for the other profile. Signing is hedged with caller randomness or deterministic, takes a context of up to 255 bytes, and writes into a caller-provided buffer; a failed call zero-fills it. Verification takes the signature as a byte slice. Secret-key import regenerates the public root, and keys have RFC 9909 SubjectPublicKeyInfo and PKCS #8 import and export. Nothing allocates. Failures use the new `SlhDsaError` and `SlhDsaKeyError`.
