---
"rscrypto" = "patch"
---

Add EcdsaP384PublicKey::verify_sec1 and verify_sec1_sha256, which verify with an uncompressed SEC1 key used once. They reach the same decisions as from_sec1_bytes followed by verify or verify_sha256, without building the verification table that a parsed key keeps for repeated use. P-384 verification and public-key import use P-384-specific field and point arithmetic, with Montgomery multiplication, squaring, addition, and subtraction in assembly on AArch64. Verification results and public-key tables are unchanged.
