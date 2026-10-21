---
"rscrypto" = "patch"
---

Add EcdsaP256PublicKey::verify_sec1 and verify_sec1_sha384, which verify with an uncompressed SEC1 key used once. They reach the same decisions as from_sec1_bytes followed by verify or verify_sha384, without building the verification table that a parsed key keeps for repeated use. P-256 verification on AArch64 also uses assembly field addition and subtraction.
