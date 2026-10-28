---
"rscrypto" = "minor"
---

Add P-256/SHA-384 and P-384/SHA-256 ECDSA signing: EcdsaP256SecretKey and EcdsaP256Keypair gain try_sign_sha384 and try_sign_sha384_blinded_with, and EcdsaP384SecretKey and EcdsaP384Keypair gain try_sign_sha256 and try_sign_sha256_blinded_with, for protocols such as TLS 1.2 that name ECDSA's hash independently of the curve. They use the existing deterministic, blinding-capable construction, with nonces derived under their own domains so they never repeat a same-curve signature's nonce. The existing verify_sha384 and verify_sha256 verify them.
