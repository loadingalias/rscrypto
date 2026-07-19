---
"rscrypto" = "minor"
---

RSA JWT/JWS verification is now bound to one verifier-owned
`RsaJwtAlgorithm`. Peer-controlled `alg` metadata can only match that fixed
policy; string-to-profile helpers and runtime algorithm-name signing and
verification APIs have been removed.
