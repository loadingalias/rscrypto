---
"rscrypto" = "minor"
---

Add the `p384-ecdh` feature with ephemeral P-384 Diffie-Hellman key agreement:
`P384EphemeralSecret`, `P384PublicKey`, `P384SharedSecret`, `P384KeyGenerationError`, and
`P384PublicKeyError`. Peer keys must be canonical uncompressed SEC1 points on the curve.
The `key-exchange` umbrella now enables `p384-ecdh`.
