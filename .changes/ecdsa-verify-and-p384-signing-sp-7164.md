---
"rscrypto" = "patch"
---

ECDSA verification adds the generator term into the key chain, so it needs no doublings of its own: on an Apple M1 Pro, P-256 verification with a prepared key runs 11% faster and P-384 14% faster. On x86-64 CPUs with BMI2 and ADX, P-256 verification multiplies and squares with owned MULX assembly. On x86-64 Linux and Windows, ECDSA P-384 signing and key derivation run the owned P-384 ECDH fixed-base kernel instead of the generic comb.
