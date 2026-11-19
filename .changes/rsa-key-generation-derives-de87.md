---
"rscrypto" = "patch"
---

RSA key generation derives lcm(p - 1, q - 1) and the private exponent with fixed-work arithmetic: a binary GCD and long division with fixed iteration counts, and a masked scan that inverts the public exponent. The previous Euclid and inverse search ran in time that depended on the secret primes.
