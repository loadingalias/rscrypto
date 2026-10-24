---
"rscrypto" = "patch"
---

Speed up RSA public-key operations with an imported key. A one-time operation enters the Montgomery domain
with one long division by the modulus instead of computing R² mod n, and each public-key operation leaves it
with its final multiplication instead of a separate reduction. Verification with a newly imported key costs
about a third less on x86-64 and AArch64. RsaPublicKey::public_scratch computes R² mod n once with two such
divisions, and RsaPublicScratch::new now only allocates. Results and errors are unchanged.
