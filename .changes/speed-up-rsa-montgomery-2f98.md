---
"rscrypto" = "patch"
---

Speed up RSA Montgomery setup and private operations. `R² mod n` now takes six Montgomery squarings instead of
about 64 modular doublings per limb, and new scratch copies a key's precomputed value. Verifying with a newly
imported RSA-2048 key drops from 123 µs to 20 µs on Intel Granite Rapids, and creating private-key scratch no
longer repeats the computation. Private operations reduce their CRT inputs with one Montgomery reduction instead
of a bit-serial reduction, so RSA-3072 and RSA-4096 signing is about 20% faster. Results are unchanged.
