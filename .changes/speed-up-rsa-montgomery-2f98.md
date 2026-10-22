---
"rscrypto" = "patch"
---

Speed up RSA Montgomery setup and private operations. `R² mod n` now takes about one modular doubling per
limb and six Montgomery squarings instead of about 64 doublings per limb, and new scratch copies a key's
precomputed value. Verifying with a newly imported RSA-2048 key drops from 123 µs to 20 µs on Intel Granite
Rapids, and creating private-key scratch no longer repeats the computation. Private operations reduce their
CRT inputs with one Montgomery reduction instead of a bit-serial reduction, so RSA-3072 and RSA-4096 signing
is about 20% faster. On Linux x86-64 with BMI2 and ADX, private exponentiation squares with the dedicated
square kernel, a further 15–22% for signing on Granite Rapids before the AVX-512 IFMA path below.
The per-operation blinding-factor inverse runs batches of 62 Bernstein–Yang divsteps on every target except
RISC-V and IBM Z, a further 5–10% for signing. On x86-64 with AVX-512 IFMA, 2048-, 3072- and 4096-bit
private operations run both CRT exponentiations together in radix 2^52. On Intel Granite Rapids, RSA-4096,
RSA-3072 and RSA-2048 signing take 66%, 61% and 49% less time. Results are unchanged.

Blinded private operations invert the blinding factor multiplied by a fresh random mask, then multiply the
result by the mask, so the inverse never reads the factor itself on any target. Native timing tests on
Graviton4 and Apple Silicon separated a fixed factor from random ones in the unmasked 62-bit inverse.
Each blinding attempt with a valid candidate also requests a modulus-width mask, redrawn while it is zero or
not below the modulus, so the `sign_*_with_random_fill*` methods call their entropy callback at least once
more per attempt. A failed mask request returns `EntropyUnavailable`.
