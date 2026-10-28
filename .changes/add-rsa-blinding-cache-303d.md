---
"rscrypto" = "minor"
---

Add the opt-in `RsaBlindingCache` for RSA signing.
`RsaPrivateKey::sign_signature_with_random_fill_and_blinding_cache`, which takes caller entropy,
and `sign_signature_with_blinding_cache`, which uses `getrandom`, reuse one blinding pair for up
to 32 signatures and square it before each reuse, as AWS-LC does, instead of drawing, inverting,
and exponentiating a new factor for every signature. Factors within an epoch are therefore
correlated. Any signing error, a different key, or, on Unix with `std`, a changed process ID draws
a new factor. The cache is not `Clone`, shows only its width and use count in `Debug`, and
zeroizes its pair on error and drop.
