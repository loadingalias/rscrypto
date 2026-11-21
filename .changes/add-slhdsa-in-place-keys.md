---
"rscrypto" = "minor"
---

Add in-place SLH-DSA secret-key constructors behind `alloc`, for every pure and HashSLH-DSA profile: `generate_keypair_in`, `try_from_slice_in`, and `from_pkcs8_der_in`. Each builds the key in a box from the caller's allocator, so moving the key moves only a pointer and leaves no copy of SK.seed or SK.prf. The checks and errors match `generate_keypair`, `try_from_slice`, and `from_pkcs8_der`.
