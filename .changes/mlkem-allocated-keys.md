---
"rscrypto" = "minor"
---

Add ML-KEM `generate_keypair_in`, `try_generate_keypair_in`,
`DecapsulationKey::try_from_slice_in`, and `DecapsulationKey::prepare_in`. With
`alloc`, they take an `Allocator` and write the decapsulation key or prepared
key directly into its `Box`, so no by-value copy of the secret is left behind.
Measured on QEMU RV32 and Cortex-M, generating, importing, or preparing a key by
value leaves up to a full secret-key copy in the caller's dead stack; the new
constructors leave none on the stack or in the freed allocation. The by-value
constructors remain for core-only use.
