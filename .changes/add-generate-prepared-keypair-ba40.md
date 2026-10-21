---
"rscrypto" = "patch"
---

Add generate_prepared_keypair_in to MlKem512, MlKem768, and MlKem1024. It returns the encapsulation key and a prepared decapsulation key built from the same matrix expansion, equal to generate_keypair_in followed by prepare_in without expanding the matrix or hashing the encapsulation key again. Use it for ephemeral key shares that decapsulate once.
