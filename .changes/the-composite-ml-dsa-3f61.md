---
"rscrypto-rustls" = "patch"
---

The composite ML-DSA loader imports each key directly into its heap storage with from_pkcs8_der_in, not by value. Compact mode no longer moves the imported secret from the stack into a box, and prepared mode keeps the prepared key in its own storage box without a second box.
