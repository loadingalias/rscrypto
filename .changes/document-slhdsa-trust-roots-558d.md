---
"rscrypto" = "patch"
---

Add the SLH-DSA guide, `docs/slhdsa.md`, and the `slhdsa_update_verify` example. The guide names SLH-DSA-SHA2-128s as the profile for trust roots and update keys, and SLH-DSA-SHA2-192s for security category 3, and lists the sizes, the API flow, the evidence, and the open limits of each set. The example pins a root key as its RFC 9909 SubjectPublicKeyInfo, signs an update manifest under a context, and verifies it from the pinned DER, as a device does. SLH-DSA-SHA2-128s key generation, signing, and verification now have stack bounds on the five Linux CI targets. Native residue runs on macOS AArch64 and on Linux x86-64 and AArch64 find no derived secret after each call. `ct.toml` adds DudeCT cases for SHA2-128s key generation and deterministic signing. SLH-DSA keeps its best-effort entry and has no timing claim. Keys, signatures, and the API are unchanged.
