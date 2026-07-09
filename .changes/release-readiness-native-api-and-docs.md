---
"rscrypto" = "minor"
---

Closed native API gaps with HKDF-SHA512, HMAC-SHA3, KMAC128, standalone
Poly1305, `getrandom` key-generation helpers, AEAD `*_to_vec` helpers, generic
signing/verification traits, and dedicated AEAD, signature, RSA, and ML-KEM
examples.

Hardened release and security automation with versioned constant-time evidence
bundles, release-path audit and semver checks, weekly ASan fuzz-corpus replay,
OpenSSF Scorecard, CODEOWNERS, CONTRIBUTING guidance, and committed minimized
fuzz seeds.

Pruned duplicated public Markdown, made `THREAT_MODEL.md` the canonical audit
entry point, folded advisory readiness into `SECURITY.md`, and kept
maintainer-only release and benchmark evidence out of README onboarding.
