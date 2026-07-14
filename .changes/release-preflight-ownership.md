---
"rscrypto" = "patch"
---

Made release preflight consume the exact-commit Cargo graph assurance gate instead
of repeating the exhaustive target and feature sweep. Duplicate runs for the same
release tag now collapse immediately while signed-tag, audit, SemVer, constant-time,
RSA, package-integrity, provenance, and approval gates remain blocking.
