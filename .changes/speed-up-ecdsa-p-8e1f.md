---
"rscrypto" = "patch"
---

Speed up ECDSA P-256 verification and public-key import. Verification and comb-table construction use P-256-specific field and point arithmetic, with flag-chained Montgomery multiplication and squaring in assembly on AArch64, and verification accumulates its generator and key terms in independent chains. Verification results and public-key tables are unchanged.
