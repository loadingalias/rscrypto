---
"rscrypto" = "patch"
---

Removed variable-latency scalar multiplication from ECDSA P-256/P-384 secret
arithmetic on s390x and RISC-V while preserving signing support, and restored
the affected native timing cases as required release gates.

Made constant-time assembly triage fail closed on target-sensitive arithmetic,
grouped every reachable finding with exact review-bound waivers, and bound
release evidence to the complete lane set, source commit, toolchain, target
configuration, generated artifacts, timing results, and BINSEC results. Weekly
and release publication now both require the dedicated RSA evidence workflow.
Cross-platform checks now keep host-only feature-matrix flags out of target
crate selection and keep target-only diagnostics plus CPUID's MSRV safety
transition lint-clean across the full matrix.

Corrected the public constant-time boundary: `ct.toml` records intent, while a
claim exists only for configurations with passing evidence in the matching
attested release bundle.

Removed secret-fed overflow panic branches from the fixed-work ECDSA limb
multiply-accumulate used on s390x and RISC-V. Blinded signing now additively
masks the private scalar during `r·d`, with direct full-width carry equivalence
tests and regenerated target evidence.
