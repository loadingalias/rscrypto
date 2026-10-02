---
"rscrypto" = "patch"
---

Require evidence for the ML-DSA secret kernels in the constant-time harness.
Cover all parameter sets with production timing adapters, retained linked-binary roots,
and a bounded proof root for the portable Montgomery multiplication.
The existing sample budgets and thresholds do not change.
Whole signing keeps its variable retries, and stays outside a strict constant-time claim.

Protect x86-64 scalar selections and secret sampler masks from branches that the compiler creates.
The standard arithmetic and sampling order do not change.

Follow direct tail transfers when building linked-code call closures,
so that optimized evidence wrappers cannot hide the production kernel from assembly review.
