---
"rscrypto" = "patch"
---

Clear AES-GCM's materialized CTR keystream buffers on return and unwind in the portable,
batched, and native tail paths. Isolate generic CTR work and rustls cipher construction
behind borrowed-only workers with post-call stack scrubs for compiler-created stack
copies. Generated-code coverage stays target and compiler bound; live registers retain
their separate qualification limits.
