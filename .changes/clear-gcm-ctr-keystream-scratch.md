---
"rscrypto" = "patch"
---

Clear AES-GCM's materialized CTR keystream buffers on return and unwind in the portable,
batched, and native tail paths. Run generic CTR work in a borrowed-only worker, then clear
3 KiB of its dead stack in one pass to remove compiler-created copies. Generated-code coverage
stays target and compiler bound; live registers retain their separate qualification limits.
