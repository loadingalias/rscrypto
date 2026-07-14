---
"rscrypto" = "patch"
---

Hardened portable P-256 and P-384 caller-blinded fixed-base multiplication so
projective randomization is established before secret-selected points enter
field arithmetic. P-256 also randomizes the scalar representative with a group
order multiple. Corrected the timing harness's random-secret distribution and
extended RISC-V assembly screening to conditional branches.

Made release tagging require successful CT and RSA evidence from the exact
candidate commit. Release publication now promotes that Weekly run's raw CT
artifacts instead of launching duplicate multi-hour evidence workflows, and RSA
workflow concurrency can no longer cancel its reusable-workflow caller.
