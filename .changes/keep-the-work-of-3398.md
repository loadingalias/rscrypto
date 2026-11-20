---
"rscrypto" = "patch"
---

Keep the work of every Arm data-independent timing (`PSTATE.DIT`) scope between the DIT writes with compiler barriers, and detect `FEAT_DIT` without `std` on Linux, Android, and Apple targets. On AArch64, a scope that returns a secret by value, such as HPKE `setup`, keeps one more uncleared stack copy of it; the `*_in` forms do not.
