---
"rscrypto" = "patch"
---

Keep the work of every Arm data-independent timing (`PSTATE.DIT`) scope between the DIT writes with compiler barriers, and detect `FEAT_DIT` without `std` on Linux, Android, and Apple targets.
