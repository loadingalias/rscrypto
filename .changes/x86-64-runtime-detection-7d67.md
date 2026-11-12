---
"rscrypto" = "patch"
---

x86-64 runtime detection now reports a capability only together with every feature that Rust's target feature implies for it, so an incomplete CPUID, such as a hypervisor mask, can no longer select a kernel whose implied instructions the CPU lacks.
