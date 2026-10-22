---
"rscrypto" = "patch"
---

Fix compiler-introduced secret branches in portable P-256 ECDH: the signed-window table selection on RISC-V, and signed-digit correction and point doubling on IBM Z. Keep the selection masks opaque and add a blocking compiled-code check and, on AArch64, a bounded signed-selector proof. Results are unchanged.
