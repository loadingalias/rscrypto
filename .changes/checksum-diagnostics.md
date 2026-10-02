---
"rscrypto" = "minor"
---

Reduce the CRC-32 and CRC-64 selection diagnostics to these fields: polynomial, input length,
architecture, selection reason, effective force setting, and selected kernel.
Remove old policy thresholds, stream counts, capability flags,
and placeholder values that no longer described active dispatch.
Remove the unused `SelectionReason::BelowSmallThreshold` and `BelowSimdThreshold` variants.
The numeric discriminants of the remaining selection reasons change.
Checksum computation and backend selection do not change.
