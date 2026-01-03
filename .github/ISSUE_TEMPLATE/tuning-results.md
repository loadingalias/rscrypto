---
name: Tuning Results Submission
about: Submit benchmark results from your hardware to improve rscrypto defaults
title: "tune: [CPU/Platform Name]"
labels: ["tuning", "enhancement"]
---

## Platform Information

**CPU:** <!-- e.g., AMD Ryzen 9 7950X, Intel Xeon w9-3595X, Apple M3 Max -->
**OS:** <!-- e.g., Linux 6.8, macOS 15.2, Windows 11 -->
**Rust version:** <!-- output of `rustc --version` -->

## Tuning Output

<!-- Paste the full output from `just tune` below -->

```
<paste here>
```

## Checklist

- [ ] I ran `just tune` (not `just tune-quick`) for accurate results
- [ ] The system was mostly idle during benchmarking
- [ ] Peak throughput numbers look reasonable for my hardware
- [ ] No excessive "high variance" warnings appeared

## Additional Context

<!-- Optional: Any notes about your setup, BIOS settings, etc. -->
