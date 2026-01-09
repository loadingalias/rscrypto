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

<!-- Paste the output from `just tune-contribute` below -->

```
<paste here>
```

## Checklist

- [ ] Ran `just tune-contribute` on an idle system
- [ ] Throughput looks reasonable (30-80 GiB/s for CRC-64)
- [ ] No excessive variance warnings

## Notes

<!-- Optional: BIOS settings, VM, etc. -->
