# Contributing to rscrypto

Submit tuning results from your hardware. Every contribution helps users on similar platforms get better performance automatically.

## Quick Start

```bash
git clone https://github.com/loadingalias/rscrypto.git
cd rscrypto
just tune-contribute
```

Paste the output into a [new issue](../../issues/new?template=tuning-results.md).

## Platform Coverage

### Measured

| Platform | TuneKind | Peak CRC-64 |
|----------|----------|-------------|
| AMD Zen 4 | `Zen4` | 75 GiB/s |
| Apple M1-M3 | `AppleM1M3` | 63 GiB/s |
| AWS Graviton 2 | `Graviton2` | 33 GiB/s |

### Inferred

| Platform | Inferred From |
|----------|---------------|
| AMD Zen 5/5c | Zen4 |
| Apple M4/M5 | AppleM1M3 |
| Intel SPR/GNR/ICL | Generic VPCLMUL |
| AWS Graviton 3/4/5 | Graviton2 |
| ARM Neoverse N2/N3/V3 | Graviton2 |
| NVIDIA Grace, Ampere Altra | Graviton2 |

### Wanted

| Priority | Platform | TuneKind |
|----------|----------|----------|
| High | Intel Sapphire/Granite/Ice Lake | `IntelSpr`, `IntelGnr`, `IntelIcl` |
| Medium | AWS Graviton 3/4 | `Graviton3`, `Graviton4` |
| Medium | IBM Power 9/10 | `Power9`, `Power10` |
| Medium | IBM z14/z15 | `Z14`, `Z15` |
| Low | RISC-V with Zbc/Zvbc | — |

## Validation

Before submitting, verify results are reasonable:

- **Peak throughput**: 30-80 GiB/s for CRC-64 on modern CPUs
- **Stream counts**: 1-8 (x86_64), 1-3 (aarch64)

If something looks off, run again. High variance is normal on busy systems.

## Questions

Open a [Discussion](../../discussions) or tag `@maintainers` in your issue.
