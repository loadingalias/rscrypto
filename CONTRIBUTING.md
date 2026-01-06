# Contributing to rscrypto

Thank you for your interest in making rscrypto faster! This guide covers the most impactful way to contribute: **submitting tuning results from your hardware**.

## Why Tuning Matters

rscrypto automatically selects optimal algorithms based on your CPU. But we need real benchmark data from diverse hardware to make good choices. Every platform contribution helps users on similar hardware get better performance automatically. I've only really got access to my Macbook M1 and Namespace/Github CI runners. If you have access to other hardware, please consider contributing!

## Quick Start: Submit Your Tuning Results

**Time required:** ~5 minutes

### Step 1: Run the tuner

```bash
# Clone and build
git clone https://github.com/loadingalias/rscrypto.git
cd rscrypto

# Run tuning and generate contribution-ready output
just tune-contribute
```

### Step 2: Copy the output

The output is formatted markdown ready to paste:

```markdown
## Tuning Results

**Platform:** `Caps(aarch64, [...]) (Apple M1-M3)`
**Tune preset:** `AppleM1M3`

| Algorithm | Best Kernel | Streams | Peak GiB/s |
|-----------|-------------|---------|------------|
| crc16-ccitt | `aarch64/pmull` | 3 | 62.1 |
...
```

### Step 3: Open an issue

1. Go to [Issues → New Issue](../../issues/new?template=tuning-results.md)
2. Select **"Tuning Results Submission"**
3. Paste your output
4. Submit!

That's it. We'll review and merge your results.

---

## Advanced: Direct PR Submission

If you're comfortable with PRs, you can apply results directly:

```bash
# Apply tuning to source files
just tune-apply

# Review changes
git diff crates/checksum/src/*/tuned_defaults.rs

# Commit and PR
git checkout -b tune/my-platform
git add -p  # Review each change
git commit -m "tune: add [Your CPU] defaults"
git push -u origin tune/my-platform
```

### What gets changed

The tuner updates files like `crates/checksum/src/crc64/tuned_defaults.rs`:

```rust
pub const CRC64_TUNED_DEFAULTS: &[(TuneKind, Crc64TunedDefaults)] = &[
  // BEGIN GENERATED (rscrypto-tune)
  (TuneKind::Zen4, Crc64TunedDefaults { ... }),
  (TuneKind::AppleM1M3, Crc64TunedDefaults { ... }),  // ← Your contribution
  // END GENERATED (rscrypto-tune)
];
```

---

## Hardware We Need

We especially want results from:

| Priority | Platform | TuneKind |
|----------|----------|----------|
| High | Intel Sapphire Rapids | `IntelSpr` |
| High | Intel Granite Rapids | `IntelGnr` |
| High | Intel Ice Lake | `IntelIcl` |
| Medium | AWS Graviton3/4 | `Graviton3`, `Graviton4` |
| Medium | Ampere Altra | `AmpereAltra` |
| Medium | NVIDIA Grace | `NvidiaGrace` |
| Medium | IBM Power9/10 | `Power9`, `Power10` |
| Low | AMD EPYC (Zen3) | Contact us first |

Already have good coverage:
- AMD Zen4/Zen5 ✓
- Apple M1-M4 ✓
- AWS Graviton2 ✓

---

## Tuning Options

```bash
# Standard run (recommended for submissions)
just tune

# Quick run (faster, noisier - good for testing)
just tune-quick

# Verbose output (shows per-kernel benchmarks)
just tune -- --verbose

# JSON output (for automation)
just tune -- --format json > results.json
```

---

## Validation

Before submitting, verify your results make sense:

1. **Peak throughput** should be reasonable for your CPU
   - Modern x86_64: 30-80 GiB/s for CRC64
   - Modern aarch64: 40-80 GiB/s for CRC64

2. **Stream counts** typically:
   - x86_64: 1-8 streams
   - aarch64: 1-3 streams

3. **Thresholds** typically:
   - `portable_to_*`: 32-256 bytes
   - `min_bytes_per_lane`: 32-16384 bytes

If something looks off, try running again. High variance warnings are normal on busy systems.

---

## Questions?

- Open a [Discussion](../../discussions) for questions
- Check existing [Issues](../../issues) for known hardware
- Tag `@maintainers` if unsure about your results

Thank you for contributing!
