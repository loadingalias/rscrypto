# Blake3 Gap Gate

This gate tracks a small set of BLAKE3 one-shot sizes where `rscrypto` has historically trailed `official`.

## Pinned Cases

- 256 B
- 1 KiB
- 4 KiB
- 16 KiB
- 64 KiB

## Current Max Allowed Gap (Need %+ to match official)

- 256 B: `+4.8%`
- 1 KiB: `+6.8%`
- 4 KiB: `+7.8%`
- 16 KiB: `+13.0%`
- 64 KiB: `+4.5%`

Gate implementation: `scripts/bench/blake3-gap-gate.py`.

## Local Usage

```bash
just bench-blake3-gate
```

## CI Usage

Manual Bench workflow (`.github/workflows/bench.yaml`) input:

- `enforce_blake3_gap_gate=true`
- `enforce_blake3_x86_kernel_gate=true`

Notes:

- Requires `quick=false`.
- Runs against Criterion data under `target/criterion`.
- Fails if any pinned case exceeds its allowed gap.
- Ratchet policy: when a case improves by >=1.0% sustained across CI hosts, reduce that case limit by 0.5-1.0%.

## x86 Kernel Gate

When `enforce_blake3_x86_kernel_gate=true` on x86 lanes (`intel-icl`, `intel-spr`, `amd-zen4`, `amd-zen5`):

- CI runs `blake3/kernel-ab` with diagnostics enabled.
- Gate compares `official` against the best `rscrypto/x86_64/*` kernel per pinned size.
- Thresholds are lane-specific in `scripts/ci/run-bench.sh`:
  - `intel-spr`: `256=8,1024=7,4096=6,16384=6,65536=6`
  - `intel-icl`: `256=9,1024=8,4096=7,16384=7,65536=7`
  - `amd-zen4` / `amd-zen5`: `256=9,1024=8,4096=7,16384=7,65536=7`
- Non-x86 lanes skip this gate automatically.
