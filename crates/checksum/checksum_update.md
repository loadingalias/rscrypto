# Checksum Baseline Update (2026-02-25)

## Baseline run captured

- Workflow run: `22379444492` (all 8 lanes green)
- Commit under test: `1c171f19246836f550f8ccb2794da66385214e23`
- Generated CSVs:
  - `/private/tmp/checksum-bench-22379444492/crc32_baseline_comp_summary.csv`
  - `/private/tmp/checksum-bench-22379444492/crc32_baseline_vs_external.csv`
  - `/private/tmp/checksum-bench-22379444492/crc32_ours_vs_22372289591.csv`

## Baseline status (CRC32 + CRC32C)

- Total cases: `80`
- Ahead of best external: `60`
- Behind best external: `20`
- Largest external deficits:
  - `graviton4 crc32/ieee xs`: `-29.227%`
  - `graviton4 crc32/ieee s`: `-24.759%`
  - `intel-spr crc32/ieee xs`: `-18.145%`
  - `ibm-power10 crc32/ieee xs`: `-16.638%`
  - `ibm-s390x crc32/ieee xs`: `-11.755%`
  - `intel-spr crc32c/castagnoli xl`: `-10.718%`
  - `intel-icl crc32c/castagnoli l`: `-10.498%`

## Experiment log

## 2026-02-25: Graviton4 dedicated table, CRC32 IEEE xs/s -> hwcrc (FAILED)

- Candidate commit: `b0b9de75534b476cba9241f22ea142a0825655ae`
- Validation run: `22380785922` (graviton3 + graviton4)
- Comparison baseline: `22379444492`

Results on target cells (Graviton4, CRC32 IEEE):
- `xs`: `9.484 -> 2.191 GiB/s` (`-76.900%`)
- `s`: `13.660 -> 7.006 GiB/s` (`-48.713%`)
- External margin degradation:
  - `xs`: `-29.227% -> -83.908%`
  - `s`: `-24.759% -> -60.453%`

Conclusion:
- Hardware CRC kernels are not viable for Graviton4 `crc32/ieee` at `xs/s` in current implementation.
- Revert this candidate immediately.

Artifacts:
- `/private/tmp/checksum-bench-22380785922/crc32_ours_vs_22379444492.csv`
- `/private/tmp/checksum-bench-22380785922/crc32_baseline_vs_external.csv`
- `/private/tmp/checksum-bench-22380785922/crc32_external_gap_delta_vs_22379444492.csv`

## Next plan (ordered)

1. Graviton4 `crc32/ieee` xs/s kernel shootout with PMULL-family only:
- `aarch64/pmull-small` (current baseline)
- `aarch64/pmull-v9s3x2e-s3`
- `aarch64/pmull-v9s3x2e-s3-2way`
- `aarch64/pmull-v9s3x2e-s3-3way`

2. Intel SPR `crc32c` l/xl retune:
- Current `x86_64/fusion-vpclmul-v3x2-2way`
- Evaluate `{2way,4way,7way,8way}` for `64KiB` and `1MiB`.

3. POWER10 and s390x CRC32 IEEE xs tiny-path retune:
- Verify `portable/bytewise` vs `portable/slice16` at `xs`.

Execution rule:
- One small change at a time, rerun targeted CI lanes, keep only clear wins.

## 2026-02-25: CRC32 focused validation for commit `362bbce` (ICL revert + hasher dispatch cache)

- Run A: `22405777646` (graviton3/graviton4/intel-icl/intel-spr, `checksum`, `comp+kernels`, `only=crc32`, `quick=false`)
- Run B (repeat on same SHA): `22407775339` (same matrix and settings)
- Commit under test: `362bbce3252326827a21965322c21d6e17f6f41e`

### Stable conclusions from A+B (treat as real)

- Persistent CRC32 IEEE losses:
  - `graviton4 crc32/ieee xs`: `-32.47%` (A) -> `-34.76%` (B)
  - `graviton4 crc32/ieee s`: `-26.60%` (A) -> `-29.62%` (B)
  - `intel-icl crc32/ieee s`: `-11.84%` (A) -> `-5.53%` (B)
  - `intel-spr crc32/ieee s`: `-3.33%` (A) -> `-3.96%` (B)
- `graviton3 crc32/ieee s` recovered and stayed green:
  - `+2.85%` (A), `+2.02%` (B)

### Noisy/non-actionable in this pass (do not optimize yet)

- `intel-spr crc32/ieee xl`: `-5.17%` (A) -> `+2.59%` (B), sign flip
- Several crc32c near-zero cells also flipped sign between A and B

### Dispatch headroom check on persistent CRC32 IEEE losses

- `graviton4 xs`: selected already best (`pmull-small`), headroom `0.0%`
- `graviton4 s`: selected `pmull-small`, best `sve2-pmull`, headroom only `~0.06%`
- `intel-icl s`: selected `vpclmul`, best `vpclmul-2way`, headroom `~2.69%`
- `intel-spr s`: selected `vpclmul-7way`, best `vpclmul`, headroom `~2.48%`

Conclusion:
- Graviton4 `xs/s` is not a table selection problem; we need kernel/code-path work.
- ICL/SPR `s` still have small dispatch retune headroom worth testing one-at-a-time.

## Focused next plan (persistent CRC32 IEEE only)

1. `intel-icl crc32/ieee s`: retune dispatch to `x86_64/vpclmul-2way` and rerun same 4-lane matrix.
2. If #1 is clean, `intel-spr crc32/ieee s`: retune to `x86_64/vpclmul` and rerun same 4-lane matrix.
3. Graviton4 `crc32/ieee xs/s`: start kernel-path optimization (not table churn):
   - keep `pmull-small` mapping;
   - optimize tiny/small path internals and call overhead;
   - validate with Graviton-only `comp+kernels` non-quick before full 4-lane check.
