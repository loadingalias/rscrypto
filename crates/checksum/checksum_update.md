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

## 2026-02-25: CRC32 hasher auto-dispatch table cache (SUCCESS, new targeted baseline)

- Commit under test: `ee5eee46e46757de8ccddf2611b368b496c0fbb1`
- Validation run: `22418728015` (graviton3/graviton4/intel-icl/intel-spr, `checksum`, `comp+kernels`, `only=crc32`, `quick=false`)
- Comparison baseline for deltas: `22414190122`

### Result summary

- Total cases: `40`
- Ahead of best external: `35`
- Behind best external: `5`
- Loss count improved: `11 -> 5` (`-6` cells)

### Key wins from this change

- `graviton4 crc32/ieee xs`: `-16.591% -> +30.960%` (LOSE -> WIN), ours `11.422 -> 18.138 GiB/s`
- `graviton4 crc32/ieee s`: `-6.513% -> +4.701%` (LOSE -> WIN), ours `17.311 -> 19.489 GiB/s`
- `graviton3 crc32/ieee s`: `-0.182% -> +29.465%` (LOSE -> WIN), ours `14.251 -> 18.551 GiB/s`
- `graviton4 crc32/ieee l`: `-0.671% -> +0.299%` (LOSE -> WIN)

### Remaining deficits (new baseline to close)

- `intel-icl crc32/ieee s`: `-5.878%` (`x86_64/vpclmul`, `15.965` vs `crc32fast/auto` `16.962 GiB/s`)
- `graviton3 crc32/ieee xl`: `-2.537%` (`aarch64/pmull-eor3-v9s3x2e-s3`, `44.684` vs `crc-fast/auto` `45.847 GiB/s`)
- `intel-spr crc32/ieee s`: `-2.320%` (`x86_64/vpclmul-7way`, `19.198` vs `crc32fast/auto` `19.654 GiB/s`)
- `intel-icl crc32c/castagnoli m`: `-0.627%` (new small regression)
- `graviton4 crc32/ieee xl`: `-0.146%` (near noise floor)

### Artifacts

- `/private/tmp/checksum-bench-22418728015/comp_parsed.csv`
- `/private/tmp/checksum-bench-22418728015/kernels_parsed.csv`
- `/private/tmp/checksum-bench-22418728015/comp_vs_competitors.csv`
- `/private/tmp/checksum-bench-22418728015/current_losses.csv`
- `/private/tmp/checksum-bench-22418728015/kernel_selected_vs_best.csv`
- `/private/tmp/checksum-bench-22418728015/delta_vs_22414190122.csv`

### Updated next plan (from this baseline)

1. `intel-icl crc32/ieee s`: retune selection away from `vpclmul` (trial `vpclmul-2way` then `vpclmul-8way`), rerun same 4-lane matrix.
2. `intel-spr crc32/ieee s`: retune from `vpclmul-7way` to best measured small-size variant (`vpclmul-2way` candidate), rerun same 4-lane matrix.
3. `graviton3 crc32/ieee xl`: kernel-path investigation (no internal dispatch headroom in current table).

## 2026-02-26: Full-suite checksum run attempt (INCOMPLETE, partial data only)

- Workflow run: `22421227461`
- Commit under test: `4eab23c03474eb6cad5f55bad48862a271e868db`
- Requested scope: all checksum CRC families (`comp+kernels`) across 8 lanes.

### Run outcome

- Workflow conclusion: `failure`
- Lane status:
  - Completed with artifacts: `amd-zen5`, `graviton4`, `ibm-power10`, `ibm-s390x`
  - Incomplete/cancelled (no artifact): `amd-zen4`, `intel-icl`, `intel-spr`
  - Failed/cancelled run step: `graviton3`

This is **not** a valid full baseline for all arches because 4 lanes are missing.

### Partial snapshot (4 completed lanes only)

- Cases captured: `140` (`4 arches x 7 algos x 5 sizes`)
- Ahead of best external: `119`
- Behind best external: `21`
- By arch:
  - `amd-zen5`: `31` wins / `4` losses
  - `graviton4`: `23` wins / `12` losses
  - `ibm-power10`: `33` wins / `2` losses
  - `ibm-s390x`: `32` wins / `3` losses

### Largest partial deficits

- `ibm-power10 crc64/xz xs`: `-20.636%` (selected `portable/slice16`, no internal headroom)
- `ibm-power10 crc64/nvme xs`: `-19.999%` (selected `portable/slice16`, no internal headroom)
- `ibm-s390x crc32/ieee xs`: `-14.597%` (selected `portable/slice16`, no internal headroom)
- `amd-zen5 crc64/xz l`: `-12.650%` (dispatch headroom exists: `vpclmul-4x512 -> vpclmul-4way`)
- `graviton4 crc16/ccitt xl`: `-8.980%` (dispatch headroom exists: `pmull -> pmull-2way`)

### Artifacts (partial run data)

- `/private/tmp/checksum-bench-22421227461/benchmark-amd-zen5/output.txt`
- `/private/tmp/checksum-bench-22421227461/benchmark-graviton4/output.txt`
- `/private/tmp/checksum-bench-22421227461/benchmark-ibm-power10/output.txt`
- `/private/tmp/checksum-bench-22421227461/benchmark-ibm-s390x/output.txt`
- `/private/tmp/checksum-bench-22421227461/comp_parsed.csv`
- `/private/tmp/checksum-bench-22421227461/kernels_parsed.csv`
- `/private/tmp/checksum-bench-22421227461/comp_vs_competitors.csv`
- `/private/tmp/checksum-bench-22421227461/current_losses.csv`
- `/private/tmp/checksum-bench-22421227461/kernel_selected_vs_best.csv`

### Baseline policy update

- Do **not** replace full-suite baseline with run `22421227461`.
- Use this run only as a partial directional snapshot for the 4 completed arches.

## 2026-02-26: Full-suite checksum baseline (SUCCESS)

- Workflow run: `22425195531` (`Bench`, all 8 lanes green)
- Commit under test: `4eab23c03474eb6cad5f55bad48862a271e868db`
- Run URL: `https://github.com/loadingalias/rscrypto/actions/runs/22425195531`

### Baseline status (all CRC families, all 8 arches)

- Total cases: `280`
- Ahead of best external: `247`
- Behind best external: `33`

By arch (wins / losses):
- `amd-zen4`: `34 / 1`
- `amd-zen5`: `34 / 1`
- `graviton3`: `29 / 6`
- `graviton4`: `23 / 12`
- `intel-icl`: `33 / 2`
- `intel-spr`: `28 / 7`
- `ibm-s390x`: `33 / 2`
- `ibm-power10`: `33 / 2`

By algorithm (wins / losses):
- `crc16/ccitt`: `33 / 7`
- `crc16/ibm`: `33 / 7`
- `crc24/openpgp`: `40 / 0`
- `crc32/ieee`: `34 / 6`
- `crc32c/castagnoli`: `34 / 6`
- `crc64/nvme`: `36 / 4`
- `crc64/xz`: `37 / 3`

### Largest deficits (current priority queue)

- `ibm-power10 crc64/xz xs`: `-19.745%` (`portable/slice16`, `1.5153` vs `1.8881 GiB/s`)
- `ibm-power10 crc64/nvme xs`: `-19.458%` (`portable/slice16`, `1.5179` vs `1.8846 GiB/s`)
- `intel-icl crc32/ieee s`: `-12.301%` (`x86_64/vpclmul-8way`, `15.785` vs `17.999 GiB/s`)
- `graviton3 crc16/ibm xl`: `-10.821%` (`aarch64/pmull`, `30.699` vs `34.424 GiB/s`)
- `graviton3 crc16/ccitt xl`: `-10.259%` (`aarch64/pmull`, `31.204` vs `34.771 GiB/s`)
- `graviton4 crc16/ccitt xl`: `-8.747%`
- `graviton4 crc16/ccitt l`: `-8.465%`
- `graviton4 crc16/ibm l`: `-8.444%`
- `graviton4 crc16/ibm xl`: `-8.391%`
- `amd-zen4 crc32c/castagnoli l`: `-7.233%`

### Artifacts

- `/private/tmp/checksum-bench-22425195531/benchmark-amd-zen4/output.txt`
- `/private/tmp/checksum-bench-22425195531/benchmark-amd-zen5/output.txt`
- `/private/tmp/checksum-bench-22425195531/benchmark-graviton3/output.txt`
- `/private/tmp/checksum-bench-22425195531/benchmark-graviton4/output.txt`
- `/private/tmp/checksum-bench-22425195531/benchmark-intel-icl/output.txt`
- `/private/tmp/checksum-bench-22425195531/benchmark-intel-spr/output.txt`
- `/private/tmp/checksum-bench-22425195531/benchmark-ibm-s390x/output.txt`
- `/private/tmp/checksum-bench-22425195531/benchmark-ibm-power10/output.txt`
- `/private/tmp/checksum-bench-22425195531/comp_parsed.csv`
- `/private/tmp/checksum-bench-22425195531/kernels_parsed.csv`
- `/private/tmp/checksum-bench-22425195531/comp_vs_competitors.csv`
- `/private/tmp/checksum-bench-22425195531/current_losses.csv`
- `/private/tmp/checksum-bench-22425195531/kernel_selected_vs_best.csv`

### Focus after this baseline

1. `intel-icl crc32/ieee s`: fix small-size x86 kernel/dispatch choice first (largest x86 deficit).
2. `graviton3/graviton4 crc16 (l/xl)`: investigate AArch64 PMULL kernel throughput gap vs `crc-fast`.
3. `ibm-power10 crc64 xs`: targeted tiny-input path work (`portable/slice16` vs competitor xs behavior).
