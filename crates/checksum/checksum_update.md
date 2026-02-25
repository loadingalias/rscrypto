# Checksum Baseline Update (2026-02-25)

## Baseline run captured

- Workflow run: `22379444492` (all 8 lanes green)
- Commit under test: `1c171f19246836f550f8ccb2794da66385214e23` (revert of dispatch-cache change)
- Prior comparison baseline: `22372289591`
- Generated CSVs:
  - `/private/tmp/checksum-bench-22379444492/crc32_baseline_comp_summary.csv`
  - `/private/tmp/checksum-bench-22379444492/crc32_baseline_vs_external.csv`
  - `/private/tmp/checksum-bench-22379444492/crc32_ours_vs_22372289591.csv`

## Headline numbers (CRC32 + CRC32C)

- Total cases: `80`
- Ahead of best external: `60`
- Behind best external: `20`
- Biggest external deficits:
  - `graviton4 crc32/ieee xs`: `-29.227%` vs `crc32fast/auto`
  - `graviton4 crc32/ieee s`: `-24.759%` vs `crc32fast/auto`
  - `intel-spr crc32/ieee xs`: `-18.145%` vs `crc32fast/auto`
  - `ibm-power10 crc32/ieee xs`: `-16.638%` vs `crc32fast/auto`
  - `ibm-s390x crc32/ieee xs`: `-11.755%` vs `crc32fast/auto`
  - `intel-spr crc32c/castagnoli xl`: `-10.718%` vs `crc-fast/auto`
  - `intel-icl crc32c/castagnoli l`: `-10.498%` vs `crc-fast/auto`

## Stability check vs prior baseline (`22372289591`)

- Same selected `rscrypto` kernel in all `80/80` cases (`impl_new == impl_old`).
- Deltas: `33` wins / `47` losses.
- Median delta: `-0.0907%`.
- Mean delta: `-0.4613%`.
- Cases within `±3%`: `68/80` (`85%`).

Interpretation: baseline is mostly stable; large swings are concentrated in a small number of lane/size tails.

## Where to start (highest ROI first)

## 1) Graviton4 CRC32 IEEE tiny/small (xs/s)

Why first:
- Largest deficits in the whole matrix (`-29%`, `-25%`).
- Graviton4 currently reuses Graviton3 family table instead of a dedicated table.
- Current selected kernels on these sizes are `aarch64/pmull-small`.

Code evidence:
- Family mapping of Graviton4 -> Graviton3 table:
  - `crates/checksum/src/dispatch.rs` (`family_match`, `TuneKind::Graviton4 | TuneKind::Graviton5 => GRAVITON3_TABLE`)
- Graviton3 xs/s CRC32 selection:
  - `crates/checksum/src/dispatch.rs` (`GRAVITON3_TABLE`, `crc32_ieee = aarch64/pmull-small` for `xs` and `s`)

Immediate experiment:
- On Graviton4 lane, benchmark CRC32 IEEE kernels for `xs`/`s` only:
  - `aarch64/hwcrc`
  - `aarch64/hwcrc-2way`
  - `aarch64/hwcrc-3way`
  - `aarch64/pmull-small` (current)
  - `aarch64/pmull-v9s3x2e-s3`
- If hwcrc wins at tiny sizes, split Graviton4 table from Graviton3 and switch only xs/s entries.

Success criteria:
- Reduce worst external gap on Graviton4 xs/s to <= `-5%` without regressing `l/xl` by more than `1%`.

## 2) Intel SPR CRC32C large/xlarge (l/xl)

Why second:
- Stable double-digit gap on `xl`, material gap on `l`.
- Current table uses `fusion-vpclmul-v3x2-2way` for `l/xl`; may be under-parallelized for this CPU.

Code evidence:
- `INTEL_SPR_TABLE` large selection for `crc32c`:
  - `crates/checksum/src/dispatch.rs` (`crc32c = x86_64/fusion-vpclmul-v3x2-2way` in `l`)

Immediate experiment:
- Compare `fusion-vpclmul-v3x2-{2way,4way,7way,8way}` on SPR for `64KiB` and `1MiB`.
- Keep xs/s/m untouched while testing.

## 3) POWER10 + s390x CRC32 IEEE xs

Why third:
- Both are behind only at tiny size (`xs`), and both route xs/s to `PORTABLE_SET`.
- This is likely a tiny-size portable-kernel choice issue, not a SIMD large-buffer issue.

Code evidence:
- `POWER10_TABLE` and `S390X_Z15_TABLE`:
  - `xs: PORTABLE_SET`, `s: PORTABLE_SET`, with boundaries `[64,64,4096]`.

Immediate experiment:
- Tiny-size shootout (`xs` only): `portable/slice16` vs `portable/bytewise` for CRC32 IEEE.
- If bytewise wins on these arches, introduce a tiny-only portable CRC32 kernel choice for xs.

## Execution order

1. Graviton4 xs/s fix (largest and most clearly table-mismatch-driven gap).
2. Intel SPR crc32c l/xl lane-width retune.
3. POWER10/s390x xs tiny-portable retune.

This order maximizes expected gain per change while keeping each patch narrow and easy to validate/revert.
