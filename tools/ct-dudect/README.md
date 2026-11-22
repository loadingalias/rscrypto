# DudeCT timing export

Run `just ct-test` to check the exporter and the report validation.
Run timing cases through `just ct-dudect` or `just ct-full`.

## HMAC host controls

The separate `hmac_host_controls` binary diagnoses the short HMAC-SHA256 case.
It calls production `HmacSha256::verify_tag` with the release case's message,
random keys, class sequence, and seed. Only expected-tag preparation changes.
Set `RSCRYPTO_CT_HMAC_CONTROL` to `valid-invalid`, `invalid-valid`, `valid-valid`,
or `invalid-invalid`, and set `RSCRYPTO_CT_DUDECT_SAMPLES` to a positive count.
Both settings are required. They have no effect on the release timing binary.
An unknown control mode prints a diagnostic and exits with status 2 before collecting samples.

The two identical-work controls separate class-label effects from tag validity.
Reversing validity checks whether a difference follows the result or the label.
The control binary has its own code layout; it cannot qualify the release binary.
Retain exact binaries, raw CSVs, compiler identity, CPU affinity, and all planned
repetitions when comparing them. A passing control does not dismiss a failing
release case.

## CSV format

The CSV columns are `benchname,sequence,class,runtime_ns`.

- Each completed `run_one` adds one row.
- Class `0` is the left class.
  Class `1` is the right class.
- `sequence` starts at zero for each case and follows execution order, also when the class counts differ.
  In continuous mode, the sequence continues across batches.
  It records order only, not wall-clock timestamps or pauses between observations.
- Durations are the same integer nanoseconds that the in-memory statistics receive.

## Report

- The report needs every requested observation, both classes, and contiguous sequence numbers.
- Its `raw_csv` counts and artifact hash bind the retained CSV.
- `dudect_runner_sources` binds the local runner sources.
- The report rejects legacy exports.
  It does not treat them as complete measurements.
  [`scripts/ct/dudect_report.py`](../../scripts/ct/dudect_report.py) sets the current schema version.

## Local runner patch

`vendor/dudect-bencher` keeps the selected upstream 0.7.0 source and its licenses.

- `UPSTREAM.json` records the identity of the crate archive and the original file hashes.
- Its `local_files` entries record the exact hashes of the three patched files: `src/ctbench.rs`
  (the timing loop), `src/macros.rs`, and `src/stats.rs`. The identity test checks every recorded hash,
  and keeps the originals.
- The local patch records each duration and its class after the timed closure, in execution order,
  without branching on the class. It splits the durations by class only after the case completes,
  and exports every sample in execution order.
  Upstream's per-class push branched on the class right after the end timestamp.
  On Graviton5 that branch gave the classes different timings even when both ran identical work;
  label-independent recording removed the difference
  ([2026-10-09 record](https://github.com/loadingalias/rscrypto/blob/50c708b29905bb3221e7b22b9c7cce8fcc700818/benchmark_results/OVERVIEW.md#2026-10-09-graviton5-dudect-class-label-artifact)).
- The maximum t skips a cropped test whose t is not finite. A crop where each class holds one repeated
  duration has no variance, so its t is undefined. The RISC-V CI timer ticks every 333 ns, and equal tick
  counts convert to durations 1 ns apart, so such a crop gave t = -inf (CT 44476221075). The full test
  keeps every sample, so a constant difference between the classes still gives an infinite t.
- The runner measures the smallest step of `Instant` at start and prints it as `timer tick N ns`.
  On the first sample of a case, it doubles the calls of the operation in one sample until each of
  16 timed samples lasts twice `--min-ticks` ticks. One cold or preempted sample therefore cannot
  set the count. Both classes use that count, which it prints as `bench NAME timed N calls per sample`.
- `--seed` replaces the seed of each selected case, so a confirmation measures new inputs.
- Buffered output is flushed before a result is reported.
- Clap's optional terminal color support is disabled to remove `atty` and
  `ansi_term` from the tool.
- The export tests cover unequal classes, deterministic duration and order fixtures,
  replay of real measured samples through the same statistic, and the repetition of short operations.

The local dependency is unpublished, and only this evidence harness uses it.
Vendored code is excluded from automatic dependency-requirement rewriting
and from first-party style lints.
`just ct-test` has explicit regression coverage for its exporter.
Review any runner update against the retained upstream hashes.

## Historical exports

Historical upstream CSV labelled both classes `0`.
It paired the class arrays with `zip`, which truncated the larger class,
and it discarded execution order.
Those files cannot recover the lost information.
This does not show an error in the in-memory statistic.

The new collection bookkeeping is outside each timed closure,
but it can affect the measurement environment.
Collect target evidence again for the new binary.
Do not carry historical timing claims over to it.
