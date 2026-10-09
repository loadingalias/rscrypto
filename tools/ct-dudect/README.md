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
- Its `local_files` entry records the exact hash of the documented whitespace normalization in the macros.
  The identity test checks that hash, and keeps the original.
- The local patch records each duration and its class after the timed closure, in execution order,
  without branching on the class. It splits the durations by class only after the case completes,
  and exports every sample in execution order.
  Upstream's per-class push branched on the class right after the end timestamp.
  On Graviton5 that branch gave the classes different timings even when both ran identical work;
  label-independent recording removed the difference
  ([2026-10-09 record](../../benchmark_results/OVERVIEW.md#2026-10-09-graviton5-dudect-class-label-artifact)).
- Buffered output is flushed before a result is reported.
- The statistical implementation in `src/stats.rs` does not change.
- The export tests cover unequal classes, deterministic duration and order fixtures,
  and replay of real measured samples through the same statistic.

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
