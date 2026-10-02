# DudeCT timing export

Run `just ct-test` to check the exporter and the report validation.
Run timing cases through `just ct-dudect` or `just ct-full`.

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
- The local patch records the class order after the timed closure,
  and exports every sample in that order.
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
