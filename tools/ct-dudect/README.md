# DudeCT timing export

Run `just ct-test` to check the exporter and report validation. Run timing cases
through `just ct-dudect` or `just ct-full`.

The CSV columns are `benchname,sequence,class,runtime_ns`. Each completed
`run_one` contributes one row. Class `0` means left; class `1` means right.
`sequence` starts at zero per case and follows execution order, including
unequal class counts. Durations are the same integer nanoseconds supplied to
the in-memory statistics. Sequence continues across batches in continuous mode.
It records order, not wall-clock timestamps or pauses between observations.

The report requires every requested observation, both classes, and contiguous
sequence numbers. Its `raw_csv` counts and artifact hash bind the retained CSV;
`dudect_runner_sources` binds the local runner sources. Report schema 3 rejects
legacy exports rather than treating them as complete measurements.

## Local runner patch

`vendor/dudect-bencher` retains the selected upstream 0.7.0 source and licenses.
`UPSTREAM.json` records the crate archive identity and original file hashes.
The local patch records class order after the timed closure and exports every
sample using that order. Buffered output is flushed before reporting a result.
Macro documentation whitespace is normalized. The statistical implementation
in `src/stats.rs` is unchanged. The export tests
exercise unequal classes, deterministic duration/order fixtures, and replay of
actual measured samples through the same statistic.

The local dependency is unpublished and used only by this evidence harness.
Vendored code is excluded from automatic dependency-requirement rewriting and
first-party style lints; its exporter has explicit regression coverage through
`just ct-test`. Review any runner update against the retained upstream hashes.

Historical upstream CSV labelled both classes `0`, paired class arrays with
`zip` (truncating the larger class), and discarded execution order. Those files
cannot recover the lost information. This does not establish an error in the
in-memory statistic. New collection bookkeeping is outside each timed closure
but can affect the measurement environment; rerun target evidence for the new
binary rather than transferring historical timing claims.
