# Benchmarking

Benchmark the exact primitive, operation, input size, feature set, and target
you plan to deploy. A crate-wide aggregate is not a deployment result.

## Read published results

[`benchmark_results/OVERVIEW.md`](../benchmark_results/OVERVIEW.md) owns the
published summary. Raw Criterion output lives under:

```text
benchmark_results/<date>/<os>/<arch>/
```

Development-machine runs use an immutable, collision-free run directory under
`benchmark_results/criterion/<run-id>/` on the remote. The benchmark command
prints that run ID; collect it before destroying the machine with
`just ssh-collect-bench <target> <run-id> <new-local-directory>`. The collected
artifact includes CPU, OS, compiler, worktree, and source-file identity
evidence. A collection destination must not already exist.

Each result records its commit and platform. Compare individual rows when a
message size or operation matters. Do not transfer a result between CPU
families.

Speedup is:

```text
comparison_time / rscrypto_time
```

Above `1.00x` favors `rscrypto`; below `1.00x` favors the comparison. Summary
tables treat `0.95x` through `1.05x` as a tie.

[`.config/benchmark-matrix.json`](../.config/benchmark-matrix.json) owns
benchmark binaries, required features, aliases, and filters. The benchmark
source owns each timed operation. Inspect both before claiming equivalent work.

## Measure locally

Discover selectors with `just --list`, then run the narrowest useful case:

```sh
just bench bench=sha2
just bench crate=rscrypto bench=auth filter='^ecdsa-p256/'
just bench p256-ecdh
just bench mlkem
```

Criterion measures elapsed time. `just bench-structural` uses Gungraun and
Valgrind to count instructions and cache events on supported Linux hosts; those
counts do not prove wall-clock speed.

After a benchmark exposes a concrete cost, inspect it with:

```sh
just profile sha2 'sha256/64' 10
just perf-codegen --asm <function>
just perf-llvm-lines --filter <pattern>
```

Keep raw results and run metadata for any published claim. Local measurements
without that evidence are useful only for the machine that produced them.
P-256 ECDH uses the `p256-ecdh` benchmark alias. Its operation rows compare
caller-filled generation, public derivation, canonical SEC1 parsing, agreement,
and a TLS-shaped two-party roundtrip; raw target results and the overview remain
the only performance record.
