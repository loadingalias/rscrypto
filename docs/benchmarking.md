# Benchmarking

Benchmark the exact primitive, operation, input size, feature set,
and target that you plan to deploy.
A crate-wide aggregate is not a deployment result.

## Read published results

[`benchmark_results/OVERVIEW.md`](../benchmark_results/OVERVIEW.md) owns the published summary.
Keep raw results and run metadata for every published claim.
A local measurement without that evidence applies only to the machine that made it.

## Run records

Each new local or development-machine run writes its own directory:

```text
benchmark_results/criterion/<run-id>/
```

Each run records:

- the literal requests and the log;
- source-file hashes;
- the compiler and Cargo identity;
- the build and runtime environment;
- CPU and operating-system information;
- the raw `criterion/` data;
- `status.txt`.

Successful discovery adds the resolved case plan.

`status.txt` starts as `running`.
It becomes `complete` or `failed`, with the exit code of the execution.
`complete` needs, for every planned case, the case identity, the statistical samples, and the estimates.
It also needs the comparison estimates when a baseline was given for that case.
If the process succeeds but results are missing, the run fails.
A run that is interrupted abruptly can stay `running`.
If packaging fails, the run is kept and no checksum is published.
Source hashes identify a dirty worktree.
They do not replace keeping the source.

### Export and collect

Export a run only when you need a portable artifact:

```bash
just bench-export benchmark_results/criterion/<run-id>
```

This archives the run, including failed-run evidence, under `benchmark_results/.transfers/` with a SHA-256 checksum.
Export refuses to overwrite an existing archive.
`output_dir=<path>` changes the results root, and keeps the `criterion/` and `.transfers/` layout.

On development machines, use the default root.
Collection exports the selected run, then downloads it:

```bash
just ssh-collect-bench <target> <run-id> <new-local-directory>
```

Collect before you destroy the machine.
The destination must not exist yet.
Historical result directories, named by date, operating system, and architecture, do not change.

### Compare with a baseline

Each run starts with new Criterion output.
To compare with a completed run, select it explicitly:

```bash
just bench sha256 baseline=benchmark_results/criterion/<previous-run-id>
```

- Only validated Criterion `base` data for matching cases and configurations is copied into the new run.
  The previous run does not change.
- Compatibility includes the Cargo command, compiler, manifest profiles,
  Cargo configuration file hashes, CPU identity, build and runtime controls,
  and effective Criterion settings.
- The source revision and the watchdog budget are recorded, but they are not part of compatibility.
- A baseline needs a resolved plan and verified measurements.
- If no configuration or case matches, the command fails.
  Otherwise, cases without a match run without a comparison.
- Matching metadata does not prove the same thermal, power, or system-load conditions.
  Control those before you interpret a comparison.

### Speedup

```text
speedup = comparison_time / rscrypto_time
```

A value above `1.00x` favors `rscrypto`.
A value below `1.00x` favors the comparison.
Summary tables treat `0.95x` through `1.05x` as a tie.

[`.config/benchmark-matrix.json`](../.config/benchmark-matrix.json) owns the benchmark targets, required features,
selector names, filters, and comparison declarations.
The benchmark source owns each timed operation.
Inspect both before you claim equivalent work.

## Comparison rules

A comparison measures one task, such as "verify a signature with a key parsed once".
It does not compare code paths.
Every comparison follows these rules:

1. A group measures one task.
   When an operation has more than one task, the group name says which one,
   for example `ed25519/verify/reuse-key` and `ed25519/verify/import-key`.
1. Each library joins a group with its best public API for that task, and with its best supported build.
   A library that has no API for the task stays out of the group,
   and the source comment at the group says so.
1. `rscrypto` runs in its default build.
   In each group and input, exactly one row is named `rscrypto`: its best public API for the task.
   Rows named `rscrypto-<variant>` show other `rscrypto` APIs. They never enter a ratio.
1. Differences that come from a library's design, such as what it caches or how it selects instructions, are results.
   Differences that come from the harness, such as a weaker API, missing build flags, or skipped validation,
   are defects.
1. Every row checks its output against an independent oracle before it is timed; see [Output checks](#output-checks).
   Where a task includes validation, the check also requires the timed code path to reject a malformed input,
   when that is practical.

The source comment at each group states the task, what is timed,
what each library caches or validates, and any remaining API-cost difference.

### Scorecard

The catalog's `comparisons` map declares each comparison:
its task, the target that measures it, a case pattern, a family, and whether it is in the headline.
The headline holds each primitive's primary operations.
Streaming, batch, collection-hasher, construction, and parsing comparisons are reported by family, outside the headline.

```bash
just bench-compare x86_64-linux-intel=benchmark_results/criterion/<run-id> --json scorecard.json
just bench all --list > cases.txt
just bench-compare --cases cases.txt
```

- Each argument is `PLATFORM=RUN_DIRECTORY`.
  Repeat a label to combine runs from one host. The scorer rejects runs from different hosts under one label.
- One comparison point is one comparison, variant, and input on one platform.
  Its ratio divides the fastest external median by the `rscrypto` median.
  The JSON output also gives an interval from the two medians' confidence intervals.
- A comparison can name `competitor_benches`, which are other builds whose external rows also count.
  The `rscrypto` row always comes from the comparison's own target.
- A comparison can restrict `architectures` when no external library offers the operation elsewhere.
  Its required `availability` explanation names that limit. The rscrypto case still runs on every supported target.
  On an excluded architecture, the report retains it as a standalone measurement with the explanation, without a ratio.
  An unexpected external row there fails the report so the catalog must be updated when support changes.
- The scorer fails when a point lacks its `rscrypto` row or an external row,
  when a platform lacks a competitor build that applies to it,
  and when a measured group has external rows but no declaration.
  `--partial` reports the first kind as notes, for runs whose case filter omitted rows.
- `--cases` checks the declarations against a case listing without measuring.
  It uses this host's architecture; add `--architecture s390x`, for example, for a listing from another host.

The Bench workflow's `Score comparisons` job extracts and scores the run's own measurement archives.
It writes the tables to the job summary and keeps the JSON as the `bench-score-RUN-ATTEMPT` artifact.
Published summaries come from that output.

If only the comparison report failed after every native measurement job passed, dispatch Bench with
`rescore=RUN_ID/ATTEMPT`, for example `rescore=39106333869/1`.
This mode ignores the measurement inputs and starts only a report job.
It requires a completed Bench attempt from this repository's `main`, successful native jobs,
and exactly one retained archive for every such job. Only the source report may have failed.
It revalidates every retained measurement and requires complete comparison groups.
The summary identifies the measured revision. The JSON also records the revision of the corrected report.
Re-scoring an older run does not measure the current source revision.

### Competitor builds

Some libraries choose their SIMD code only at compile time.
xxhash-rust compiles its SSE2, AVX2, or AVX-512 code only when the build enables those target features.
Default benchmark builds pass no CPU flags, so on x86-64 it would run SSE2,
while `rscrypto` selects AVX2 or AVX-512 at run time.

- The `xxh3_v4` target builds the same binary with `-C target-cpu=x86-64-v4`, on x86-64 only.
- The `xxh3` comparison takes each competitor's fastest row from either build.
  `rscrypto` stays in its default build.
- `xxh3-baseline-build` keeps the default build only.
  It shows what runtime dispatch gives a binary built for the x86-64 baseline.
- `xxh3-v4-build` compares every library compiled for x86-64-v4.
- A binary compiled for CPU features that the host lacks refuses to start.

The catalog's `architectures` field limits a target to some architectures,
and `rustflags` builds a target as its own configuration with extra flags.
Profiling and code inspection do not apply catalog `rustflags`, so they reject those targets.

## Bench workflow

The [Bench workflow](../.github/workflows/bench.yml) runs only on manual request.
Select the revision with the GitHub branch selector, then fill in the form:

| Input        | Example                         | Meaning |
| ------------ | ------------------------------- | ------- |
| `benchmarks` | `sha256`                        | One catalog algorithm. |
| `benchmarks` | `sha256,blake3-mixed`           | More than one name, separated by commas or spaces. |
| `benchmarks` | `sha2`, `mldsa`, `auth`         | A whole benchmark target. |
| `benchmarks` | `hashes`, `checksums`, `mlkem`  | A catalog group. |
| `benchmarks` | `all`                           | Every default target. |
| `filter`     | `^sha256/rscrypto/64$`          | Limit each selected target to matching Criterion cases. |
| Platforms    | `x86_64-linux`, `riscv64-linux` | Check one box for each platform. |
| `quick`      | Checked                       | Run every selected case with Criterion's quick sampling and report coverage. |

The platform checkboxes are `x86_64-linux`, `aarch64-linux`, `x86_64-win`, `s390x-linux`,
`powerpc64le-linux`, and `riscv64-linux`. Only `x86_64-linux` is checked by default.
Each x86-64 platform runs separate Intel and AMD jobs.
`benchmarks` accepts the same names as `just bench`; see [Select cases](#select-cases).

The planner rejects unknown names, a form with no platform checked,
and targets that need an external prerequisite, such as `tls-provider`,
before it starts any build or measurement runner.
A case filter that matches nothing in a target fails that target's job during discovery.

Optional sampling fields override the shared Criterion settings in
[`.config/criterion.json`](../.config/criterion.json). Empty fields keep those settings.
The diagnostic checkbox enables diagnostic features for the selected targets.
It does not select other targets.

Use `quick` for a short validation run. It keeps the selected cases, production operations,
independent output checks, and ten-minute invocation limit. Leave `sample_size` and `warmup_ms`
blank in this mode; `measure_ms` controls Criterion's quick measurement window.
The local equivalent for one algorithm is `just bench hmac-sha256 quick=true measure_ms=10`.
Quick artifacts record their sampling mode, and the report gives case and comparison coverage
without performance rankings. Quick runs cannot be used as performance baselines or mixed with
regular runs in one report.

### Runners

A small planning job validates the request and groups the selected targets by primitive family.
The catalog gives each Criterion target one `family`:
`checksum`, `hashes`, `aead`, or `auth`, after the root modules, or `tls` for the provider comparison.
Each checked platform and selected family pair becomes one job.
The job runs one `just bench` for each selected target of its family, one after another.
A split target, such as `slhdsa`, runs one `just bench` for each of its catalog algorithms instead;
see [Time budget](#time-budget).
For example, `sha256,blake3,crc32c` on `x86_64-linux` runs four jobs.
`Measure x86_64-linux-intel / hashes` and `Measure x86_64-linux-amd / hashes` each run `sha2`, then `blake3`.
The two `checksum` jobs run `crc`.
The plan log prints the `just bench` command that each target runs.
Every platform receives the same plan.
A job skips a target that the catalog limits to other architectures, such as `xxh3_v4` outside x86-64,
and prints the skip.
After every measurement job passes, the `Score comparisons` job scores the run; see [Scorecard](#scorecard).

- AWS supplies fixed On-Demand instance types for Linux x86-64, Linux ARM64, and Windows x86-64.
  Both x86-64 operating systems run separate Intel and AMD jobs, with different artifact names.
- [`runs-on.yml`](../.github/runs-on.yml) defines the machine shapes.
  Benchmark preparation uses `bench-cross-build`.
  Measurement uses the `measure-*` profiles.
  They are sized separately from the CI and CT profiles.
- These profiles use the fixed processor families named in the catalog, with no AVX2-only baseline.
- IBM and RISE supply their existing native runners.
- macOS benchmarks run locally on the Apple Silicon Mac.

All preparation jobs start together, then all measurement jobs.
RunsOn starts an AWS host for every queued job and ignores `max-parallel`,
so the number of family jobs bounds the concurrent hosts.
Each RunsOn job has its own runner label, because a job can take any host whose label matches.
`scripts/test/cross_workflow_test.py` checks the largest possible run
against the account's Spot and On-Demand vCPU quotas.

- RISC-V, POWER, and IBM Z compile the selected targets of each family on x86-64
  with `--ci-cross-build TARGET`, in a `Prepare PLATFORM / FAMILY` job.
  Their native jobs use `--ci-cross-run` with the verified tools archive,
  then discover and measure the transferred binaries.
- Other Linux jobs use `--ci-bench`.
  Windows uses `-CiBench`.

No caches and no speed-regression gates are enabled.
Donated hosts can be shared, and fixed AWS instance types do not remove host noise.
Equal vCPU counts do not mean equal physical core counts.
Use the recorded CPU topology to interpret parallel results.

### Artifacts and limits

Each measurement job keeps `target/bench/` as one `bench-RUNNER-FAMILY-RUN-ATTEMPT.tar.gz` artifact,
for example `bench-x86_64-linux-intel-hashes-<run>-1.tar.gz`, with one run directory for each `just bench`.
GitHub stores the archive as uploaded, without zipping it;
zipped uploads of these directories crashed `upload-artifact` on the RISC-V runners.
The packing step runs after any job outcome, so failed and cancelled jobs keep their evidence.
It includes failed-run evidence, source and machine identity, the resolved case plan, logs,
and raw Criterion results.
A cross-target run omits its transferred binaries; `input/bundle.json` keeps their digests,
and the preparation artifact keeps the files.
Preparation jobs keep their evidence as `bench-prepare-PLATFORM-FAMILY-RUN-ATTEMPT`.

Each target, or each algorithm of a split target, runs as its own `just bench`,
so the ten-minute [time budget](#time-budget) applies to each one separately.
A target can still exceed its budget; narrow it with a case filter or sampling settings.
A job stops at its first failed `just bench`.
Preparation jobs time out after 30 minutes.
Measurement job timeouts (120 minutes, 180 on RISC-V) cover provisioning and transfer.
Each further `just bench` in a family job adds 10 minutes to either timeout.
One failed job cancels the rest of the run.
Manual dispatch is available after the workflow reaches the default branch.

### Cross-build preparation

Cross-build preparation never runs target code.
It seals each unique catalog build configuration, the source identity,
the compiler and linker evidence, the exact binary hash, and the requested sampling settings.

Native consumption rejects changed sources, settings, configurations,
or ELF architectures before discovery.
It uses the same case filtering, measurement, and result verification as an ordinary run,
without compiling again.
The retained input manifest identifies the build host.
Result compatibility records the measurement host and its runtime settings.
Cross-built and native-built results have different build identities for baseline comparisons.

The two invocations use matching selections and settings:

```bash
just bench ... target=TARGET prepare_archive=ARCHIVE
just bench ... target=TARGET run_archive=ARCHIVE
```

The same selections work locally:

```bash
just bench sha256 blake3
just bench hashes
just bench all
just bench sha2 'filter=^sha256/rscrypto/64$'
```

## Profile workflow

The [Profile workflow](../.github/workflows/profile.yml) runs only on manual request.
It takes one native Linux architecture and one curated primitive.

- The benchmark catalog maps each primitive to one benchmark target and one exact production case.
  It also owns the diagnostic-feature policy.
- The default selection is `aead/aes`.
  It profiles `aes-128-gcm/copy-and-encrypt/rscrypto/4096` for five seconds.
- The ML-DSA presets use ML-DSA-65: `auth/mldsa65-keygen`, `auth/mldsa65-prepared-sign`, and `auth/mldsa65-prepared-verify`.
  Signing is deterministic.
  Signing and verification do not include key preparation.
- The SLH-DSA presets are `auth/slhdsa-sha2-128s-sign`, `auth/slhdsa-shake-128s-sign`, and `auth/slhdsa-sha2-128s-verify`.

The GitHub UI shows the same two inputs as the CLI:

```bash
gh workflow run profile.yml --ref BRANCH \
  -f architecture=riscv64-linux -f primitive=hashes/sha256
```

How a capture runs:

1. The planning job validates the request before it reserves native hardware.
1. An x86-64 job cross-builds and seals the production benchmark.
1. The native job verifies the archive and discovers the case exactly once.
1. It records with `perf record`.
1. It writes a symbol hotspot report with `perf report --stdio --no-inline`.
   The report includes call paths when the runner supports them.
1. The GitHub job summary shows the report.
   The artifact keeps the exact binary, the raw `perf.data`, the report, and the machine identity.

Collector setup:

- Native setup prefers the runner's `perf`.
- The pinned RISC-V kernel has no matching Ubuntu tools package,
  so setup builds `perf` from the matching pinned upstream stable source.
- Other donated runners fall back to the pinned Ubuntu generic userspace tool.
- The selected collector version is recorded next to the kernel identity.
- A live capability probe decides whether the tool and the host can collect evidence.
- When runner policy denies access, native setup enables perf events.
  The benchmark stays unprivileged.

The workflow keeps the evidence for missing packages, denied permissions, unsupported events,
and partial captures.
Native reports and raw evidence stay downloadable for 30 days, also when capture fails.
Preparation and native capture each have a 20-minute cap.
Planning has a five-minute cap.
A newer request for the same architecture cancels an older request that is still running.
Inspect the uncertainty, and repeat matched measurements, before you make a performance claim.

## Output checks

Every Criterion row checks its output before Criterion times it.

- Rows register through `checked`, `checked_with`, or `checked_routine` in
  [`benches/common/criterion.rs`](../benches/common/criterion.rs).
- The check runs once per row, outside timing, when Criterion first runs the row.
  Listing runs no checks.
- The check receives the operation's second output, so reused state has already served one call.
- Every row, comparison rows included, must match an independent oracle:
  another implementation or a published vector.
  An independent verifier checks randomized outputs, such as hedged signatures and new keys.
- Two row families have no independent implementation to compare with.
  Ascon-CXOF128 rows reproduce a published vector and match rscrypto's incremental API.
  RapidHash collection hashers, which no external specification defines, match pinned known answers.
- A mismatch stops the run before that row is timed.
- A planned run fails when a measured row skipped its check.
  Each binary prints `Output checks: N of N measured rows passed`,
  and the Bench workflow adds one such line per binary to its job summary.

## Timed workload boundaries

### BLAKE3 file inputs

The optional `blake3_file` target measures `Blake3::update_reader`, four
caller-scheduled subtrees, upstream `update_mmap_rayon`, and standalone `b3sum`.
Discover its 40 cases
with `just bench blake3_file --list`. It is excluded from default runs because
it needs explicit file fixtures and Linux cache-residency checks.

The artifact printed by discovery accepts `--rscrypto-prepare-file DIRECTORY BYTES`
to create one new deterministic file and digest manifest, and
`--rscrypto-verify-file DIRECTORY BYTES` to check the three library operations
and the CLI when configured. The fixed
sizes are 1 MiB, 16 MiB, 256 MiB, 1 GiB and 10 GiB. Preparation refuses to overwrite
a fixture. Set `RSCRYPTO_BLAKE3_FILE_DIR` to that directory and
`RSCRYPTO_BLAKE3_FILE_CACHE_LOG` to the retained JSONL output path before invoking
`just bench blake3_file` with the desired case filter.

The `upstream-b3sum-4` rows require `RSCRYPTO_BLAKE3_B3SUM` and
`RSCRYPTO_BLAKE3_B3SUM_RECEIPT`: an executable and a retained build receipt for
the same upstream revision as the locked BLAKE3 dependency. `benches/blake3_file.rs` defines the required receipt fields. `--rscrypto-verify-b3sum` validates the receipt,
binary hash and version without hashing a file. A receipt attests its build;
retain the source and build logs that substantiate it. The shared benchmark
metadata hashes both external files, preventing baseline reuse after a change
at the same path. Discovery and filtered library rows require neither file.

Every iteration conditions the file outside timing and checks residency with
`fincore`. Cold cases require zero resident pages after file-scoped eviction
advice; warm cases require the entire file resident. Warm conditioning checks
residency first and reads the file once only if pages are missing, then checks
again. The log retains both observations when a read is needed. This does not
promise CPU-cache warmth, and results must not be pooled with the older method
that reread every warm file. A failed
check stops measurement and retains the observation. These are Linux page-cache
states, not claims about device or storage-service caches. A host must have enough
available memory for the warm fixture; a shared machine is unsuitable for cache
qualification. Retain the fixtures' manifests, cache log, host/storage identity,
activity observations and complete Criterion output with the result.

Timing includes open/seek, hash-state construction, reading, allocation and input
cleanup, finalization, and destruction. The subtree row also includes scoped
thread creation/join and public parent merges. A shared four-worker Rayon pool
is initialized before timing for upstream and the Linux AArch64 plain reader.
Pool startup is therefore excluded; subtree thread creation stays included.
The standalone CLI includes command construction, process and four-worker
startup, hashing, output capture and exit in every timed call. Its provenance
and correctness checks run outside timing.
The comparison reports those actual
caller costs. It enables mmap only in the upstream development dependency;
rscrypto's production reader uses ordinary reads. The benchmark source owns
these boundaries. This target does not qualify short-prefix streaming or keyed
file performance.

### BLAKE3 threads and builds

`rscrypto` BLAKE3 can use Rayon threads only with the `parallel` feature, which is not a default feature.
Two catalog targets build the same binary:

- **`blake3_serial`, without `parallel`:** single-thread groups under `blake3/serial/`:
  `hash`, `keyed`, `derive-key/context`, `derive-key/prepared-context`, `xof`, `xof-output`, `ordered`,
  and `ordered-alignment`.
  `xof-output` squeezes 1 KiB, 16 KiB, and 1 MiB from a 32-byte seed into a reused buffer.
  `bao-decode` verifies a whole Bao combined encoding of 1,024, 1,025, 65,536, or 1,048,576 bytes from memory
  against bao 0.13.1, the format's reference. `bao` calls its one-shot `decode`, which returns a new `Vec`;
  `bao-reader` and `rscrypto` read their decoders to EOF into a reused buffer.
  Each check also requires the timed path to reject the encoding with its last byte changed.
  Upstream rows use its one-shot or `Hasher` APIs, which are single-threaded.
  The headline uses the `hash` and `keyed` groups, because a default install hashes on one thread.
- **`blake3`, with `parallel`:** groups that allow threads, under `blake3/parallel/`:
  `hash`, `keyed`, `derive-key/context`, and `xof`.
  `rscrypto` threads when its CPU profile's cost model admits the input.
  `blake3` is upstream's single-threaded API, and `blake3-rayon` is `Hasher::update_rayon`.
  At each size, the faster upstream row counts.
  Both libraries use the global Rayon pool, which starts before timing.
  Report these results with the host's core and thread topology.
- The `blake3` build also runs `batch`, `mixed-batch`, and `streaming`. No row of those groups uses threads.
- `derive-key/context` hashes the context string in every timed call, on both sides.
  `derive-key/prepared-context` hashes it before timing.
  There, `rscrypto` passes a `Blake3DeriveContext`; `rscrypto-cached` calls `Blake3::derive_key`,
  which reuses the context from a per-thread cache; and upstream builds a hazmat `Hasher` from the context key in each call.

### BLAKE3 mixed batches

`just bench blake3-mixed` compares batch and serial rscrypto calls with locked
upstream serial calls over the same ordered inputs. Its 54 rows span plain,
keyed and derive-key modes, three length mixes, and 16 or 65 inputs. Fixture
allocation, correctness checks and derive-context hashing stay outside timing;
production construction, internal cleanup and output writes remain timed.
Each library retains its own cleanup policy. Caller output storage is reused.

`RSCRYPTO_BLAKE3_BENCH_ISA` selects `auto`, `portable`, `sse41`, `avx2`, `avx512`
or `neon` before cached detection. The override only removes capabilities and
rejects unavailable tiers. Upstream remains explicitly labeled `blake3-auto`.
Process output records capability masks; optional `--diag` also checks the
selected bulk backend. Native performance qualification is separate from
compilation, case discovery and output equality.

### Other operation boundaries

Choose the timed boundary from the question that the workload answers.
Write it next to the benchmark group in the source.
Include input restoration, allocation, key or state construction, output handling, and destruction.
Put material included or excluded work in the case identity.
Implementation names must identify the library or backend that is actually called.
A renamed boundary starts a new baseline.

- **Reusable-buffer operation:** allocate storage and prepare reusable state outside timing.
  Time the operation on that state.
  If the operation needs fresh state, describe any batched setup explicitly.
  Do not call its allocation part of the measured operation.
- **Copy plus operation:** inside timing, restore the input into preallocated storage,
  then run the operation on it.
  Use `copy-and-…` in the operation name.
  Use this boundary to measure the cost of keeping an immutable source message.
- **Complete application operation:** include the real lifecycle under study, and name its stages,
  for example `copy-and-construct-and-seal`.
  State which application costs are still excluded.
  Constructing a cipher does not mean that packet allocation, entropy, or transport is included.

Do not move setup out of timing only to get a smaller number.

- `iter` times the work and destruction inside its closure, and the destruction of its return value.
- `iter_batched` excludes the setup closure and defers destruction of the returned output.
  Consumed inputs can still be destroyed inside the timed closure.
- `iter_batched_ref` also defers destruction of the setup object.

The P-256 ECDH batches prepare new consumed keys outside timing.
RapidHash map-insertion batches allocate empty maps outside timing.
These boundaries are different from the timed buffer restoration in AEAD rows.

### AEAD rows

The AEAD `copy-and-encrypt`, `copy-and-decrypt`, `copy-and-seal`, and `copy-and-open` groups reuse preallocated buffers and cipher contexts.

- **Timed:** input restoration, cryptography, and per-call output handling and cleanup.
- **Not timed:** fixture generation, the first buffer allocation,
  and construction and destruction of the reusable context.
- Rows labeled `appended-tag` copy or produce the combined ciphertext-and-tag form.
  Other rows use detached tags.
- Throughput counts message bytes, not restoration traffic or tag bytes.
- AES-SIV `copy-and-construct-and-seal` also constructs and destroys a cipher in each iteration.
- `copy-and-construct-and-encrypt` measures one message per key for the other AEADs, at 64 bytes and 1 KiB:
  each call builds a cipher from the key bytes, restores the input, encrypts with a detached tag, and destroys the cipher.
  ring and aws-lc-rs build an `UnboundKey` and a `LessSafeKey`; the aegis crate binds the nonce when it builds its cipher.
  Decryption builds the same key schedule and has no separate group.

These are not cryptography-only measurements,
and they are not complete packet-processing measurements.

### QUIC header protection

The groups compare the RFC 9001 header-protection APIs of `rscrypto`, ring, and aws-lc-rs.
RustCrypto has no such API, so its `rustcrypto-composed` rows build the mask from `aes` and `chacha20`.

- **`construct`:** build a reusable context from borrowed key bytes, then destroy it, in each timed call.
  `rscrypto` also copies the bytes into its typed key, and zeroizes that key and the context.
  ring does not zeroize.
  aws-lc-rs builds an unused AES decrypt schedule and zeroizes both AES schedules.
  RustCrypto `aes` is built without its `zeroize` feature.
  RustCrypto has no ChaCha20 construct row, because it has no key-only ChaCha20 object.
- **`mask`:** compute one five-byte mask with a context built outside timing.
  ring and aws-lc-rs take a slice and return a `Result`.
  The composed AES row encrypts a full block. The composed ChaCha20 row keys a new core on every call.
- Every row must reproduce the RFC 9001 Appendix A.2 and A.5 masks before timing,
  plus the bench-sample mask on which RustCrypto and ring agree.
  AES-256 has no published vector, so that agreement is its only oracle.

### Other primitive rows

- Byte-oriented one-shot groups share one size matrix: 0 and 1 byte, 32 bytes, and powers of two from 64 bytes to 1 MiB.
  Hash groups add the lengths at which their primitive changes regime, in the same group:
  - SHA-224 and SHA-256: 55 and 56 bytes, the longest message whose padding fits one block and the shortest that does not.
    SHA-384, SHA-512, and SHA-512/256: 111 and 112.
  - SHA-3 and SHAKE: one byte below the rate, and the rate (143/144, 135/136, 103/104, 71/72, 167/168).
  - XXH3-64 and XXH3-128: 16/17, 128/129, and 240/241, where the short-input algorithm changes.
  - BLAKE2: 65 and 129, one byte past each block size. BLAKE3 `hash` and `keyed`: 65 and 1,025, one byte past a block
    and a chunk.
- Ed25519 and ECDSA verification has two tasks.
  - `verify/reuse-key` builds each library's key object outside timing and times verification.
    `rscrypto` and Dalek reuse the decoded Ed25519 point. `rscrypto` reuses its ECDSA verification table,
    and RustCrypto reuses a `VerifyingKey`.
    aws-lc-rs reuses a `ParsedPublicKey`, which saves parsing and allocation.
    AWS-LC still decodes the Ed25519 point in each call, and re-encodes an ECDSA signature as DER in each call.
    ring and dryoc have no API that reuses a parsed key, so they are absent.
  - `verify/import-key` gives every library the same encoded key and signature bytes,
    and times key import and validation, signature parsing, verification, and destruction,
    as a TLS certificate or handshake check does.
    `rscrypto` also builds the ECDSA verification table that key reuse amortizes. aws-lc-rs allocates an `EVP_PKEY`.
  - Each check also requires the timed path to reject a key that is not a curve point.
  - The diagnostic `verify-phase/portable-double-scalar` and `verify-phase/aarch64-asm-double-scalar` rows
    include portable public-point decoding and result encoding; they do not measure cached-key verification alone.
- ECDSA `sign/rscrypto` is the default `try_sign` API, which needs no entropy source.
  `rscrypto-blinded` returns the same signature through the blinded path.
- ECDH agreement has two tasks.
  `agreement/parsed-peer` parses and validates the peer key outside timing, and times the scalar multiplication.
  aws-lc-rs clones its `ParsedPublicKey` in each call, because `agree` consumes the peer.
  ring and libcrux have no parsed-peer API, so they are absent.
  `agreement/encoded-peer` gives every library the NIST peer key as identical uncompressed SEC1 bytes,
  and also times parsing and validation.
  X25519 `diffie-hellman/encoded-peer` builds the peer key from its 32 bytes inside timing in every row.
  Dalek also runs its contributory check.
  Each check also requires the timed path to reject an invalid peer.
- ECDH `key-generation` turns a caller scalar into a private key, without deriving the public key.
  `public-key` derives the public key from a private key that the caller holds.
  `public-key-from-secret` imports the scalar bytes and derives the encoded public key.
  aws-lc-rs takes part only in `public-key-from-secret`, because its key import already derives and caches
  the public point. libcrux has no key object, so it is absent from `public-key`.
  ring cannot import a scalar, so it is absent from `public-key-from-secret`.
- The ChaCha diagnostic group `chacha20-copy-and-xor` restores the message and applies the keystream in the timed closure,
  with a reused allocated buffer.
  It is not a complete AEAD operation.
- CRC `streaming` checksums 1 MiB through updates of 64 bytes or 4 KiB, with each library's incremental API.
  CRC `combine` joins the checksums of a 1 KiB part and a second part of 64, 65,535, or 16,777,215 bytes.
  Those lengths have every low bit set, which gives square-and-multiply combination its most multiplications.
  crc32fast combines two hasher states built from each checksum and length; those constructions stay timed.
  CRC-24/OpenPGP has no comparison library with combination, so its group has only the `rscrypto` row.
- SHA-3 and SHAKE rows compare RustCrypto (`sha3`, and `shake` for SHAKE), AWS-LC, and tiny-keccak.
  RustCrypto's Keccak selects the AArch64 SHA3 instructions at run time.
  AWS-LC runs s2n-bignum Keccak assembly on x86-64 and AArch64, and portable C elsewhere.
  `aws-lc-rs` rows use its digest API for SHA3-256, SHA3-384, and SHA3-512.
  `aws-lc` rows call AWS-LC's one-call `EVP_Digest` through aws-lc-sys for SHA3-224 and SHAKE,
  which aws-lc-rs does not expose; it allocates and frees its hash state in each call.
  SHA3-224 streaming has no AWS-LC row.
- SHAKE `xof-output` absorbs a 32-byte seed and squeezes 1 KiB, 16 KiB, and 1 MiB into a reused buffer.
  The one-shot SHAKE groups squeeze 32 bytes. dryoc also joins `xof-output`.
- Poly1305 `authenticate` reads immutable fixture bytes, sets up a one-time key, and returns a tag,
  without restoring a message buffer.
  RustCrypto `poly1305`, dryoc `crypto_onetimeauth`, and AWS-LC's public `CRYPTO_poly1305_*` API through aws-lc-sys
  are the comparison rows. ring and aws-lc-rs expose no standalone Poly1305.
  Cleanup policies differ: `rscrypto` clears its key and state, and AWS-LC leaves its caller's state uncleared.
- BLAKE2 `short-oneshot` and `short-keyed` keep only the 16-byte and 128-byte inputs that the main size matrix does not have.
  `single-update` measures construction, one update, and finalization at the small sizes.
  It is different from the multi-chunk streaming workload.
  Plain parameter-group duplicates are removed.
  The main one-shot rows are the baselines for salt and personalization hashing.
  All of these are complete hash operations, not isolated host overhead.
- BLAKE3 `ordered` rows compare identical ordered bytes through one-shot, one-update,
  and two-update calls. Prefix/bulk and bulk/suffix cases have separate inputs;
  reversing their byte order is not an equivalent comparison. Construction, output,
  and destruction stay timed; fixture allocation does not. Derive rows hash the
  context string in every call, on both sides. These groups run in `blake3_serial`.
  Upstream's `zeroize` feature is disabled, so cleanup costs are not equivalent.
  Every measured operation is checked against upstream before timing.
- BLAKE3 `ordered-alignment` repeats selected ordered cases with identical bytes
  at addresses congruent to 0 and 1 modulo 64. It retains the `ordered` timing
  boundary. Diagnostic `leaf-alignment-cvs` rows call existing full-chunk workers
  with independently controlled input and output alignment. They include wrapper
  validation and consume every CV, with allocation outside timing. Every CV is
  checked against upstream's non-root hash at its chunk offset.
  These component rows do not measure public dispatch or complete hashing.
- HKDF `expand` builds every pseudorandom key outside timing, from the same PRK bytes.
  aws-lc-rs uses `Prk::new_less_safe`, because its `Salt::extract` only stores the inputs
  and would run Extract again inside every `fill`.
  AWS-LC's `HKDF_expand` still keys HMAC in each call: two more compressions than the libraries that keep a keyed state.
- PBKDF2 rows derive into a reused output buffer and time password keying, every iteration, and the output.
  Below the password policy (1, 100, and 1,000 iterations), `rscrypto` calls `derive_key_primitive`,
  because the policy API rejects those counts.
  At the policy minimum (600,000 iterations for SHA-256 and 220,000 for SHA-512) with one output block,
  `rscrypto` calls the policy API `derive_key`, which also checks the iteration count and salt length.
  The `verify` groups at the same counts also compare with a stored key: `rscrypto` calls `verify_password`,
  which also bounds the work, and ring and aws-lc-rs call `pbkdf2::verify`, which check neither.
  RustCrypto's `pbkdf2` has no raw-key verification API, so it is absent there.
  Each verification check also requires the timed path to reject a stored key with one changed byte.
- RSA `verify-*/reuse-key` prepares the key outside timing, and each call verifies one signature.
  `rscrypto` reuses a scratch that caches R²; `rscrypto-oneshot` rebuilds it in each call.
  aws-lc-rs reuses a `ParsedPublicKey`, which caches its Montgomery context, and builds an `EVP_PKEY_CTX` in each call.
  RustCrypto keeps only n and e, and converts the signature in each call.
  ring has no prepared-key API, so its `RsaPublicKeyComponents` row rebuilds the modulus in each call.
- RSA `verify-*/import-key` parses identical PKCS#1 bytes, validates the key, and verifies, in each call.
  All four libraries reject non-minimal DER and an even modulus.
  `rscrypto` requires e = 65537 and its policy's size range.
  ring, AWS-LC, and RustCrypto accept odd e from 3 to 2³³−1.
  RustCrypto stops at 4096 bits; aws-lc-rs checks its 2048–8192-bit range only when it verifies.
- RSA `parse-spki` parses and validates identical SPKI bytes. aws-lc-rs tries PKCS#1 first and copies the input.
  ring has no SPKI parser.
  RustCrypto's parser stops at 4096 bits. At 8192 bits, AWS-LC is the only external parser in this suite.
  The suite excludes AWS-LC on POWER and IBM Z, so those platforms retain the rscrypto 8192-bit measurement
  without a comparison; the catalog still requires the external row on x86-64, ARM64, and RISC-V.
- RSA signing and OAEP name their entropy: `fixed` is a caller blinding pair,
  `caller-entropy` is a fill closure or RNG that writes fixed bytes,
  and `fresh-entropy` is new randomness in every call.
  `rscrypto` and RustCrypto use `getrandom`. aws-lc-rs uses AWS-LC's DRBG and a cached blinding pair.
  ring signs without blinding. RustCrypto and aws-lc-rs allocate their outputs inside timing.
  PKCS #1 v1.5 signing pads deterministically, so its `fresh-entropy` group draws only blinding randomness,
  and every row must produce RustCrypto's unblinded signature.
  Key generation stays outside the headline, because prime search makes its time vary widely.
- RSA `pkcs1v15-encryption/decrypt/fresh-entropy` decrypts a valid Wycheproof RSAES-PKCS1-v1_5 vector
  with fresh blinding. ring has no RSA decryption.
  `rscrypto` applies implicit rejection: every call hashes the private exponent, keys HMAC-SHA256 with the ciphertext,
  and expands a modulus-length substitute message, which it returns only for invalid padding.
  RustCrypto 0.9 and AWS-LC return an error for invalid padding, so they skip that work.
  The difference is a design result, and the group stays outside the headline as a legacy operation.
- HMAC-SHA2 verification has two tasks, at 32 and 1,024 bytes.
  `verify/reuse-key` prepares each key outside timing: `rscrypto` absorbs into its keyed state, verifies, and resets it;
  RustCrypto clones its keyed state; aws-lc-rs and ring call `hmac::verify` with a prepared `hmac::Key`.
  `verify/import-key` keys from bytes in every call; `rscrypto` calls `verify_tag`.
  Each check also requires the timed path to reject the tag with its last bit flipped.
- ECDSA `signature-der/encode` and `signature-der/decode` time the RFC 3279 `Ecdsa-Sig-Value` codec alone,
  against RustCrypto. ring and aws-lc-rs parse DER only inside verification, so they are absent.
  Every encoding must equal RustCrypto's and verify under ring's DER verifier.
  Every decoding must return the fixed-width signature, and the timed path must reject the encoding without its last byte.
- HMAC-SHA3 rows time one tag from a key set up once outside timing.
  `rscrypto` and `aws-lc` (AWS-LC's C `HMAC_CTX` through aws-lc-sys) both cache the Keccak states
  after the ipad and opad blocks. `aws-lc` reuses them through `HMAC_Init_ex` with a null key, inside timing.
  `rustcrypto-simplehmac` caches the ipad state and the raw opad block, and absorbs the opad block again for every tag.
  That is one more Keccak permutation per tag: 1.5x the work below the rate, and under 1.01x from 16 KiB.
  RustCrypto's `hmac::Hmac` would cache the opad state, but `sha3` 0.12 does not expose the block-level core it needs.
  The aws-lc-rs and ring APIs have no HMAC-SHA3.
  The `aws-lc` row exists only where aws-lc-sys is a dev-dependency.
- Ascon's `rscrypto/scalar-loop` rows compare repeated calls to the `rscrypto` scalar API with its batch API.
  They do not compare with an external library.
- RapidHash `rapidhash-v3-64` and `rapidhash-stream` rows compute the same C++ RapidHash V3 function in both libraries.
  The `rapidhash-buildhasher`, `rapidhash-hash-one`, and `rapidhash-hashmap` rows compare three different
  collection hashers. rscrypto mixes buffered integer writes in before each byte write and ends with one mix.
  Upstream `rapidhash-fast` keeps integers buffered and has no avalanche round, so it does less work.
  Upstream `rapidhash-quality` adds avalanche rounds, so it does more.
  Quote an rscrypto collection result against both rows.

## ML-KEM and Argon2 comparison contracts

Compare only rows in the same operation group, with matching build and host identities.
These contracts replace the old ML-KEM IDs, and the Argon2 IDs without `salt16-raw32`.
Do not reuse those old measurements as baselines.
The effect of the old mismatches on the reported ratios has not been measured.

### ML-KEM

ML-KEM uses fixed 64-byte key-generation seeds,
and fixed 32-byte encapsulation randomness in the `derand` groups.
Fixture construction is not timed.
Decapsulation uses the same key material and ciphertext in every implementation.
It consumes no entropy.
Each parameter set has its own groups.

| Operation suffix | Timed input and preparation | Timed output |
| --- | --- | --- |
| `keygen/derand-encoded` | Seed from the caller; generation and export | Encoded public key and expanded secret key as fixed byte arrays |
| `keygen/internal-entropy-encoded` | AWS-LC generation, internal entropy, and export | The same key encodings as fixed byte arrays |
| `encapsulate/derand-reuse-key` | Each library's best reused key, checked once outside timing; randomness from the caller | Ciphertext array and 32-byte shared-secret array |
| `encapsulate/derand-reuse-encoded` | Reused `rscrypto`, libcrux, or fips203 encoded-key wrapper; caller randomness; per-call decoding stays timed | The same ciphertext and secret arrays |
| `encapsulate/derand-import-encoded` | Identical public-key bytes; each API's import, validation, and encapsulation with caller randomness | The same ciphertext and secret arrays |
| `encapsulate/internal-entropy-import-encoded` | The same public-key bytes; AWS-LC import and internal entropy | Ciphertext and secret converted to fixed arrays |
| `decapsulate/prepare-key` | `rscrypto` only: the preparation that its `reuse-key` row amortizes | A prepared key |
| `decapsulate/reuse-key` | Each library's best reused key, and the ciphertext | 32-byte shared-secret array |
| `decapsulate/reuse-encoded` | Reused `rscrypto`, libcrux, or fips203 encoded-key wrapper and typed ciphertext; per-call decoding stays timed | 32-byte shared-secret array |
| `decapsulate/import-encoded` | Identical expanded secret-key and ciphertext bytes; each API's import, validation, and decapsulation | 32-byte shared-secret array |

- The `reuse-key` groups are the task view: the keys cache different work, by design.
  - `rscrypto`'s prepared key holds the decoded key, the expanded public matrix, and H(ek),
    plus the decryption mask for decapsulation.
  - RustCrypto's key holds the decoded key and H(ek), and samples the matrix in each call.
  - libcrux and fips203 keys hold the encoded bytes, so decoding and matrix sampling stay timed.
  - AWS-LC cannot take caller randomness, so it joins only decapsulation.
    There it holds the encoded key and runs the hash check in each call.
- The `reuse-encoded` groups compare equal key depth.
  There, `rscrypto` also runs the FIPS 203 key check in every call, and libcrux and fips203 checked once outside timing.
- Import rows run the FIPS 203 checks inside timing.
  libcrux import does not check, so its rows call `validate_public_key` or `validate_private_key`.
  `rscrypto` runs the check twice: in `try_from_slice`, and again in `Kem::encapsulate` or `Kem::decapsulate`.
  Each import check also requires the timed path to reject a key with a coefficient of at least q, or a changed H(ek).
- All ML-KEM rows include output conversion and the destruction of per-call objects.
  `Criterion::iter` includes the destruction of returned arrays.
- Reused keys are constructed and destroyed outside timing.
  Imported keys are constructed and destroyed inside timing.
- Internal allocations and their cleanup stay timed,
  including the ciphertext and shared-secret buffers and key objects that AWS-LC allocates.
- The harness does not supply reusable scratch storage,
  and it does not equalize the cleanup policies of the libraries.
  The rows measure the selected APIs on valid inputs.
  They do not show identical validation or zeroization guarantees.
- RustCrypto takes part in encapsulation and in reused-key decapsulation.
  Its decoded secret key is built from the fixture seed outside timing.
- Expanded-key generation and import rows use implementations that have expanded-key APIs.

Correctness checks before timing:

- Each deterministic ML-KEM row runs its actual timed closure once outside timing,
  and checks the complete output against the shared fixture.
- Key-generation checks compare both encoded keys.
  Encapsulation checks compare the ciphertext and the secret.
  Decapsulation checks compare the secret.
- AWS-LC's randomized generation and encapsulation closures are checked
  through decapsulation by another implementation.
- A mismatch stops execution before that row is timed.
- Discovery lists identities only.
  It does not replace the correctness checks of the selected rows.

### Argon2 and scrypt

Argon2 comparison groups include `salt16-raw32` in their IDs.

- All rows use the same password, the full 16-byte salt, Argon2 version 0x13,
  the same memory, time, and lane parameters, and 32-byte raw output.
- They consume no entropy and do no PHC encoding.
- Parameter objects and caller-owned output buffers are prepared outside timing.
- Each call includes the selected API's scratch allocation, computation, and scratch cleanup.
- The output buffer is reused, and destroyed outside timing.
- Each library keeps its own cleanup policy.
- Untimed checks compare all 32 output bytes of every row with RustCrypto.
- Argon2 parallel-scaling rows use the same salt and output size, and change the lane count.
- scrypt and PHC fixtures are separate workloads.

The reused-memory groups (`argon2id-owasp/salt16-raw32-reused-memory` and `scrypt-owasp-reused-memory`) isolate work-memory reuse at the OWASP shapes.

- `rscrypto/fresh-allocation` calls `derive`,
  so its allocation, zero fill, computation, and cleanup are timed.
- `reused-memory` rows lend one buffer that is allocated and freed outside timing.
  `rscrypto` still clears every block it used, inside timing.
- `rustcrypto/reused-memory` calls `hash_password_into_with_memory`, built without RustCrypto's `zeroize` feature.
  It does not clear its buffer, so the rows do not do equal cleanup.
- Untimed checks compare all 32 output bytes with RustCrypto, after a call has already used the memory.

## ML-DSA comparison contracts

ML-DSA groups use a 32-byte message. Keys are built outside timing unless a group imports them.

- **Task groups:** `sign/reuse-key-deterministic`, `sign/reuse-key-randomized`, `verify/reuse-key`, and `verify/import-key`.
  They use an empty context, because aws-lc-rs takes none.
  Each library uses its best API, so how much each key caches is a design difference.
  - Verify rows parse the encoded signature inside timing. `import-key` also imports identical public-key bytes.
  - Randomized rows take 32 bytes from AWS-LC's RNG in each call, in every library.
    That group exists only where aws-lc-rs builds.
    RustCrypto signs through `sign_internal` with the empty-context prefix,
    because its randomized API needs a `rand_core` feature that the benchmarks do not enable.
  - `rscrypto-compact` decodes its key in every call, as AWS-LC does.
  - AWS-LC has no deterministic signing API, so it is absent from the deterministic group.
  - Each verify check runs the timed path. It must accept the signature,
    and reject a tampered and a malformed one.
- **Equal key-depth groups:** `encoded-key` and `prepared-key`, with an 18-byte context.
  Encoded-key rows decode the key in every call.
  Prepared-key rows cache the matrix and the transformed vectors.
  In `verify/prepared-key`, RustCrypto's signature is decoded outside timing, while `rscrypto` decodes inside.
- **`keygen/seed-encoded`:** turns a seed into the encoded public key and the expanded secret key.
  Generation, export, and destruction are timed.
- There is no signing import group: `rscrypto` and AWS-LC validate an imported secret key by recomputing
  the public key, and libcrux and RustCrypto do not. Such a group would compare validation policies.

## Measure locally

### Settings

[`.config/criterion.json`](../.config/criterion.json) supplies one configuration for every Criterion harness,
also for direct Cargo invocations: 20 samples, 100 ms warmup, 400 ms requested measurement time,
10,000 bootstrap resamples, 95% confidence, 5% significance, and a 1% noise threshold.
Benchmark groups must not override these settings.

`warmup_ms=`, `measure_ms=`, and `sample_size=` override the shared defaults for every selected case in that invocation.
Their `BENCH_` environment variables have lower precedence than explicit arguments.
Boolean controls reject unknown values and empty strings.

These are bounded development defaults, not a promise of statistical precision.
Inspect the confidence intervals.
Repeat a focused selection when the uncertainty cannot support the claim you want to make.
For slow operations, Criterion can extend the requested measurement window to collect the requested
samples.
For example, Criterion estimated about 34 seconds per RustCrypto SLH-DSA-SHAKE-192s signing iteration
on RISC-V in [run 39100888583](https://github.com/loadingalias/rscrypto/actions/runs/39100888583).
The full comparison exceeded the ten-minute limit even at ten samples and 10 ms sampling windows.
Quick mode validates those cases with fewer iterations; it does not qualify the regular statistical campaign.
Criterion 0.8.2 retains two quick samples and can derive the second from the first for a slow operation.
Those estimates do not establish uncertainty or a performance winner.
`argon2id` includes small, OWASP, and parallel workloads.
No opt-in for expensive workloads is needed.

### Time budget

`just bench` limits the whole pipeline to 10 minutes: build, discovery, measurement, analysis,
and result verification.
`just bench-structural` and `just profile` use the same limit, including their builds.
You can lower the limit, but it cannot be more than 600 seconds.

- Shutdown starts before the deadline.
  It keeps up to five seconds to save failed-run evidence, then stops the remaining child processes.
- A run that times out exits with status 124.
  Partial results are not a complete run.
- A plan whose requested sampling windows alone use up the budget is rejected before measurement.
  Build cost, analysis, and slow operations can still make a smaller plan reach the deadline.
- Direct Cargo invocation limits each Criterion harness separately.
  Use `just bench` to limit a selection that spans more than one harness, together with its builds.
  Use `just bench-structural` for the structural benchmark deadline.
- `just bench-mldsa-tail` and `just bench-provider-scorecard` are separate tools with their own timeouts:
  20 minutes per build and 60 per run for the ML-DSA tail, 10 minutes for each scorecard build and run.

A planned comparison must fit within this limit, including every repetition.
Do not split an over-budget comparison across repeated invocations to bypass it.
A comparison is one Criterion benchmark group: one operation of one algorithm, across its implementations.
A catalog target marked `"split": "algorithms"` runs in CI as one `just bench` per catalog algorithm,
which never divides a group.
SLH-DSA uses this, because its 24 FIPS 205 algorithms do not fit one budget on the slower platforms.
PBKDF2 uses it too: at the password-policy iteration count, one SHA-256 call takes about 1.25 s on the RISC-V runner,
extrapolated from its 1,000-iteration rows.
The `auth` target also splits by algorithm: its cold Windows build plus all 15 algorithms exceeded one budget
in [Bench run 39091310295](https://github.com/loadingalias/rscrypto/actions/runs/39091310295).
The invocations reuse the same compiled binary and preserve every case and sampling setting.
Every case of a split target must belong to exactly one of its algorithms; discovery fails otherwise.
RSA key generation has its own `rsa_keygen` target, separate from `rsa` verification, signing, and encryption.
CI runs `just bench rsa_keygen` with its own budget in the `auth` family job; its case IDs remain `rsa-{bits}-keygen/...`.
If required evidence cannot fit, record that requirement as unmet. A smaller future
experiment needs a prospectively stated question and gates; it does not complete
an older, larger qualification plan. Preserve historical frozen plans and failed
captures unchanged; their old budgets do not authorize rerunning them.

### Select cases

Select cases by catalog name. Benchmark targets, algorithms, and groups share one name space,
and the catalog rejects a name that two of them would share.
Matching ignores case and punctuation, so `password-hashing` selects the `password_hashing` target.

| Name kind | Example | Selects |
| --------- | ------- | ------- |
| Target    | `sha2`, `auth`, `blake3_file` | Every case in one benchmark target. |
| Algorithm | `sha256`, `ed25519`, `blake3-mixed` | The cases for one algorithm, inside its target. |
| Group     | `hashes`, `checksums`, `crc64`, `mlkem` | The algorithms and targets that the group lists. |
| `all`     | `all` | Every default target. Running with no name does the same. |

Default targets are the catalog's `required` Criterion targets.
A target with an `external` prerequisite is never a default target.
`tls_connections` needs OpenSSL 4.0.3 and is excluded from default runs.
`tls-provider` is its umbrella group for discovery.
Measure one `tls-handshake-<credential>` or scenario algorithm per invocation; the complete umbrella exceeds the run budget.
The [TLS benchmark guide](../tools/provider-bench/README.md) defines every family and its comparison boundary.

`quic_connections` has the same external prerequisite and is also excluded from default runs.
`quic-provider` lists its seven selectors: `quic-handshake`, `quic-packet`, `quic-header`, `quinn-connect`, `quinn-retry`, `quinn-address-token`, and `quinn-reset`.
Measure one selector per invocation.
The [QUIC benchmark guide](../tools/provider-bench/README.md#quic-and-quinn) separates TLS crypto,
packet protection and complete in-memory Quinn connections.

`filter=<pattern>` limits the selected algorithms or targets.
For example, `sha256 filter=rscrypto` stays within SHA-256, but `sha2 filter=rscrypto` searches the whole SHA-2 target.

- Repeat `filter=<pattern>` for more than one pattern.
- Each build configuration is listed once.
- A lightweight invocation of the same executable matches all patterns with Criterion's regex
  engine, without building benchmark fixtures.
- The matched cases run as one measurement process for each build configuration.
  The harness reads the resolved case set from a file and applies an anchored, escaped union filter.
- A pattern that matches no cases fails before measurement.
- Patterns are passed exactly as written.
  Commas are regex characters, not separators.
  Quote each argument for your shell.
- `BENCH_FILTER` supplies one literal pattern, in addition to any `filter=` arguments.
- Empty `filter=` values are rejected.
  For an unfiltered run, omit the argument.
- Positional selectors accept catalog names.
  Use `filter=` for raw regexes.
- An exact algorithm name selects only that algorithm.
  Use a group name, such as `crc64`, to select more than one.
- `blake2` is a target name, so it includes all implementations and operations, including the dryoc one-shot and keyed cases.
- SLH-DSA has one algorithm name for each FIPS 205 algorithm.
  `slhdsa-shake-192s` selects pure SLH-DSA-SHAKE-192s, and `hash-slhdsa-shake-192s` selects its HashSLH-DSA variant.

### Discover cases

Discover the actual cases before you choose a measurement scope:

```bash
just bench crc64-nvme --list
just bench sha2 --list
just bench blake3 --diag --list
just bench aead_kernels --list
```

`--list` builds the selected configuration and lists its cases.
It does not measure, create a run, or copy baseline data.
It uses the same filters and case deduplication as measurement.
Each row shows its benchmark binary, the exact case name, and its work class:

- `ordinary`: public operation and comparison workloads.
- `expensive`: high-cost workloads that the catalog declares,
  including password hashing, PBKDF2, and RSA private signing.
- `diagnostic`: internal components, backend experiments, and overhead probes.

The classes come from `.config/benchmark-matrix.json`.
They describe the intent of a workload, not its measured duration,
and they give no timing guarantee.
A diagnostic case can also be costly.
Classes do not block execution.

`--diag` (or `diag=true`) enables `diag` and the internal compiler cfg for the selected benchmark builds.
Some target configurations already need it.
The cases depend on the compiled features and the host capabilities.
Dedicated diagnostic targets, such as `aead_kernels`, need explicit selection.
`all` and runs with no name include only the default targets.

### Run

Run the narrowest useful case:

```bash
just bench sha2
just bench auth filter='^ecdsa-p256/'
just bench sha256 'filter=^sha256/rscrypto/\d+$' 'filter=^sha256/rscrypto/[0-9]{1,3}$'
just bench p256-ecdh
just bench p384-ecdh
just bench mlkem
```

Target names, including unfiltered `sha2`, run without a scope override.

- P-256 ECDH uses the `p256-ecdh` selector.
  Exact profiling uses the `auth` catalog target that owns it.
  Its rows compare caller-filled generation, public derivation, canonical SEC1 parsing,
  agreement with a parsed or an encoded peer key, and a TLS-shaped two-party roundtrip.
- P-384 ECDH uses the `p384-ecdh` selector, with caller-filled generation, public-derivation, parsing, and agreement rows.

The raw target results and the overview are the only performance record.

### Run files

- `requests.json` keeps the resolved target and filter requests and the run budget.
- `plan.json` records, for each configuration: the selected cases, the Cargo command and artifact,
  the executable hash, the compatibility evidence, the effective settings, the baseline cases,
  the execution command, and the output location.
- Raw results are under `criterion/<binary>-<configuration-id>/`, in Criterion's directory layout.
- `output.txt` is the build, discovery, and measurement log.
- `source.json` and `source-state.json` identify the source files and the worktree.

The runner verifies all planned measurements once before it marks the run complete.

The shared environment collector separates build inputs from runtime controls,
including Rayon thread controls, CRC backend overrides, and `RSCRYPTO_FORCE_AVX512`.
For the `auth` benchmark on an AVX2-capable x86-64 host,
`RSCRYPTO_BENCH_DISABLE_IFMA=1` removes IFMA from the detected capabilities before initialization.
This measures the production AVX2 Curve25519 fallback on IFMA hosts.
It does not change the separate Linux assembly paths.
Known runtime controls that are not set are recorded as explicit JSON nulls.
Benchmark plans and profile metadata carry the same compatibility evidence.

Criterion measures elapsed time.
`just bench-structural` uses Gungraun and Valgrind to count instructions and cache events on supported Linux hosts.
Those counts do not prove wall-clock speed.

## Profile locally

After a benchmark shows a specific cost, inspect it:

```bash
just profile sha2 --list
just profile sha2 'sha256/rscrypto/64' 10
just profile blake3 --diag --list
just perf-codegen sha2 -- --asm <function>
just perf-llvm-lines sha2 -- --filter <pattern>
```

- Profiling needs one exact case name.
- `--list` builds the selected target and lists its cases without recording.
- Capture checks that the name occurs exactly once,
  then runs that executable with only the resolved case selected.
- Unrelated workload groups skip fixture construction.
- The requested duration applies to that case.
  Process startup and profiler overhead add to the total elapsed time.

Each local Samply capture gets its own directory under `target/profiles/`.
It contains `profile.json.gz`, `cases.json`, `metadata.json`, a log, and source evidence.

- The metadata records the exact case, the executable path and SHA-256,
  the Cargo artifact description, the build and capture commands, compiler and tool versions,
  the build and runtime environment, and the capture outcome.
- The source evidence records input hashes, the revision, and the worktree status.
- Keep the matching executable and its symbols when you investigate a saved profile.

Transferred CI captures instead keep the verified input bundle, raw `perf.data`, the native `perf-report.txt`,
host and capability facts, metadata, status, and an outer sealed manifest.
Only a complete sampled capture exits successfully.
A partial, unavailable, or failed capture stays downloadable and keeps the workflow from passing.

## Build configuration

Benchmarking and profiling share the Cargo command and the CPU-flag policy.

- Both use the `bench` profile.
  It inherits the release optimization settings and keeps debug symbols, without stripping.
- Explicit `RUSTFLAGS` or `CARGO_ENCODED_RUSTFLAGS` take precedence.
  Otherwise, local macOS runs use `-C target-cpu=native`.
  Match these flags, the target, and the Cargo features when you compare with a deployment build.
- A target with catalog `rustflags`, such as `xxh3_v4`, appends them to those flags for its own build only.
- Each target uses its explicit catalog features with Cargo defaults disabled.
  Algorithm selectors, raw filters, and multi-target selection do not change this.
- Each distinct target configuration is built once, before its selected cases run.
- Profiling and code inspection use the same catalog target configuration.
  `--diag` enables the same feature and internal compiler cfg in each command.
- The `blake3`, `blake3_file`, and password-hashing targets enable `parallel`, where their workloads use it.
  `blake3_serial` builds the same binary without it, for the single-thread BLAKE3 comparisons.
- Cargo ignores the panic setting for benchmarks,
  so the release profile's `panic = "abort"` stays a difference
  ([Cargo profiles](https://doc.rust-lang.org/cargo/reference/profiles.html)).
