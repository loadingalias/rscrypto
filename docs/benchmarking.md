# Benchmarking

Benchmark the exact primitive, operation, input size, feature set, and target
you plan to deploy. A crate-wide aggregate is not a deployment result.

## Read published results

[`benchmark_results/OVERVIEW.md`](../benchmark_results/OVERVIEW.md) owns the
published summary. New local and development-machine runs each write a unique directory:

```text
benchmark_results/criterion/<run-id>/
```

Each run records its literal requests, log, source-file hashes, compiler
and Cargo identity, build and runtime environment, CPU/OS information, raw `criterion/`
data, and `status.txt`. Successful discovery adds the resolved case plan.
Status starts as `running` and becomes `complete` or
`failed` with the execution exit code. `complete` requires every planned case's
identity, statistical samples, and estimates, plus comparison estimates when a
baseline was supplied for that case. A successful process with missing results
fails the run. An abruptly interrupted run may remain
`running`. Packaging failures retain the run and do not publish a checksum.
Source hashes identify dirty worktrees; they do not replace retaining the source.

Export only when you need a portable artifact:

```sh
just bench-export benchmark_results/criterion/<run-id>
```

This archives that run, including failed-run evidence, under
`benchmark_results/.transfers/` with a SHA-256 checksum. Export refuses an
existing archive. `output_dir=<path>` changes the results root, preserving the
`criterion/` and `.transfers/` layout. On development machines, use the default
root; collection exports the selected run before downloading it:

```sh
just ssh-collect-bench <target> <run-id> <new-local-directory>
```

Collect before destroying the machine. The destination must not already exist.
Historical date/OS/architecture result directories remain unchanged.

Every run starts with fresh Criterion output. To compare against a completed
run, select it explicitly:

```sh
just bench sha256 baseline=benchmark_results/criterion/<previous-run-id>
```

Only validated Criterion `base` data for matching cases and configurations is
copied into the new run; the previous run is unchanged. Compatibility includes
the Cargo command, compiler, manifest profiles, Cargo configuration file hashes,
CPU identity, build/runtime controls, and effective Criterion settings. Source
revision and the watchdog budget are recorded separately from compatibility.
Baselines require a resolved plan and verified measurements. The command fails
if there are no matching configurations/cases; unmatched cases otherwise run
without comparison. Matching metadata does not prove identical thermal, power,
or system-load conditions: control those before interpreting a comparison.

Speedup is:

```text
comparison_time / rscrypto_time
```

Above `1.00x` favors `rscrypto`; below `1.00x` favors the comparison. Summary
tables treat `0.95x` through `1.05x` as a tie.

[`.config/benchmark-matrix.json`](../.config/benchmark-matrix.json) owns
benchmark binaries, required features, aliases, and filters. The benchmark
source owns each timed operation. Inspect both before claiming equivalent work.

## Run a manual workflow

The [Bench workflow](../.github/workflows/bench.yml) runs only on manual
request. Select the revision with GitHub's branch selector, then choose:

| Input | Examples | Meaning |
| --- | --- | --- |
| `architectures` | `x86_64-linux` | One native platform. |
| `architectures` | `s390x-linux,powerpc64le-linux,riscv64-linux` | Any subset, separated by commas or spaces. |
| `architectures` | `all` | Linux x86-64, Linux ARM64, Windows x86-64, IBM Z, IBM POWER, and RISC-V. |
| `selection` | `sha256` | One catalog algorithm. |
| `selection` | `sha256,blake3` | Multiple algorithms. |
| `selection` | `hashes`, `checksums`, `auth`, `aead` | A catalog group. Groups can also be combined. |
| `selection` | `all` | All algorithms in the catalog's `all` selector. |
| `selection` | `bench=sha2,auth` | Entire benchmark targets, including cases beyond an individual algorithm. |
| `filter` | `^sha256/rscrypto/64$` | Narrow the selected scope to matching Criterion cases. |

The remaining platform names are `aarch64-linux` and `x86_64-win`.
Algorithm/group selectors and explicit `bench=` targets are alternative forms
of `selection`; the workflow rejects invalid architecture and catalog selections
before starting measurement runners. A case filter that matches nothing fails during
discovery. The catalog remains the authority for available selectors and targets.

Optional sampling fields override the shared Criterion settings; blank fields
preserve the repository defaults. The diagnostic checkbox enables diagnostic
features for the selected targets; it does not select separate targets.

A small planning job validates the request and creates the exact runner matrix.
AWS provides fixed on-demand instance types for Linux x86-64/ARM64 and Windows
x86-64. IBM and RISE provide their existing native runners. The selected
architectures run concurrently; benchmark configurations run sequentially on
each machine. Installers use `--ci-bench` on Linux and `-CiBench` on Windows.
No caches or speed-regression gates are enabled. Donated hosts may be shared,
and fixed AWS instance types do not eliminate host noise. Inspect uncertainty
and repeat matched measurements before making performance claims.

Each job retains `target/bench/` as a GitHub artifact, including failed-run
evidence, source and machine identity, the resolved case plan, logs, and raw
Criterion results. The benchmark runner keeps its one-hour pipeline budget;
`all` is a selection, not a guarantee that every case will fit that budget.
Narrow large runs by algorithm, group, target, or case filter. The workflow
allows additional provisioning time, especially on RISC-V.
Manual dispatch becomes available after the workflow reaches the default branch.

The same granular selections work locally:

```sh
just bench sha256 blake3
just bench hashes
just bench all
just bench bench=sha2 'filter=^sha256/rscrypto/64$'
```

## Timed workload boundaries

Choose the timed boundary from the question the workload answers. State it next
to the benchmark group in source, including input restoration, allocation, key
or state construction, output handling, and destruction. Put material included
or excluded work in the case identity; implementation names must identify the
library or backend actually called. A renamed boundary starts a new baseline.

- **Reusable-buffer operation:** allocate storage and prepare reusable state
  outside timing. Time the operation on that state. If fresh state is required,
  describe any batched setup explicitly; do not call its allocation part of the
  measured operation.
- **Copy plus operation:** restore the input into preallocated storage inside
  timing, then operate on it. Use `copy-and-…` in the operation name. Retain this
  boundary when measuring the cost of preserving an immutable source message.
- **Complete application operation:** include the actual lifecycle being
  studied, and name its stages, such as `copy-and-construct-and-seal`. State which
  application costs remain excluded; constructing a cipher does not imply that
  packet allocation, entropy, or transport is included.

Do not move setup out of timing just to obtain a smaller number. `iter` includes
work and destruction inside its closure and destruction of its return value.
`iter_batched` excludes the setup closure and defers returned-output destruction;
consumed inputs can still be destroyed inside the timed closure.
`iter_batched_ref` also defers destruction of the setup object. The existing
P-256 ECDH batches prepare fresh consumed keys outside timing; RapidHash map
insertion batches allocate empty maps outside timing. These are different
boundaries from AEAD's timed buffer restoration.

The AEAD `copy-and-encrypt`, `copy-and-decrypt`, `copy-and-seal`, and
`copy-and-open` groups reuse preallocated buffers and cipher contexts. They time
input restoration, cryptography, and per-call output handling and cleanup.
Fixture generation, initial buffer allocation, and reusable-context construction
and destruction are excluded. Rows labeled `appended-tag` copy or produce the
combined ciphertext/tag representation; other rows use detached tags. Throughput
counts message bytes, not restoration traffic or tag bytes. These are not
cryptography-only measurements or complete packet-processing measurements.
AES-SIV `copy-and-construct-and-seal` additionally constructs and destroys a
cipher inside each iteration. Construction-only and header-mask groups state
their own boundaries next to their registrations.

The ChaCha diagnostic `chacha20-copy-and-xor` group restores the message and
applies the keystream in the timed closure, reusing an allocated buffer. Poly1305
instead reads immutable fixture bytes and returns a tag without restoring a
message buffer. Neither is a complete AEAD operation.

BLAKE2 `short-oneshot` and `short-keyed` retain only the 16- and 128-byte inputs
absent from the main size matrix. `single-update` measures construction, one
update, and finalization at the small sizes; it is distinct from the multi-chunk
streaming workload. Plain parameter-group duplicates are removed; the main
one-shot rows remain the baselines for salt/personalization hashing. All are
complete hash operations, not isolated host overhead.
Ascon's `rscrypto/scalar-loop` rows compare repeated rscrypto scalar API calls
with its batch API, not with an external library.

## ML-KEM and Argon2 comparison contracts

Compare only rows in the same operation group, with matching build and host
identities. These contracts supersede the old ML-KEM IDs and the Argon2 IDs
without `salt16-raw32`. Do not reuse their measurements as baselines. The effect
of the old mismatches on reported ratios has not been measured.

ML-KEM uses fixed 64-byte key-generation seeds and fixed 32-byte encapsulation
randomness in `derand` groups. Fixture construction is untimed. Decapsulation
uses the same expanded secret-key bytes and ciphertext in every implementation;
it consumes no entropy. The three parameter sets have separate groups.

| Operation suffix | Timed input and preparation | Timed output |
| --- | --- | --- |
| `keygen/derand-encoded` | Caller-supplied seed; generation and export | Encoded public key and expanded secret key as fixed byte arrays |
| `keygen/internal-entropy-encoded` | AWS-LC generation, internal entropy, and export | The same key encodings as fixed byte arrays |
| `encapsulate/derand-reuse-matrix-prepared` | Reused rscrypto key with cached public matrix; caller-supplied randomness | Ciphertext array and 32-byte shared-secret array |
| `encapsulate/derand-reuse-decoded` | Reused RustCrypto decoded key and cached key hash; matrix sampling and caller-supplied randomness remain timed | The same ciphertext and secret arrays |
| `encapsulate/derand-reuse-encoded` | Reused rscrypto, libcrux, or fips203 encoded-key wrapper; caller-supplied randomness; per-call decoding remains timed | The same ciphertext and secret arrays |
| `encapsulate/derand-import-encoded` | Identical public-key bytes; each API's import, validation, and encapsulation with caller-supplied randomness | The same ciphertext and secret arrays |
| `encapsulate/internal-entropy-reuse-native` | Reused AWS-LC key object; internal entropy | Ciphertext and secret converted to fixed arrays |
| `encapsulate/internal-entropy-import-encoded` | The same public-key bytes; AWS-LC import and internal entropy | Ciphertext and secret converted to fixed arrays |
| `decapsulate/reuse-matrix-prepared` | Reused rscrypto key with cached public matrix and typed ciphertext | 32-byte shared-secret array |
| `decapsulate/reuse-decoded` | Reused RustCrypto decoded key and typed ciphertext; re-encryption samples the public matrix inside timing | 32-byte shared-secret array |
| `decapsulate/reuse-encoded` | Reused rscrypto, libcrux, or fips203 encoded-key wrapper and typed ciphertext; per-call decoding remains timed | 32-byte shared-secret array |
| `decapsulate/reuse-native` | Reused AWS-LC key object and borrowed ciphertext bytes | Shared secret converted to a 32-byte array |
| `decapsulate/import-encoded` | Identical expanded secret-key and ciphertext bytes; each API's import, validation, and decapsulation | 32-byte shared-secret array |

All ML-KEM rows include output conversion and destruction of per-call objects.
`Criterion::iter` includes destruction of returned arrays. Reused keys are
constructed and destroyed outside timing; imported keys are constructed and
destroyed inside timing. Internal allocations and their cleanup remain timed,
including AWS-LC's allocated ciphertext/shared-secret buffers and key objects.
The harness does not supply reusable scratch storage or equalize library-specific
cleanup policies. These measure the selected APIs on valid inputs, not identical
validation or zeroization guarantees. RustCrypto expanded-key import/export uses
its deprecated compatibility API intentionally to keep key encodings identical.

Each deterministic ML-KEM row runs its actual timed closure once outside timing
and checks the complete output against the shared fixture. Key-generation checks
compare both encoded keys; encapsulation checks compare ciphertext and secret;
decapsulation checks compare the secret. AWS-LC's randomized generation and
encapsulation closures are checked through cross-implementation decapsulation.
A mismatch aborts execution before that row is timed. Discovery lists identities;
it does not substitute for executing the selected rows' correctness checks.

Argon2 competitor groups include `salt16-raw32` in their IDs. They use the same
password, full 16-byte salt, Argon2 version 0x13, memory/time/lane parameters,
and 32-byte raw output. They consume no entropy and do no PHC encoding. Parameter
objects and caller-owned output buffers are prepared outside timing; each call
includes the selected API's scratch allocation, computation, and scratch
cleanup. The output buffer is reused and destroyed outside timing. Cleanup
policies remain those of each library. Untimed checks compare all 32 output
bytes with RustCrypto and, where its parameter limits allow a row, dryoc.
Argon2 parallel-scaling rows use the same salt and output size while varying
lane count. Scrypt and PHC fixtures are separate workloads.

## Measure locally

[`.config/criterion.json`](../.config/criterion.json) supplies one configuration
for every Criterion harness, including direct Cargo invocations: 20 samples,
100 ms warmup, 400 ms requested measurement time, 10,000 bootstrap resamples,
95% confidence, 5% significance, and a 1% noise threshold. Benchmark groups may
not override these settings. `warmup_ms=`, `measure_ms=`, and `sample_size=`
override the shared defaults for every selected case in that invocation; their
`BENCH_` environment equivalents have lower precedence than explicit arguments.
Boolean controls reject unrecognized values and empty strings.

These are bounded development defaults, not a promise of statistical precision.
Inspect confidence intervals and repeat a focused selection when the uncertainty
cannot support the intended claim. Criterion can extend the requested measurement
window to collect the requested samples for slow operations. `argon2id` includes
small, OWASP, and parallel workloads; no expensive-workload opt-in is required.

`just bench` bounds the whole pipeline—build, discovery, measurement, analysis,
and result verification—to at most one hour. `just profile` uses the same limit for build,
discovery, and capture. The configured limit may be lowered but cannot exceed
3,600 seconds. Shutdown starts before the deadline, reserving up to five seconds
to retain failed-run evidence before stopping surviving child processes. A timed
out run exits with status 124; partial results do not constitute a complete run.
Plans whose requested sampling windows alone exhaust the budget are rejected
before measurement. Build costs, analysis, and slow operations can still make a
smaller plan hit the deadline. Direct Cargo invocation bounds each harness;
use `just bench` to bound a selection spanning multiple harnesses and its builds.

Use an algorithm or family selector, or choose explicit benchmark targets with
`bench=<target>` (`bench=<csv>` for several). `filter=<pattern>` narrows the
selected algorithms or targets. For example, `sha256 filter=rscrypto` stays
within SHA-256; `bench=sha2 filter=rscrypto` searches the entire SHA-2 target.
Repeat `filter=<pattern>` for multiple patterns. Each build configuration is
listed once. A lightweight invocation of the same executable matches all
patterns using Criterion's regex engine, without constructing benchmark fixtures.
The matched cases run as one measurement process per build configuration. The harness reads the
resolved case set from a file and applies an anchored, escaped union filter. A pattern matching no
cases fails before measurement. Patterns are passed verbatim: commas are regex characters, not
separators. Quote each argument for your shell. `BENCH_FILTER` supplies one
literal pattern in addition to any `filter=` arguments. Empty `filter=` values
are rejected; omit the argument for an unfiltered run. Positional selectors accept catalog names; use `filter=` for raw regexes.
Exact algorithm names select only that algorithm; use family names such as
`crc64` to select several. `blake2` includes all implementations and operations,
including dryoc one-shot and keyed cases.

Discover the actual cases before choosing a measurement scope:

```sh
just bench crc64-nvme --list
just bench bench=sha2 --list
just bench blake3 --diag --list
just bench bench=aead_kernels --list
```

`--list` builds the selected configuration and lists its cases without measuring,
creating a run, or copying baseline data. It uses the same filters and case
deduplication as measurement. Each row shows its benchmark binary, exact case
name and work class:

- `ordinary`: public operation and comparison workloads.
- `expensive`: catalog-declared high-cost workloads, including password hashing,
  PBKDF2, and RSA private signing.
- `diagnostic`: internal components, backend experiments, and overhead probes.

Classes come from `.config/benchmark-matrix.json`; they describe workload intent,
not measured duration or a timing guarantee. A diagnostic case can also be costly.
Classes do not block execution. `--diag` (or `diag=true`) enables the `diag`
feature for the selected benchmark builds; some target configurations already
require it. Cases depend on the compiled features and host capabilities.
Dedicated diagnostic targets such as `aead_kernels` require explicit selection;
generic runs include the catalog's required Criterion targets.

Run the narrowest useful case:

```sh
just bench bench=sha2
just bench bench=auth filter='^ecdsa-p256/'
just bench sha256 'filter=^sha256/rscrypto/\d+$' 'filter=^sha256/rscrypto/[0-9]{1,3}$'
just bench p256-ecdh
just bench mlkem
```

Explicit targets, including unfiltered `bench=sha2`, run without a scope override.

`requests.json` preserves the resolved target/filter requests and run budget. `plan.json` records
each configuration's selected cases, Cargo command/artifact, executable hash,
compatibility evidence, effective settings, baseline cases, execution command,
and output location. Raw results live under
`criterion/<binary>-<configuration-id>/` using Criterion's directory layout.
`output.txt` is the build, discovery, and measurement log; `source.json` and
`source-state.json` identify the source files and worktree. The runner verifies
all planned measurements once before marking the run complete.

The shared environment collector separates build inputs from runtime controls,
including Rayon thread controls, CRC backend overrides, and
`RSCRYPTO_FORCE_AVX512`. Unset known runtime controls are explicit JSON nulls.
Benchmark plans and profile metadata carry the same compatibility evidence.

Criterion measures elapsed time. `just bench-structural` uses Gungraun and
Valgrind to count instructions and cache events on supported Linux hosts; those
counts do not prove wall-clock speed.

After a benchmark exposes a concrete cost, inspect it with:

```sh
just profile sha2 --list
just profile sha2 'sha256/rscrypto/64' 10
just profile blake3 --diag --list
just perf-codegen sha2 -- --asm <function>
just perf-llvm-lines sha2 -- --filter <pattern>
```

Profiling requires one exact case name. `--list` builds the selected target and
lists its cases without recording. Capture checks that the name occurs exactly
once, then runs that executable with only the resolved case selected. Unrelated
workload groups skip fixture construction. The
requested duration applies to that case; process startup and profiler overhead
add to the total elapsed time.

Benchmarking and profiling share the Cargo command and CPU-flag policy. Both use
`bench`, which inherits release optimization settings and retains debug symbols
without stripping. Explicit `RUSTFLAGS` or `CARGO_ENCODED_RUSTFLAGS` take
precedence; local macOS runs otherwise use `-C target-cpu=native`. Match those
flags, target, and Cargo features when comparing with a deployment build.

Each target uses its explicit catalog features with Cargo defaults disabled,
independent of algorithm selectors, raw filters, or multi-target selection.
Each distinct target configuration is built once before its selected cases run.
Profiling and code inspection use the same catalog target configuration;
`--diag` enables the same additional feature in each command. Only BLAKE3 and password-hashing
targets enable `parallel`, where their workloads exercise it. Cargo ignores the
panic setting for benchmarks,
so release's `panic = "abort"` remains a difference
([Cargo profiles](https://doc.rust-lang.org/cargo/reference/profiles.html)).

Each capture gets a unique directory under `target/profiles/` containing
`profile.json.gz`, `cases.json`, `metadata.json`, a log, and source evidence. Metadata records the exact
case, executable path and SHA-256, Cargo artifact description, build and capture
commands, compiler and tool versions, build/runtime environment, and capture outcome. Source evidence records input
hashes, revision, and worktree status. Keep the matching executable and
its symbols available when investigating a saved profile.

Keep raw results and run metadata for any published claim. Local measurements
without that evidence are useful only for the machine that produced them.
P-256 ECDH uses the `p256-ecdh` benchmark alias. Its operation rows compare
caller-filled generation, public derivation, canonical SEC1 parsing, agreement,
and a TLS-shaped two-party roundtrip; raw target results and the overview remain
the only performance record.
