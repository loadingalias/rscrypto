# Contributing

Develop and commit directly on `main`.
`main` is the only development branch.
Validate every commit locally before you push it.

## Start a change

Start from a clean, current `main`:

```bash
git status --short
git switch main
git pull --ff-only
```

Do not discard unrelated work to make the worktree clean.
Keep it safe before you switch or update the checkout.

## Record release intent

Add a `.changes/*.md` file when crate users will see a change in the API, behavior, security, performance,
compatibility, or release artifacts:

```bash
cargo rail change add rscrypto --bump patch --message "Describe the user-visible result."
```

- Use `minor` or `major` when compatibility needs it.
- Internal tooling and maintainer-only documentation usually need no change file.
- Review the release intent manually before you commit.
- Keep pending notes about user-visible results.
- Merge notes that overlap, but keep distinct changes and their bump levels.
- Maintainer-only runner adjustments belong in the runner catalog and the tooling guide.

`cargo rail change status` validates and lists the pending intent.
The release command moves it into `CHANGELOG.md`.
Do not add unreleased work to the changelog section of a published version.

## Configure compiler reuse

Cargo Rail can reuse compiler results across Cargo, Nextest, Just, and IDE invocations.

1. Choose a configured `rscrypto` target from `~/dev-machines/dev-machine list rscrypto`.
1. Get a short-lived credential, and enroll this checkout in the remote cache
   that trusted development machines and CI use:

   ```bash
   eval "$("$HOME/dev-machines/dev-machine" cache-env rscrypto <target>)"
   just rail-cache-setup --max-size 10GiB
   just cache-status
   ```

1. Run `cache-env` again when the lease expires.

`cache-env` owns the remote URL, the credentials, and the read/write authority.
None of them belong in repository configuration.
Without that environment, `rail-cache-setup` installs and checks private local reuse only.
`dev-machine ssh` and `dev-machine just` refresh the lease of the remote machine before they run.

CI uses the same remote cache.
It has read-write authority only on trusted `main` pushes.
Other jobs with credentials are read-only, and pull requests from forks stay cold.
Use `CARGO_RAIL_CACHE=off` only when a check needs a cold compiler process,
for example Miri and machine-code zeroization evidence.

## Validate

### Local hooks

Every pushed tip queues macOS ARM64 qualification on the maintainer's physical Apple Silicon Mac.
The push returns while qualification runs; a release waits for its passing GitHub status.

Run `just install-hooks` once for each checkout.

- The pre-commit and pre-merge-commit hooks run `just ci-check` on the staged tree:
  formatting, native and portable host lints, and documentation.
  The check runs in one persistent worktree, `rscrypto-ci-check/checkout` in the common Git directory,
  with its own Cargo target directory.
  One check runs at a time.
  Unstaged edits and untracked files stay out of the check and do not block the commit,
  so sessions can commit disjoint paths from one checkout.
- The pre-push hook queues `just check-macos` in a detached worktree of the pushed commit:
  native checks, complete release tests with native and portable dispatch (including doctests),
  internal evidence regressions, and the Apple Silicon RSA assembly gate.
- The pushed tip must be the clean checkout. Later work in that checkout cannot change the background job.
  Jobs run one at a time and keep their logs and passing tree/compiler records in the common Git directory.
  A tree that already passed skips testing, from any worktree, and publishes the result for the pushed commit.
- Both hooks reuse a pass when the new tree differs from the nearest passing first-parent ancestor
  only in paths that no check reads: `docs/`, `.changes/`, `.github/`, `benchmark_results/`,
  `.config/tooling.toml`, `CHANGELOG.md`, `CONTRIBUTING.md`, `SECURITY.md`, and `THREAT_MODEL.md`.
  `scripts/check/qualified.py` owns this list. A change to any other path, or a different compiler, runs the checks.
  `README.md` is checked because the crate documentation includes it.
- `check-macos` skips `just ci-check` when the clean checkout already passed it in the pre-commit hook.
- In a push of more than one commit, the intermediate commits get only `just ci-check`.

Install the prerequisites with `scripts/tooling/aarch64-macos.sh` when you need them.
Authenticate `gh` as a repository maintainer with permission to write commit statuses
(`repo:status` for an OAuth token, or Commit statuses: write for a fine-grained token).
See GitHub's [commit status API](https://docs.github.com/en/rest/commits/statuses).
Run `just macos-status` to find each job's result and log.
After fixing a failed check, commit and push the fix.
For an interrupted job or failed status publication, run `just qualify-macos` on the clean commit to retry.
The retry reuses a completed local pass when only publication failed.
`just check-macos` remains available as a foreground diagnostic; it does not publish a status.

Do not bypass the hooks.
Release preflight, packaging, and publication require the latest `rscrypto/macos` status to pass
for that commit, its exact Git tree, and the pinned compiler distribution.
A missing, failed, pending, or mismatched status blocks release.
Pushes to destinations other than GitHub keep their results local.
Commits made remotely still need qualification on the physical Mac before release.

### Checks

Run `just --list` to see the current recipes.
Start with:

```bash
just check
just test
```

- `just check` repairs the source, then validates it.
  It covers the host and every entry in `.config/target-matrix.json`.
  Missing target libraries or Clippy components fail before the repairs start.
  The repair pass applies rustfmt and Clippy suggestions, also in a dirty or staged worktree.
  Review the resulting diff.
- `just ci-check` validates only the native host, without source fixes.
- `just ci-policy` checks dependencies across the full supported target graph.
- Neither command selects only the affected work.

What the checks cover:

- Every target gets a release/native Clippy pass and a debug/portable Clippy pass.
- The host checks all Cargo targets.
  Cross checks compile the library without foreign C dependencies of tests or benchmarks.
- Bare-metal and browser WASM use `full`, plus the applicable serialization features, without `std`, threads,
  or operating-system entropy.
  WASI adds `std` and entropy, without threads.
- Every target uses the one repository-pinned nightly from `rust-toolchain.toml`.
- Validation also checks the independent workspaces, dependencies, and documentation.
- Check recipes verify vector inventories and checksums before compilation.
  Add or update each payload's adjacent `SHA256SUMS` with its provenance when changing vectors.
  Missing or unlisted payloads fail validation, including new untracked inputs.

### Tests

`just test` selects the production-auto feature set: every crate feature except `portable-only`,
with production dispatch enabled.
Use `just test --portable` for the portable-only lane.
That lane uses Cargo's all-feature set, which includes `portable-only`.
An all-feature host run is therefore portable-only evidence, never native backend evidence.
Both modes print their dispatch profile.
`--all` widens the test scope, independently of that choice.

Run `just test-evidence` for changes to internal evidence hooks or forced-kernel tests.
It runs their production-auto and portable-only regressions through the internal build boundary.
Ordinary test builds keep that boundary closed.
ChaCha20 differential tests report how often the accelerated backend and kernels ran,
and they report explicitly when no accelerated backend ran.

Use the same command for a focused loop:

```bash
just test --test aead_kernel_equivalence
just test --test aead_kernel_equivalence chacha20
just test -- --lib -- --exact checksum::crc16::tests::test_vectors_crc16_ccitt_x25 --nocapture
```

How `just test` handles arguments:

- It uses the pinned Nextest runner.
  It needs `cargo-nextest`, and it has no `cargo test` fallback.
- Put repository options first: `--all`, `--release`, `--native`, `--portable`.
- `--release` selects optimized builds for both Nextest and doctests.
- The first runner argument, or an explicit `--`, starts verbatim forwarding to `cargo nextest run`.
- The wrapper consumes the first `--`.
  A second `--` reaches Nextest for its libtest-compatible arguments, such as `--skip` and `--exact`.
- `--test` selects an integration binary, `--lib` selects library tests, and a name filters tests.
- Forwarded Cargo feature flags are rejected, because the dispatch profile owns feature selection.
- Runner arguments select explicit work, independent of the affected scope, and they skip doctests.
  Runs without runner arguments keep the separate Cargo doctest step.
- `RSCRYPTO_TEST_THREADS` sets `NEXTEST_TEST_THREADS`.
  Nextest's explicit `--test-threads` option takes precedence.

Examples:

```bash
just test --portable -- --release --lib
just test -- --no-run
just test -- --lib -- --skip slow_test
```

### Coverage

Run `just test-coverage` when you need source coverage.

- It runs the complete native and portable test suites,
  and it replays the committed corpus in the full and scoped fuzz workspaces.
- It writes `coverage/total.lcov`, `coverage/SUMMARY.txt`, browsable `coverage/html/index.html`,
  and `coverage/provenance.json` with source, tool, suite, environment, and artifact evidence.
- In a coverage job, use it instead of a separate `just test` step.
  Reporting does not run the tests again.
- Ordinary uninstrumented test results cannot produce coverage later.
- The merged profile and the executable list stay in `coverage/` for report diagnosis.
- Coverage provenance records discovered tests, ignored/filter status, actual executions,
  executable hashes, and profile counts per test and suite.
  Discovery profiles are excluded.
  A missing execution, missing test profile, or changed executable fails collection.
  `provenance.json` is the completion marker; failed collection or publication removes old reports.
- `coverage/suites/` retains each suite's LCOV contribution, linked from provenance.
  Corpus receipts identify replayed files;
  their SHA-256 hashes must match before and after the suite.
  This includes ignored discoveries when local replay is requested.

Corpus replay uses the paths in `fuzz/committed-seeds.txt` by default, with their working-tree contents.
Unlisted files, including local fuzz discoveries, are excluded.
To include all local corpus files, run `RSCRYPTO_FUZZ_CORPUS=local just test-coverage` or `RSCRYPTO_FUZZ_CORPUS=local just test-fuzz-asan --all`.
The same variable applies to direct Cargo replay tests.
`committed` selects the default explicitly.
Replay never deletes discoveries.

To promote a minimized regression, add its seed file, and add its repository-relative path to `fuzz/committed-seeds.txt`
(sorted, one path per line).
`just test-scripts` checks that this list matches the tracked corpus files.
Stage new seed files before you run that check.

Coverage uses the development toolchain, `cargo-nextest`, `cargo-llvm-cov`, and the `llvm-tools-preview` rustup component.
It measures Rust source under `src/` on the host, including inline tests, with the existing test profile.

- Doctest coverage is deferred.
- Live fuzzing, sanitizers, Miri, timing checks, release-only paths,
  and other target architectures are separate evidence.
- Corpus replay reuses the fuzz implementations without starting nightly libFuzzer.
- Reporting validates the LLVM function mappings before it publishes.
  A failed run publishes no report.

### Tooling tests

- Run `just test-scripts` after you change command selection or script orchestration.
  It uses substitute executors and runs no cryptographic workloads.
- Run `just ct-test` for CT tooling regressions,
  including DudeCT balancing and the raw-exporter self-tests, without timing cases.

### Evidence by risk

For broad or compatibility-sensitive changes, run:

```bash
just check
just test --all
just test --all --portable
```

Then add the evidence that the change reaches:

| Change                                          | Required evidence |
| ----------------------------------------------- | ----------------- |
| Parser, import, DER, PHC, hex, or hostile input | `just test-fuzz <target>` or `just test-fuzz --all` |
| Unsafe Rust, SIMD, assembly, or dispatch        | Backend differential tests, and `just test-fuzz-asan --all` where native |
| Portable unsafe path                            | `just test-miri`  |
| Constant-time claim boundary                    | `just ct-full --target <triple>`; change `ct.toml` only with matching evidence |
| Apple Silicon RSA assembly                      | `just test-rsa-macos-asm` on physical Apple Silicon |
| Public API, examples, or compatibility          | `just test-examples`; review callers, tests, docs, explicit API removals, and release intent |
| Dependency                                      | `just check`; inspect the selected graph |

Cross-compilation proves compilation only.
It does not prove runtime behavior, constant-time execution, or performance.
Record the target lanes that cannot run.

### Cross-built targets

RISC-V, POWER, and IBM Z CI separate cross-compilation from native execution,
to avoid long builds on the physical runner.

- The x86-64 producers use the shared Cargo Rail cache, under the CI read/write policy above.
- The native runners consume archives bound to the source.
  They do not compile the crate.
- The native-dispatch and portable release suites, doctests, and the full CT campaign stay required.
- [scripts/README.md](scripts/README.md) documents the transfer commands and integrity requirements.
- A successful preparation job does not qualify the target.
  Its execution job must also pass for the same source and artifacts.

## Review and submit

Inspect and commit only the files you intend:

```bash
git status --short
git diff --check
git add <files>
git diff --cached
git commit -m "module: imperative outcome"
```

Push the validated commits:

```bash
git push origin main
```

Before you push, resolve review findings, inspect the final diff,
and confirm the required local and target-specific evidence.

## Release

### Prepare

You can preview the exact local release plan at any time.
On a clean `main` checkout, prepare the release from the reviewed change files:

```bash
just release-check
just release-prepare
```

- `release-prepare` repeats the local release check.
  It then creates the local version, the changelog, the auxiliary lockfiles, and the release commit.
- It does not tag, push, publish, or create a forge release.
- Pass an exact bump or version only when the reviewed intent needs it,
  for example `just release-check minor`.
- The Surface gate (`just release-surface`) is paused for releases until its target preflight passes.
  It is not part of routine planning or validation.

If preparation stops before it finishes, inspect and resume the retained transaction:

```bash
cargo rail release status
cargo rail release resume
```

Review the prepared commit and its complete diff,
including the manifests and lockfiles in independent workspaces.
Validate it, then push it to `main`.
The `release.auxiliary_cargo_manifests` list in [`.config/rail.toml`](.config/rail.toml) names the standalone workspaces whose
lockfiles must follow the package version.
The background Mac job includes physical Apple Silicon RSA assembly qualification.
Complete any separately required timing qualification locally before submission;
`check-macos` does not run it. macOS does not run in hosted CI.

### One-time publishing setup

1. Create a GitHub environment named `release`, restricted to `main`.
1. Configure the crates.io Trusted Publisher for `loadingalias/rscrypto`,
   workflow `release.yml`, environment `release`.

The workflow gets a short-lived token, so no crates.io secret is needed.
See the [crates.io setup instructions](https://crates.io/docs/trusted-publishing).

### Deploy

Select **Actions → Release → Run workflow → main**, and enter the version from `Cargo.toml` without the `v` prefix,
or run `gh workflow run release.yml --ref main -f version=<version>`.
The run is named `Release v<version>`.
Wait for the release commit's `rscrypto/macos` status before starting the workflow.

- The workflow rejects a version input that differs from `Cargo.toml`, unconsumed change files,
  a version and changelog mismatch, and a tag that points elsewhere.
- Three workflows run at the same time against the triggering commit: CI
  (with macOS ARM64 qualified locally for the triggering commit),
  full CT on all configured CI architectures, and fuzzing on both architectures plus Miri.
  Publication needs all three to pass.
- Benchmarks are separate.
- The final job packages the same commit and publishes it to crates.io.
  It then creates `v<version>` and a GitHub Release with the reviewed changelog entry.
  Only this job gets registry authentication and repository write permission.

After a transient failure, use **Re-run failed jobs** on the same run.

- A retry accepts an existing crates.io version only if its checksum matches the local package
  and it is not yanked.
- A retry never moves an existing tag, and never overwrites a GitHub Release.
- If the qualification artifacts have expired, rerun all jobs.
- Resolve checksum, tag, and release-note conflicts before you retry.
  Do not bypass them.

## Security and test evidence

Do not widen constant-time, audit, FIPS, compliance, secret-lifecycle,
or platform claims without matching evidence.
[`THREAT_MODEL.md`](THREAT_MODEL.md), [`ct.toml`](ct.toml), and the linked evidence documents define the security boundaries.
Report vulnerabilities privately through [`SECURITY.md`](SECURITY.md).

Use official vectors or an independent implementation as the oracle for cryptographic correctness.
Keep vector provenance, licensing, transforms, and coverage reviewable.
Assembly derived from upstream sources needs an `output` row in a `src/**/*_assembly_provenance.tsv` manifest
that pins the upstream archive, the members, and the output SHA-256, and it must keep the upstream license notice.
`just check` fails on derived assembly without a manifest row and on any output that differs from its pinned hash.
Fuzz targets live in [`fuzz/`](fuzz/) and [`fuzz-packages/`](fuzz-packages/).
Commit only small, minimized seeds that exercise production paths.
