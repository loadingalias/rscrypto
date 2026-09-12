# Contributing

Develop every change on a short-lived branch and merge it through a pull
request. The protected `main` branch is releasable history, not a working
branch.

## Start a change

Start from a clean, current `main`:

```bash
git status --short
git switch main
git pull --ff-only
git switch -c <short-feature-name>
```

Do not discard unrelated work to make the worktree clean. Preserve it or move
it to its own branch first.

## Record release intent

Add a `.changes/*.md` file when crate users will observe an API, behavior,
security, performance, compatibility, or release-artifact change:

```bash
cargo rail change add rscrypto --bump patch --message "Describe the user-visible result."
```

Use `minor` or `major` when compatibility requires it. Internal tooling and
maintainer-only documentation normally need no change file. Review release
intent manually before committing.

## Configure compiler reuse

Cargo Rail can reuse compiler results across Cargo, Nextest, Just, and IDE
invocations. For the first-party development fleet, choose a configured
rscrypto target from `~/dev-machines/dev-machine list rscrypto`, acquire a
short-lived credential, and install the canonical remapped policy into the
active Cargo home:

```bash
eval "$("$HOME/dev-machines/dev-machine" cache-env rscrypto <target>)"
just rail-cache-setup --max-size 10GiB
just cache-status
```

Run `cache-env` again when its short-lived R2 lease expires. `dev-machine ssh`
and `dev-machine just` refresh the corresponding remote-machine lease before
execution. Keep cache credentials outside repository configuration. Use
`CARGO_RAIL_CACHE=off` only when a check requires a cold compiler process,
including Miri and machine-code zeroization evidence.

## Validate

Run `just --list` to discover the current recipes. Start with:

```bash
just check
just test
```

`just check` repairs sources before validation. It covers the host and every
entry in `.config/target-matrix.json`; missing target libraries or Clippy
components fail before repairs start. The repair pass applies rustfmt and Clippy
suggestions, including in a dirty or staged worktree. Review the resulting diff.
`just ci-check` validates only the native host without source fixes.
`just ci-policy` checks dependencies across the full supported target graph.
Neither command uses affected-work selection.

Every target receives release/native and debug/portable Clippy passes. The host
checks all Cargo targets; cross checks compile the library without foreign test
or benchmark C dependencies. Bare-metal and browser WASM use `full` plus
applicable serialization features without std, threads, or OS
entropy. WASI adds std and entropy, without threads. POWER, IBM Z, and RISC-V
use the repository-pinned nightly; other targets use the development toolchain.
Validation also checks independent workspaces, dependencies, and docs.

`just test` enables every crate feature except `portable-only`, so runtime
capability detection selects native backends where supported. Use
`just test --portable` to test forced portable dispatch. Both modes print their
dispatch profile; `--all` widens test scope independently of that choice.
ChaCha20 differential tests report accelerated backend and kernel execution
counts, including an explicit message when no accelerated backend ran.

Use the same command for a focused loop:

```bash
just test --test aead_kernel_equivalence
just test --test aead_kernel_equivalence chacha20
just test -- --lib -- --exact checksum::crc16::tests::test_vectors_crc16_ccitt_x25 --nocapture
```

`just test` uses the pinned Nextest runner; it requires `cargo-nextest` and has
no Cargo-test fallback. Put repository options (`--all`, `--release`, `--native`,
`--portable`) first. `--release` selects optimized builds for both Nextest and doctests.
The first runner argument, or an explicit `--`, starts verbatim forwarding
to `cargo nextest run`. For example:

```bash
just test --portable -- --release --lib
just test -- --no-run
just test -- --lib -- --skip slow_test
```

The wrapper consumes the first `--`; a second one reaches Nextest for its
libtest-compatible arguments such as `--skip` and `--exact`. `--test` selects an
integration binary, `--lib` selects library tests, and a name filters tests.
Runner arguments select explicit work regardless of affected scope and skip
doctests. Runs without runner arguments retain the separate Cargo doctest step.
`RSCRYPTO_TEST_THREADS` sets `NEXTEST_TEST_THREADS`; Nextest's explicit
`--test-threads` option takes precedence.

Run `just test-coverage` when you need source coverage. It runs the complete
native and portable test suites plus committed corpus replay in the full and
scoped fuzz workspaces, then writes `coverage/total.lcov`, `coverage/SUMMARY.txt`,
and browsable `coverage/html/index.html`. Use it instead of a separate `just test`
step in a coverage job; reporting does not rerun tests. Ordinary uninstrumented
test results cannot retroactively produce coverage. The merged profile and
executable list remain in `coverage/` for report diagnosis.

Corpus replay defaults to the paths in `fuzz/committed-seeds.txt`, using their
working-tree contents. Unlisted files, including local fuzz discoveries, are
excluded. To include all local corpus files, run
`RSCRYPTO_FUZZ_CORPUS=local just test-coverage` or
`RSCRYPTO_FUZZ_CORPUS=local just test-fuzz-asan --all`. The same variable applies
to direct Cargo replay tests; `committed` explicitly selects the default.
Replay never deletes discoveries. Promote a minimized regression by adding its
seed file and repository-relative path to `fuzz/committed-seeds.txt` (sorted,
one path per line). `just test-scripts` checks that this inventory matches the
tracked corpus files; stage new seed files before running that check.

Coverage uses the development toolchain, cargo-nextest, cargo-llvm-cov, and the
`llvm-tools-preview` rustup component. It measures Rust source under `src/` on
the host, including inline tests, with the existing test profile. Doctest
coverage is deferred until supported without nightly. Live fuzzing, sanitizers,
Miri, timing checks, release-only paths, and other target architectures remain
separate evidence; corpus replay reuses the fuzz implementations without
launching nightly libFuzzer. Reporting validates LLVM function mappings before
publishing; a failed run does not publish a report.

Run `just test-scripts` after changing command selection or script orchestration.
It uses substitute executors without running cryptographic workloads.

Run `just test-harnesses` for DudeCT balancing and raw-exporter self-tests without
timing cases. `just ct-test` includes those tests plus CT tooling regressions.

For broad or compatibility-sensitive changes, run:

```bash
just check
just test --all
just test --all --portable
```

Add the risk-specific evidence reached by the change:

| Change | Required evidence |
| --- | --- |
| Parser, import, DER, PHC, hex, or hostile input | `just test-fuzz <target>` or `just test-fuzz --all` |
| Unsafe Rust, SIMD, assembly, or dispatch | Backend differential tests; `just test-fuzz-asan --all` where native |
| Portable unsafe path | `just test-miri` |
| Constant-time claim boundary | `just ct-full --target <triple>`; update `ct.toml` only with matching evidence |
| Apple Silicon RSA assembly | `just test-rsa-macos-asm` on physical Apple Silicon |
| Public API, examples, or compatibility | Run `just test-examples`; review callers, tests, docs, migration guidance, and release intent |
| Dependency | `just check`; inspect the selected graph |

Cross-compilation proves compilation, not runtime behavior, constant-time
execution, or performance. Record target lanes that cannot run.

RISC-V, POWER, and IBM Z CI separate cross-compilation from native execution to avoid long builds
on the physical runner. Both native-dispatch and portable release suites,
doctests, and the full CT campaign remain required. The transfer commands and
integrity requirements are documented in [scripts/README.md](scripts/README.md).
A successful preparation job does not qualify the target; its execution job
must also pass for the same source and artifacts.

## Review and submit

Inspect and commit only the intended files:

```bash
git status --short
git diff --check
git add <files>
git diff --cached
git commit -m "module: imperative outcome"
```

Push the current branch:

```bash
git push --set-upstream origin HEAD
```

Open a draft pull request:

```bash
gh pr create --base main --fill --draft
```

Before merging, resolve every review thread, inspect the final diff, and confirm
the required local and target-specific evidence.

## Release

Prepare the version and changelog on a clean release branch, using the reviewed
change files:

```bash
cargo rail release run rscrypto --bump auto --skip-tag --allow-non-default-branch
```

Review the generated diff, including manifests and lockfiles in independent
workspaces, validate it, and merge through a PR. Complete physical Apple Silicon
RSA assembly and timing qualification locally before submission; hosted macOS
CI does not replace that evidence.

For the one-time publishing setup, create a GitHub environment named `release`
restricted to `main`. Configure rscrypto's crates.io Trusted Publisher for
`loadingalias/rscrypto`, workflow `release.yml`, and environment `release`.
The workflow obtains a short-lived token; no crates.io secret is required.
See the [crates.io setup instructions](https://crates.io/docs/trusted-publishing).

To deploy, select **Actions → Release → Run workflow → main**. No version input
is needed. The workflow rejects unconsumed change files, a version/changelog
mismatch, or a tag pointing elsewhere. CI (including hosted macOS ARM64), full
CT on all configured CI architectures, and both fuzz architectures plus Miri
run concurrently against the triggering commit. Publication requires all three
workflows to succeed. Benchmarks are separate.

The final job packages the same commit, publishes to crates.io, then creates
`v<version>` and a GitHub Release using the reviewed changelog entry. Only this
job receives registry authentication and repository write permission.

After a transient failure, use **Re-run failed jobs** on the same run. A retry
accepts an existing crates.io version only when its checksum matches the local
package and it is not yanked. It never moves an existing tag or overwrites a
GitHub Release. If qualification artifacts have expired, rerun all jobs. Resolve
checksum, tag, or release-note conflicts before retrying; do not bypass them.

## Security and test evidence

Do not broaden constant-time, audit, FIPS, compliance, secret-lifecycle, or
platform claims without matching evidence. Security boundaries are defined by
[`THREAT_MODEL.md`](THREAT_MODEL.md), [`ct.toml`](ct.toml), and the linked
evidence documents. Report vulnerabilities privately through
[`SECURITY.md`](SECURITY.md).

Use official vectors or an independent implementation as the oracle for
cryptographic correctness. Keep vector provenance, licensing, transforms, and
coverage reviewable. Fuzz targets live in [`fuzz/`](fuzz/) and
[`fuzz-packages/`](fuzz-packages/); commit only small, minimized seeds that
exercise production paths.
