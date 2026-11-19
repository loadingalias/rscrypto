set positional-arguments

[default]
[private]
_default:
    @just --list

# Remote dev. Provider mechanics live in ~/dev-machines.

export DEV_MACHINE_EXECUTOR := env_var_or_default("DEV_MACHINE_BIN", home_directory() / "dev-machines/dev-machine")

# Run a command on a repository development machine.
[group('remote')]
ssh target *args:
    @"$DEV_MACHINE_EXECUTOR" ssh rscrypto "$@"

# Verify the synchronized remote repository state without opening a shell.
[group('remote')]
ssh-check target:
    @"$DEV_MACHINE_EXECUTOR" ssh rscrypto "$1" --check

# Run a targeted Cargo command on a repository development machine.
[group('remote')]
ssh-cargo target *args:
    @target="$1"; shift; "$DEV_MACHINE_EXECUTOR" just rscrypto "$target" _remote-cargo "$@"

[private]
_remote-cargo *args:
    scripts/lib/toolchain.sh --exec cargo "$@"

[private]
_remote-install-llvm-tools:
    rustup component add llvm-tools-preview

# Verify that a development machine is ready for repository work.
[group('remote')]
ssh-preflight target:
    @"$DEV_MACHINE_EXECUTOR" preflight rscrypto "$1"

# Create a repository development machine.
[group('remote')]
ssh-create target *args:
    @"$DEV_MACHINE_EXECUTOR" create rscrypto "$@"

# Start a repository development machine.
[group('remote')]
ssh-start target:
    @"$DEV_MACHINE_EXECUTOR" start rscrypto "$1"

# Deallocate a repository development machine while preserving it.
[group('remote')]
ssh-deallocate target:
    @"$DEV_MACHINE_EXECUTOR" deallocate rscrypto "$1"

# Permanently destroy a repository development machine.
[group('remote')]
ssh-kill target:
    @"$DEV_MACHINE_EXECUTOR" kill rscrypto "$1"

# Show one or all repository development machines.
[group('remote')]
ssh-status target="":
    @if [ -n "$1" ]; then "$DEV_MACHINE_EXECUTOR" status rscrypto "$1"; else "$DEV_MACHINE_EXECUTOR" status rscrypto; fi

# Bootstrap a repository development machine.
[group('remote')]
ssh-bootstrap target:
    @"$DEV_MACHINE_EXECUTOR" bootstrap rscrypto "$1"

# Run a Just recipe on a repository development machine.
[group('remote')]
ssh-just target *args:
    @"$DEV_MACHINE_EXECUTOR" just rscrypto "$@"

# Collect a benchmark run from a repository development machine.
[group('remote')]
ssh-collect-bench target run_id destination:
    @"$DEV_MACHINE_EXECUTOR" just rscrypto "$1" bench-export "benchmark_results/criterion/$2"
    @"$DEV_MACHINE_EXECUTOR" collect-results rscrypto "$1" criterion "$2" "$3"

# List repository development machines.
[group('remote')]
ssh-list:
    @"$DEV_MACHINE_EXECUTOR" list rscrypto

# Builds
# Build every workspace target with every feature; accepts Cargo build arguments.
build *args:
    scripts/lib/toolchain.sh --exec cargo build --locked --workspace --all-targets --all-features "$@"

# Checks
# Explain the affected Cargo Rail work; accepts planner arguments.
plan *args:
    @cargo rail plan --explain "$@"

# Check the complete supported declaration surface before preparing a release.
release-surface:
    @cargo rail surface --check --explain

# Preview the next release and run every release check without any effect; exit 1 means checks are pending.
release-check bump="auto":
    @status=0; cargo rail release rscrypto rscrypto-rustls --bump "$1" --dry-run || status=$?; [ "$status" -le 1 ] || exit "$status"

# Surface is paused: restore `release-surface` as the first dependency when its target preflight passes.
# release bump="auto": release-surface (release-check bump)

# Run the Release workflow on main; cargo-rail commits, tags, publishes, and creates the GitHub Release.
release bump="auto": (release-check bump)
    @gh workflow run release.yml --ref main -f bump="$1"

# Repair (rustfmt and Clippy fixes), validate this Mac, and run the CI tooling checks. CI validates the other targets.
check: && test-scripts ct-test
    @scripts/check/check.sh check
    @scripts/check/dependencies.sh

# Validate Apple Silicon checks, native/portable release tests, and RSA assembly locally.
check-macos:
    @scripts/check/macos.sh

# Queue isolated Mac qualification and publish its result for the current commit.
qualify-macos:
    @scripts/lib/python.sh scripts/check/macos_async.py queue

# Inspect queued Mac qualification and find its retained logs.
macos-status:
    @scripts/lib/python.sh scripts/check/macos_async.py status

# Prepare or execute complete, source-bound cross-compiled test artifacts; `run ... --emulated` under QEMU.
test-cross operation target archive *flags:
    @scripts/lib/python.sh scripts/test/cross.py "$@"

# Exercise TLS with the pinned OpenSSL and Go peers; retain process evidence locally.
[group('tests')]
test-provider-interop *args:
    @scripts/lib/python.sh scripts/test/provider_interop.py "$@"

# Record individual TLS latency, CPU, bytes and process RSS for one catalog profile.
[group('benchmarks')]
bench-provider-scorecard *args:
    @scripts/lib/python.sh scripts/bench/provider_scorecard.py "$@"

# Observe live TLS allocations with shared keys and up to 64 admitted handshakes.
[group('benchmarks')]
bench-provider-resources *args:
    @scripts/lib/python.sh scripts/bench/provider_resources.py "$@"

# Record ML-DSA per-signature latency tails and rejection-sampling attempt counts.
[group('benchmarks')]
bench-mldsa-tail *args:
    @scripts/lib/python.sh scripts/bench/mldsa_tail.py "$@"

# Run pinned BoGo TLS 1.2/1.3 cases and retain failures as well as passes.
[group('tests')]
test-provider-bogo *args:
    @scripts/lib/python.sh scripts/test/provider_bogo.py "$@"

# Execute the full native and portable suites against the host's musl target.
test-musl:
    @scripts/test/test-musl.sh

# Tests
# Run Nextest with repository scope/dispatch options, then -- NEXTEST_ARGS.
test *args:
    @scripts/test/test.sh "$@"

# Test script selection and failure handling without running cryptographic workloads.
[group('tooling')]
test-scripts:
    #!/usr/bin/env bash
    set -euo pipefail
    # `just ct-test` runs scripts/ct.
    for test in $(find scripts -name '*_test.py' -not -path 'scripts/ct/*' | sort); do
      scripts/lib/python.sh --group tooling "$test"
    done

# Exercise artifact transfer and the pinned rustdoc build/run contract.
[group('tests')]
test-transfer:
    @scripts/lib/python.sh scripts/test/transfer_test.py

# Execute every runnable example with its minimum feature set.
[group('tests')]
test-examples:
    @scripts/test/test-examples.sh

# Test portable unsafe paths under Miri.
[group('tests')]
test-miri *args:
    @scripts/test/test-miri.sh "$@"

# Test Apple Silicon RSA assembly on a physical supported host.
[group('tests')]
test-rsa-macos-asm:
    @scripts/test/test-rsa-asm.sh macos

# Test the owned x86-64 RSA Montgomery kernel on a physical Linux host.
[group('tests')]
test-rsa-linux-asm:
    @scripts/test/test-rsa-asm.sh linux

# Run live fuzzing; --all includes the full and scoped fuzz packages.
[group('tests')]
test-fuzz *args:
    @scripts/test/test-fuzz.sh "$@"

# Run fuzz targets with AddressSanitizer.
[group('tests')]
test-fuzz-asan *args:
    @scripts/test/test-fuzz-asan.sh "$@"

# Constant-Time (CT) Validation Engine
# Test CT report validation, orchestration, harness, and raw timing exporter.
[group('constant-time')]
ct-test:
    @scripts/ct/test.sh

# Run DudeCT Timing Checks
[group('constant-time')]
ct-dudect *args:
    @scripts/ct/dudect.sh "$@"

# Build CT Artifacts; Run Timing Evidence; Emit CT Reports
[group('constant-time')]
ct-full *args:
    @scripts/lib/python.sh scripts/ct/full.py "$@"

# Measure prepared cross-target CT cases on a pinned CPU; diagnostic evidence only.
[group('constant-time')]
ct-replay *args:
    @scripts/lib/python.sh scripts/ct/replay.py "$@"

# Run BINSEC; Manifest-Declared Binary CT Kernels
[group('constant-time')]
ct-binsec *args:
    @scripts/lib/python.sh scripts/ct/binsec.py "$@"

# Build CT Harness Artifacts
[group('constant-time')]
ct-artifacts *args:
    @scripts/ct/artifacts.sh "$@"

# Validate CT Manifest & Generated Artifacts
[group('constant-time')]
ct-validate *args:
    @scripts/lib/python.sh scripts/ct/validate.py "$@"

# Prepare the review of stale CT asm waivers; write a review record
[group('constant-time')]
ct-renew *args:
    @scripts/lib/python.sh scripts/ct/renew_waivers.py "$@"

# Secret stack evidence
# Review secret-worker stack depth against its scrub in linked release binaries.
[group('secret-lifecycle')]
stack-frames *args:
    @scripts/lib/python.sh scripts/stack/frames.py "$@"

# Regenerate the owned stack budgets from the current bounds after a reviewed stack change.
[group('secret-lifecycle')]
stack-budgets *args:
    @scripts/lib/python.sh scripts/stack/frames.py --whole-calls all --all-configurations --write-budgets .config/stack-budgets.json "$@"

# Measure the declared libc edges' stack use on this Linux GNU host on a painted stack.
[group('secret-lifecycle')]
stack-allowances *args:
    @scripts/lib/python.sh scripts/stack/painted.py edges "$@"

# Measure the provider TLS worker's painted-stack high-water mark for every scenario.
[group('secret-lifecycle')]
stack-worker *args:
    @scripts/lib/python.sh scripts/stack/painted.py worker "$@"

# Measure secret residue on QEMU boards, or native ML-DSA signing and SLH-DSA-SHA2-128s stacks.
[group('secret-lifecycle')]
stack-residue *args:
    @scripts/lib/python.sh scripts/stack/residue.py "$@"

# Coverage

# Run native/portable tests and corpus replay, then report combined source coverage.
[group('tests')]
test-coverage:
    @scripts/lib/python.sh scripts/test/test-coverage.py

# Benches

# Measure Criterion cases, or discover them with --list; --diag enables diagnostics.
# The bench and profile runners run Python in UTF-8 mode so Windows consoles can print Criterion output.
[group('benchmarks')]
bench *args:
    @export PYTHONUTF8=1; python="$(scripts/lib/python.sh --print)"; "$python" scripts/bench/bounded.py --bench "$@"

# Stable instruction/cache-cost benchmarks. Requires gungraun-runner and Valgrind.
[group('benchmarks')]
bench-structural *args:
    @command -v gungraun-runner >/dev/null || { echo "error: gungraun-runner is required" >&2; exit 1; }
    @command -v valgrind >/dev/null || { echo "error: Valgrind is required" >&2; exit 1; }
    @export PYTHONUTF8=1; python="$(scripts/lib/python.sh --print)"; "$python" scripts/bench/bounded.py scripts/lib/toolchain.sh --exec cargo bench --locked --profile bench --features 'checksums,sha2,blake3' --bench structural "$@"

# Record one exact case, or discover cases with --list; --diag enables diagnostics.
[group('benchmarks')]
profile *args:
    @export PYTHONUTF8=1; python="$(scripts/lib/python.sh --print)"; "$python" scripts/bench/bounded.py "$python" scripts/bench/runner.py profile "$@"

# Inspect optimized code for an explicit benchmark target configuration.
[group('benchmarks')]
perf-codegen target *args:
    @scripts/lib/python.sh scripts/bench/runner.py codegen "$@"

# Attribute LLVM IR for an explicit benchmark target configuration.
[group('benchmarks')]
perf-llvm-lines target *args:
    @scripts/lib/python.sh scripts/bench/runner.py llvm-lines "$@"

# Export a completed or failed benchmark run for collection.
[group('benchmarks')]
bench-export run:
    @scripts/lib/python.sh scripts/bench/runner.py export "$@"

# Score retained runs as PLATFORM=RUN_DIRECTORY against catalog comparisons; --cases checks a --list output.
[group('benchmarks')]
bench-compare *args:
    @scripts/lib/python.sh scripts/bench/compare.py "$@"

# Run internal-hook regressions with native and portable dispatch.
[group('tests')]
test-evidence:
    @scripts/lib/python.sh scripts/ct/internal.py --target "${CARGO_BUILD_TARGET:-$(scripts/lib/toolchain.sh --print-host)}" -- scripts/lib/python.sh scripts/test/evidence_suite.py
