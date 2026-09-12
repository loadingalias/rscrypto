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
ssh-bootstrap target profile="":
    @if [ -n "$2" ]; then "$DEV_MACHINE_EXECUTOR" bootstrap rscrypto "$1" "$2"; else "$DEV_MACHINE_EXECUTOR" bootstrap rscrypto "$1"; fi

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

# Preview, install, and verify the canonical remapped Cargo Rail cache policy.
[group('tooling')]
rail-cache-setup *args:
    @status=0; cargo rail cache setup --check --remote "$CARGO_RAIL_CACHE_REMOTE" --remote-mode "$CARGO_RAIL_CACHE_MODE" --root-portability remap "$@" || status=$?; [ "$status" -le 1 ] || exit "$status"
    @cargo rail cache setup --remote "$CARGO_RAIL_CACHE_REMOTE" --remote-mode "$CARGO_RAIL_CACHE_MODE" --root-portability remap "$@"
    @cargo rail cache probe --json

# Report the effective Cargo Rail cache policy and usage.
[group('tooling')]
cache-status:
    @cargo rail cache status --scope local --format json

# Builds
# Build every workspace target with every feature; accepts Cargo build arguments.
build *args:
    scripts/lib/toolchain.sh --exec cargo build --locked --workspace --all-targets --all-features "$@"

# Checks
# Explain the affected Cargo Rail work; accepts planner arguments.
plan *args:
    @cargo rail plan --explain "$@"

# Repair, then validate the host and the explicit supported target catalog.
check:
    @scripts/check/check.sh check

# Validate the native host without source fixes or cross-target prerequisites.
ci-check:
    @scripts/check/check.sh native

# Validate Apple Silicon checks, native/portable release tests, and RSA assembly locally.
check-macos:
    @scripts/check/macos.sh

# Enable mandatory local macOS validation for commits and merge commits in this checkout.
[group('tooling')]
install-hooks:
    @git config --local core.hooksPath .githooks

# Cross-check the complete target native CI compilation surface.
ci-check-target target:
    @scripts/check/check.sh target {{quote(target)}}

# Prepare or execute complete, source-bound cross-compiled test artifacts.
test-cross operation target archive:
    @scripts/lib/python.sh scripts/test/cross.py {{quote(operation)}} {{quote(target)}} {{quote(archive)}}

# Check dependency policy for every supported target, once per CI workflow.
ci-policy:
    @scripts/check/dependencies.sh

# Compile feature/MSRV/bare-metal compatibility and execute WASM/WASI vectors.
ci-compat:
    @scripts/lib/python.sh scripts/check/compat.py

# Verify the publishable archive and external std/core/alloc consumers.
ci-package:
    @scripts/lib/python.sh scripts/check/package.py

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
    @scripts/lib/python.sh scripts/release/release_test.py
    @scripts/lib/python.sh scripts/test/test_runner_test.py
    @scripts/lib/python.sh scripts/test/just_arguments_test.py
    @scripts/lib/python.sh scripts/tooling/toolchain_test.py
    @scripts/lib/python.sh scripts/tooling/install_test.py
    @scripts/lib/python.sh scripts/bench/ci_test.py
    @scripts/lib/python.sh scripts/test/fuzz_features_test.py
    @scripts/lib/python.sh scripts/check/macos_test.py
    @scripts/lib/python.sh scripts/check/check_runner_test.py
    @scripts/lib/python.sh scripts/check/compat_test.py
    @scripts/lib/python.sh scripts/test/fuzz_runner_test.py

# Exercise artifact transfer and the pinned rustdoc build/run contract.
[group('tests')]
test-transfer:
    @scripts/lib/python.sh scripts/test/transfer_test.py

# Run CT harness and exporter self-tests without timing cases.
[group('constant-time')]
test-harnesses:
    scripts/lib/toolchain.sh --exec cargo test --locked --manifest-path tools/ct-dudect/Cargo.toml -p rscrypto-ct-dudect -p dudect-bencher --lib --bins

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

# Test x86-64 RSA assembly on a physical Linux host.
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

# Coverage

# Run native/portable tests and corpus replay, then report combined source coverage.
[group('tests')]
test-coverage:
    @scripts/lib/python.sh scripts/test/test-coverage.py

# Benches

# Measure Criterion cases, or discover them with --list; --diag enables diagnostics.
[group('benchmarks')]
bench *args:
    @python="$(scripts/lib/python.sh --print)"; "$python" scripts/bench/bounded.py "$python" scripts/bench/runner.py bench "$@"

# Stable instruction/cache-cost benchmarks. Requires gungraun-runner and Valgrind.
[group('benchmarks')]
bench-structural:
    @command -v gungraun-runner >/dev/null || { echo "error: gungraun-runner is required" >&2; exit 1; }
    @command -v valgrind >/dev/null || { echo "error: Valgrind is required" >&2; exit 1; }
    scripts/lib/toolchain.sh --exec cargo bench --locked --profile bench --features 'checksums,sha2,blake3' --bench structural

# Record one exact case, or discover cases with --list; --diag enables diagnostics.
[group('benchmarks')]
profile *args:
    @python="$(scripts/lib/python.sh --print)"; "$python" scripts/bench/bounded.py "$python" scripts/bench/runner.py profile "$@"

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

# Update tool pins, stable Rust, and every Cargo manifest.
[group('tooling')]
update *args:
    @scripts/update-all.sh "$@"
