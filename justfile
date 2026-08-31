# Remote dev. Provider mechanics live in ~/dev-machines.

dev_machine := env_var_or_default("DEV_MACHINE_BIN", env_var("HOME") + "/dev-machines/dev-machine")

# Run a command on a repository development machine.
ssh target *args="":
    @"{{ dev_machine }}" ssh rscrypto "{{ target }}" {{ args }}

# Run a command after checking the remote repository state.
ssh-check target *args="":
    @"{{ dev_machine }}" ssh rscrypto "{{ target }}" --check {{ args }}

# Verify that a development machine is ready for repository work.
ssh-preflight target:
    @"{{ dev_machine }}" preflight rscrypto "{{ target }}"

# Create a repository development machine.
ssh-create target *args="":
    @"{{ dev_machine }}" create rscrypto "{{ target }}" {{ args }}

# Start a repository development machine.
ssh-start target:
    @"{{ dev_machine }}" start rscrypto "{{ target }}"

# Deallocate a repository development machine while preserving it.
ssh-deallocate target:
    @"{{ dev_machine }}" deallocate rscrypto "{{ target }}"

# Permanently destroy a repository development machine.
ssh-kill target:
    @"{{ dev_machine }}" kill rscrypto "{{ target }}"

# Show one or all repository development machines.
ssh-status target="":
    @if [ -n "{{ target }}" ]; then "{{ dev_machine }}" status rscrypto "{{ target }}"; else "{{ dev_machine }}" status rscrypto; fi

# Bootstrap a repository development machine.
ssh-bootstrap target profile="":
    @if [ -n "{{ profile }}" ]; then "{{ dev_machine }}" bootstrap rscrypto "{{ target }}" "{{ profile }}"; else "{{ dev_machine }}" bootstrap rscrypto "{{ target }}"; fi

# Run a Just recipe on a repository development machine.
ssh-just target *args="":
    @"{{ dev_machine }}" just rscrypto "{{ target }}" {{ args }}

# Collect a benchmark run from a repository development machine.
ssh-collect-bench target run_id destination:
    @"{{ dev_machine }}" collect-bench rscrypto "{{ target }}" "{{ run_id }}" "{{ destination }}"

# List repository development machines.
ssh-list:
    @"{{ dev_machine }}" list rscrypto

# Install and verify the canonical remapped Cargo Rail cache policy.
rail-cache-setup *args="":
    @cargo rail cache setup --remote "$CARGO_RAIL_CACHE_REMOTE" --remote-mode "$CARGO_RAIL_CACHE_MODE" --root-portability remap {{ args }}
    @cargo rail cache setup --check --remote "$CARGO_RAIL_CACHE_REMOTE" --remote-mode "$CARGO_RAIL_CACHE_MODE" --root-portability remap {{ args }}
    @cargo rail cache probe

# Report the effective Cargo Rail cache policy and usage.
cache-status:
    @cargo rail cache status --scope local --format json

# Builds
# Build every workspace target with every feature; accepts Cargo build arguments.
build *args="":
    cargo build --locked --workspace --all-targets --all-features {{ args }}

# Checks
# Run affected local checks; pass --all for the full workspace.
check *args="":
    @scripts/check/check.sh {{ args }}

# Run the broad local check set.
check-all:
    @scripts/check/check-all.sh

# Check every supported feature profile.
check-feature-matrix:
    @scripts/check/check-feature-matrix.sh

# Rebuild and verify optimized zeroization evidence.
check-zeroize-evidence:
    @scripts/check/zeroize-evidence.sh

# Run the CI quality policy locally.
ci-check:
    @scripts/ci/ci-check.sh

# Test every supported feature profile.
test-feature-matrix:
    @scripts/test/test-feature-matrix.sh

# Tests
# Test the affected scope or the full workspace with --all.
test *args="":
    @scripts/test/test.sh {{ args }}

# Execute every runnable example with its minimum feature set.
test-examples:
    @scripts/test/test-examples.sh

# Test portable unsafe paths under Miri.
test-miri *args="":
    @scripts/test/test-miri.sh {{ args }}

# Run the RSA leakage evidence harness.
test-rsa-leakage:
    @scripts/test/test-rsa-leakage.sh

# Test Apple Silicon RSA assembly on a physical supported host.
test-rsa-macos-asm:
    @scripts/test/test-rsa-macos-asm.sh

# Run fuzz targets or replay the full fuzz set with --all.
test-fuzz *args="":
    @scripts/test/test-fuzz.sh {{ args }}

# Run fuzz targets with AddressSanitizer.
test-fuzz-asan *args="":
    @scripts/test/test-fuzz-asan.sh {{ args }}

# Constant-Time (CT) Validation Engine
# Run DudeCT Timing Checks
ct-dudect *args="":
    @scripts/ct/dudect.sh {{ args }}

# Build CT Artifacts; Run Timing Evidence; Emit CT Reports
ct-full *args="":
    @scripts/lib/python.sh scripts/ct/full.py {{ args }}

# Run BINSEC; Manifest-Declared Binary CT Kernels
ct-binsec *args="":
    @scripts/lib/python.sh scripts/ct/binsec.py {{ args }}

# Build CT Harness Artifacts
ct-artifacts *args="":
    @scripts/ct/artifacts.sh {{ args }}

# Validate CT Manifest & Generated Artifacts
ct-validate *args="":
    @scripts/lib/python.sh scripts/ct/validate.py {{ args }}

# Coverage

# Generate total coverage, or select --nextest or --fuzz.
test-coverage *args="":
    @scripts/test/test-coverage.sh {{ args }}

# Benches
# Results land in benchmark_results/<YYYY-MM-DD>/<os>/<arch>/results.txt

# Run Criterion benchmarks selected by name or key-value arguments.
bench *args="":
    @scripts/bench/bench.sh {{ args }}

# Stable instruction/cache-cost benchmarks. Requires gungraun-runner and Valgrind.
bench-structural:
    @command -v gungraun-runner >/dev/null || { echo "error: gungraun-runner is required" >&2; exit 1; }
    @command -v valgrind >/dev/null || { echo "error: Valgrind is required" >&2; exit 1; }
    cargo bench --locked --profile bench --features 'checksums,sha2,blake3' --bench structural

# Record one Criterion profiling window with samply.
profile bench filter="" seconds="10":
    @scripts/bench/profile.sh "{{ bench }}" "{{ filter }}" "{{ seconds }}"

# Inspect optimized MIR, LLVM IR, assembly, WASM, or llvm-mca output.
perf-codegen *args="":
    @command -v cargo-asm >/dev/null || { echo "error: cargo-show-asm is required" >&2; exit 1; }
    cargo asm --locked --lib --features full {{ args }}

# Attribute generic instantiation and LLVM IR volume.
perf-llvm-lines *args="":
    @command -v cargo-llvm-lines >/dev/null || { echo "error: cargo-llvm-lines is required" >&2; exit 1; }
    cargo llvm-lines --locked --release --lib --features full {{ args }}

# Maintenance

# Build and apply one Cargo Rail release transaction, including standalone lockfiles.
release-prepare:
    cargo rail release run rscrypto --bump auto --yes --pr

# Verify exact-commit release evidence and create the signed release tag.
release-tag:
    scripts/ci/repository-controls-evidence.sh \
      --commit "$(git rev-parse HEAD)" \
      --output target/repository-controls.json
    scripts/ci/release-evidence-check.sh --commit "$(git rev-parse HEAD)"
    cargo rail release finalize rscrypto --yes --skip-publish

# Update coordinated Cargo manifests
# Update coordinated Cargo manifests, or preview with --check.
update *args="":
    @scripts/update/update-all.sh {{ args }}

# Validate GitHub Actions, local actions, and their repository policy tests.
check-actions:
    @scripts/ci/check-action-pins.sh
    @scripts/ci/check-action-pins-test.sh
    @scripts/ci/tool-integrity-test.sh
    @scripts/ci/remote-cache-recipes-test.sh
    @scripts/ci/capture-cache-status-test.sh
    @scripts/ci/dependabot-smoke-test.sh
    @scripts/ci/check-ci-ownership.sh
    @scripts/ci/check-ci-ownership-test.sh
    @scripts/ci/run-rust-job-test.sh
    @scripts/test/test-fuzz-scheduler-test.sh
    @scripts/ci/emit-manual-matrix-test.sh
    @scripts/ci/materialize-rail-plan-test.sh
    @scripts/ci/changed-test-planning-test.sh
    @scripts/ci/check-worktree-test.sh
    @scripts/ci/pre-push-test.sh
    @scripts/ci/release-evidence-check-test.sh
    @scripts/ci/release-ct-recovery-check-test.sh
    @scripts/ci/repository-controls-evidence-test.sh
    @scripts/ci/release-identity-test.sh
    @scripts/ci/publish-immutable-release-test.sh
    @scripts/ci/release-recipes-test.sh
    @actionlint
    @zizmor .github/workflows .github/actions

# Run the pre-push policy and push the current branch.
push:
    @scripts/ci/pre-push.sh
    git push --set-upstream origin HEAD

# Assets

# Regenerate README Perf SVG from benchmark_results/OVERVIEW.md.
chart:
    @mkdir -p target
    @rustc --edition 2024 -O scripts/render_perf_chart.rs -o target/render_perf_chart
    @target/render_perf_chart
