# Remote dev. Provider mechanics live in ~/dev-machines.

dev_machine := env_var_or_default("DEV_MACHINE_BIN", env_var("HOME") + "/dev-machines/dev-machine")

ssh target *args="":
    @"{{ dev_machine }}" ssh rscrypto "{{ target }}" {{ args }}

ssh-check target *args="":
    @"{{ dev_machine }}" ssh rscrypto "{{ target }}" --check {{ args }}

ssh-preflight target:
    @"{{ dev_machine }}" preflight rscrypto "{{ target }}"

ssh-create target *args="":
    @"{{ dev_machine }}" create rscrypto "{{ target }}" {{ args }}

ssh-start target:
    @"{{ dev_machine }}" start rscrypto "{{ target }}"

ssh-deallocate target:
    @"{{ dev_machine }}" deallocate rscrypto "{{ target }}"

ssh-kill target:
    @"{{ dev_machine }}" kill rscrypto "{{ target }}"

ssh-status target="":
    @if [ -n "{{ target }}" ]; then "{{ dev_machine }}" status rscrypto "{{ target }}"; else "{{ dev_machine }}" status rscrypto; fi

ssh-bootstrap target profile="":
    @if [ -n "{{ profile }}" ]; then "{{ dev_machine }}" bootstrap rscrypto "{{ target }}" "{{ profile }}"; else "{{ dev_machine }}" bootstrap rscrypto "{{ target }}"; fi

ssh-just target *args="":
    @"{{ dev_machine }}" just rscrypto "{{ target }}" {{ args }}

ssh-collect-bench target run_id destination:
    @"{{ dev_machine }}" collect-bench rscrypto "{{ target }}" "{{ run_id }}" "{{ destination }}"

ssh-list:
    @"{{ dev_machine }}" list

# Builds
build:
    cargo build --locked --workspace --all-targets --all-features

build-release:
    cargo build --locked --workspace --all-targets --all-features --release

# Checks
check *args="":
    @scripts/check/check.sh {{ args }}

check-all *args="":
    @scripts/check/check-all.sh {{ args }}

check-feature-matrix:
    @scripts/check/check-feature-matrix.sh

check-zeroize-evidence:
    @scripts/check/zeroize-evidence.sh

ci-check:
    @scripts/ci/ci-check.sh

test-feature-matrix:
    @scripts/test/test-feature-matrix.sh

test-native-api:
    cargo test --locked --no-default-features --features 'alloc,aead,ed25519,x25519,ecdsa,ml-kem' --test api_consistency
    cargo test --locked --features 'aead,signatures,key-exchange,getrandom' --test api_consistency
    cargo test --locked --features 'signatures,key-exchange,getrandom' --test getrandom_smoke

# Tests
test *crates="":
    @scripts/test/test.sh {{ crates }}

test-all:
    @scripts/test/test.sh --all

test-miri *crates="":
    @scripts/test/test-miri.sh {{ crates }}

test-rsa-leakage:
    @scripts/test/test-rsa-leakage.sh

test-rsa-macos-asm:
    @scripts/test/test-rsa-macos-asm.sh

test-fuzz *args="":
    @scripts/test/test-fuzz.sh {{ args }}

test-fuzz-asan *args="":
    @scripts/test/test-fuzz-asan.sh {{ args }}

# Constant-Time (CT) Validation Engine
ct *args="":
    @scripts/ct/artifacts.sh {{ args }}
    @scripts/ct/python.sh scripts/ct/validate.py {{ args }}

# Run DudeCT Timing Checks
ct-dudect *args="":
    @scripts/ct/dudect.sh {{ args }}

# Build CT Artifacts; Run Timing Evidence; Emit CT Reports
ct-full *args="":
    @scripts/ct/python.sh scripts/ct/full.py {{ args }}

# Run BINSEC; Manifest-Declared Binary CT Kernels
ct-binsec *args="":
    @scripts/ct/python.sh scripts/ct/binsec.py {{ args }}

# Build CT Harness Artifacts
ct-artifacts *args="":
    @scripts/ct/artifacts.sh {{ args }}

# Validate CT Manifest & Generated Artifacts
ct-validate *args="":
    @scripts/ct/python.sh scripts/ct/validate.py {{ args }}

# Coverage

# Total Coverage: nextest + fuzz corpus replay
test-coverage:
    @scripts/test/test-coverage.sh

# Nextest LCOV
test-nextest-coverage:
    @scripts/test/test-coverage.sh --nextest

# Fuzz-corpus replay LCOV
test-fuzz-coverage:
    @scripts/test/test-coverage.sh --fuzz

# Benches
# Results land in benchmark_results/<YYYY-MM-DD>/<os>/<arch>/results.txt

bench *args="":
    @scripts/bench/bench.sh {{ args }}

bench-quick *args="":
    @scripts/bench/bench.sh --quick {{ args }}

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

# Release adapters not yet expressible as typed Cargo Rail release policy.
release-prepare:
    cargo rail release check rscrypto --extended
    cargo rail release run rscrypto --bump auto --yes --pr
    cargo update --manifest-path tools/ct-harness/Cargo.toml -p rscrypto
    cargo update --manifest-path tools/ct-dudect/Cargo.toml -p rscrypto
    cargo update --manifest-path tools/ct-binsec-harness/Cargo.toml -p rscrypto
    git add tools/ct-harness/Cargo.lock tools/ct-dudect/Cargo.lock tools/ct-binsec-harness/Cargo.lock
    git diff --cached --quiet || git commit -m "workspace: sync CT tool locks for release"
    git push

release-tag:
    scripts/ci/repository-controls-evidence.sh \
      --commit "$(git rev-parse HEAD)" \
      --output target/repository-controls.json
    scripts/ci/release-evidence-check.sh --commit "$(git rev-parse HEAD)"
    cargo rail release finalize rscrypto --yes --skip-publish

# Update coordinated Cargo manifests
update:
    @scripts/update/update-all.sh

update-check:
    @scripts/update/update-all.sh --check

check-actions:
    @scripts/ci/check-action-pins.sh
    @scripts/ci/check-action-pins-test.sh
    @scripts/ci/tool-integrity-test.sh
    @scripts/ci/dependabot-smoke-test.sh
    @scripts/ci/check-ci-ownership.sh
    @scripts/ci/check-ci-ownership-test.sh
    @scripts/ci/run-rust-job-test.sh
    @scripts/test/test-fuzz-scheduler-test.sh
    @scripts/ci/emit-manual-matrix-test.sh
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

push:
    @scripts/ci/pre-push.sh
    git push --set-upstream origin HEAD

# Assets

# Regenerate README Perf SVG from benchmark_results/OVERVIEW.md.
chart:
    @mkdir -p target
    @rustc --edition 2024 -O scripts/render_perf_chart.rs -o target/render_perf_chart
    @target/render_perf_chart
