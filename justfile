# Remote dev. Provider mechanics live in ~/dev-machines.

ssh target:
    @"$HOME/dev-machines/dev-machine" ssh rscrypto "{{ target }}"

ssh-check target:
    @"$HOME/dev-machines/dev-machine" ssh rscrypto "{{ target }}" --check

ssh-create target *args="":
    @"$HOME/dev-machines/dev-machine" create rscrypto "{{ target }}" {{ args }}

ssh-kill target:
    @"$HOME/dev-machines/dev-machine" kill rscrypto "{{ target }}"

ssh-status target="":
    @if [ -n "{{ target }}" ]; then "$HOME/dev-machines/dev-machine" status rscrypto "{{ target }}"; else "$HOME/dev-machines/dev-machine" status rscrypto; fi

ssh-bootstrap target:
    @"$HOME/dev-machines/dev-machine" bootstrap rscrypto "{{ target }}"

ssh-aws-linux-x64:
    @"$HOME/dev-machines/dev-machine" ssh rscrypto aws-linux-x64

ssh-aws-linux-arm64:
    @"$HOME/dev-machines/dev-machine" ssh rscrypto aws-linux-arm64

ssh-aws-windows-x64:
    @"$HOME/dev-machines/dev-machine" ssh rscrypto aws-windows-x64

ssh-azure-linux-x64:
    @"$HOME/dev-machines/dev-machine" ssh rscrypto azure-linux-x64

ssh-azure-linux-arm64:
    @"$HOME/dev-machines/dev-machine" ssh rscrypto azure-linux-arm64

ssh-azure-windows-x64:
    @"$HOME/dev-machines/dev-machine" ssh rscrypto azure-windows-x64

ssh-azure-windows-arm64:
    @"$HOME/dev-machines/dev-machine" ssh rscrypto azure-windows-arm64

ssh-aws-linux-x64-perf:
    @"$HOME/dev-machines/dev-machine" ssh rscrypto aws-linux-x64-perf

ssh-aws-linux-intel-gnr-profile:
    @"$HOME/dev-machines/dev-machine" ssh rscrypto aws-linux-intel-gnr-profile

ssh-aws-linux-intel-spr-profile:
    @"$HOME/dev-machines/dev-machine" ssh rscrypto aws-linux-intel-spr-profile

ssh-aws-linux-amd-zen5-profile:
    @"$HOME/dev-machines/dev-machine" ssh rscrypto aws-linux-amd-zen5-profile

ssh-aws-linux-amd-zen4-profile:
    @"$HOME/dev-machines/dev-machine" ssh rscrypto aws-linux-amd-zen4-profile

ssh-aws-linux-arm64-graviton4-profile:
    @"$HOME/dev-machines/dev-machine" ssh rscrypto aws-linux-arm64-graviton4-profile

ssh-azure-linux-intel-gnr-profile:
    @"$HOME/dev-machines/dev-machine" ssh rscrypto azure-linux-intel-gnr-profile

ssh-azure-linux-intel-emr-profile:
    @"$HOME/dev-machines/dev-machine" ssh rscrypto azure-linux-intel-emr-profile

ssh-azure-linux-amd-zen5-profile:
    @"$HOME/dev-machines/dev-machine" ssh rscrypto azure-linux-amd-zen5-profile

ssh-azure-linux-amd-zen4-profile:
    @"$HOME/dev-machines/dev-machine" ssh rscrypto azure-linux-amd-zen4-profile

ssh-azure-linux-arm64-cobalt-profile:
    @"$HOME/dev-machines/dev-machine" ssh rscrypto azure-linux-arm64-cobalt-profile

ssh-azure-linux-arm64-ampere-profile:
    @"$HOME/dev-machines/dev-machine" ssh rscrypto azure-linux-arm64-ampere-profile

ssh-azure-windows-amd-zen5-profile:
    @"$HOME/dev-machines/dev-machine" ssh rscrypto azure-windows-amd-zen5-profile

ssh-azure-windows-intel-gnr-profile:
    @"$HOME/dev-machines/dev-machine" ssh rscrypto azure-windows-intel-gnr-profile

ssh-azure-windows-arm64-profile:
    @"$HOME/dev-machines/dev-machine" ssh rscrypto azure-windows-arm64-profile

ssh-aws-test-rdma:
    @"$HOME/dev-machines/dev-machine" ssh rscrypto aws-test-rdma

ssh-azure-test-rdma:
    @"$HOME/dev-machines/dev-machine" ssh rscrypto azure-test-rdma

# Builds
build:
    cargo build --workspace --all-targets --all-features

build-release:
    cargo build --workspace --all-targets --all-features --release

# Checks
check *args="":
    @scripts/check/check.sh {{ args }}

check-all *args="":
    @scripts/check/check-all.sh {{ args }}

check-feature-matrix:
    @scripts/check/check-feature-matrix.sh

check-zeroize-evidence:
    @scripts/check/zeroize-evidence.sh

check-unify:
    cargo rail config validate --strict
    cargo rail config migrate --check
    cargo rail unify --check --explain

ci-check:
    @scripts/ci/ci-check.sh

test-feature-matrix:
    @scripts/test/test-feature-matrix.sh

test-native-api:
    cargo test --no-default-features --features 'alloc,aead,ed25519,x25519,ecdsa,ml-kem' --test api_consistency
    cargo test --features 'aead,signatures,key-exchange,getrandom' --test api_consistency
    cargo test --features 'signatures,key-exchange,getrandom' --test getrandom_smoke

# Tests
test *crates="":
    @scripts/test/test.sh {{ crates }}

test-all:
    @scripts/test/test.sh --all

test-miri *crates="":
    @scripts/test/test-miri.sh {{ crates }}

test-rsa-leakage:
    @scripts/test/test-rsa-leakage.sh

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

# Maintenance

# Release
release-change bump message:
    cargo rail change add rscrypto --bump "{{ bump }}" --message "{{ message }}"

release-status:
    cargo rail change status

release-check:
    just check-unify
    cargo rail change status
    cargo rail release check rscrypto --extended
    scripts/ci/release-plan-check.sh rscrypto

check-repository-controls:
    @scripts/ci/repository-controls-evidence.sh \
      --commit "$(git rev-parse HEAD)" \
      --output target/repository-controls.json

release-prepare:
    just check-unify
    cargo rail release check rscrypto --extended
    cargo rail release run rscrypto --bump auto --yes --pr
    cargo update --manifest-path tools/ct-harness/Cargo.toml -p rscrypto
    cargo update --manifest-path tools/ct-dudect/Cargo.toml -p rscrypto
    cargo update --manifest-path tools/ct-binsec-harness/Cargo.toml -p rscrypto
    git add tools/ct-harness/Cargo.lock tools/ct-dudect/Cargo.lock tools/ct-binsec-harness/Cargo.lock
    git diff --cached --quiet || git commit -m "workspace: sync CT tool locks for release"
    git push

release-tag:
    just check-unify
    just check-repository-controls
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
    @scripts/ci/emit-manual-matrix-test.sh
    @scripts/ci/changed-test-planning-test.sh
    @scripts/ci/check-worktree-test.sh
    @scripts/ci/pre-push-test.sh
    @scripts/ci/release-evidence-check-test.sh
    @scripts/ci/repository-controls-evidence-test.sh
    @scripts/ci/release-identity-test.sh
    @scripts/ci/publish-immutable-release-test.sh
    @scripts/ci/release-recipes-test.sh
    @actionlint
    @zizmor .github/workflows .github/actions

push remote="origin":
    @scripts/ci/pre-push.sh --light
    git push --set-upstream "{{ remote }}" HEAD

push-full remote="origin":
    @scripts/ci/pre-push.sh --full
    git push --set-upstream "{{ remote }}" HEAD

# Assets

# Regenerate README Perf SVG from benchmark_results/OVERVIEW.md.
chart:
    @mkdir -p target
    @rustc --edition 2024 -O scripts/render_perf_chart.rs -o target/render_perf_chart
    @target/render_perf_chart
