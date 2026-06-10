# Linux Dev Box via Tailscale + AWS EC2 + Mutagen; Mutagen is one-way sync from the local repo to remote.

ssh-linux target="linux":
    @"$HOME/dev-machines/connect.sh" "{{ target }}" rscrypto

ssh-linux-arch arch:
    @"$HOME/dev-machines/connect.sh" "{{ arch }}" rscrypto

ssh-create-linux-arch arch *args="":
    @DEV_MACHINE_CREATE_VOLUME_SIZE_GB="${DEV_MACHINE_CREATE_VOLUME_SIZE_GB:-30}" "$HOME/dev-machines/create.sh" {{ args }} "{{ arch }}"

ssh-join-linux-arch arch:
    @"$HOME/dev-machines/join-tailscale.sh" "{{ arch }}"

ssh-kill-linux target="linux":
    @"$HOME/dev-machines/kill.sh" "{{ target }}" rscrypto

ssh-kill-linux-arch arch:
    @"$HOME/dev-machines/kill.sh" "{{ arch }}" rscrypto

ssh-status:
    @"$HOME/dev-machines/status.sh"

ssh-bootstrap-linux target="linux":
    @"$HOME/dev-machines/bootstrap.sh" "{{ target }}" rscrypto

ssh-bootstrap-linux-arch arch:
    @"$HOME/dev-machines/bootstrap.sh" "{{ arch }}" rscrypto

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

ci-check:
    @scripts/ci/ci-check.sh

test-feature-matrix:
    @scripts/test/test-feature-matrix.sh

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

# Constant-Time (CT) Validation Engine
ct *args="":
    @scripts/ct/artifacts.sh {{ args }}
    @scripts/ct/validate.py {{ args }}

# Run DudeCT Timing Checks
ct-dudect *args="":
    @scripts/ct/dudect.sh {{ args }}

# Build CT Artifacts; Run Timing Evidence; Emit CT Reports
ct-full *args="":
    @scripts/ct/full.py {{ args }}

# Run BINSEC; Manifest-Declared Binary CT Kernels
ct-binsec *args="":
    @scripts/ct/binsec.py {{ args }}

# Build CT Harness Artifacts
ct-artifacts *args="":
    @scripts/ct/artifacts.sh {{ args }}

# Validate CT Manifest & Generated Artifacts
ct-validate *args="":
    @scripts/ct/validate.py {{ args }}

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

# Update Root/Fuzz Manifests & GHA Pins
update:
    @scripts/update/update-all.sh

update-check:
    @scripts/update/update-all.sh --check

# Refresh .github/actions-lock.yaml SHAs
pin-actions:
    @scripts/ci/pin-actions.sh --update-lock

check-actions:
    @scripts/ci/pin-actions.sh --verify-only

push *args="":
    @scripts/ci/pre-push.sh --light
    git push --no-verify {{ args }}

push-full *args="":
    @scripts/ci/pre-push.sh --full
    git push --no-verify {{ args }}

# Assets

# Regenerate README Perf SVG from benchmark_results/OVERVIEW.md.
chart:
    @mkdir -p target
    @rustc --edition 2024 -O scripts/render_perf_chart.rs -o target/render_perf_chart
    @target/render_perf_chart
