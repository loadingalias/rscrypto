# Local Linux dev box (Tailscale + EC2 + one-way Mutagen).
# Scripts live at ~/dev-machines and sync this workspace to /home/ubuntu/rscrypto.

# Wake/connect the Linux dev box. Optional .github/runs-on.yml runner aliases
# select metadata only; x86 aliases still use the existing linux-dev Tailnet node.
ssh-linux target="linux":
    @"$HOME/dev-machines/connect.sh" "{{ target }}" rscrypto

ssh-kill-linux target="linux":
    @"$HOME/dev-machines/kill.sh" "{{ target }}" rscrypto

ssh-status:
    @"$HOME/dev-machines/status.sh"

ssh-bootstrap-linux target="linux":
    @"$HOME/dev-machines/bootstrap.sh" "{{ target }}" rscrypto

build:
    cargo build --workspace --all-targets --all-features

# Build the workspace (release).
build-release:
    cargo build --workspace --all-targets --all-features --release

# ─── Checks ────────────────────────────────────────────────────────

# Host fmt, check, clippy, docs (+ rscrypto feature matrix when in scope).
check *args="":
    @scripts/check/check.sh {{ args }}

# Host + cross-targets (windows, linux, ibm, no_std, wasm).
check-all *args="":
    @scripts/check/check-all.sh {{ args }}

# Native CI quality lane (runs on target machines in CI).
ci-check:
    @scripts/ci/ci-check.sh

# Executable feature-flag matrix: compile + test 26 combinations.
test-feature-matrix:
    @scripts/test/test-feature-matrix.sh

# ─── Tests ─────────────────────────────────────────────────────────

# Rail-scoped tests (cargo-rail picks affected crates). Pass crate names to narrow (e.g. `just test aead`).
test *crates="":
    @scripts/test/test.sh {{ crates }}

# Full workspace tests.
test-all:
    @scripts/test/test.sh --all

# Miri memory-safety tests. Pass a crate name to narrow.
test-miri *crates="":
    @scripts/test/test-miri.sh {{ crates }}

# Fuzz: build scoped packages + run full harness. Pass a target name to run a single target.
test-fuzz *args="":
    @scripts/test/test-fuzz.sh {{ args }}

# ─── Coverage ──────────────────────────────────────────────────────

# Total source coverage: nextest + deterministic fuzz corpus replay.
test-coverage:
    @scripts/test/test-coverage.sh

# Nextest-only LCOV coverage.
test-nextest-coverage:
    @scripts/test/test-coverage.sh --nextest

# Fuzz-corpus replay LCOV coverage.
test-fuzz-coverage:
    @scripts/test/test-coverage.sh --fuzz

# Alias for total source coverage.
test-all-coverage:
    @scripts/test/test-coverage.sh

# ─── Benches ───────────────────────────────────────────────────────
# Results land in benchmark_results/<YYYY-MM-DD>/<os>/<arch>/results.txt
# (same layout the /extract-bench skill uses for CI artifacts).

# Full workspace bench. Pass a scope (aead, blake3, crc64, ...) to narrow.
bench *args="":
    @scripts/bench/bench.sh {{ args }}

# Criterion --quick samples (fast local sanity check).
bench-quick *args="":
    @scripts/bench/bench.sh --quick {{ args }}

# ─── Maintenance ───────────────────────────────────────────────────

# Update root/fuzz manifests and GitHub Actions refs/pins.
update:
    @scripts/update/update-all.sh

# Show what `just update` would change without applying.
update-check:
    @scripts/update/update-all.sh --check

# Refresh .github/actions-lock.yaml SHAs for the currently locked refs.
pin-actions:
    @scripts/ci/pin-actions.sh --update-lock

# Verify workflow actions match the pin lock file.
check-actions:
    @scripts/ci/pin-actions.sh --verify-only

# Local pre-push hook entry (symlink into .git/hooks/pre-push).
ci-pre-push:
    @scripts/ci/pre-push.sh

# ─── Assets ────────────────────────────────────────────────────────

# Regenerate README perf chart SVGs from benchmark_results/OVERVIEW.md.
chart:
    @mkdir -p target
    @rustc --edition 2024 -O scripts/render_perf_chart.rs -o target/render_perf_chart
    @target/render_perf_chart
