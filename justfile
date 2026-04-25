# Build the workspace (debug).
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

# Nextest LCOV coverage.
test-coverage:
    @scripts/test/test-coverage.sh --nextest

# Fuzz-corpus LCOV coverage.
test-fuzz-coverage:
    @scripts/test/test-coverage.sh --fuzz

# Nextest + fuzz + merged HTML report.
test-all-coverage:
    @scripts/test/test-coverage.sh --html

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

# Update lockfile + manifests to latest compatible versions.
update:
    @scripts/update/update-all.sh

# Show what `just update` would change without applying.
update-check:
    @scripts/update/update-all.sh --check

# Update .github/actions-lock.yaml to latest upstream SHAs.
pin-actions:
    @scripts/ci/pin-actions.sh --update-lock

# Verify workflow actions match the pin lock file.
check-actions:
    @scripts/ci/pin-actions.sh --verify-only

# Local pre-push hook entry (symlink into .git/hooks/pre-push).
ci-pre-push:
    @scripts/ci/pre-push.sh
