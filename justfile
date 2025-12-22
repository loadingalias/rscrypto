# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# rscrypto Development Commands
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Organization:
#   scripts/check/   - Local development quality checks
#   scripts/ci/      - CI-specific scripts (native builds, strict mode)
#   scripts/test/    - Test runners (unit, miri, fuzz)
#   scripts/bench/   - Benchmarking and tuning
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ─────────────────────────────────────────────────────────────────────────────
# Build
# ─────────────────────────────────────────────────────────────────────────────

build:
    cargo build --workspace --all-targets --all-features
    @echo "✅ Build complete!"

build-release:
    cargo build --workspace --all-targets --all-features --release
    @echo "✅ Release build complete!"

# ─────────────────────────────────────────────────────────────────────────────
# Testing
# ─────────────────────────────────────────────────────────────────────────────

# Run unit and integration tests
test *crates="":
    @scripts/test/test.sh {{crates}}

# Run Miri memory safety tests
test-miri *crates="":
    @scripts/test/test-miri.sh {{crates}}

# Run fuzz tests
# Usage: just test-fuzz                  - Smoke test (60s per target)
#        just test-fuzz fuzz_crc64       - Run specific target
#        just test-fuzz fuzz_crc64 300   - Run target for 300 seconds
#        just test-fuzz --all            - Run all targets
#        just test-fuzz --list           - List available targets
test-fuzz *args="":
    @scripts/test/test-fuzz.sh {{args}}

# Build fuzz targets without running
test-fuzz-build:
    @scripts/test/test-fuzz.sh --build

# Run extended property tests (more iterations)
test-proptests:
    PROPTEST_CASES=10000 cargo nextest run --workspace --all-features -E 'test(/proptest/)' --test-threads=1

# Run all tests: unit/integration, fuzz (smoke), and Miri
test-all:
    just test
    just test-fuzz
    just test-miri

# ─────────────────────────────────────────────────────────────────────────────
# Benchmarking
# ─────────────────────────────────────────────────────────────────────────────

# Run Criterion benchmarks
bench crate="":
    @scripts/bench/bench.sh "{{crate}}"

# Run benchmarks with native CPU optimizations
bench-native crate="":
    RUSTFLAGS='-C target-cpu=native' scripts/bench/bench.sh "{{crate}}"

# Run CRC64 tuning discovery matrix
# Usage: just bench-crc64-tune           - Full matrix (precise)
#        just bench-crc64-tune --quick   - Fast discovery (noisy but useful)
bench-crc64-tune mode="":
    @bash scripts/bench/crc64-tune.sh {{mode}}

# Fast CRC64 tuner (no Criterion; prints recommended RSCRYPTO_CRC64_* exports)
# Usage: just tune-crc64
#        just tune-crc64 --quick
tune-crc64 *args="":
    RUSTC_WRAPPER= cargo run -p checksum --release --bin crc64-tune -- {{args}}

# Summarize Criterion results as TSV
bench-summary group="" only="oneshot":
    @python3 scripts/bench/criterion-summary.py --group-prefix '{{group}}' --only {{only}}

# ─────────────────────────────────────────────────────────────────────────────
# Quality Checks
# ─────────────────────────────────────────────────────────────────────────────

# Host checks (fmt, check, clippy, deny, docs, audit)
# Usage: just check --all | just check foo bar
check *args="":
    @scripts/check/check.sh {{args}}

# Cross-compile for Windows targets (x86_64, aarch64)
check-win *args="":
    @scripts/check/check-win.sh {{args}}

# Cross-compile for Linux targets (x86_64/aarch64 gnu/musl)
check-linux *args="":
    @scripts/check/check-linux.sh {{args}}

# Complete cross-platform check: host + windows + linux + constrained targets
check-all *args="":
    @scripts/check/check-all.sh {{args}}

# CI checks (native only, stricter - runs on actual target machines)
# Runs: fmt --check, check, clippy -D warnings, deny, docs, audit
ci-check:
    @scripts/ci/ci-check.sh

# Quick format and lint (no cross-compilation)
lint:
    cargo fmt --all
    cargo clippy --workspace --all-targets --all-features --fix --allow-dirty -- -D warnings

# ─────────────────────────────────────────────────────────────────────────────
# Maintenance
# ─────────────────────────────────────────────────────────────────────────────

# Update dependencies
update:
    cargo update --workspace
    cargo upgrade --recursive

# Pin GitHub Actions to commit SHAs
pin-actions:
    @scripts/ci/pin-actions.sh --update-lock

# Verify GitHub Actions are pinned
verify-actions:
    @scripts/ci/pin-actions.sh --verify-only

# Clean build artifacts
clean:
    cargo clean

# Clean fuzz artifacts
clean-fuzz:
    @scripts/test/test-fuzz.sh --clean

# ─────────────────────────────────────────────────────────────────────────────
# Coverage & Profiling
# ─────────────────────────────────────────────────────────────────────────────

# Generate fuzz coverage report
# Usage: just fuzz-coverage fuzz_crc64
fuzz-coverage target:
    @scripts/test/test-fuzz.sh --coverage {{target}}
