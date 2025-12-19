build:
    cargo build --workspace --all-targets --all-features
    @echo "✅ Build complete!"

build-release:
    cargo build --workspace --all-targets --all-features --release
    @echo "✅ Release build complete!"

test *crates="":
    @scripts/test/test.sh {{crates}}

test-miri *crates="":
    @scripts/test/test-miri.sh {{crates}}

# Run fuzz tests
# Usage: just test-fuzz                  - Smoke test (30s per target)
#        just test-fuzz fuzz_crc32c      - Run specific target
#        just test-fuzz fuzz_crc32c 60   - Run target for 60 seconds
#        just test-fuzz --all            - Run all targets
#        just test-fuzz --list           - List available targets
test-fuzz *args="":
    @scripts/test/test-fuzz.sh {{args}}

# Build fuzz targets without running
test-fuzz-build:
    @scripts/test/test-fuzz.sh --build

# Run all tests: unit/integration, fuzz (smoke), and Miri
test-all:
    just test
    just test-fuzz
    just test-miri

bench crate="":
    @scripts/bench/bench.sh "{{crate}}"

bench-native crate="":
    RUSTFLAGS='-C target-cpu=native' scripts/bench/bench.sh "{{crate}}"

# Local checks with cross-compilation verification
check:
    @scripts/ci/check.sh

# CI checks (no cross-compile, stricter - runs natively on each platform)
ci-check:
    @scripts/ci/check.sh --ci

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

# Generate fuzz coverage report
# Usage: just fuzz-coverage fuzz_crc32c
fuzz-coverage target:
    @scripts/test/test-fuzz.sh --coverage {{target}}

# Clean fuzz artifacts
clean-fuzz:
    @scripts/test/test-fuzz.sh --clean
