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

check:
    cargo fmt --all
    cargo check --workspace --all-targets --all-features
    cargo clippy --workspace --all-targets --all-features --fix --allow-dirty -- -D warnings
    cargo deny check all
    RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --all-features
    cargo audit
    @echo "🔍 Cross-target compile matrix (checksum crate)"
    RUSTC_WRAPPER= cargo check -p checksum --all-features --target x86_64-unknown-linux-gnu
    RUSTC_WRAPPER= cargo check -p checksum --all-features --target x86_64-pc-windows-msvc
    RUSTC_WRAPPER= cargo check -p checksum --all-features --target aarch64-unknown-linux-gnu
    RUSTC_WRAPPER= cargo check -p checksum --no-default-features
    RUSTC_WRAPPER= cargo check -p checksum --no-default-features --features alloc
    RUSTC_WRAPPER= cargo check -p checksum --no-default-features --target wasm32-unknown-unknown
    RUSTC_WRAPPER= cargo check -p checksum --no-default-features --features alloc --target wasm32-unknown-unknown
    @echo "✅ All checks passed!"

# CI checks (no auto-fix, fail-fast)
ci-check:
    cargo fmt --all -- --check
    cargo check --workspace --all-targets --all-features
    cargo clippy --workspace --all-targets --all-features -- -D warnings
    cargo deny check all
    RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --all-features
    cargo audit
    @echo "✅ CI checks passed!"

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

# Clean fuzz artifacts
clean-fuzz:
    @scripts/test/test-fuzz.sh --clean
