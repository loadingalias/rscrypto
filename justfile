build:
    cargo build --workspace --all-targets --all-features

build-release:
    cargo build --workspace --all-targets --all-features --release

check *args="":
    @scripts/check/check.sh {{args}}

check-all *args="":
    @scripts/check/check-all.sh {{args}}

ci-check:
    @scripts/ci/ci-check.sh

ci-infra-check:
    @scripts/ci/check-infra.sh

test *crates="":
    @scripts/test/test.sh {{crates}}

test-changed *args="":
    @scripts/test/test.sh --changed {{args}}

test-feature-matrix:
    @scripts/test/test-feature-matrix.sh

test-miri *crates="":
    @scripts/test/test-miri.sh {{crates}}

test-fuzz *args="":
    @scripts/test/test-fuzz.sh {{args}}

test-fuzz-full *args="":
    @scripts/test/test-fuzz.sh --full {{args}}

test-fuzz-scoped *args="":
    @scripts/test/test-fuzz.sh --scoped {{args}}

test-fuzz-build:
    @scripts/test/test-fuzz.sh --build

test-fuzz-build-scoped:
    @scripts/test/test-fuzz.sh --scoped-build

test-proptests:
    PROPTEST_CASES=10000 cargo nextest run --workspace --all-features -E 'test(/proptest/)' --test-threads=1

test-all:
    just test --all
    just test-fuzz
    just test-miri

# Coverage reporting.
coverage *args="":
    @scripts/test/test-coverage.sh {{args}}

coverage-nextest:
    @scripts/test/test-coverage.sh --nextest

coverage-fuzz:
    @scripts/test/test-coverage.sh --fuzz

coverage-html:
    @scripts/test/test-coverage.sh --html

# Bench frontdoor (local + CI parity via scripts/ci/run-bench.sh).
# Usage:
#   just bench
#   just bench auth
#   just bench blake3
#   just bench crc32c quick=true
#   just bench checksum
#   just bench blake3 crates=hashes benches=blake3 filter=streaming
bench *args="":
    @scripts/bench/bench.sh {{args}}

bench-native *args="":
    @RUSTFLAGS='-C target-cpu=native' scripts/bench/bench.sh {{args}}

bench-quick *args="":
    @BENCH_QUICK=true scripts/bench/bench.sh {{args}}

blake3-codegen-audit target="x86_64-unknown-linux-gnu" out="target/blake3-codegen":
    @scripts/bench/blake3-codegen-audit.sh {{target}} {{out}}


# Summarize Criterion results as TSV
bench-summary group="" only="oneshot":
    @python3 scripts/bench/criterion-summary.py --group-prefix '{{group}}' --only {{only}}

# Run `checksum` comparison benches and report non-wins vs competitors.
bench-compare group="" ours="rscrypto" min_pct="0":
    RUSTC_WRAPPER= cargo bench --features parallel,checksums --bench crc
    @python3 scripts/bench/criterion-summary.py --group-prefix '{{group}}' --non-wins --ours '{{ours}}' --min-improvement-pct {{min_pct}}

bench-blake3-compare min_pct="0" ours="rscrypto":
    RUSTC_WRAPPER= cargo bench --features parallel,blake3 --bench blake3
    @python3 scripts/bench/criterion-summary.py --group-prefix 'blake3' --non-wins --ours '{{ours}}' --min-improvement-pct {{min_pct}}

bench-blake3-core:
    @scripts/bench/bench.sh benches=blake3 filter=oneshot,streaming,keyed,derive-key,xof

bench-blake3-diag:
    @RSCRYPTO_BLAKE3_BENCH_DIAGNOSTICS=1 scripts/bench/bench.sh benches=blake3

bench-blake3-gate:
    @BENCH_ENFORCE_BLAKE3_GAP_GATE=true scripts/bench/bench.sh benches=blake3 filter=oneshot quick=false

bench-blake3-kernel-gate platform="intel-icl":
    @BENCH_ENFORCE_BLAKE3_KERNEL_GATE=true BENCH_PLATFORM={{platform}} scripts/bench/bench.sh benches=blake3 filter=kernel-ab quick=false

comp-check path:
    @python3 scripts/bench/comp-check.py {{path}}

gen-blake3-x86-asm-ports:
    @scripts/gen_blake3_x86_asm_ports.py

target-matrix-shell:
    @bash scripts/lib/target-matrix.sh --format shell

target-matrix-json key:
    @bash scripts/lib/target-matrix.sh --format json --key "{{key}}"

gen-hashes-testdata:
    @python3 scripts/gen_hashes_testdata.py


ci-pin-actions:
    @scripts/ci/pin-actions.sh --update-lock

ci-verify-actions:
    @scripts/ci/pin-actions.sh --verify-only

# Update Rust dependencies in the lockfile and manifests.
update:
    @scripts/update/update-all.sh

update-check:
    @scripts/update/update-all.sh --check

# Legacy convenience wrappers used in contributor docs.
pin-actions:
    @just ci-pin-actions

verify-actions:
    @just ci-verify-actions

ci-pre-push:
    @scripts/ci/pre-push.sh

fuzz-coverage:
    @scripts/test/test-coverage.sh --fuzz
