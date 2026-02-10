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

test-miri *crates="":
    @scripts/test/test-miri.sh {{crates}}

test-fuzz *args="":
    @scripts/test/test-fuzz.sh {{args}}

test-fuzz-build:
    @scripts/test/test-fuzz.sh --build

test-proptests:
    PROPTEST_CASES=10000 cargo nextest run --workspace --all-features -E 'test(/proptest/)' --test-threads=1

test-all:
    just test
    just test-fuzz
    just test-miri

bench crate="" bench="":
    @scripts/bench/bench.sh "{{crate}}" "{{bench}}"

bench-native crate="" bench="":
    @RUSTFLAGS='-C target-cpu=native' scripts/bench/bench.sh "{{crate}}" "{{bench}}"


# Tuning
# NOTE: The 'tune' crate is dev-only and not part of the workspace build.
# It's built on-demand when these commands are run, keeping workspace builds fast.
# Usage:
#   just tune
#   just tune-measure dir=target/tune
#   just tune-derive raw=target/tune/raw-results.json
#   just tune-quick -- --crate hashes --only blake3
#   just tune -- --repeats 5 --aggregate trimmed-mean
#   just tune-report dir=target/tune -- --enforce-targets
tune *args="":
    RUSTC_WRAPPER= RUSTFLAGS='-C target-cpu=native' cargo run -p tune --release --bin rscrypto-tune -- {{args}}

# Developer preview only (faster, noisier). Not for dispatch decisions.
tune-quick *args="":
    RUSTC_WRAPPER= RUSTFLAGS='-C target-cpu=native' cargo run -p tune --release --bin rscrypto-tune -- --quick {{args}}

tune-apply *args="":
    RUSTC_WRAPPER= RUSTFLAGS='-C target-cpu=native' cargo run -p tune --release --bin rscrypto-tune -- --apply {{args}}

tune-measure dir="target/tune" *args="":
    RUSTC_WRAPPER= RUSTFLAGS='-C target-cpu=native' cargo run -p tune --release --bin rscrypto-tune -- --measure-only --report-dir "{{dir}}" --raw-output "{{dir}}/raw-results.json" {{args}}

tune-derive raw="target/tune/raw-results.json" dir="target/tune" *args="":
    RUSTC_WRAPPER= RUSTFLAGS='-C target-cpu=native' cargo run -p tune --release --bin rscrypto-tune -- --derive-from "{{raw}}" --report-dir "{{dir}}" {{args}}

tune-report dir="target/tune" raw="target/tune/raw-results.json" *args="":
    RUSTC_WRAPPER= RUSTFLAGS='-C target-cpu=native' cargo run -p tune --release --bin rscrypto-tune -- --derive-from "{{raw}}" --report-dir "{{dir}}" {{args}}

tune-contribute *args="":
    RUSTC_WRAPPER= RUSTFLAGS='-C target-cpu=native' cargo run -p tune --release --bin rscrypto-tune -- --format contribute {{args}}


# Summarize Criterion results as TSV
bench-summary group="" only="oneshot":
    @python3 scripts/bench/criterion-summary.py --group-prefix '{{group}}' --only {{only}}

# Run `checksum` comparison benches and report non-wins vs competitors.
bench-compare group="" ours="rscrypto/checksum" min_pct="0":
    RUSTC_WRAPPER= cargo bench -p checksum --bench comp
    @python3 scripts/bench/criterion-summary.py --group-prefix '{{group}}' --non-wins --ours '{{ours}}' --min-improvement-pct {{min_pct}}

bench-blake3-compare min_pct="0":
    RUSTC_WRAPPER= cargo bench -p hashes --bench comp
    @python3 scripts/bench/criterion-summary.py --group-prefix 'blake3/' --non-wins --ours 'rscrypto/blake3' --min-improvement-pct {{min_pct}}

comp-check path:
    @python3 scripts/bench/comp-check.py {{path}}

gen-blake3-x86-asm-ports:
    @scripts/gen_blake3_x86_asm_ports.py


ci-pin-actions:
    @scripts/ci/pin-actions.sh --update-lock

ci-verify-actions:
    @scripts/ci/pin-actions.sh --verify-only

# Update Rust dependencies in the lockfile and manifests.
update:
    cargo update --workspace
    cargo upgrade --recursive

# Legacy convenience wrappers used in contributor docs.
pin-actions:
    @just ci-pin-actions

verify-actions:
    @just ci-verify-actions

ci-pre-push:
    @scripts/ci/pre-push.sh

fuzz-coverage target:
    @scripts/test/test-fuzz.sh --coverage {{target}}
