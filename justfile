build:
    cargo build --workspace --all-targets --all-features

build-release:
    cargo build --workspace --all-targets --all-features --release

check *args="":
    @scripts/check/check.sh {{args}}

check-win *args="":
    @scripts/check/check-win.sh {{args}}

check-linux *args="":
    @scripts/check/check-linux.sh {{args}}

check-all *args="":
    @scripts/check/check-all.sh {{args}}

ci-check:
    @scripts/ci/ci-check.sh

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
#   just tune-quick -- --crate hashes --only blake3
#   just tune-report dir=target/tune -- --enforce-targets
#   just tune-list
tune *args="":
    RUSTC_WRAPPER= RUSTFLAGS='-C target-cpu=native' cargo run -p tune --release --bin rscrypto-tune -- {{args}}

tune-quick *args="":
    RUSTC_WRAPPER= RUSTFLAGS='-C target-cpu=native' cargo run -p tune --release --bin rscrypto-tune -- --quick {{args}}

tune-apply *args="":
    RUSTC_WRAPPER= RUSTFLAGS='-C target-cpu=native' cargo run -p tune --release --bin rscrypto-tune -- --apply {{args}}

tune-quick-apply *args="":
    RUSTC_WRAPPER= RUSTFLAGS='-C target-cpu=native' cargo run -p tune --release --bin rscrypto-tune -- --quick --apply {{args}}

tune-list:
    RUSTC_WRAPPER= RUSTFLAGS='-C target-cpu=native' cargo run -p tune --release --bin rscrypto-tune -- --list

tune-crate crate *args="":
    RUSTC_WRAPPER= RUSTFLAGS='-C target-cpu=native' cargo run -p tune --release --bin rscrypto-tune -- --crate "{{crate}}" {{args}}

tune-report dir="target/tune" *args="":
    RUSTC_WRAPPER= RUSTFLAGS='-C target-cpu=native' cargo run -p tune --release --bin rscrypto-tune -- --quick --report-dir "{{dir}}" {{args}}

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


update:
    cargo update --workspace
    cargo upgrade --recursive

pin-actions:
    @scripts/ci/pin-actions.sh --update-lock

verify-actions:
    @scripts/ci/pin-actions.sh --verify-only

fuzz-coverage target:
    @scripts/test/test-fuzz.sh --coverage {{target}}
