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
# Unified tuning engine - benchmarks all checksum algorithms and finds optimal settings
# Usage: just tune            - Full tuning run with detailed output
#        just tune --quick    - Faster, noisier measurements
#        just tune --verbose  - Show progress during tuning
#        just tune --format env   - Output as shell export statements
#        just tune --format json  - Output as JSON
tune *args="":
    RUSTC_WRAPPER= RUSTFLAGS='-C target-cpu=native' cargo run -p tune --release --bin rscrypto-tune -- {{args}}

# Quick tune alias (faster measurements, still useful for development)
tune-quick *args="":
    RUSTC_WRAPPER= RUSTFLAGS='-C target-cpu=native' cargo run -p tune --release --bin rscrypto-tune -- --quick {{args}}

# Apply tuned defaults into the repo for this machine (writes into crates/checksum)
tune-apply *args="":
    RUSTC_WRAPPER= RUSTFLAGS='-C target-cpu=native' cargo run -p tune --release --bin rscrypto-tune -- --apply {{args}}

# Quick mode + apply (faster, noisier; still useful for iterative tuning)
tune-quick-apply *args="":
    RUSTC_WRAPPER= RUSTFLAGS='-C target-cpu=native' cargo run -p tune --release --bin rscrypto-tune -- --quick --apply {{args}}

# Generate markdown for contributing tuning results (copy-paste to GitHub issue)
tune-contribute:
    RUSTC_WRAPPER= RUSTFLAGS='-C target-cpu=native' cargo run -p tune --release --bin rscrypto-tune -- --format contribute


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
