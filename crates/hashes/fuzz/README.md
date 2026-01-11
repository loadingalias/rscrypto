# Hash Fuzzing (Differential)

This fuzz workspace compares `hashes` against well-known reference crates.

## Prereqs

- `cargo install cargo-fuzz`

## Run

From the repo root:

- `RUSTC_WRAPPER= cargo fuzz run sha256 --manifest-path crates/hashes/fuzz/Cargo.toml`
- `RUSTC_WRAPPER= cargo fuzz run sha3 --manifest-path crates/hashes/fuzz/Cargo.toml`
- `RUSTC_WRAPPER= cargo fuzz run blake3 --manifest-path crates/hashes/fuzz/Cargo.toml`

