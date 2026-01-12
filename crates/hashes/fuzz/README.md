# Hash Fuzzing (Differential)

This fuzz workspace compares `hashes` against well-known reference crates.

## Prereqs

- `cargo install cargo-fuzz`

## Run

From the repo root:

- `RUSTC_WRAPPER= cargo fuzz run sha256 --manifest-path crates/hashes/fuzz/Cargo.toml`
- `RUSTC_WRAPPER= cargo fuzz run sha2_family --manifest-path crates/hashes/fuzz/Cargo.toml`
- `RUSTC_WRAPPER= cargo fuzz run sha512 --manifest-path crates/hashes/fuzz/Cargo.toml`
- `RUSTC_WRAPPER= cargo fuzz run sha3 --manifest-path crates/hashes/fuzz/Cargo.toml`
- `RUSTC_WRAPPER= cargo fuzz run sha3_family --manifest-path crates/hashes/fuzz/Cargo.toml`
- `RUSTC_WRAPPER= cargo fuzz run shake256 --manifest-path crates/hashes/fuzz/Cargo.toml`
- `RUSTC_WRAPPER= cargo fuzz run shake --manifest-path crates/hashes/fuzz/Cargo.toml`
- `RUSTC_WRAPPER= cargo fuzz run blake3 --manifest-path crates/hashes/fuzz/Cargo.toml`
- `RUSTC_WRAPPER= cargo fuzz run siphash --manifest-path crates/hashes/fuzz/Cargo.toml`
- `RUSTC_WRAPPER= cargo fuzz run blake2 --manifest-path crates/hashes/fuzz/Cargo.toml`
- `RUSTC_WRAPPER= cargo fuzz run ascon --manifest-path crates/hashes/fuzz/Cargo.toml`
- `RUSTC_WRAPPER= cargo fuzz run cshake --manifest-path crates/hashes/fuzz/Cargo.toml`
- `RUSTC_WRAPPER= cargo fuzz run kmac --manifest-path crates/hashes/fuzz/Cargo.toml`
