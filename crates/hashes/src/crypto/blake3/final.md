# BLAKE3 World-Class Execution Plan (2026-02-17)

## Hard Facts

- Real SIMD BLAKE3 leaf kernels are now implemented for `s390x`, `powerpc64`, and `riscv64`.
- IBM/RISC-V full-chunk paths are no longer wrapper-only scalar forwarding.
- x86 has verified oneshot cliffs around `65B` and `16KiB`.
- Work is not done until x86 boundary retune and cross-arch baselines are fully closed.

## Current Evidence Snapshot

- `cargo asm` (`s390x`, `riscv64`) and `--emit=asm` fallback (`powerpc64le`) now show wrapper hot loops calling arch-local chunk helpers (`hash_one_chunk_*`, `chunk_compress_blocks_*`) instead of scalar forwarding helpers.
- `cargo llvm-lines` on all three targets no longer lists `hash_one_chunk_unchecked`/`chunk_compress_blocks_portable` in the audited hot-set.
- SIMD round kernels are live for IBM/RISC-V leaf/compress and 4-lane full-chunk paths.
- Parent folding on IBM/RISC-V now has a SIMD path (`parent_cvs_many4_simd`).
- x86 tiny finalize path now uses kernel-bound function pointers (reduced hot-path control flow).
- Cross-arch evidence workflow is now scripted in:
  - `scripts/bench/blake3-codegen-audit.sh`
- Baseline parity enforcement now includes IBM + RISC-V tune arches:
  - `config/target-matrix.toml` `[tune].arches` now includes `s390x`, `powerpc64le`, `riscv64`.
  - legacy tune enforcement required matching BLAKE3 baseline files.

## Priority Order

1. `s390x/vector` real leaf kernel
2. `powerpc64/vsx` real leaf kernel
3. `riscv64/v` real leaf kernel
4. x86 boundary retune (`65B`, `16KiB`)

## P0: Ship Real IBM/RISC-V Kernels

### P0.1 s390x/vector

- [x] Implement non-portable leaf bodies for:
  - `compress_s390x_vector`
  - `chunk_compress_blocks_s390x_vector`
  - `hash_many_contiguous_s390x_vector`
- [x] Ensure full-chunk hot path does not route through portable/scalar fallback.
- Acceptance:
  - [x] `cargo asm --target s390x-unknown-linux-gnu` shows vector ops in hot loop.
  - [x] `cargo llvm-lines` no longer dominated by scalar fallback on this path.

### P0.2 powerpc64/vsx

- [x] Implement non-portable leaf bodies for:
  - `compress_power_vsx`
  - `chunk_compress_blocks_power_vsx`
  - `hash_many_contiguous_power_vsx`
- [x] Keep Power asm extraction robust (cargo-asm parse fallback stays supported).
- Acceptance:
  - [x] asm evidence (via `cargo asm` or fallback extraction) shows VSX path doing real work.
  - [x] scalar fallback no longer dominates routing for full-chunk hot path.

### P0.3 riscv64/v

- [x] Implement non-portable leaf bodies for:
  - `compress_riscv_v`
  - `chunk_compress_blocks_riscv_v`
  - `hash_many_contiguous_riscv_v`
- [x] Keep runtime/capability gates strict and correct.
- Acceptance:
  - [x] `cargo asm --target riscv64gc-unknown-linux-gnu` shows RVV-driven hot path.
  - [x] no scalar-forwarding for full-chunk path.

## P1: x86 Cliff Removal (`65B`, `16KiB`)

- [ ] Retune `dispatch_tables.rs` boundaries/kernels (especially `64 -> 65` crossover).
- [ ] Re-check AVX-512 small-payload penalties and avoid wrong tiny-size picks.
- [x] Reduce control-flow overhead in single-chunk finalize hot paths.
- Acceptance:
  - worst x86 delta in `65..65536` is within `-8%` vs official on tuned hosts.

## P2: Baseline + Parity Enforcement

- [ ] Capture BLAKE3 baselines for IBM + RISC-V after real kernels land.
- [x] Keep automated parity checks failing on missing baseline files.
- [x] Ensure reproducible evidence artifacts for asm/llvm-lines and benchmark runs.
- Acceptance:
  - CRC + BLAKE3 arch coverage aligned across release targets.

## Commands (Source of Truth)

### x86 tuning/codegen

```bash
cargo asm -p hashes --lib --target x86_64-unknown-linux-gnu 'hashes::crypto::blake3::digest_one_chunk_root_hash_words_x86' --rust
cargo asm -p hashes --lib --target x86_64-unknown-linux-gnu 'hashes::crypto::blake3::single_chunk_output' --rust
cargo asm -p hashes --lib --target x86_64-unknown-linux-gnu 'hashes::crypto::blake3::x86_64::chunk_compress_blocks_sse41' --rust
cargo llvm-lines -p hashes --lib --release --target x86_64-unknown-linux-gnu | rg 'blake3::'
```

### IBM/RISC-V codegen audit

```bash
scripts/bench/blake3-codegen-audit.sh s390x-unknown-linux-gnu
scripts/bench/blake3-codegen-audit.sh powerpc64le-unknown-linux-gnu
scripts/bench/blake3-codegen-audit.sh riscv64gc-unknown-linux-gnu
```

### Runtime profiling

```bash
samply record --save-only -o /tmp/blake3_65.json -- target/release/deps/blake3-<id> 'blake3/oneshot/rscrypto/65'
samply record --save-only -o /tmp/blake3_16k.json -- target/release/deps/blake3-<id> 'blake3/oneshot/rscrypto/16384'
```

## Done Definition

1. Real SIMD/ASM BLAKE3 kernels are live on `s390x`, `powerpc64`, `riscv64` (not wrapper-only scalar forwarding). ✅
2. x86 oneshot cliff is capped at `-8%` worst-case in `65..65536` vs official on tuned hosts.
3. Evidence is checked with asm + llvm-lines + runtime profiles. ✅ (codegen/scripted evidence)
4. Baselines are present and parity enforcement is active for all release arch targets.
