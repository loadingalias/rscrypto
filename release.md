# rscrypto v0.1.0 Release Tracker

Status: **In Progress**

---

## Phase 1: Security & Correctness Fixes — DONE

All changes applied in the existing multi-crate workspace. Compiles and tests pass.

### Step 1.1: Fix bare arithmetic → `strict_*` — DONE

Replaced `+=`, `-`, `+` with `strict_add`, `strict_sub` on counters/lengths/indices
in hash update paths and Blake3 parallelism, per CLAUDE.md.

| File | Changes |
|------|---------|
| `crates/hashes/src/crypto/sha256/mod.rs` | `block_len += take` → `strict_add`; slice index math; `full_len` subtraction; finalize `block_len += 1` |
| `crates/hashes/src/crypto/sha224.rs` | Same pattern |
| `crates/hashes/src/crypto/sha384.rs` | Same pattern |
| `crates/hashes/src/crypto/sha512/mod.rs` | Same pattern |
| `crates/hashes/src/crypto/sha512_256.rs` | Same pattern |
| `crates/hashes/src/crypto/keccak.rs` | `buf_len += take`; `pos += take` in squeeze |
| `crates/hashes/src/crypto/ascon.rs` | `buf_len += take`; `pos += take` in XOF squeeze |
| `crates/hashes/src/crypto/blake3/mod.rs` | `thread_range()`: bare `*`, `+`, `/` → `strict_mul`, `strict_add`, `strict_div` |

### Step 1.2: Targeted safety fixes — DONE

| Sub-step | File | Change |
|----------|------|--------|
| 1.2a | `blake3/mod.rs` | `SendPtr` SAFETY comments strengthened |
| 1.2b | `platform/src/detect/cache_override.rs` | `Slot<T>: Sync` → `Slot<T: Sync>: Sync` |
| 1.2c | `hashes/src/fast/xxh3.rs` | `read_u32_le`/`read_u64_le` → `unsafe fn` with `# Safety` docs; all call sites wrapped in `unsafe {}` with SAFETY comments |

### Step 1.3: Add `constant_time_eq` + `zeroize` — DONE

- New file: `crates/traits/src/ct.rs` with `constant_time_eq()` and `zeroize()`
- Added `pub mod ct;` to `crates/traits/src/lib.rs`
- Updated `VerificationError` doc example to use `constant_time_eq`

### Step 1.4: Zeroize on Drop — DONE

Added `impl Drop` with volatile zeroing for:
- `Sha256`, `Sha224`, `Sha384`, `Sha512`, `Sha512_256` — state words + block buffer
- `KeccakCoreImpl<RATE, P>`, `KeccakXofImpl<RATE, P>` — state + buf
- `Sponge` (Ascon), `AsconXof128Xof` — state + buf
- `Blake3` — `key_words`, `chaining_value`, `block`, `cv_stack`, `pending_chunk_cv`

### Step 1.5: Clear Blake3 keyed hash thread_local cache — DONE

In `Blake3::drop()`, when `flags & KEYED_HASH != 0`, zeroizes and clears
`KEYED_WORDS_LOCAL_CACHE` thread_local (gated on `#[cfg(feature = "std")]`).

### Step 1.6: `Digest::Output` — add `Default` bound — SKIPPED

`[u8; N]` for N > 32 does not implement `Default` on current stable/nightly Rust.
`[u8; 48]` (SHA-384) and `[u8; 64]` (SHA-512) would fail. Skipped to avoid
requiring a nightly feature.

---

## Phase 2: API Surface Lockdown — DONE

### Step 2.1: `pub unsafe fn` → `pub(crate) unsafe fn` — DONE

141 `pub unsafe fn` entries across 13 SIMD backend files changed to `pub(crate) unsafe fn`.
Also changed `pub use` re-exports in prefetch modules to `pub(crate) use`.

Files changed:
- `crates/hashes/src/common/prefetch.rs`
- `crates/checksum/src/common/prefetch.rs`
- `crates/checksum/src/crc64/{aarch64,power,x86_64,riscv64,s390x}.rs`
- `crates/checksum/src/crc32/x86_64.rs`
- `crates/hashes/src/crypto/blake3/{aarch64,x86_64,x86_64/avx2,x86_64/sse41,x86_64/avx512}.rs`

### Step 2.2: Tighten module visibility — DONE

| Crate | Module | Change |
|-------|--------|--------|
| `checksum` | `bench` | `#[doc(hidden)]` added |
| `checksum` | `diag` | `#[doc(hidden)]` added |
| `checksum` | `dispatch` | `#[doc(hidden)]` added |
| `checksum` | `dispatchers` | `#[doc(hidden)]` added |
| `hashes` | `common` | `#[doc(hidden)]` added |
| `hashes` | `bench` | `#[doc(hidden)]` added |

Kept `pub` (not `pub(crate)`) because fuzz targets and integration tests access
these modules as external crates. `#[doc(hidden)]` hides them from public docs.

### Step 2.3: `#[non_exhaustive]` on public enums — DONE

Added `#[non_exhaustive]` to 19 enums/structs:
- `backend`: `KernelTier`, `SelectionError`, `KernelSubfamily`
- `checksum`: `Crc16Force`, `Crc24Force`, `Crc32Force`, `Crc64Force`, `SelectionReason`, `Crc32Polynomial`, `Crc64Polynomial`
- `hashes`: `Blake3KernelId`, `Sha256KernelId`, `Sha224KernelId`, `Sha384KernelId`, `Sha512KernelId`, `Sha512_256KernelId`, `Keccakf1600KernelId`, `AsconPermute12KernelId`, `RapidHashKernelId`

No cross-crate exhaustive matches found — no wildcard arms needed.

---

## Phase 3: Flatten to Single Crate — IN PROGRESS

### Overview

Flatten 6 workspace crates into a single publishable `rscrypto` crate at the
**repo root** (`src/`, not `crates/rscrypto/src/`). This matches the target
structure in `docs/tasks/rscrypto_publish.md`.

```
rscrypto/                          ← repo root = the crate
├── Cargo.toml                     ← workspace + [package] merged
├── src/
│   ├── lib.rs
│   ├── traits/
│   ├── platform/
│   ├── backend/
│   ├── checksum/  (feature = "checksums")
│   └── hashes/    (feature = "hashes")
├── tests/
├── benches/
├── examples/
└── testdata/
```

### Step 3.1: Copy source trees into staging location — DONE

Source trees copied into `crates/rscrypto/src/` as a staging area.
All 5 sub-crate source trees now live under `crates/rscrypto/src/{traits,platform,backend,checksum,hashes}/`.

### Step 3.2: Write new root `lib.rs` — DONE

Merged crate-level attributes from all 5 sub-crates. Feature-gated `checksum`
and `hashes` modules. Re-exports the public API at root.

### Step 3.3: Fix imports — DONE

Rewrote ~400 import paths across the flattened source tree:
- Removed `#![no_std]`, `extern crate alloc/std`, redundant crate-level lints/features from 5 mod.rs files
- `use crate::` in sub-modules → `use crate::{module}::` via sed
- `use platform/traits/backend::` → `use crate::platform/traits/backend::` (both `use` and inline)
- Fixed `dispatch_caps()` references in checksum sub-files
- Fixed `include_str!` relative path in platform/target_matrix.rs
- Fixed doc test imports to use `rscrypto::` prefix (58 doc tests pass)
- Merged duplicate `portable_simd` feature gate in lib.rs
- Added `#[allow(unreachable_patterns)]` for `#[non_exhaustive]` wildcard arms

### Step 3.4: Fix macro `$crate::` paths — DONE

| Macro | File | Change |
|-------|------|--------|
| `candidates!` | `backend/dispatch.rs` | `$crate::dispatch::Candidate` → `$crate::backend::dispatch::Candidate` |
| `define_dispatcher!` | `backend/dispatch.rs` | `$crate::cache::OnceCache` → `$crate::backend::cache::OnceCache`; `$crate::dispatch::Selected` → `$crate::backend::dispatch::Selected` |
| `define_buffered_crc!` | `checksum/macros.rs` | `$crate::Checksum` resolves correctly via root re-export (no change needed) |
| `define_crc_dispatch!` | `checksum/common/kernels.rs` | `$crate::__internal::` → `$crate::checksum::__internal::` |
| `crc_test_suite!` | `checksum/common/tests.rs` | `$crate::common::tests::` → `$crate::checksum::common::tests::` |

### Step 3.5: Update Cargo.toml — DONE

Consolidated all dependencies into `crates/rscrypto/Cargo.toml`:
- `rayon` as optional dep for `parallel` feature
- All dev-dependencies from checksum + hashes (criterion, proptest, oracle crates)
- Features: `default = ["std", "checksums", "hashes"]`, `parallel`, `diag`, `testing`

### Step 3.6: Decouple rayon from `std` — DONE

In `crates/rscrypto/src/hashes/crypto/blake3/mod.rs`:
- `RayonJoin` enum + impl: gated on `#[cfg(feature = "parallel")]`
- 3 `_parallel_rayon` functions: gated on `#[cfg(feature = "parallel")]`
- `root_output_oneshot_join_parallel`: gated on `#[cfg(feature = "parallel")]`
- `rayon::current_num_threads()` in `available_parallelism_cached`: conditional on `parallel`
- `hash_full_chunks_cvs_scoped`: serial fallback when `parallel` is off
- `hash_power_of_two_subtree_roots` call site: dispatches to serial when `parallel` is off
- `reduce_power_of_two_chunk_cvs_any`: `fold_level!` macro for cfg-gated dispatch
- Oneshot join path: `#[cfg(feature = "parallel")]` instead of `#[cfg(feature = "std")]`
- Test call sites: gated on `#[cfg(feature = "parallel")]`

### Step 3.7: Move source from staging to repo root — DONE

Moved `crates/rscrypto/src/*` → `src/` at the repo root.

- Updated `include_str!` path in `src/platform/target_matrix.rs`
  (from `"../../../../config/target-matrix.toml"` to `"../../config/target-matrix.toml"`)
- Merged `crates/rscrypto/Cargo.toml` `[package]` + `[features]` + `[dependencies]`
  into the root `Cargo.toml` (workspace + package in one file)
- Removed `crates/rscrypto/` entirely

### Step 3.8: Move tests, benches, examples, testdata — DONE

All integration tests, benches, examples, and testdata at repo root.
Fixed all imports:
- `platform::` → `rscrypto::platform::` in bench utils
- `crypto::` → `hashes::crypto::` in bench/test files
- `fast::` → `hashes::fast::` in bench files
- `bench` → `checksum::bench` / `hashes::bench` in kernel bench files
- `dispatch` → `checksum::dispatch` in vectored dispatch test

638 tests pass (439 lib + 141 integration + 58 doc). All cross-platform checks pass.

### Step 3.9: Wire Blake3 prefetch into hot loops — DONE

Wired `prefetch_read_l1` into the aarch64 NEON `hash_many_contiguous_neon` hot loop
in `src/hashes/crypto/blake3/aarch64.rs`. Prefetches the next 4/8-chunk batch at
the top of each loop iteration.

**Benchmark results (Apple Silicon, criterion):**

| Size | Before | After | Change |
|------|--------|-------|--------|
| 16384 (16 chunks) | 9.113 µs | 9.065 µs | **-0.5%** (p=0.01) |
| 1048576 (1024 chunks) | 597.0 µs | 590.2 µs | **-1.1%** (p=0.00) |

Improvement is modest but statistically significant on large inputs. Apple Silicon
has best-in-class hardware prefetchers; improvement likely larger on Graviton/x86.

**Prefetch module cleanup:**
- Trimmed `hashes/common/prefetch.rs` to only `prefetch_read_l1` on aarch64
- Removed unused variants: `prefetch_read_l2`, `prefetch_read_nta/stream`,
  `prefetch_hash_block`, `prefetch_next_chunk`, and all unused constants
- Removed no-op fallbacks for architectures with no callers
- Removed `#[allow(dead_code)]` on the module declaration — no dead code remains
- x86_64 implementation deferred until an x86_64 kernel is wired up and benchmarked

### Step 3.10: Update workspace, justfile, scripts — DONE

- Root `Cargo.toml`: workspace `members = ["."]`, removed old crate path deps
- `justfile`: updated bench commands: `-p rscrypto --bench checksum_comp` etc.,
  `crates=rscrypto` instead of `crates=hashes`
- `scripts/check/check-all.sh`: constrained crates list → `["rscrypto"]`,
  `crate_supports_alloc` handles root Cargo.toml
- `scripts/check/check-win.sh`: xwin init uses `-p rscrypto --no-default-features`
- `scripts/test/test-miri.sh`: `MIRI_CRATES="rscrypto"`
- `scripts/gen_blake3_x86_asm_ports.py`: asm path → `src/hashes/...`
- `scripts/gen_hashes_testdata.py`: testdata path → `testdata/`
- Bench runner (`scripts/ci/run-bench.sh`) needs further updates — deferred
  (complex 530-line script, not blocking)

### Step 3.11: Remove old crate directories — DONE

```
rm -rf crates/
```

No `crates/` directory remains. The repo root IS the crate.

### Step 3.12: Clean up dead code annotations — DONE

Audited `#[allow(dead_code)]` in `src/`. Remaining instances are legitimate:

| Category | Files | Reason |
|----------|-------|--------|
| Kernel dispatch tables | `sha*/kernels.rs`, `keccak/dispatch.rs`, `xxh3/kernels.rs`, `rapidhash/kernels.rs` | Compiler can't see through fn ptr dispatch |
| SIMD helpers | `blake3/aarch64.rs`, `blake3/x86_64.rs` | Architecture-conditional, called from asm |
| Kernel registry | `crc16/kernels.rs`, `crc24/kernels.rs` | Used by bench + policy dispatch |
| Platform constants | `aarch64.rs` | Upcoming Apple hardware constants |
| Checksum prefetch | `checksum/common/prefetch.rs` | Same pattern as hashes (unused l2/nta/stream variants) — pre-existing, clean up as follow-up |
| Test harness | `common/tests.rs` | Test-only struct |

No `#[allow(dead_code)]` was introduced by the flatten. The hashes prefetch dead code
(the original concern) was fixed in Step 3.9.

---

## Post-Flatten Verification — DONE

### Compilation Matrix

| Configuration | Result |
|---------------|--------|
| `--all-features` | Clean |
| `--no-default-features` (no_std) | Clean |
| `--no-default-features --features checksums` | 36 warnings (bench/test-only kernel items, pre-existing) |
| `--no-default-features --features hashes` | Clean |
| `--features "hashes,parallel"` | Clean |

### Correctness

| Check | Result |
|-------|--------|
| `cargo test --all-features` | 522 tests pass (437 lib + 85 integration) |
| `cargo test --doc --all-features` | 58 doc tests pass |

### Quality

| Check | Result |
|-------|--------|
| `cargo clippy --all-features --all-targets` | Zero warnings |
| `cargo doc --all-features --no-deps` | Zero warnings |

### Dependency Verification

| Check | Result |
|-------|--------|
| `cargo tree -e normal` (default features) | Zero runtime dependencies |
| `cargo tree -e normal --features parallel` | Only rayon tree |

### Publish Readiness

| Check | Result |
|-------|--------|
| `cargo publish --dry-run` | Zero code warnings, package 289 files / 6.5 MiB (1.3 MiB compressed) |

### Fix: Blake3 parallel cfg gating

During verification, `cargo publish --dry-run` revealed 17 warnings in Blake3: parallel-only
code was gated on `#[cfg(feature = "std")]` instead of `#[cfg(feature = "parallel")]`. Since
`parallel` implies `std` in `Cargo.toml`, changing the gate to `#[cfg(feature = "parallel")]`
is correct and eliminates all dead code when `parallel` is off.

Files changed:
- `src/hashes/crypto/blake3/mod.rs` — ~25 items re-gated
- `src/hashes/crypto/blake3/dispatch.rs` — ~14 items re-gated

### Remaining: checksums-only warnings

36 warnings appear in `--no-default-features --features checksums` build. These are
bench/test-only items (kernel name arrays, slice4 portable functions, SVE2 kernel
arrays) that have callers only in bench/test code which isn't compiled by `cargo build`.
These are pre-existing and do not affect the default or publish build. Fix as follow-up
with `#[cfg(any(test, feature = "testing"))]` gating.

---

## Notes

- Step 1.6 (`Default` bound) was skipped because `[u8; N]` for N>32 lacks `Default`
  in current Rust. Revisit when the std blanket impl ships.
