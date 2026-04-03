# v0.1.0 Release Plan

> **Last updated:** 2026-04-03
> **Status:** Pre-release — correctness validated, packaging verified, non-loss gate passed, pure-win gate approaching

---

## Release Readiness Summary

### Correctness: STRONG

No correctness blocker found. The crate passes:

- `just check-all` (host + 12 cross targets including no_std, WASM, Windows, s390x, POWER)
- `just test` (816+ tests: unit, integration, doctest, compile-fail)
- 7 official vector test suites against published standards
- 10 differential test suites against competitor crates
- 14 proptest property-based test suites

### Performance: COMPETITIVE, BELOW PUBLIC-RELEASE BAR

Canonical benchmark source: [`docs/bench/BENCHMARKS.md`](bench/BENCHMARKS.md),
CI run `#23822408700` on 2026-03-31.

Overall scoreboard: `1300W / 445T / 217L` across `1962` comparisons (updated 2026-04-03 after POLY-4).

Release gates:

1. non-loss rate `((W + T) / total)` = `1745 / 1962 = 88.94%` — **passes** the `80%` gate
2. pure win rate `(W / total)` = `1300 / 1962 = 66.26%` — **approaching** the `70%` public-release bar (was 64.78%)

| Category | Win% | Status |
|----------|------|--------|
| Checksums | 85% | Dominant |
| SHA-2 | 59% | Competitive, but weak on Grav3/Grav4/POWER10 |
| SHA-3 | 58% | Competitive |
| SHAKE | 87% | Strong |
| Blake3 | 49% | Competitive |
| XXH3 | 54% | Competitive |
| RapidHash | 6% | Mostly parity, not wins |
| Auth | 57% | HMAC strong; Ed25519 sign wins, verify lags |
| AEAD | 64% | Competitive on x86_64/aarch64; fallback platforms still drag the total |

**Acceleration gap tracker:** [`docs/tasks/acceleration.md`](acceleration.md)

Key gaps still holding the pure-win gate below `70%`:
- SHA-2 on Grav3/Grav4/POWER10
- Ed25519 verification throughput, even though Ed25519 signing now wins
- AES-family AEAD on s390x and POWER10, which still falls back to portable code
- Large-message ChaCha-family AEAD on x86_64 at `4KiB+`

### Security: SOLID FOUNDATIONS

| Defense | Status | Details |
|---------|--------|---------|
| Constant-time verification | Implemented | `ct::constant_time_eq` at all MAC/signature verify sites |
| Zeroize on Drop | Implemented | 30+ sites: hash state, key material, expanded secrets |
| Volatile writes + fence | Implemented | `write_volatile` + `compiler_fence(SeqCst)` |
| Opaque verification errors | Implemented | `VerificationError` is `Copy`, no message, no timing leak |
| Unsafe documentation | Complete | All `unsafe` blocks have `// SAFETY:` comments |
| Unsafe scope | Narrow | Concentrated in SIMD intrinsics and volatile writes |

---

## Test Coverage Inventory

### Official Vector Tests (7 suites)

| File | Algorithm | Standard |
|------|-----------|----------|
| `tests/sha2_official_vectors.rs` | SHA-224/256/384/512/512-256 | NIST FIPS 180-4 |
| `tests/sha3_official_vectors.rs` | SHA3-224/256/384/512, SHAKE128/256 | NIST FIPS 202 |
| `tests/blake3_official_vectors.rs` | Blake3 (hash, keyed, derive, XOF) | BLAKE3 spec |
| `tests/ascon_official_vectors.rs` | AsconHash256, AsconXof128 | ASCON v1.2 |
| `tests/ed25519_rfc8032_vectors.rs` | Ed25519 sign/verify | RFC 8032 |
| `tests/hmac_sha256_vectors.rs` | HMAC-SHA256 | RFC 4231 |
| `tests/hkdf_sha256_vectors.rs` | HKDF-SHA256 | RFC 5869 |

### Differential Tests (10 suites)

| File | Algorithm | Competitor Crate |
|------|-----------|-----------------|
| `tests/blake3_differential.rs` | Blake3 | `blake3 v1.8.3` |
| `tests/sha256_differential.rs` | SHA-256, SHA-224 | `sha2 v0.10.9` |
| `tests/sha512_differential.rs` | SHA-512, SHA-384, SHA-512/256 | `sha2 v0.10.9` |
| `tests/sha3_differential.rs` | SHA3-224/256/384/512 | `sha3 v0.10.8` |
| `tests/shake128_differential.rs` | SHAKE128 | `sha3 v0.10.8` |
| `tests/shake256_differential.rs` | SHAKE256 | `sha3 v0.10.8` |
| `tests/xxh3_differential.rs` | XXH3-64/128 | `xxhash-rust v0.8.15` |
| `tests/rapidhash_differential.rs` | RapidHash64/128 | `rapidhash v4.4.1` |
| `tests/ascon_differential.rs` | AsconHash256, AsconXof | Internal reference |
| `tests/ed25519_oracle.rs` | Ed25519 | `ed25519-dalek v2.1.1` |

### Property Tests (14 suites)

All proptest-based with randomized inputs (0-8192B), variable chunk sizes,
and streaming-vs-oneshot equivalence:

- `tests/crc16_properties.rs` — CRC-16 CCITT/IBM vs crc-fast + portable ref
- `tests/crc24_properties.rs` — CRC-24 vs crc crate + bitwise ref
- `tests/crc32_properties.rs` — CRC-32/32C vs crc-fast + portable ref
- `tests/crc64_properties.rs` — CRC-64 XZ/NVME vs crc64fast + portable ref
- `tests/common_properties.rs` — CRC combine correctness, chunking equivalence
- `tests/blake3_differential.rs` — proptest: oneshot vs streaming (0-4096B)
- `tests/sha256_differential.rs` — proptest: oneshot vs streaming vs oracle
- `tests/sha512_differential.rs` — proptest: oneshot vs streaming vs oracle
- `tests/sha3_differential.rs` — proptest: oneshot vs streaming
- `tests/shake128_differential.rs` — proptest: variable output 0-2048B
- `tests/shake256_differential.rs` — proptest: variable output 0-2048B
- `tests/xxh3_differential.rs` — proptest: random seeds, 0-4096B
- `tests/rapidhash_differential.rs` — proptest: random seeds, 0-4096B
- `tests/ascon_differential.rs` — proptest: hash + XOF, 0-8192B input, 0-2048B output

### Structural Tests

- `tests/root_surface.rs` — compile-time root export verification
- `tests/api_consistency.rs` — trait contract (new/update/finalize/reset) for all algorithms
- `tests/portable_fallback.rs` — portable CRC implementations reachable on all platforms
- `tests/vectored_dispatch.rs` — multi-buffer API matches one-shot

---

## What Must Be Done Before v0.1.0

### M1. Cargo.toml `include` rules ✅

Resolved on 2026-03-31.

- `Cargo.toml` now carries an explicit `include` list
- publishable inputs are intentionally scoped to crate sources, licenses,
  `README.md`, and `testdata/` needed by dev-only vector tests

### M2. `#[doc(hidden)] pub` boundary audit ✅

Resolved on 2026-03-31.

- test-only checksum internals no longer leak through `checksum::__internal`
- `traits::io::SealedMarker` is crate-private
- Ascon kernel-selecting batch helpers are crate-private, while the supported
  batched public APIs (`digest_many`, `hash_many_into`) are documented public
- target met: zero remaining `#[doc(hidden)] pub` items in `src/`

### M3. Benchmark-based positioning ✅

Resolved on 2026-03-31.

- README and crate docs now anchor performance claims to the canonical report in
  [`docs/bench/BENCHMARKS.md`](bench/BENCHMARKS.md)
- release gate 1 passes: `84.97%` non-loss rate
- release gate 2 misses: `64.78%` pure win rate vs the `70%` public-release bar
- Ed25519 messaging now reflects the real split:
  - `ed25519-sign`: `27W / 1T / 0L`
  - `ed25519-verify`: `3W / 4T / 21L`
- the old blanket "behind dalek" claim is deleted

### M4. Verify `cargo package` is clean ✅

Resolved on 2026-03-31.

- `cargo package --list --allow-dirty` completed successfully
- `cargo publish --dry-run --allow-dirty` completed successfully
- expected warning: benchmark `benches/xxh3.rs` is not shipped in the published
  package

---

## What Should Be Done Before v0.1.0 (High Value)

### S1. Naming consistency pass ✅

Resolved on 2026-03-31.

- root and module XOF readers now use the `*XofReader` convention:
  `Blake3XofReader`, `Shake128XofReader`, `Shake256XofReader`,
  `Cshake256XofReader`, `AsconXofReader`
- the remaining spec alias now follows the same reader naming law:
  `AsconXof128Reader`

### S2. Remove stale `TODO` comments in src/ ✅

Resolved on 2026-04-01.

- Added `CPUFAMILY_ARM_HIDRA` (`0x1d5a_87e8`) and `CPUFAMILY_ARM_SOTRA` (`0xf76c_5b1a`)
  for M5/M5 Pro/Max detection via Xcode SDK `mach/machine.h`
- Added `CPUFAMILY_ARM_TILOS` and `CPUFAMILY_ARM_THERA` for A19/A19 Pro (≈ M5 architecture)
- `AppleSiliconGen::M5` is now reachable — removed `#[allow(dead_code)]`
- Zero TODO comments remain in `src/platform/`

### S3. Security skill audit

Run the `/sec` security audit skill against the crate. The foundations are solid
(constant-time, zeroize, opaque errors), but a systematic audit may catch:
- Timing variability in Ed25519 scalar operations
- Missing zeroize sites in less-obvious paths
- Unsafe scope that could be narrowed further

---

## What Can Wait for v0.1.x

### P1. Unified policy module

The current per-algorithm config/force system works. A unified `rscrypto::policy`
module would be cleaner but is not needed for initial release.

### P2. Ed25519 field arithmetic optimization (AVX2/NEON)

Ed25519 sign/keygen already **beats dalek by 1.3-2x** (basepoint table + dedicated
doubling + Straus verify are all shipped). Small-message verify is ~12% behind dalek,
large-message verify wins. The remaining gap is in portable 5×51 field arithmetic —
AVX2 `vpmuludq` or NEON vectorized field mul would close it. See ED25519-4/6 in
[`acceleration.md`](acceleration.md).

### P3. aarch64 SHA-3 improvement

SHA-3 ties on Graviton (~72T) are a performance opportunity, not a correctness
issue. Approaches: EOR3 theta, zeroization opt-out for non-secret contexts.

### P4. Full 8-platform benchmark baseline

Run a complete benchmark suite (all algorithms, all platforms) and archive as the
v0.1.0 baseline for future comparison.

---

## Public API Audit

### What Is Strong

- Root exports are curated: `Checksum`, `Digest`, `Mac`, `Xof`, `FastHash` + algorithm types
- Core traits share one mental model: `new` → `update` → `finalize` → `reset`
- Advanced surfaces are explicit: `checksum::config`, `checksum::introspect`, `hashes::introspect`, `hashes::fast`, `platform`
- Root API is guarded by compile-fail tests (`tests/root_surface.rs`)

### M2 Resolution

- hidden-public checksum test hooks were removed from the public crate surface
- hidden-public Ascon batch hooks were split cleanly into crate-private kernel
  selectors and documented public batched APIs
- `pub` now means supported surface here; internal glue stays `pub(crate)`

### Naming Law (S1)

- Root aliases: short and opinionated (`Xxh3`, `RapidHash`, `Crc32`)
- Explicit names: algorithm-width (`Xxh3_64`, `RapidHash128`)
- Tuned variants: module-only (`RapidHashFast64` under `hashes::fast`)
- XOF readers: standardize on `*XofReader`

---

## Release Checklist

### Must-do (M)

- [x] **M1** Add `include` rules to `Cargo.toml`
- [x] **M2** Audit and fix `#[doc(hidden)] pub` boundary
- [x] **M3** Update README/crate docs with honest Ed25519 positioning
- [x] **M4** Verify `cargo package --list` and `cargo publish --dry-run`

### Should-do (S)

- [x] **S1** Naming consistency pass (XOF reader suffixes)
- [x] **S2** Clean TODO comments in platform detection
- [ ] **S3** Run `/sec` security audit skill

### Release verification

```bash
just check-all
just test
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
cargo doc --no-deps --all-features
cargo package
cargo publish --dry-run
```

### Tag and publish

```bash
git tag v0.1.0
git push origin main --tags
cargo publish
```

---

## Positioning

Supportable claim for v0.1.0:

> Pure Rust checksums, digests, XOFs, MACs, HKDF, Ed25519, and fast hashes.
> Zero dependencies by default. `no_std` first. Hardware-accelerated where it helps.
> Broadly competitive and often faster across the current benchmark sweep, but not
> yet over this project's `70%` pure-win public-release bar.

Honest caveats:
- Current benchmark gates are `84.97%` non-loss and `64.78%` pure wins
- Ed25519 is split: sign is strong, verify is still a drag on the total auth score
- RapidHash is at parity, not faster
- Blake3 is competitive but not universally faster than the official crate
