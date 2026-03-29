# v0.1.0 Release Plan

> **Last updated:** 2026-03-29
> **Status:** Pre-release — correctness validated, performance competitive, surface cleanup remaining

---

## Release Readiness Summary

### Correctness: STRONG

No correctness blocker found. The crate passes:

- `just check-all` (host + 12 cross targets including no_std, WASM, Windows, s390x, POWER)
- `just test` (774+ tests: unit, integration, doctest, compile-fail)
- 7 official vector test suites against published standards
- 10 differential test suites against competitor crates
- 14 proptest property-based test suites

### Performance: COMPETITIVE

~69% overall win rate (~1064W / ~379T / ~97L across 7 platforms).

| Category | Win% | Status |
|----------|------|--------|
| Checksums | 85% | Dominant |
| SHA-2 | 74% | Strong (SPR AVX-512 gap remaining) |
| SHA-3 | ~49% | Improved (was 26%, lane-complementing on x86) |
| SHAKE | ~90% | Dominant |
| Blake3 | 49% | Competitive |
| XXH3 | 54% | Competitive |
| RapidHash | 8% | Parity (ties, not losses) |
| Auth | 48% | HMAC strong; Ed25519 correct but 5.9x slower than dalek |

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

### M1. Cargo.toml `include` rules (BLOCKING)

No `include`/`exclude` defined. `cargo publish` will ship internal docs, scripts,
testdata, CI configs, and bench infrastructure.

**Action:** Add `include` list covering `src/`, `Cargo.toml`, `LICENSE-*`, `README.md`,
`testdata/` (needed for official vector tests in dev-deps).

### M2. `#[doc(hidden)] pub` boundary audit (BLOCKING)

`pub` is semver surface whether rustdoc shows it or not. Internal benchmark hooks,
dispatch plumbing, and test-only modules that are `pub` or `#[doc(hidden)] pub` will
become accidental API commitments.

**Action:** Inventory all `pub` items. Convert internal items to `pub(crate)`. Target:
zero `#[doc(hidden)] pub` except intentionally supported advanced APIs.

### M3. Ed25519 positioning (BLOCKING — messaging only)

Ed25519 is correct (RFC 8032 vectors + dalek oracle pass) but 5.9x slower than
ed25519-dalek. This is the single largest credibility risk if the crate is positioned
as "fast crypto."

**Action:** Crate docs and README must be honest:
- Ed25519 is included for correctness and zero-dep convenience
- Not yet optimized for speed (optimization roadmap exists)
- Users needing peak Ed25519 throughput should use dalek until optimization lands

### M4. Verify `cargo package` is clean (BLOCKING)

**Action:** Run `cargo package --list` and verify no sensitive or unnecessary files
are included. Run `cargo publish --dry-run` for final validation.

---

## What Should Be Done Before v0.1.0 (High Value)

### S1. Naming consistency pass

Standardize XOF reader suffix to `*XofReader` everywhere. Current inconsistencies:
- `Blake3Xof` vs `Shake256Xof` vs `AsconXofReader`

Not a correctness issue. Affects perceived quality.

### S2. Remove stale `TODO` comments in src/

Two TODO markers in `src/platform/detect/arch/aarch64.rs` (lines 401, 438) for
Apple M5 CPU detection values not yet public. Cosmetic — mark as known limitation
in a comment rather than TODO.

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

### P2. Ed25519 optimization (8-bit basepoint table, AVX2 field mul)

The 28 Ed25519 losses are real but the algorithm is correct. Optimization is a
separate workstream (documented in `next_steps.md`).

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

### What Still Needs Work (M2)

- `#[doc(hidden)] pub` modules in `checksum/mod.rs` and `hashes/mod.rs` leak semver surface
- Internal benchmark hooks are public for bench crate access
- Target: `pub` = stable API, `pub(crate)` = internal, zero `#[doc(hidden)] pub`

### Naming Law (S1)

- Root aliases: short and opinionated (`Xxh3`, `RapidHash`, `Crc32`)
- Explicit names: algorithm-width (`Xxh3_64`, `RapidHash128`)
- Tuned variants: module-only (`RapidHashFast64` under `hashes::fast`)
- XOF readers: standardize on `*XofReader`

---

## Release Checklist

### Must-do (M)

- [ ] **M1** Add `include` rules to `Cargo.toml`
- [ ] **M2** Audit and fix `#[doc(hidden)] pub` boundary
- [ ] **M3** Update README/crate docs with honest Ed25519 positioning
- [ ] **M4** Verify `cargo package --list` and `cargo publish --dry-run`

### Should-do (S)

- [ ] **S1** Naming consistency pass (XOF reader suffixes)
- [ ] **S2** Clean TODO comments in platform detection
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
> Competitive or faster than established crates on x86-64, aarch64, s390x, and POWER.

Honest caveats:
- Ed25519 is correct but not yet speed-optimized (5.9x gap vs dalek)
- RapidHash is at parity, not faster
- Blake3 is competitive but not universally faster than the official crate
