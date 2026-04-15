# Auth Roadmap

## Current Status (2026-04-12)

Phase 1 auth primitives are **shipped and competitive**:

- `HMAC-SHA256` — 66% win rate (112W/40T/16L across 168 comparisons)
- `HMAC-SHA384` — shipped
- `HMAC-SHA512` — shipped
- `HKDF-SHA256` — 93.8% win rate (30W/1T/1L) after raw-state expand rewrite
- `HKDF-SHA384` — shipped
- `Ed25519` — sign 100% WIN on x86 (IFMA/AVX2), verify competitive
- `X25519` — shipped (portable Montgomery ladder, RFC 7748 coverage, dalek differential tests)
- `KMAC256` / `cSHAKE256` — shipped (SP 800-185)
- `Ascon-CXOF128` — shipped (SP 800-232)

Canonical benchmarks: `benchmark_results/` (organized by date/platform)

## Status

The classical auth surface is now closed before shifting focus to PQ
(ML-KEM, ML-DSA, SLH-DSA — see [`pqe_pqc.md`](pqe_pqc.md)).

### X25519 (Curve25519 ECDH Key Exchange)

**Priority:** Shipped — unblocks hybrid PQ key exchange (X25519 + ML-KEM).

Every modern key agreement protocol depends on X25519: TLS 1.3, SSH,
WireGuard, Signal, Noise framework. Without it, rscrypto cannot participate
in key establishment.

**What ships now:** X25519 uses two paths over the existing Curve25519 5x51
field arithmetic in `src/auth/ed25519/field.rs`:

- `X25519SecretKey::public_key()` uses fixed-base Edwards multiplication with
  Montgomery conversion, inheriting the existing Ed25519 basepoint dispatch
  (`AVX2` / `IFMA` on x86 where available, portable table-driven fallback
  everywhere else).
- `X25519SecretKey::diffie_hellman(&X25519PublicKey)` stays on the dedicated
  RFC 7748 Montgomery ladder for arbitrary peer inputs.

The public API remains typed and byte-oriented:

- `X25519SecretKey`
- `X25519PublicKey`
- `X25519SharedSecret`
- `X25519SecretKey::public_key()`
- `X25519SecretKey::diffie_hellman(&X25519PublicKey)`

Secret material is masked in `Debug`, supports `display_secret()`, and is
zeroized on drop.

**Validation bar met:**

- RFC 7748 scalar-multiplication vectors
- RFC 7748 iterated ladder vectors (1 and 1,000 iterations)
- RFC 7748 Alice/Bob Diffie-Hellman vector
- Differential tests against `x25519-dalek`
- All-zero shared-secret rejection for low-order inputs
- Non-canonical public-input acceptance per RFC 7748
- `no_std` compile coverage in the auth feature matrix

**Performance posture:** fixed-base public-key derivation is now accelerated
through the existing Ed25519 basepoint machinery and is materially faster on
real hosts. The remaining headroom is in variable-base Diffie-Hellman, which
still uses the portable Montgomery ladder today.

**Spec:** RFC 7748 (Elliptic Curves for Security)

## Ship Order

Classical auth/key-agreement primitives are now shipped:

- `HMAC-SHA256` / `HMAC-SHA384` / `HMAC-SHA512`
- `HKDF-SHA256` / `HKDF-SHA384`
- `KMAC256`
- `Ed25519`
- `X25519`

## Acceleration Posture

X25519 currently inherits the following portability posture:

| Primitive | x86_64 | aarch64 | s390x | powerpc64 | riscv64 |
|-----------|--------|---------|-------|-----------|---------|
| X25519 public-key derivation | Ed25519 fixed-base dispatch (`AVX2` / `IFMA` where available) | portable fixed-base table | portable fixed-base table | portable fixed-base table | portable fixed-base table |
| X25519 Diffie-Hellman | portable Montgomery ladder | portable Montgomery ladder | portable Montgomery ladder | portable Montgomery ladder | portable Montgomery ladder |

X25519 now has a fast fixed-base path everywhere and inherits x86 SIMD from
the Ed25519 basepoint engine. Variable-base ECDH acceleration remains a
follow-up optimization track.

## SHA-512 Auth Acceleration Plan

The remaining benchmark-critical auth work is not new primitive design; it is
performance closure for the SHA-512-family wrappers:

- `HMAC-SHA384`
- `HMAC-SHA512`
- `HKDF-SHA384`

These are wrapper surfaces over the existing SHA-512-family dispatch and kernel
stack, not separate SIMD kernels. The performance bar therefore splits into
two parts:

- **Kernel inheritance:** always hit the best SHA-512-family backend already
  available for the current target.
- **Wrapper overhead:** eliminate avoidable keyed-state setup, buffer motion,
  and per-call control-flow cost so auth benches stay close to the raw hash
  lower bound.

### Runner Baseline

The baseline performance lanes are the dedicated runners in
`.github/runs-on.yml`:

- `amd-zen4`
- `amd-zen5`
- `intel-spr`
- `intel-icl`
- `graviton3`
- `graviton4`

Generic CI lanes (`linux-x64-ci`, `linux-arm64-ci`) are compile/test coverage,
not performance truth.

### Required Coverage Checklist

#### Bench Governance

- [ ] `hmac-sha384` is published in the canonical bench workflow output
- [ ] `hmac-sha512` is published in the canonical bench workflow output
- [ ] `hkdf-sha384/expand` is published in the canonical bench workflow output
- [ ] all three appear in `docs/bench/BENCHMARKS.md` for every dedicated runner
- [ ] the canonical report includes per-platform win/tie/loss signal, not just
  local one-off runs

#### Oracle / Differential Validation

- [x] `HMAC-SHA384` vector coverage
- [x] `HMAC-SHA512` vector coverage
- [x] `HKDF-SHA384` vector coverage
- [x] `HMAC-SHA384` differential fuzzing vs RustCrypto
- [x] `HMAC-SHA512` differential fuzzing vs RustCrypto
- [x] `HKDF-SHA384` differential fuzzing vs RustCrypto
- [x] forced portable vs accelerated backend equivalence tests for auth wrappers
- [ ] feature-matrix / `no_std` coverage remains green with the new fuzz/oracle additions

#### Wrapper Overhead Closure

- [ ] `HMAC-SHA384::new()` keyed-state precompute is measured and profiled on short inputs
- [ ] `HMAC-SHA512::new()` keyed-state precompute is measured and profiled on short inputs
- [ ] `HKDF-SHA384::extract()` keyed-state precompute is measured and profiled
- [ ] `HMAC-SHA384::mac()` short-message path is within an acceptable fixed overhead of raw `Sha384`
- [ ] `HMAC-SHA512::mac()` short-message path is within an acceptable fixed overhead of raw `Sha512`
- [ ] `HKDF-SHA384::expand()` minimizes per-block buffer rebuild and copy cost

#### Per-Runner Kernel Policy

- [ ] `amd-zen4`: verify `HMAC-SHA384`, `HMAC-SHA512`, and `HKDF-SHA384` inherit the best measured AVX2-based SHA-512-family path
- [ ] `amd-zen5`: same verification with Zen5-specific dispatch reality
- [ ] `intel-spr`: verify the auth surfaces inherit the best measured AVX-512VL/SHA-512 path
- [ ] `intel-icl`: same verification with Ice Lake-specific dispatch reality
- [ ] `graviton3`: verify all three surfaces are actually riding `aarch64-sha512`
- [ ] `graviton4`: same verification with Graviton4-specific numbers
- [ ] if benchmark governance extends to IBM lanes, add explicit auth benchmark publication for `s390x` and `POWER10`

### Non-Goals

Avoid ornamental work:

- do **not** add auth-specific SIMD kernels that duplicate the SHA-512-family kernels
- do **not** chase generic abstractions over `HMAC-SHA256/384/512` if they cost hot-path clarity
- do **not** claim “world-class” on a platform that lacks dedicated benchmark truth

## After These Three

The classical auth surface is closed. Remaining auth-adjacent work:

- **Argon2id** — deferred unless password hashing enters scope (requires BLAKE2b
  or a new BLAKE3-based design; see conversation notes)
- **AEGIS-128L** — AEAD, not auth; optional companion to AEGIS-256
- **X25519 fixed-base acceleration** — optimize public-key derivation after the
  portable baseline is benchmarked across the canonical matrix
- **Ed25519 verify optimization** — Phase 9 (IFMA double reduce elimination);
  incremental, not blocking

Focus shifts to PQ: [`pqe_pqc.md`](pqe_pqc.md) (ML-KEM, ML-DSA, SLH-DSA).

## Sources

- RFC 7748: Elliptic Curves for Security (X25519, X448)
  https://datatracker.ietf.org/doc/html/rfc7748
- RFC 2104: HMAC
  https://datatracker.ietf.org/doc/html/rfc2104
- NIST FIPS 198-1: HMAC
  https://csrc.nist.gov/pubs/fips/198-1/final
- RFC 5869: HKDF
  https://datatracker.ietf.org/doc/html/rfc5869
- RFC 8446: TLS 1.3 (cipher suite requirements)
  https://datatracker.ietf.org/doc/html/rfc8446
- NIST SP 800-224 IPD: updated HMAC guidance
  https://csrc.nist.gov/pubs/sp/800/224/ipd
