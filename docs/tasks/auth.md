# Auth Roadmap

## Current Status (2026-04-12)

Phase 1 auth primitives are **shipped and competitive**:

- `HMAC-SHA256` — 66% win rate (112W/40T/16L across 168 comparisons)
- `HKDF-SHA256` — 93.8% win rate (30W/1T/1L) after raw-state expand rewrite
- `Ed25519` — sign 100% WIN on x86 (IFMA/AVX2), verify competitive
- `KMAC256` / `cSHAKE256` — shipped (SP 800-185)
- `Ascon-CXOF128` — shipped (SP 800-232)

Canonical benchmarks: [`docs/bench/BENCHMARKS.md`](../bench/BENCHMARKS.md)

## Remaining Gaps

Three primitives are needed to close the classical auth surface before
shifting focus to PQ (ML-KEM, ML-DSA, SLH-DSA — see [`pqe_pqc.md`](pqe_pqc.md)).

### 1. X25519 (Curve25519 ECDH Key Exchange)

**Priority:** Critical — blocks hybrid PQ key exchange (X25519 + ML-KEM).

Every modern key agreement protocol depends on X25519: TLS 1.3, SSH,
WireGuard, Signal, Noise framework. Without it, rscrypto cannot participate
in key establishment.

**What exists:** Ed25519 is shipped with full Curve25519 field arithmetic
(`src/auth/ed25519/field.rs`, `point.rs`), including AVX2 and IFMA vectorized
backends. The field code is shared between Edwards (signing) and Montgomery
(key exchange) operations.

**What to build:**

- `X25519SecretKey` — 32-byte clamped scalar
- `X25519PublicKey` — 32-byte Montgomery u-coordinate
- `X25519SharedSecret` — 32-byte Diffie-Hellman output
- `X25519PublicKey::from(secret)` — basepoint scalar multiply
- `X25519SharedSecret::diffie_hellman(secret, public)` — variable-base scalar multiply
- RFC 7748 Montgomery ladder (constant-time, no Edwards conversion needed)
- Zeroize secret key and shared secret on Drop

**Implementation path:**

The Montgomery ladder operates in the (u, u') representation using only
field arithmetic (add, sub, mul, square, invert). No point decompression,
no extended coordinates. This is ~150-200 lines of new code on existing
field infrastructure.

For SIMD: the Montgomery ladder has a different parallelism shape than
Edwards (it's a 2-way differential chain, not 4-way coordinate parallel).
Start with the portable 5x51 field and evaluate AVX2/IFMA vectorization
after the portable path is benchmarked.

**Spec:** RFC 7748 (Elliptic Curves for Security)

**Test bar:**

- RFC 7748 Section 6.1 test vectors (all 3: basepoint, iterated, Alice/Bob)
- Differential test against `x25519-dalek`
- All-zero / low-order point rejection
- Clamping correctness
- `no_std` compile coverage

### 2. HMAC-SHA384 / HMAC-SHA512

**Priority:** High — needed for TLS 1.3 `TLS_AES_256_GCM_SHA384` suite.

TLS 1.3 mandates two cipher suites:

| Suite | Hash | HMAC | HKDF | Status |
|-------|------|------|------|--------|
| `TLS_AES_128_GCM_SHA256` | SHA-256 | HMAC-SHA256 | HKDF-SHA256 | shipped |
| `TLS_AES_256_GCM_SHA384` | SHA-384 | HMAC-SHA384 | HKDF-SHA384 | **missing** |

SSH, IPsec, and JOSE also use HMAC-SHA384 and HMAC-SHA512.

**What exists:** SHA-384 and SHA-512 are shipped with hardware acceleration
(SHA-NI on x86, SHA2-CE on aarch64, KIMD on s390x, Zknh on RISC-V).
`HmacSha256` is shipped with a battle-tested oneshot `mac()` path that
bypasses streaming `Sha256` construction.

**What to build:**

- `HmacSha384` — 48-byte tag, 128-byte block size
- `HmacSha512` — 64-byte tag, 128-byte block size
- Both implement the `Mac` trait (new/update/finalize/reset/verify)
- Oneshot `mac(key, data)` path following the SHA-256 pattern (raw state
  arrays, single dispatch resolve, no intermediate struct Drop)
- Inherent `mac()` and `verify_tag()` methods on each type

**Implementation path:**

The HMAC construction is identical across hash sizes — only block size,
tag size, and the underlying compress function change. Two approaches:

1. **Concrete types** (preferred): `HmacSha384` and `HmacSha512` as
   standalone types, each with their own oneshot `mac()` using raw
   SHA-384/SHA-512 state arrays. Follows the existing `HmacSha256` pattern.
   More code but zero abstraction overhead.

2. **Generic HMAC<H>**: Parameterized over a hash. Cleaner but risks
   adding a trait bound that constrains the oneshot optimization path.
   Evaluate only if the concrete types show unacceptable duplication.

SHA-384 and SHA-512 share the same compression function (SHA-512 core with
different IVs and output truncation). `HmacSha384` and `HmacSha512` can
share the same raw oneshot implementation with const-generic block/tag sizes.

**Test bar:**

- RFC 4231 test vectors (cases 1-7 for both SHA-384 and SHA-512)
- Oneshot matches streaming equivalence (boundary-critical data lengths)
- Constant-time verification
- `no_std` compile coverage

### 3. HKDF-SHA384

**Priority:** High — needed for TLS 1.3 `TLS_AES_256_GCM_SHA384` suite.

**What exists:** `HkdfSha256` is shipped with a raw-state expand loop that
bypasses streaming `Sha256` entirely — direct `[u32; 8]` state arrays with
cached ipad/opad compressions and a single dispatch-resolved compress function.

**What to build:**

- `HkdfSha384` — extract + expand using SHA-384 (128-byte blocks, 48-byte output)
- Same raw-state expand pattern as `HkdfSha256` (using `[u64; 8]` state arrays
  for the SHA-512 family)
- `new(salt, ikm)`, `extract()`, `expand(info, okm)`, `expand_array::<N>()`
- `derive()` and `derive_array::<N>()` oneshot helpers

**Implementation path:**

Follow the `HkdfSha256` raw-state pattern directly. The SHA-512 family uses
64-byte state (`[u64; 8]`) and 128-byte blocks, so the ipad/opad blocks and
padding arithmetic change, but the structure is identical. The outer hash
block template changes to: inner_hash(48B) + 0x80 + zeros + length(192 bytes
= 1536 bits as u128).

Note: SHA-512 uses a 128-bit bit-length field in padding (vs SHA-256's 64-bit).
The expand helpers need to account for this.

**Test bar:**

- RFC 5869 test vectors (cases 1-3, SHA-384 variant — note: RFC 5869 only
  specifies SHA-256 vectors; use HKDF test vectors from other sources or
  derive SHA-384 vectors from the construction)
- Cross-validation: extract + expand matches oneshot `derive()`
- Output length boundary tests
- `no_std` compile coverage

## Ship Order

1. **HMAC-SHA384 / HMAC-SHA512** — mechanical, high confidence, unblocks HKDF
2. **HKDF-SHA384** — depends on HMAC-SHA384 for extract phase
3. **X25519** — independent, but highest protocol value

All three can be developed in parallel. HMAC-SHA384 and HKDF-SHA384 are
mechanical extensions of shipped code. X25519 is new algorithmic work but
builds on existing field infrastructure.

## Acceleration Posture

All three primitives inherit existing hardware acceleration:

| Primitive | x86_64 | aarch64 | s390x | powerpc64 | riscv64 |
|-----------|--------|---------|-------|-----------|---------|
| HMAC-SHA384/512 | inherit SHA-NI | inherit SHA2-CE | inherit KIMD | portable | inherit Zknh |
| HKDF-SHA384 | inherit SHA-NI | inherit SHA2-CE | inherit KIMD | portable | inherit Zknh |
| X25519 | AVX2/IFMA (shared Ed25519 field) | NEON (when ED25519-6 ships) | portable | portable | portable |

No new SIMD kernels are needed for HMAC/HKDF — they inherit the SHA-2 dispatch
matrix automatically. X25519 can start on the portable 5x51 field and optionally
pick up AVX2/IFMA vectorization from the Ed25519 field backends.

## After These Three

The classical auth surface is closed. Remaining auth-adjacent work:

- **Argon2id** — deferred unless password hashing enters scope (requires BLAKE2b
  or a new BLAKE3-based design; see conversation notes)
- **AEGIS-128L** — AEAD, not auth; optional companion to AEGIS-256
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
