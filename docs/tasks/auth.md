# Auth Roadmap

## Current Status (2026-04-01)

Phase 1 (HMAC-SHA256, HKDF-SHA256, Ed25519) is **shipped and benchmarked**.
Phase 3 (`cSHAKE256`, `KMAC256`) is **now shipped**.
Phase 4 (`Ascon-CXOF128`) is **now shipped**.

AEAD work has now split cleanly into its own track:

- AEAD foundations are in tree under [`docs/tasks/aead.md`](aead.md).
- `XChaCha20-Poly1305` and `ChaCha20-Poly1305` are shipped there, with native ChaCha dispatch across x86-64/aarch64/wasm/s390x/ppc64/riscv64.
- `AES-256-GCM-SIV` is shipped there on the portable constant-time baseline (algebraic AES S-box, Pornin bmul64 POLYVAL, RFC 8452 vectors). Hardware acceleration (AES-NI + PCLMULQDQ, ARM AES + PMULL) is next.
- `auth.md` now treats `Ascon-CXOF128` as the remaining auth-side Ascon item instead of implicitly carrying the AEAD rollout too.

**Historical CI benchmark results** — [run #23544327679](https://github.com/loadingalias/rscrypto/actions/runs/23544327679) on Zen5, SPR, Graviton4, s390x:

| Primitive | vs | Win Rate | Key Finding |
|-----------|-----|---------|-------------|
| **HMAC-SHA256** | `hmac`+`sha2` | **53%** (19W/12T/5L) | Grav4: 3.3-3.7x WIN (SHA-CE). s390x: 1.7-9.4x WIN (KIMD). Zen5/SPR: 4-6% loss at 0-32 B (key setup overhead). |
| **HKDF-SHA256** | `hkdf`+`sha2` | ~38% | Grav4: 1.6-2.0x WIN. Older Zen5/SPR runs showed **1.5-1.9x LOSS**; remeasure after the cached `prk_mac` path now in tree. |
| **Ed25519** | `ed25519-dalek` | **~65%** | Sign/keygen 1.3-2x WIN (basepoint table). Verify 12% behind dalek on small messages, wins on large messages. |

### Performance Priorities

> **Master acceleration tracker:** [`docs/tasks/acceleration.md`](acceleration.md)

1. **HKDF expand on x86-64** — rerun the matrix and measure the residual gap.
   The old x86-64 loss was tied to repeated keyed-HMAC setup. The current code
   already caches keyed HMAC state inside `HkdfSha256`, so the next step is
   evidence, not another speculative rewrite.

2. **Ed25519 small-message verify** — ~12% behind dalek (51µs vs 45µs on M4).
   Sign/keygen already wins by 1.3-2x thanks to shipped optimizations:
   - Precomputed 64×8 radix-16 basepoint table (ED25519-1 ✅)
   - Dedicated dbl-2008-hwcd doubling formula (ED25519-2 ✅)
   - Straus/Shamir verification with variable-base radix-16 (ED25519-3 ✅)
   The remaining verify gap is in portable 5×51 field mul/square. AVX2 vectorization
   (ED25519-4) or NEON (ED25519-6) would close it. Target: parity with dalek.
   **Remaining tasks:** ED25519-4/5/6 in [`acceleration.md`](acceleration.md).

3. **HMAC-SHA256 small-size overhead** — 4-6% loss at 0-32 B on Zen5/SPR. The SHA-256
   compression is fast (SHA-NI), but HMAC wrapping adds two compressions (ipad + opad).
   Investigate whether the inner/outer key pads can be pre-compressed to a midstate.

4. **KMAC256/cSHAKE256 inherit Keccak regression** — The Keccak-f dispatch bug
   (KECCAK-1 in [`acceleration.md`](acceleration.md)) means SHA-3-based auth primitives
   are 26-45% slower than they should be on x86_64. Fix is trivial: disable the
   x86-avx512 kernel in dispatch tables.

## Authenticated Crypto Contract

`auth.md` owns MACs, KDFs, and signatures. [`docs/tasks/aead.md`](aead.md) owns AEAD
algorithms. They still need to meet the same subsystem bar.

That bar is:

- typed public surfaces for fixed-width secrets, nonces, tags, signatures, and public keys
- one-shot helpers plus stateful or in-place APIs where the primitive naturally supports them
- detached and combined outputs where AEAD semantics require both
- caller-provided output buffers for the core path; no hidden allocation as the default contract
- explicit error boundaries between buffer misuse, malformed inputs, and authentication failure
- portable constant-time baselines first, then internal dispatch
- zeroize owned secret material on drop
- `no_std` as a first-class constraint; `std` only widens runtime dispatch and ergonomics

This matters because “best API in the ecosystem” is not just a longer method list.
It means one coherent contract across authentication and AEAD:

- typed, misuse-resistant inputs
- no allocation pressure hidden behind convenience
- no backend-specific semantic drift
- no fake platform support where the fast path is unmeasured or unauditable

---

## Recommended Set

Build this authentication and key-derivation stack first:

- `HMAC-SHA256` ✅ shipped
- `HKDF-SHA256` ✅ shipped
- `Ed25519` / `EdDSA` ✅ shipped (correctness-first, performance pending)
- `KMAC256` and `cSHAKE256` ✅ shipped
- `Ascon-CXOF128` ✅ shipped
- `Argon2id` only if password hashing enters scope

Keep this split clear:

- protocol compatibility: `HMAC-SHA256`, `HKDF-SHA256`, `Ed25519`
- NIST-native SHA-3 track: `KMAC256`, `cSHAKE256`
- Ascon standards track: `Ascon-Hash256`, `Ascon-XOF128`, `Ascon-CXOF128`; `Ascon-AEAD128` is tracked under [`docs/tasks/aead.md`](aead.md)
- AES-family AEAD: `AES-256-GCM-SIV` (shipped), `AES-256-GCM` (next); tracked under [`docs/tasks/aead.md`](aead.md)
- internal-only keyed hashing / derivation: `BLAKE3` keyed mode and `derive_key`

## Why This Set

As of 2026-03-30, the boring, defensible baseline is still:

- `HMAC-SHA256` for request signing, token MACs, and compatibility surfaces
- `HKDF-SHA256` for extract-then-expand key derivation
- `Ed25519` for fast modern signatures
- `KMAC256` and `cSHAKE256` where you want a NIST SHA-3-native MAC / PRF / customization story
- `Ascon-CXOF128` to complete the Ascon family that NIST finalized in SP 800-232

This is the right split because it covers:

- IETF protocol reality
- NIST reality
- internal systems that want modern keyed derivation without a dependency tower

## Ship Now — ✅ ALL SHIPPED

### 1. HMAC-SHA256 ✅

Shipped. 53% win rate vs RustCrypto. Dominant on Grav4 (3.3-3.7x) and s390x (1.7-9.4x)
due to SHA-256 hardware acceleration inheritance. Small-size overhead on x86-64 (4-6%)
is the remaining optimization target.

### 2. HKDF-SHA256 ✅

Shipped. Wins on Grav4 (1.6-2.0x) and s390x (large sizes). **1.5-1.9x behind on x86-64** —
the expand loop needs HMAC key pad caching to avoid redundant SHA-256 compressions.

### 3. Ed25519 ✅

Shipped (correctness-first). Passes all RFC 8032 vectors, differential-tested against dalek.
**8-13x behind dalek** — the portable scalar field baseline is deliberately unoptimized.
Performance is the primary post-ship optimization target (see Performance Priorities above).

## Ship Second — ✅ SHIPPED

### 4. KMAC256 and cSHAKE256 ✅

Shipped.

- They align with NIST SP 800-185.
- They give you a SHA-3-native MAC / PRF / customizable hash story.
- They matter if you want `rscrypto` to cover the NIST world cleanly, not just the IETF world.

### 5. Ascon-CXOF128 ✅

Shipped.

- NIST SP 800-232 finalized `Ascon-AEAD128`, `Ascon-Hash256`, `Ascon-XOF128`,
  and `Ascon-CXOF128` on August 13, 2025.
- The AEAD half is already tracked separately in [`docs/tasks/aead.md`](aead.md), so the auth-side gap is now `Ascon-CXOF128`, not the whole family.
- `rscrypto` already ships the hash/XOF half of that family.
- `Ascon-CXOF128` lets the Ascon track cover customization cleanly instead of
  leaving that story entirely to Keccak.

## Defer

### 6. Argon2id

If password hashing becomes a real requirement, `Argon2id` is the right choice.

It is not first-wave work for `rscrypto` unless password authentication is actually in scope. It is a separate subsystem with different tuning, memory-pressure, and benchmarking needs.

## What Not To Prioritize

- `CMAC`: only for standards compatibility where AES-based MAC is mandated
- generic “MAC over every digest”: adds API surface and very little value on day one
- RSA signatures: important for interop, not for a clean modern default
- P-256 ECDSA before Ed25519: only do it when a protocol forces it

## SIMD / HW Rules

These rules should hold for all authenticated primitives, including the AEAD
track in [`docs/tasks/aead.md`](aead.md):

- No lookup-table AES for anything new. If AES is used, use hardware AES paths or constant-time bitsliced code.
- No secret-dependent memory access in signature, MAC, or KDF hot paths.
- Prefer scalar constant-time reference code first, then add ISA paths.
- x86_64 priorities: `SHA-NI` / `AES-NI` / `PCLMUL` where the primitive inherits them, `AVX2` for software-first lanes, `AVX-512` or `IFMA` only when the end-to-end win is real.
- aarch64 priorities: ARMv8 SHA / AES extensions and `PMULL` where applicable, `NEON` for software-first lanes, then `SVE2` only where it materially helps.
- `s390x`, `powerpc64`, and `riscv64` priorities: use crypto or vector facilities only where the primitive maps cleanly and the maintenance cost is justified by measured wins.
- Keep the public API stable while dispatch remains internal.
- Do not tie correctness to SIMD. Every accelerated path must differential-test against the portable path.

## Acceleration Posture

As of 2026-03-30, the right rule is:

- pursue full coverage where the primitive naturally inherits an existing accelerated core
- do not force fake ISA parity where the primitive does not map cleanly to the checksum/hash backend model
- measure first, dispatch second, document third

For auth, that means:

- `HMAC-SHA256` and `HKDF-SHA256` should inherit the `SHA-256` backend matrix directly
- `KMAC256` and `cSHAKE256` should inherit the `Keccak` backend matrix directly
- `Ascon-CXOF128` should inherit the `Ascon` permutation backend matrix directly
- `Ed25519` should get a portable constant-time baseline first, then targeted SIMD on the architectures where it clearly pays off
- `Argon2id` is a separate memory-hard subsystem and should not be judged by the same “every ISA gets a bespoke kernel” bar

### Platform Ladder Discipline

The platform story should be ambitious but not theatrical.

- `x86_64`: treat the ladder as `portable -> AVX2 -> AVX-512` for software-first primitives and `portable -> SHA-NI` or `AES-NI/PCLMUL -> VAES/VPCLMUL/IFMA` for hardware-first primitives. Do not skip straight to the fanciest ISA and call the lower rungs “good enough.”
- `aarch64`: treat the ladder as `portable -> NEON` for software-first primitives and `portable -> SHA2/AES/PMULL -> SVE2` for hardware-assisted or data-parallel primitives. `SVE2` is not a default; it is an earned optimization.
- `s390x`: exploit CPACF and vector facilities where they give a real lane-native win, especially for SHA/AES-class work. Do not promise bespoke curve or permutation kernels without measurement.
- `powerpc64`: use POWER crypto or VSX backends only where the implementation is maintainable and benchmarked on actual POWER hardware.
- `riscv64`: target scalar or vector crypto extensions opportunistically, but do not let ecosystem immaturity turn into speculative API or dispatch complexity.
- `wasm32`: support the contract and correctness story. Treat acceleration as optional and environment-specific, never as the semantic baseline.

### What “In Line” Means

For `HMAC-SHA256` / `HKDF-SHA256`, “in line” means:

- `no_std`: yes
- `wasm32`: yes, via the existing `SHA-256` portable / `simd128` story
- `x86_64`: yes, via `SHA-NI`
- `aarch64`: yes, via ARM SHA2 extensions
- `s390x`: yes, via `KIMD`
- `riscv64`: yes, via `Zknh`
- `powerpc64`: only when the `SHA-256` POWER backend is revalidated and actually dispatched

For `KMAC256` / `cSHAKE256`, “in line” means:

- `no_std`: yes
- `wasm32`: yes for compatibility, but portable first
- `x86_64`: only use SIMD if it beats the current portable Keccak path
- `aarch64`: use SHA3 CE only where it wins, not by ideology
- `s390x`: use `KIMD` / batch-absorb paths where they materially help
- `powerpc64` / `riscv64`: portable first until there is measured evidence for a better path

This matters because `rscrypto` already has proof that “SIMD exists” does not imply “SIMD wins” for Keccak.

### Ed25519 Acceleration Plan

`Ed25519` should not be held to the same dispatch shape as `SHA-256`.

The correct order is:

1. scalar constant-time portable baseline
2. `x86_64` `AVX2` fixed-base / multi-scalar work where it clearly reduces sign / verify cost
3. `x86_64` `AVX-512IFMA` only if it is measurably better and still auditable
4. `aarch64` `NEON` hybrid path
5. `SVE2` only if it shows a real win over `NEON`
6. `powerpc64`, `s390x`, `riscv64`, and `wasm32` stay on portable until there is strong measured evidence for more

Strong opinions:

- do not write a weak `SSSE3` / `SSE4.1` Ed25519 backend just to complete a visual ladder
- do not promise `s390x` / `POWER` / `RISC-V` bespoke Ed25519 kernels in v1 unless we can benchmark and maintain them
- do not make `wasm32` SIMD a correctness dependency for elliptic-curve code

### Argon2id Acceleration Plan

If `Argon2id` enters scope:

- it should be `alloc`-using and `no_std + alloc`, not sold as a tiny `core`-only primitive
- portable first
- `x86_64` `AVX2` next
- `aarch64` `NEON` next
- `wasm32` compatibility is plausible, but performance promises should be modest
- do not spend early cycles on `POWER` / `s390x` / `RISC-V` vector backends unless there is demand and benchmark evidence

This is not a failure of ambition. `Argon2id` is memory-hard; bandwidth and layout dominate earlier than “one more vector backend.”

## Recommended Implementation Order

1. `HMAC-SHA256`
2. `HKDF-SHA256`
3. `Ed25519`
4. `KMAC256`
5. `cSHAKE256`
6. `Ascon-CXOF128`
7. `Argon2id` if password auth becomes real

## Phase 1 API Shape

The first auth delivery should look like the rest of the crate.

- add a dedicated `rscrypto::auth` module
- keep the root exports small: `HmacSha256`, `HkdfSha256`, and `Mac`
- keep auth-specific errors under `rscrypto::auth`
- do not put protocol auth primitives under `hashes::*`

Phase 1 public shapes:

- `HmacSha256::mac(key, data)`
- `HmacSha256::new(key)` then `update`, `finalize`, `reset`, `verify`
- `HkdfSha256::new(salt, ikm)` / `extract(salt, ikm)`
- `expand(info, out)` and `derive(salt, ikm, info, out)`
- `derive_array::<N>(...)` for fixed-size protocol key schedules

That keeps continuity with the existing checksum and digest APIs:

- one-shot helpers on the concrete type
- stateful usage for incremental flows
- root exports only for stable user-facing types
- advanced or niche pieces kept explicit under the subsystem module

## Ed25519 Breakdown

Do this in one direction only: establish the byte-level typed surface first, then
fill in the arithmetic and protocol behavior underneath it.

### Phase 2A. Typed Surface and Module Layout

- keep Ed25519 under `rscrypto::auth::ed25519`
- do not root-export it until sign and verify are real
- use explicit wrappers, not raw `&[u8]`:
  - `Ed25519SecretKey`
  - `Ed25519PublicKey`
  - `Ed25519Signature`
- redact secret-key `Debug`
- zeroize owned secret material on drop

### Phase 2B. Arithmetic Core

- field element arithmetic mod `2^255 - 19`
- scalar arithmetic mod group order `L`
- Edwards point encode / decode
- fixed-base scalar multiplication for public-key derivation
- variable-base verification path

Break this into explicit deliverables:

#### Phase 2B1. Internal Layout

- create internal `field`, `scalar`, and `point` modules under `auth::ed25519`
- lock in the portable baseline layout explicitly instead of letting it leak out of ad hoc helpers
- keep these internal until sign / verify are real

#### Phase 2B2. Field Arithmetic

- addition / subtraction
- carry propagation and normalization
- multiplication / squaring
- inversion
- canonical encode / decode support

Current status:

- the portable 5x51 field baseline is in place
- add / sub / mul / square / normalize are wired
- canonical 32-byte encode / decode is wired
- inversion is wired
- square-root recovery for decompression is wired

#### Phase 2B3. Scalar Arithmetic

- canonical reduction mod `L`
- multiply-add for signature equation work
- scalar clamp helper for secret-key expansion

Current status:

- canonical scalar decode and encode are wired
- reduction from arbitrary byte strings mod `L` is wired
- modular add / multiply / multiply-add are wired
- the current implementation is correctness-first, not optimized

#### Phase 2B4. Point Arithmetic

- extended Edwards coordinates
- point addition / doubling
- fixed-base multiplication
- variable-base multiplication for verify
- point decompression with malformed-input rejection

Current status:

- extended Edwards point storage is in place
- affine construction and affine conversion are in place
- the portable complete addition law is in place
- doubling currently routes through point addition for correctness-first simplicity
- compressed point encode / decode is wired
- fixed-base multiplication is wired through the portable double-and-add baseline
- variable-base multiplication for verify is wired through the same portable scalar-multiplication baseline

#### Phase 2B5. Hash Integration

- secret-key expansion through SHA-512
- nonce derivation path for signing
- challenge hash path for verification

Current status:

- the internal SHA-512 secret-key expansion and scalar-clamp path is in place
- keep the expanded secret material internal until public sign / verify exist
- do not collapse nonce-prefix handling into generic scalar helpers

Do not blur these together. Field, scalar, point, and hashing should remain auditable as separate layers.

### Phase 2C. Signing and Verification API ✅ DONE

Minimum concrete API — all shipped and crate-root exported:

- `Ed25519Keypair::from_secret_key(secret)` ✅
- `Ed25519Keypair::sign(message)` ✅
- `Ed25519PublicKey::verify(message, signature)` ✅
- one-shot verify helper returning `VerificationError` ✅

Hold the line here:

- no batch verify in phase 1
- no prehash variant in phase 1
- no generic signature trait until a second signature algorithm exists
- no trait dependency lock-in to an external ecosystem crate

Typed surface promoted: `Ed25519SecretKey`, `Ed25519PublicKey`,
`Ed25519Signature`, `Ed25519Keypair` are crate-root re-exports.

### Phase 2E. Performance (NEW — post-ship)

**Target:** ≤3x of dalek for sign/verify by v0.2.

Current gap: 8-13x. Breakdown by operation (Zen5):

| Operation | rscrypto | dalek | Gap |
|-----------|----------|-------|-----|
| Public key | 97.6 µs | 11.5 µs | 8.5x |
| Sign (32 B) | 99.1 µs | 12.0 µs | 8.3x |
| Verify (32 B) | 247 µs | 18.8 µs | 13.1x |

Acceleration roadmap (in order):

1. **Precomputed basepoint table** — fixed-base scalar mul using a precomputed
   table of `[B, 2B, 4B, ..., 2^252·B]` (or windowed NAF). Eliminates
   double-and-add for keygen/sign. Expected: 5-10x improvement.
2. **Dedicated doubling formula** — current doubling routes through general
   addition. Use the Edwards dedicated doubling formula (fewer field muls).
3. **Straus double-scalar mul for verify** — `aR + sB` via interleaved
   Straus method instead of two separate scalar muls. Expected: 2-3x verify.
4. **AVX2 field arithmetic** — vectorize 5×51-limb field mul/square across
   two independent field elements (common in addition/doubling).
5. **AVX-512IFMA** — 52-bit integer fused multiply-add, natural fit for
   5×51-limb representation. Only where measurably better.

### Phase 2D. Validation

- RFC 8032 vectors
- malformed-point decode tests
- negative verification tests
- differential tests against a well-audited oracle crate
- no_std compile coverage

Current status:

- RFC 8032 vector 1 key-derivation and signing coverage is in place
- RFC 8032 vectors 2, 3, 1024-byte, and `SHA(abc)` are in place
- malformed compressed-point and non-canonical `S` rejection tests are in place
- negative modified-message verification coverage is in place
- differential testing against `ed25519-dalek` is in place
- `no_std` auth compile coverage is in place

## Phase 3. SHA-3 Native Auth Track ✅ DONE

This is where the roadmap stops being “HMAC, but Keccak-shaped.”

Do not force `KMAC256` into the current fixed-size `Mac` mental model just to
reuse a trait. `KMAC256` is naturally variable-output. The API should admit that
instead of pretending every MAC is a 32-byte tag forever.

### Phase 3A. cSHAKE256 API Shape

`cSHAKE256` should look like a customized XOF, not like a digest with hidden
domain-separation state.

Minimum public shape:

- `Cshake256::new(function_name, customization)`
- `Cshake256::xof(function_name, customization, data)`
- `update`, `finalize_xof`, `reset`
- reader side conforms to the existing `Xof` contract

Hold the line here:

- do not hide `function_name` and `customization` behind an options builder
- do not collapse empty and non-empty customization cases into undocumented aliases
- do not invent a generic “configurable sponge” public type first

### Phase 3B. KMAC256 API Shape

`KMAC256` should be explicit about output sizing.

Minimum public shape:

- `Kmac256::new(key, customization)`
- `Kmac256::mac_into(key, customization, data, out)`
- `update`, `finalize_into`, `reset`
- `verify(key, customization, data, expected)`
- `mac_array::<N>(...)` for fixed-size protocol tags where a const generic is useful

Strong opinion:

- do not wedge `KMAC256` into `traits::Mac` unless the trait is intentionally widened for variable-length outputs
- do not split this into fake “KMAC” and “KMACXOF” user types unless the standards semantics genuinely diverge in the public contract

### Phase 3C. SHA-3 Validation Bar

- SP 800-185 vectors for `cSHAKE256` and `KMAC256`
- empty `function_name` / empty `customization` cases
- split-update equivalence vs one-shot usage
- variable output lengths, including short tags and multi-block outputs
- differential tests against a well-audited Keccak oracle where semantics align
- `no_std` compile coverage

## Phase 4. Ascon-CXOF128 ✅ DONE

This should mirror the existing Ascon hash/XOF ergonomics, not bolt on a Keccak-shaped API.

### Phase 4A. API Shape

Minimum public shape:

- `AsconCxof128::new(customization)`
- `AsconCxof128::xof(customization, data)`
- `update`, `finalize_xof`, `reset`
- `hash_into(customization, data, out)` for direct one-shot filling

Hold the line here:

- keep the customization input explicit in the type construction or one-shot helper
- reuse the existing `Xof` reader model for squeezing
- do not introduce a separate trait just for “customizable XOF”

### Phase 4B. Validation Bar

- SP 800-232 vectors for `Ascon-CXOF128`
- empty and non-empty customization cases
- long-output squeeze coverage across multiple permutation rounds
- split-update equivalence vs one-shot usage
- differential testing against a known-good oracle implementation
- `no_std` compile coverage

### Phase 4C. Acceleration Posture

`Ascon-CXOF128` inherits the Ascon permutation matrix directly.

- `x86_64`: portable first, then `AVX2`, then `AVX-512` only if measured
- `aarch64`: portable first, then `NEON` only if measured
- `powerpc64`, `s390x`, `riscv64`, `wasm32`: portable first until data says otherwise

Do not build a dispatch cathedral around a primitive whose core cost is already compact.

## Auth Target Matrix

This is the release bar the auth subsystem should aim at.

| Primitive | `no_std` | `wasm32` | `x86_64` | `aarch64` | `powerpc64` | `s390x` | `riscv64` |
|-----------|----------|----------|----------|-----------|-------------|---------|-----------|
| `HMAC-SHA256` | yes | yes | inherit `SHA-NI` | inherit SHA2 CE | inherit only when validated | inherit `KIMD` | inherit `Zknh` |
| `HKDF-SHA256` | yes | yes | inherit `SHA-NI` | inherit SHA2 CE | inherit only when validated | inherit `KIMD` | inherit `Zknh` |
| `KMAC256` / `cSHAKE256` | yes | yes | measured dispatch only | measured dispatch only | portable first | `KIMD` / absorb where it wins | portable first |
| `Ascon-CXOF128` | yes | yes | measured dispatch only | measured dispatch only | portable first | portable first | portable first |
| `Ed25519` | yes | yes | portable, then `AVX2`, then maybe `AVX-512IFMA` | portable, then `NEON`, then maybe `SVE2` | portable first | portable first | portable first |
| `Argon2id` | `alloc`-centric | maybe | portable, then `AVX2` | portable, then `NEON` | portable first | portable first | portable first |

## Sources

- NIST FIPS 198-1: HMAC
  https://csrc.nist.gov/pubs/fips/198-1/final
- NIST SP 800-224 IPD: updated HMAC guidance
  https://csrc.nist.gov/pubs/sp/800/224/ipd
- RFC 5869: HKDF
  https://datatracker.ietf.org/doc/html/rfc5869
- NIST SP 800-108 Rev. 1: KDFs using HMAC, CMAC, KMAC
  https://csrc.nist.gov/pubs/sp/800/108/r1/upd1/final
- NIST SP 800-185: cSHAKE, KMAC, TupleHash, ParallelHash
  https://csrc.nist.gov/pubs/sp/800/185/final
- NIST decision to revise SP 800-185 / update FIPS 202
  https://csrc.nist.gov/news/2025/decision-to-update-fips-202-and-revise-sp-800-185
- NIST SP 800-232: Ascon standard family
  https://csrc.nist.gov/pubs/sp/800/232/final
- NIST Lightweight Cryptography project
  https://csrc.nist.gov/projects/lightweight-cryptography
- NIST FIPS 186-5 / SP 800-186 announcement
  https://csrc.nist.gov/News/2023/nist-releases-fips-186-5-and-sp-800-186
- NIST SP 800-186: Edwards curves for government use
  https://csrc.nist.gov/pubs/sp/800/186/final
- RFC 8032: Ed25519 / EdDSA
  https://datatracker.ietf.org/doc/html/rfc8032
- RFC 9106: Argon2
  https://datatracker.ietf.org/doc/html/rfc9106
- HACLxN: verified SIMD crypto, including vectorized ChaCha20-Poly1305 and integration into HMAC/HKDF/Ed25519 stacks
  https://eprint.iacr.org/2020/572.pdf
- Size, Speed, and Security: An Ed25519 Case Study
  https://eprint.iacr.org/2021/471.pdf
- AVX-512 Curve25519-family vectorization
  https://eprint.iacr.org/2020/388.pdf
- Assembly-optimized Curve25519 for Cortex-M4 / Cortex-M33
  https://eprint.iacr.org/2025/523
