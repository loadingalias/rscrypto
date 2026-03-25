# Auth Roadmap

## Recommended Set

Build this authentication and key-derivation stack first:

- `HMAC-SHA256`
- `HKDF-SHA256`
- `Ed25519` / `EdDSA`
- `KMAC256` and `cSHAKE256`
- `Argon2id` only if password hashing enters scope

Keep this split clear:

- protocol compatibility: `HMAC-SHA256`, `HKDF-SHA256`, `Ed25519`
- NIST-native SHA-3 track: `KMAC256`, `cSHAKE256`
- internal-only keyed hashing / derivation: `BLAKE3` keyed mode and `derive_key`

## Why This Set

As of 2026-03-24, the boring, defensible baseline is still:

- `HMAC-SHA256` for request signing, token MACs, and compatibility surfaces
- `HKDF-SHA256` for extract-then-expand key derivation
- `Ed25519` for fast modern signatures
- `KMAC256` where you want a NIST SHA-3-native MAC/PRF instead of HMAC

This is the right split because it covers:

- IETF protocol reality
- NIST reality
- internal systems that want a modern fast keyed hash

## Ship Now

### 1. HMAC-SHA256

This is mandatory.

- AWS SigV4 and a lot of existing protocol glue still depend on it.
- It is small work on top of the SHA-256 code already in `rscrypto`.
- NIST is moving HMAC guidance from FIPS 198-1 into SP 800-224, but HMAC itself is not going away.

### 2. HKDF-SHA256

This should ship with HMAC.

- It is the boring interoperable KDF.
- It is better than inventing ad hoc “keyed hash as KDF” APIs for protocol surfaces.
- Your existing BLAKE3-based derivation can stay as an internal fast path, but it should not be the only KDF story.

### 3. Ed25519

If `rscrypto` wants to be taken seriously as a modern systems crypto crate, signatures need a clean path.

- EdDSA is in NIST FIPS 186-5.
- Ed25519 is the practical first target.
- It is the modern small-key, fast-signature, easy-to-use answer for application auth.

## Ship Second

### 4. KMAC256 and cSHAKE256

These are worth doing after HMAC and HKDF, not before.

- They align with NIST SP 800-185.
- They give you a SHA-3-native MAC / PRF / customizable hash story.
- They matter if you want `rscrypto` to cover the NIST world cleanly, not just the IETF world.

## Defer

### 5. Argon2id

If password hashing becomes a real requirement, `Argon2id` is the right choice.

It is not first-wave work for `rscrypto` unless password authentication is actually in scope. It is a separate subsystem with different tuning, memory-pressure, and benchmarking needs.

## What Not To Prioritize

- `CMAC`: only for standards compatibility where AES-based MAC is mandated
- generic “MAC over every digest”: adds API surface and very little value on day one
- RSA signatures: important for interop, not for a clean modern default
- P-256 ECDSA before Ed25519: only do it when a protocol forces it

## SIMD / HW Rules

These rules should hold for all auth primitives:

- No lookup-table AES for anything new. If AES is used, use hardware AES paths or constant-time bitsliced code.
- No secret-dependent memory access in signature, MAC, or KDF hot paths.
- Prefer scalar constant-time reference code first, then add ISA paths.
- x86_64 priorities: `SHA-NI`, `AVX2`, `AVX-512` only when it clearly wins.
- aarch64 priorities: ARMv8 SHA extensions, NEON, then SVE2 only where it materially helps.
- Keep the public API stable while dispatch remains internal.
- Do not tie correctness to SIMD. Every accelerated path must differential-test against the portable path.

## Acceleration Posture

As of 2026-03-24, the right rule is:

- pursue full coverage where the primitive naturally inherits an existing accelerated core
- do not force fake ISA parity where the primitive does not map cleanly to the checksum/hash backend model
- measure first, dispatch second, document third

For auth, that means:

- `HMAC-SHA256` and `HKDF-SHA256` should inherit the `SHA-256` backend matrix directly
- `KMAC256` and `cSHAKE256` should inherit the `Keccak` backend matrix directly
- `Ed25519` should get a portable constant-time baseline first, then targeted SIMD on the architectures where it clearly pays off
- `Argon2id` is a separate memory-hard subsystem and should not be judged by the same “every ISA gets a bespoke kernel” bar

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
6. `Argon2id` if password auth becomes real

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

### Phase 2C. Signing and Verification API

Minimum concrete API:

- `Ed25519Keypair::from_secret_key(secret)`
- `Ed25519Keypair::sign(message)`
- `Ed25519PublicKey::verify(message, signature)`
- one-shot verify helper returning `VerificationError`

Hold the line here:

- no batch verify in phase 1
- no prehash variant in phase 1
- no generic signature trait until a second signature algorithm exists
- no trait dependency lock-in to an external ecosystem crate

Current status:

- `Ed25519SecretKey::public_key` is wired
- `Ed25519SecretKey::sign` is wired
- `Ed25519Keypair::from_secret_key` and `sign` are wired
- `Ed25519PublicKey::verify` is wired
- the module-level one-shot `verify(...)` helper is wired
- Ed25519 should be promoted into the stable `auth` and crate-root re-export surface now that sign / verify are real

Decision:

- do promote the typed Ed25519 surface: `Ed25519SecretKey`, `Ed25519PublicKey`, `Ed25519Signature`, `Ed25519Keypair`
- do not promote every internal helper; keep arithmetic and point modules internal
- keep the one-shot free function explicit as `verify_ed25519` at the root so it stays discoverable without colliding with other future signature schemes

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

## Auth Target Matrix

This is the release bar the auth subsystem should aim at.

| Primitive | `no_std` | `wasm32` | `x86_64` | `aarch64` | `powerpc64` | `s390x` | `riscv64` |
|-----------|----------|----------|----------|-----------|-------------|---------|-----------|
| `HMAC-SHA256` | yes | yes | inherit `SHA-NI` | inherit SHA2 CE | inherit only when validated | inherit `KIMD` | inherit `Zknh` |
| `HKDF-SHA256` | yes | yes | inherit `SHA-NI` | inherit SHA2 CE | inherit only when validated | inherit `KIMD` | inherit `Zknh` |
| `KMAC256` / `cSHAKE256` | yes | yes | measured dispatch only | measured dispatch only | portable first | `KIMD` / absorb where it wins | portable first |
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
