# AEAD Roadmap

## Current Status (2026-04-01)

**All six AEAD primitives are shipped.** The AEAD portfolio is complete.

`XChaCha20-Poly1305` is shipped on the typed AEAD surface.
The plain `ChaCha20-Poly1305` interop companion is shipped on the same core.
The ChaCha-family core now has explicit backend routing for `x86_64` (`portable -> AVX2 -> AVX-512`), `aarch64` (`portable -> NEON`), `wasm32/wasm64` (`portable -> simd128`), and explicit `s390x` / `powerpc64` / `riscv64` vector routes. The x86_64 backends use `vprold` (AVX-512) and `vpshufb` (AVX2) for single-instruction rotations, plus SIMD matrix transpose for the keystream XOR phase.
`Poly1305` now follows the same primitive-keyed dispatch contract, with architecture-specific block kernels on `x86_64` (`AVX2` / `AVX-512`), `aarch64` (`NEON`), and `wasm32` (`simd128`), plus explicit fallback routes everywhere else.

`AES-256-GCM-SIV` is shipped with **x86_64 + aarch64 hardware acceleration** plus **x86_64 VAES-512 + VPCLMULQDQ wide path**:
- AES-256: enum-based `Aes256EncKey` dispatches AES-NI (x86_64), AES-CE (aarch64), or portable algebraic S-box
- POLYVAL: `clmul128_reduce()` dispatches PCLMULQDQ (x86_64), PMULL (aarch64), or portable Pornin bmul64
- Wide path (Zen4+/ICL+): VAES-512 4-block CTR + VPCLMULQDQ 4-block POLYVAL (schoolbook-then-reduce)
- Full RFC 8452 construction: key derivation, POLYVAL tag, AES-CTR
- Verified against RFC 8452 Appendix A (POLYVAL) and Appendix C.2 (AES-256-GCM-SIV) test vectors

`AES-256-GCM` is shipped with **x86_64 + aarch64 hardware acceleration** plus **x86_64 VAES-512 + VPCLMULQDQ wide path**:
- Reuses the same AES-256 core with AES-NI / AES-CE dispatch
- GHASH: accelerated via shared `clmul128_reduce` from POLYVAL (GHASH↔POLYVAL bridge preserved)
- Wide path (Zen4+/ICL+): VAES-512 4-block CTR + VPCLMULQDQ 4-block GHASH with precomputed H powers
- AES-CTR with big-endian 32-bit counter (bytes 12-15) per NIST SP 800-38D
- Full GCM construction: H = AES(K, 0), J0 = IV || 0x00000001, tag = AES(K, J0) XOR GHASH(H, AAD, C)
- Verified against NIST SP 800-38D Test Cases 13, 14, 15, 16 (AES-256)
- Hardware acceleration shared with GCM-SIV: same dispatch path benefits both

`Ascon-AEAD128` is shipped on the portable constant-time baseline:
- Full NIST SP 800-232 construction with 128-bit key, 128-bit nonce, 128-bit tag
- Portable Ascon permutation (rate=64 bits, PA=12, PB=6), no ornamental SIMD dispatch
- Verified against official Ascon LWC KAT vectors (11 test cases across varied AAD/PT sizes)

`AEGIS-256` is shipped with **x86_64 AES-NI + aarch64 AES-CE hardware acceleration**:
- Full draft-irtf-cfrg-aegis-aead-18 construction with 256-bit key, 256-bit nonce, 128-bit tag
- Three backends: portable (algebraic AES S-box), x86_64 AES-NI (`_mm_aesenc_si128`), aarch64 AES-CE (`vaeseq_u8` + `vaesmcq_u8`)
- Runtime dispatch via `platform::caps()` at construction time
- Verified against all 5 spec test vectors (Appendix A.3) plus AESRound and Update unit vectors
- VAES-512 wide path (AEGIS-256X2/X4) is future work

## Recommended Set

Build this AEAD portfolio:

- `XChaCha20-Poly1305`
- `AES-256-GCM-SIV`
- `AES-256-GCM`
- `Ascon-AEAD128`

Watch closely, but do not let it derail first delivery:

- `AEGIS-256`

Companion interop profiles, once the core lands:

- `ChaCha20-Poly1305` from the same ChaCha20 / Poly1305 core
- `AES-128-GCM` and `AES-128-GCM-SIV` if downstream protocols actually need them

## Default Choices

Use these defaults unless interoperability forces something else:

- software-first default: `XChaCha20-Poly1305`
- AES-family default: `AES-256-GCM-SIV`
- interop default: `AES-256-GCM`
- constrained / lightweight default: `Ascon-AEAD128`

Keep one rule straight:

- `XChaCha20-Poly1305` is the clean first build
- plain `ChaCha20-Poly1305` is the interop companion, not the design center

## Why This Set

### XChaCha20-Poly1305

This should be the first AEAD you ship.

- Portable
- Fast without AES hardware
- Large nonce space
- Operationally simpler than 96-bit nonce schemes

If you control both ends, this is the cleanest first answer.

The obvious follow-on is the RFC 8439 / TLS-style `ChaCha20-Poly1305` profile.
That should be treated as a companion export once the XChaCha core and API shape
are stable, not as a separate implementation project.

### AES-256-GCM-SIV

This is the right AES-family default for new designs.

- Nonce-misuse resistant
- Much better operational story than plain GCM
- Still leverages AES hardware well

### AES-256-GCM

You still need it, but mostly for interop.

- TLS
- ecosystem compatibility
- standards baggage

Do not treat it as the best new default. Treat it as the compatibility workhorse.

### Ascon-AEAD128

This is not optional if the goal is “important standards and the latest NIST world.”

- NIST finalized SP 800-232 in August 2025.
- It is the modern lightweight AEAD standard.
- It fits the pure-Rust, broad-platform, no-FFI ethos well.
- It keeps the Ascon family coherent with `Ascon-Hash256`, `Ascon-XOF128`, and the still-missing `Ascon-CXOF128`.

### AEGIS-256

This is the aggressive watch item.

Official NIST standard? No.
Interesting and current? Yes.

Libsodium’s current guidance says:

- if the CPU has hardware AES acceleration, `AEGIS-256` is the safest choice
- otherwise use `XChaCha20-Poly1305-IETF`

That does not make `AEGIS-256` the first thing to build here. It makes it the thing to evaluate after the core standards portfolio is in place.

## Ship Order

1. `XChaCha20-Poly1305`
2. `AES-256-GCM-SIV`
3. `AES-256-GCM`
4. `Ascon-AEAD128`
5. companion interop profiles (`ChaCha20-Poly1305`, then AES-128 variants if needed)
6. `AEGIS-256` if the project still wants to push the frontier

Reality check as of 2026-04-01:

- item 1 is shipped
- item 2 (`AES-256-GCM-SIV`) is shipped with **x86_64 AES-NI + PCLMULQDQ + VAES-512 + VPCLMULQDQ and aarch64 AES-CE + PMULL hardware dispatch**; wide path processes 4 blocks per iteration on Zen4+/ICL+
- item 3 (`AES-256-GCM`) is shipped with the same hardware dispatch plus wide path; GHASH accelerated via shared `clmul128_reduce` + 4-block `accumulate_4blocks` (GHASH↔POLYVAL bridge preserved)
- item 4 (`Ascon-AEAD128`) is shipped on the portable constant-time baseline with NIST SP 800-232 KAT vectors
- the `ChaCha20-Poly1305` companion from item 5 is also shipped because it falls out of the same core and is too cheap to defer artificially
- item 6 (`AEGIS-256`) is shipped with x86_64 AES-NI + aarch64 AES-CE hardware dispatch (2026-04-01)
- `benches/aead.rs` is live: all 6 shipped primitives benchmarked against RustCrypto competitors
- **The AEAD ship order is complete.**

## Acceleration Gap Status

> **Master tracker:** [`docs/tasks/acceleration.md`](acceleration.md)

### AES-GCM and AES-GCM-SIV: x86_64 + aarch64 hardware-accelerated

**x86_64** dispatches to AES-NI + PCLMULQDQ and **aarch64** dispatches to AES-CE + PMULL,
both at construction time (enum-based `Aes256EncKey` in `aes.rs`, combined `clmul128_reduce`
in `polyval.rs`). Both GCM and GCM-SIV benefit from the same hardware paths. CTR functions
auto-dispatch.

| Platform | AES Block | GHASH/POLYVAL | Status |
|----------|-----------|---------------|--------|
| x86_64 | **AES-NI** (AES-1 ✅) + **VAES-512** (AES-2 ✅) | **PCLMULQDQ** (CLMUL-1 ✅) + **VPCLMULQDQ** (CLMUL-2 ✅) | **Full wide pipeline** |
| aarch64 | **AES-CE** (AES-3 ✅) | **PMULL** (CLMUL-3 ✅) | **Competitive** |
| s390x | Portable — need CPACF/KM (AES-4) | Portable — need VGFM (CLMUL-4) | **340x gap** |
| powerpc64 | Portable — need vcipher (AES-5) | Portable — need VPMSUM (CLMUL-5) | **340x gap** |
| riscv64 | Portable — need Zvkned (AES-6) | Portable — need Zvkg (CLMUL-6) | **340x gap** |
| wasm32 | Portable (expected — no HW AES in wasm) | Portable (expected) | N/A |

**Completed P1 tasks:**
- AES-2 + CLMUL-2: x86_64 VAES-512 + VPCLMULQDQ wide path shipped (2026-03-31). 4-block parallel AES-CTR + 4-block schoolbook-then-reduce GHASH/POLYVAL on Zen4+/ICL+.

### ChaCha20-Poly1305: x86 optimized, powerpc64 + s390x hardware-accelerated

ChaCha20 and Poly1305 have **real** SIMD kernels on x86_64 (AVX2/AVX-512), aarch64 (NEON),
wasm32 (SIMD128), powerpc64 (VSX), and s390x (z/Vector). One platform remains stubbed:

| Platform | ChaCha20 | Poly1305 | Task IDs |
|----------|----------|----------|----------|
| x86_64 | **AVX-512** (`vprold` + 16×16 transpose) / **AVX2** (`vpshufb` + 16×8 transpose) | AVX2/AVX-512 | ✅ CHACHA-4, CHACHA-5 |
| powerpc64 | **VSX** (`vadduwm`/`vxor`/`vrlw`) | **VSX** (`vmulouw`/`vaddudm`) | ✅ CHACHA-1, POLY-1 |
| s390x | **z/Vector** (`vaf`/`vx`/`verll`) | **z/Vector** (`vmlof`/`vag`) | ✅ CHACHA-2, POLY-2 |
| riscv64 | STUB → portable | STUB → portable | CHACHA-3, POLY-3 |

The x86_64 ChaCha20 kernels were optimized (2026-04-01) to close a 0.42-0.69x gap at
4KiB-1MiB on Zen4/Zen5/SPR. The root cause was scalar byte-by-byte XOR after SIMD keystream
generation — replaced with SIMD matrix transpose + load-XOR-store, plus `vprold` (AVX-512)
and `vpshufb` (AVX2) for single-instruction rotations. See CHACHA-4/5 in [`acceleration.md`](acceleration.md).

riscv64 XChaCha20-Poly1305 and ChaCha20-Poly1305 are still running at portable speed.

## SIMD / HW Rules

These are non-negotiable:

- No table-based AES in production code.
- AES paths must be hardware-backed on x86_64 / aarch64 or implemented in constant-time portable code.
- `AES-GCM` and `AES-GCM-SIV` should use CLMUL / PMULL / POLYVAL acceleration where available.
- `XChaCha20-Poly1305` must have a constant-time portable baseline before any SIMD tuning.
- `Ascon-AEAD128` should remain bitsliced / permutation-centric and avoid ornamental dispatch layers.
- In-place APIs are required.
- Detached-tag APIs are required.
- Combined ciphertext+tag APIs are required.
- Nonce types must be explicit: `Nonce96`, `Nonce192`, etc.
- Failure paths must be uniform and constant-time as far as practical.
- Every accelerated backend must differential-test against the portable backend and official vectors.

## Acceleration Posture

As of 2026-03-28, the state of the art is not “accelerate every AEAD on every ISA equally.”

The right rule is:

- reuse real hardware support where the primitive structurally matches it
- keep a constant-time portable baseline for every AEAD
- do not force SIMD paths that lose to scalar or bitslice code
- treat `wasm32` compatibility and `wasm32` acceleration as separate questions

### Per-Primitive Reality

#### XChaCha20-Poly1305

This should be the software-first AEAD, and that has consequences:

- `no_std`: yes
- `wasm32`: yes
- `x86_64`: `AVX2` first, `AVX-512` only if it clearly wins end-to-end
- `aarch64`: `NEON` first
- `wasm32` / `wasm64`: portable baseline, then `simd128` if it stays honest end-to-end
- `powerpc64` / `s390x` / `riscv64`: explicit vector routes are allowed, but only portable-first claims count until benchmarked

Do not overfit this to AES-heavy hardware. The point of `XChaCha20-Poly1305` is exactly that it stays strong where AES hardware is absent or awkward.

#### AES-256-GCM-SIV and AES-256-GCM

These should be hardware-first, not software-first:

- `x86_64`: `AES-NI` + `PCLMULQDQ`, with `VAES` + `VPCLMULQDQ` where it materially improves throughput
- `aarch64`: AES instructions + `PMULL`
- `s390x`: CPACF / hardware AES-GCM class instructions should be a serious target if AES AEAD enters scope
- `powerpc64`: only if VSX / crypto backends can be shown to win and stay maintainable
- `riscv64`: portable first until the Rust and hardware ecosystem for vector / AES extensions is solid enough
- `wasm32`: compatibility yes, “hardware AES” no; portable constant-time fallback only

For AES-family AEADs, the acceleration story is only worth doing if it is genuinely hardware-backed or constant-time bitsliced/fixsliced. No lookup-table AES.

#### Ascon-AEAD128

This is the lightweight / broad-platform AEAD.

- `no_std`: yes
- `wasm32`: yes
- portable bitslice / permutation-first baseline
- `x86_64`: `AVX2` / `AVX-512` only if measured against the portable permutation
- `aarch64`: `NEON` only if measured against the portable permutation
- avoid building a dispatch tower that costs more than the permutation work itself

Ascon is exactly the kind of primitive where ornamental SIMD is a trap.

#### AEGIS-256

This is only interesting if the CPU has strong AES hardware.

- `x86_64`: yes, high-priority if `AES-NI` / `VAES` are available
- `aarch64`: yes, high-priority if ARM AES instructions are available
- `s390x`: maybe, if the hardware AES path is clean and measurable
- no AES hardware: do not prioritize it ahead of `XChaCha20-Poly1305`

That matches current operational guidance from libsodium more than it matches standards politics.

### New Research / Current Pressure

Recent work strengthens a few choices here:

- `AES-GCM` is still the interop workhorse, but recent collision-focused analysis reinforces that nonce discipline is not optional
- `AES-GCM-SIV` remains the more robust AES-family default for new designs because misuse resistance is worth real operational value
- verified SIMD work has continued to make `ChaCha20-Poly1305` on `AVX2` / `AVX-512` / `NEON` a very credible fast software path
- NIST finalized `Ascon-AEAD128` in SP 800-232, so “lightweight but standard” is no longer hypothetical
- recent embedded work on fixsliced / bitsliced `AES-GCM` reinforces the fallback rule: if hardware AES is absent, use constant-time software, not tables

## AEAD Target Matrix

This is the release bar the AEAD subsystem should aim at.

| Primitive | `no_std` | `wasm32` | `x86_64` | `aarch64` | `powerpc64` | `s390x` | `riscv64` |
|-----------|----------|----------|----------|-----------|-------------|---------|-----------|
| `XChaCha20-Poly1305` | yes | yes | portable, then `AVX2`, then `AVX-512` if it wins | portable, then `NEON` | portable first, explicit vector route | portable first, explicit vector route | portable first, explicit vector route |
| `AES-256-GCM-SIV` | yes | yes | `AES-NI` / `PCLMUL`, then `VAES` / `VPCLMUL` | AES + `PMULL` | only if proven | hardware AES/GCM if clean | portable first |
| `AES-256-GCM` | yes | yes | `AES-NI` / `PCLMUL`, then `VAES` / `VPCLMUL` | AES + `PMULL` | only if proven | hardware AES/GCM if clean | portable first |
| `Ascon-AEAD128` | yes | yes | portable first, SIMD only if measured | portable first, SIMD only if measured | portable first | portable first | portable first |
| `AEGIS-256` | yes | yes | only worth doing on AES hardware | only worth doing on AES hardware | low priority | maybe | low priority |

## API Surface

The crate should expose both:

- low-level reusable cipher types
- one-shot convenience helpers

Required minimum shapes:

- `encrypt`
- `decrypt`
- `encrypt_in_place`
- `decrypt_in_place`
- `seal` / `open` aliases only if they map onto the exact same semantics
- detached-tag variants
- typed key and nonce wrappers

## What Not To Do

- Do not start with three AEADs in parallel.
- Do not ship portable AES that leaks through tables.
- Do not hide nonce discipline under vague `&[u8]` APIs.
- Do not couple API design to one external trait crate forever.
- Do not promise `wasm32` “hardware acceleration” for AES-family AEADs when the environment does not provide it.
- Do not build `POWER` / `s390x` / `RISC-V` bespoke backends just to fill a table unless the benchmarks justify the maintenance cost.

## Sources

- RFC 8452: AES-GCM-SIV
  https://datatracker.ietf.org/doc/html/rfc8452
- RFC 8439: ChaCha20-Poly1305
  https://datatracker.ietf.org/doc/rfc8439/
- Libsodium ChaCha20 / XChaCha20 guidance
  https://doc.libsodium.org/secret-key_cryptography/aead/chacha20-poly1305
- NIST SP 800-38D: GCM
  https://csrc.nist.gov/pubs/sp/800/38/d/final
- NIST SP 800-232: Ascon-AEAD128 and Ascon family
  https://csrc.nist.gov/pubs/sp/800/232/final
- NIST Lightweight Cryptography project
  https://csrc.nist.gov/projects/lightweight-cryptography
- Libsodium AEAD guidance
  https://doc.libsodium.org/secret-key_cryptography/aead
- HACLxN: verified SIMD crypto, including vectorized ChaCha20-Poly1305
  https://eprint.iacr.org/2020/572.pdf
- EverCrypt: verified cross-platform AES-GCM and dispatch-oriented crypto provider
  https://eprint.iacr.org/2019/757
- Recent collision analysis pressure on AES-GCM nonce practice
  https://eprint.iacr.org/2024/1111
- Recent constant-time embedded AES-GCM optimization work
  https://eprint.iacr.org/2025/512
- Recent AEGIS parallelism work
  https://eprint.iacr.org/2023/523
