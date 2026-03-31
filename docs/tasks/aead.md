# AEAD Roadmap

## Current Status (2026-03-30)

`XChaCha20-Poly1305` is shipped on the typed AEAD surface.
The plain `ChaCha20-Poly1305` interop companion is shipped on the same core.
The ChaCha-family core now has explicit backend routing for `x86_64` (`portable -> AVX2 -> AVX-512`), `aarch64` (`portable -> NEON`), `wasm32/wasm64` (`portable -> simd128`), and explicit `s390x` / `powerpc64` / `riscv64` vector routes.
`Poly1305` now follows the same primitive-keyed dispatch contract, with architecture-specific block kernels on `x86_64` (`AVX2` / `AVX-512`), `aarch64` (`NEON`), and `wasm32` (`simd128`), plus explicit fallback routes everywhere else.

`AES-256-GCM-SIV` is shipped on the portable constant-time baseline:
- AES-256: algebraic S-box (Fermat x^254 in GF(2^8)), no lookup tables
- POLYVAL: Pornin bmul64 (16 integer muls/call), Karatsuba 128×128, 2-pass Montgomery reduction
- Full RFC 8452 construction: key derivation, POLYVAL tag, AES-CTR
- Verified against RFC 8452 Appendix A (POLYVAL) and Appendix C.2 (AES-256-GCM-SIV) test vectors
- Hardware acceleration (AES-NI + PCLMULQDQ, ARM AES + PMULL) is the next step for this primitive

Still missing from the primary rollout:

- `AES-256-GCM`
- `Ascon-AEAD128`
- `AEGIS-256`

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

Reality check as of 2026-03-30:

- item 1 is shipped
- item 2 (`AES-256-GCM-SIV`) is shipped on the portable constant-time baseline (algebraic AES S-box, Pornin bmul64 POLYVAL, RFC 8452 test vectors passing); hardware acceleration (AES-NI, PCLMULQDQ, ARM AES+PMULL) is the next step
- the `ChaCha20-Poly1305` companion from item 5 is also shipped because it falls out of the same core and is too cheap to defer artificially
- items 3, 4, and 6 remain open

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
