# PQE / PQC Roadmap

## Recommended Set

For the first post-quantum wave, focus on:

- `ML-KEM`
- `ML-DSA`
- `SLH-DSA`

Track closely:

- `HQC`
- `FN-DSA`
- hybrid `TLS 1.3`
- PQ / hybrid `HPKE`

## Current State

As of 2026-03-24, the official NIST baseline is:

- `FIPS 203` — `ML-KEM`
- `FIPS 204` — `ML-DSA`
- `FIPS 205` — `SLH-DSA`
- `SP 800-227` — KEM guidance

NIST has also:

- selected `HQC` on 2025-03-11 as a fourth-round KEM for future standardization
- published draft migration guidance in `NIST IR 8547`

So the serious answer is no longer “wait and see.” The standards exist now.

## What To Build First

### 1. ML-KEM

This is the first PQC primitive that matters here.

- KEMs are the practical bridge into real protocols.
- `ML-KEM-768` is the center of gravity for general deployment.
- `ML-KEM-512` and `ML-KEM-1024` can follow the standard profile set.

If `rscrypto` ever grows a public-key encryption / key-establishment layer, this is where it starts.

### 2. ML-DSA

This is the practical PQ signature to pair with ML-KEM.

- Lattice-based
- NIST-finalized
- good “modern standard” position

### 3. SLH-DSA

This should exist, but not as the first signature shipped.

- It is slower and larger.
- It is the conservative hash-based backup.
- It matters for diversity and high-assurance use cases.

## What To Watch, Not Ship First

### HQC

Important, but not first.

- NIST selected it in 2025 as the second KEM track.
- It is not yet the immediate first thing to build ahead of ML-KEM.

### FN-DSA

This is the signature watch item.

- It is active in NIST’s additional signature work.
- It is not a finalized baseline yet.
- Track it, do not let it destabilize the first PQ signature portfolio.

### Hybrid TLS and HPKE

These are the protocol destinations, not the first primitive implementations.

- hybrid TLS 1.3 is still draft work
- PQ and hybrid HPKE are active draft work

That means:

- build primitives first
- build hybrid composition APIs second
- build full protocol glue last

## SIMD / HW Rules

PQC changes the implementation game. The rules still apply:

- portable constant-time baseline first
- no secret-dependent memory access
- no “it only works fast on AVX2” design shortcuts
- vectorization must be an optimization layer, never the algorithm boundary

Hot-path guidance:

- x86_64: expect `AVX2` to matter for NTT / polynomial arithmetic first
- aarch64: expect `NEON` and later `SVE2` to matter
- keep scalar reference code readable enough to audit
- isolate packing / NTT / sampling / reduction kernels cleanly
- benchmark cache behavior and layout, not just instruction count

For PQC especially:

- deterministic test vectors are mandatory
- cross-check against official test vectors is mandatory
- serialization / encoding tests are mandatory
- malformed-input behavior must be explicit and tested

## Build Order

1. `ML-KEM`
2. `ML-DSA`
3. hybrid composition helpers around classical + `ML-KEM`
4. `SLH-DSA`
5. `HQC`
6. protocol-layer hybrid `HPKE` / `TLS` helpers only after the primitive layer is solid

## Strong Opinion

Do not bolt PQC into `rscrypto` as a sidecar curiosity.

If this project takes PQC on, it should do it the same way it is approaching hashing:

- portable first
- accelerated where it matters
- benchmarked on real CPUs
- aggressively tested
- no dependency tower

But do not try to ship every PQ candidate. That would be a mess.

## Sources

- NIST FIPS 203: ML-KEM
  https://csrc.nist.gov/pubs/fips/203/final
- NIST FIPS 204: ML-DSA
  https://csrc.nist.gov/pubs/fips/204/final
- NIST FIPS 205: SLH-DSA
  https://csrc.nist.gov/pubs/fips/205/final
- NIST SP 800-227: KEM guidance
  https://csrc.nist.gov/pubs/sp/800/227/final
- NIST PQC project page
  https://csrc.nist.gov/projects/post-quantum-cryptography
- NIST IR 8547 IPD: transition guidance
  https://csrc.nist.gov/pubs/ir/8547/ipd
- NIST HQC selection
  https://csrc.nist.gov/News/2025/hqc-announced-as-a-4th-round-selection
- NIST approval of FIPS 203 / 204 / 205
  https://csrc.nist.gov/News/2024/postquantum-cryptography-fips-approved
- Hybrid TLS 1.3 draft
  https://datatracker.ietf.org/doc/html/draft-ietf-tls-hybrid-design
- PQ / hybrid HPKE draft
  https://datatracker.ietf.org/doc/html/draft-ietf-hpke-pq
