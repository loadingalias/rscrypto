# ML-KEM Performance Ledger

Internal task ledger for ML-KEM performance work. This is not release copy, not a public speed claim, and not a
roadmap promise.

Last updated: 2026-06-21.

## Position

We are not winning on AWS Graviton3 or AWS Graviton4 yet.

The current Linux aarch64 path has real rscrypto-owned assembly for forward NTT, inverse NTT, SampleNTT rejection
parsing, and K2/K3/K4 basemul accumulation. The latest full ML-KEM G3/G4 bench run still loses to `aws-lc-rs` on every
top-level keygen, encapsulate, and decapsulate row.

The important correction: standalone basemul dispatch was not the wall. The latest K2/K3/K4 Linux basemul production
dispatch moved end-to-end results by noise-level amounts only. More standalone basemul reshuffling is likely another
loop unless a phase bench proves otherwise.

## Latest Evidence

Source run:

- GitHub Actions run: `27915624820`.
- Workflow: `Bench`.
- Commit: `efdfce89bfba91c27be446ac5466b64572088b7a`.
- Inputs: `targets=ml-kem`, `platforms=g3,g4`, `quick=false`.
- Result: workflow succeeded on both AWS Graviton3 and AWS Graviton4.

Midpoint times, `rscrypto` versus `aws-lc-rs`, lower is better:

| Platform | Operation | rscrypto | aws-lc-rs | Gap |
|---|---|---:|---:|---:|
| G3 | ML-KEM512 keygen | 12.635 us | 8.564 us | +47.5% |
| G3 | ML-KEM512 encapsulate | 11.418 us | 10.234 us | +11.6% |
| G3 | ML-KEM512 decapsulate | 15.423 us | 11.680 us | +32.0% |
| G3 | ML-KEM768 keygen | 16.701 us | 13.953 us | +19.7% |
| G3 | ML-KEM768 encapsulate | 15.899 us | 15.833 us | +0.4% |
| G3 | ML-KEM768 decapsulate | 21.271 us | 18.153 us | +17.2% |
| G3 | ML-KEM1024 keygen | 27.736 us | 19.851 us | +39.7% |
| G3 | ML-KEM1024 encapsulate | 26.295 us | 23.496 us | +11.9% |
| G3 | ML-KEM1024 decapsulate | 33.204 us | 26.366 us | +25.9% |
| G4 | ML-KEM512 keygen | 10.545 us | 7.166 us | +47.2% |
| G4 | ML-KEM512 encapsulate | 9.437 us | 8.571 us | +10.1% |
| G4 | ML-KEM512 decapsulate | 12.670 us | 9.693 us | +30.7% |
| G4 | ML-KEM768 keygen | 14.644 us | 11.584 us | +26.4% |
| G4 | ML-KEM768 encapsulate | 13.577 us | 13.256 us | +2.4% |
| G4 | ML-KEM768 decapsulate | 18.033 us | 15.113 us | +19.3% |
| G4 | ML-KEM1024 keygen | 23.864 us | 16.307 us | +46.3% |
| G4 | ML-KEM1024 encapsulate | 22.252 us | 19.055 us | +16.8% |
| G4 | ML-KEM1024 decapsulate | 28.056 us | 22.502 us | +24.7% |

Versus the prior commit `1b3781d`, the latest commit was neutral:

- G3 end-to-end movement ranged from about -0.5% to +0.2%.
- G4 end-to-end movement ranged from about -0.9% to +0.1%.
- That is not a real win. Treat it as noise until a repeated run says otherwise.

## Tried Work

### Correctness and promotion gates

- Proved aarch64 forward NTT assembly against the scalar oracle.
- Added ML-KEM promotion gating on Graviton evidence.
- Added Linux aarch64 gate coverage for NTT, basemul, ACVP/FIPS 203 vectors, operation tests, property tests, fuzz
  corpus replay, and benchmark gates.
- Added scalar-oracle coverage for full forward NTT, inverse NTT paths, basemul accumulation, and the latest K2
  basemul path.
- Added emitted-assembly scans for constant-time evidence. The current aarch64 release artifact did not contain
  division-family instructions in the checked scope.

Relevant commits:

- `ad23fa5` - prove aarch64 ML-KEM NTT asm against scalar oracle; gate ML-KEM promotion on Graviton evidence.
- `766132e` - include inverse add asm in aarch64 gate filter.
- `efdfce8` - dispatch Linux aarch64 ML-KEM basemul assembly.

### Forward NTT

- Made raw aarch64 NTT assembly exact against the scalar/FIPS path.
- Fixed canonicalization.
- Dispatched Linux aarch64 NTT assembly.
- Routed Apple/aarch64 through NEON earlier; Apple is not the current G3/G4 blocker.

Relevant commits:

- `05a56da` - route aarch64 ML-KEM NTT through NEON.
- `25a367e` - make aarch64 ML-KEM NTT asm exact.
- `67cee6d` - dispatch Linux aarch64 ML-KEM NTT asm.
- `1e41083` - fix aarch64 ML-KEM NTT canonicalization.

### Inverse NTT

- Added Linux aarch64 inverse NTT diagnostics.
- Fused inverse NTT add.
- Added and reverted one inverse final-scale fusion attempt.
- Precomputed inverse reducers.
- Unrolled the inverse NTT assembly schedule.
- Tightened inverse final scale.

Relevant commits:

- `ba59869` - fuse aarch64 ML-KEM inverse NTT add.
- `11afb4a` - fuse aarch64 ML-KEM inverse NTT final scale.
- `5c79304` - revert inverse final scale fusion.
- `133b8b6` - add Linux aarch64 ML-KEM inverse NTT asm diagnostics.
- `7271015` - precompute aarch64 ML-KEM inverse reducers.
- `2d725d0` - unroll aarch64 ML-KEM inverse NTT asm.
- `551f1b6` - tighten aarch64 ML-KEM inverse final scale.

### Sampling, rejection, and SHAKE feed

- Added aarch64 SampleNTT NEON extraction.
- Parsed ML-KEM samples and sample tails directly from XOF state.
- Kept SampleNTT candidates in registers.
- Added a Linux aarch64 rejection parser.
- Compacted aarch64 rejection lanes.
- Fed fused ML-KEM products from compact rejection.
- Added a quad rejection parser and reverted it.
- Added aarch64 triple SHAKE scheduling for ML-KEM sampling.

Relevant commits:

- `c7c2d89` - add aarch64 ML-KEM SampleNTT NEON extractor.
- `56e234a` - parse aarch64 ML-KEM quad samples from XOF state.
- `75b1551` - parse aarch64 ML-KEM sample tails from XOF state.
- `5a94fbe` - keep aarch64 ML-KEM sample candidates in registers.
- `951be2c` - add aarch64 ML-KEM rejection parser.
- `13a50c8` - compact aarch64 ML-KEM rejection lanes.
- `a05a23b` - feed fused ML-KEM products from compact aarch64 rejection.
- `2f00fa7` - add aarch64 triple SHAKE path for ML-KEM sampling.
- `9708c1d` - compact ML-KEM fused rejection into NEON chunks.
- `a2620a5` - add aarch64 ML-KEM quad rejection parser.
- `133627a` - revert the quad rejection parser.

### Product-domain conversion and reduction

- Vectorized product-domain conversion.
- Tightened product-domain reduction.

Relevant commits:

- `c665306` - vectorize aarch64 ML-KEM product-domain conversion.
- `bf5bfe5` - tighten aarch64 ML-KEM product-domain reduction.

### Basemul and dot-product accumulation

- Added owned basemul diagnostics.
- Added fused basemul diagnostics.
- Tightened the aarch64 basemul assembly schedule.
- Added fused K-way accumulation.
- Rescheduled accumulation and reverted one schedule.
- Added K2 row-dot path.
- Ported K3 and K4 basemul schedules.
- Inlined aarch64 fused chunk multiply.
- Added production Linux aarch64 K2/K3/K4 basemul assembly dispatch.

Relevant commits:

- `cbd32a8` - add owned aarch64 ML-KEM basemul diagnostics.
- `67caab7` - add fused aarch64 ML-KEM basemul diagnostics.
- `c95d99f` - tighten aarch64 ML-KEM basemul asm schedule.
- `95ea2e5` - fuse aarch64 ML-KEM K-way accumulate.
- `e28eaee` - add aarch64 K2 ML-KEM row-dot path.
- `6573866` - reschedule aarch64 ML-KEM basemul accumulation.
- `5bd2052` - revert basemul accumulation reschedule.
- `abcf8d8` - port aarch64 ML-KEM k4 basemul schedule.
- `2d65c61` - port aarch64 ML-KEM k3 basemul schedule.
- `1b3781d` - inline aarch64 ML-KEM fused chunk multiply.
- `efdfce8` - dispatch Linux aarch64 ML-KEM basemul assembly.

Evidence from `27915624820`: this category did not move the end-to-end G3/G4 result.

### PKE and matrix paths

- Parsed fused products from XOF state.
- Dispatched a fused ML-KEM1024 PKE matrix path.
- Added existing phase benches for matrix sample, arithmetic, PKE phases, and decapsulation phases.

Relevant commits:

- `75b1551` - parse aarch64 ML-KEM sample tails from XOF state.
- `1e5bfa8` - parse fused aarch64 ML-KEM products from XOF state.
- `369e422` - dispatch fused ML-KEM1024 PKE matrix path.

## What This Means

The latest evidence rules out a simple explanation:

- It is not enough to "enable the asm." We already did that for the obvious Linux aarch64 NTT and basemul pieces.
- It is not enough to tweak standalone basemul. The latest K2/K3/K4 production assembly dispatch was neutral.
- It is not enough to optimize only ML-KEM768 encapsulation. That row is close, but keygen and decapsulate still lose
  hard, and ML-KEM512/1024 keygen are the worst gaps.
- The largest losses are keygen and decapsulation, so the missing work is probably in full-pipeline shape: matrix
  sampling, NTT scheduling around sampling, byte packing/compression/decompression, cached product layout, or avoidable
  materialization between phases.

## Do Not Repeat Without New Evidence

- Do not do another standalone basemul schedule change unless `mlkem-arithmetic` proves basemul is the top remaining
  phase on G3 or G4.
- Do not reintroduce the quad rejection parser without a scalar oracle and a phase bench win. It was already reverted.
- Do not promote diagnostics-only assembly just because it compiles. Promotion requires scalar-oracle equivalence,
  aarch64 Linux execution, CT review, and bench movement.
- Do not use Apple results to predict Graviton results. Apple is already in better shape and has different scheduling
  behavior.
- Do not run only top-level `ml-kem` benches after each micro-change. They are useful for final verdicts, not for
  choosing the next kernel.

## Still Open, Not Yet Exhausted

These are the remaining non-duplicate low-level directions.

1. Run G3/G4 phase benches for the current commit:
   - `mlkem-matrix-sample`
   - `mlkem-arithmetic`
   - `mlkem-pke-phases`
   - `mlkem-decap-phases`

   The output must rank phase cost before more assembly is written.

2. Build a direct phase map against AWS-LC:
   - keygen: matrix expansion, secret/error sampling, NTTs, matrix-vector product, encode/decode work.
   - encapsulate: public-key decode, matrix/vector work, inverse NTT, compression/encoding.
   - decapsulate: ciphertext decode/decompress, secret dot product, inverse NTT, re-encryption compare.

   The point is to identify missing whole-pipeline fusion, not to copy implementation.

3. Add aarch64-native packing/compression/decompression if phase benches justify it.

   Current code has fused scalar Rust encode/decode/compress helpers and s390x vector compression helpers. There is no
   comparable aarch64 NEON/ASM compression and byte-packing backend for D=10, D=11, D=4, D=5, or D=12. This is a real
   untried gap, especially for encapsulate and decapsulate.

4. Evaluate a real cached-multiply representation.

   AWS-LC-class ML-KEM paths usually avoid paying the same product setup repeatedly. Our latest work made K2/K3/K4
   basemul assembly callable in production, but it did not introduce a first-class cached product layout that changes
   the surrounding algorithm. If phase benches point at arithmetic, the next move is cached layout plus fused consumers,
   not another local basemul loop shuffle.

5. Split G3 and G4 scheduling decisions only after a phase or instruction-level profile proves divergence.

   One generic Linux aarch64 schedule may not be optimal for both Neoverse V1-class and V2-class cores. Separate
   schedules add maintenance cost and should be justified by a measured difference.

## Next Command Set

Use the phase selectors before writing another kernel:

```bash
gh workflow run Bench --ref main -f targets=mlkem-matrix,mlkem-arith,mlkem-pke,mlkem-decap -f platforms=g3,g4 -f quick=false
```

After it completes, extract the artifacts and compare:

```bash
gh run download <run-id> --repo LoadingALIAS/rscrypto --dir /tmp/rscrypto-mlkem-phase
rg -n "mlkem-(matrix-sample|arithmetic|pke-phases|decap-phases)" /tmp/rscrypto-mlkem-phase/benchmark-*/results.txt
```

If phase benches still do not isolate the wall, use Linux profiling on a native/remote host:

```bash
just ssh-linux
# On the host:
cargo bench --profile bench --features parallel,hmac,hkdf,pbkdf2,ecdsa,ed25519,x25519,ml-kem,diag --bench auth -- '^mlkem'
```

## Decision Rule

The next accepted kernel must satisfy all of this:

1. A phase bench identifies the target as a top contributor on G3 or G4.
2. The kernel has a scalar/FIPS oracle test for exact output.
3. The memory schedule is fixed with respect to secret values.
4. The aarch64 Linux CI gate executes the path, not just cross-compiles it.
5. Full G3/G4 `ml-kem` benches show a real end-to-end win, not noise.
