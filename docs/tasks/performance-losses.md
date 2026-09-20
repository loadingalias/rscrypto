# Cross-target performance losses

## Outcome

Explain and close the catastrophic P-256 and ECDSA losses measured on RISC-V
and IBM Z in benchmark run [#34874736834](https://github.com/loadingalias/rscrypto/actions/runs/34874736834).
Build a reusable CI-only profiling path first
because these machines are not available through an interactive shell.

Do not treat the ten rows as ten optimizations.
They strongly suggest shared field, scalar, inversion, table-selection, or generated-code costs.
One demonstrated cause should drive one bounded production change.

This task is complete when:

- the exact workloads have native CI profiles with attributable symbols and retained raw evidence;
- each shared cause is demonstrated rather than inferred from portable source;
- corrected cases reach the repository's `0.95x` tie boundary,
  or a reviewed equivalent-work analysis records why a remaining comparison is not a valid target;
  and
- the complete benchmark matrix and applicable correctness, CT, cleanup, dispatch,
  and target evidence pass on the final revision.

## Retained baseline

- Source: `ae6f54afedaa652858fd2bcbd8f56f339e663a4f` on `main`.
- Measurement date: 2026-09-14.
- Toolchain: `rustc 1.99.0-nightly (3d6c19bb9 2026-08-11)`.
- Profile: repository `bench` profile.
- Criterion: 20 samples, 100 ms warm-up, 400 ms measurement, 10,000 resamples, 95% confidence.
- Artifacts: `bench-s390x-linux-34874736834-1` and `bench-riscv64-linux-34874736834-1`.
- Ratio: fastest equivalent external time divided by rscrypto time; lower is worse.

| Rank | Target | Exact rscrypto case | Rscrypto | Fastest external | Ratio |
| ---: | --- | --- | ---: | --- | ---: |
| 1 | IBM Z | `p256-ecdh/public-key/rscrypto-selected` | 9.627 ms | `crrl-pure-rust` | 0.00750x |
| 2 | IBM Z | `p256-ecdh/agreement/rscrypto-selected` | 19.075 ms | `crrl-pure-rust` | 0.00924x |
| 3 | RISC-V | `p256-ecdh/public-key/rscrypto-selected` | 13.584 ms | `crrl-pure-rust` | 0.00992x |
| 4 | RISC-V | `p256-ecdh/agreement/rscrypto-selected` | 30.860 ms | `crrl-pure-rust` | 0.01076x |
| 5 | RISC-V | `ecdsa-p384/public-key/rscrypto-blinded` | 159.707 ms | `rustcrypto-p384` | 0.01540x |
| 6 | IBM Z | `ecdsa-p384/public-key/rscrypto-blinded` | 96.934 ms | `rustcrypto-p384` | 0.01560x |
| 7 | IBM Z | `p256-ecdh/parse/rscrypto` | 22.726 us | `crrl-pure-rust` | 0.01577x |
| 8 | RISC-V | `ecdsa-p256/public-key/rscrypto-blinded` | 36.110 ms | `rustcrypto-p256` | 0.01638x |
| 9 | RISC-V | `ecdsa-p384/sign/rscrypto-deterministic/0` | 50.215 ms | `aws-lc-rs` | 0.01702x |
| 10 | RISC-V | `ecdsa-p384/sign/rscrypto-deterministic/32` | 50.220 ms | `aws-lc-rs` | 0.01710x |

The retained confidence intervals are narrow compared with these 58x-133x gaps.
The measurements establish a severe cost, not its cause.

## Phase 1 — Native profile evidence (2026-09-19)

The manual `profile.yml` accepts one architecture and one curated primitive.
The catalog maps that pair to one exact Criterion case; capture lasts five seconds.
Preparation cross-builds and seals the optimized benchmark with debug information.
The native runner verifies and executes that binary without rebuilding it.
Each run retained the binary, source-bound manifest, host/capability facts,
raw `perf.data` when capture started, text output, and final status.
The downloaded binary hashes match the manifests.
The initial six runs used `a314ea81dd615e40d925efd5c05a5be5e719cb4f`; the successful POWER rerun used `23dc2f703680257de0f5fc9fd766dc320b96da63`.

| Architecture | Exact case | Workflow | Native evidence |
| --- | --- | --- | --- |
| Intel x86-64 | `aes-128-gcm-siv/copy-and-encrypt/rscrypto/32` | [success](https://github.com/loadingalias/rscrypto/actions/runs/35411075950) | [499 samples, 0 lost; DWARF](https://github.com/loadingalias/rscrypto/actions/runs/35411075950/artifacts/10574468080) |
| AMD x86-64 | `aes-siv-cmac-256/construct/rscrypto` | [success](https://github.com/loadingalias/rscrypto/actions/runs/35411075977) | [533 samples, 0 lost; DWARF](https://github.com/loadingalias/rscrypto/actions/runs/35411075977/artifacts/10573767508) |
| AArch64 | `argon2id-owasp/salt16-raw32/rscrypto/m=19MiB_t=2_p=1` | [success](https://github.com/loadingalias/rscrypto/actions/runs/35411075935) | [513 samples, 0 lost; DWARF](https://github.com/loadingalias/rscrypto/actions/runs/35411075935/artifacts/10574627850) |
| POWER | `blake3/keyed/rscrypto/64` | [success](https://github.com/loadingalias/rscrypto/actions/runs/35415762447) | [414 samples, 0 lost; flat symbols](https://github.com/loadingalias/rscrypto/actions/runs/35415762447/artifacts/10575707870) |
| IBM Z | `p256-ecdh/public-key/rscrypto-selected` | [failed](https://github.com/loadingalias/rscrypto/actions/runs/35411075962) | [perf access denied; no report](https://github.com/loadingalias/rscrypto/actions/runs/35411075962/artifacts/10574103643) |
| RISC-V | `p256-ecdh/public-key/rscrypto-selected` | [success](https://github.com/loadingalias/rscrypto/actions/runs/35411076148) | [497 samples, 0 lost; flat symbols](https://github.com/loadingalias/rscrypto/actions/runs/35411076148/artifacts/10574449230) |

The [initial POWER run](https://github.com/loadingalias/rscrypto/actions/runs/35411075973) lost all 347 DWARF samples yet incorrectly passed with a header-only report.
The probe had accepted a nonempty `perf.data` file without checking for a decodable sample;
the final report check had not required a positive sample count and symbol row.
The repaired probe rejected the DWARF capture after it lost all eight probe samples,
selected flat sampling, and retained 414 attributable `cycles:u` samples with none lost on POWER10 (`6.12.0-264.el10.ppc64le`, `perf 6.8.12`).
The downloaded benchmark binary's SHA-256 matches the sealed manifest.

IBM Z's verified binary and exact-case discovery succeeded,
but all native `perf` probes were denied with `perf_event_paranoid=4` and no effective capabilities.
The runner owner must grant `CAP_PERFMON` or agree to a host configuration that permits user-space sampling.
Do not change donated-runner security settings in the workflow.
Rerun the same case after the owner makes sampling available.

`perf` cycles samples locate CPU time;
they do not measure elapsed speedups or prove why an external implementation is faster.
DWARF report percentages are inclusive unless marked self and must not be added.
The RISC-V and POWER collectors have only flat symbols, so they cannot attribute callers.
These five-second reports include Criterion warm-up as well as the timed profile loop.
Use the exact benchmark and target-native correctness/constant-time evidence for changes.

## Per-architecture optimization targets

The four additional selections below are the largest equivalent-work losses
for their Linux architectures in the [September benchmark campaign](https://github.com/loadingalias/rscrypto/actions/runs/34874736834).
These are historical baselines at `ae6f54af`; the profiles used the later revision above.
A sampled hotspot narrows an investigation but does not close a benchmark loss.

| Architecture | Historical rscrypto / external median | Ratio | First production target |
| --- | --- | ---: | --- |
| Intel | AES-128-GCM-SIV 32 B: 203.16 / 91.1 ns (AWS-LC) | 0.449x | Short-message tag, key schedule, and CTR path |
| AMD | AES-SIV-CMAC-256 construction: 77.8 / 30.6 ns (RustCrypto) | 0.393x | AES-128 key expansion in context construction |
| AArch64 | Argon2id OWASP: 21.965 / 11.392 ms (RustCrypto) | 0.519x | NEON block compression |
| POWER | keyed BLAKE3 64 B: 229.4 / 115.8 ns (official BLAKE3) | 0.505x | Portable one-shot digest and compression codegen |

### RISC-V — P-256 public derivation

The flat `cycles:u` report attributes 97.14% self to `p256_portable::ct_mul_u64_wide` and 1.62% to `montgomery_mul`.
The selected P-256 field path calls this deliberately fixed-work 64-by-64-bit multiply through
Montgomery arithmetic.
Its 64 conditional-add rounds per product explain
where this production operation spends its sampled cycles;
the profile does not establish the fraction of the gap against CRRL caused by that choice.
Inspect target assembly and CT evidence for a faster fixed-work field representation/multiply,
then compare exact public derivation and agreement on native RISC-V.
Do not replace it with target multiplication without proving secret-independent latency
and preserving the CT contract.

### IBM Z — P-256 public derivation

No sampled hot path exists yet.
The same fixed-work multiply is selected by `cfg(target_arch = "s390x")`, making it the first hypothesis,
not a measured IBM Z finding.
Resolve native perf access, capture the exact public-key case,
and check whether field multiplication dominates before sharing a RISC-V fix.
The existing public-key/agreement/parse rows remain open.

### Intel x86-64 — AES-128-GCM-SIV, 32-byte seal

The profile attributes 35.65% inclusive to `compute_tag_wide`
(including 15.41% self in `polyval::pclmul::clmul128_reduce`),
23.67% inclusive to `aes128_expand_key`, 20.63% self to wide CTR encryption, and 12.10% inclusive to `derive_keys`.
Per-nonce subkeys must still be derived by the algorithm.
Inspect generated code and isolate tag/POLYVAL
and per-message key expansion on the real 32-byte seal path;
compare the same complete encryption contract against AWS-LC
before choosing a bounded production change.

### AMD x86-64 — AES-SIV-CMAC-256 construction

`AesSivCmac256::new` accounts for 81.27% inclusive and `aes128_expand_key` for 72.00% inclusive.
Construction expands both 16-byte key halves and derives CMAC subkeys.
Several large libc addresses lack function names;
do not label them cleanup or copying without symbol evidence.
Inspect both AES expansions, CMAC setup, and destruction on the exact construct/destroy workload;
compare equivalent RustCrypto construction on the same AMD host.

### AArch64 — Argon2id OWASP profile

`argon2::aarch64::compress_neon` accounts for 95.23% inclusive (94.02% self) inside `fill_segment_inner`; matrix cleanup is 2.22% self.
The workload uses 19 MiB, two passes, one lane, a 16-byte salt, and 32-byte raw output.
Inspect the production NEON compression's generated rounds, spills, and memory traffic,
then compare equivalent full hashes against RustCrypto on the same host.
Do not reduce Argon2 work factors to improve this ratio.

### POWER — keyed BLAKE3, 64 bytes

The exact-case [native report](https://github.com/loadingalias/rscrypto/actions/runs/35415762447/artifacts/10575707870) retained 414 flat `cycles:u` samples with no loss.
It assigns 50.49% to `digest_public_oneshot`, 25.45% to `compress`, 11.86% to `digest_oneshot_words`, and 11.56% to Criterion's `Bencher::iter`.
These are symbol-level samples, not call-chain percentages: inlined work may be charged to `digest_public_oneshot` or `digest_oneshot_words`.
Do not interpret 50.49% as dispatch overhead; inlined compression may be charged to that symbol.

The selected benchmark calls the production `Blake3::keyed_digest` on 64 bytes.
Diagnostics select the portable kernel on POWER;
this input takes the tiny one-block path through `hash_tiny_to_root_words` and `compress_chunk_tail_to_root_words`.
First inspect the exact POWER binary's code in the two one-shot symbols and `compress`:
separate compression rounds from dispatch, block preparation, word conversion, and key cleanup,
then compare equivalent official BLAKE3 keyed hashing on the same host.
Confirm any proposed change with a repeat native capture and same-workload elapsed benchmark.
Preserve keyed-hash output, constant-time behavior, and secret cleanup;
the profile alone does not establish which operation explains the 0.505x gap.

## Phase 2 — Profile the shared cause

Use the RISC-V wide-multiply finding first; IBM Z is gated on native sampling:

1. Inspect and confirm the RISC-V `ct_mul_u64_wide` cost with target codegen
   and a repeat or controlled production-path experiment.
1. After the runner owner enables perf, profile IBM Z
   `p256-ecdh/public-key/rscrypto-selected` and test whether it shares that cost.
1. Revisit IBM Z and RISC-V `p256-ecdh/agreement/rscrypto-selected`,
   IBM Z P-256 parsing, then RISC-V P-256/P-384 public derivation and P-384 signing.
1. Use a valid POWER control capture if a proposed change touches shared code.

- [ ] Attribute fixed-base multiplication, arbitrary-point multiplication,
      field multiplication/reduction, scalar reduction, inversion, coordinate conversion,
      masked table selection, encoding, entropy, and cleanup.
- [ ] Compare native instruction and branch counts with the exact external winner
      when equivalent symbols and work can be identified.
      Do not compare totals across different operation contracts.
- [ ] Determine whether compiler runtime division/multiplication helpers, missed inlining,
      limb width, excessive masked table scans, spills,
      or an algorithmic representation explains the gap.
- [ ] Confirm the leading cause with a repeat capture
      or one controlled perturbation of the real production path.
- [ ] Record a cause once, then link every affected row.
      Do not open separate implementations until the shared-cause hypothesis is falsified.

## Phase 3 — Fix and prove

- [ ] Prefer target-shaped safe Rust, arithmetic representation,
      and data layout before intrinsics or assembly.
- [ ] Change only production-reachable code.
      Preserve portable authority, deterministic output, blinding, failure opacity, secret cleanup,
      constant-time selection, feature independence, and fallback behavior.
- [ ] Route any unsafe, intrinsic, SIMD, assembly, ABI,
      or target-feature change through the required specialist proof.
- [ ] Rerun the exact profile and a longer Criterion baseline/candidate comparison on the same
      machine identity.
- [ ] Run independent vectors and portable-versus-optimized differentials, target-native tests,
      CT evidence, optimized cleanup checks, dispatch evidence,
      and codegen review for the changed boundary.
- [ ] Run the full benchmark matrix and update `benchmark_results/OVERVIEW.md` only from complete retained artifacts.
- [ ] Verify POWER and other targets sharing the changed code do not regress materially.

## Deferred performance queue

After the catastrophic cross-target rows close:

1. Linux x86-64 P-384 signing.
1. `RapidStreamHasher` large one-write throughput on x86-64.
1. ML-KEM decapsulation, especially where the current aggregate loses.
1. The Intel AES-128-GCM-SIV, AMD AES-SIV construction, AArch64 Argon2id,
   and POWER keyed BLAKE3 cases localized or blocked above; confirm candidates
   with focused same-host elapsed measurements before a production change.
1. Other short-message AES-GCM fixed cost and RISC-V XXH3 only if a fresh
   focused run confirms material impact.

Do not restore the stale ML-KEM key-generation priority:
the September campaign measured key generation as a win.

## Non-goals

- Hiding target losses in cross-platform averages.
- Weakening blinding, validation, cleanup, failure opacity, or constant-time work.
- Treating cross-compilation, source inspection, static counters,
  or a changed profiling build as native wall-clock proof.
- Adding a copied algorithm or an external implementation as rscrypto's benchmark/profile path.
