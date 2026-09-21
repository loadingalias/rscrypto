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
- corrected cases aim for at least 10% lower median elapsed time than the fastest current
  equivalent implementation; a reviewed target-native analysis may instead close a case
  when both implementations have reached the same physical or platform lower bound,
  or when the remaining comparison is not equivalent work;
  and
- the complete benchmark matrix and applicable correctness, CT, cleanup, dispatch,
  and target evidence pass on the final revision.

## Current work order

1. POWER keyed BLAKE3 at 64 bytes.

RISC-V P-256 follow-up is deferred after the major correction failed to close the CRRL gap.
IBM Z P-256 and ECDSA work is deferred until the native runner permits attributable `perf` sampling.

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

#### RISC-V result (2026-09-20; major defect fixed, CRRL gap remains, follow-up deferred)

Status: the catastrophic loss is fixed, but rscrypto did not win. Public derivation takes 1.538x
CRRL's time, and agreement takes 2.478x CRRL's time. Both performance rows remain open. Defer
further RISC-V work to a later pass rather than treating this slice as closed.

Revision `11fee328f57297f69303371bf7593e4e33aa7a75` replaces the 64-round software product on RV64
with safe Rust that lowers to `mul` and `mulhu`. It normalizes both multiplier operands to set the high bit,
then applies a branchless correction so operand magnitude does not vary with the secret. The same bounded
change adds a sparse P-256 Montgomery reduction, a 10-product square, a fixed inversion chain, direct
Jacobian conversion, and a three-way width-seven fixed-base comb. The comb performs 12 doublings,
36 mixed additions, and 37 fixed-count full-table scans. Its three 8 KiB tables are shared with ECDSA;
the change adds no dependency, vendored implementation, unsafe code, intrinsic, or assembly.

The exact native [RV64 benchmark](https://github.com/loadingalias/rscrypto/actions/runs/35542196605) reduced
public derivation from the retained 13.584 ms to 208.04 us, a 65.3x speedup. CRRL measured 135.28 us in the
same binary, so rscrypto remains 53.8% slower. Agreement fell from 30.860 ms to 832.31 us, a 37.1x speedup;
CRRL measured 335.92 us, leaving rscrypto 147.8% slower. These are major corrections, but neither case clears
the 10% goal and no physical lower bound has been demonstrated. Do not mark either row closed.

The follow-up [native profile](https://github.com/loadingalias/rscrypto/actions/runs/35543835843) retained
493 cycle samples with none lost. It attributed 26.02% self to RV64 field squaring, 19.54% to field
multiplication, and 18.89% to Montgomery reduction; the remaining 28.00% inlined into public derivation.
The field kernels therefore account for at least 64.45% of sampled cycles. Static review also found that the
fixed-base comb scans about 296 KiB of table coordinates per public key. Field arithmetic and constant-time
table selection remain the demonstrated optimization targets.

The full [RV64 constant-time run](https://github.com/loadingalias/rscrypto/actions/runs/35542201632) failed
globally on an unrelated Ed25519 timing case and pre-existing manifest coverage gaps. Both applicable P-256
DudeCT cases passed on native hardware: public derivation reported `|t| = 2.3471`, and agreement reported
`|t| = 6.90385`, below the configured threshold of 10. Generated-code heuristics reported no `needs-fix`
finding; manual review confirmed only fixed-count branches. This supports this revision's P-256 timing claim,
but does not clear the unrelated repository-wide CT failures.

A correction-aggregation experiment at `c99a45d1536048e8d1f8a48100a73fb23a1c2b8e` reduced RV64 multiply
memory operations from 114 to 66, but the exact native public-key benchmark was statistically flat at
208.23 us versus the 208.04 us accepted baseline. It was reverted because generated-code cleanliness without
a measured speedup does not meet the acceptance rule. A canonical-field Solinas-reduction prototype passed
the focused differential tests but required roughly 1,100 instructions per multiply and 860 per square,
versus 458 and 325 for the accepted Montgomery kernels, so it was rejected before spending another CI run.

Further RISC-V work is deferred. The next active target is POWER keyed BLAKE3 at 64 bytes.

### Deferred — IBM Z P-256 public derivation

No sampled hot path exists yet.
The same fixed-work multiply is selected by `cfg(target_arch = "s390x")`, making it the first hypothesis,
not a measured IBM Z finding.
Resolve native perf access, capture the exact public-key case,
and check whether field multiplication dominates before sharing a RISC-V fix.
The existing public-key/agreement/parse rows remain open.
Do not resume IBM Z optimization work until the runner can produce attributable native samples.

### Intel x86-64 — AES-128-GCM-SIV, 32-byte seal

The profile attributes 35.65% inclusive to `compute_tag_wide`
(including 15.41% self in `polyval::pclmul::clmul128_reduce`),
23.67% inclusive to `aes128_expand_key`, 20.63% self to wide CTR encryption, and 12.10% inclusive to `derive_keys`.
Per-nonce subkeys must still be derived by the algorithm.
Inspect generated code and isolate tag/POLYVAL
and per-message key expansion on the real 32-byte seal path;
compare the same complete encryption contract against AWS-LC
before choosing a bounded production change.

#### Intel result (2026-09-20; performance row closed)

The exact `aes-128-gcm-siv/copy-and-encrypt/rscrypto/32` case was rerun on an Intel Xeon 6975P
Granite Rapids `c8i.4xlarge`. Baseline run `20260921T030958Z-z9tlcty5` measured rscrypto at
202.69 ns [202.23 ns, 203.03 ns] and AWS-LC at 92.501 ns [92.039 ns, 93.135 ns]. Final run
`20260921T031552Z-j9iwvp8i` measured rscrypto at 75.371 ns [74.899 ns, 75.878 ns] and AWS-LC
at 91.620 ns [91.156 ns, 92.255 ns]. Criterion reports a 62.893% median decrease from the
rscrypto baseline, with a 95% confidence interval of [62.769%, 62.997%] and `p = 0.000`.
The final rscrypto median is 17.73% below AWS-LC and clears the 10% acceptance target.

The profile was causal: the 32-byte seal paid separately for per-nonce key derivation, derived
AES-128 key expansion, generic POLYVAL batching, tag encryption, and CTR setup. The bounded x86-64
path now handles transcripts of at most four padded blocks with four-lane VAES key derivation,
native derived-key expansion, one VPCLMUL aggregate, direct tag encryption, and four-lane CTR.
Longer messages and non-VAES backends retain the existing portable-authority path. The shared CTR
counter builder was simplified instead of leaving a second legacy construction, the AES-GCM-SIV
leaf-feature gate for the shared reduction was repaired, and a stale generated ECDSA table type is
now gated with its constants.

Security evidence includes the independent AES-128-GCM-SIV oracle on the native Granite Rapids
path, full ASan fuzz-target coverage, native Linux and macOS RSA assembly gates, and focused native
DudeCT runs. The largest focused absolute t-statistic was 3.06490, below the repository threshold
of 10. Generated-code review confirmed fixed secret-processing instruction flow and explicit wipes
for the authentication key, derived encryption key, round schedule, POLYVAL state and powers, and
partial-block keystream. The repository CT artifact checks, strict validation, and zeroization
sentinel passed. Full `ct-full` proof did not complete because the ephemeral Ubuntu 26.04 image lacks
BINSEC and the installer intentionally supports Ubuntu 24.04; do not convert that environment gap
into a full-CT claim.
The final revision passed `just check`, `just test --all`, and `just test --all --portable`.

### AMD x86-64 — AES-SIV-CMAC-256 construction

`AesSivCmac256::new` accounts for 81.27% inclusive and `aes128_expand_key` for 72.00% inclusive.
Construction expands both 16-byte key halves and derives CMAC subkeys.
Several large libc addresses lack function names;
do not label them cleanup or copying without symbol evidence.
Inspect both AES expansions, CMAC setup, and destruction on the exact construct/destroy workload;
compare equivalent RustCrypto construction on the same AMD host.

#### AMD result (2026-09-21; construction row removed)

Status: closed as a non-equivalent comparison. No production change was justified.

Native run `20260921T040845Z-_mvsaf0q` used an AMD EPYC 9R45 `c8a.4xlarge`,
Rust 1.98.1, the repository `bench` profile, and the retained Criterion settings.
It reproduced the construction-only gap at 83.025 ns for rscrypto versus 30.575 ns
for RustCrypto. Generated-code inspection of the exact rscrypto artifact confirmed
that the AES-NI branch expands both 16-byte halves and moves each 176-byte schedule
into the backend-dispatched key owner. The AES-NI expansion kernel itself accounted
for only 18.68% self time in the retained profile.

The two constructors do different work. rscrypto expands both the CMAC and CTR keys,
derives the CMAC subkeys, and later wipes both schedules and subkeys. RustCrypto
`aes-siv` 0.8.0 expands the CMAC key but retains the raw CTR key and constructs that
AES cipher inside every `apply_keystream`; the benchmark dependency also omits its
optional `zeroize` feature. The standalone constructor row therefore rewards deferred
work and weaker cleanup rather than equivalent useful work.

The same native run measured the complete copy, construction, seal, and destruction
lifecycle:

| Plaintext | rscrypto | RustCrypto | rscrypto relative to RustCrypto |
| ---: | ---: | ---: | ---: |
| 0 bytes | 138.51 ns | 161.85 ns | 14.4% lower |
| 16 bytes | 165.68 ns | 181.89 ns | 8.9% lower |
| 64 bytes | 170.52 ns | 185.88 ns | 8.3% lower |
| 256 bytes | 303.01 ns | 300.59 ns | 0.8% higher |
| 1232 bytes | 1.0126 us | 973.64 ns | 4.0% higher |

With reusable contexts, rscrypto seal was faster at every measured size, from 51.1%
lower at zero bytes to 2.4% lower at 1232 bytes; open showed the same result. The large
construction-only ratio does not survive either equivalent lifecycle. Remove its benchmark,
CI profile preset, and workflow choice instead of retaining a misleading legacy target.

### AArch64 — Argon2id OWASP profile

`argon2::aarch64::compress_neon` accounts for 95.23% inclusive (94.02% self) inside `fill_segment_inner`; matrix cleanup is 2.22% self.
The workload uses 19 MiB, two passes, one lane, a 16-byte salt, and 32-byte raw output.
Inspect the production NEON compression's generated rounds, spills, and memory traffic,
then compare equivalent full hashes against RustCrypto on the same host.
Do not reduce Argon2 work factors to improve this ratio.

#### AArch64 result (2026-09-21; performance row closed)

The loss came from dispatching to the AArch64 NEON compressor. On the exact
19 MiB, two-pass, one-lane Argon2id workload, Graviton4 baseline run
`20260921T042307Z-hzs70uav` measured rscrypto at 30.686 ms and RustCrypto at
18.199 ms. Final run `20260921T042730Z-a9wfxhfr` selected the portable authority
and measured rscrypto at 18.187 ms and RustCrypto at 18.327 ms. That is a 40.7%
rscrypto reduction, and the final median is 0.76% below RustCrypto.

The result generalizes across the representative AArch64 machines available to
the repository. On Graviton3, the portable run `20260921T043343Z-jh_lh12t`
measured 18.768 ms versus 33.719 ms for the NEON run
`20260921T043531Z-cusd6jg4`, a 44.3% reduction. On Apple M1 Pro, the portable
path measured 15.898 ms versus 17.229 ms for NEON, a 7.7% reduction.

Generated-code inspection explains the direction. The NEON kernel reserves a
2 KiB stack frame, zeroes both 1 KiB temporaries before overwriting them, and
lowers each BlaMka multiply through lane narrowing plus `umull`. The portable
authority lets the scalar AArch64 core schedule independent 64-bit multiplies
directly and wins despite doing the same Argon2 work.

Production AArch64 dispatch now selects the portable authority. The NEON kernel
and its identifier remain available only for existing diagnostic differential
evidence; removing that public identifier would be an unrelated compatibility
break. No cryptographic algorithm, work factor, memory access rule, output,
failure behavior, or cleanup path changed. The selected compressor is already
the portable correctness authority and the leaf covered by the Argon2i BINSEC
contract.

The portable-dispatch candidate passed the native Graviton3 evidence suite in
normal and portable modes, plus `just test --all` and
`just test --all --portable` on the same machine. The final
compatibility-preserving selector then passed local `just test-evidence`: 1,212
native-dispatch tests and 1,190 portable-only tests. The focused Apple M1 Pro
Argon2i Dudect run used 20,000 samples and reported a maximum absolute
t-statistic of 1.39840, below the repository threshold of 10. The CT artifacts
and manifest passed `just ct-validate`, and the dedicated macOS AArch64 RSA
assembly gate passed in debug, release, and public-operation comparison modes.
A fresh AArch64 Linux BINSEC run did not start: the repository Zig cross-linker
rejected Rust's
`--fix-cortex-a53-843419` linker argument while building the evidence binary.
The portable compressor itself is unchanged; retain the existing manifest
proof boundary and treat this as an environment limitation, not fresh formal
evidence.

### POWER — keyed BLAKE3, 64 bytes

The exact-case [native report](https://github.com/loadingalias/rscrypto/actions/runs/35415762447/artifacts/10575707870) retained 414 flat `cycles:u` samples with no loss.
It assigns 50.49% to `digest_public_oneshot`, 25.45% to `compress`, 11.86% to `digest_oneshot_words`, and 11.56% to Criterion's `Bencher::iter`.
These are symbol-level samples, not call-chain percentages: inlined work may be charged to `digest_public_oneshot` or `digest_oneshot_words`.
Do not interpret 50.49% as dispatch overhead; inlined compression may be charged to that symbol.

The selected benchmark calls the production `Blake3::keyed_digest` on 64 bytes.
Diagnostics select the portable kernel on POWER;
this input takes the tiny one-block path through `hash_tiny_to_root_words` and `compress_chunk_tail_to_root_words`.
Inspection of the retained POWER10 binary established the leading structural cost.
The exact 64-byte path copied the input into a zero-filled 64-byte block,
copied the complete resolved dispatch aggregate before selecting one kernel,
and passed key words through nested by-value owners.
Each secret owner was correctly cleared, but the nesting produced repeated volatile stores and POWER `sync` barriers.
This explains the sampled time outside `compress` without treating the 50.49% symbol attribution as dispatch alone.

The first candidate removes the monolithic resolved-dispatch aggregate,
caches compact size-class kernel identifiers separately from streaming and parallel state,
borrows one key-word owner through the tiny portable path,
and reads an exact 64-byte input directly instead of materializing a padded copy.
It retains volatile cleanup for the message words, full compression result, digest words, and sole key owner.
Cross-generated POWER10 release-LTO code for an exact production-API wrapper contains no input `memcpy`
and two cleanup barriers: one after clearing the secret-derived message words and one after clearing the digest words and key owner.
The temporary inspection wrapper was removed after review.

As supporting evidence on Apple Silicon, the exact production benchmark moved from a 98.901 ns baseline median
to 84.700 ns after the final cleanup-complete change, a reduction of 14.4%.
The immediately preceding comparison measured official BLAKE3 at 76.286 ns,
placing rscrypto at 0.901x by the table's external-time / rscrypto-time ratio and within the 10% target.
This is not POWER acceptance evidence.

The exact paired [POWER10 benchmark](https://github.com/loadingalias/rscrypto/actions/runs/35568638076)
measured rscrypto at 147.01 ns `[146.64, 147.46]` and official BLAKE3 at
118.46 ns `[118.31, 118.68]` across 200 samples after a three-second warm-up and ten-second measurement.
The safe-Rust cleanup reduced the historical rscrypto median by 35.9%, from 229.4 ns,
but rscrypto remains 24.1% slower than the same-run external implementation.
The matching [native profile](https://github.com/loadingalias/rscrypto/actions/runs/35567963259/artifacts/10624811596)
retained 310 flat `cycles:u` samples with none lost.
It assigns 90.23% to portable `compress`, 5.20% to Criterion's `Bencher::iter`,
and 4.57% to `digest_oneshot_words`.
The removed dispatch aggregate, input copy, and nested key-owner costs no longer appear as sampled hotspots.

A measured POWER10 dispatch experiment selected the existing VSX compression kernel for the 64-byte class.
Its exact paired [benchmark](https://github.com/loadingalias/rscrypto/actions/runs/35570144259)
measured 145.51 ns `[143.26, 148.43]` against official BLAKE3 at
118.07 ns `[117.99, 118.15]`: 23.2% slower, and only 1.0% below the portable candidate with overlapping intervals.
The [matching profile](https://github.com/loadingalias/rscrypto/actions/runs/35570180583/artifacts/10626320578)
retained 316 flat `cycles:u` samples with none lost and assigned 79.98% to
`compress_power_vsx`, proving that the production VSX kernel executed.
The dispatch experiment was rejected and the portable short-message policy restored.
Keep the task open with portable scalar compression as the next measured target.

The final code revision `c95819f10f922dbbf18f383dce543b29efa85891` passed the complete
[CI workflow](https://github.com/loadingalias/rscrypto/actions/runs/35576174499), including native
POWER, RISC-V, s390x, AArch64, Linux x86-64, and Windows x86-64 execution plus compatibility and
package lanes. Its matching [native POWER CT workflow](https://github.com/loadingalias/rscrypto/actions/runs/35576174521)
retained a clean-commit artifact with 94 of 94 gated DudeCT cases passing, zero blockers, and zero
diagnostics. Artifact generation, strict validation, the zeroization sentinel, and DudeCT import
passed. The largest relevant absolute t-statistics were 2.03190 for BLAKE3, 3.22876 for
AES-SIV-CMAC-256, and 3.08138 for RSA, below their configured thresholds of 8 or 10. BINSEC remains
not applicable on this target because its shipped PPC64 decoder does not support little-endian PPC64;
the retained POWER evidence is generated-code heuristics plus native DudeCT.

Disassembly of the retained `f9902f96` profile binary narrows the remaining loss to
unneeded full-output work in portable compression. The rscrypto `compress` body has
965 instructions, 86 loads, and 59 stores, while the same binary's official
`blake3::portable::compress_in_place` body has 941 instructions, 81 loads, and 44 stores.
Both perform the same seven rounds. rscrypto then materializes all 16 output words even though
the tiny one-block path retains only the first eight chaining-value words; the official path
finalizes only those eight words. A source-layout experiment that replaced compression locals
with an indexed state array increased the rscrypto body to 967 instructions, 86 loads, and
60 stores under the same generic POWER build inputs, so it was rejected.

The current safe-Rust candidate shares the seven rounds between two finalizers and sends only
the tiny portable path through an 8-word chaining-value finalizer. Cross-generated generic
POWER bench-profile assembly reduces that finalizer to 939 instructions, 83 loads, and 44 stores,
while the full 16-word compression path remains at 964 instructions, 86 loads, and 59 stores.
The candidate finalizer has no calls, divisions, floating-point instructions, conditional branches,
or secret-dependent addresses. Keyed message words retain volatile cleanup, and the full-output
path retains its existing cleanup.

Two detached-baseline runs at revision `ce05765b` on Apple Silicon measured rscrypto at
81.287 ns and 81.274 ns. The current candidate measured 73.819 ns `[73.746, 73.893]`,
a 9.22% reduction from the selected baseline. The same-run official implementation measured
75.681 ns, but its row moved by 3.66% from the selected baseline, so this is supporting evidence,
not a cross-implementation acceptance claim. The release binary's Mach-O text segment remained
2,572,288 bytes. Native and forced-portable evidence suites pass, with 1,214 and 1,192 tests,
respectively, and `ct-validate` passes. `just check` also passes the full release-native and
debug-portable Clippy matrix, independent workspaces, dependency policy, and Rustdoc. Native POWER
profiling and elapsed measurement remain the acceptance gate for this candidate.

## Phase 2 — Localize the active cause

Work the retained native profiles in the current order:

1. POWER keyed BLAKE3 one-shot digest and compression codegen.

Resume IBM Z only after native sampling access exists. Circle back to RISC-V P-256 field arithmetic,
reduction, and constant-time table selection in a later pass.

- [x] Separate portable compression from dispatch selection, block preparation,
      byte-to-word conversion, and keyed cleanup in the retained POWER binary.
- [x] Identify the redundant input copy, aggregate dispatch copy,
      nested key owners, and repeated cleanup barriers in the exact production path.
- [x] Confirm the structural hypothesis with one safe-Rust production candidate,
      cross-generated POWER10 code, and a same-workload local elapsed comparison.
- [x] Repeat the exact profile and rscrypto-versus-official elapsed benchmark on the POWER runner.
- [x] Accept, revise, or reject the candidate from native POWER evidence.
- [x] Compare the retained portable compression body with the official implementation and
      isolate full-output finalization as the next structural difference.
- [x] Reject the indexed-state source experiment and produce a CV-only safe-Rust candidate with
      local codegen, elapsed, correctness, cleanup, and constant-time evidence.
- [ ] Repeat the exact profile and elapsed benchmark for the CV-only candidate on the POWER runner,
      then accept, revise, or reject it from native evidence.

## Phase 3 — Fix and prove

- [x] Prefer target-shaped safe Rust, arithmetic representation,
      and data layout before intrinsics or assembly.
- [x] Change only production-reachable code.
      Preserve portable authority, deterministic output, blinding, failure opacity, secret cleanup,
      constant-time selection, feature independence, and fallback behavior.
- [x] Route any unsafe, intrinsic, SIMD, assembly, ABI,
      or target-feature change through the required specialist proof.
- [x] Rerun the exact profile and a longer Criterion baseline/candidate comparison on the same
      machine identity.
- [x] Run independent vectors and portable-versus-optimized differentials, target-native tests,
      CT evidence, optimized cleanup checks, dispatch evidence,
      and codegen review for the changed boundary.
- [ ] Run the full benchmark matrix and update `benchmark_results/OVERVIEW.md` only from complete retained artifacts.
- [ ] Verify POWER and other targets sharing the changed code do not regress materially.

## Deferred performance queue

After the active per-architecture targets above:

1. RISC-V P-256 field arithmetic, reduction, and constant-time table selection.
1. IBM Z P-256 and ECDSA, only after native `perf` access is available.
1. Linux x86-64 P-384 signing.
1. `RapidStreamHasher` large one-write throughput on x86-64.
1. ML-KEM decapsulation, especially where the current aggregate loses.
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
