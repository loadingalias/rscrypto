# BLAKE3 performance evidence

As of 2026-10-09, accepted evidence supports the WASM root specialization,
the Linux little-endian AArch64 reader improvement, and the NEON and AVX-512
partial-chunk streaming forest. The complete sizes ×
targets × modes qualification remains open. Missing cells below are not passes;
functional tests and compilation do not establish throughput.

All comparisons use locked upstream BLAKE3 1.8.7 with the features recorded in
each result. Lower elapsed time is better. Preserve each capture's original
statistics: E10 and S24 use paired intervals, S25 has descriptive per-pass
ratios only, and the current ARM capture uses both repetitions' 95% mean-ratio
envelope. Do not pool these methods or hosts. The 3% regression classification
does not replace the separate 5% ordered-streaming target or an
ahead-of-upstream requirement.

## Coverage matrix

| Sizes and operation | Linux AArch64 native | Linux x86-64 native | WASI on Intel | WASI on Graviton | Apple Silicon native | Browser |
|---|---|---|---|---|---|---|
| 0/64 B, 1/4/64 KiB; plain, keyed, derive-key one-shot | No complete all-mode matrix | Pending | E10: all modes | E10: all modes | Qualified timing missing | Qualified timing missing |
| 4/16/64/256 KiB, 1 MiB; plain one-shot | Graviton3 controls: all five ahead of upstream | Pending | Only overlapping E10 sizes | Only overlapping E10 sizes | Qualified timing missing | Qualified timing missing |
| 64 independent equal-length inputs; 21/64/256 B, 1 KiB | No qualified native matrix here | No qualified native matrix here | E10 | E10 | Functional evidence only | Qualified timing missing |
| 64 B / 4 KiB update-granularity streaming cases | Ordered-prefix gate remains open | Ordered-prefix gate remains open | E10 retained cases | E10 retained cases | Qualified timing missing | Qualified timing missing |
| Retained 4/64 KiB XOF cases | No complete keyed/streaming/XOF matrix | Pending | E10 | E10 | Qualified timing missing | Qualified timing missing |
| Warm 1/16 MiB reader, four-worker comparison | Accepted reader improvement; source-equivalent S24 ARM evidence | S24 entry failed; zero timing rows | Not measured | Not measured | Qualified timing missing | Not measured |
| Cold/warm 1/16/256 MiB, 1/10 GiB; reader, subtrees, mmap/Rayon | 59/60 rows; whole comparison failed | Native controls failed; no qualified matrix | Not measured | Not measured | Functional evidence only | Not measured |
| Same file sizes; standalone `b3sum --num-threads 4` | Qualified timing missing | Qualified timing missing | Not measured | Not measured | 1 MiB functional preflight; no timing | Not measured |
| Derive-key bulk then 60 B suffix; 980–16,384 B | Accepted: within 1.075× one update; −61% at 4,052 B | Accepted gain (−36% at 4,052 B, −42% at 16,340 B); 1.53×/1.31× one update | Not measured | Not measured | Not qualified | Not measured |
| Mixed lengths; plain/keyed/derive; counts 16/65 | NEON replacement rejected | Fixed idle gate failed; zero timing rows | No qualified mixed matrix | No qualified mixed matrix | Functional tests only | Qualified timing missing |
| SVE, 4 KiB–1 MiB | Implementation/native gain gate open | Not applicable | Not applicable | Not applicable | Not applicable | Not applicable |

The [benchmark catalog](../.config/benchmark-matrix.json) owns executable case
names. Mixed batches retain all three ordered distributions: short refill,
block boundaries (including 0/63/64/65/1,023/1,024 B), and tree fallback through
16,385 B. No equal-length result substitutes for mixed-length qualification.

The missing ordered-streaming cells retain prefixes 1/24/70/1,023 B, bulk
1–64 KiB, named 70+4096 and 24+3104 byte inputs, and 3/5/6/7 chunks with tails.
Bulk-then-suffix comparisons remain separate and byte-equivalent, including
the named 980/4000/4052/4096/16340/16384 B bulks followed by 60 B. Their results
follow below; the E10 streaming rows do not establish them.

## Results and limits

**Bulk then suffix:** Rail's granule digest hashes an envelope, then 60 B of
fields, in derive-key mode. Before this change, an envelope that ended inside a
chunk after 3 or 15 complete chunks hashed those chunks partly one at a time. The
partial-chunk forest shares one SIMD pass between the complete chunks and the
partial chunk's leading blocks. Each host ran production (A), the candidate (B),
and production again (A′), each within the 600-second budget. No candidate row
is more than 3% slower than both A and A′.

| Two updates, derive-key | Graviton4 c8g.xlarge | Granite Rapids c8i.xlarge |
|---|---|---|
| 4,052 B + 60 B versus production | −61.4% | −36.4% |
| 16,340 B + 60 B versus production | −36.0% | −42.1% |
| Worst ratio to one update over the same bytes | 1.075 (4,095 B) | 1.534 (4,052 B) |
| 4,052 B + 60 B versus upstream two updates | 0.575× | 0.671× |

Rail's 5% target is below the cost floor at 4 KiB: two updates still need about
one extra block compression and one extra parent compression, an estimated 6–7%
on these hosts. The accepted bar was 1.10× one update. Graviton4 meets it.
Granite Rapids does not: the owned partial-lane kernels are slower than the
production assembly pass, so 4,052 B + 60 B stays at 1.53× and 16,340 B + 60 B
at 1.31×. A second production run moved unchanged x86 one-shot rows with
1,023-byte tails by 3.6–9.2%; treat single-run differences of that size as drift.

**WASM:** E10's complete Intel table has 14 cases ahead of upstream,
eight behind, and one unresolved comparison. Intel losses are plain 4/64 KiB
(+0.89%/+0.36%), keyed 0 B/4 KiB/64 KiB (+2.63%/+2.84%/+1.25%), derive-key
64 KiB (+1.09%), 4 KiB-update streaming (+1.72%), and the 64 KiB XOF case
(+0.25%). Derive-key 4 KiB spans −0.01% to +0.05%. All 23
Graviton cases are ahead of upstream. The tables retain every
absolute time, baseline change and paired 95% interval, including small
baseline-relative losses and uncertainty.

These are Wasmtime 49.0.0 measurements on c8i.xlarge/c8g.xlarge, not native
Rust or browser timings. E10's accepted decision retains **one Intel steal
tick during measurement**; Graviton recorded zero. Physical host exclusivity
was not proved. Repeated-context derive uses rscrypto's warm context cache
against upstream's normal setup, and the implementations have distinct
cleanup contracts. See the full E10 method and verification.

**Reader:** The accepted ARM change reuses S24's complete warm-file comparison:
reader ratios to its earlier production baseline are 0.44005
[0.41508, 0.47301] at 1 MiB and 0.33568 [0.33320, 0.33877] at 16 MiB.
These are about 56% and 66% improvements, not wins over upstream. S24's
full table and raw analysis retain reader, subtree and upstream
controls. Its two-target candidate remains unaccepted: x86 failed the fixed
idle zero-steal gate before timing. S25 separately accepted only the proven
Linux AArch64 scope.

**Large files:** S25's complete retained table and
CSV include both passes, losses and all missing qualifications.
The watchdog stopped the planned capture at 595.219 seconds: 59/60 rows and
590/600 samples. Pass-two warm 10 GiB reader is missing, and the closing
window's four completed reader rows remain unqualified. Eleven other windows
passing individually does not make the comparison pass. Reader point ratios
to mmap/Rayon are all above one; subtree directions vary by size and pass.
No ratio confidence interval or resolved win/loss is inferred from these two
passes. Warm means verified page residency, not CPU-cache or device-cache
warmth. This capture did not include standalone b3sum.
Its residency-first warm method is not pooled with earlier always-reread
captures.

**Current ARM mixed batches:** The 226.105-second Graviton3 capture completed
two repetitions of all 64 rows, 20 samples each, with entry, inherited affinity,
observer occupancy, activity and output gates passing. Of 18 mixed cases,
15 regress versus serial rscrypto and three tree-fallback/65 cases are stable.
Against upstream-auto, 17 regress and plain tree-fallback/65 is stable;
there are no resolved mixed-batch wins. All absolute estimates and both
comparison envelopes retain all 128 rows. The decision table
reports every case. The rejected source and binary remain preserved; public
calls now use the prior serial/tree fallback while accepted equal-length
plain hashing remains. The mixed-lane candidate is removed from production;
its original source and binaries remain archived. That correction has
functional evidence only.

The same capture's plain one-shot controls are ahead of upstream at 4/16/64/256
KiB and 1 MiB: respective time-ratio envelopes are 0.93140–0.93257,
0.89893–0.90027, 0.89302–0.89409, 0.92266–0.92601 and 0.92022–0.92152.
These results do not establish SVE gains, keyed-path costs or streaming targets.

**Current x86 mixed batches:** The fixed comparison stopped after 183.076
seconds at its mandatory idle gate: one steal tick appeared in CPU3 and the
aggregate counter. No SSE4.1, AVX2 or AVX-512 timing row ran, and no retry was
made. The complete result and raw idle observations
are retained with the ordinary binary, complete source and structural output.
A transfer decoding error was repaired by recovering the full packet; its
byte count and SHA-256 verify the recovered archive. This does
not change the failed native gate.

## Source and gate boundaries

Source binding verifies that current `wasm32.rs` matches E10
(`d640eaee…`), and current `io.rs` (`adef0aa2…`) and `parallel.rs`
(`ce9d4822…`) match S25. Full hashes and original inventories are retained.
Matching these individual files preserves their evidence scope; it does not
turn historical artifacts into qualification of every later API or final build.

Instruction-count gate work remains incomplete. Representative
production cases and deliberate failure propagation exist. The final ARM
static-musl profile stopped at its fixed fixture-symbol gate after one
repetition; the x86 profile completed two repetitions but failed its rejection
control observer budget (1.0695% against a 1% limit). Both full archives are
retained. Neither failed profile supplies accepted CI limits, and the CI patch
remains unapplied. An earlier ARM GNU capture lost its raw files during failed
export; it supplies no count or cause claim. A baseline-initialization run is
not a regression pass. Apple Silicon and browser timing cells remain open.

WASM leaf/upstream-gap diagnosis, fixed-count/PMU capture and Intel host
qualification remain deferred at **61/80 workloads and 8/16 controls**, with
their original failed gates. S35 and other rejected candidates are not promoted
by this matrix. Portable-state, outer-key, native-fallback, unwind and
compiler-spill limitations remain; named-owner cleanup is not whole-route
erasure or a constant-time performance claim.

## Local evidence

The raw bundles below are retained locally in ignored evidence directories.
Durable archival outside this workspace remains an explicit retention gap;
tracked summaries do not replace those raw observations.

- E10 method and verification: `benchmark_results/blake3-wasm-root-specialization-20261006T132000Z/report.md`
- E10 Intel table: `benchmark_results/blake3-wasm-root-specialization-20261006T132000Z/table-x64.md`
- E10 Graviton table: `benchmark_results/blake3-wasm-root-specialization-20261006T132000Z/table-arm64.md`
- S24 report: `benchmark_results/blake3-linux-reader-20261008T074000Z/report.md`
- S24 ARM analysis: `benchmark_results/blake3-linux-reader-20261008T074000Z/arm64/analysis.json`
- S25 report: `benchmark_results/blake3-full-file-matrix-20261008T090400Z/report.md`
- S25 measured rows: `benchmark_results/blake3-full-file-matrix-20261008T090400Z/arm64/all-measured-rows.md`
  and `all-measured-rows.csv`
- ARM batch and Graviton3 controls: `benchmark_results/blake3-delivery-20261008T185226Z/batch-native-summary.md`
  and `batch-native-summary.json`
- `b3sum` functional preflight: `benchmark_results/blake3-delivery-20261008T185226Z/b3sum-functional/report.md`
- Matrix source bindings: `benchmark_results/blake3-delivery-20261008T185226Z/matrix-source-binding.json`
- Structural result: `benchmark_results/blake3-delivery-20261008T185226Z/structural-result.md`
- x86 native result and idle gate: `benchmark_results/blake3-delivery-20261008T185226Z/x86-native/blake3-delivery-20261008-x86/results/`
  (`result.json`, `activity/idle.json`)
- x86 recovery: `benchmark_results/blake3-delivery-20261008T185226Z/x86-native-recovery.json`
- Partial-chunk forest, first candidate (rejected): `benchmark_results/blake3-rail-forest-20261009T020000Z/report.md`
- Partial-chunk forest, accepted: `benchmark_results/blake3-rail-forest-v2-20261009T031500Z/report.md`
  (`plan.md`, `x86-gates.txt`, `arm-gates.txt`, and raw A/B/A′ runs per host)
