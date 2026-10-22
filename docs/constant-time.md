# Constant-time claims

In this project, constant time means that secret values do not change control flow,
memory addresses, or the operands of variable-latency instructions within a defined operation.
Public input lengths, algorithm parameters, target features, allocation, scheduling,
and external entropy sources can still change the time.

[`ct.toml`](../ct.toml) is the inventory of operations.
An operation has a claim only for the targets, features, compiler, linked binary,
and evidence that `ct.toml` names.
Code that `ct.toml` does not list has no constant-time claim.

## Evidence model

Release evidence combines:

- source review for secret-dependent branches, indexing, comparisons,
  and variable-latency instructions;
- differential tests that bind accelerated paths to portable Rust behavior;
- inspection of optimized linked binaries for the retained entry points;
- BINSEC proofs for declared fixed-shape kernels;
- DudeCT timing tests for declared end-to-end cases.

Source code that looks branchless is not proof.
Compiler lowering, inlining, target features, and linking can change machine behavior.
For example, a checked `strict_*` operation on a secret-derived value tests that value.
RV64GC has no conditional move, so the test can compile to a secret-dependent branch.
Bounded secret arithmetic therefore uses `wrapping_*`, with the bound written at the definition,
as the Curve25519 field and scalar code does.
Evidence for the release harness does not cover a downstream binary that is compiled differently.

LLVM can also turn a masked selection back into a branch on the mask.
The binary inspection reports a conditional branch whose two paths rejoin after straight-line code
as `select_branch`, the shape of that lowering.
For the primitives under `[asm_select_branch]` in `ct.toml`, an unclassified `select_branch` fails the gate.
An `asm_waiver` binds a reviewed condition to its target, primitive, function, finding kind,
and condition identity (`slice_sha256`): a digest of the backward slice of the condition in the linked binary.
The slice holds every instruction whose result reaches the condition through registers, flags,
or memory inside the function, each with the placement-independent identity described below.
A load reaches the stores that can write the loaded bytes.
Stack slots are tracked by their offset from the entry stack pointer,
and fields read through an unmodified argument pointer by their offset from it.
Any other store that can overlap a load joins its slice.
If a frame address is stored, passed to a call, or used in arithmetic, every call can write the frame.

- A slice is *closed* when it ends only at function inputs, entry memory, immediates,
  and read-only data bound by contents.
  Code outside a closed slice cannot change the condition,
  so its waiver keeps matching when other code in the function or the file changes.
- A slice is *open* when a call, an instruction of unknown effect, or an indirect jump can change a value it reads.
  An open identity also binds the complete function hash, so any change in the function requires renewed review.

A count-only `asm_public_operand` rule cannot exempt a branch:
another instruction in the same function could depend on a secret while staying within that count.
Instruction identities use function-relative positions. Resolved direct transfers retain their
instruction length, mnemonic, registers, condition, and exact destination symbol and offset instead
of the placement-dependent displacement bytes.
The same rule applies to resolved address references in the linked binary:
x86-64 RIP-relative operands, AArch64 `adrp`, `adr`, and literal loads, RISC-V `auipc` pairs,
POWER TOC-relative pairs and the global-entry TOC setup, and IBM Z relative-long instructions such as `larl`.
Each keeps its length, mnemonic, and registers, and replaces the displacement with a description of the target:

- code: the containing function and offset, plus the 16 target bytes for an interior of another function;
- writable data: the containing symbol and offset, only when that symbol has a stable name;
- read-only data: the stable name, if any, the offset, the size, and a digest of the complete allocation unit;
- a slot that the loader fills, such as a GOT, TOC, or import-table entry: the imported symbol or the described target.

Inside read-only data, each pointer that the loader relocates is replaced by the description of its target,
up to three levels deep, so moved targets with unchanged contents keep the digest.
ELF allocation units are linker-map input sections, and Mach-O units are ld64 map atoms.
Merged string and literal sections bind only the referenced literal.
On PE, a unit runs from the nearest known object start to the next one;
x86-64 code generation does not merge globals.
Mach-O artifacts without an ld64 map, such as the `543fa738` baseline, use the same bound.
A merged unit, such as an LLVM `_MergedGlobals` block on POWER, binds all of its bytes,
so a change anywhere in it requires renewed review.
A high part (`adrp`, `auipc`, or `addis`) is described only when its uses are resolved low parts
in the straight-line code that it dominates, and every later path overwrites or clobbers the register
before another read.
Other bytes and operands remain exact, including unresolved targets, anonymous writable data,
immediates that add an offset to an address that is already formed, and indirect transfers.
Because the identity binds read-only contents, a changed constant also requires renewed review,
even at an unchanged address. Different compiler or linker output can therefore require renewed review.
The artifact hashes retain the exact binary identity.
A waiver that matches no finding fails the verdict, but evidence collection and timing continue.
`just ct-renew` prepares the review of stale waivers and changes nothing;
see [the renewal workflow](../scripts/README.md#waiver-renewal).
The retained Linux and Windows reviews use the workflow checkout path, pinned
compiler, target flags, and native or GNU cross linker recorded with each artifact.
A Mac/Zig build or another checkout path can change embedded source locations,
data placement, and function hashes, and with them the identities of open slices.
Reclassify those exact artifacts before rebinding a waiver; the review does not grant a portable hash exemption.
Artifact builds require the configured `rust-src` component and record its
source root and `std/src/lib.rs` digest. Rustc can translate standard-library
source locations to installed sources, so component presence and path affect
embedded diagnostic strings, linked addresses, and exact function identities.
Waivers distinguish public operands from existing claim exclusions and allowed disclosures. The latter
two require a reference to the existing `ct.toml` contract and do not classify the operand as public.
An exclusion cites `primitive:<id>:notes`; an allowed disclosure cites
`primitive:<id>:may_leak:<value>`, where the value must already belong to that primitive.
Other primitives report the shape for review.
If an enforced primitive shares a finding with other primitives, every affected
primitive needs a classification before the shared finding is accepted.

The enforced roots follow the claims as written.
The RSA sign and decrypt roots borrow a key that a setup entry imported,
and they create their private scratch for each call.
Key import, validation, release, and DER export run only in setup and diagnostic roots.
The `rsa.private_key_material` root is the bounded same-width component-validation leaf
that its claim names, which BINSEC and DudeCT also cover.
Broad import, validation, and export keep their diagnostic DudeCT cases and the diagnostic
`ct_entry_rsa_private_key_pkcs8_roundtrip` root.
That root is not a release equality root, so the public-length comparisons in import validation
have source review but no linked-binary equality evidence; `public_len_comparison` in `ct.toml` counts them.
`unreachable_symbols` under `[asm_select_branch]` lists the import, validation, and release symbols.
The scanner fails when the linked direct-call and tail-transfer closure of an enforced root reaches
one of them, and when one of them is missing from the linked binary.
Indirect transfers inside the closure remain separate findings.

Linked disassembly must include every mapped function start. If stripped inline
tables make x86 decoding cross a function boundary, the artifact collector restarts
the matching decoder at that mapped address on the original binary. It retains the
raw dump, bounded recovery command and output, and binary hash. A missing start
after recovery fails evidence collection.
Fixed x86 GOT calls require matching instruction bytes and dynamic relocations.
POWER import-thunk resolution checks the complete supported instruction sequence,
TOC slot and named relocation. The s390x resolver checks the complete 14-byte
`LARL %r1` / `LG %r1` / `BR %r1` sequence inside `.plt`, computes its signed
halfword displacement, and requires a named `R_390_JMP_SLOT` at the resulting
slot. Missing or mismatched relocation evidence rejects artifact collection.
Exact decoder-provided PLT names take precedence over broad inferred symbol
ranges. Recognized local labels inside a mapped function remain part of that
function's body, so they cannot hide later instructions from review. Unresolved
internal destinations fail the enforced call-closure check.

Build and validate the local evidence artifacts:

```bash
just ct-artifacts
just ct-validate
```

`ct-validate` rejects generated artifacts that are missing or stale.
`just ct-full` builds the artifacts, runs the available timing checks, and writes the reports.
A claim for a target needs the evidence that `ct.toml` requires for that target.
A local host cannot stand in for another target.

For strict manifest and artifact coverage, build the artifacts, then run `just ct-validate --strict-coverage`.
This checks compiler output and coverage only.
Timing and formal evidence need their own target runs.

The bounded PBKDF2 proof hooks use `verify_primitive` with one iteration and a fixed salt.
They must reach key derivation and comparison.
The application password-policy verifier rejects these weak parameters before comparison,
so earlier proofs of the policy-rejecting hooks do not prove PBKDF2 verification coverage.
Regression tests for the internal hooks check both successful and failed verification.
Application password policies are separate and do not change.

### Linked-artifact review, 2026-10-18

The POWER and IBM Z preparation failures in [run 39091286840](https://github.com/loadingalias/rscrypto/actions/runs/39091286840)
invalidated 84 and 63 exact waivers, respectively. Comparison with
[passing run 38842297553](https://github.com/loadingalias/rscrypto/actions/runs/38842297553)
bound the earlier reviews at `ed86bcb3` to the linked artifacts at `c19b49bb`.
The builds used `nightly-2026-10-14`, GNU cross-linkers, and the checkout path
`/home/runner/work/rscrypto/rscrypto`.

All 84 POWER branch locators and their conditions were unchanged. The 13 enclosing
functions differed at 36 address-loading instructions. The relocated data contained
the same two 16-byte vector constants and the same panic source paths, lines, and
columns. The comparison decoded those values from both linked ELF files; it did
not infer equivalence from matching instruction counts. The existing public-input,
claim-exclusion, and disclosure rationales therefore remain applicable.

IBM Z was rebuilt in an Ubuntu 24.04 x86-64 container with the same compiler,
installed Rust sources, checkout path, GNU cross-linker, and target flags.
It reproduced the same 63 unmatched waivers as CI. All 63 locators were unchanged.
The 12 enclosing functions differed only at 61 `larl` instructions referring to
30 relocated data objects. Their complete referenced contents were identical,
including all four 4,096-byte P-256 tables, field and SHA-256 constants, RSA policy,
prime tables, DER bytes, domain strings, and panic text. No condition or index
calculation changed.

The original failure skipped the native lanes. Rebuilding Linux x86-64 at the
same checkout path exposed 64 additional unmatched waivers. All their locators
were unchanged. Seven enclosing functions differed only at 17 RIP-relative loads
or address calculations. The ten referenced data ranges were byte-identical:
RSA policy, domain strings, DER bytes, the complete small-prime table,
Miller–Rabin bases, and the one-byte value used by key consistency checks.
A fresh Linux AArch64 artifact build passed with its existing waivers.

The Windows-only [run 39094250382](https://github.com/loadingalias/rscrypto/actions/runs/39094250382)
exposed nine unmatched waivers at the same source revision. All nine locators were
unchanged. Three enclosing functions differed only at eight `lea` instructions.
The six referenced objects moved by 64 bytes: four panic locations retained their
exact source paths, lines, and columns; an eight-byte zero constant and a one-byte
value were identical. The checkout path was
`C:\actions-runner\_work\rscrypto\rscrypto`.

The reviewed linked binary SHA-256 values were:

| Target | Source | SHA-256 |
| --- | --- | --- |
| POWER | `ed86bcb3` | `45d508d40f34612142749c14599bf2821fabf8e24479b167d49104b29d2a09d1` |
| POWER | `c19b49bb` | `124bc2e4e19a03e4c0db7524ffda8acbf5a8d34a7032e8045ed96b750dce8a2d` |
| IBM Z | `ed86bcb3` | `606c5dd9071c299f6031fc9cd8dbf5c9c55ae19d04b28f318c2338321fa54d60` |
| IBM Z | `c19b49bb` | `9c4dd9424a3d479f16efedacc5ac1a513f4dfc07ff2176a259d0999b2f7d8335` |
| Linux x86-64 | `ed86bcb3` | `0674d42597af5e2a7a966fc2accb8cf80cb492e0a4f136934429454dad57321d` |
| Linux x86-64 | `c19b49bb` | `e4b65d2a609fc1d35e42696dd5dd8893872676a972989f450639736ac70d383f` |
| Windows x86-64 | `ed86bcb3` | `1f1bfc89e4e185f484db6b2eea8faef6bb9c19fcd83d9045fe3a7b37d5ad46c9` |
| Windows x86-64 | `c19b49bb` | `25e7076dd72f42a55773a8c4c9066ff29c53113a5140bdef179fba62c1a5ac40` |

The refresh changes only the reviewed function hashes and review dates in `ct.toml`.
Instruction locators, classifications, claim boundaries, and exact matching remain
unchanged. The refreshed artifacts pass the assembly scanner with no
unresolved failures. This review establishes artifact classification, not a new
native timing result.

### Identity v2 re-encoding, 2026-10-21

The address-reference identity changes every function hash, so the 457 waivers that remain
after the RSA scope change were re-encoded on the artifacts they were reviewed against:
[Constant-Time 39100887496](https://github.com/loadingalias/rscrypto/actions/runs/39100887496) at `0c959150`
for Linux, Windows, POWER, and IBM Z, its RISC-V run
[39101292966](https://github.com/loadingalias/rscrypto/actions/runs/39101292966),
and the local Apple Silicon artifact from `543fa738`.
A waiver moved to the v2 locator and hash only when its exact v1 locator and hash named the same
instruction in that artifact. Classifications, rationales, and review dates are unchanged.
With the re-encoded waivers, the v2 scanner accepts all seven artifact sets with no unmatched waiver,
the same result as v1. The re-encoding binds the existing reviews to the same binaries; it is not a
review of later source.

### Condition-slice waiver identity, 2026-10-22

Waivers moved from function identity (artifact, instruction locator, and function hash) to condition identity.
Function identity made a waiver stale whenever anything in its function changed,
including the source line numbers in panic locations, which any edit above the function moves.
At `4a52f75c`, 33 of the 42 RISC-V waivers were stale; 31 of their conditions were unchanged.

The 498 waivers were re-encoded on the artifacts they were reviewed against:
[Constant-Time 39554592321](https://github.com/loadingalias/rscrypto/actions/runs/39554592321) at `6b5ea2f1`,
its RISC-V run [39555382384](https://github.com/loadingalias/rscrypto/actions/runs/39555382384),
and the local Apple Silicon build of `6b5ea2f1`.
Each identity is the slice digest of the waiver's own reviewed finding.
302 slices are closed and 196 are open.
50 waivers that covered the same condition in the same function were merged;
each merged pair had the same classification and contract, so 448 waivers remain.
With them, the scanner accepts all seven artifact sets with no stale waiver and unchanged counts.
The one exception is the vendored AArch64 Montgomery kernel, which is still in those binaries,
although `68d942e2` removed it and its waiver.
At `4a52f75c`, 12 RISC-V waivers are stale under condition identity:
4 conditions changed, and 8 open slices bind a function that changed.

The gate also separates evidence from the verdict.
A check whose evidence is complete but whose findings fail the verdict exits with status 3.
Preparation, sealing, and timing continue,
and the Constant-Time workflow's verdict job decides the run from every platform's report.
A finding on one platform no longer cancels timing on the others,
and diagnostic replays run while waivers await review.

### Linked-artifact review, 2026-10-21

The `select_branch` waivers of all seven targets were reviewed again on the linked binaries of `0b66b4ad`,
and the x86-64 waivers again at `4266f7b5`. The RSA performance commits through `929955e7` and the
later harness changes had made 304 of the 457 waivers stale.

The new artifacts came from one Constant-Time run per architecture at `0b66b4ad`:
Linux x86-64 [39545147429](https://github.com/loadingalias/rscrypto/actions/runs/39545147429),
Linux AArch64 [39545154612](https://github.com/loadingalias/rscrypto/actions/runs/39545154612),
Windows x86-64 [39545162420](https://github.com/loadingalias/rscrypto/actions/runs/39545162420),
IBM Z [39545170015](https://github.com/loadingalias/rscrypto/actions/runs/39545170015),
POWER [39545177824](https://github.com/loadingalias/rscrypto/actions/runs/39545177824),
and RISC-V [39545184709](https://github.com/loadingalias/rscrypto/actions/runs/39545184709).
Each run failed at the stale waivers and retained its artifacts.
The Apple Silicon artifact was built locally from the clean `0b66b4ad` tree, with an ld64 map.
`just ct-renew` compared each target with the artifacts of the
[identity v2 re-encoding](#identity-v2-re-encoding-2026-10-21):

- 75 waivers still matched exactly.
- 78 waivers had an unchanged branch and condition slice and were renewed with their existing rationales.
- The other 304 stale waivers were deleted.
- All 347 enforced findings without a waiver were reviewed in their new binaries and received new exact waivers.
  Each rationale names the registers and offsets that define the condition, follows frame-slot reloads
  to their stores, and cites the source line at the reviewed revision.

Of the 347 conditions, 346 are public:

- zeroization tests that compare a buffer address with its public length;
- loop, copy, and odd-limb bounds over public limb counts;
- the PKCS#1 implicit-rejection shift bound, where the secret shift bit forms only a mask;
- the PSS unused-top-bits test and the saved DIT state;
- harness null checks;
- digit-conversion and table-length checks in the AVX-512 IFMA CRT kernel.

The remaining IBM Z condition maps the returned status to a length,
which the `opaque_success_or_failure` disclosure allows.
No reviewed condition depends on a key, limb, plaintext, or padding value. The waiver set now has 500 entries.

The IFMA kernel from `929955e7` had no reviewed linked binary before this review.
In both the Linux and the Windows x86-64 binaries, every conditional branch in its functions,
select-shaped or not, tests a public length, loop counter, iterator index, or overflow check on a public index.
The window selection had a different defect: LLVM folded each table load into the merge-masked `vporq` operand.
The architecture lets a masked memory operand skip its masked-off lanes,
so the secret window could choose which table lines were read.
`4266f7b5` passes each loaded vector through an empty register barrier, so every entry is read
with an unmasked load and the window reaches only the lane mask.
The portable selector already reads every limb with volatile loads for the same reason.
The x86-64 waivers were then renewed against Constant-Time runs
[39549084956](https://github.com/loadingalias/rscrypto/actions/runs/39549084956) (Linux) and
[39549094432](https://github.com/loadingalias/rscrypto/actions/runs/39549094432) (Windows) at `4266f7b5`:
17 renewed with unchanged slices, and one branch with renamed registers received a new waiver.

On an AWS c8i.xlarge host (Intel Xeon 6975P-C, AVX-512F and AVX-512 IFMA) at `4266f7b5`,
each `rsa_crt_pair_*` DudeCT case confirmed that dispatch selected the IFMA kernel.
The largest |t| values were 2.74 (exponent high byte), 1.44 (random exponents), and 1.52 (random inputs),
against the threshold of 10. The internal regression suite passed with native and portable dispatch,
including the comparison of the CRT pair with independent modular exponentiation.
These single-host `ct-dudect` runs are diagnostic; release evidence is the Constant-Time campaign.

The portable Montgomery code still adds 128-bit limb products with `strict_add`.
On x86-64 and RV64, each overflow check compiles to a branch on a carry derived from secret limbs.
No input takes it, because the sum is at most 2^128 − 1, so the branch direction never changes.
These branches are not select-shaped and have no waiver. Converting them to `wrapping_add`,
with the bound written at the definition as [the evidence model](#evidence-model) requires, remains open.

The reviewed linked binary SHA-256 values are:

| Target | Source | SHA-256 |
| --- | --- | --- |
| Apple Silicon | `0b66b4ad` | `bd883d317f1a596c2c7241d9be21c0dd4884babdaa9a53ad0271da80957ec11a` |
| Linux AArch64 | `0b66b4ad` | `1e9ae799368916ba0d8947af4fd2fd5c542e93d239e9b97ed9ab9ea24f15b5ba` |
| POWER | `0b66b4ad` | `4e9d356652410ac1523aa261d4f242d96d503fa6f59126c06b9872e6c4bfca69` |
| IBM Z | `0b66b4ad` | `8afed39def07024d77d9384537e34dc47fc7b95c8309260e1412a8a89912fb83` |
| RISC-V | `0b66b4ad` | `863b207ed46b6ca643ac65b8df3e4395a9ff990aaf56d63589addec468812918` |
| Linux x86-64 | `4266f7b5` | `828925eba6a66dce9a31f77b145a21d7ad87e66fa065fb203c2cd5cc93fb081f` |
| Windows x86-64 | `4266f7b5` | `a68ab698ab62731e942c061bc5416f798a01dad18a113886fec646142daa3b55` |

## Public decisions and exclusions

Ordinary equality is allowed for public values, such as nonces, encoded public keys,
ciphertext lengths, and signature inputs.
Where a comparison must stay opaque until explicit declassification, secret owners return a `CtDecision`.

These operations are outside the constant-time claims on purpose:

- unkeyed hashes, checksums, and XOFs on public data;
- signature verification and public-key parsing;
- RSA prime generation;
- Argon2d, the data-dependent phase of Argon2id, and scrypt memory access;
- caller callbacks, operating-system allocation, thread scheduling, and entropy acquisition;
- external implementations of public traits;
- diagnostic APIs, which expose evidence values on purpose.

Authentication failures stay opaque, also when their inputs are public.

### P-256 ECDH

Scalar sampling and canonical SEC1 validation are public prelude operations.
They are outside the private-arithmetic claim.
When a valid scalar and peer point exist, public derivation and agreement use: fixed loop bounds,
full-table scans for secret digits, masked selection of exceptional points,
and no secret-dependent addresses.

`ct.toml` sets the required linked-binary and target evidence.
A source-level fixed-work design is not a release claim by itself.
The operation entry separates the portable implementation from the selected assembly for Apple
and Linux AArch64, Linux x86-64, and Windows x86-64.
The [September 2026 P-256 ECDH snapshot](../benchmark_results/OVERVIEW.md#p-256-ecdh-development-snapshot) keeps the historical Graviton3, Graviton4,
and Intel Granite Rapids results.
Those development bundles keep binaries and raw timing samples,
but later source changes need new evidence for the exact candidate.

### P-384 ECDH

P-384 ECDH has the same public prelude boundary.

- Public derivation uses a fixed 48-row comb with full 256-entry table scans.
- Agreement uses 77 fixed signed radix-32 windows with full 16-entry table scans,
  masked conditional negation, and masked handling of infinity and equal operands.
- AArch64 replaces the field arithmetic and the comb scan with inline assembly
  that keeps the same fixed schedule.
  It replaces Fermat inversion with a fixed 1116-step Bernstein–Yang divstep inversion.
- x86-64 uses the same divstep inversion and the inline-assembly field kernels.
  Runtime detection selects BMI2/ADX multiplication and fused point doubling.
  Window additions use the portable formula over those kernels.
- Other targets run the portable implementation.

`ct.toml` declares the DudeCT cases and the bounded BINSEC selector kernels.
The full Constant-Time workflow measures the DudeCT cases on every native platform.
A manual run can skip RISC-V; see [Native evidence by platform](#native-evidence-by-platform).

### DudeCT decision

Each required DudeCT case is decided in two steps on the same prepared binary.

1. A screening measurement runs with the case's sample budget.
2. If its |t| reaches `review_fraction` of the threshold, or exceeds the threshold,
   a confirmation measurement runs with `sample_factor` times the samples.
   The confirmation decides the case.

`[dudect_confirmation]` in `ct.toml` sets both values.
Neither step changes a threshold.
With stable distributions and independent samples, a fixed class difference grows
expected |t| by about the square root of the sample ratio.
Four times the samples therefore roughly doubles the expected |t|.
Each measurement launches a new process, immediately after the preceding step
on the same host.
Persistent host or measurement bias can survive confirmation.
A confirmation result does not by itself distinguish that bias from
secret-dependent behavior.
The near-threshold rule also measures borderline passes again, so a real difference that screening
underestimated can fail instead of passing on one draw.
The report keeps the screening result under `screening`.
A screening failure that the confirmation does not reproduce is reported as a `dudect_unconfirmed` diagnostic.

### ECDSA

Recent ECDSA hardening keeps masked point selection on AArch64 and Windows,
masked selection in portable P-256, and table traversal with fixed bounds.
RISC-V generator-table loads stay unconditional under LLVM optimization.
These changes keep signature behavior the same.
They do not establish a timing claim for a release or a downstream build by themselves.

### Secret limb multiplication on RISC-V and IBM Z

Owner decisions, 2026-10-21: ECDSA, ECDH, and RSA private arithmetic, including the RSA blinding
inverse, multiply secret limbs with the hardware multiplier on IBM Z and 64-bit RISC-V.

- IBM Z uses the plain 64-bit by 64-bit `MLGR` product.
  IBM publishes no timing guarantee for it.
- 64-bit RISC-V sets the top bit of both operands, multiplies, and subtracts the correction terms.
  This defends against multipliers that finish early on leading zeros.
  The open RTL of the RISC-V runner's C910 core shows a fixed-latency multiplier.
- 32-bit RISC-V keeps the fixed-work software multiplier, because no runner gives it timing evidence.

The CT artifact gate reports these multiplies for review instead of rejecting them.
Native DudeCT runs on each target qualify the claim; the instruction choice alone does not.

### ML-DSA

ML-DSA has a separate required boundary, `signature.mldsa.secret_kernels`.
Its timing cases, listed in `ct.toml`, cover transforms, products, accumulation, the fused row product
that prepared signing uses for `Â·ŷ`, norm-rejection position, rounding, secret samplers, and valid-key
preparation for ML-DSA-44/65/87.

- Polynomial classes use canonical coefficients.
- Preparation compares one fixed key with a pool of 32 independently seeded valid keys.
- Key generation and input selection happen outside timing.
- Public matrix expansion and redundant-field validation are outside the preparation boundary.
- Retained roots support closure review of the linked binary, including direct tail transfers.
- The bounded BINSEC root analyzes the production portable Montgomery leaf over all operand pairs
  below 2q.
  It does not prove full transforms, accelerated arithmetic, samplers, preparation,
  or hardware instruction latency.

These are requirements for candidate qualification.
They do not state that every target has passed.
Standard signing retries until the first accepted candidate, so it stays variable-time.
Whole ML-DSA signing is best effort, with no strict constant-time claim.

### SLH-DSA

`ct.toml` registers `signature.slhdsa` as best effort.
It has no harness, DudeCT case, or BINSEC kernel yet, so SLH-DSA has no timing claim.

## Native evidence by platform

Windows x86-64 needs the same evidence as the other selected native lanes:
the native timing campaign, the compiler API inventory, artifact validation,
and the cleanup sentinel.
BINSEC proofs for Windows (PE) binaries are not part of the release gate.
"Required" means that the evidence must be collected.
It does not mean that a candidate has passed.
Each release still needs passing native results for the exact source on every selected architecture.
Cross-compilation is not timing proof, and neither is a result from a different microarchitecture.

For RISC-V, POWER, and IBM Z, CI prepares the CT artifacts on x86-64,
then measures the transferred executables on matching native hardware.
The preparation bundle binds the exact source, compiler, binary, disassembly,
and validation evidence.
The timing reports keep both host identities.
Preparation alone gives no timing result.
Transferred execution keeps the same required cases and acceptance thresholds.
See [the transfer workflow](../scripts/README.md#constant-time-evidence).

RISC-V measurement runs on a donated runner pool, in a separate `RISC-V` run.
In a manual run, if no runner starts it within 30 minutes, the job passes with a skip notice,
and that run gives no RISC-V timing evidence.
Release qualification sets `require_riscv` and waits for a runner.

The [Constant-Time workflow](../.github/workflows/ct.yml) runs on manual dispatch and in release qualification.
It does not run on each pull request or push.
Its AWS measurement profiles use fixed On-Demand instances.
They are sized separately from the benchmark profiles in [`.github/runs-on.yml`](../.github/runs-on.yml).
Preparation uses Spot instances, and its completion gives no timing evidence.
Diagnostic replay is separate from the full release campaign.

## Arm data-independent timing

On AArch64 cores with `FEAT_DIT`, the architecture guarantees data-independent timing
for its listed instructions only while `PSTATE.DIT` is set.
On Linux and macOS, it is clear by default.

`rscrypto` sets DIT for the full duration of every asymmetric, post-quantum, and password operation,
then restores the caller's state: X25519, Ed25519, ECDSA, P-256 and P-384 ECDH, ML-KEM, ML-DSA,
SLH-DSA, RSA private operations, Argon2, scrypt, and PBKDF2.
One toggle costs about 30 ns on Apple Silicon,
which is less than 1% of these operations at realistic parameters.

Short symmetric operations do not toggle DIT on each call: MACs and tag verification, AEADs,
keyed hashes, HKDF, and fixed-size `ct_eq`.
For them, a toggle would cost between 14% and 15×.
To use DIT there, wrap the work, or a whole worker loop, in `rscrypto::traits::ct::with_data_independent_timing`.
A unit test checks that every covered operation enters the guard.
DIT is hardening.
It does not replace the evidence in `ct.toml`.
Cores without `FEAT_DIT` run unchanged.

## Power and frequency channels

The constant-time claim covers control flow, memory addresses, and variable-latency operands.
Data-dependent power is outside it, as [`THREAT_MODEL.md`](../THREAT_MODEL.md) states for physical side channels.
Power can still affect timing.
On x86, frequency scaling reacts within milliseconds (Hertzbleed).
POWER9 and POWER10 slow the clock within nanoseconds when current causes a voltage droop.

The POWER CI hosts show this effect.
On some POWER10 hosts, DudeCT probes find vector Montgomery products faster
when every coefficient repeats one value than when the coefficients are random.
Random operands with the same distribution cannot be told apart,
and scalar Montgomery products show no difference.
This is a power effect on those hosts, not a secret-dependent instruction choice.
The `mldsa_probe_*` diagnostic cases in `ct.toml` reproduce it.

ML-DSA secret polynomials have small, heavily repeated coefficients.
On the POWER vector backend, their forward NTT is therefore blinded:
key generation and key preparation transform `s + r`, then subtract `NTT(r)`, where `r` comes from the secret seed `K`.
The outputs do not change.
Other targets transform directly.

The ML-DSA gate measures the operand distribution that production uses.
POWER sends four ML-DSA kernels to its vector unit: forward NTT, inverse NTT, product,
and accumulation.

- The `mldsa_*_dense_fixed_vs_random` cases compare one fixed dense polynomial with new ones, at 200,000 samples.
  They are required on every target,
  because NTT-domain secrets and the signing mask are dense residues.
- The all-zero cases of those four kernels, and of the fused row product, which runs accumulation on
  POWER, are required on every target except POWER.
  On POWER, `diagnostic_targets` in `ct.toml` marks them as non-gating diagnostics and gives the reason:
  production never gives those kernels zero or small repeated secrets on POWER.
  Products, accumulations, and inverse NTTs take NTT-domain operands.
  Small secrets reach the vector NTT only blinded, and the required `mldsa_prepare*` cases measure that.
- The zero scalar Montgomery and scalar inverse NTT cases are required on every target.

ML-KEM decapsulation follows FIPS 203.
It decompresses the ciphertext polynomial before its NTT,
so an attacker cannot choose the sparse NTT inputs that frequency attacks on the inverse NTT need
(Yu et al., CHES 2024).
Prepared decapsulation keys add a second barrier.
Before the inverse NTT, decryption adds a secret dense polynomial,
derived once from the implicit-rejection secret `z`.
The fused final pass of the inverse NTT removes its transform.
The message and the work for each call do not change.
One-shot decapsulation does not mask yet.

See [`secret-ownership.md`](secret-ownership.md) for comparison capabilities,
and [`secret-lifecycle.md`](secret-lifecycle.md) for cleanup evidence.
