# Provider resource qualification

The 2026-11-03 ML-DSA resource measurements fit the approved heap and shared-key
caps on the measured Apple M1 Pro. On the five reviewed Linux targets, every
whole-call root has a whole-call stack bound: its static owned bound composed
with an allowance measured for each libc edge it reaches on the named glibc
deployment. The provider TLS worker's painted-stack high-water mark is measured
on every native target, Apple and Windows included
([Measured stack use](#measured-stack-use)). Both are evidence for the measured
hosts and paths, not static proofs of glibc or rustls.

This record covers Gap 2 in the local delivery plan. It does not restart the
deferred performance campaign or qualify G1-D traffic.

## Source and evidence

The production library source is commit
`6926f298` (`build: renew exact CT condition reviews for AES/TLS`). The working
change adds evidence tooling and a private allocator regression; it changes no
production cryptographic operation, dependency pin, or dispatch policy.

The compiler is `nightly-2026-10-24`, rustc
`1.101.0-nightly (48f4b0553 2026-10-23)`, LLVM 23.1.4. Runtime measurements used
`aarch64-apple-darwin` on an Apple M1 Pro, MacBookPro18,3, with ten logical CPUs.
Resource builds use the provider benchmark release profile and
`-C target-cpu=native`, with `ml-dsa`, then `ml-dsa,portable-only`.

Raw evidence is retained locally in
`docs/tasks/evidence/gap2-20261103/`. Each resource row retains its command,
source file hashes, compiler/host identity, executable and hash, raw JSON,
stderr, build log, exit code, and `SHA256SUMS`. `resource-runs.json` lists the
42 invocations; `resource-summary.json` indexes their results and hashes.
`stack-reviewed-native/` and `stack-reviewed-portable/` retain ten linked Linux
artifacts, disassemblies, relocations, symbol/frame tables, complete parsed call
graphs, source hashes, and reports. These cross-built artifacts were not run.
`stack-summary.json` confirms that their binaries match those used by the
separate cleanup review. Earlier failed parser runs remain in the evidence
directory; the `stack-reviewed-*` reports supersede their graph analysis.
`stack-declared-native/` and `stack-declared-portable/` retain the declared-edge
reports, their `SHA256SUMS`, and the run logs. They come from a 2026-11-03
rebuild with the same compiler, at `1d8dadfe` plus the declared-edge tooling, in
another checkout. Their binaries differ by hash from the `stack-reviewed-*`
binaries, but every full-graph frame sum and unresolved-edge count is identical.

## Production paths and storage

All three advertised parameter sets follow the same provider path in
[`sign/mldsa.rs`](../rscrypto-rustls/src/sign/mldsa.rs):

1. `Provider::load_private_key` bounds and owns the DER input. ML-DSA loading
   uses `from_pkcs8_der_in(der, Global)` and obtains the validated public key.
2. The default loader calls `prepare_in(Global)`, shares the prepared key by
   `Arc`, and drops the compact secret key. Compact mode retains that compact
   owner instead.
3. A prepared signer borrows the shared prepared key. A compact signer calls
   `prepare_in(Global)` for each signature and drops that temporary on both
   success and recoverable signing failure. Both use hedged signing and the
   empty FIPS 204 context required by the TLS profile.

In [`src/auth/mldsa.rs`](../src/auth/mldsa.rs), borrowed `prepare` writes into
caller-owned storage without allocating. `prepare_in` allocates that storage
once and decodes directly into it. Optimized Unix artifacts call `calloc` for
the zeroed allocation and `free` after destruction. The pinned Rust standard
library's `sys/alloc/unix.rs` confirms these allocator paths. Allocation failure
has the documented `Box::new_in` behavior; it is not a recoverable `MlDsaError`.

| Parameter set | Prepared signing storage, requested bytes | Compact owner, including public key | Prepared provider key retained, usable bytes | Compact provider key retained, usable bytes |
| --- | ---: | ---: | ---: | ---: |
| ML-DSA-44 | 28,800 | 3,872 | 34,400 | 5,728 |
| ML-DSA-65 | 48,256 | 5,984 | 51,296 | 8,288 |
| ML-DSA-87 | 81,024 | 7,488 | 85,088 | 11,360 |

The last two columns include allocator rounding, SPKI, the shared owner, and
the boxed provider key. The secret encodings alone are 2,560 / 4,032 / 4,896
bytes; they are not the compact owners' total storage. Owned-DER loading made
six allocation calls in prepared mode and five in compact mode. One signature,
including signer construction and the returned signature, made two calls in
prepared mode and three in compact mode. Destruction allocated nothing and
released all four retained provider-key allocations in these observations.
These are storage/allocation costs, not latency measurements.

Preparation remains configuration-bound. Trust-anchor preparation is owned by
the configured verifiers. Peer CertificateVerify keys and intermediate keys
are imported for their operations; this work introduces no peer-key cache.
Rail still owns admission, scheduling, eviction, cancellation, locked storage,
and shutdown. An arbitrary user-supplied allocator has no generic stack bound.

## Measured heap and admission

The matrix uses X25519MLKEM768, all three TLS 1.3 suites, all three ML-DSA sets,
prepared and compact signing, and both production-auto and portable-only
dispatch. Supplementary ML-DSA-87/AES-256 rows cover intermediate chains,
stateless resumption, and stateful resumption in both dispatch configurations.
Every row checks eight individual pairs and holds 64 concurrent pairs live.
Each pair contains both TLS endpoints; its heap is never divided by two.

| Approved limit | Largest measured increment | Result and boundary |
| --- | ---: | --- |
| 1 MiB incremental TLS heap per connection | 400,752 B per pair | Pass for the measured rows; includes both endpoints and transport |
| 64 simultaneous handshakes, 64 MiB aggregate connection allowance | 24,805,744 B for 64 live pairs | Pass for the measured rows |
| 8 MiB shared prepared-key storage | 782,512 B configuration-construction peak | Pass for these configurations; includes keys, verifiers, and a warmup connection |
| 8 MiB stack per worker | 56,928 B painted high-water of one TLS worker | Pass for the measured scenarios and hosts ([Measured stack use](#measured-stack-use)) |

Every row rejects an extra admission before constructing TLS state and returns
all 64 permits after destruction. This exercises the harness's caller-owned
bounded channel. It does not install an application admission limit.

The observer counts live System allocator usable payload bytes, including
simultaneously live old/new buffers during moving reallocation. It excludes
allocator arena metadata, C-library allocations, and OS thread stacks. The
64 stack requests total 512 MiB outside the heap allowance; actual mappings
and guard pages are operating-system dependent.
Random signing retries change work, but this matrix does not establish a
universal hostile-input memory or denial-of-service bound. Other deployment
targets still need matching native resource evidence; the standard configured
AWS x86-64/AArch64 machines had no instances during this run.

## Static stack evidence

`just stack-frames --mldsa-preparation --evidence DIR` reviews preparation and
preparation-plus-destruction roots for every set. Repeating with `--portable`
selects the other dispatch configuration. The release profile uses fat LTO,
one codegen unit, overflow checks, and panic abort. `-Z emit-stack-sizes` adds
frame records; no target CPU override is supplied. As the
[Rust compiler documentation](https://doc.rust-lang.org/nightly/unstable-book/compiler-flags/emit-stack-sizes.html)
explains, frame records do not include callees and can be absent for dynamic
stack allocation. They are not whole-call bounds. On x86-64 and AArch64, code
without a record, such as hand-written assembly, gets a frame derived from its
instructions only when every stack-pointer change is bounded; each derived
frame is listed as an assumption, and any other recordless code stays
unresolved.

Each root has two results. The owned bound covers the code linked into the
artifact: rscrypto and the linked parts of the Rust standard library. It stops
at the declared libc edges that
[`scripts/README.md`](../scripts/README.md#preparation-bounds) lists, and
charges each edge only its caller's frame and call slot. Cold first-call
capability detection is owned code and stays inside the owned bound: the
lock-free cache behind `platform::caps()`, CPUID, STFLE, and the AArch64
`DataIndependentTiming` probe. Its only declared edge is the libc
`getauxval` read on AArch64, POWER, and IBM Z; detection reads no environment
variable. The report gives the maximum stack depth at each edge's
entry over all paths. The whole-call bound composes the owned bound with a
measured allowance for each reached edge.

Largest owned bound by root, in bytes, native / portable:

| Linux target | `prepare_in` | `prepare` | `prepare_in`, then drop | `prepare`, then drop |
| --- | ---: | ---: | ---: | ---: |
| x86-64 GNU | 2,008 / 1,224 | 1,960 / 1,096 | 2,040 / 1,256 | 1,992 / 1,128 |
| AArch64 GNU | 2,032 / 1,472 | 1,824 / 1,184 | 2,080 / 1,520 | 1,872 / 1,232 |
| POWER64LE GNU | 9,840 / 1,744 | 9,840 / 1,408 | 9,920 / 1,824 | 9,920 / 1,488 |
| IBM Z GNU | 1,824 / 1,592 | 1,648 / 1,416 | 2,008 / 1,776 | 1,832 / 1,600 |
| RISC-V 64 GNU | 1,408 / 1,408 | 1,200 / 1,200 | 1,440 / 1,440 | 1,232 / 1,232 |

Largest stack depth at a declared edge's entry, in bytes, native / portable.
A dash means that no root reaches the edge:

| Linux target | `calloc` | `free` | `memcpy` | `memset` | `getauxval` |
| --- | ---: | ---: | ---: | ---: | ---: |
| x86-64 GNU | 184 / 184 | 40 / 40 | 1,720 / 1,000 | - / - | - / - |
| AArch64 GNU | 352 / 352 | 48 / 48 | 1,504 / 1,360 | - / - | 2,080 / 400 |
| POWER64LE GNU | 2,576 / 480 | 112 / 112 | 8,224 / 1,536 | 8,048 / 1,536 | 7,888 / - |
| IBM Z GNU | 360 / 360 | 184 / 184 | - / - | - / - | 2,008 / - |
| RISC-V 64 GNU | 256 / 256 | 32 / 32 | 1,184 / 1,184 | 1,184 / 1,184 | - / - |

Only the `prepare_in` roots, with or without the drop, reach `calloc` and
`free`. No root reaches `malloc`, `memmove`, `memcmp`, or `bcmp`. Each parameter
set reaches the same edges; the tables give the maximum over the three sets.

All 120 roots have an owned bound. On every target and configuration, the
owned graphs contain no other import, IFUNC relocation, indirect transfer,
missing frame record, or recursion. Owned bounds are not whole-call bounds;
[Measured stack use](#measured-stack-use) composes them with the measured libc
allowances. The six portable IBM Z roots that call `prepare` reach no declared
edge and keep their static whole-call bounds: 1,416 B for preparation, or 1,600 B
including successful prepared-key destruction, for each parameter set. Caller
storage and enclosing worker/TLS frames are outside every whole-call bound.

The full-graph traversal still descends into every declared edge and lists what
an allowance must cover. Native AArch64, POWER, and IBM Z reach `getauxval`
through capability detection, and portable AArch64 reaches it through the DIT
probe; below it, the traversal finds only the libc import. No detection path
reaches the standard library's `Once`, a futex, an indirect call, or unwinding.
Native x86-64 detection is owned code with no declared edge: CPUID, the cache,
and the in-code `platform::expert::allow_hybrid_avx512` setting, which replaced
the `RSCRYPTO_FORCE_AVX512` environment read. That read's `std::env::var`
subgraph reached indirect calls, `syscall`, unwinding without frame records,
`getenv`, `malloc`, and `dlsym`, so its edge could not be bounded without a
measured allowance. After its removal, the x86-64 reviews were repeated: every
x86-64 bound and entry depth above is unchanged. x86-64, portable POWER,
portable IBM Z, and both RISC-V configurations reach no detection edge, so
their roots need only libc allowances.

`--whole-calls all` adds 27 roots per configuration beyond preparation:
ML-DSA deterministic signing, prepared hedged signing, and verification; ML-KEM
key generation, encapsulation, decapsulation, and prepared decapsulation; and
in-place AEAD record seal and open. Largest owned bound, native / portable,
with the number of roots that have one:

| Linux target | ML-DSA signing (9 roots) | ML-KEM (12 roots) | AEAD records (6 roots) |
| --- | ---: | ---: | ---: |
| x86-64 GNU | 9 / 9, 59,112 / 58,552 B | 12 / 12, 33,976 / 33,880 B | 6 / 6, 3,736 / 3,352 B |
| AArch64 GNU | 9 / 9, 59,392 / 58,720 B | 12 / 12, 38,256 / 36,688 B | 6 / 6, 3,472 / 3,472 B |
| POWER64LE GNU | 9 / 9, 59,360 / 59,280 B | 12 / 12, 34,416 / 34,416 B | 6 / 6, 4,496 / 3,632 B |
| IBM Z GNU | 9 / 9, 59,080 / 59,080 B | 12 / 12, 33,960 / 31,456 B | 6 / 6, 3,888 / 3,776 B |
| RISC-V 64 GNU | 9 / 9, 58,752 / 58,752 B | 12 / 12, 31,728 / 31,728 B | 6 / 6, 3,232 / 3,280 B |

AEAD records call their kernels directly. The `polyval` multiplication, the
Poly1305 block function, and the ChaCha20 keystream each test the cached
capabilities and call the selected kernel; none stores a function pointer. The
x86-64 Linux VAES-512 batch loop calls each kernel width in its own branch. No
AEAD graph, on any target or configuration, reaches the standard library's
`Once`, a futex, an indirect call, or unwinding. Earlier trees routed every
AEAD root through a std `OnceLock` of function pointers, which ran `Once`
(indirect calls, futex `syscall`, no frame records) and made each dispatched
call indirect. The AES-GCM bounds are dominated by the fixed 3 KiB frame of
the CTR scrub, `backend::stack::scrub_dead_stack::<384>`, which clears the stack
below the CTR worker. The AEAD columns of both tables come from `just ci-stack`
on 2026-11-09, after that scrub shrank from 8 KiB; the other columns keep their
earlier reviews. The roots
reach `memcpy` and `memset` on every target except IBM Z, and `getauxval`
through first-call detection on native AArch64, POWER, and IBM Z. Portable IBM
Z AEAD roots reach no declared edge and have static whole-call bounds: 3,760 B
for AES-GCM seal, 3,776 B for AES-GCM open, and 856 B for ChaCha20-Poly1305.

CI enforces these owned bounds. Its `stack` row reviews all 39 whole-call roots
of both builds on the five Linux targets and fails when a bound is unresolved
or exceeds its budget in
[`.config/stack-budgets.json`](../.config/stack-budgets.json): the reviewed
bound plus 12.5%, rounded up to 1 KiB. A reviewed change that moves a bound
updates that file in the same commit
([scripts guide](../scripts/README.md#stack-budgets)).

Every AEAD root has an owned bound on every target and configuration. Owned
assembly has no compiler frame record, so the tool derives its frames from the
instructions and lists each one among the report's assumptions (scripts guide,
"Frame review"). The twelve x86-64 Linux VAES-512 GCM kernels are leaves of
648, 1,160, and 2,184 B for 16-, 64-, and 128-block batches: two pushes, a
constant subtraction, at most 56 B to align `rsp` to 64 bytes, and a reload of
`rsp` from the slot that saved it. Portable builds link them too, because a key
stores its selected backend. The x86-64 ChaCha20-Poly1305 seal and open
routines share one 376 B frame, seven pushes and 320 B, with their AVX2
continuations, which jump back into the SSE routines. ELF-visible labels split
the AArch64 ChaCha20-Poly1305 routines into 151 (seal) and 28 (open) symbols;
their frames are 112 B and 80 B.

Every signing and verification root has an owned bound on every target and
configuration. On x86-64, IBM Z, and RISC-V, `mldsa::encoding::pack` and
`unpack_raw` reach their width-specific packers through compiler jump tables.
The tool reads each table from the linked image and resolves the register jump
only when an unsigned bound check precedes it, every instruction in between is
a table step or a store, no branch enters that span, and every entry for every
checked index lands on an instruction of the same function. That function's
frame record and calls already cover every destination; each resolved table is
listed among the assumptions. Portable IBM Z signing and verification reach no
declared edge, so they also have static whole-call bounds, up to 59,080 B for
ML-DSA-87 deterministic signing. Every ML-KEM root now has an owned bound: the tool
follows a GOT-loaded import through full-width copies into caller-saved
registers, which calls and implicit register writers end.

These results come from the tree that reads the capability cache once on both
paths (2026-11-04). An inlined `platform::caps()` check no longer copies the
whole `Detected` value onto the stack, so AArch64 preparation bounds drop by
48 B and other hot frames shrink. The detected value now waits in the cold
`initialize` frame instead, which raises the IBM Z bounds that run through
first-call detection by 40 B and the POWER `getauxval` entry by 16 B. The
AArch64 DIT guard is a load, a zero test, and a bit test, and its probe is a
separate cold function, which adds 16 B to the portable `getauxval` entry and
the native signing bound.

The tool continues below missing frame records, resolves identified GOT/PLT
imports, distinguishes IFUNC relocations, and retains unresolved transfers by
instruction address. Regression fixtures cover missing frames, allocator
cleanup, IFUNCs, branch joins, register clobbers, unlabelled IBM Z PLT entries,
repeated assembly stack decrements, assembly frames shared across labels and
jump continuations, x86-64 `rsp` save-slot reloads, data after a final return,
checked jump tables on x86-64, IBM Z, and RISC-V, declared-edge entry depths after calls and tail calls, register copies of held
imports, detection without a declared edge, and allowance composition.

The executable's import name cannot bound whichever implementation the loader
or an IFUNC resolver selects, so each allowance is measured on its deployment,
and a warm capability cache cannot remove the cold path from the contract: the
composition charges every reached edge at its deepest entry. The frame tool
reviews Linux ELF only; Apple and Windows have painted-stack worker evidence
instead. Borrowed storage must be charged to its caller when the caller places
it on the stack.

Aborting panic/OOM exits are listed separately; normal and recoverable-error
paths remain in the graph. No unknown callee is assigned a zero-stack bound.

## Measured stack use

Static owned bounds stop at the declared libc edges. `tools/stack-paint` measures
what those edges use on the deployed libc, and what the provider TLS worker
uses on every native target, on painted, guarded private stacks
([scripts guide](../scripts/README.md#painted-stacks)). These are measurements of
the reached paths on the named hosts, not static proofs of glibc, rustls, or
every input.

### libc allowances

`just stack-allowances` calls each declared edge through the symbol the
executable binds, so the loader's IFUNC selection for the host CPU applies. An
allowance is the deepest use below the edge's entry stack pointer over the
measured calls, rounded up to 16 bytes, plus the ABI red zone, then 12.5% more,
rounded up to 64 bytes. [`.config/stack-allowances.json`](../.config/stack-allowances.json)
records each target's allowances and libc identity.

| Linux target | `calloc` | `malloc` | `free` | `memcpy` | `memmove` | `memset` | `memcmp` | `bcmp` | `getauxval` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| x86-64 GNU | 576 | 576 | 576 | 192 | 192 | 192 | 192 | 192 | 192 |
| AArch64 GNU | 640 | 576 | 512 | 0 | 0 | 0 | 0 | 0 | 64 |
| POWER64LE GNU | 896 | 896 | 960 | 384 | 384 | 384 | 384 | 448 | 384 |
| IBM Z GNU | 1,216 | 1,152 | 1,024 | 0 | 0 | 0 | 0 | 0 | 0 |
| RISC-V 64 GNU | 576 | 576 | 512 | 128 | 128 | 64 | 64 | 64 | 64 |

The measured deployments are Ubuntu 24.04 glibc 2.39 on the CI runners of
CI 42061504225 at `9211e330`: x86-64 on AMD EPYC 9R14 (2.39-0ubuntu8.8),
AArch64 on Graviton3 (2.39-0ubuntu8.8), POWER10, IBM z15, and RISC-V
(2.39-0ubuntu8.9). For x86-64 and AArch64, the committed value is the larger
of that run and an Ubuntu 26.04 glibc 2.43 (2.43-2ubuntu2.4) measurement on the
Granite Rapids and Graviton4 dev machines. Each entry names its exact libc
build hash, kernel, and CPU; `--check` reports a different identity.

The memory functions and `getauxval` are frameless or nearly so on every
measured target. The allocator edges are deepest in the pseudo-random workload
and the first calls on a fresh thread, which create and attach an arena. Their
failure returns, `free(NULL)`, frees of other threads' chunks, and bulk release
are also measured. Allocator paths the workload does not reach, such as heap
corruption reports, are not covered; they abort.

### Whole-call bounds

`just ci-stack` composes these allowances with the owned bounds:
`max(owned bound, max over reached edges of (entry depth + allowance))`. Every
`--whole-calls all` root on all five Linux targets, native and portable, now
has a whole-call bound, and CI fails a root without one or above 8 MiB.

Largest whole-call bound by root group, in bytes, native / portable:

| Linux target | ML-DSA preparation | ML-DSA signing | ML-KEM | AEAD records |
| --- | ---: | ---: | ---: | ---: |
| x86-64 GNU | 2,040 / 1,256 | 59,112 / 58,552 | 33,976 / 33,880 | 3,736 / 3,352 |
| AArch64 GNU | 2,144 / 1,520 | 59,456 / 58,720 | 38,288 / 36,688 | 3,472 / 3,472 |
| POWER64LE GNU | 9,920 / 1,920 | 59,360 / 59,280 | 34,416 / 34,416 | 4,496 / 3,632 |
| IBM Z GNU | 2,008 / 1,776 | 59,080 / 59,080 | 33,960 / 31,456 | 3,888 / 3,776 |
| RISC-V 64 GNU | 1,440 / 1,440 | 58,752 / 58,752 | 31,728 / 31,728 | 3,232 / 3,280 |

The entry depths at `memcpy` and `memset` sit inside the deepest owned frames,
so the allowances raise only the roots whose deepest point is a libc call.
The bounds apply to the measured libc identities and kernels only. Other glibc
builds, musl, signal handlers, arbitrary caller-supplied allocators, and caller
frames outside the call need their own evidence.

### Provider TLS worker

`just stack-worker` runs each of the 483 scenarios of
`tools/stack-paint/src/bin/worker.rs` in a fresh process, on an 8 MiB painted
stack, natively on each target. The first call builds both endpoint
configurations, which loads and prepares keys and runs first-call capability
detection, then drives the connections in memory: handshake, about 16 KB of
application data each way, close notification, and destruction. A second call
repeats the connections with the built configurations. Scenarios cover every
credential, key exchange group, and suite; mutual and server-only
authentication; prepared and WebPKI verifiers; intermediate chains; ticket and
stateful resumption; HelloRetryRequest; compact ML-DSA keys; TLS 1.2; and each
recoverable failure: corrupted signatures in both roles, untrusted and expired
certificates, corrupted handshake and application records, invalid key shares
in both directions, failing client and server entropy at each request, a
foreign ticket key, no common group or suite, and a truncated private key.
Every scenario ended as expected on every target.

Deepest painted-stack high-water mark over all scenarios, in bytes, and its host:

| Native target | Host | Scenarios | Deepest cold, native / portable | Deepest repeated, native / portable |
| --- | --- | ---: | ---: | ---: |
| x86-64 GNU | AMD EPYC 9R14 (CI) | 483 | 53,856 / 53,856 | 50,848 / 50,848 |
| AArch64 GNU | Neoverse V1, Graviton3 (CI) | 483 | 54,656 / 54,288 | 51,664 / 51,296 |
| POWER64LE GNU | POWER10 (CI) | 483 | 55,392 / 55,312 | 52,288 / 52,208 |
| IBM Z GNU | IBM z15 T02, machine 8562 (CI) | 483 | 54,112 / 54,112 | 51,808 / 51,792 |
| RISC-V 64 GNU | riscv64, Scaleway (CI) | 483 | 53,904 / 53,904 | 50,960 / 50,960 |
| x86-64 Windows MSVC | AMD Zen 4 (CI) | 483 | 56,928 / 54,864 | 53,920 / 51,856 |
| AArch64 macOS | Apple M1 Pro (local) | 483 | 54,208 / 54,032 | 51,216 / 51,040 |


The deepest scenario everywhere is ML-DSA-87 mutual authentication, where the
client's signature fails verification: its signing frame dominates, as in the
static review. The measured high-water is below the static owned bound for
ML-DSA-87 signing because the static bound takes the deepest path through
every branch. Both are far below the 8 MiB worker cap.
[`.config/stack-worker-budgets.json`](../.config/stack-worker-budgets.json)
holds each configuration's budget, the measurement plus 12.5% rounded up to
1 KiB, and CI fails a scenario that ends unexpectedly or a configuration that
exceeds its budget. The worker frames above the provider are this harness's
in-memory driver; an application's own frames add to them.

## Cleanup and timing

The private allocator regression invokes production `prepare_in` for all three
sets. It observes nonzero prepared state first, then checks transformed
polynomials and all 128 seed bytes at the live allocation immediately before
deallocation, on success and recoverable invalid-key failure. Native and
portable release tests pass, as does Miri's check of that observer. Existing
borrowed-storage regressions check failure and reuse. Provider regressions
cover partial entropy failure and signing after failure.
The existing all-set signing-exhaustion regression also passes in native and
portable release builds: forced exhaustion clears the output, then reused
state produces the official ACVP signature with valid parameters.

`cleanup/` retains source/MIR/IR/assembly analysis and linked success/error/drop
excerpts. Detailed linked wipe review covers native x86; the other target
artifacts retain destructor reachability, not a complete wipe-dominance proof.
The external audit tool's semantic Rustdoc probe failed and its assembly parser
recognized no functions. Those probes remain incomplete and are not credited as
passing checks. The public matrix is intentionally not secret-cleared. Heap-owner
cleanup does not establish clearance of every compiler spill or register.

`just stack-residue --native-mldsa --output DIR` passed 96 production signing
calls per dispatch configuration, 192 total, on private painted 1 MiB stacks.
It covers selected masks, NTT masks, rejected responses/commitments, and hint
maps, with planted/cleared controls and signatures matched to the isolated
capture build. It is residue observation, not a stack high-water measurement
or a proof about every preparation exit, heap byte, register, or shorter secret
fragment. Raw snapshots and capture/build identities are in `residue/`.

The separate full Apple CT report passes all 135 required cases, with
4,024,528 samples and a largest absolute t-statistic of 3.803
against the unchanged threshold of 10. `apple-ct-qualified/` retains the complete
report, raw samples, binaries, and provenance; `apple-condition-review/` retains
the fresh review of sixteen existing public RSA conditions. Only those exact
condition bindings changed in `ct.toml`. The run uses the production commit above
plus that manifest overlay in `/private/tmp/rscrypto-gap2-ct-6926f298`; its path
and compiler are part of the artifact identity. The original report with stale
bindings remains unchanged in `apple-ct/`.

Artifact compilation overlapped an unrelated build. The CT orchestrator was
paused before timing until that project's full gate ended; the host-window
receipt records the reserved timing period. Existing green workflows are
accepted without reconciliation or polling. ML-DSA secret-kernel timing evidence
does not change standard signing's variable-retry boundary or create a
whole-signing constant-time claim. G1-D traffic qualification remains deferred.

## Validation and reproduction

`just check` passed across the configured target matrix, including independent
workspaces, dependency policy, and documentation. `just test-scripts` passed;
the final frame-parser changes also passed `python3 scripts/stack/frames_test.py`.
The native/portable core and provider regressions and Miri commands are retained
in `cleanup/tests/`. The final import-only test correction uses the same allocator
types through their defining `core::alloc` module; its exact equivalence receipt
is retained there.

To reproduce the resource matrix, copy the retained `run-resources.py` into a
new evidence directory and execute that copy from the repository root. It writes
beside itself; `resource-runs.json` records each invocation. The individual
front door is `just bench-provider-resources`; `--help` describes the supported
credentials, modes, workloads, repetition limits, and output option.
The retained `resource-runs.json` identifies the two valid rows reused by the
matrix script instead of measuring them twice.

The full Apple command is `CARGO_RAIL_CACHE=off just ct-full --target
aarch64-apple-darwin`. `run-apple-qualified.py` records its exact checkout and
manifest overlay. Reproducing its exact artifact identity requires production
commit `6926f298c16d3d6777973e6fa5d29ad86e0ca48d`, the retained `ct-after.toml`,
the pinned compiler, and the recorded checkout path. Use a quiet host and retain
a fresh output bundle; a changed path or compiler may require another condition
review. `apple-ct-summary.json` binds the passing report and timing binary hashes.

For the static and residue evidence, use new output directories:

```sh
just stack-frames --mldsa-preparation --evidence /path/to/new-native-frames
just stack-frames --mldsa-preparation --portable --evidence /path/to/new-portable-frames
just stack-residue --native-mldsa --output /path/to/new-residue
```

The preparation reviews exit nonzero while a root has an unresolved owned edge
or reaches a declared edge without an allowance; pass
`--allowances .config/stack-allowances.json` to compose the committed ones.
`just stack-allowances --check` and `just stack-worker --all-configurations
--check` repeat the measurements on a host against the committed files. The legacy scrub review passes on all ten retained artifacts; its
explicit memory-leaf assumptions do not apply to the preparation gate.
`check.log`, `test-scripts.log`, `frames-tests-final.log`, and
`scrub-regressions/` retain the corresponding validation output.
