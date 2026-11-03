# Provider resource qualification

The 2026-11-03 ML-DSA resource measurements fit the approved heap and shared-key
caps on the measured Apple M1 Pro. Whole-call stack qualification remains open.
The linked graph reaches library imports and indirect calls whose stack use is
not bounded by the retained artifacts. No single-frame measurement, requested
thread stack, or successful execution closes that gate.

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
| 8 MiB stack per worker | No whole-call bound | Open; the requested 8 MiB thread stack is not a measurement |

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
stack allocation. They are not whole-call bounds.

| Linux target | Largest root frame, native / portable | Largest recorded-frame path sum, native / portable | Whole-call bound |
| --- | ---: | ---: | --- |
| x86-64 GNU | 136 / 136 B | 2,024 / 1,256 B | Unresolved |
| AArch64 GNU | 304 / 304 B | 2,080 / 1,520 B | Unresolved |
| POWER64LE GNU | 2,496 / 400 B | 9,920 / 1,824 B | Unresolved |
| IBM Z GNU | 184 / 184 B | 2,008 / 1,776 B | Owned unresolved; portable borrowed subset below |
| RISC-V 64 GNU | 224 / 224 B | 1,440 / 1,440 B | Unresolved |

The path sums are diagnostic sums of available records, not upper bounds.
Of 120 reviewed roots, 114 retain `whole_call_bound_bytes: null`. The six
portable IBM Z borrowed roots are bounded: 1,416 B for preparation, or 1,600 B
including successful prepared-key destruction, for each parameter set. These
contain no unresolved library calls. Caller storage and enclosing worker/TLS
frames are outside those bounds. Every owned root remains unresolved, so the
worker qualification gate still fails. The tool continues below missing frame
records, resolves
identified GOT/PLT imports, distinguishes IFUNC relocations, and retains
unresolved transfers by instruction address. Regression fixtures cover missing
frames, allocator cleanup, IFUNCs, branch joins, register clobbers, unlabelled
IBM Z PLT entries, and repeated assembly stack decrements.

The exact remaining proof needs are:

- The deployment's `calloc`/`free`, memory routines, loader/IFUNC implementations,
  and their callbacks, with final library identities. The executable's import
  name cannot bound whichever implementation the loader selects.
- Indirect dispatch and cold capability-detection paths, including standard
  library callbacks and syntactically reachable cleanup/unwind paths. A warm
  capability cache cannot silently remove the cold path from the contract.
- Enclosing provider/TLS/worker frames, and native Apple/Windows stack evidence.
  The frame tool currently reviews Linux ELF only. Borrowed storage must also
  be charged to its caller when the caller places it on the stack.

Aborting panic/OOM exits are listed separately; normal and recoverable-error
paths remain in the graph. No unknown callee is assigned a zero-stack bound.

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

Both preparation reviews deliberately exit nonzero while unresolved edges
remain. The legacy scrub review passes on all ten retained artifacts; its
explicit memory-leaf assumptions do not apply to the preparation gate.
`check.log`, `test-scripts.log`, `frames-tests-final.log`, and
`scrub-regressions/` retain the corresponding validation output.
