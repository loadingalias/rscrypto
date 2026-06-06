# Constant-Time Validation Status

Internal task file for the rscrypto constant-time evidence pipeline.

Source snapshot: 2026-06-06.

## Thesis

We are not building a universal constant-time verifier. We are building a
release evidence pipeline that can make narrow, reproducible claims about exact
compiled artifacts.

The claim boundary is:

```text
LLVM backend only, declared secrets only, declared public inputs only, exact
target/build configuration only, and only where the required evidence gates
pass for that target.
```

The policy lives in [`docs/constant-time.md`](../constant-time.md). The
machine-readable source of truth is [`ct.toml`](../../ct.toml).

## Current Release Boundary

The current hard gate is:

```bash
just ct-check
```

`just ct-check` and `just ct-full` are intentionally the same command path.
They build artifacts, validate manifest coverage, run DudeCT, run BINSEC where
the target requires it, and emit release-style reports.

Current platform scope:

| Target class | Current CT evidence |
|---|---|
| Linux `x86_64`/`aarch64`, GNU and MUSL | Artifact/heuristics + DudeCT + BINSEC. |
| Linux `powerpc64le`/`riscv64gc` | Artifact/heuristics + DudeCT + BINSEC, subject to runner/tool completion. |
| Linux `s390x` | Artifact/heuristics + DudeCT. BINSEC is not claimed for s390x today. |
| macOS `aarch64`/`x86_64` | Artifact/heuristics + DudeCT. BINSEC is not claimed for Mach-O today. |
| Windows MSVC `x86_64`/`aarch64` | Artifact/heuristics + DudeCT. BINSEC is not claimed for PE/COFF today. |
| `no_std` bare-metal | Not release-claimed today. |
| WASM | Not release-claimed today. |

The practical rule: if a native target ships an accelerated ASM, SIMD,
hardware-instruction, or portable fallback path for a CT-critical primitive,
that path must be represented in the manifest and must pass the required gates
for that target class.

## Done

### Policy And Claim Language

- Wrote the CT policy in [`docs/constant-time.md`](../constant-time.md).
- Updated [`docs/security.md`](../security.md) with the public CT claim
  boundary.
- Defined the strict leakage rule:
  public length may leak; secret-dependent branch, address, table index,
  variable loop bound, divisor, panic path, allocation behavior, formatting,
  dispatch, and secret parsing rejection timing may not leak.
- Scoped current claims to LLVM only. Cranelift and GCC codegen are deferred.

### Manifest

- Created [`ct.toml`](../../ct.toml) as the source of truth.
- Classified CT-critical primitives and target policies.
- Declared required evidence profiles.
- Added manifest-driven DudeCT cases.
- Added manifest-driven BINSEC kernel rows for supported Linux targets.
- Added target policy rows that explicitly exclude macOS Mach-O, Windows
  PE/COFF, `no_std`, WASM, and s390x BINSEC from current release claims.

### Harnesses

- Added the artifact harness in [`tools/ct-harness`](../../tools/ct-harness).
- Added the DudeCT harness in [`tools/ct-dudect`](../../tools/ct-dudect).
- Added the BINSEC harness in
  [`tools/ct-binsec-harness`](../../tools/ct-binsec-harness).
- Added stable leaf entrypoints for CT-critical kernels so checkers do not have
  to start from ergonomic public APIs with parsing, allocation, dispatch, and
  error conversion mixed in.

### Gate 1: Artifacts And Heuristics

- Implemented artifact generation through
  [`scripts/ct/artifacts.sh`](../../scripts/ct/artifacts.sh).
- Captured provenance, LLVM IR, assembly, object disassembly, symbol maps,
  artifact hashes, and evidence indexes.
- Implemented assembly/object heuristics through
  [`scripts/ct/asm_heuristics.py`](../../scripts/ct/asm_heuristics.py).
- Made strict manifest/artifact validation part of the full gate through
  [`scripts/ct/validate.py`](../../scripts/ct/validate.py).

Gate behavior:

- Missing required artifacts fail.
- Missing manifest coverage fails.
- Hard heuristic findings fail unless explicitly waived.
- Heuristic warnings remain review evidence and must be audited before a
  release claim.

### Gate 2: DudeCT

- Wired the repo-local DudeCT runner through
  [`scripts/ct/dudect.sh`](../../scripts/ct/dudect.sh).
- Normalized DudeCT output through
  [`scripts/ct/dudect_report.py`](../../scripts/ct/dudect_report.py).
- Made every manifest-declared DudeCT case run from
  [`scripts/ct/full.py`](../../scripts/ct/full.py).
- Made missing required DudeCT coverage, timeout, crash, or threshold failure a
  blocker in `ct-check`.

Local status:

- `just ct-full --target aarch64-apple-darwin --dudect-timeout 300` passes on
  local Apple Silicon.
- Latest local report shape: `40` DudeCT cases, `43` artifacts, `0` report
  issues for `aarch64-apple-darwin`.

### Gate 3: BINSEC

- Integrated BINSEC directly through
  [`scripts/ct/binsec.py`](../../scripts/ct/binsec.py).
- Chose direct integration over `cargo-checkct` so the rscrypto manifest owns
  kernels, targets, assumptions, required status, and evidence artifacts.
- Added repo-local BINSEC harness entrypoints for small CT-critical kernels.
- Made required Linux BINSEC kernels fail `ct-check` unless the result is
  `secure`.
- Emit per-kernel BINSEC artifacts under
  `target/ct/<target>/<profile>/binsec/<kernel>/`.

Current BINSEC scope:

- BINSEC is used for small, analyzable CT kernels.
- BINSEC is not used as a whole-public-API verifier.
- AEAD whole APIs remain covered by artifacts/heuristics and DudeCT; BINSEC
  targets the relevant leaf kernels such as AES round, GHASH, POLYVAL,
  Poly1305, and Ascon tag paths.
- macOS Mach-O, Windows PE/COFF, WASM, bare-metal, and s390x BINSEC are not
  claimed today.

Local status:

- A local `aarch64-unknown-linux-gnu` BINSEC run completed with all current
  required reports marked `secure`.

### CI Wiring

- Added [`.github/workflows/ct.yaml`](../../.github/workflows/ct.yaml).
- The workflow supports manual trigger and a weekly scheduled run.
- `platforms=all` currently emits native Linux, macOS, Windows, and Linux MUSL
  CT lanes.
- Linux CT lanes use `tools_mode=ct-linux`, which installs Linux CT support and
  BINSEC through the shared setup path.
- Artifacts are uploaded even on failure:
  host logs, CT logs, and compressed `target/ct` evidence bundles.

## Not Done Yet

These are the remaining blockers before making a public release CT claim.

### 1. Run The CI Matrix

The workflow is wired, but the full GitHub matrix has not yet produced a clean
set of release artifacts.

Next action:

```text
Push the current branch, manually run .github/workflows/ct.yaml with
platforms=all, and inspect every uploaded ct-* artifact.
```

Expected outcome:

- Native Linux, macOS, Windows, and MUSL lanes either pass or produce precise
  artifacts showing what failed.
- Any failure is classified as tool setup, harness/configuration, real timing
  signal, BINSEC unsupported instruction, or actual CT bug.

### 2. Audit Heuristic Warnings

The heuristic gate blocks hard failures, but warnings still need human review.

Remaining work:

- Review every conditional branch, call, indexed load, panic/allocation symbol,
  and dispatch warning in generated CT artifacts.
- Add narrow waivers only when the warning is public-data-driven or otherwise
  outside the secret-dependent leakage policy.
- Keep waivers versioned and target-scoped.

### 3. Publish A CT Matrix

The docs now describe the current target boundary, but they do not yet include
the actual pass/fail evidence from CI.

Remaining work after the first clean CI run:

- Add a public CT matrix to the docs.
- Link or summarize uploaded artifacts.
- State which primitive/target pairs are claimed, intended, unsupported, or
  deferred.

Likely destination:

- [`docs/constant-time.md`](../constant-time.md) for policy and target matrix.
- [`docs/security.md`](../security.md) for user-facing security boundary.

### 4. Add Artifact Diff Review

The pipeline emits hashes and reports, but the review workflow for changed
CT-critical assembly is not complete yet.

Remaining work:

- Store reviewed baseline hashes for CT-critical artifacts.
- Fail CI when a reviewed CT-critical symbol changes without CT review.
- Require reviewer-owned waivers for accepted changes.

### 5. no_std And WASM

These targets are deliberately outside the current release CT claim.

`no_std` needs:

- Separate artifact builds for bare-metal target triples.
- Panic/allocation/profile-specific provenance.
- BINSEC only where an ELF/ISA harness is supported.
- Real hardware timing evidence before a timing claim.

WASM needs:

- `.wasm` bytecode checks for secret-dependent branch/table/call/memory shape.
- Engine-specific timing evidence for Wasmtime, Node/V8, and any browser
  engines the project wants to claim.
- Claim language per engine, not a generic "WASM is constant-time" statement.

### 6. Backend Expansion

Deferred until LLVM evidence is stable:

- GCC codegen.
- Cranelift.
- Additional linker-specific claims beyond the current target rows.

### 7. Performance Recheck

CT fixes can affect performance. After the CT matrix is clean:

- Run the normal benchmark matrix.
- Compare Ed25519, BLAKE3, AEAD, RSA, Curve25519, and KDF hot paths against the
  latest saved benchmark artifacts.
- Recover performance only with branchless, CT-safe implementations.

## Current Commands

Local artifact/heuristic gate:

```bash
just ct
```

Local full gate for the native host:

```bash
just ct-check
just ct-full
```

Narrow local DudeCT debugging:

```bash
just ct-dudect --filter x25519
just ct-dudect --smoke
```

Linux BINSEC gate:

```bash
just ct-binsec
```

CI release evidence:

```text
Run .github/workflows/ct.yaml manually with platforms=all.
```

## Success Criteria For The Next Push

The next push is successful when:

- `ct.yaml` completes for every selected native lane, or every failing lane has
  a concrete artifact explaining the failure.
- All CT-critical primitives in the manifest have required DudeCT evidence on
  host-executable targets.
- All Linux targets with `binsec = "required"` have required BINSEC kernels
  marked `secure`.
- macOS and Windows pass the two gates available today.
- `no_std`, WASM, s390x BINSEC, GCC, and Cranelift remain explicitly outside
  the release claim until their own evidence exists.

## Final Claim Shape

Use this wording only after the relevant CI evidence exists:

```text
For the configurations listed in ct-report.json, rscrypto's manifest-declared
CT-critical primitives passed the required LLVM artifact/heuristic, DudeCT, and
where supported Linux BINSEC gates with respect to the declared secret inputs
and public assumptions.
```

Do not write:

```text
rscrypto is proven constant-time on every platform.
```
