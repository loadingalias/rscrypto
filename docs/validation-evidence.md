# Validation evidence

The validation work completed on 2026-10-04 adds checks to existing runners.
It adds no dependencies, policy configuration, or cryptographic API changes.

- Backend evidence must account for every available ChaCha20 backend.
- Check recipes validate the vector inventory and checksums before compilation.
- Coverage reconciles discovered tests with actual execution and profile files.
  Discovery profiles are excluded.
  Executable hashes, replayed corpus paths and hashes,
  and per-suite LCOV files accompany the report.
  Failed collection or export cannot publish a completed report.

The collector hashes corpus candidates before and after each suite
and compares them with paths recorded by the replay helper after consumption.
These checks detect missing, changed, and misattributed inputs under an ordinary local test run.
They do not authenticate a compromised runner or detect files changed and restored between checks.
Normal replay does not write receipts.

## Executed checks

On Apple M1 Pro, `aarch64-apple-darwin`, using the pinned `nightly-2026-09-30`:

- Thirteen coverage regressions exercise missing execution, missing profiles, changed binaries
  and corpora, suite attribution, and collection/export/publication failures.
- Six fuzz-support tests pass, including committed/local selection and consumed-path receipts.
  Targeted Clippy and formatting checks pass.
- `just test-scripts` passes, with four existing installer tests skipped for unavailable host tools.
  `git diff --check` passes.
- The actual collector ran SHA-2 and CRC32 replay and LLVM export in committed and local modes.
  The committed run consumed one seed per suite;
  the local run consumed 69 and 617 files respectively.
  Both runs retained separate suite LCOV contributions and the combined report.

The local reports are under `target/validation-evidence/coverage-{committed,local}/`.
Source identities, report hashes, selected tests,
and mutation diagnostics are retained in [validation-evidence.json](validation-evidence.json).
This qualification covers the collector on Apple Silicon,
not a complete crate coverage run or other hosts.

## Bounded mutation experiment

Three existing tests exercise the shared production hex parser directly: invalid character,
invalid length, and mixed-case decoding.
Their expectations are explicit errors and the bytes `[0xaa, 0xbb, 0xcc]`, independent of the parser.
The baseline passed all three before mutation.
Cargo-mutants 27.1.0 used an isolated source copy with `std,x25519` and no default features.

| Production fault                               | Detecting test           | Result |
| ---------------------------------------------- | ------------------------ | ------ |
| Invert the length check (`!=` to `==`)         | `from_hex_mixed_case`    | Assertion failure after successful compilation |
| Assemble nibbles with `&` instead of `\|`      | `from_hex_mixed_case`    | Wrong decoded bytes |
| Return `Ok(())` without decoding or validating | All three selected tests | Wrong bytes and acceptance of invalid input |

The third fault initially failed compilation because removing the body made arguments
and a helper unused.
That result was recorded as **unviable**, not detected.
A second baseline and run with cargo-mutants' `--cap-lints true` compiled it and produced three test failures.
Repository lints and tests were unchanged.
There were no surviving selected mutants or timeouts, and no tests were removed.
This bounded result does not measure mutation adequacy across other primitives.

Reproduce the initial experiment:

```bash
cargo mutants --no-config --file src/hex.rs \
  --re 'replace from_hex .* with Ok\(\(\)\)|replace != with == in from_hex|replace \| with & in from_hex' \
  --no-default-features --features std,x25519 --jobs 1 \
  --timeout 60 --build-timeout 300 --output target/validation-evidence --caught \
  -- --lib hex::tests::from_hex_
```

For the body-replacement follow-up, use only `--re 'replace from_hex .* with Ok\(\(\)\)'`, add `--cap-lints true`, and use `--output target/validation-evidence/capped`.
The original logs remain in those output directories;
the retained JSON includes diffs and test diagnostics.
