# Scripts

Repository scripts implement local development, testing, evidence, and
benchmark commands. User-facing entry points are the recipes reported by
`just --list`. Every script in this directory is locally executable.

## Check entry points

| Script | Caller |
| --- | --- |
| `check/affected.sh` | `just check`, `just validate` |
| `check/check-all.sh` | `just check-all` |
| `check/msrv.sh` | `just msrv`, `check/check-all.sh` |
| `check/feature-contracts.sh` | `just feature-contracts`, `check/affected.sh`, `check/check-all.sh` |
| `check/zeroize-evidence.sh` | `just check-zeroize-evidence`, `check/check-all.sh` |
| `check/policy.sh` | `check/affected.sh`, `check/check-all.sh` |
| `check/check.sh` | `check/affected.sh`, `check/check-all.sh` |
| `check/asm-ledger.sh` | `check/policy.sh` |
| `check/rsa-asm-provenance.sh` | `check/asm-ledger.sh` |
| `check/lint-independent-workspaces.sh` | `check/check.sh --all` |
| `check/zig-cc.sh` | `ct/artifacts.sh`, `ct/binsec.py` |

The Python files in `check/` validate feature boundaries, vector provenance,
and assembly provenance. `check/policy.sh` and `check/asm-ledger.sh` own their
normal execution.

## Test entry points

| Script | Caller |
| --- | --- |
| `test/test.sh` | `just test`, `check/affected.sh` |
| `test/test-examples.sh` | `just test-examples`, `check/affected.sh`, `check/check-all.sh` |
| `test/test-miri.sh` | `just test-miri`, `test/miri-contracts.sh` |
| `test/miri-contracts.sh` | `just miri-contract`, `check/affected.sh` |
| `test/test-fuzz.sh` | `just test-fuzz`, `test/fuzz-contracts.sh` |
| `test/fuzz-contracts.sh` | `just fuzz-contract`, `check/affected.sh` |
| `test/test-fuzz-asan.sh` | `just test-fuzz-asan` |
| `test/test-coverage.sh` | `just test-coverage` |
| `test/test-rsa-leakage.sh` | `just test-rsa-leakage` |
| `test/test-rsa-linux-asm.sh` | `just test-rsa-linux-asm` |
| `test/test-rsa-macos-asm.sh` | `just test-rsa-macos-asm` |

## Constant-time evidence

| Script | Caller |
| --- | --- |
| `ct/artifacts.sh` | `just ct-artifacts`, `ct/full.py`, `ct/structural.sh` |
| `ct/dudect.sh` | `just ct-dudect`, `ct/full.py` |
| `ct/structural.sh` | `just ct-structural` |

`ct/full.py`, `ct/binsec.py`, and `ct/validate.py` back `just ct-full`,
`just ct-binsec`, and `just ct-validate`. The remaining Python files under
`ct/` implement local artifact provenance, disassembly analysis, report
parsing, and their focused regression tests.

## Benchmarks and updates

| Script | Caller |
| --- | --- |
| `bench/bench.sh` | `just bench` |
| `bench/run.sh` | `bench/bench.sh` |
| `bench/blake3-gap-gate.sh` | `bench/run.sh` when explicitly enabled |
| `bench/profile.sh` | `just profile` |
| `update/update-all.sh` | `just update` |

`bench/benchmark_catalog.py` owns benchmark selection.
`bench/benchmark_catalog_test.py` runs from `check/policy.sh`.
`render_perf_chart.rs` backs `just chart`.

Local benchmark results land under:

```text
benchmark_results/<date>/<os>/<arch>/results.txt
```

Development-machine runs first write an immutable run directory under
`benchmark_results/criterion/<run-id>/`. Collect it with
`just ssh-collect-bench TARGET RUN_ID DESTINATION` before deallocating the
machine.

## Shared libraries

| Script | Sourced or invoked by |
| --- | --- |
| `lib/common.sh` | Check and test entry points |
| `lib/rail-plan.sh` | `lib/common.sh` |
| `lib/feature-profiles.sh` | `check/feature-contracts.sh` |
| `lib/fuzz-packages.sh` | Fuzz and fuzz-coverage scripts |
| `lib/python.sh` | Python-backed check, CT, and benchmark scripts |
| `lib/toolchain.sh` | `lib/common.sh`, MSRV, Miri, and fuzz scripts |

Python tooling requires Python 3.11 or newer and uses only the standard
library.
