#!/usr/bin/env -S uv run python
"""Regression tests for fail-closed CT evidence identity checks."""

from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from unittest.mock import patch
import tempfile
from pathlib import Path

import full
import validate as manifest_validation
from asm_heuristics import (
  FunctionBody,
  all_direct_callees,
  apply_public_operand_rules,
  apply_waivers,
  build_call_graph,
  parse_disassembly,
  reachable_closure,
  scan_symbol,
  summarize,
  summarize_closure,
  validate_waivers,
)
from full import build_findings, configure_target_environment
from provenance import codegen_value, resolve_executable, resolved_linker_version, symbol_objects
from symbolize_linked_binary import (
  Symbol,
  parse_indirect_symbols,
  parse_final_nm,
  parse_link_map,
  parse_dynamic_relocations,
  parse_power_import_thunks,
  parse_s390_import_thunks,
  resolve_x86_got_transfer,
  symbolize,
)
def expect_failure(action) -> None:
  try:
    action()
  except ValueError:
    return
  raise AssertionError("evidence mutation unexpectedly passed")


def test_linked_import_resolution() -> None:
  # GNU s390x lacks decoded PLT names. Bind every import to its actual
  # 14-byte register/width/branch sequence and independently decoded slot.
  with tempfile.TemporaryDirectory() as temporary:
    raw = Path(temporary) / "s390.raw.txt"
    fixture = (
      "evidence: file format elf64-s390\nDisassembly of section .plt:\n"
      "5e48: c0 10 00 06 c7 34\tlarl\t%r1, 0xdecb0\n"
      "5e4e: e3 10 10 00 00 04\tlg\t%r1, 0(%r1)\n"
      "5e54: 07 f1\tbr\t%r1\n"
    )
    raw.write_text(fixture)
    relocations = {0xDECB0: ("R_390_JMP_SLOT", "memcpy")}
    assert parse_s390_import_thunks(raw, relocations) == [Symbol(0x5E48, 14, "memcpy@plt")]
    expect_failure(lambda: parse_s390_import_thunks(raw, None))
    expect_failure(lambda: parse_s390_import_thunks(raw, {}))
    expect_failure(lambda: parse_s390_import_thunks(raw, {0xDECB0: ("R_390_GLOB_DAT", "memcpy")}))
    expect_failure(lambda: parse_s390_import_thunks(raw, {**relocations, 0xDECB8: ("R_390_JMP_SLOT", "other")}))
    raw.write_text(fixture.replace("Disassembly of section .plt:", "Disassembly of section .text:"))
    expect_failure(lambda: parse_s390_import_thunks(raw, relocations))
    # Printed operands are not evidence. Signed negative displacements must
    # still resolve the encoded slot when that annotation is wrong.
    raw.write_text(fixture.replace("00 06 c7 34", "ff ff ff f8"))
    assert parse_s390_import_thunks(raw, {0x5E38: ("R_390_JMP_SLOT", "memcpy")}) == [
      Symbol(0x5E48, 14, "memcpy@plt"),
    ]
    for original, replacement in (
      ("c0 10", "c0 20"), ("c7 34", "c7 35"), ("e3 10 10 00", "e3 10 20 00"),
      ("00 04", "00 14"), ("07 f1", "07 f2"),
    ):
      raw.write_text(fixture.replace(original, replacement))
      expect_failure(lambda: parse_s390_import_thunks(raw, relocations))

  # Final POWER libc calls use unnamed 20-byte ELFv2 thunks. Their instruction
  # bytes and the dynamic JMP_SLOT, rather than the .text label, identify memcpy.
  with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    link_map, raw, got, dynamic = [root / name for name in ("map", "raw", "got", "dynamic")]
    link_map.write_text("1091cc 1091cc 14 16 <internal>:(.text.thunk)\n")
    power_raw = (
      "fixture: file format elf64-powerpcle\n"
      "1091cc: 18 00 41 f8\tstd 2, 24(1)\n"
      "1091d0: 00 00 82 3d\taddis 12, 2, 0\n"
      "1091d4: 40 82 8c e9\tld 12, -32192(12)\n"
      "1091d8: a6 03 89 7d\tmtctr 12\n"
      "1091dc: 20 04 80 4e\tbctr\n"
    )
    raw.write_text(power_raw)
    got.write_text("fixture: file format elf64-powerpcle\nContents of section .got:\n 12a310 10231300 00000000\n")
    dynamic.write_text("000000000012a550 R_PPC64_JMP_SLOT memcpy\n")
    relocations = parse_dynamic_relocations(dynamic)
    assert parse_power_import_thunks(link_map, raw, got, relocations) == [Symbol(0x1091CC, 20, "memcpy@plt")]
    for original, changed in (
      ("18 00 41 f8", "18 00 61 f8"),  # save another register
      ("00 00 82 3d", "00 00 83 3d"),  # address from a different base
      ("40 82 8c e9", "44 82 8c e9"),  # load an unrelocated slot
      ("a6 03 89 7d", "a6 03 69 7d"),  # transfer a different register
      ("20 04 80 4e", "21 04 80 4e"),  # different transfer semantics
    ):
      raw.write_text(power_raw.replace(original, changed))
      expect_failure(lambda: parse_power_import_thunks(link_map, raw, got, relocations))
    raw.write_text(power_raw)
    for changed in ({}, {0x12A550: ("R_PPC64_RELATIVE", "memcpy")}):
      expect_failure(lambda: parse_power_import_thunks(link_map, raw, got, changed))
    got.write_text(got.read_text().replace("10231300", "18231300"))
    expect_failure(lambda: parse_power_import_thunks(link_map, raw, got, relocations))
    dynamic.write_text(dynamic.read_text() * 2)
    expect_failure(lambda: parse_dynamic_relocations(dynamic))
    # A GNU map has no LLD synthetic thunk range. It gains no synthesized
    # names; unresolved .text/.plt destinations remain the closure gate's job.
    link_map.write_text(
      " .text.unlikely.imports.stub\n"
      "                0x0000000000008800      0x8c0 linker stubs\n"
      "                0x0000000000008860                0000004c.plt_call.memcpy@@GLIBC_2.17\n"
      "                0x0000000000008880                0000004c.plt_call.abort@@GLIBC_2.17\n"
    )
    assert parse_power_import_thunks(link_map, raw, got, relocations) == []
    assert Symbol(0x8860, 0x20, "0000004c.plt_call.memcpy@@GLIBC_2.17") in parse_link_map(link_map, {})

    # The final Linux x86 P-256 caller reaches ASM through a RELATIVE GOT slot.
    # Resolve the encoded slot, not any incidental numeric data annotation.
    call = "fcc20: ff 15 2a 85 06 00\tcallq\t*0x6852a(%rip) # 0x165150"
    assembly = Symbol(0x44A34, 1, "rscrypto_p256_scalarmul_alt")
    symbols = {assembly.address: assembly}
    relative = {0x165150: ("R_X86_64_RELATIVE", "*ABS*+0x44a34")}
    resolved = call.replace("# ", "# GOT ") + " <rscrypto_p256_scalarmul_alt>"
    assert resolve_x86_got_transfer(call, relative, symbols) == resolved
    jump = call.replace("ff 15", "ff 25").replace("callq", "jmpq")
    assert resolve_x86_got_transfer(jump, relative, symbols) == jump.replace("# ", "# GOT ") + " <rscrypto_p256_scalarmul_alt>"
    for kind in ("R_X86_64_GLOB_DAT", "R_X86_64_JUMP_SLOT"):
      assert resolve_x86_got_transfer(call, {0x165150: (kind, "memcpy")}, symbols) == call.replace("# ", "# GOT ") + " <memcpy@plt>"
    assert resolve_x86_got_transfer(call, {}, symbols) == call
    assert resolve_x86_got_transfer(call, {0x165158: relative[0x165150]}, symbols) == call
    data = call.replace("callq", "movq")
    assert resolve_x86_got_transfer(data, relative, symbols) == data
    for changed in (
      call.replace("ff 15", "ff 25"),
      call.replace("2a 85", "2b 85"),
      call.replace("# 0x165150", "# 0x165158"),
    ):
      expect_failure(lambda: resolve_x86_got_transfer(changed, relative, symbols))
    expect_failure(lambda: resolve_x86_got_transfer(call, {0x165150: ("R_X86_64_RELATIVE", "*ABS*+0x44a35")}, symbols))
    raw.write_text("fixture: file format elf64-x86-64\n44a34: c3\tretq\n" + call + "\n")
    output = root / "symbolized.disasm.txt"
    symbolize(raw, output, [assembly, Symbol(0xFCC20, 6, "ct_entry_p256_ecdh_agree")], dynamic_relocations=relative)
    functions, _ = parse_disassembly(output)
    assert reachable_closure("ct_entry_p256_ecdh_agree", build_call_graph(functions)) == {
      "ct_entry_p256_ecdh_agree", "rscrypto_p256_scalarmul_alt",
    }


def manifest_errors(mutate) -> list[str]:
  root = Path(__file__).resolve().parents[2]
  manifest = manifest_validation.load_toml(root / "ct.toml")
  mutate(manifest)
  selected_target = "aarch64-apple-darwin"
  selected = next(target for target in manifest["target"] if target["name"] == selected_target)

  original_load = manifest_validation.load_toml
  original_snapshot = manifest_validation.compiler_public_api_snapshot
  manifest_validation.load_toml = lambda _path: manifest
  manifest_validation.compiler_public_api_snapshot = lambda _root, _target, _prefixes, _errors: (
    selected["compiler_api_item_count"],
    selected["compiler_api_sha256"],
  )
  try:
    errors: list[str] = []
    manifest_validation.validate_manifest(root, selected_target, errors, [])
    return errors
  finally:
    manifest_validation.load_toml = original_load
    manifest_validation.compiler_public_api_snapshot = original_snapshot


SOURCE = {"commit": "candidate", "sha256": "current"}


def complete_report(
  *, name="fixture", status="pass", gate="required", source=SOURCE, requested=100, rows=100, classes=("0", "1")
):
  """A child DudeCT report with every requested sample, from `source`."""
  labels = {label: rows // len(classes) for label in classes}
  row = {
    "name": name, "gate": gate, "status": status, "requested_samples": requested,
    "raw_csv": {"row_count": rows, "labels": labels},
  }
  return json.dumps({"source": source, "cases": [row]})


def test_dudect_invocation_evidence() -> None:
  with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    child = root / "child.py"
    child.write_text("""
import json, os, sys
from pathlib import Path
mode = json.loads(Path('mode.json').read_text())
evidence = Path(sys.argv[sys.argv.index('--evidence-dir') + 1])
if mode['report'] is not None:
  (evidence / 'dudect-report.json').write_text(mode['report'])
  (evidence / 'dudect-raw.csv').write_text('current samples')
sys.exit(124 if '--timeout' in sys.argv else mode['exit'])
""")
    case = {"name": "fixture", "filter": "fixture", "primitive": "fixture"}

    def invoke(report, exit_code=0, *, timeout=None, gate="required"):
      (root / "mode.json").write_text(json.dumps({"report": report, "exit": exit_code}))
      with patch.object(full, "python_script", return_value=[sys.executable, str(child)]):
        return full.dudect_case_result(
          root, root / "logs",
          100, 10.0, {**case, "gate": gate}, timeout, root / "out/dudect/run/shared/prepared.json", SOURCE,
        )

    def report(status="pass", gate="required"):
      return complete_report(status=status, gate=gate)

    # Seed both legacy locations as well as a successful current invocation.
    legacy = root / "out/dudect"
    (legacy / "cases/fixture").mkdir(parents=True)
    for directory in (legacy, legacy / "cases/fixture"):
      (directory / "dudect-report.json").write_text(report())
      (directory / "dudect-raw.csv").write_text("stale samples")

    for payload, exit_code, timeout, expected_command in (
      (None, 2, None, "fail"),
      (None, 0, 0, "timeout"),
      (None, 0, None, "pass"),
      ('{"cases": [', 1, None, "fail"),
      ('{"cases": [', 0, None, "pass"),
      ('[]', 0, None, "pass"),
      ('{"cases": [null]}', 0, None, "pass"),
      ('{"cases": [{"name": "fixture"}]}', 0, None, "pass"),
      (report().replace('fixture', 'other'), 0, None, "pass"),
    ):
      with_success = invoke(report())
      assert with_success["status"] == "pass"
      assert build_findings([], [with_success], [], []) == ([], [])
      row = invoke(payload, exit_code, timeout=timeout)
      assert row["status"] == ("timeout" if timeout == 0 else "tooling-fail"), row
      assert row["command_result"]["status"] == expected_command
      assert row["statistical_status"] is None
      assert row["report_error"]
      assert row["report"] != with_success["report"]
      assert row["command_result"]["stdout"] != with_success["command_result"]["stdout"]
      if payload is None:
        assert row["report"] is None and row["artifacts"] == []
      findings, diagnostics = build_findings([], [row], [], [])
      assert len(findings) == 1 and not diagnostics
      assert findings[0]["category"] == ("evidence_inconclusive" if timeout == 0 else "tooling_failure")

    for gate in ("required", "diagnostic"):
      row = invoke(report(gate=gate), 2, gate=gate)
      assert row["status"] == "tooling-fail" and row["statistical_status"] == "pass"
      assert row["command_result"]["returncode"] == 2
      findings, diagnostics = build_findings([], [row], [], [])
      assert findings[0]["category"] == "tooling_failure" and not diagnostics
      row = invoke(None, 2, gate=gate)
      assert build_findings([], [row], [], [])[0][0]["category"] == "tooling_failure"

    def timeout_after_report(command, **kwargs):
      evidence = Path(command[command.index("--evidence-dir") + 1])
      (evidence / "dudect-report.json").write_text(report())
      raise subprocess.TimeoutExpired(command, 1)

    with patch.object(full.subprocess, "run", side_effect=timeout_after_report):
      row = invoke(report(), timeout=1)
    assert row["status"] == "timeout" and row["statistical_status"] == "pass"
    assert row["command_result"]["status"] == "timeout"
    assert build_findings([], [row], [], [])[0][0]["category"] == "evidence_inconclusive"

    row = invoke(report("fail"), 1)
    assert row["status"] == "fail" and row["failure_count"] == 1
    assert row["command_result"]["status"] == "fail"
    assert build_findings([], [row], [], [])[0][0]["category"] == "timing_failure"
    row = invoke(report("diagnostic-fail", "diagnostic"), gate="diagnostic")
    assert row["status"] == "diagnostic-fail"
    findings, diagnostics = build_findings([], [row], [], [])
    assert not findings and len(diagnostics) == 1


def test_dudect_smoke_summary_is_insufficient() -> None:
  with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    report = root / "target/ct/fixture/release/dudect/dudect-report.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({
      "schema_version": 3, "kind": "rscrypto.ct.dudect", "smoke": True,
      "requested_samples_by_case": {"fixture": 16},
      "measurements": ["runs/fixture/selection/0/dudect-report.json"],
      "cases": [{"name": "fixture", "status": "pass"}],
      "case_count": 1, "failure_count": 0,
    }))
    errors: list[str] = []
    with patch.object(
      manifest_validation.subprocess, "check_output", side_effect=AssertionError("smoke reached release evidence checks"),
    ) as probe:
      manifest_validation.validate_dudect(root, "fixture", "release", {}, errors, [])
    assert errors == ["dudect smoke evidence is insufficient for --require-dudect; run without --smoke"]
    probe.assert_not_called()


def test_dudect_source_identity_rejects_stale_worktree() -> None:
  expected = {"commit": "fixture", "sha256": "current"}
  with patch.object(manifest_validation.bundle, "source_identity", return_value=expected):
    errors: list[str] = []
    manifest_validation.validate_dudect_source(Path("."), {"source": expected}, errors)
    assert errors == []

    for source in (None, {"commit": "fixture", "sha256": "stale"}):
      errors = []
      manifest_validation.validate_dudect_source(Path("."), {"source": source}, errors)
      assert errors == ["dudect source identity mismatch; rebuild and rerun timing evidence"]


def test_qualification_negative_controls() -> None:
  """Each defect in otherwise valid timing evidence must block `ct-full`."""
  with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    child = root / "child.py"
    child.write_text("""
import json, sys
from pathlib import Path
mode = json.loads(Path('mode.json').read_text())
evidence = Path(sys.argv[sys.argv.index('--evidence-dir') + 1])
(evidence / 'dudect-report.json').write_text(mode['report'])
sys.exit(mode['exit'])
""")
    planned = [
      {"name": "fixture", "filter": "fixture", "primitive": "fixture", "gate": "required"},
      {"name": "second", "filter": "second", "primitive": "fixture", "gate": "required"},
      {"name": "probe", "filter": "probe", "primitive": "fixture", "gate": "diagnostic"},
    ]

    def qualify(report, exit_code=0, *, cases=planned):
      (root / "mode.json").write_text(json.dumps({"report": report, "exit": exit_code}))
      rows = []
      with patch.object(full, "python_script", return_value=[sys.executable, str(child)]):
        for case in cases:
          rows.append(full.dudect_case_result(
            root, root / "logs", 100, 10.0, case, None, root / "out/dudect/run/shared/prepared.json", SOURCE,
          ))
      # Only the first planned case runs, so the other required case is also
      # reported missing; the defect under test must be the first finding.
      findings, _ = build_findings([], rows, [], [], full.missing_required_dudect_cases(planned, rows))
      return [finding["category"] for finding in findings]

    def report_for(case, **overrides):
      return complete_report(name=case["name"], gate=case["gate"], **overrides)

    # Positive control: current, complete, passing evidence for every case.
    rows = []
    with patch.object(full, "python_script", return_value=[sys.executable, str(child)]):
      for case in planned:
        (root / "mode.json").write_text(json.dumps({"report": report_for(case), "exit": 0}))
        rows.append(full.dudect_case_result(
          root, root / "logs", 100, 10.0, case, None, root / "out/dudect/run/shared/prepared.json", SOURCE,
        ))
    assert build_findings([], rows, [], [], full.missing_required_dudect_cases(planned, rows)) == ([], [])

    # A missing required case blocks even when its primitive has other results;
    # a missing diagnostic case does not.
    missing = full.missing_required_dudect_cases(planned, rows[:1] + rows[2:])
    assert missing == [{"name": "second", "primitive": "fixture"}]
    assert full.missing_required_dudect_cases(planned, rows[:2]) == []
    findings, _ = build_findings([], rows[:1] + rows[2:], [], [], missing)
    assert [finding["kind"] for finding in findings] == ["missing_dudect_case"]

    first = planned[:1]
    # A stale artifact: a report measured from other source.
    stale = report_for(first[0], source={"commit": "candidate", "sha256": "older"})
    assert qualify(stale, cases=first)[0] == "tooling_failure"
    # Lost samples: fewer rows than requested, or one class absent.
    assert qualify(report_for(first[0], rows=99), cases=first)[0] == "tooling_failure"
    # A complete run of a different sample budget is not this run's evidence.
    assert qualify(report_for(first[0], requested=50, rows=100), cases=first)[0] == "tooling_failure"
    assert qualify(report_for(first[0], classes=("0",)), cases=first)[0] == "tooling_failure"
    # A detected timing failure.
    assert qualify(report_for(first[0], status="fail"), 1, cases=first)[0] == "timing_failure"


def test_msvc_flagless_assembly() -> None:
  # Native MSVC flags Rust functions, but not global assembly functions. A
  # matching object text symbol must recover the latter without admitting data
  # or arbitrary CODE labels as function boundaries.
  with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    symbols = (
      "00000000 T rscrypto_p256_scalarmul\n"
      "00000004 t Llocal_loop\n"
      "00000008 D data_label\n"
      "0000000c U undefined_label\n"
      "00000000 T outside_code\n"
    )
    (root / "input.o.raw-symbols.txt").write_text(symbols)
    (root / "input.o.symbols.txt").write_text(symbols)
    (root / "map").write_text(
      " Start Length Name Class\n"
      " 0001:00000000 00000030H .text CODE\n"
      " 0002:00000000 00000010H .data DATA\n"
      " Address Publics by Value Rva+Base Lib:Object\n"
      " 0001:00000000 rscrypto_p256_scalarmul 0000000140001000 input.o\n"
      " 0001:00000004 Llocal_loop 0000000140001004 input.o\n"
      " 0001:00000008 data_label 0000000140001008 input.o\n"
      " 0001:0000000c undefined_label 000000014000100c input.o\n"
      " 0001:00000010 code_label 0000000140001010 input.o\n"
      " 0001:00000020 ct_entry_p256_ecdh_agree 0000000140001020 f input.o\n"
      " 0002:00000000 outside_code 0000000140002000 input.o\n"
    )
    (root / "nm").write_text("")  # The stripped PE has no final symbol table.
    (root / "raw").write_text(
      "fixture: file format coff-x86-64\nDisassembly of section .text:\n"
      "0000000140001000 <.text>:\n"
      "140001000: c3 retq\n"
      "140001020: e8 db ff ff ff callq 0x140001000 <.text>\n"
      "140001025: c3 retq\n"
    )
    subprocess.run([
      sys.executable, str(Path(__file__).with_name("symbolize_linked_binary.py")),
      "--artifact-dir", str(root), "--raw-disassembly", str(root / "raw"),
      "--nm-symbols", str(root / "nm"), "--link-map", str(root / "map"),
      "--out-disassembly", str(root / "out"), "--out-symbols", str(root / "symbols"),
    ], check=True)
    assert (root / "symbols").read_text().splitlines() == [
      "0000000140001000 0000000000000020 T rscrypto_p256_scalarmul",
      "0000000140001020 0000000000000010 T ct_entry_p256_ecdh_agree",
    ]
    functions, _ = parse_disassembly(root / "out", {})
    assert reachable_closure("ct_entry_p256_ecdh_agree", build_call_graph(functions)) == {
      "ct_entry_p256_ecdh_agree", "rscrypto_p256_scalarmul",
    }


def main() -> None:
  test_msvc_flagless_assembly()
  test_linked_import_resolution()
  test_dudect_invocation_evidence()
  test_dudect_smoke_summary_is_insufficient()
  test_dudect_source_identity_rejects_stale_worktree()
  test_qualification_negative_controls()
  root = Path(__file__).resolve().parents[2]
  target_matrix = manifest_validation.json.loads((root / ".config" / "target-matrix.json").read_text())
  assert manifest_validation.matrix_targets(target_matrix) == {
    target["name"] for target in manifest_validation.load_toml(root / "ct.toml")["target"]
  }

  assert codegen_value(["-C", "target-cpu=native", "-C", "target-cpu=x86-64"], "target-cpu") == "x86-64"

  target_environment: dict[str, str] = {}
  configure_target_environment("s390x-unknown-linux-gnu", target_environment)
  assert target_environment == {
    "CARGO_TARGET_S390X_UNKNOWN_LINUX_GNU_RUSTFLAGS": "-C target-feature=+vector"
  }
  target_environment["CARGO_TARGET_S390X_UNKNOWN_LINUX_GNU_RUSTFLAGS"] = "-C target-cpu=z16"
  configure_target_environment("s390x-unknown-linux-gnu", target_environment)
  assert target_environment["CARGO_TARGET_S390X_UNKNOWN_LINUX_GNU_RUSTFLAGS"] == "-C target-cpu=z16"
  unrelated_environment: dict[str, str] = {}
  configure_target_environment("powerpc64le-unknown-linux-gnu", unrelated_environment)
  assert unrelated_environment == {}

  findings, diagnostics = build_findings(
    [{"name": "ct-binsec", "status": "fail"}],
    [],
    [{"kernel": "completed-before-failure", "required": True, "status": "secure"}],
    [],
  )
  assert diagnostics == []
  assert findings == [
    {
      "kind": "gate_failure",
      "category": "tooling_failure",
      "severity": "blocker",
      "summary": "ct-binsec failed before complete evidence could be collected",
    }
  ]

  captured_rustdoc: dict[str, object] = {}
  original_run = manifest_validation.subprocess.run

  def capture_rustdoc(command, **kwargs):
    captured_rustdoc["command"] = command
    captured_rustdoc["env"] = kwargs.get("env")
    return subprocess.CompletedProcess(command, 1, "", "rustdoc unavailable")

  manifest_validation.subprocess.run = capture_rustdoc
  try:
    with tempfile.TemporaryDirectory() as temporary:
      inventory_errors: list[str] = []
      assert (
        manifest_validation.compiler_public_api_snapshot(
          Path(temporary),
          "aarch64-apple-darwin",
          ("rscrypto::auth",),
          inventory_errors,
        )
        is None
      )
      assert inventory_errors == ["compiler public-API inventory failed: rustdoc unavailable"]
  finally:
    manifest_validation.subprocess.run = original_run

  with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    target = "x86_64-pc-windows-msvc"
    rustdoc_path = root / "target/ct-api-inventory" / target / "doc/rscrypto.json"
    rustdoc_path.parent.mkdir(parents=True)
    for filename, expected in (("src/auth/key.rs", "rscrypto::auth::Key::verify\n"),
                               ("src\\auth\\key.rs", "rscrypto::auth::Key::verify\n"),
                               ("library/core/src/key.rs", "\n")):
      rustdoc_path.write_text(json.dumps({
        "paths": {"1": {"crate_id": 0, "path": ["rscrypto", "auth", "Key"], "kind": "struct"}},
        "index": {
          "1": {"visibility": "public", "inner": {"struct": {"impls": [2]}}},
          "2": {"inner": {"impl": {"items": [3]}}},
          "3": {"visibility": "public", "name": "verify", "inner": {"function": {}},
                "span": {"filename": filename}},
        },
      }))
      with patch.object(manifest_validation.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)):
        errors = []
        assert manifest_validation.compiler_public_api_snapshot(root, target, ("rscrypto::auth",), errors) == (
          int(expected != "\n"), hashlib.sha256(expected.encode()).hexdigest())
        assert not errors

    linker = root / "Build Tools/link.exe"
    linker.parent.mkdir()
    linker.touch()
    assert resolve_executable(str(linker)) == linker.resolve()
    banner = "Microsoft (R) Incremental Linker Version 14.51.36256.0"
    for status, output, expected in ((0, banner, banner), (1100, banner, banner),
                                     (1104, banner, None), (1100, "usage without version", None)):
      with patch("provenance.subprocess.run", return_value=subprocess.CompletedProcess([], status, output)) as version:
        assert resolved_linker_version(linker, str(linker), root) == expected
        version.assert_called_once_with([str(linker), "/?"], cwd=root, text=True, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, check=False)

    symbols = root / "rscrypto_ct_evidence.obj.symbols.txt"
    symbols.write_text(
      "00000000 R .refptr.ct_entry_argon2i_verify\n"
      "00000000 T ct_entry_argon2i_verify\n"
      "00000000 R ct_entry_data_only\n"
    )
    binary_symbols = root / "rscrypto-ct-evidence.exe.binary.symbols.txt"
    binary_symbols.write_text("0000000140001020 0000000000000010 T ct_entry_argon2i_verify\n")
    assert manifest_validation.symbol_counts(symbols) == {"ct_entry_argon2i_verify": 1}
    assert manifest_validation.generated_symbols(root) == {"ct_entry_argon2i_verify"}
    assert [row["object"] for row in symbol_objects(root)["ct_entry_argon2i_verify"]] == [
      "rscrypto_ct_evidence.obj", "rscrypto-ct-evidence.exe.binary"]

  rustdoc_env = captured_rustdoc["env"]
  assert isinstance(rustdoc_env, dict)
  assert rustdoc_env["RUSTC_BOOTSTRAP"] == "rscrypto"
  assert captured_rustdoc["command"][-4:] == ["-Z", "unstable-options", "--output-format", "json"]

  def duplicate_target(manifest) -> None:
    manifest["target"].append(dict(manifest["target"][0]))

  assert "duplicate target x86_64-unknown-linux-gnu" in manifest_errors(duplicate_target)

  assert "asm_select_branch must be a table" in manifest_errors(
    lambda manifest: manifest.update(asm_select_branch=[])
  )
  for enforced in ["kx.p256_ecdh", ["kx.p256_ecdh", "kx.p256_ecdh"]]:
    assert "asm_select_branch enforced_primitives must be a list of unique strings" in manifest_errors(
      lambda manifest: manifest.update(asm_select_branch={"enforced_primitives": enforced})
    )
  assert "asm_select_branch enforces unknown, which is not a ct-intended primitive" in manifest_errors(
    lambda manifest: manifest.update(asm_select_branch={"enforced_primitives": ["unknown"]})
  )
  for unreachable in ["ct_entry_rsa_fixture_key_import", ["ct_entry_rsa_fixture_key_import"] * 2, [" "]]:
    assert "asm_select_branch unreachable_symbols must be a list of unique strings" in manifest_errors(
      lambda manifest: manifest["asm_select_branch"].update(unreachable_symbols=unreachable)
    )
  # An enforced root is reachable from itself, so listing one would make every artifact fail.
  assert (
    "enforced CT root ct_entry_rsa_pss_sign_fixed_blinding cannot also be unreachable from the enforced roots"
    in manifest_errors(
      lambda manifest: manifest["asm_select_branch"]["unreachable_symbols"].append("ct_entry_rsa_pss_sign_fixed_blinding")
    )
  )

  def public_operand_unknown_target(manifest) -> None:
    manifest["asm_public_operand"][0]["targets"] = ["unknown-target"]

  assert "asm_public_operand[0] targets must be a non-empty list of unique ct.toml targets" in manifest_errors(
    public_operand_unknown_target
  )

  def required_target_without_snapshot(manifest) -> None:
    target = next(row for row in manifest["target"] if row["name"] == "x86_64-unknown-linux-musl")
    target["physical_timing"] = "required"

  assert (
    "target x86_64-unknown-linux-musl requires a compiler public-API snapshot for release evidence"
    in manifest_errors(required_target_without_snapshot)
  )

  def public_operand_primitive_without_root(manifest) -> None:
    manifest["asm_public_operand"][0]["primitives"].append("owner_equality.fixed")

  assert (
    "asm_public_operand[0] has no root owned by primitive owner_equality.fixed"
    in manifest_errors(public_operand_primitive_without_root)
  )

  def evidence_unit_with_undeclared_variant(manifest) -> None:
    unit = next(
      row
      for row in manifest["evidence_unit"]
      if row["id"] == "aead.symmetric_transform.aes128_header_protection"
    )
    unit["variant"] = "UndeclaredVariant"

  assert (
    "evidence unit aead.symmetric_transform.aes128_header_protection variant 'UndeclaredVariant' "
    "is not declared by primitive aead.symmetric_transform"
    in manifest_errors(evidence_unit_with_undeclared_variant)
  )

  def dudect_case(manifest, name):
    return next(case for case in manifest["dudect_case"] if case["name"] == name)

  def diagnostic_target_unknown(manifest) -> None:
    dudect_case(manifest, "mldsa_product_fixed_vs_random")["diagnostic_targets"] = ["sparc64-unknown-linux-gnu"]

  assert (
    "DudeCT case mldsa_product_fixed_vs_random diagnostic_targets names unknown target(s): sparc64-unknown-linux-gnu"
    in manifest_errors(diagnostic_target_unknown)
  )

  def diagnostic_target_without_reason(manifest) -> None:
    del dudect_case(manifest, "mldsa_product_fixed_vs_random")["diagnostic_targets_reason"]

  assert (
    "DudeCT case mldsa_product_fixed_vs_random diagnostic_targets requires diagnostic_targets_reason"
    in manifest_errors(diagnostic_target_without_reason)
  )

  def diagnostic_target_on_diagnostic_case(manifest) -> None:
    case = dudect_case(manifest, "mldsa_probe_product_zero_vs_zero")
    case["diagnostic_targets"] = ["powerpc64le-unknown-linux-gnu"]
    case["diagnostic_targets_reason"] = "fixture"

  assert (
    "DudeCT case mldsa_probe_product_zero_vs_zero diagnostic_targets applies only to a required case"
    in manifest_errors(diagnostic_target_on_diagnostic_case)
  )

  # A target-scoped demotion must leave every evidence unit with a case required on that target.
  power = "powerpc64le-unknown-linux-gnu"
  demoted = manifest_validation.load_toml(Path(__file__).resolve().parents[2] / "ct.toml")
  unit = next(row for row in demoted["evidence_unit"] if row["id"] == "signature.mldsa.secret_kernels.44")
  for name in unit["dudect"]:
    dudect_case(demoted, name)["diagnostic_targets"] = [power]
  for target, expected in ((power, True), ("x86_64-unknown-linux-gnu", False)):
    errors: list[str] = []
    manifest_validation.validate_strict_coverage(demoted, errors, target)
    missing = f"primitive signature.mldsa.secret_kernels evidence unit {unit['id']} has no DudeCT case required on {target}"
    assert (missing in errors) == expected, errors

  with tempfile.TemporaryDirectory() as temporary:
    temporary_path = Path(temporary)
    link_map = temporary_path / "link-map.txt"
    link_map.write_text(
      "             VMA              LMA     Size Align Out     In      Symbol\n"
      "            1000             1000       10    16 input.o:(.text.ct_entry_owner_eq_16)\n"
      "            1010             1010       10    16 input.o:(.text._RNv_test)\n"
    )
    mapped = parse_link_map(link_map, {"_RNv_test": "rscrypto::fixed_eq"})
    assert mapped == [Symbol(0x1000, 0x10, "ct_entry_owner_eq_16"), Symbol(0x1010, 0x10, "rscrypto::fixed_eq")]

    # Assembly exports can have size zero inside a shared .text section.
    # Bound them by the next named function, retaining local loop labels inside.
    assembly_map = temporary_path / "assembly-map.txt"
    assembly_map.write_text(
      "1000 1000 40 16 input.o:(.text)\n"
      "1000 1000 0 1 .L0\n"
      "1000 1000 0 1 $x\n"
      "1000 1000 0 1 rscrypto_p256_scalarmul_alt\n"
      "1008 1008 0 1 Lp256_loop\n"
      "1010 1010 0 1 .L0\n"
      "1020 1020 0 1 rscrypto_p256_scalarmulbase_alt\n"
    )
    assembly_symbols = [Symbol(0x1000, 0x20, "rscrypto_p256_scalarmul_alt"),
                        Symbol(0x1020, 0x20, "rscrypto_p256_scalarmulbase_alt")]
    assert parse_link_map(assembly_map, {}) == assembly_symbols
    assembly_nm = temporary_path / "assembly-nm.txt"
    assembly_nm.write_text(
      "1000 T rscrypto_p256_scalarmul_alt\n"
      "1008 t Lp256_loop\n"
      "1010 t $x\n"
      "1020 T rscrypto_p256_scalarmulbase_alt\n"
      "1040 T next_function\n"
    )
    assert parse_final_nm(assembly_nm)[:2] == assembly_symbols
    assembly_map.write_text(
      " .text 0x1000 0x40 input.o\n"
      " 0x1000 rscrypto_p256_scalarmul_alt\n"
      " 0x1008 Lp256_loop\n"
      " 0x1020 rscrypto_p256_scalarmulbase_alt\n"
    )
    assert parse_link_map(assembly_map, {}) == assembly_symbols

    gnu_link_map = temporary_path / "gnu-link-map.txt"
    gnu_link_map.write_text(
      "Discarded input sections\n"
      "\n"
      " .text.ct_entry_discarded\n"
      "                0x0000000000000000       0x20 input.o\n"
      "\n"
      "Memory Configuration\n"
      "\n"
      "Linker script and memory map\n"
      " .text.ct_entry_owner_eq_16\n"
      "                0x0000000000002000       0x10 input.o\n"
      " .text._RNv_test\n"
      "                0x0000000000002010       0x10 input.o\n"
    )
    assert parse_link_map(gnu_link_map, {"_RNv_test": "rscrypto::fixed_eq"}) == [
      Symbol(0x2000, 0x10, "ct_entry_owner_eq_16"),
      Symbol(0x2010, 0x10, "rscrypto::fixed_eq"),
    ]

    msvc_link_map = temporary_path / "msvc-link-map.txt"
    msvc_link_map.write_text(
      " Start         Length     Name                   Class\n"
      " 0001:00000020 00000030H .text$mn                CODE\n"
      " 0001:00000080 00000010H .text$x                 CODE\n"
      " 0002:00000000 00000020H .data                   DATA\n"
      " Address         Publics by Value              Rva+Base       Lib:Object\n"
      " 0001:00000020 ct_entry_owner_eq_16             0000000140001020 f input.obj\n"
      " 0001:00000024 code_label                       0000000140001024   input.obj\n"
      " 0001:00000030 _RNv_test                        0000000140001030 f input.obj\n"
      " 0001:00000030 folded_alias                     0000000140001030 f input.obj\n"
      " 0001:00000040 memcpy                           0000000140001040 f libcmt:memcpy.obj\n"
      " 0001:00000080 last_function                    0000000140001080 f input.obj\n"
      " 0002:00000000 data_value                       0000000140002000   input.obj\n"
    )
    msvc_symbols = [
      Symbol(0x140001020, 0x10, "ct_entry_owner_eq_16"),
      Symbol(0x140001030, 0x10, "rscrypto::fixed_eq"),
      Symbol(0x140001030, 0x10, "folded_alias"),
      Symbol(0x140001040, 0x10, "memcpy"),
      Symbol(0x140001080, 0x10, "last_function"),
    ]
    assert parse_link_map(msvc_link_map, {"_RNv_test": "rscrypto::fixed_eq"}) == msvc_symbols
    lld_coff_map = temporary_path / "lld-coff-map.txt"
    lld_coff_map.write_text("\n".join(
      line.replace(" f ", "   ") for line in msvc_link_map.read_text().splitlines() if "code_label" not in line
    ) + "\n 0001:00000028 Llocal_loop 0000000140001028 input.obj\n"
      + " 0001:000000a0 memset 00000001400010a0 vcruntime:VCRUNTIME140.dll\n"
      + " 0002:00000010 __imp_memset 0000000140002010 vcruntime:VCRUNTIME140.dll\n")
    assert parse_link_map(lld_coff_map, {"_RNv_test": "rscrypto::fixed_eq"}) == [
      Symbol(0x1400010a0, 1, "memset"), *msvc_symbols,
    ]
    msvc_link_map.write_text(msvc_link_map.read_text().replace("0000000140001030", "0000000140002030"))
    expect_failure(lambda: parse_link_map(msvc_link_map, {}))

    indirect_symbols = temporary_path / "indirect-symbols.txt"
    indirect_symbols.write_text(
      "Indirect symbols for (__TEXT,__stubs) 1 entries\n"
      "address            index name\n"
      "0x00000001000719d8   864 _memcpy\n"
    )
    assert parse_indirect_symbols(indirect_symbols) == [Symbol(0x1000719D8, 1, "_memcpy")]

    raw_disassembly = temporary_path / "raw.disasm.txt"
    raw_disassembly.write_text(
      "Disassembly of section .text:\n"
      "    1000: e8 0b 00 00 00 callq 0x1010 <.text+0x10>\n"
      "    1005: e8 06 00 00 00 callq 0x1010 <<rscrypto::Owner>::verify::<128>>\n"
      "    100a: c0 e5 00 00 00 03 brasl %r14, 0x1010\n"
      "    1010: c3 retq\n"
    )
    symbolized = temporary_path / "symbolized.disasm.txt"
    symbolize(raw_disassembly, symbolized, mapped)
    output = symbolized.read_text()
    assert "<ct_entry_owner_eq_16>:" in output and "<rscrypto::fixed_eq>" in output
    assert "0x1010 <rscrypto::fixed_eq>" in output
    assert "brasl %r14, 0x1010 <rscrypto::fixed_eq>" in output
    assert "Owner" not in output
    linked_functions, _ = parse_disassembly(symbolized)
    assert reachable_closure("ct_entry_owner_eq_16", build_call_graph(linked_functions)) == {
      "ct_entry_owner_eq_16", "rscrypto::fixed_eq",
    }
    expect_failure(lambda: symbolize(raw_disassembly, symbolized, [Symbol(0x1020, 1, "missing")]))

    # GNU ELF nm bounds _init by the next named text symbol, spanning .plt.
    # A decoder-confirmed import entry must not become an offset inside _init.
    raw_disassembly.write_text(
      "00001000 <_init>:\n1000: d65f03c0\tret\n"
      "00001040 <bcmp@plt>:\n1040: d61f0220\tbr\tx17\n"
      "00001100 <ct_entry_p256_ecdh_agree>:\n"
      "1100: 97ffffd0\tbl\t0x1040 <bcmp@plt>\n"
      "1104: 97ffffd1\tbl\t0x1048 <bcmp@plt>\n"
      "1108: d65f03c0\tret\n"
    )
    import_symbols = [Symbol(0x1000, 0x100, "_init"), Symbol(0x1100, 12, "ct_entry_p256_ecdh_agree")]
    symbolize(raw_disassembly, symbolized, import_symbols)
    output = symbolized.read_text()
    assert "bl\t0x1040 <bcmp@plt>" in output
    # A call-site annotation alone does not prove an import at another address.
    assert "bl\t0x1048 <_init+0x48>" in output
    linked_functions, _ = parse_disassembly(symbolized)
    assert all_direct_callees(linked_functions["ct_entry_p256_ecdh_agree"]) == {"bcmp", "_init"}

    # Native GNU ARM retains local ASM labels. They must not truncate the
    # mapped RSA body before its real +0x64c conditional-copy selection.
    assembly_name = "rscrypto_rsa_bn_mul_mont_words_aarch64_elf"
    raw_disassembly.write_text(
      f"0001a1e0 <{assembly_name}>:\n"
      "1a1e0: f24008bf\ttst\tx5, #0x7\n"
      "0001a1f0 <Lmul_mont>:\n1a1f0: a9bc7bfd\tstp\tx29, x30, [sp, #-0x40]!\n"
      "0001a4d0 <Lsqr8x_outer_loop>:\n1a4d0: d503201f\tnop\n"
      "0001a810 <Lsqr8x_break>:\n"
      "1a810: a9401c06\tldp\tx6, x7, [x0]\n"
      "1a82c: b4ffe52e\tcbz\tx14, 0x1a4d0 <Lsqr8x_outer_loop>\n"
      "1a830: a9005053\tstp\tx19, x20, [x2]\n"
      "1a834: a94051f3\tldp\tx19, x20, [x15]\n"
      "1a838: a9015855\tstp\tx21, x22, [x2, #0x10]\n"
      "1a83c: a94159f5\tldp\tx21, x22, [x15, #0x10]\n"
      "1a840: a9026057\tstp\tx23, x24, [x2, #0x20]\n"
      "1a844: a94261f7\tldp\tx23, x24, [x15, #0x20]\n"
      "1a848: a9036859\tstp\tx25, x26, [x2, #0x30]\n"
      "1a84c: aa0f03e2\tmov\tx2, x15\n"
      "1a850: a94369f9\tldp\tx25, x26, [x15, #0x30]\n"
      "1a854: 17ffff1f\tb\t0x1a4d0 <Lsqr8x_outer_loop>\n"
      "0001b440 <Loutside_mapped_function>:\n1b440: d65f03c0\tret\n"
    )
    symbolize(raw_disassembly, symbolized, [Symbol(0x1A1E0, 0x1260, assembly_name)])
    linked_functions, _ = parse_disassembly(symbolized)
    assert "Lmul_mont" not in linked_functions and "Lsqr8x_break" not in linked_functions
    assert "Loutside_mapped_function" in linked_functions
    assembly_findings = scan_symbol(
      "aarch64-unknown-linux-gnu", assembly_name, linked_functions[assembly_name], set(linked_functions),
      scope="ct_intended_call_closure", primitive_ids=["rsa.private_ops"], roots=[],
    )
    assert any(row["kind"] == "select_branch" and row["offset"] == 0x64C for row in assembly_findings)

    # Inline x86 table data can consume the first byte of the next function.
    # Recover the whole function from original bytes at its mapped boundary.
    raw_disassembly.write_text("100f: 77 f3\tja 0x1004\n1011: 0f 1e\t<unknown>\n1013: fa\tcli\n1014: c3\tretq\n")
    original = raw_disassembly.read_bytes()
    binary = temporary_path / "original-binary"
    binary.write_bytes(b"unchanged linked code")
    recovered = "1010: f3 0f 1e fa\tendbr64\n1014: c3\tretq\n"
    with patch("symbolize_linked_binary.subprocess.check_output", return_value=recovered) as decoder:
      symbolize(raw_disassembly, symbolized, [Symbol(0x1010, 5, "rscrypto::fixed_eq")],
                binary=binary, objdump=Path("llvm-objdump"))
    assert "1010: f3 0f 1e fa" in symbolized.read_text() and "1011:" not in symbolized.read_text()
    assert raw_disassembly.read_bytes() == original and binary.read_bytes() == b"unchanged linked code"
    assert "--start-address=0x1010" in decoder.call_args.args[0]
    assert "--stop-address=0x1015" in decoder.call_args.args[0]
    recovery = json.loads(symbolized.with_name("symbolized.boundary-recovery.json").read_text())
    assert recovery["recoveries"][0]["disassembly"] == recovered
    with patch("symbolize_linked_binary.subprocess.check_output", return_value="1011: c3 retq\n"):
      expect_failure(lambda: symbolize(raw_disassembly, symbolized, [Symbol(0x1010, 5, "rscrypto::fixed_eq")],
                                       binary=binary, objdump=Path("llvm-objdump")))

    # Linked RISC-V register operands may still have a statically resolved
    # section annotation from llvm-objdump. Preserve that call-graph edge.
    raw_disassembly.write_text(
      "Disassembly of section .text:\n"
      "00001000 <.text>:\n"
      "1000: 00000097\tauipc\tra, 0x0\n"
      "1004: 010080e7\tjalr\t0x10(ra) <.text+0x10>\n"
      "1008: 8082\tret\n"
      "1010: 8082\tret\n"
    )
    symbolize(raw_disassembly, symbolized, mapped)
    assert "jalr\t0x10(ra) <rscrypto::fixed_eq>" in symbolized.read_text()
    linked_functions, _ = parse_disassembly(symbolized)
    assert reachable_closure("ct_entry_owner_eq_16", build_call_graph(linked_functions)) == {
      "ct_entry_owner_eq_16", "rscrypto::fixed_eq",
    }

  finding = {
    "symbol": "rscrypto::auth::argon2::fill_segment_inner",
    "kind": "variable_latency_division",
    "severity": "fail",
    "primitive_ids": ["password.argon2i"],
    "roots": ["ct_entry_argon2i_verify"],
    "locator": "fixture",
    "operand_class": "unproven",
    "disposition": "needs-fix",
    "waived": False,
  }
  rule = {
    "primitives": ["password.argon2i"],
    "roots": ["ct_entry_argon2i_verify"],
    "symbol": finding["symbol"],
    "kind": finding["kind"],
    "max_count": 1,
    "source": "src/auth/argon2/mod.rs:1434",
    "rationale": "public addressing inputs",
  }
  assert apply_public_operand_rules([finding], [rule]) == []
  assert finding["operand_class"] == "public" and finding["disposition"] == "accepted"
  assert apply_waivers([finding], [], "aarch64-apple-darwin") == []
  assert finding["unresolved_primitive_ids"] == []

  functions = {
    finding["symbol"]: FunctionBody(finding["symbol"], Path("fixture"), 0, []),
    rule["roots"][0]: FunctionBody(rule["roots"][0], Path("fixture"), 0, []),
  }
  symbol_summary = summarize(set(functions), functions, [finding])[finding["symbol"]]
  assert symbol_summary["unwaived_fail_count"] == 0
  assert symbol_summary["accepted_count"] == 1

  closures = {rule["primitives"][0]: {rule["roots"][0]: set(functions)}}
  primitive_summary = summarize_closure(closures, functions, [finding])[rule["primitives"][0]]
  assert primitive_summary["unwaived_fail_count"] == 0
  assert primitive_summary["accepted_count"] == 1

  mixed = dict(
    finding,
    primitive_ids=[rule["primitives"][0], "fixture.unresolved"],
    operand_class="unproven",
    disposition="needs-fix",
    waived=False,
  )
  assert apply_public_operand_rules([mixed], [rule]) == []
  assert apply_waivers([mixed], [], "aarch64-apple-darwin") == []
  assert mixed["unresolved_primitive_ids"] == ["fixture.unresolved"]
  assert mixed["operand_class"] == "unproven" and mixed["disposition"] == "needs-fix"

  mixed_closures = {
    rule["primitives"][0]: {rule["roots"][0]: set(functions)},
    "fixture.unresolved": {rule["roots"][0]: set(functions)},
  }
  mixed_summary = summarize_closure(mixed_closures, functions, [mixed])
  assert mixed_summary[rule["primitives"][0]]["unwaived_fail_count"] == 0
  assert mixed_summary[rule["primitives"][0]]["accepted_count"] == 1
  assert mixed_summary["fixture.unresolved"]["unwaived_fail_count"] == 1
  assert mixed_summary["fixture.unresolved"]["accepted_count"] == 0

  extra = dict(finding, locator="fixture-2", operand_class="unproven", disposition="needs-fix")
  assert apply_public_operand_rules([finding, extra], [rule])

  # A target-scoped rule bounds each listed target and accepts nothing elsewhere.
  scoped = dict(rule, targets=["s390x-unknown-linux-gnu"])
  other = dict(finding, operand_class="unproven", disposition="needs-fix")
  assert apply_public_operand_rules([other], [scoped], "riscv64gc-unknown-linux-gnu") == []
  assert other["operand_class"] == "unproven" and other["disposition"] == "needs-fix"
  assert apply_public_operand_rules([other], [scoped], "s390x-unknown-linux-gnu") == []
  assert other["operand_class"] == "public" and other["disposition"] == "accepted"
  assert apply_public_operand_rules([other], [dict(rule, targets=[])], "s390x-unknown-linux-gnu")

  # A count bound cannot distinguish a reviewed public branch from a secret
  # branch in the same function. In particular, the old s390x signed-digit
  # correction also fit the public window-bound rule's max_count of one.
  assert apply_public_operand_rules([], [dict(rule, kind="select_branch")])
  assert "asm_public_operand[0] kind is not a public-operand-classifiable instruction" in manifest_errors(
    lambda manifest: manifest["asm_public_operand"][0].update(kind="select_branch")
  )
  target = "s390x-unknown-linux-gnu"
  public_site = dict(
    finding, symbol="<rscrypto::auth::p256_ecdh::P256EphemeralSecret>::diffie_hellman",
    kind="select_branch", primitive_ids=["kx.p256_ecdh"], public_classification=None,
    file="artifacts/rscrypto_ct_evidence.o.disasm.txt", locator="public-window-bound+0x1e0",
    function_sha256="a" * 64, operand_class="unproven", disposition="needs-fix", waived=False,
  )
  waiver = {
    "target": target, "primitive": "kx.p256_ecdh", "symbol": public_site["symbol"],
    "kind": "select_branch", "artifact": public_site["file"], "locator": public_site["locator"],
    "function_sha256": public_site["function_sha256"], "source": "src/auth/p256_portable.rs:246",
    "classification": "public", "rationale": "The compared value is the public window index.",
    "reviewer": "regression fixture", "reviewed_at": "2026-10-15",
  }
  assert validate_waivers([waiver]) == []
  assert apply_waivers([public_site], [waiver], target) == []
  assert public_site["operand_class"] == "public" and public_site["waived"]
  assert apply_waivers([], [waiver], "riscv64gc-unknown-linux-gnu") == []
  for changes in (
    {"locator": "secret-digit-correction+0x278"},
    {"function_sha256": "b" * 64},
    {"file": "artifacts/another.o.disasm.txt"},
  ):
    changed = dict(public_site, **changes, operand_class="unproven", disposition="needs-fix", waived=False)
    assert apply_waivers([changed], [waiver], target)
    assert not changed["waived"] and changed["disposition"] == "needs-fix"

  for classification in ("outside-claim", "allowed-disclosure"):
    bounded = dict(waiver, classification=classification)
    assert validate_waivers([bounded])
    bounded["contract"] = "primitive:kx.p256_ecdh:notes"
    assert validate_waivers([bounded]) == []
    classified = dict(public_site, operand_class="unproven", disposition="needs-fix", waived=False)
    assert apply_waivers([classified], [bounded], target) == []
    assert classified["operand_class"] == "unproven" and classified["waived"]
    assert classified["waivers"][0]["contract"] == "primitive:kx.p256_ecdh:notes"

  boundary = dict(waiver, classification="outside-claim", contract="primitive:kx.p256_ecdh:notes")
  assert not manifest_errors(lambda manifest: manifest.update(asm_waiver=[boundary]))
  disclosure = dict(waiver, classification="allowed-disclosure", primitive="rsa.private_ops",
                    contract="primitive:rsa.private_ops:may_leak:opaque_success_or_failure")
  assert not manifest_errors(lambda manifest: manifest.update(asm_waiver=[disclosure]))
  for contract in ("ct.toml:1", "primitive:missing:notes", "primitive:rsa.private_ops:may_leak:padding_octets"):
    invalid = dict(disclosure, contract=contract)
    assert any("contract must reference" in error for error in manifest_errors(
      lambda manifest: manifest.update(asm_waiver=[invalid])
    ))


if __name__ == "__main__":
  main()
