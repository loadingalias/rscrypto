#!/usr/bin/env python3
"""Regression tests for fail-closed CT evidence identity checks."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import validate as manifest_validation
from asm_heuristics import (
  FunctionBody,
  apply_public_operand_rules,
  apply_waivers,
  summarize,
  summarize_closure,
)
from full import build_findings, configure_target_environment
from provenance import codegen_value
from symbolize_linked_binary import (
  Symbol,
  parse_indirect_symbols,
  parse_link_map,
  symbolize,
)
def expect_failure(action) -> None:
  try:
    action()
  except ValueError:
    return
  raise AssertionError("evidence mutation unexpectedly passed")


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


def main() -> None:
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

  rustdoc_env = captured_rustdoc["env"]
  assert isinstance(rustdoc_env, dict)
  assert rustdoc_env["RUSTC_BOOTSTRAP"] == "rscrypto"
  assert captured_rustdoc["command"][-4:] == ["-Z", "unstable-options", "--output-format", "json"]

  def duplicate_target(manifest) -> None:
    manifest["target"].append(dict(manifest["target"][0]))

  assert "duplicate target x86_64-unknown-linux-gnu" in manifest_errors(duplicate_target)

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
      "    1010: c3 retq\n"
    )
    symbolized = temporary_path / "symbolized.disasm.txt"
    symbolize(raw_disassembly, symbolized, mapped)
    output = symbolized.read_text()
    assert "<ct_entry_owner_eq_16>:" in output and "<rscrypto::fixed_eq>" in output
    assert "0x1010 <rscrypto::fixed_eq>" in output
    assert "Owner" not in output
    expect_failure(lambda: symbolize(raw_disassembly, symbolized, [Symbol(0x1020, 1, "missing")]))

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


if __name__ == "__main__":
  main()
