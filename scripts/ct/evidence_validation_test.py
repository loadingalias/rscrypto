#!/usr/bin/env python3
"""Regression tests for fail-closed CT evidence identity checks."""

from __future__ import annotations

import tempfile
from pathlib import Path

from asm_heuristics import apply_public_operand_rules
from provenance import codegen_value
from symbolize_linked_binary import Symbol, parse_indirect_symbols, parse_link_map, symbolize
from validate_release_evidence import (
  parse_hashes,
  records_by_path,
  records_by_name,
  validate_equality_index,
  validate_exact_candidate,
  validate_heuristics,
)


def expect_failure(action) -> None:
  try:
    action()
  except ValueError:
    return
  raise AssertionError("evidence mutation unexpectedly passed")


def equality_fixture() -> tuple[dict, dict, dict]:
  owner = "ct_entry_owner_eq_16"
  public = "ct_entry_kmac256_verify"
  ct = {
    "equality_evidence": {
      "release_binary": {
        "owner_symbols": [owner],
        "public_len_symbols": [public],
      }
    }
  }
  symbols = []
  for name in (owner, public):
    symbols.append(
      {
        "name": name,
        "locations": [
          {"object": "rscrypto_ct_evidence-deadbeef.o"},
          {"object": "rscrypto-ct-evidence.binary"},
        ],
      }
    )
  evidence = {"primitives": [{"harness": {"symbols": symbols}}]}
  heuristics = {
    "schema_version": 2,
    "kind": "rscrypto.ct.asm-heuristics",
    "target": "x86_64-unknown-linux-gnu",
    "profile": "release",
    "needs_fix_count": 0,
    "needs_binsec_count": 0,
    "accepted_count": 0,
    "unclassified_count": 0,
    "unwaived_fail_count": 0,
    "missing_symbols": [],
    "finding_count": 0,
    "symbol_summary": {owner: {"present": True}, public: {"present": True}},
    "ct_intended_call_closure": {
      "primitive_summary": {
        "owner_equality.fixed": {
          "root_symbols": [owner],
          "missing_root_symbols": [],
        }
      }
    },
    "disassembly_files": [
      {
        "path": "artifacts/rscrypto-ct-evidence.binary.disasm.txt",
        "sha256": "a" * 64,
      }
    ],
    "final_equality_call_closure": {
      "artifact": "artifacts/rscrypto-ct-evidence.binary.disasm.txt",
      "artifact_sha256": "a" * 64,
      "root_symbols": [owner, public],
      "missing_root_symbols": [],
      "unresolved_internal_calls": [],
      "terminal_call_sites": [],
    },
  }
  return ct, evidence, heuristics


def main() -> None:
  assert codegen_value(["-C", "target-cpu=native", "-C", "target-cpu=x86-64"], "target-cpu") == "x86-64"

  commit = "a" * 40
  validate_exact_candidate("1.2.3", commit, "1.2.3", commit)
  expect_failure(lambda: validate_exact_candidate("1.2.3", commit, "1.2.4", commit))
  expect_failure(lambda: validate_exact_candidate("1.2.3", commit, "1.2.3", "b" * 40))

  expect_failure(lambda: records_by_name([{"name": "same"}, {"name": "same"}], "fixture"))
  expect_failure(lambda: records_by_path([{"path": "same"}, {"path": "same"}], "fixture"))
  with tempfile.TemporaryDirectory() as temporary:
    temporary_path = Path(temporary)
    hashes = temporary_path / "hashes.txt"
    hashes.write_text(f"{'0' * 64}  artifact\n{'1' * 64}  artifact\n")
    expect_failure(lambda: parse_hashes(hashes))

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

  ct, evidence, heuristics = equality_fixture()
  validate_equality_index(evidence, "fixture", ct)
  validate_heuristics(heuristics, "fixture", "x86_64-unknown-linux-gnu", ct)
  evidence["primitives"][0]["harness"]["symbols"][0]["locations"].pop()
  expect_failure(lambda: validate_equality_index(evidence, "fixture", ct))
  _, _, missing_terminal_calls = equality_fixture()
  missing_terminal_calls["final_equality_call_closure"].pop("terminal_call_sites")
  expect_failure(
    lambda: validate_heuristics(missing_terminal_calls, "fixture", "x86_64-unknown-linux-gnu", ct)
  )
  heuristics["missing_symbols"] = ["ct_entry_owner_eq_16"]
  expect_failure(lambda: validate_heuristics(heuristics, "fixture", "x86_64-unknown-linux-gnu", ct))

  finding = {
    "symbol": "rscrypto::auth::argon2::fill_segment_inner",
    "kind": "variable_latency_division",
    "primitive_ids": ["password.argon2i"],
    "roots": ["ct_entry_argon2i_verify"],
    "locator": "fixture",
    "operand_class": "unproven",
    "disposition": "needs-fix",
  }
  rule = {
    "primitive": "password.argon2i",
    "root": "ct_entry_argon2i_verify",
    "symbol": finding["symbol"],
    "kind": finding["kind"],
    "max_count": 1,
    "source": "src/auth/argon2/mod.rs:1434",
    "rationale": "public addressing inputs",
  }
  assert apply_public_operand_rules([finding], [rule]) == []
  assert finding["operand_class"] == "public" and finding["disposition"] == "accepted"
  extra = dict(finding, locator="fixture-2", operand_class="unproven", disposition="needs-fix")
  assert apply_public_operand_rules([finding, extra], [rule])


if __name__ == "__main__":
  main()
