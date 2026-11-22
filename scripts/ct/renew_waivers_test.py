#!/usr/bin/env -S uv run python
"""Regression tests for the waiver review record."""

import json
from pathlib import Path

from asm_heuristics import (FunctionBody, apply_select_branch_enforcement, apply_waivers,
                            attach_condition_identities, scan_symbol)
from renew_waivers import (Slicer, apply_identities, candidates, function_changes, reencoding,
                           unwaived_enforced_findings)

TARGET = "x86_64-unknown-linux-gnu"


def function(lines: list[str]) -> dict[str, FunctionBody]:
  return {"f": FunctionBody("f", Path("fixture.binary.disasm.txt"), 0x1000, list(enumerate(lines, start=1)))}


def findings(functions: dict[str, FunctionBody]) -> list[dict]:
  found = [item for item in scan_symbol(TARGET, "f", functions["f"], set(functions), scope="review",
                                        primitive_ids=["kx.p256_ecdh"], roots=[], image=None)
           if item["kind"] == "conditional_branch"]
  attach_condition_identities(TARGET, found, functions, "artifacts/fixture.binary.disasm.txt", set(functions), None,
                              {"conditional_branch"})
  return found


def loop(limit: str = "0x34", head: list[str] | None = None, start: str = "31 c0\txorl\t%eax, %eax") -> list[str]:
  prefix = head or []
  at = 0x1000 + 3 * len(prefix)
  return [*prefix, f"{at:x}: {start}", f"{at + 2:x}: 48 83 c0 01\taddq\t$0x1, %rax",
          f"{at + 6:x}: 48 83 f8 {limit[2:]}\tcmpq\t${limit}, %rax",
          f"{at + 10:x}: 75 f6\tjne\t0x{at + 2:x} <f+0x{at + 2 - 0x1000:x}>", f"{at + 12:x}: c3\tretq"]


def main() -> None:
  reviewed_functions = function(loop())
  reviewed = findings(reviewed_functions)[0]
  waiver = {"symbol": "f", "kind": "conditional_branch", "primitive": "kx.p256_ecdh",
            "slice_sha256": reviewed["slice_sha256"]}
  reviewed_record = Slicer(TARGET, reviewed_functions, None).reviewed(waiver)
  assert reviewed_record is not None and reviewed_record["reasons"] == [], reviewed_record

  # Code outside a closed slice keeps the condition identity, so the gate keeps the waiver.
  moved = findings(function(loop(head=["1000: 48 89 fe\tmovq\t%rdi, %rsi"])))[0]
  assert moved["slice_sha256"] == waiver["slice_sha256"] and moved["function_sha256"] != reviewed["function_sha256"]

  # A changed condition is a changed candidate.
  changed_functions = function(loop(limit="0x35"))
  changed = candidates(waiver, findings(changed_functions), reviewed_record, Slicer(TARGET, changed_functions, None))
  assert len(changed) == 1 and not changed[0]["unchanged"], changed
  assert changed[0]["slice_sha256"] != waiver["slice_sha256"]

  # An open slice binds its function: moving it changes the identity but not the slice.
  call = "e8 fb 0f 00 00\tcallq\t0x2000 <g>"
  open_functions = function(loop(start=call))
  open_finding = findings(open_functions)[0]
  open_waiver = dict(waiver, slice_sha256=open_finding["slice_sha256"])
  open_record = Slicer(TARGET, open_functions, None).reviewed(open_waiver)
  assert open_record is not None and open_record["reasons"], open_record
  shifted_functions = function(loop(start=call, head=["1000: 48 89 fe\tmovq\t%rdi, %rsi"]))
  shifted = candidates(open_waiver, findings(shifted_functions), open_record, Slicer(TARGET, shifted_functions, None))
  assert len(shifted) == 1 and shifted[0]["unchanged"], shifted
  assert shifted[0]["slice_sha256"] != open_waiver["slice_sha256"]
  # The record shows what staled an open slice: here the added head instruction and the loop branch whose
  # function-relative destination moved with it. An unchanged function has no identity changes.
  changes = function_changes(Slicer(TARGET, open_functions, None), Slicer(TARGET, shifted_functions, None), "f")
  assert changes is not None and "+48 89 fe\tmovq\t%rdi, %rsi" in changes["diff"], changes
  assert changes["changed_lines"] == 3 and all(line[0] in "+-" for line in changes["diff"]), changes
  same = function_changes(Slicer(TARGET, open_functions, None), Slicer(TARGET, open_functions, None), "f")
  assert same == {"changed_lines": 0, "diff": []}, same
  assert function_changes(Slicer(TARGET, open_functions, None), Slicer(TARGET, {}, None), "f") is None
  # A waived finding is not a candidate.
  assert candidates(waiver, [dict(moved, waived=True)], reviewed_record, Slicer(TARGET, reviewed_functions, None)) == []
  reencoding_regressions(waiver, reviewed_functions)
  enforced_finding_regressions()


def reencoding_regressions(waiver: dict, functions: dict[str, FunctionBody]) -> None:
  """A re-encoding moves a waiver to a new identity only for the slice it reviewed."""
  # The reviewed scanner's records cross a process boundary as JSON. The current ones stay in this process.
  [old] = json.loads(json.dumps(Slicer(TARGET, functions, None).branches(waiver, None)))
  assert old["slice_sha256"] == waiver["slice_sha256"], old

  def at(changed: dict[str, FunctionBody], identity: str | None = None) -> list[dict]:
    rows = Slicer(TARGET, changed, None).branches(waiver, [old["line"]])
    return [dict(row, slice_sha256=identity or row["slice_sha256"]) for row in rows]

  # A changed identity of the same slice is a re-encoding; the same identity is unchanged.
  assert reencoding([old], at(functions, "a" * 64)) == ("re-encoded", "a" * 64)
  assert reencoding([old], at(functions)) == ("unchanged", old["slice_sha256"])
  # A slice with other instructions or edges needs a review: here the loop starts from an input.
  status, reason = reencoding([old], at(function(loop(start="48 89 f8\tmovq\t%rdi, %rax"))))
  assert status == "refused" and "instructions or edges" in reason, reason
  # So does a slice that a call now opens, and a waiver that the reviewed artifacts do not hold.
  status, reason = reencoding([old], at(function(loop(start="e8 fb 0f 00 00\tcallq\t0x2000 <g>"))))
  assert status == "refused", reason
  assert reencoding([], [])[0] == "refused"
  # Two branches that shared the reviewed identity must keep one identity.
  second = dict(old, line=old["line"] + 100)
  assert reencoding([old, second], [dict(old, slice_sha256="a" * 64), dict(second, slice_sha256="b" * 64)])[0] == "refused"

  # Applying a re-encoding changes only the identity of the named waiver.
  manifest = (
    '[[asm_waiver]]\ntarget = "t"\nslice_sha256 = "0"\nrationale = "same"\n\n'
    '[[asm_waiver]]\ntarget = "t"\nslice_sha256 = "0"\nrationale = "same"\n'
  )
  updated = apply_identities(manifest, {1: ("0", "1")})
  assert updated == manifest[: manifest.rindex('slice_sha256 = "0"')] + 'slice_sha256 = "1"\nrationale = "same"\n', updated
  try:
    apply_identities(manifest, {0: ("2", "1")})
  except SystemExit:
    pass
  else:
    raise AssertionError("a waiver without the old identity was changed")


def enforced_finding_regressions() -> None:
  """An enforced row fails the gate until each primitive that reaches it has a waiver, enforced or not.

  IBM Z CT 44965044968 failed on such a row that the record did not list: only the non-enforced
  `signature.ecdsa_p256_public_derivation` lacked a waiver."""
  functions = function(["1000: 40 f6 c7 08\ttestb\t$0x8, %dil", "1004: 74 05\tje\t0x100b <f+0xb>",
                        "1006: b8 01 00 00 00\tmovl\t$0x1, %eax", "100b: c3\tretq"])
  found = scan_symbol(TARGET, "f", functions["f"], set(functions), scope="ct_intended_call_closure",
                      primitive_ids=["kx.p256_ecdh", "kem.mlkem768"], roots=[], image=None)
  apply_select_branch_enforcement(found, {"kx.p256_ecdh"})
  attach_condition_identities(TARGET, found, functions, "artifacts/fixture.binary.disasm.txt", set(functions), None,
                              {"select_branch"})
  [branch] = [row for row in found if row["kind"] == "select_branch"]
  waiver = {"target": TARGET, "primitive": "kx.p256_ecdh", "symbol": "f", "kind": "select_branch",
            "slice_sha256": branch["slice_sha256"], "classification": "public", "source": "fixture",
            "rationale": "fixture", "reviewer": "fixture", "reviewed_at": "2026-11-21"}
  assert apply_waivers(found, [waiver], TARGET) == []
  [row] = unwaived_enforced_findings(found, Slicer(TARGET, functions, None))
  assert row["primitives"] == ["kem.mlkem768"] and row["slice_sha256"] == branch["slice_sha256"], row
  # A waiver for each primitive clears the row.
  assert apply_waivers(found, [waiver, dict(waiver, primitive="kem.mlkem768")], TARGET) == []
  assert unwaived_enforced_findings(found, Slicer(TARGET, functions, None)) == []


if __name__ == "__main__":
  main()
