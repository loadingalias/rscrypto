#!/usr/bin/env -S uv run python
"""Regression tests for the waiver review record."""

from pathlib import Path

from asm_heuristics import FunctionBody, attach_condition_identities, scan_symbol
from renew_waivers import Slicer, candidates

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
  # A waived finding is not a candidate.
  assert candidates(waiver, [dict(moved, waived=True)], reviewed_record, Slicer(TARGET, reviewed_functions, None)) == []


if __name__ == "__main__":
  main()
