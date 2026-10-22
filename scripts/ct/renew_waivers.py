#!/usr/bin/env -S uv run python
"""Prepare the review of asm waivers that no longer match.

A waiver binds the backward slice of the branch condition it reviewed (`slice_sha256`). The gate keeps a
waiver while a finding in the same function has that condition, so a waiver goes stale only when the
condition changed or, for an open slice, when its function changed. For each stale waiver, this tool finds
the reviewed branch in the artifacts the waiver was reviewed against and lists the new branches of the same
kind in the same function, with their slices and digests. A candidate with an unchanged slice differs only
because an open slice also binds its function; a reviewer still confirms its open leaves. The record also
lists every enforced finding that no waiver covers. The tool writes the record and changes nothing.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Embedded Windows Python omits the script directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import asm_heuristics as scanner
from condition_slice import FunctionSlices
from provenance import sha256_file

ROOT = Path(__file__).resolve().parents[2]


def linked_functions(artifact_dir: Path) -> tuple[dict[str, scanner.FunctionBody], Any, Path]:
  paths = sorted(artifact_dir.glob("*.binary.disasm.txt"))
  if len(paths) != 1:
    raise SystemExit(f"expected one linked disassembly in {artifact_dir}, found {len(paths)}")
  functions, _ = scanner.parse_disassembly(paths[0])
  image = scanner.load_linked_image(artifact_dir)
  if image is None:
    raise SystemExit(f"cannot read the linked binary in {artifact_dir}")
  image.set_functions(scanner.function_ranges(functions))
  return functions, image, paths[0]


def scan_new(target: str, artifact_dir: Path) -> tuple[list[dict[str, Any]], set[int]]:
  """Run the gate's scanner and return its findings and the indexes of stale waivers."""
  with tempfile.TemporaryDirectory() as out_dir:
    saved = sys.argv
    sys.argv = ["asm_heuristics.py", "--target", target, "--profile", "release", "--artifact-dir", str(artifact_dir),
                "--out-dir", out_dir]
    try:
      status = scanner.main()
    finally:
      sys.argv = saved
    report_path = Path(out_dir) / "asm-heuristics.json"
    if not report_path.exists():
      raise SystemExit(f"the scanner stopped before writing its report (status {status}); fix that failure first")
    report = json.loads(report_path.read_text())
  stale = {int(re.match(r"asm_waiver\[(\d+)\]", message).group(1)) for message in report["stale_waivers"]}
  return report["findings"], stale


class Slicer:
  """Condition slices and identities of findings in one linked binary."""

  def __init__(self, target: str, functions: dict[str, scanner.FunctionBody], image: Any) -> None:
    self.target, self.functions, self.image = target, functions, image
    self.analyses: dict[str, FunctionSlices] = {}

  def record(self, symbol: str, finding: dict[str, Any]) -> dict[str, Any]:
    if symbol not in self.analyses:
      self.analyses[symbol] = FunctionSlices(self.target, self.functions[symbol], self.image, set(self.functions))
    result = self.analyses[symbol].slice(finding["line"])
    return {
      "offset": finding["offset"], "text": finding["text"], "canonical": result.canonical, "reasons": result.reasons,
      "slice_sha256": scanner.condition_identity(result.canonical, result.reasons, finding["function_sha256"]),
    }

  def reviewed(self, waiver: dict[str, Any]) -> dict[str, Any] | None:
    """The finding in this binary whose condition identity is the waiver's."""
    body = self.functions.get(waiver["symbol"])
    if body is None:
      return None
    for finding in scanner.scan_symbol(self.target, waiver["symbol"], body, set(self.functions), scope="review",
                                       primitive_ids=[waiver["primitive"]], roots=[], image=self.image):
      if finding["kind"] == waiver["kind"]:
        record = self.record(waiver["symbol"], finding)
        if record["slice_sha256"] == waiver["slice_sha256"]:
          return record
    return None


def candidates(waiver: dict[str, Any], findings: list[dict[str, Any]], reviewed: dict[str, Any] | None,
               slicer: Slicer) -> list[dict[str, Any]]:
  """Unwaived new findings of the waiver's kind in its function; `unchanged` marks the reviewed slice."""
  rows = []
  for finding in findings:
    # Only linked-binary findings carry a condition identity.
    if (finding["symbol"], finding["kind"]) != (waiver["symbol"], waiver["kind"]) or finding.get("waived") \
        or waiver["primitive"] not in finding.get("primitive_ids", []) or "slice_sha256" not in finding:
      continue
    record = slicer.record(finding["symbol"], finding)
    record["unchanged"] = reviewed is not None and record["canonical"] == reviewed["canonical"]
    rows.append(record)
  return sorted(rows, key=lambda row: (not row["unchanged"], row["offset"] if row["offset"] is not None else -1))


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--target", required=True)
  parser.add_argument("--reviewed-artifacts", required=True, type=Path,
                      help="artifact directory that the current waivers were reviewed against")
  parser.add_argument("--artifacts", required=True, type=Path, help="new artifact directory")
  parser.add_argument("--out", required=True, type=Path, help="directory for the review record")
  args = parser.parse_args()

  configured = scanner.waivers(ROOT)
  errors = scanner.validate_waivers(configured)
  if errors:
    for error in errors:
      print(error)
    return 2
  findings, stale = scan_new(args.target, args.artifacts)
  old_functions, old_image, old_disassembly = linked_functions(args.reviewed_artifacts)
  new_functions, new_image, new_disassembly = linked_functions(args.artifacts)
  reviewed_slicer = Slicer(args.target, old_functions, old_image)
  new_slicer = Slicer(args.target, new_functions, new_image)

  rows = []
  for index in sorted(stale):
    waiver = configured[index]
    reviewed = reviewed_slicer.reviewed(waiver)
    rows.append({"index": index, "waiver": waiver, "reviewed": reviewed,
                 "candidates": candidates(waiver, findings, reviewed, new_slicer)})

  enforced = scanner.select_branch_enforced_primitives(ROOT)
  unwaived = []
  for finding in findings:
    missing = [primitive for primitive in finding.get("unresolved_primitive_ids", []) if primitive in enforced]
    if finding["kind"] == "select_branch" and missing and "slice_sha256" in finding:
      unwaived.append({"symbol": finding["symbol"], "primitives": missing, **new_slicer.record(finding["symbol"], finding)})

  counts = collections.Counter(
    "unchanged slice" if any(candidate["unchanged"] for candidate in row["candidates"]) else
    ("not in reviewed artifacts" if row["reviewed"] is None else "changed slice") for row in rows)
  record = {
    "kind": "rscrypto.ct.waiver-review",
    "schema_version": 2,
    "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    "target": args.target,
    "identity": scanner.SLICE_IDENTITY_VERSION,
    "ct_manifest_sha256": sha256_file(ROOT / "ct.toml"),
    "reviewed_disassembly": {"path": str(old_disassembly), "sha256": sha256_file(old_disassembly)},
    "new_disassembly": {"path": str(new_disassembly), "sha256": sha256_file(new_disassembly)},
    "stale_waivers": len(rows),
    "counts": dict(counts),
    "waivers": rows,
    "unwaived_enforced_findings": unwaived,
  }
  args.out.mkdir(parents=True, exist_ok=True)
  (args.out / "renewal-record.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
  (args.out / "renewal-record.md").write_text(markdown(record))
  print(f"{args.target}: {len(rows)} stale waivers {dict(counts)}; {len(unwaived)} enforced findings need review")
  return 0


def markdown(record: dict[str, Any]) -> str:
  lines = [
    f"# Waiver review record: `{record['target']}`",
    "",
    f"- Generated: {record['generated_at_utc']}",
    f"- Identity: `{record['identity']}`",
    f"- Reviewed disassembly SHA-256: `{record['reviewed_disassembly']['sha256']}`",
    f"- New disassembly SHA-256: `{record['new_disassembly']['sha256']}`",
    f"- Stale waivers: {record['stale_waivers']} ({', '.join(f'{name} {count}' for name, count in sorted(record['counts'].items()))})",
    f"- Enforced findings without a waiver: {len(record['unwaived_enforced_findings'])}",
    "",
    "A stale waiver's condition changed, or its open slice binds a function that changed. Review each",
    "candidate in the new artifact; to renew, set the waiver's `slice_sha256` to the candidate's digest.",
    "",
  ]
  for row in record["waivers"]:
    waiver = row["waiver"]
    lines.extend([f"## asm_waiver[{row['index']}] `{waiver['primitive']}` `{waiver['symbol']}`", "",
                  f"- Classification: {waiver['classification']}; source: `{waiver['source']}`"])
    if row["reviewed"] is None:
      lines.append("- The reviewed artifacts do not contain this condition.")
    else:
      lines.extend(["", "Reviewed slice:", "", "```text", row["reviewed"]["canonical"], "```"])
      lines.extend(f"- Open leaf: {reason}" for reason in row["reviewed"]["reasons"])
    for candidate in row["candidates"]:
      state = "unchanged slice" if candidate["unchanged"] else "changed slice"
      lines.extend(["", f"Candidate +0x{candidate['offset']:x} ({state}): `{candidate['slice_sha256']}`"])
      lines.extend(f"- Open leaf: {reason}" for reason in candidate["reasons"])
      if not candidate["unchanged"]:
        lines.extend(["", "```text", candidate["canonical"], "```"])
    lines.append("")
  if record["unwaived_enforced_findings"]:
    lines.extend([f"## Enforced findings without a waiver ({len(record['unwaived_enforced_findings'])})", ""])
    for finding in record["unwaived_enforced_findings"]:
      lines.extend([f"### `{finding['symbol']}` +0x{finding['offset']:x}", "",
                    f"- `slice_sha256 = \"{finding['slice_sha256']}\"` for {', '.join(finding['primitives'])}",
                    *[f"- Open leaf: {reason}" for reason in finding["reasons"]],
                    "", "```text", finding["canonical"], "```", ""])
  return "\n".join(lines)


if __name__ == "__main__":
  raise SystemExit(main())
