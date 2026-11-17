#!/usr/bin/env -S uv run python
"""Prepare the review of asm waivers that no longer match.

A waiver binds the backward slice of the branch condition it reviewed (`slice_sha256`). The gate keeps a
waiver while a finding in the same function has that condition, so a waiver goes stale only when the
condition changed or, for an open slice, when its function changed. For each stale waiver, this tool finds
the reviewed branch in the artifacts the waiver was reviewed against and lists the new branches of the same
kind in the same function, with their slices and digests. A candidate with an unchanged slice differs only
because an open slice also binds its function; a reviewer still confirms its open leaves. The record also
lists every enforced finding that no waiver covers. The tool writes the record and changes nothing.

With `--reencode-from COMMIT`, the tool instead binds the current waivers to a fixed instruction identity. It
finds each waiver's branches with the scanner of COMMIT and the same branches with the current scanner, both in
the artifacts the waivers were reviewed against. A waiver moves to its new identity only when each branch keeps
its slice: the same instructions, edges, and open reasons. `--apply` then changes only those identities in ct.toml.
"""

from __future__ import annotations

import argparse
import collections
import difflib
import json
import re
import subprocess
import sys
import tempfile
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Embedded Windows Python omits the script directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import asm_heuristics as scanner
from condition_slice import FunctionSlices, strip_offset
from provenance import sha256_file

ROOT = Path(__file__).resolve().parents[2]


def linked_functions(artifact_dir: Path, target: str) -> tuple[dict[str, scanner.FunctionBody], Any, Path]:
  paths = sorted(artifact_dir.glob("*.binary.disasm.txt"))
  if len(paths) != 1:
    raise SystemExit(f"expected one linked disassembly in {artifact_dir}, found {len(paths)}")
  functions, _ = scanner.parse_disassembly(paths[0])
  try:
    image = scanner.load_linked_image(artifact_dir)
  except ValueError as error:
    raise SystemExit(f"cannot describe the linked binary's data in {artifact_dir}: {error}") from error
  if image is None:
    raise SystemExit(f"cannot read the linked binary in {artifact_dir}")
  scanner.prepare_linked_image(target, functions, image)
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
    analysis = self.analyses[symbol]
    # A re-encoding runs this method with the reviewed scanner too, whose findings carry no table loads.
    result = analysis.slice(finding["line"], frozenset(analysis.index_by_line[line] for line in finding["table_loads"])) \
      if finding.get("table_loads") else analysis.slice(finding["line"])
    return {
      "line": finding["line"], "offset": finding["offset"], "text": finding["text"], "canonical": result.canonical,
      "reasons": result.reasons,
      # The slice without its instruction identities: the offset, edges, and effect kind of each node.
      "structure": [[node["offset"], sorted(node["inputs"].items()), node["load"], node["call"], node["unknown"]]
                    for node in result.nodes],
      "slice_sha256": scanner.condition_identity(result.canonical, result.reasons, finding["function_sha256"]),
    }

  def branches(self, waiver: dict[str, Any], lines: list[int] | None) -> list[dict[str, Any]]:
    """The waiver's branches: those whose identity is the waiver's, or those on `lines`."""
    body = self.functions.get(waiver["symbol"])
    if body is None:
      return []
    rows = []
    for finding in scanner.scan_symbol(self.target, waiver["symbol"], body, set(self.functions), scope="review",
                                       primitive_ids=[waiver["primitive"]], roots=[], image=self.image):
      if finding["kind"] != waiver["kind"] or (lines is not None and finding["line"] not in lines):
        continue
      record = self.record(waiver["symbol"], finding)
      if lines is not None or record["slice_sha256"] == waiver["slice_sha256"]:
        rows.append(record)
    return rows

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


FUNCTION_CHANGE_LIMIT = 40


def function_changes(reviewed: Slicer, new: Slicer, symbol: str) -> dict[str, Any] | None:
  """The instruction identities that differ between the reviewed and new function, ignoring offsets.

  An open slice binds its whole function, so this diff is what staled it. An empty diff, or one made only of
  placement such as a moved symbol name, means the identity bound placement: fix the identity, not the waiver.
  """
  if symbol not in reviewed.functions or symbol not in new.functions:
    return None

  def lines(slicer: Slicer) -> list[str]:
    identities = scanner.instruction_identities(slicer.target, slicer.functions[symbol], set(slicer.functions),
                                                slicer.image)
    return [strip_offset(identity) for identity in identities.values()]

  diff = [line for line in difflib.unified_diff(lines(reviewed), lines(new), lineterm="", n=0)
          if not line.startswith(("---", "+++", "@@"))]
  return {"changed_lines": len(diff), "diff": diff[:FUNCTION_CHANGE_LIMIT]}


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


def reencoding(old: list[dict[str, Any]], new: list[dict[str, Any]]) -> tuple[str, str]:
  """("re-encoded" or "unchanged", the new identity), or ("refused", the reason).

  `old` holds a waiver's branches under the identity it was reviewed with, and `new` the branches on the same
  lines under the current identity. A branch whose slice changed needs a review, not a new identity.
  """
  if not old:
    return "refused", "no branch in the reviewed artifacts has the waiver's identity"
  by_line = {record["line"]: record for record in new}
  for record in old:
    current = by_line.get(record["line"])
    if current is None:
      return "refused", f"the current scanner finds no branch at +0x{record['offset']:x}"
    # The reviewed records arrive as JSON, which turns tuples into lists.
    if json.loads(json.dumps(current["structure"])) != json.loads(json.dumps(record["structure"])):
      return "refused", f"the slice at +0x{record['offset']:x} has other instructions or edges"
    if current["reasons"] != record["reasons"]:
      return "refused", f"the slice at +0x{record['offset']:x} has other open reasons"
  identities = {by_line[record["line"]]["slice_sha256"] for record in old}
  if len(identities) != 1:
    return "refused", "the waiver's branches have different current identities"
  identity = identities.pop()
  return ("unchanged" if identity == old[0]["slice_sha256"] else "re-encoded"), identity


def apply_identities(text: str, changes: dict[int, tuple[str, str]]) -> str:
  """ct.toml with the `slice_sha256` of waiver `index` changed from `old` to `new`, for each (old, new) change."""
  starts = [match.start() for match in re.finditer(r"^\[\[asm_waiver\]\]$", text, re.M)] + [len(text)]
  pieces, position = [], 0
  for index, (old, new) in sorted(changes.items()):
    start, end = starts[index], starts[index + 1]
    line = f'slice_sha256 = "{old}"'
    if text[start:end].count(line) != 1:
      raise SystemExit(f"asm_waiver[{index}] has no single line {line}")
    pieces += [text[position:start], text[start:end].replace(line, f'slice_sha256 = "{new}"')]
    position = end
  updated = "".join(pieces) + text[position:]
  before, after = tomllib.loads(text)["asm_waiver"], tomllib.loads(updated)["asm_waiver"]
  for index, (waiver, changed) in enumerate(zip(before, after, strict=True)):
    if index in changes:
      waiver = dict(waiver, slice_sha256=changes[index][1])
    if waiver != changed:
      raise SystemExit(f"applying the identities would change asm_waiver[{index}] beyond its identity")
  return updated


def slices_of(target: str, artifact_dir: Path, requests: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
  """The branches of each requested waiver, by `Slicer.branches`, with this checkout's scanner."""
  functions, image, _ = linked_functions(artifact_dir, target)
  slicer = Slicer(target, functions, image)
  return {str(request["index"]): slicer.branches(request["waiver"], request.get("lines")) for request in requests}


def reviewed_slices(commit: str, target: str, artifact_dir: Path, requests: list[dict[str, Any]]) -> dict[str, Any]:
  """`slices_of` with the scanner of `commit`: this tool runs on that commit's `scripts/ct`."""
  with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    archive = subprocess.run(["git", "-C", str(ROOT), "archive", commit, "scripts/ct"], check=True,
                             capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", str(root)], input=archive, check=True)
    tool = root / "scripts" / "ct" / Path(__file__).name
    tool.write_bytes(Path(__file__).read_bytes())
    (root / "requests.json").write_text(json.dumps(requests))
    subprocess.run([sys.executable, str(tool), "--target", target, "--slices-of", str(artifact_dir),
                    "--requests", str(root / "requests.json"), "--out", str(root / "slices.json")], check=True)
    return json.loads((root / "slices.json").read_text())


def reencode(args: argparse.Namespace, configured: list[dict[str, Any]]) -> int:
  requests = [{"index": index, "waiver": waiver} for index, waiver in enumerate(configured)
              if waiver["target"] == args.target]
  old = reviewed_slices(args.reencode_from, args.target, args.reviewed_artifacts, requests)
  new = slices_of(args.target, args.reviewed_artifacts, [
    dict(request, lines=[record["line"] for record in old[str(request["index"])]]) for request in requests])
  rows = []
  for request in requests:
    key = str(request["index"])
    status, value = reencoding(old[key], new[key])
    rows.append({"index": request["index"], "waiver": request["waiver"], "status": status,
                 "identity": value if status != "refused" else None, "reason": value if status == "refused" else None,
                 "reviewed": old[key], "current": new[key]})
  counts = collections.Counter(row["status"] for row in rows)
  record = {
    "kind": "rscrypto.ct.waiver-reencoding",
    "schema_version": 1,
    "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    "target": args.target,
    "reviewed_scanner": args.reencode_from,
    "ct_manifest_sha256": sha256_file(ROOT / "ct.toml"),
    "counts": dict(counts),
    "waivers": rows,
  }
  args.out.mkdir(parents=True, exist_ok=True)
  (args.out / "reencoding-record.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
  (args.out / "reencoding-record.md").write_text(reencoding_markdown(record))
  print(f"{args.target}: {dict(counts)}")
  if args.apply:
    changes = {row["index"]: (row["waiver"]["slice_sha256"], row["identity"]) for row in rows
               if row["status"] == "re-encoded"}
    manifest = ROOT / "ct.toml"
    manifest.write_text(apply_identities(manifest.read_text(), changes))
    print(f"{args.target}: changed {len(changes)} waiver identities in ct.toml")
  return 0 if not counts["refused"] else scanner.FINDINGS_EXIT


def reencoding_markdown(record: dict[str, Any]) -> str:
  lines = [
    f"# Waiver re-encoding record: `{record['target']}`",
    "",
    f"- Generated: {record['generated_at_utc']}",
    f"- Reviewed identities: the scanner of `{record['reviewed_scanner']}`",
    f"- Waivers: {', '.join(f'{name} {count}' for name, count in sorted(record['counts'].items()))}",
    "",
    "A refused waiver's slice changed, or the reviewed artifacts do not hold it. It needs a review.",
    "",
  ]
  for row in record["waivers"]:
    waiver = row["waiver"]
    if row["status"] == "unchanged":
      continue
    lines.append(f"## asm_waiver[{row['index']}] `{waiver['primitive']}` `{waiver['symbol']}`: {row['status']}")
    lines.append("")
    if row["status"] == "re-encoded":
      lines.extend([f"- `{waiver['slice_sha256']}` -> `{row['identity']}`", ""])
      continue
    lines.extend([f"- {row['reason']}", ""])
    by_line = {record["line"]: record for record in row["current"]}
    for reviewed in row["reviewed"]:
      current = by_line.get(reviewed["line"])
      if current is None:
        continue
      diff = [line for line in difflib.unified_diff(reviewed["canonical"].splitlines(), current["canonical"].splitlines(),
                                                    lineterm="", n=0) if not line.startswith(("---", "+++", "@@"))]
      lines.extend([f"Slice at +0x{reviewed['offset']:x}:", "", "```diff", *diff, "```", ""])
  return "\n".join(lines)


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--target", required=True)
  parser.add_argument("--reviewed-artifacts", type=Path,
                      help="artifact directory that the current waivers were reviewed against")
  mode = parser.add_mutually_exclusive_group()
  mode.add_argument("--artifacts", type=Path, help="new artifact directory")
  mode.add_argument("--reencode-from", metavar="COMMIT",
                    help="move the waivers from the identities of COMMIT's scanner to the current ones")
  mode.add_argument("--slices-of", type=Path, help=argparse.SUPPRESS)
  parser.add_argument("--apply", action="store_true", help="with --reencode-from, change the identities in ct.toml")
  parser.add_argument("--requests", type=Path, help=argparse.SUPPRESS)
  parser.add_argument("--out", required=True, type=Path, help="directory for the record")
  args = parser.parse_args()
  if args.slices_of is not None:
    # The reviewed scanner's half of a re-encoding; see `reviewed_slices`.
    args.out.write_text(json.dumps(slices_of(args.target, args.slices_of, json.loads(args.requests.read_text()))))
    return 0
  if args.reviewed_artifacts is None or (args.artifacts is None and args.reencode_from is None):
    parser.error("--reviewed-artifacts and one of --artifacts or --reencode-from are required")
  if args.apply and args.reencode_from is None:
    parser.error("--apply needs --reencode-from")

  configured = scanner.waivers(ROOT)
  errors = scanner.validate_waivers(configured)
  if errors:
    for error in errors:
      print(error)
    return 2
  if args.reencode_from is not None:
    return reencode(args, configured)
  findings, stale = scan_new(args.target, args.artifacts)
  old_functions, old_image, old_disassembly = linked_functions(args.reviewed_artifacts, args.target)
  new_functions, new_image, new_disassembly = linked_functions(args.artifacts, args.target)
  reviewed_slicer = Slicer(args.target, old_functions, old_image)
  new_slicer = Slicer(args.target, new_functions, new_image)

  rows = []
  for index in sorted(stale):
    waiver = configured[index]
    reviewed = reviewed_slicer.reviewed(waiver)
    rows.append({"index": index, "waiver": waiver, "reviewed": reviewed,
                 "function_changes": function_changes(reviewed_slicer, new_slicer, waiver["symbol"]),
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
    "schema_version": 3,
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
    changes = row["function_changes"]
    if changes is not None and changes["changed_lines"] == 0:
      lines.extend(["", "Function instruction identities are unchanged."])
    elif changes is not None:
      shown = len(changes["diff"])
      more = f", first {shown} shown" if shown < changes["changed_lines"] else ""
      lines.extend(["", f"Function instruction identity changes ({changes['changed_lines']} lines{more}):", "",
                    "```diff", *changes["diff"], "```"])
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
