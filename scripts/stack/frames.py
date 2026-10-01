#!/usr/bin/env python3
"""Review secret-worker stack depth in linked release artifacts.

A secret worker runs below a public frame, and the scrub that follows it
overwrites a fixed region of that dead stack. This review proves, for one
linked binary, that every worker path fits inside the scrub region.

Frames come from the compiler's `.stack_sizes` records (`-Z emit-stack-sizes`).
Calls come from the linked disassembly and are resolved by address. A path is
unbounded when it reaches a function without a frame record, an indirect
transfer, an unresolved external target, or recursion. Unbounded paths and
capability detection reachable from a worker are reported, never dropped, and
each fails the review. Panic exits are listed separately: the review artifact
uses `panic = "abort"`, so they terminate the process instead of returning.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CRATE = ROOT / "tools" / "frame-review"
BINARY = "rscrypto-frame-review"


@dataclass(frozen=True)
class Arch:
  # Bytes a call pushes below the caller's stack pointer before the callee's
  # recorded frame begins.
  call_slot: int
  # Bytes below the stack pointer that a function may use without allocating.
  red_zone: int


ARCHES = {
  "x86_64-unknown-linux-gnu": Arch(call_slot=8, red_zone=128),
  "aarch64-unknown-linux-gnu": Arch(call_slot=0, red_zone=0),
  "powerpc64le-unknown-linux-gnu": Arch(call_slot=0, red_zone=288),
  "s390x-unknown-linux-gnu": Arch(call_slot=0, red_zone=0),
  "riscv64gc-unknown-linux-gnu": Arch(call_slot=0, red_zone=0),
}


@dataclass(frozen=True)
class Boundary:
  """One scrubbed secret-worker boundary in the source."""

  name: str
  workers: re.Pattern[str]
  scrub: re.Pattern[str]
  source: Path
  constant: str


BOUNDARIES = [
  Boundary(
    name="ml-dsa secret SHAKE256",
    workers=re.compile(r"^rscrypto::auth::mldsa::sampling::absorb_and_squeeze$"),
    scrub=re.compile(r"^rscrypto::auth::mldsa::sampling::scrub_dead_stack$"),
    source=ROOT / "src" / "auth" / "mldsa" / "sampling.rs",
    constant="HASH_STACK_SCRUB_WORDS",
  ),
]

DETECTION = re.compile(
  r"rscrypto::platform::detect|rscrypto::platform::caps|std_detect|"
  r"\bgetauxval\b|\bis_[a-z0-9_]+_feature_detected\b"
)
PANIC_EXIT = re.compile(
  r"^(core::panicking::|core::slice::index::|core::str::slice_error_fail|"
  r"core::option::(unwrap|expect)_failed|core::result::unwrap_failed|"
  r"alloc::alloc::handle_alloc_error|alloc::raw_vec::handle_error|"
  r"std::process::abort|std::sys::pal::.*::abort_internal|core::cell::panic_already_)"
)
# libc memory primitives are leaf routines. They have no frame records, so the
# review counts them as frameless leaves (plus the target red zone) and lists
# the assumption in every report.
EXTERNAL_LEAVES = {"memcpy", "memmove", "memset", "memcmp", "bcmp"}

HEADER = re.compile(r"^([0-9a-f]+) <(.+)>:$")
INSTRUCTION = re.compile(r"^\s*([0-9a-f]+):\s+(\S+)(?:\s+(.*))?$")
ANNOTATION = re.compile(r"\s*<([^>]+)>\s*$")


@dataclass
class Edge:
  kind: str  # call, tail, import, import-tail, indirect, external
  source: int
  target: int | None = None
  name: str | None = None


@dataclass
class Function:
  address: int
  name: str
  demangled: str
  frame: int | None = None
  edges: list[Edge] = field(default_factory=list)


@dataclass
class Program:
  target: str
  functions: dict[int, Function]
  starts: list[int]

  def containing(self, address: int) -> Function | None:
    low, high = 0, len(self.starts)
    while low < high:
      middle = (low + high) // 2
      if self.starts[middle] <= address:
        low = middle + 1
      else:
        high = middle
    if low == 0:
      return None
    return self.functions[self.starts[low - 1]]

  def named(self, pattern: re.Pattern[str]) -> list[Function]:
    return [function for function in self.functions.values() if pattern.search(function.demangled)]


def llvm_tool(name: str) -> Path:
  sysroot = subprocess.run(
    ["rustc", "--print", "sysroot"], check=True, capture_output=True, text=True, cwd=ROOT
  ).stdout.strip()
  host = subprocess.run(
    ["rustc", "--print", "host-tuple"], check=True, capture_output=True, text=True, cwd=ROOT
  ).stdout.strip()
  path = Path(sysroot) / "lib" / "rustlib" / host / "bin" / name
  if not path.is_file():
    raise SystemExit(f"{name} is missing from the pinned toolchain; install the llvm-tools component")
  return path


def run(tool: str, *args: str) -> str:
  return subprocess.run([str(llvm_tool(tool)), *args], check=True, capture_output=True, text=True).stdout


def parse_symbols(mangled: str, demangled: str) -> dict[int, tuple[str, str]]:
  """Map each defined function address to its first mangled and demangled name."""
  symbols: dict[int, tuple[str, str]] = {}
  rows = zip(mangled.splitlines(), demangled.splitlines(), strict=True)
  for raw, readable in rows:
    raw_fields = raw.split(maxsplit=2)
    readable_fields = readable.split(maxsplit=2)
    if len(raw_fields) != 3 or raw_fields[1] not in {"t", "T", "W", "w"} or raw_fields[2].startswith(".L"):
      continue
    if raw_fields[:2] != readable_fields[:2]:
      raise ValueError(f"mangled and demangled symbol tables disagree at {raw!r}")
    symbols.setdefault(int(raw_fields[0], 16), (raw_fields[2], readable_fields[2]))
  return symbols


def parse_stack_sizes(text: str) -> dict[str, int]:
  sizes: dict[str, int] = {}
  names: list[str] = []
  for line in text.splitlines():
    line = line.strip()
    if line.startswith("Functions: ["):
      names = [name.strip() for name in line.removeprefix("Functions: [").removesuffix("]").split(",")]
    elif line.startswith("Size: "):
      size = int(line.removeprefix("Size: "), 0)
      for name in names:
        sizes[name] = size
      names = []
  return sizes


def target_address(operands: str) -> int | None:
  """Return the absolute target objdump prints as the last branch operand."""
  operand = ANNOTATION.sub("", operands).rsplit(",", 1)[-1].strip()
  if re.fullmatch(r"0x[0-9a-f]+", operand) is None:
    return None
  return int(operand, 16)


def classify(target: str, mnemonic: str, operands: str) -> str | None:
  """Return call, branch, indirect-call, indirect-jump, or None for one instruction."""
  arch = target.split("-", 1)[0]
  if arch == "x86_64":
    if mnemonic.startswith("call"):
      return "indirect-call" if "*" in operands else "call"
    if mnemonic.startswith("j"):
      return "indirect-jump" if "*" in operands else "branch"
    return None
  if arch == "aarch64":
    if mnemonic in {"bl"}:
      return "call"
    if mnemonic in {"blr", "blraa", "blraaz", "blrab", "blrabz"}:
      return "indirect-call"
    if mnemonic in {"br", "braa", "braaz", "brab", "brabz"}:
      return "indirect-jump"
    if mnemonic == "b" or mnemonic.startswith("b.") or mnemonic in {"cbz", "cbnz", "tbz", "tbnz"}:
      return "branch"
    return None
  if arch == "powerpc64le":
    if mnemonic in {"bl", "bla"}:
      return "call"
    if mnemonic.endswith("ctrl") or mnemonic.endswith("lrl"):
      return "indirect-call"
    if mnemonic in {"bctr", "bcctr"} or (mnemonic.endswith("ctr") and mnemonic.startswith("b")):
      return "indirect-jump"
    if mnemonic.endswith("lr"):
      return None  # returns and conditional returns
    if mnemonic.startswith("b"):
      return "branch"
    return None
  if arch == "s390x":
    if mnemonic in {"brasl", "bras", "jasl", "jas"}:
      return "call"
    if mnemonic in {"basr", "bas"}:
      return "indirect-call"
    if mnemonic in {"br", "bcr"} or (mnemonic.startswith("b") and mnemonic.endswith("r")):
      return None if "%r14" in operands else "indirect-jump"
    if mnemonic.startswith("j") or mnemonic.startswith("br"):
      return "branch"
    return None
  if arch.startswith("riscv"):
    if mnemonic in {"jal", "c.jal"}:
      first = operands.split(",", 1)[0].strip()
      return "branch" if first == "zero" else "call"
    if mnemonic in {"j", "c.j"} or mnemonic.startswith("b") or mnemonic.startswith("c.b"):
      return "branch"
    return None
  raise ValueError(f"unsupported target {target}")


RISCV_REGISTER_OFFSET = re.compile(r"(-?0x[0-9a-f]+|-?\d+)\((\w+)\)")


def riscv_jalr(operands: str, upper: dict[str, int]) -> tuple[str, int | None]:
  """Resolve `jalr`/`jr` through a preceding `auipc`; return (kind, target)."""
  text = ANNOTATION.sub("", operands).strip()
  parts = [part.strip() for part in text.split(",")]
  link = "ra"
  if len(parts) == 2:
    link, text = parts
  elif len(parts) == 3:
    link, base, offset = parts
    text = f"{offset}({base})"
  match = RISCV_REGISTER_OFFSET.fullmatch(text)
  if match is None:
    base, offset = text, 0
  else:
    offset, base = int(match.group(1), 0), match.group(2)
  calls = link != "zero"
  if base == "ra" and not calls:
    return ("return", None)
  if base not in upper:
    return ("indirect-call" if calls else "indirect-jump", None)
  return ("call" if calls else "branch", (upper[base] + offset) & ((1 << 64) - 1))


GOT_SLOT = re.compile(r"\*0x[0-9a-f]+\(%rip\)\s+#\s+0x([0-9a-f]+)")


def parse_dynamic_symbols(text: str) -> dict[int, str]:
  """Map each GOT slot with a symbolic dynamic relocation to its import."""
  slots: dict[int, str] = {}
  for line in text.splitlines():
    fields = line.split()
    if len(fields) == 3 and re.fullmatch(r"[0-9a-f]+", fields[0]) and not fields[2].startswith("*ABS*"):
      slots[int(fields[0], 16)] = fields[2].split("@", 1)[0]
  return slots


def parse_disassembly(
  target: str,
  text: str,
  symbols: dict[int, tuple[str, str]],
  sizes: dict[str, int],
  imports: dict[int, str] | None = None,
) -> Program:
  functions: dict[int, Function] = {}
  for address, (mangled, demangled) in symbols.items():
    functions[address] = Function(address, mangled, demangled, sizes.get(mangled))

  current: Function | None = None
  rows: list[tuple[Function, int, str, str]] = []
  for line in text.splitlines():
    if header := HEADER.match(line):
      address = int(header.group(1), 16)
      name = header.group(2)
      if name.startswith(".L"):
        continue  # assembler-local label inside the current function
      current = functions.get(address)
      if current is None:
        # Linker-synthesized entries (PLT stubs, section labels) have no
        # symbol-table function. Keep them as frameless targets.
        current = Function(address, name, name, None)
        functions[address] = current
      continue
    if current is None or (instruction := INSTRUCTION.match(line)) is None:
      continue
    rows.append((current, int(instruction.group(1), 16), instruction.group(2), instruction.group(3) or ""))

  program = Program(target, functions, sorted(functions))
  upper: dict[str, int] = {}
  previous: Function | None = None
  for function, address, mnemonic, operands in rows:
    if function is not previous:
      upper = {}
      previous = function
    is_riscv = target.startswith("riscv")
    if is_riscv and mnemonic == "auipc":
      register, immediate = (part.strip() for part in operands.split(","))
      value = int(immediate, 0) & 0xFFFFF
      value -= (value & 0x80000) << 1  # the 20-bit immediate is signed
      upper[register] = (address + (value << 12)) & ((1 << 64) - 1)
      continue
    if is_riscv and mnemonic not in {"jalr", "jr", "c.jalr", "c.jr"}:
      # Any other write to a register ends what `auipc` proved about it.
      # Treating stores and branches as writes only makes a later `jalr`
      # indirect, which is the conservative direction.
      upper.pop(operands.split(",", 1)[0].strip(), None)
    if is_riscv and mnemonic in {"jalr", "jr", "c.jalr", "c.jr"}:
      if mnemonic in {"jr", "c.jr"}:
        operands = f"zero, {operands}"
      kind, destination = riscv_jalr(operands, upper)
      if kind == "return":
        continue
      if kind in {"call", "indirect-call"}:
        upper.pop("ra", None)  # the link register now holds the return address
    elif is_riscv and mnemonic == "ret":
      continue
    else:
      kind = classify(target, mnemonic, operands)
      destination = target_address(operands) if kind in {"call", "branch"} else None
    if kind is None:
      continue
    if kind in {"indirect-call", "indirect-jump"}:
      # A call through a GOT slot names its import in the dynamic relocations.
      slot = GOT_SLOT.search(operands)
      if slot is not None and imports and int(slot.group(1), 16) in imports:
        stub = imports[int(slot.group(1), 16)]
        function.edges.append(Edge("import" if kind == "indirect-call" else "import-tail", address, name=stub))
        continue
      function.edges.append(Edge("indirect", address, name=f"{mnemonic} {operands}".strip()))
      continue
    if destination is None:
      function.edges.append(Edge("indirect", address, name=f"{mnemonic} {operands}".strip()))
      continue
    callee = program.containing(destination)
    annotation = ANNOTATION.search(operands)
    if callee is None:
      function.edges.append(Edge("external", address, destination, annotation.group(1) if annotation else None))
      continue
    if kind == "branch" and callee is function:
      continue
    function.edges.append(Edge("call" if kind == "call" else "tail", address, callee.address))
  return program


def external_name(program: Program, function: Function) -> str | None:
  """Return the imported symbol behind a linker-synthesized PLT stub."""
  name = function.name
  for prefix in ("__plt_",):
    if name.startswith(prefix):
      return name.removeprefix(prefix)
  if name.endswith("@plt"):
    return name.removesuffix("@plt")
  return None


@dataclass
class Depth:
  bytes: int
  path: list[str]
  unbounded: list[dict[str, object]]
  detection: list[list[str]]
  panic_exits: set[str]
  assumptions: set[str]


def depth_of(program: Program, root: Function) -> Depth:
  arch = ARCHES[program.target]
  memo: dict[int, Depth] = {}
  active: list[int] = []

  def visit(function: Function) -> Depth:
    if function.address in memo:
      return memo[function.address]
    if function.address in active:
      cycle = [program.functions[address].demangled for address in active[active.index(function.address) :]]
      return Depth(0, [function.demangled], [{"reason": "recursion", "path": cycle}], [], set(), set())
    active.append(function.address)
    name = function.demangled
    imported = external_name(program, function)
    if imported in EXTERNAL_LEAVES:
      result = Depth(arch.red_zone, [name], [], [], set(), {f"{imported}: external leaf, frame 0 + red zone"})
    elif function.frame is None:
      reason = "external" if imported is not None else "no frame record"
      result = Depth(0, [name], [{"reason": reason, "path": [name]}], [], set(), set())
    else:
      result = Depth(function.frame, [name], [], [], set(), set())
      calls = [edge for edge in function.edges if edge.kind in {"call", "import"}]
      if not calls:
        result.bytes = function.frame + arch.red_zone
      for edge in function.edges:
        if edge.kind in {"import", "import-tail"}:
          if edge.name in EXTERNAL_LEAVES:
            result.assumptions.add(f"{edge.name}: external leaf, frame 0 + red zone")
            base = function.frame + arch.call_slot if edge.kind == "import" else 0
            result.bytes = max(result.bytes, base + arch.red_zone)
          else:
            result.unbounded.append({"reason": "external", "path": [name, str(edge.name)]})
          if DETECTION.search(str(edge.name)):
            result.detection.append([name, str(edge.name)])
          continue
        if edge.kind in {"indirect", "external"}:
          label = edge.name or (f"{edge.target:#x}" if edge.target is not None else "?")
          result.unbounded.append({"reason": edge.kind, "path": [name, label]})
          continue
        callee = program.functions[edge.target]
        if PANIC_EXIT.search(callee.demangled):
          result.panic_exits.add(callee.demangled)
          continue
        child = visit(callee)
        base = function.frame + arch.call_slot if edge.kind == "call" else 0
        result.unbounded.extend({**item, "path": [name, *item["path"]]} for item in child.unbounded)
        result.detection.extend([name, *path] for path in child.detection)
        result.panic_exits |= child.panic_exits
        result.assumptions |= child.assumptions
        if DETECTION.search(callee.demangled):
          result.detection.append([name, callee.demangled])
        if base + child.bytes > result.bytes:
          result.bytes = base + child.bytes
          result.path = [name, *child.path]
    active.pop()
    memo[function.address] = result
    return result

  return visit(root)


def scrub_bound(boundary: Boundary) -> int:
  match = re.search(rf"const {boundary.constant}: usize = (\d+);", boundary.source.read_text())
  if match is None:
    raise ValueError(f"{boundary.constant} is missing from {boundary.source}")
  return int(match.group(1)) * 8


def review(program: Program, boundaries: list[Boundary] = BOUNDARIES) -> dict[str, object]:
  arch = ARCHES[program.target]
  results = []
  for boundary in boundaries:
    bound = scrub_bound(boundary)
    problems: list[str] = []
    scrubs = program.named(boundary.scrub)
    workers = program.named(boundary.workers)
    if len(scrubs) != 1:
      problems.append(f"expected one linked scrub, found {len(scrubs)}")
    if not workers:
      problems.append("no linked worker")
    scrub = scrubs[0] if len(scrubs) == 1 else None
    # A leaf scrub may keep part of its buffer in the red zone, so compare the
    # buffer with the scrub's full extent rather than its recorded frame.
    scrub_extent = depth_of(program, scrub) if scrub is not None else None
    if scrub_extent is not None and (scrub_extent.unbounded or scrub_extent.bytes < bound):
      problems.append(f"scrub extent {scrub_extent.bytes} B cannot hold its {bound}-byte buffer")
    worker_reports = []
    for worker in workers:
      depth = depth_of(program, worker)
      total = depth.bytes + arch.call_slot
      caller_functions = [
        function
        for function in program.functions.values()
        if any(edge.kind == "call" and edge.target == worker.address for edge in function.edges)
      ]
      callers = sorted(function.demangled for function in caller_functions)
      unscrubbed = sorted(
        function.demangled
        for function in caller_functions
        if scrub is None or not any(edge.kind == "call" and edge.target == scrub.address for edge in function.edges)
      )
      worker_problems = []
      if depth.unbounded:
        worker_problems.append(f"{len(depth.unbounded)} unbounded path(s)")
      if depth.detection:
        worker_problems.append(f"{len(depth.detection)} capability-detection path(s)")
      if total > bound:
        worker_problems.append(f"depth {total} exceeds the {bound}-byte scrub")
      if not callers:
        worker_problems.append("no direct caller")
      if unscrubbed:
        worker_problems.append(f"callers without the scrub: {', '.join(unscrubbed)}")
      worker_reports.append(
        {
          "worker": worker.demangled,
          "frame": worker.frame,
          "depth": total,
          "deepest_path": depth.path,
          "callers": callers,
          "unbounded": depth.unbounded,
          "detection": depth.detection,
          "panic_exits": sorted(depth.panic_exits),
          "assumptions": sorted(depth.assumptions),
          "problems": worker_problems,
        }
      )
    passed = not problems and all(not report["problems"] for report in worker_reports)
    results.append(
      {
        "boundary": boundary.name,
        "bound": bound,
        "scrub_frame": scrub.frame if scrub is not None else None,
        "workers": worker_reports,
        "problems": problems,
        "passed": passed,
      }
    )
  return {"target": program.target, "boundaries": results, "passed": all(item["passed"] for item in results)}


def load(target: str, binary: Path) -> Program:
  mangled = run("llvm-nm", "--defined-only", "-n", str(binary))
  demangled = run("llvm-nm", "--defined-only", "-n", "-C", str(binary))
  symbols = parse_symbols(mangled, demangled)
  sizes = parse_stack_sizes(run("llvm-readobj", "--stack-sizes", str(binary)))
  if not sizes:
    raise SystemExit(f"{binary} has no .stack_sizes records; build it with -Z emit-stack-sizes")
  disassembly = run("llvm-objdump", "-d", "--no-show-raw-insn", str(binary))
  imports = parse_dynamic_symbols(run("llvm-objdump", "-R", str(binary)))
  return parse_disassembly(target, disassembly, symbols, sizes, imports)


def build(target: str, portable: bool, rustflags: list[str]) -> Path:
  if target not in ARCHES:
    raise SystemExit(f"unsupported target {target}; choose from {', '.join(ARCHES)}")
  target_dir = ROOT / "target" / "frame-review" / ("portable" if portable else "native")
  environment = dict(os.environ)
  key = target.upper().replace("-", "_")
  # The default is the shipping configuration: no target flags beyond the
  # frame records the review reads.
  flags = ["-Z", "emit-stack-sizes", *rustflags]
  environment["CARGO_ENCODED_RUSTFLAGS"] = "\x1f".join(flags)
  environment.pop("RUSTFLAGS", None)
  host = subprocess.run(["rustc", "--print", "host-tuple"], check=True, capture_output=True, text=True).stdout.strip()
  if target != host:
    environment[f"CARGO_TARGET_{key}_LINKER"] = str(ROOT / "scripts" / "ct" / "zig-cc.sh")
    environment["ZIG_CC_TARGET"] = target
  command = [
    "cargo",
    "build",
    "--locked",
    "--release",
    "--manifest-path",
    str(CRATE / "Cargo.toml"),
    "--target",
    target,
    "--target-dir",
    str(target_dir),
  ]
  if portable:
    command += ["--features", "portable-only"]
  subprocess.run(command, check=True, env=environment, cwd=ROOT)
  return target_dir / target / "release" / BINARY


def markdown(report: dict[str, object]) -> str:
  lines = [f"## {report['target']}: {'pass' if report['passed'] else 'FAIL'}", ""]
  for boundary in report["boundaries"]:
    lines.append(
      f"### {boundary['boundary']} (scrub {boundary['bound']} B, scrub frame {boundary['scrub_frame']} B)"
    )
    lines.extend(f"- problem: {problem}" for problem in boundary["problems"])
    for worker in boundary["workers"]:
      status = "pass" if not worker["problems"] else "FAIL: " + "; ".join(worker["problems"])
      lines.append(f"- `{worker['worker']}`: frame {worker['frame']} B, depth {worker['depth']} B, {status}")
      lines.append(f"  - deepest: {' -> '.join(worker['deepest_path'])}")
      for path in worker["detection"]:
        lines.append(f"  - detection: {' -> '.join(path)}")
      for item in worker["unbounded"]:
        lines.append(f"  - unbounded ({item['reason']}): {' -> '.join(item['path'])}")
      for assumption in worker["assumptions"]:
        lines.append(f"  - assumption: {assumption}")
      if worker["panic_exits"]:
        lines.append(f"  - panic exits (abort): {len(worker['panic_exits'])}")
    lines.append("")
  return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--target", action="append", help="target to build and review; repeatable (default: all)")
  parser.add_argument("--portable", action="store_true", help="build with the portable-only feature")
  parser.add_argument("--rustflags", default="", help="extra rustc flags for the build, space-separated")
  parser.add_argument("--binary", type=Path, help="review an existing binary (requires one --target)")
  parser.add_argument("--json", type=Path, help="write the full report as JSON")
  args = parser.parse_args(argv)

  targets = args.target or list(ARCHES)
  if args.binary is not None and len(targets) != 1:
    parser.error("--binary requires exactly one --target")
  reports = []
  for target in targets:
    binary = args.binary or build(target, args.portable, args.rustflags.split())
    report = review(load(target, binary))
    report["binary"] = str(binary)
    report["portable"] = args.portable
    report["rustflags"] = args.rustflags
    reports.append(report)
    print(markdown(report))
  if args.json is not None:
    args.json.write_text(json.dumps(reports, indent=2) + "\n")
  return 0 if all(report["passed"] for report in reports) else 1


if __name__ == "__main__":
  sys.exit(main())
