#!/usr/bin/env -S uv run python
"""Review secret-worker stack depth in linked release artifacts.

A secret worker runs below a public frame, and the scrub that follows it
overwrites a fixed region of that dead stack. This review proves, for one
linked binary, that every worker path fits inside the scrub region.

Frames come from the compiler's `.stack_sizes` records (`-Z emit-stack-sizes`).
On x86-64 and AArch64, code without a record, such as hand-written assembly,
gets a frame derived from its instructions when every stack-pointer change is
provably bounded (see `assembly_region`); each derived frame is listed as an
assumption. Calls come from the linked disassembly and are resolved by address;
a checked jump table whose entries stay inside its function is resolved too
(see `resolve_jump_tables`).
A path is unbounded when it reaches a function without a frame record or a
derivable frame, an indirect transfer, an unresolved external target, or
recursion. Unbounded paths and
capability detection reachable from a worker are reported, never dropped, and
each fails the review. Panic exits are listed separately: the review artifact
uses `panic = "abort"`, so they terminate the process instead of returning.

`--mldsa-preparation` reviews whole preparation calls instead. Besides the
whole-call traversal, it reports an owned bound that stops at declared libc
imports and cold capability detection, with each edge's entry depth. A
whole-call bound composes that owned bound with measured `--allowances`.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
from datetime import datetime, timezone
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
  # Targets whose artifact links the workers; None means every reviewed target.
  targets: frozenset[str] | None = None
  # Whether a portable-only build links the workers.
  portable: bool = True


SECRET_HASHING = ROOT / "src" / "hashes" / "crypto" / "keccak" / "secret.rs"
BOUNDARIES = [
  Boundary(
    name="secret SHA-3/SHAKE (ML-DSA; ML-KEM G, J, PRF)",
    workers=re.compile(
      r"^rscrypto::hashes::crypto::keccak::secret::"
      r"(absorb_and_squeeze::<\d+>|squeeze_seeded|squeeze_seeded_pair|squeeze_seeded64_pair)$"
    ),
    scrub=re.compile(r"^rscrypto::backend::stack::scrub_dead_stack::<256>$"),
    source=SECRET_HASHING,
    constant="STACK_SCRUB_WORDS",
  ),
  Boundary(
    name="secret four-state SHAKE256 (ML-KEM PRF)",
    workers=re.compile(r"^rscrypto::hashes::crypto::keccak::secret::squeeze_seeded_quad$"),
    scrub=re.compile(r"^rscrypto::backend::stack::scrub_dead_stack::<512>$"),
    source=SECRET_HASHING,
    constant="QUAD_STACK_SCRUB_WORDS",
  ),
  Boundary(
    name="AES-GCM generic CTR compiler spills",
    workers=re.compile(r"^rscrypto::aead::aes::aes(?:128|256)_ctr32_encrypt_be_worker$"),
    scrub=re.compile(r"^rscrypto::backend::stack::scrub_dead_stack::<384>$"),
    source=ROOT / "src" / "aead" / "aes.rs",
    constant="CTR_STACK_SCRUB_WORDS",
  ),
  Boundary(
    name="AES-GCM x86-64 VAES message tails",
    workers=re.compile(r"^rscrypto::aead::aes::x86_short::(?:seal|open)_worker::<(?:11|15)>$"),
    scrub=re.compile(r"^rscrypto::backend::stack::scrub_dead_stack_avx512::<40>$"),
    source=ROOT / "src" / "aead" / "aes" / "x86_64_short.rs",
    constant="STACK_SCRUB_WORDS",
    targets=frozenset({"x86_64-unknown-linux-gnu"}),
    portable=False,
  ),
  Boundary(
    name="ChaCha20 x86-64 AVX-512 column batches",
    workers=re.compile(r"^rscrypto::backend::chacha20::x86_avx512::columns$"),
    scrub=re.compile(r"^rscrypto::backend::stack::scrub_dead_stack_avx512::<160>$"),
    source=ROOT / "src" / "backend" / "chacha20" / "x86_64_avx512.rs",
    constant="COLUMN_STACK_SCRUB_WORDS",
    targets=frozenset({"x86_64-unknown-linux-gnu"}),
    portable=False,
  ),
  Boundary(
    name="ChaCha20-Poly1305 x86-64 AVX-512 one-pass open",
    workers=re.compile(r"^rscrypto::aead::chacha20poly1305::x86_64_avx512::open_one_pass::<[12]>$"),
    scrub=re.compile(r"^rscrypto::backend::stack::scrub_dead_stack_avx512::<40>$"),
    source=ROOT / "src" / "aead" / "chacha20poly1305" / "x86_64_avx512.rs",
    constant="ONE_PASS_STACK_SCRUB_WORDS",
    targets=frozenset({"x86_64-unknown-linux-gnu"}),
    portable=False,
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
# Historical scrub reviews assume these libc routines are leaves. Whole-call
# preparation reviews do not make this assumption: imports and IFUNC selection
# require evidence from the actual deployed libc and loader.
EXTERNAL_LEAVES = {"memcpy", "memmove", "memset", "memcmp", "bcmp"}
# Preparation reviews stop at these edges and record the stack depth at their
# entry. Each edge needs a separately measured allowance for the deployed libc
# (and, for `getauxval`, the kernel); none is assumed frameless. `getauxval` is
# the AArch64, POWER, and IBM Z Linux capability read. Cold first-call capability
# detection is otherwise owned code: the lock-free cache, CPUID, STFLE, and
# RISC-V hwprobe. Any other import, including IFUNC relocations, remains unresolved.
DECLARED_LIBC = {"calloc", "malloc", "free", "memcpy", "memmove", "memset", "memcmp", "bcmp", "getauxval"}

HEADER = re.compile(r"^([0-9a-f]+) <(.+)>:$")
INSTRUCTION = re.compile(r"^\s*([0-9a-f]+):\s+(\S+)(?:\s+(.*))?$")
ANNOTATION = re.compile(r"\s*<([^>]+)>\s*$")


@dataclass
class Edge:
  kind: str  # call, tail, import, import-tail, indirect, external, jump-table
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
  instructions: list[tuple[str, str]] = field(default_factory=list)
  instruction_addresses: set[int] = field(default_factory=set)


@dataclass
class Program:
  target: str
  functions: dict[int, Function]
  starts: list[int]
  # Every resolved direct branch destination in the binary.
  branch_targets: set[int] = field(default_factory=set)

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
    function = self.functions[self.starts[low - 1]]
    return function if address in function.instruction_addresses else None

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
    if mnemonic.startswith(("j", "br", "crj", "cgrj", "cij", "cgij", "clrj", "clgrj", "clij", "clgij")):
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
GOT_LOAD = re.compile(r"^-?0x[0-9a-f]+\(%rip\),\s*(%\w+)\s+#\s+0x([0-9a-f]+)")
X86_PRESERVED_IMPORTS = {"%rbx", "%rbp", "%r12", "%r13", "%r14", "%r15"}
X86_CALL_CLOBBERED = {"%rax", "%rcx", "%rdx", "%rsi", "%rdi", "%r8", "%r9", "%r10", "%r11"}
# Instructions whose only general-register write is their explicit last operand.
# Any other instruction may write a caller-saved register implicitly (`mul`,
# `cqto`, string operations, `syscall`, ...), so it ends every caller-saved proof.
X86_EXPLICIT_WRITES = {
  "movq", "movl", "movw", "movb", "movabsq", "movzbl", "movzwl", "movzbq", "movzwq", "movsbl", "movswl",
  "movsbq", "movswq", "movslq", "leaq", "leal", "xorl", "xorq", "addl", "addq", "subl", "subq", "andl",
  "andq", "orl", "orq", "cmpb", "cmpw", "cmpl", "cmpq", "testb", "testw", "testl", "testq", "incl", "incq",
  "decl", "decq", "negl", "negq", "notl", "notq", "shll", "shlq", "shrl", "shrq", "sarl", "sarq", "nop",
  "nopl", "nopw", "pushq", "popq", "movups", "movaps", "movdqu", "movdqa", "xorps", "pxor", "vmovups",
  "vmovaps", "vmovdqu", "vmovdqa", "vxorps", "vpxor",
}
X86_LEGACY_REGISTERS = {
  "%rax": ("%eax", "%ax", "%al", "%ah"), "%rbx": ("%ebx", "%bx", "%bl", "%bh"),
  "%rcx": ("%ecx", "%cx", "%cl", "%ch"), "%rdx": ("%edx", "%dx", "%dl", "%dh"),
  "%rsi": ("%esi", "%si", "%sil"), "%rdi": ("%edi", "%di", "%dil"), "%rbp": ("%ebp", "%bp", "%bpl"),
  "%rsp": ("%esp", "%sp", "%spl"),
}
X86_PARENT_REGISTER = {alias: parent for parent, aliases in X86_LEGACY_REGISTERS.items() for alias in aliases}


def x86_destination(operands: str) -> str | None:
  """The register an AT&T-syntax instruction writes, if its last operand is one."""
  text = operands.split("#", 1)[0].strip()
  last = text.rsplit(",", 1)[-1].strip() if text else ""
  return last if re.fullmatch(r"%\w+", last) else None


def x86_parent_register(register: str) -> str:
  """The 64-bit general register that contains `register`, or `register` itself."""
  return X86_PARENT_REGISTER.get(register, re.sub(r"^(%r(?:[89]|1[0-5]))[dwb]$", r"\1", register))


def x86_import_calls(rows, imports: dict[int, str | int]) -> dict[int, str | int]:
  """Prove GOT values in registers at calls by meeting all CFG predecessors.

  Full-width GOT loads and full-width register copies are tracked. A branch
  that bypasses a load, a partial-register write, or differing incoming values
  makes the target unknown. Calls preserve the ELF ABI's callee-saved registers
  and clobber the others, and any instruction outside `X86_EXPLICIT_WRITES`
  also ends every caller-saved proof.
  """
  by_function = {}
  for function, address, mnemonic, operands in rows:
    by_function.setdefault(function.address, []).append((address, mnemonic, operands))
  resolved = {}
  for instructions in by_function.values():
    positions = {row[0]: index for index, row in enumerate(instructions)}
    states = {0: {}}
    pending = [0]
    while pending:
      index = pending.pop()
      address, mnemonic, operands = instructions[index]
      state = dict(states[index])
      if mnemonic.startswith(("call", "j")):
        register = operands.strip().removeprefix("*")
        if register in state:
          resolved[address] = state[register]
        else:
          resolved.pop(address, None)
        if mnemonic.startswith("call"):
          state = {key: value for key, value in state.items() if key not in X86_CALL_CLOBBERED}
      else:
        if mnemonic not in X86_EXPLICIT_WRITES:
          state = {key: value for key, value in state.items() if key not in X86_CALL_CLOBBERED}
        destination = x86_destination(operands)
        copied = state.get(operands.split(",", 1)[0].strip()) if mnemonic == "movq" else None
        if destination:
          destination = x86_parent_register(destination)
          state.pop(destination, None)
          if copied is not None and destination in X86_PRESERVED_IMPORTS | X86_CALL_CLOBBERED:
            state[destination] = copied
        if mnemonic.startswith(("xchg", "xadd", "mulx", "cmpxchg", "cpuid", "leave", "enter")):
          state.clear()  # implicit or multiple destinations; no partial proof
        load = GOT_LOAD.match(operands)
        if mnemonic == "movq" and load and load.group(1) in X86_PRESERVED_IMPORTS | X86_CALL_CLOBBERED:
          if (value := imports.get(int(load.group(2), 16))) is not None:
            state[load.group(1)] = value
      successors = []
      if mnemonic.startswith("j"):
        destination = target_address(operands)
        if destination in positions:
          successors.append(positions[destination])
      if not mnemonic.startswith(("ret", "jmp", "ud2")) and index + 1 < len(instructions):
        successors.append(index + 1)
      for successor in successors:
        previous = states.get(successor)
        merged = state if previous is None else {key: value for key, value in previous.items() if state.get(key) == value}
        if previous is None or merged != previous:
          states[successor] = dict(merged)
          pending.append(successor)
  return resolved


AARCH64_SP_SUB = re.compile(r"^sp, sp, #(0x[0-9a-f]+|\d+)$")
AARCH64_SP_PREINDEX = re.compile(r"\[sp, #-(0x[0-9a-f]+|\d+)\]!")
X86_STACK_POINTER = re.compile(r"%(?:rsp|esp|spl|sp)\b")
X86_STACK_POINTER_NAMES = {"%rsp", "%esp", "%sp", "%spl"}
# `disp(%rsp)` with an optional index; group 1 is the displacement, group 2 the index part.
X86_RSP_MEMORY = re.compile(r"(-?0x[0-9a-f]+|-?\d+)?\(%rsp(,[^)]*)?\)")
X86_IMMEDIATE = re.compile(r"^\$(-?0x[0-9a-f]+|-?\d+)$")
X86_TERMINATORS = {"jmp", "jmpq", "ret", "retq", "ud2", "hlt"}
AARCH64_TERMINATORS = {"b", "br", "braa", "brab", "braaz", "brabz", "ret", "retaa", "retab", "brk", "udf", "hlt"}
ALIGNMENT_PADDING = re.compile(r"^(?:nop[a-z]*|int3)$")
# objdump prints constant tables in a code section as data directives.
DATA_DIRECTIVE = re.compile(r"(?:^|\s)\.(?:byte|short|hword|word|long|quad|xword)\b")
# Store widths, in bytes, of the moves that the save-slot check understands.
X86_STORE_WIDTHS = {"movq": 8, "movl": 4, "movw": 2, "movb": 1, "movdqa": 16, "movdqu": 16, "movaps": 16, "movups": 16}
X86_VECTOR_MOVES = re.compile(r"^vmov(?:dq[au](?:8|16|32|64)?|ap[sd]|up[sd])$")
X86_VECTOR_WIDTHS = {"zmm": 64, "ymm": 32, "xmm": 16}


@dataclass(frozen=True)
class AssemblyRegion:
  """Frameless assembly symbols that share one stack frame, and that frame's bound."""

  frame: int
  members: frozenset[int]
  note: str


def instruction_rows(function: Function) -> list[tuple[int, str, str]]:
  return [
    (address, mnemonic, operands)
    for address, (mnemonic, operands) in zip(sorted(function.instruction_addresses), function.instructions, strict=True)
  ]


def frameless_assembly(program: Program, function: Function) -> bool:
  """A defined function without a compiler frame record: hand-written assembly."""
  return function.frame is None and external_name(program, function) is None and bool(function.instructions)


def falls_through(program: Program, function: Function) -> bool:
  """Whether execution can run off the end of `function` into the next symbol.

  Alignment padding and data after the final terminator are skipped unless a
  branch targets them.
  """
  rows = instruction_rows(function)
  while rows and rows[-1][0] not in program.branch_targets and (
    ALIGNMENT_PADDING.match(rows[-1][1]) or DATA_DIRECTIVE.search(rows[-1][2])
  ):
    rows.pop()
  if not rows:
    return True
  _, mnemonic, operands = rows[-1]
  if program.target.startswith("x86_64"):
    return not (mnemonic in X86_TERMINATORS or (mnemonic == "rep" and operands.strip().startswith("ret")))
  return mnemonic not in AARCH64_TERMINATORS


def x86_operands(operands: str) -> list[str]:
  """Split AT&T operands at top-level commas, without annotations or comments."""
  text = ANNOTATION.sub("", operands).split("#", 1)[0].strip()
  parts, depth, current = [], 0, ""
  for character in text:
    if character == "," and depth == 0:
      parts.append(current.strip())
      current = ""
      continue
    depth += {"(": 1, ")": -1}.get(character, 0)
    current += character
  if current.strip():
    parts.append(current.strip())
  return parts


def x86_immediate(operand: str) -> int | None:
  match = X86_IMMEDIATE.match(operand)
  return int(match.group(1), 0) if match else None


def x86_stack_effect(mnemonic: str, operands: str, red_zone: int) -> tuple[str, int] | None:
  """Classify one x86-64 instruction's effect on `rsp`; None when it cannot be bounded.

  Kinds: `grow` (a decrement of N bytes), `align` (`and $-N, %rsp`), `restore`
  (`movq N(%rsp), %rsp`), `copy` (the stack address escapes into another
  register), and `none`.
  """
  parts = x86_operands(operands)
  text = ", ".join(parts)
  if mnemonic in {"pushq", "pushfq"}:
    return ("grow", 8)
  if mnemonic in {"popq", "popfq"}:
    return None if parts and parts[-1] in X86_STACK_POINTER_NAMES else ("none", 0)
  if mnemonic.startswith(("push", "pop", "enter", "leave")):
    return None  # other widths and frame-pointer forms are not modeled
  if X86_STACK_POINTER.search(text) is None:
    return ("none", 0)
  if mnemonic.startswith(("xchg", "cmpxchg", "xadd")):
    return None  # writes every operand
  for memory in X86_RSP_MEMORY.finditer(text):
    if memory.group(2):
      return None  # an indexed stack address has no static offset
    if int(memory.group(1) or "0", 0) < -red_zone:
      return None  # access below the stack pointer, beyond any red zone
  destination, source = parts[-1], parts[0]
  if destination in X86_STACK_POINTER_NAMES:
    immediate = x86_immediate(source) if len(parts) == 2 else None
    if mnemonic == "subq" and immediate is not None:
      return ("grow", immediate) if immediate >= 0 else ("none", 0)
    if mnemonic == "addq" and immediate is not None:
      return ("grow", -immediate) if immediate < 0 else ("none", 0)
    if mnemonic == "andq" and immediate is not None and immediate < 0 and (-immediate) & (-immediate - 1) == 0:
      return ("align", -immediate)
    memory = X86_RSP_MEMORY.fullmatch(source) if len(parts) == 2 else None
    if mnemonic == "leaq" and memory is not None:
      displacement = int(memory.group(1) or "0", 0)
      return ("grow", -displacement) if displacement < 0 else ("none", 0)
    if mnemonic == "movq" and memory is not None:
      return ("restore", int(memory.group(1) or "0", 0))
    return None
  if mnemonic.startswith("lea") or any(part in X86_STACK_POINTER_NAMES for part in parts):
    return ("copy", 0)
  return ("none", 0)


def aarch64_stack_effect(mnemonic: str, operands: str) -> tuple[str, int] | None:
  """Classify one AArch64 instruction's effect on `sp`; None when it cannot be bounded.

  Decrements are an immediate `sub sp, sp, #N`, a pre-indexed `[sp, #-N]!`
  access, or a negative post-index. Any other write to `sp` (other than
  `add sp, sp, #N`), and any access below `sp`, makes the frame unknown:
  Linux AArch64 has no red zone.
  """
  text = ANNOTATION.sub("", operands).split("//", 1)[0].strip()
  if mnemonic == "sub" and text.startswith("sp,"):
    match = AARCH64_SP_SUB.match(text)
    return ("grow", int(match.group(1), 0)) if match else None
  if match := AARCH64_SP_PREINDEX.search(text):
    return ("grow", int(match.group(1), 0))
  if "[sp]," in text:
    match = re.search(r"\[sp\], #(-?(?:0x[0-9a-f]+|\d+))$", text)
    if match is None:
      return None
    change = int(match.group(1), 0)
    return ("grow", -change) if change < 0 else ("none", 0)
  if "[sp, #-" in text:
    return None  # access below sp without a recorded stack adjustment
  if text.startswith("sp,") and not (mnemonic == "add" and AARCH64_SP_SUB.fullmatch(text)):
    return None
  return ("none", 0)


def x86_store_width(mnemonic: str, parts: list[str]) -> int | None:
  if mnemonic in X86_STORE_WIDTHS:
    return X86_STORE_WIDTHS[mnemonic]
  if X86_VECTOR_MOVES.match(mnemonic) and parts:
    register = re.match(r"%([xyz]mm)\d+", parts[0])
    return X86_VECTOR_WIDTHS[register.group(1)] if register else None
  return None


def x86_restore_is_verified(
  program: Program, rows: list[tuple[int, str, str]], restore: int, offset: int, sources: dict[int, set[int]]
) -> int | None:
  """Prove that `movq OFFSET(%rsp), %rsp` at row `restore` reloads an earlier `rsp`.

  The accepted idiom is `movq %rsp, %R`, then only immediate `sub`/`and`
  stack decrements, then `movq %R, OFFSET(%rsp)`, where the slot lies inside
  the bytes those subtractions allocate. Nothing may branch into the save
  sequence after its first instruction. Between the slot store and the
  restore: `rsp` does not change, its address does not escape, no call runs,
  no store through `rsp` overlaps the slot, and every branch into that span
  starts inside it. The restored value is then one that `rsp` held on every
  path that reaches the restore. Returns the row of the save, or None. The
  caller also rejects any other escape of the stack address in the region.
  """
  store = next(
    (
      index for index in range(restore - 1, -1, -1)
      if rows[index][1] == "movq" and x86_operands(rows[index][2])[1:] == [f"{offset:#x}(%rsp)" if offset else "(%rsp)"]
    ),
    None,
  )
  general = (set(X86_LEGACY_REGISTERS) - {"%rsp"}) | {f"%r{n}" for n in range(8, 16)}
  if store is None or x86_operands(rows[store][2])[0] not in general:
    return None
  saved = x86_operands(rows[store][2])[0]
  save, allocated = store - 1, 0
  while save >= 0 and x86_operands(rows[save][2]) != ["%rsp", saved]:
    effect = x86_stack_effect(rows[save][1], rows[save][2], 0)
    if rows[save][1] not in {"subq", "andq"} or effect is None or effect[0] not in {"grow", "align"}:
      return None
    allocated += effect[1] if effect[0] == "grow" else 0
    save -= 1
  if save < 0 or rows[save][1] != "movq" or not 0 <= offset <= allocated - 8:
    return None  # the slot must be memory that this save sequence allocated
  if any(rows[index][0] in program.branch_targets for index in range(save + 1, store + 1)):
    return None
  low, high = rows[store][0], rows[restore][0]
  for index in range(store + 1, restore):
    address, mnemonic, operands = rows[index]
    effect = x86_stack_effect(mnemonic, operands, 0)
    if effect is None or effect[0] != "none" or mnemonic.startswith("call"):
      return None
    parts = x86_operands(operands)
    if parts and (memory := X86_RSP_MEMORY.fullmatch(parts[-1])) is not None and len(parts) > 1:
      width = x86_store_width(mnemonic, parts)
      start = int(memory.group(1) or "0", 0)
      if width is None or (start < offset + 8 and offset < start + width):
        return None
  for target in range(low + 1, high + 1):
    if target in sources and any(not low <= source < high for source in sources[target]):
      return None
  return save


def assembly_region(program: Program, function: Function) -> AssemblyRegion | None:
  """Bound the frame of hand-written assembly that has no compiler record.

  The region is `function` and every frameless symbol it reaches by a direct
  jump or by running off its end. ELF-visible labels split one routine into
  such symbols, and a continuation reached by `jmp` still uses the frame of
  the code that jumped. Calls leave the region. The frame is the sum of every
  stack decrement in the region; on x86-64, `and $-N, %rsp` adds `N - 8`
  because the ABI keeps `rsp` 8-byte aligned. The sum bounds the region only
  when no decrement can repeat: a backward branch that spans a decrement, an
  unresolved jump, an unmodeled stack-pointer write, an access below the
  stack pointer beyond the red zone, or an unverified `rsp` reload makes the
  frame unknown. Supported on x86-64 and AArch64.
  """
  arch = program.target.split("-", 1)[0]
  if arch not in {"x86_64", "aarch64"} or not frameless_assembly(program, function):
    return None
  members = {function.address}
  pending = [function]
  while pending:
    current = pending.pop()
    successors = [(edge.target, False) for edge in current.edges if edge.kind == "tail"]
    if falls_through(program, current):
      index = bisect.bisect_right(program.starts, current.address)
      if index >= len(program.starts):
        return None
      successors.append((program.starts[index], True))
    for address, fallthrough in successors:
      if address in members:
        continue
      successor = program.functions[address]
      if not frameless_assembly(program, successor):
        if fallthrough:
          return None  # runs into code whose frame this region cannot own
        continue  # a tail call out of the region; the traversal charges it
      members.add(address)
      pending.append(successor)
  rows = sorted(row for address in members for row in instruction_rows(program.functions[address]))
  sources: dict[int, set[int]] = {}
  frame, decrements, branches, alignments, restores, copies, multiple_of_eight = 0, [], [], [], [], [], True
  for index, (address, mnemonic, operands) in enumerate(rows):
    kind = classify(program.target, mnemonic, operands)
    if kind == "indirect-jump":
      return None
    if kind == "branch":
      destination = target_address(operands)
      if destination is None:
        return None
      branches.append((address, destination))
    if arch == "x86_64":
      effect = x86_stack_effect(mnemonic, operands, ARCHES[program.target].red_zone)
    else:
      effect = aarch64_stack_effect(mnemonic, operands)
    if effect is None:
      return None
    if effect[0] == "grow":
      frame += effect[1]
      multiple_of_eight = multiple_of_eight and effect[1] % 8 == 0
      decrements.append(address)
    elif effect[0] == "align":
      alignments.append(effect[1])
      decrements.append(address)
    elif effect[0] == "restore":
      restores.append((index, effect[1]))
    elif effect[0] == "copy":
      copies.append(index)
  if any(destination <= decrement <= source for source, destination in branches for decrement in decrements):
    return None  # a loop can repeat a stack decrement; static sum is not a bound
  if restores:
    # Every direct branch into the region: its own, and jumps from other symbols.
    for source, destination in branches:
      sources.setdefault(destination, set()).add(source)
    for other in program.functions.values():
      for edge in other.edges:
        if edge.kind == "tail" and edge.target is not None:
          sources.setdefault(edge.target, set()).add(edge.source)
  saves = set()
  for index, offset in restores:
    if (save := x86_restore_is_verified(program, rows, index, offset, sources)) is None:
      return None
    saves.add(save)
  if restores and not set(copies) <= saves:
    return None  # another register holds a stack address that could reach the save slot
  frame += sum(alignment - 8 if multiple_of_eight else alignment - 1 for alignment in alignments)
  name = function.demangled
  leaf = not any(edge.kind in {"call", "import"} for address in members for edge in program.functions[address].edges)
  if len(members) == 1 and leaf and not alignments and not restores:
    note = f"{name}: leaf assembly frame {frame} B"
  else:
    details = [f"{len(members)} symbol(s)"]
    if alignments:
      details.append("rsp 8-byte aligned at entry")
    if restores:
      details.append("rsp reloaded from a verified save slot")
    note = f"{name}: assembly frame {frame} B ({', '.join(details)})"
  return AssemblyRegion(frame, frozenset(members), note)


class ElfImage:
  """The file-backed bytes of an ELF64 executable's loadable segments, by virtual address."""

  def __init__(self, data: bytes) -> None:
    if data[:4] != b"\x7fELF" or data[4] != 2:
      raise ValueError("expected an ELF64 file")
    self.data = data
    self.endian = "<" if data[5] == 1 else ">"
    phoff, = struct.unpack_from(self.endian + "Q", data, 0x20)
    phentsize, phnum = struct.unpack_from(self.endian + "HH", data, 0x36)
    self.segments = []
    for index in range(phnum):
      kind, _, offset, vaddr, _, filesz = struct.unpack_from(self.endian + "IIQQQQ", data, phoff + index * phentsize)
      if kind == 1:  # PT_LOAD
        self.segments.append((vaddr, offset, filesz))

  def int32(self, address: int) -> int | None:
    """The signed 32-bit value stored at `address`, or None outside file-backed memory."""
    for vaddr, offset, filesz in self.segments:
      if vaddr <= address and address + 4 <= vaddr + filesz:
        return struct.unpack_from(self.endian + "i", self.data, offset + address - vaddr)[0]
    return None


# Symbolic values while following a jump-table sequence; i is the checked index, 0 <= i <= bound.
#   ("const", c)                 c
#   ("index", scale, bound)      scale * i
#   ("entry", base, scale, bound) the signed 32-bit table entry at base + scale * i
#   ("target", origin, entry)    origin + entry
JUMP_TABLE_STEPS = 12
RISCV_STORES = {"sd", "sw", "sh", "sb", "fsd", "fsw", "c.sd", "c.sw", "c.sdsp", "c.swsp", "c.fsd", "c.fsdsp"}


def x86_jump_table_step(state: dict, mnemonic: str, operands: str) -> bool:
  """Apply one x86-64 instruction of a jump-table sequence; False if it is not one."""
  parts = x86_operands(operands)
  if mnemonic in {"movq", "movl", "movw", "movb"} and len(parts) == 2 and parts[1].endswith(")"):
    return True  # a store writes no register
  if mnemonic == "leaq" and len(parts) == 2 and parts[0].endswith("(%rip)"):
    annotated = re.search(r"#\s+0x([0-9a-f]+)", operands)
    if annotated is None:
      return False
    state[parts[1]] = ("const", int(annotated.group(1), 16))
    return True
  memory = re.fullmatch(r"\((%\w+),(%\w+),4\)", parts[0]) if mnemonic == "movslq" and len(parts) == 2 else None
  if memory is not None:
    base, index = state.get(memory.group(1)), state.get(memory.group(2))
    if base is None or base[0] != "const" or index is None or index[0] != "index" or index[1] != 1:
      return False
    state[parts[1]] = ("entry", base[1], 4, index[2])
    return True
  if mnemonic == "addq" and len(parts) == 2 and parts[0].startswith("%"):
    return add_symbolic(state, parts[1], state.get(parts[0]), state.get(parts[1]))
  return False


def s390x_jump_table_step(state: dict, mnemonic: str, operands: str) -> bool:
  """Apply one IBM Z instruction of a jump-table sequence; False if it is not one."""
  parts = [part.strip() for part in ANNOTATION.sub("", operands).split(",")]
  if mnemonic in {"stg", "st", "sty", "stc", "sth", "std", "ste", "stmg"}:
    return True
  if mnemonic == "sllg" and len(parts) == 3 and re.fullmatch(r"\d+", parts[2]):
    source = state.get(parts[1])
    if source is None or source[0] != "index":
      return False
    state[parts[0]] = ("index", source[1] << int(parts[2]), source[2])
    return True
  if mnemonic == "larl" and len(parts) == 2 and (address := target_address(operands)) is not None:
    state[parts[0]] = ("const", address)
    return True
  if mnemonic == "agf":
    match = re.fullmatch(r"(%r\d+), 0\((%r\d+),(%r\d+)\)", ANNOTATION.sub("", operands).strip())
    if match is None:
      return False
    destination, first, second = match.groups()
    origin = state.get(destination)
    values = [state.get(first), state.get(second)]
    consts = [value for value in values if value is not None and value[0] == "const"]
    indexes = [value for value in values if value is not None and value[0] == "index"]
    if origin is None or origin[0] != "const" or len(consts) != 1 or len(indexes) != 1:
      return False
    state[destination] = ("target", origin[1], ("entry", consts[0][1], indexes[0][1], indexes[0][2]))
    return True
  return False


def riscv_jump_table_step(state: dict, mnemonic: str, operands: str, address: int) -> bool:
  """Apply one RISC-V instruction of a jump-table sequence; False if it is not one."""
  parts = [part.strip() for part in ANNOTATION.sub("", operands).split(",")]
  if mnemonic in RISCV_STORES:
    return True
  if mnemonic == "auipc" and len(parts) == 2:
    value = int(parts[1], 0) & 0xFFFFF
    value -= (value & 0x80000) << 1
    state[parts[0]] = ("const", (address + (value << 12)) & ((1 << 64) - 1))
    return True
  if mnemonic in {"addi", "c.addi"} and len(parts) == 3:
    source = state.get(parts[1])
    if source is None or source[0] != "const":
      return False
    state[parts[0]] = ("const", (source[1] + int(parts[2], 0)) & ((1 << 64) - 1))
    return True
  if mnemonic in {"slli", "c.slli"} and len(parts) == 3:
    source = state.get(parts[1])
    if source is None or source[0] != "index":
      return False
    state[parts[0]] = ("index", source[1] << int(parts[2], 0), source[2])
    return True
  if mnemonic in {"add", "c.add"} and len(parts) == 3:
    return add_symbolic(state, parts[0], state.get(parts[1]), state.get(parts[2]))
  if mnemonic in {"lw", "c.lw"} and len(parts) == 2:
    match = re.fullmatch(r"0x0\((\w+)\)|0\((\w+)\)", parts[1])
    source = state.get(match.group(1) or match.group(2)) if match else None
    if source is None or source[0] != "address":
      return False
    state[parts[0]] = ("entry", source[1], source[2], source[3])
    return True
  return False


def add_symbolic(state: dict, destination: str, left: tuple | None, right: tuple | None) -> bool:
  """`destination = left + right` for the sums a jump-table sequence forms."""
  if left is None or right is None:
    return False
  pair = {left[0]: left, right[0]: right}
  if set(pair) == {"const", "index"}:
    state[destination] = ("address", pair["const"][1], pair["index"][1], pair["index"][2])
  elif set(pair) == {"const", "entry"}:
    state[destination] = ("target", pair["const"][1], pair["entry"])
  else:
    return False
  return True


def jump_table_bound(target: str, rows: list[tuple[int, str, str]], position: int) -> tuple[str, int] | None:
  """The register and inclusive bound that the unsigned check at `rows[position]` proves on its fall-through."""
  _, mnemonic, operands = rows[position]
  if target.startswith("x86_64") and mnemonic == "ja" and position > 0:
    previous = x86_operands(rows[position - 1][2])
    if rows[position - 1][1] == "cmpq" and len(previous) == 2 and (bound := x86_immediate(previous[0])) is not None:
      return (previous[1], bound) if bound >= 0 else None
  if target.startswith("s390x") and mnemonic == "clgijh":
    parts = [part.strip() for part in ANNOTATION.sub("", operands).split(",")]
    if len(parts) == 3 and re.fullmatch(r"\d+", parts[1]):
      return (parts[0], int(parts[1]))
  if target.startswith("riscv") and mnemonic == "bltu":
    parts = [part.strip() for part in ANNOTATION.sub("", operands).split(",")]
    if len(parts) != 3:
      return None
    # The limit register's last write must be `li`; stores and branches only read their first operand.
    for earlier in range(position - 1, max(position - 4, 0) - 1, -1):
      earlier_mnemonic, earlier_parts = rows[earlier][1], [part.strip() for part in rows[earlier][2].split(",")]
      if earlier_parts[0] != parts[0] or earlier_mnemonic in RISCV_STORES or earlier_mnemonic.startswith(("b", "c.b")):
        continue
      if earlier_mnemonic in {"li", "c.li"} and len(earlier_parts) == 2:
        return (parts[1], int(earlier_parts[1], 0))
      return None
  return None


def jump_table_targets(program: Program, image: ElfImage, function: Function, source: int) -> list[int] | None:
  """Every destination of the register jump at `source`, if it is a checked relative jump table.

  The jump must follow an unsigned bound check of an index register within a
  few instructions. Between the check's fall-through and the jump, every
  instruction must be a step of the table computation or a store, and no
  branch may enter. Each entry for indexes 0 through the bound is read from
  the linked image; the destinations are returned only when every entry is
  readable.
  """
  rows = instruction_rows(function)
  position = next((index for index, row in enumerate(rows) if row[0] == source), None)
  if position is None:
    return None
  for check in range(position - 1, max(position - JUMP_TABLE_STEPS, 0) - 1, -1):
    if (bound := jump_table_bound(program.target, rows, check)) is None:
      continue
    if any(rows[index][0] in program.branch_targets for index in range(check + 1, position + 1)):
      return None
    state: dict = {bound[0]: ("index", 1, bound[1])}
    for address, mnemonic, operands in rows[check + 1:position]:
      if program.target.startswith("x86_64"):
        stepped = x86_jump_table_step(state, mnemonic, operands)
      elif program.target.startswith("s390x"):
        stepped = s390x_jump_table_step(state, mnemonic, operands)
      else:
        stepped = riscv_jump_table_step(state, mnemonic, operands, address)
      if not stepped:
        return None
    _, mnemonic, operands = rows[position]
    if program.target.startswith("x86_64"):
      register = operands.strip().removeprefix("*")
    elif program.target.startswith("s390x"):
      register = operands.strip()
    else:
      register = ANNOTATION.sub("", operands).split(",")[-1].strip()
    value = state.get(register)
    if value is None or value[0] != "target":
      return None
    _, origin, (_, table, scale, limit) = value
    targets = []
    for index in range(limit + 1):
      entry = image.int32(table + scale * index)
      if entry is None:
        return None
      targets.append((origin + entry) & ((1 << 64) - 1))
    return targets
  return None


def resolve_jump_tables(program: Program, image: ElfImage) -> None:
  """Replace register jumps through checked jump tables that stay inside their function.

  Such a jump moves only between instructions of a function whose frame and
  calls the review already counts, so it is no longer an unresolved transfer.
  Every other register jump stays indirect.
  """
  for function in program.functions.values():
    for position, edge in enumerate(function.edges):
      if edge.kind != "indirect" or function.frame is None:
        continue
      targets = jump_table_targets(program, image, function, edge.source)
      if targets is not None and all(target in function.instruction_addresses for target in targets):
        function.edges[position] = Edge("jump-table", edge.source, name=f"{len(targets)} entries")


def parse_dynamic_symbols(text: str) -> dict[int, str | int]:
  """Map each GOT slot with a symbolic dynamic relocation to its import."""
  slots: dict[int, str | int] = {}
  for line in text.splitlines():
    fields = line.split()
    if len(fields) != 3 or re.fullmatch(r"[0-9a-f]+", fields[0]) is None:
      continue
    slot, relocation, value = int(fields[0], 16), fields[1], fields[2]
    if "IRELATIVE" in relocation:
      slots[slot] = f"IFUNC resolver {value} (selected implementation unresolved)"
    elif value.startswith("*ABS*+") and relocation.endswith("_RELATIVE"):
      slots[slot] = int(value.removeprefix("*ABS*+"), 0)
    elif not value.startswith("*ABS*"):
      slots[slot] = value.split("@", 1)[0]
  return slots


def parse_disassembly(
  target: str,
  text: str,
  symbols: dict[int, tuple[str, str]],
  sizes: dict[str, int],
  imports: dict[int, str | int] | None = None,
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

  if target.startswith("s390x"):
    # Some linkers emit one .plt section label, without labels per import.
    # Split only the exact larl/lg/br sequence whose GOT slot has a symbolic
    # relocation. Treating the entire section as one callee connects every
    # import to every caller and loses calloc/free's actual call sites.
    entries = {}
    for index in range(len(rows) - 2):
      owner, address, mnemonic, operands = rows[index]
      if owner.name != ".plt" or mnemonic != "larl" or not operands.startswith("%r1,"):
        continue
      slot = target_address(operands)
      value = imports.get(slot) if imports else None
      following = [(row[2], row[3].strip()) for row in rows[index + 1:index + 3]]
      if isinstance(value, str) and following == [("lg", "%r1, 0(%r1)"), ("br", "%r1")]:
        entries[address] = Function(address, f"{value}@plt", f"{value}@plt")
    functions.update(entries)
    active_entry = None
    for index, (owner, address, mnemonic, operands) in enumerate(rows):
      if owner.name != ".plt":
        active_entry = None
      elif address in entries:
        active_entry = entries[address]
      if active_entry is not None:
        rows[index] = (active_entry, address, mnemonic, operands)

  program = Program(target, functions, sorted(functions))
  x86_calls = x86_import_calls(rows, imports or {}) if target.startswith("x86_64") else {}
  branch_targets = set()
  for function, address, mnemonic, operands in rows:
    function.instruction_addresses.add(address)
    if classify(target, mnemonic, operands) == "branch":
      if (destination := target_address(operands)) is not None:
        branch_targets.add(destination)
  program.branch_targets = branch_targets
  upper: dict[str, int] = {}
  previous: Function | None = None
  for function, address, mnemonic, operands in rows:
    if function is not previous:
      upper = {}
      previous = function
    function.instructions.append((mnemonic, operands))
    if address in branch_targets:
      upper.clear()
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
    if is_riscv:
      upper.clear()  # branch joins and calls end the straight-line auipc proof
    if kind in {"indirect-call", "indirect-jump"}:
      # A call through a GOT slot, directly or through a register loaded from
      # one, names its import in the dynamic relocations.
      slot = GOT_SLOT.search(operands)
      stub = None
      if slot is not None and imports and int(slot.group(1), 16) in imports:
        stub = imports[int(slot.group(1), 16)]
      elif address in x86_calls:
        stub = x86_calls[address]
      if stub is not None:
        if isinstance(stub, int):
          if stub in program.functions:
            function.edges.append(Edge("call" if kind == "indirect-call" else "tail", address, target=stub))
          else:
            function.edges.append(Edge("external", address, target=stub))
        else:
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
class Declared:
  """The deepest entry of one declared edge, below the visited function's entry."""

  kind: str  # libc
  entry: int
  sites: set[tuple[str, str, str]]  # (caller, call instruction address, callee)


@dataclass
class Depth:
  bytes: int
  path: list[str]
  unbounded: list[dict[str, object]]
  detection: list[list[str]]
  panic_exits: set[str]
  assumptions: set[str]
  declared: dict[str, Declared] = field(default_factory=dict)


def declare(depth: Depth, name: str, kind: str, entry: int, sites: set[tuple[str, str, str]]) -> None:
  if (known := depth.declared.get(name)) is None:
    depth.declared[name] = Declared(kind, entry, set(sites))
  else:
    known.entry = max(known.entry, entry)
    known.sites |= sites


def declared_callee(program: Program, callee: Function) -> tuple[str, str] | None:
  """Return the (name, kind) of a declared edge whose entry is this function."""
  if (imported := external_name(program, callee)) in DECLARED_LIBC:
    return imported, "libc"
  return None


def depth_of(
  program: Program, root: Function, *, assume_memory_leaves: bool = True, declared_edges: bool = False
) -> Depth:
  """Return the deepest stack use below `root`'s entry and every unresolved path.

  With `declared_edges`, traversal stops at each declared libc import. Such an
  edge charges only its caller's frame and call slot, and its entry depth is
  recorded for a separately measured allowance. Everything else is accounted
  for exactly as without it.
  """
  if declared_edges and assume_memory_leaves:
    raise ValueError("declared edges replace the external-leaf assumption")
  arch = ARCHES[program.target]
  memo: dict[int, Depth] = {}
  active: list[int] = []

  def visit(function: Function) -> Depth:
    if function.address in memo:
      return memo[function.address]
    if function.address in active:
      cycle = [program.functions[address].demangled for address in active[active.index(function.address) :]]
      return Depth(0, [function.demangled], [{"reason": "recursion", "path": cycle, "address": hex(function.address)}], [], set(), set())
    active.append(function.address)
    name = function.demangled
    imported = external_name(program, function)
    if assume_memory_leaves and imported in EXTERNAL_LEAVES:
      result = Depth(arch.red_zone, [name], [], [], set(), {f"{imported}: external leaf, frame 0 + red zone"})
    elif imported is not None:
      result = Depth(0, [name], [{"reason": "external", "path": [name, imported], "address": hex(function.address)}], [], set(), set())
    else:
      frame = function.frame
      assumptions: set[str] = set()
      region = assembly_region(program, function) if frame is None else None
      edges = function.edges
      # Tail edges of compiled code follow the epilogue, so they enter at this
      # function's entry depth. An assembly jump may keep its frame allocated.
      tail_base = 0
      if region is not None:
        frame = region.frame
        assumptions.add(region.note)
        tail_base = frame
        edges = [
          edge
          for address in sorted(region.members)
          for edge in program.functions[address].edges
          if not (edge.kind == "tail" and edge.target in region.members)
        ]
      missing = [] if frame is not None else [{"reason": "no frame record", "path": [name], "address": hex(function.address)}]
      # Keep walking when this frame is unknown. Otherwise its allocator
      # callbacks, indirect calls, and cleanup edges disappear from the report.
      # The numeric sum is NOT a bound whenever any path remains unresolved.
      known_frame = frame if frame is not None else 0
      result = Depth(known_frame, [name], missing, [], set(), assumptions)
      calls = [edge for edge in edges if edge.kind in {"call", "import"}]
      if not calls or region is not None:
        # Assembly may use the red zone between its calls.
        result.bytes = known_frame + arch.red_zone
      for edge in edges:
        stop = None  # (callee, declared edge name, kind)
        if declared_edges and edge.kind in {"import", "import-tail"} and edge.name in DECLARED_LIBC:
          stop = (str(edge.name), str(edge.name), "libc")
        elif declared_edges and edge.kind in {"call", "tail"}:
          if (declared := declared_callee(program, program.functions[edge.target])) is not None:
            stop = (program.functions[edge.target].demangled, *declared)
        if stop is not None:
          # Charge only the caller's frame and call slot. The callee's own use,
          # its red zone and callbacks included, belongs to the edge's allowance.
          callee_name, label, kind = stop
          base = known_frame + arch.call_slot if edge.kind in {"call", "import"} else tail_base
          declare(result, label, kind, base, {(name, hex(edge.source), callee_name)})
          if base > result.bytes:
            result.bytes, result.path = base, [name, callee_name]
          continue
        if edge.kind in {"import", "import-tail"}:
          if assume_memory_leaves and edge.name in EXTERNAL_LEAVES:
            result.assumptions.add(f"{edge.name}: external leaf, frame 0 + red zone")
            base = known_frame + arch.call_slot if edge.kind == "import" else tail_base
            result.bytes = max(result.bytes, base + arch.red_zone)
          else:
            result.unbounded.append({"reason": "external", "path": [name, str(edge.name)], "address": hex(edge.source)})
          if DETECTION.search(str(edge.name)):
            result.detection.append([name, str(edge.name)])
          continue
        if edge.kind == "jump-table":
          result.assumptions.add(f"{name}: jump table at {edge.source:#x}, {edge.name} inside the function")
          continue
        if edge.kind in {"indirect", "external"}:
          label = edge.name or (f"{edge.target:#x}" if edge.target is not None else "?")
          result.unbounded.append({"reason": edge.kind, "path": [name, label], "address": hex(edge.source)})
          continue
        callee = program.functions[edge.target]
        if PANIC_EXIT.search(callee.demangled):
          result.panic_exits.add(callee.demangled)
          continue
        child = visit(callee)
        base = known_frame + arch.call_slot if edge.kind == "call" else tail_base
        result.unbounded.extend({**item, "path": [name, *item["path"]]} for item in child.unbounded)
        result.detection.extend([name, *path] for path in child.detection)
        result.panic_exits |= child.panic_exits
        result.assumptions |= child.assumptions
        for label, entry in child.declared.items():
          declare(result, label, entry.kind, base + entry.entry, entry.sites)
        if DETECTION.search(callee.demangled):
          result.detection.append([name, callee.demangled])
        if base + child.bytes > result.bytes:
          result.bytes = base + child.bytes
          result.path = [name, *child.path]
    # Keep one shortest witness to each unresolved edge. The retained graph
    # contains every edge, so this does not discard alternate callers. Without
    # deduplication a shared subgraph expands into thousands of
    # repeated paths and obscures the actual blockers.
    witnesses = {}
    for item in result.unbounded:
      identity = (item["reason"], item.get("address"), item["path"][-1])
      if identity not in witnesses or len(item["path"]) < len(witnesses[identity]["path"]):
        witnesses[identity] = item
    result.unbounded = list(witnesses.values())
    result.detection = list({tuple(path): path for path in result.detection}.values())
    active.pop()
    memo[function.address] = result
    return result

  return visit(root)


def scrub_bound(boundary: Boundary) -> int:
  match = re.search(rf"const {boundary.constant}: usize = (\d+);", boundary.source.read_text())
  if match is None:
    raise ValueError(f"{boundary.constant} is missing from {boundary.source}")
  return int(match.group(1)) * 8


def review(program: Program, boundaries: list[Boundary] = BOUNDARIES, portable: bool = False) -> dict[str, object]:
  arch = ARCHES[program.target]
  results = []
  for boundary in boundaries:
    if boundary.targets is not None and program.target not in boundary.targets:
      continue
    if portable and not boundary.portable:
      continue
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
      unscrubbed = []
      tail_scrubbed = []
      worker_problems = []
      for function in caller_functions:
        if scrub is not None and any(edge.kind == "call" and edge.target == scrub.address for edge in function.edges):
          continue
        if scrub is not None and any(edge.kind == "tail" and edge.target == scrub.address for edge in function.edges):
          # A tail branch runs the scrub after this caller releases its frame, so the scrub's
          # buffer starts at the caller's entry: the caller frame and the worker must both fit.
          if function.frame is None:
            worker_problems.append(f"tail-scrubbed caller without a frame record: {function.demangled}")
          elif function.frame + total > bound:
            worker_problems.append(
              f"tail-scrubbed caller {function.demangled}: frame {function.frame} + depth {total} exceeds the {bound}-byte scrub"
            )
          else:
            tail_scrubbed.append(function.demangled)
          continue
        unscrubbed.append(function.demangled)
      unscrubbed.sort()
      tail_scrubbed.sort()
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
          "tail_scrubbed_callers": tail_scrubbed,
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


PREPARATION_ROOTS = [
  f"rscrypto_frame_review::prepare_{parameter}_{form}"
  for parameter in (44, 65, 87)
  for form in ("in", "borrowed", "owned_lifecycle", "borrowed_lifecycle")
]
# Whole-call root groups: each `tools/frame-review` boundary wraps one public operation.
WHOLE_CALL_GROUPS = {
  "mldsa-preparation": PREPARATION_ROOTS,
  "mldsa-signing": [
    f"rscrypto_frame_review::{name}"
    for parameter in (44, 65, 87)
    for name in (f"sign_{parameter}_deterministic", f"sign_{parameter}_prepared", f"verify_{parameter}")
  ],
  "mlkem": [
    f"rscrypto_frame_review::mlkem_{parameter}_{operation}"
    for parameter in (512, 768, 1024)
    for operation in ("generate", "encapsulate", "decapsulate", "prepared_decapsulate")
  ],
  "aead-records": [
    f"rscrypto_frame_review::{operation}_{cipher}"
    for cipher in ("aes128gcm", "aes256gcm", "chacha20poly1305")
    for operation in ("seal", "open")
  ],
}


def whole_call_roots(groups: str) -> list[str]:
  """Return the roots of comma-separated `groups`, or of every group for `all`."""
  names = list(WHOLE_CALL_GROUPS) if groups == "all" else groups.split(",")
  unknown = [name for name in names if name not in WHOLE_CALL_GROUPS]
  if unknown or not names:
    raise ValueError(f"unknown whole-call group(s) {unknown}; choose from {', '.join(WHOLE_CALL_GROUPS)} or all")
  return [root for name in dict.fromkeys(names) for root in WHOLE_CALL_GROUPS[name]]


STACK_CAP = 8 * 1024 * 1024


def load_allowances(path: Path) -> dict[str, dict[str, object]]:
  """Read measured declared-edge allowances: target -> {libc, edges: {edge: bytes}}.

  Each allowance is the stack a declared edge may use below the stack pointer
  at its entry, red zone included, on the named deployed libc. The review only
  composes these numbers; it cannot check how they were measured.
  """
  data = json.loads(path.read_text())
  if not isinstance(data, dict):
    raise SystemExit(f"{path}: expected an object keyed by target")
  names = DECLARED_LIBC
  for target, entry in data.items():
    if target not in ARCHES:
      raise SystemExit(f"{path}: unsupported target {target}")
    if not isinstance(entry, dict) or set(entry) - {"libc", "edges", "evidence"} or not {"libc", "edges"} <= set(entry):
      raise SystemExit(f"{path}: {target} needs exactly libc, edges, and optional evidence")
    if not isinstance(entry["libc"], str) or not entry["libc"].strip():
      raise SystemExit(f"{path}: {target} needs the deployed libc identity")
    if not isinstance(entry["edges"], dict):
      raise SystemExit(f"{path}: {target} edges must map declared edge names to bytes")
    for name, value in entry["edges"].items():
      if name not in names:
        raise SystemExit(f"{path}: {target} names {name!r}, which is not a declared edge")
      if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SystemExit(f"{path}: {target} allowance for {name} must be a non-negative integer")
  return data


def review_preparation(
  program: Program, roots: list[str] = PREPARATION_ROOTS, allowances: dict[str, object] | None = None
) -> dict[str, object]:
  """Whole-call accounting; no imported routine is silently assumed frameless.

  Each root has two traversals. The whole-call traversal descends everywhere
  and is a bound only when nothing is unresolved. The owned traversal stops at
  declared edges; its bound covers the artifact's own code, and the whole-call
  bound composes it with a measured allowance for every declared edge reached.
  """
  arch = ARCHES[program.target]
  edges = allowances["edges"] if allowances is not None else {}
  reports = []
  for name in roots:
    matches = program.named(re.compile(rf"^{re.escape(name)}$"))
    if len(matches) != 1:
      reports.append({"root": name, "whole_call_bound_bytes": None, "owned_bound_bytes": None, "passed": False,
                      "problems": [f"expected one linked root, found {len(matches)}"]})
      continue
    root = matches[0]
    depth = depth_of(program, root, assume_memory_leaves=False)
    owned = depth_of(program, root, assume_memory_leaves=False, declared_edges=True)
    recorded = depth.bytes + arch.call_slot
    owned_recorded = owned.bytes + arch.call_slot
    owned_bound = None if owned.unbounded else owned_recorded
    declared = []
    for label, entry in sorted(owned.declared.items()):
      depth_at_entry = entry.entry + arch.call_slot
      declared.append({
        "name": label, "kind": entry.kind,
        # An entry depth is a bound only when every path in the owned graph is.
        "max_entry_depth_bytes": depth_at_entry if owned_bound is not None else None,
        "recorded_entry_depth_bytes": depth_at_entry,
        "allowance_bytes": edges.get(label),
        "call_sites": [{"caller": caller, "address": address, "callee": callee}
                       for caller, address, callee in sorted(entry.sites)],
      })
    missing = [edge["name"] for edge in declared if edge["allowance_bytes"] is None]
    if not depth.unbounded:
      bound, basis = recorded, "static: every frame and edge resolved"
    elif owned_bound is not None and not missing:
      bound = max([owned_bound, *(edge["max_entry_depth_bytes"] + edge["allowance_bytes"] for edge in declared)])
      basis = "owned bound composed with measured declared-edge allowances"
    else:
      bound, basis = None, None
    problems = []
    if bound is None:
      if owned_bound is None:
        problems.append("unresolved owned call paths; the owned frame sum is not a bound")
      if missing:
        problems.append(f"declared edges without a measured allowance: {', '.join(missing)}")
    elif bound > STACK_CAP:
      problems.append("exceeds the 8 MiB worker cap")
    reports.append({
      "root": name, "address": hex(root.address), "frame_bytes": root.frame,
      "whole_call_bound_bytes": bound,
      "whole_call_bound_basis": basis,
      "owned_bound_bytes": owned_bound,
      "owned_recorded_path_bytes": owned_recorded,
      "owned_path": owned.path,
      "owned_unresolved": owned.unbounded,
      "declared_edges": declared,
      "missing_allowances": missing,
      "recorded_frame_path_bytes": recorded,
      "recorded_frame_path": depth.path,
      "unresolved": depth.unbounded, "capability_detection": depth.detection,
      "panic_exits_abort": sorted(depth.panic_exits), "assumptions": sorted(depth.assumptions),
      "passed": bound is not None and bound <= STACK_CAP,
      "problems": problems,
    })
  return {
    "target": program.target, "whole_calls": reports, "stack_cap_bytes": STACK_CAP,
    "declared_libc_edges": sorted(DECLARED_LIBC),
    "allowances": allowances,
    "scope": "Linked public-operation wrappers (ML-DSA preparation including owned/borrowed destruction, ML-DSA signing and verification, ML-KEM, AEAD record seal and open); not enclosing TLS, worker, signal-handler or arbitrary Allocator callbacks. All syntactically reachable return/error paths are included; abort paths are listed separately. Unknown frames never constitute a bound. Public storage supplied by the caller is outside the borrowed call's frame. Owned bounds cover the artifact's own code and charge each declared libc edge only its caller's frame and call slot; they are not whole-call bounds until every reached declared edge has a measured allowance for the deployed libc.",
    "passed": all(item["passed"] for item in reports),
  }


ROOT_PREFIX = "rscrypto_frame_review::"
CONFIGURATIONS = ("native", "portable")


def stack_budget(owned: int) -> int:
  """A root's budget: its owned bound plus 12.5%, rounded up to a whole KiB, and at least 1 KiB."""
  return max(1024, -(-owned * 9 // (8 * 1024)) * 1024)


def load_budgets(path: Path) -> dict[str, dict[str, dict[str, int]]]:
  """Read per-root owned-bound budgets: {"targets": {target: {configuration: {root: bytes}}}}."""
  data = json.loads(path.read_text())
  targets = data.get("targets") if isinstance(data, dict) else None
  if not isinstance(targets, dict):
    raise SystemExit(f"{path}: expected an object with a `targets` object")
  for target, configurations in targets.items():
    if target not in ARCHES or not isinstance(configurations, dict) or not set(configurations) <= set(CONFIGURATIONS):
      raise SystemExit(f"{path}: {target!r} must be a reviewed target with native and/or portable budgets")
    for configuration, roots in configurations.items():
      if not isinstance(roots, dict) or not all(
        isinstance(name, str) and isinstance(value, int) and not isinstance(value, bool) and value > 0
        for name, value in roots.items()
      ):
        raise SystemExit(f"{path}: {target} {configuration} budgets must map root names to positive integers")
  return targets


def apply_budgets(
  report: dict[str, object], budgets: dict, configuration: str, complete: bool, whole_calls: bool = False
) -> list[str]:
  """Record each root's budget in `report` and return the budget problems.

  A root fails when its owned bound is unresolved, it has no budget, or the
  bound exceeds the budget. With `complete`, a budget for a root that the
  review no longer has is also a problem. With `whole_calls` (allowances were
  supplied), a root also fails when it has no whole-call bound or the bound
  exceeds the 8 MiB worker cap.
  """
  expected = budgets.get(str(report["target"]), {}).get(configuration, {})
  problems = []
  reviewed = set()
  for call in report["whole_calls"]:
    name = str(call["root"]).removeprefix(ROOT_PREFIX)
    reviewed.add(name)
    budget = expected.get(name)
    owned = call["owned_bound_bytes"]
    call["owned_budget_bytes"] = budget
    call["within_owned_budget"] = budget is not None and owned is not None and owned <= budget
    if owned is None:
      problems.append(f"{name}: owned bound unresolved")
    elif budget is None:
      problems.append(f"{name}: no budget")
    elif owned > budget:
      problems.append(f"{name}: owned bound {owned:,} B exceeds its {budget:,} B budget")
    whole = call.get("whole_call_bound_bytes")
    if whole_calls and owned is not None and whole is None:
      missing = ", ".join(call.get("missing_allowances", [])) or "unresolved"
      problems.append(f"{name}: no whole-call bound; declared edges without an allowance: {missing}")
    elif whole_calls and whole is not None and whole > STACK_CAP:
      problems.append(f"{name}: whole-call bound {whole:,} B exceeds the 8 MiB worker cap")
  if complete:
    problems.extend(f"{name}: budget for a root the review no longer has" for name in sorted(set(expected) - reviewed))
  report["budget_configuration"] = configuration
  report["budget_problems"] = problems
  return problems


def updated_budgets(existing: dict, reports: list[dict[str, object]], complete: bool) -> dict:
  """Budgets from these reports' owned bounds, merged over `existing`; refuses any unresolved bound."""
  targets = json.loads(json.dumps(existing))
  for report in reports:
    configuration = "portable" if report["portable"] else "native"
    roots = {} if complete else dict(targets.get(str(report["target"]), {}).get(configuration, {}))
    for call in report["whole_calls"]:
      owned = call["owned_bound_bytes"]
      if owned is None:
        raise SystemExit(f"{report['target']} {configuration} {call['root']}: no owned bound to budget")
      roots[str(call["root"]).removeprefix(ROOT_PREFIX)] = stack_budget(int(owned))
    targets.setdefault(str(report["target"]), {})[configuration] = dict(sorted(roots.items()))
  return {target: dict(sorted(targets[target].items())) for target in sorted(targets)}


def load(target: str, binary: Path, evidence: Path | None = None) -> Program:
  mangled = run("llvm-nm", "--defined-only", "-n", str(binary))
  demangled = run("llvm-nm", "--defined-only", "-n", "-C", str(binary))
  symbols = parse_symbols(mangled, demangled)
  stack_sizes = run("llvm-readobj", "--stack-sizes", str(binary))
  sizes = parse_stack_sizes(stack_sizes)
  if not sizes:
    raise SystemExit(f"{binary} has no .stack_sizes records; build it with -Z emit-stack-sizes")
  disassembly = run("llvm-objdump", "-d", "--no-show-raw-insn", str(binary))
  relocations = run("llvm-objdump", "-R", str(binary))
  imports = parse_dynamic_symbols(relocations)
  program = parse_disassembly(target, disassembly, symbols, sizes, imports)
  resolve_jump_tables(program, ElfImage(binary.read_bytes()))
  if evidence is not None:
    for name, content in {"symbols.txt": mangled, "symbols-demangled.txt": demangled,
                          "stack-sizes.txt": stack_sizes, "disassembly.txt": disassembly,
                          "dynamic-relocations.txt": relocations}.items():
      (evidence / name).write_text(content)
    graph = [{"address": hex(function.address), "name": function.demangled, "frame_bytes": function.frame,
              "edges": [{"kind": edge.kind, "source": hex(edge.source),
                         "target": hex(edge.target) if edge.target is not None else None, "name": edge.name}
                        for edge in function.edges]}
             for function in program.functions.values()]
    (evidence / "call-graph.json").write_text(json.dumps(graph, indent=2) + "\n")
  return program


def cross_linker(target: str) -> str:
  """The pinned Zig when this host has it, otherwise the target's GNU cross compiler."""
  if subprocess.run([str(ROOT / "scripts" / "tooling" / "zig.sh")], capture_output=True).returncode == 0:
    return str(ROOT / "scripts" / "ct" / "zig-cc.sh")
  sys.path.insert(0, str(ROOT / "scripts" / "lib"))
  import cross_build

  compiler = f"{cross_build.LINUX_TARGETS[target][1]}-gcc"
  if shutil.which(compiler) is None:
    raise SystemExit(f"{target} needs the pinned Zig or {compiler} to link")
  return compiler


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
  if target != host and f"CARGO_TARGET_{key}_LINKER" not in environment:
    environment[f"CARGO_TARGET_{key}_LINKER"] = cross_linker(target)
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
  if "whole_calls" in report:
    allowances = report.get("allowances")
    lines.append(f"Declared-edge allowances: {allowances['libc'] if allowances else 'none supplied'}")
    lines.append("")

  def size(value: int | None) -> str:
    return f"{value} B" if value is not None else "UNRESOLVED"

  for call in report.get("whole_calls", []):
    budget = f" (budget {size(call['owned_budget_bytes'])})" if "owned_budget_bytes" in call else ""
    lines.append(f"- `{call['root']}`: whole-call bound {size(call['whole_call_bound_bytes'])}, "
                 f"owned bound {size(call['owned_bound_bytes'])}{budget}")
    if "recorded_frame_path_bytes" in call:
      lines.append(f"  - recorded frame path sum: {call['recorded_frame_path_bytes']} B (not a bound when unresolved)")
    for edge in call.get("declared_edges", []):
      allowance = edge["allowance_bytes"]
      lines.append(
        f"  - declared {edge['kind']} `{edge['name']}`: entry depth {size(edge['max_entry_depth_bytes'])}"
        f" (recorded {edge['recorded_entry_depth_bytes']} B), {len(edge['call_sites'])} call site(s), "
        f"allowance {str(allowance) + ' B' if allowance is not None else 'not measured'}"
      )
    for item in call.get("owned_unresolved", []):
      lines.append(f"  - owned {item['reason']} at {item.get('address', '?')}: {' -> '.join(item['path'])}")
    if call.get("unresolved"):
      lines.append(f"  - whole-call unresolved edges, declared subgraphs included: {len(call['unresolved'])} (see report.json)")
    lines.extend(f"  - problem: {problem}" for problem in call["problems"])
  for boundary in report.get("boundaries", []):
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


def source_identity() -> dict[str, object]:
  names = subprocess.check_output(
    ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--",
     "Cargo.toml", "Cargo.lock", "build.rs", "rust-toolchain.toml", ".cargo", ".config",
     "src", "tools/frame-review", "scripts/stack", "scripts/ct/zig-cc.sh"], cwd=ROOT
  ).decode().split("\0")
  return {
    "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    "files": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() if (ROOT / name).is_file() else None
              for name in sorted(set(names) - {""})},
  }


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--target", action="append", help="target to build and review; repeatable (default: all)")
  parser.add_argument("--portable", action="store_true", help="build with the portable-only feature")
  parser.add_argument("--rustflags", default="", help="extra rustc flags for the build, space-separated")
  parser.add_argument("--binary", type=Path, help="review an existing binary (requires one --target)")
  parser.add_argument("--json", type=Path, help="write the full report as JSON")
  parser.add_argument("--mldsa-preparation", action="store_true", help="shorthand for --whole-calls mldsa-preparation")
  parser.add_argument("--whole-calls", metavar="GROUPS", help=f"review whole public calls instead of scrub boundaries; comma-separated groups from {', '.join(WHOLE_CALL_GROUPS)}, or all; unresolved paths fail")
  parser.add_argument("--evidence", type=Path, help="new directory retaining binaries, disassembly, call graphs, source identity and reports")
  parser.add_argument("--allowances", type=Path, help="JSON of measured declared-edge allowances per target and deployed libc (with --whole-calls)")
  parser.add_argument("--all-configurations", action="store_true", help="review both the native and the --portable build of each target")
  parser.add_argument("--budgets", type=Path, help="JSON of per-root owned-bound budgets (with --whole-calls); the budgets alone decide the exit status")
  parser.add_argument("--write-budgets", type=Path, help="write budgets from these owned bounds, 12.5%% above each and rounded up to 1 KiB, merged into the file")
  args = parser.parse_args(argv)
  if args.mldsa_preparation and args.whole_calls is not None:
    parser.error("--mldsa-preparation is shorthand for --whole-calls mldsa-preparation; pass one")
  groups = "mldsa-preparation" if args.mldsa_preparation else args.whole_calls
  try:
    roots = whole_call_roots(groups) if groups is not None else None
  except ValueError as error:
    parser.error(str(error))

  targets = args.target or list(ARCHES)
  if args.binary is not None and len(targets) != 1:
    parser.error("--binary requires exactly one --target")
  if args.allowances is not None and roots is None:
    parser.error("--allowances applies only to --whole-calls")
  if (args.budgets is not None or args.write_budgets is not None) and roots is None:
    parser.error("--budgets and --write-budgets apply only to --whole-calls")
  if args.all_configurations and (args.portable or args.binary is not None):
    parser.error("--all-configurations builds both configurations; omit --portable and --binary")
  configurations = list(CONFIGURATIONS) if args.all_configurations else ["portable" if args.portable else "native"]
  budgets = load_budgets(args.budgets) if args.budgets is not None else None
  allowances = load_allowances(args.allowances) if args.allowances is not None else {}
  allowance_sha256 = hashlib.sha256(args.allowances.read_bytes()).hexdigest() if args.allowances is not None else None
  if args.evidence is not None:
    args.evidence.mkdir(parents=True, exist_ok=False)
  source = source_identity()
  reports = []
  for target, configuration in ((target, configuration) for target in targets for configuration in configurations):
    portable = configuration == "portable"
    binary = args.binary or build(target, portable, args.rustflags.split())
    evidence = args.evidence / target if args.evidence is not None else None
    if evidence is not None and len(configurations) > 1:
      evidence = args.evidence / f"{target}-{configuration}"
    if evidence is not None:
      evidence.mkdir()
      shutil.copy2(binary, evidence / binary.name)
      binary = evidence / binary.name
    program = load(target, binary, evidence)
    if roots is not None:
      report = review_preparation(program, roots, allowances=allowances.get(target))
      report["whole_call_groups"] = groups
      report["allowances_file"] = (
        {"path": str(args.allowances), "sha256": allowance_sha256} if args.allowances is not None else None
      )
      if evidence is not None and args.allowances is not None:
        shutil.copy2(args.allowances, evidence / "allowances.json")
    else:
      report = review(program, portable=portable)
    if budgets is not None:
      apply_budgets(report, budgets, configuration, complete=groups == "all", whole_calls=args.allowances is not None)
    report["binary"] = str(binary)
    report["portable"] = portable
    report["rustflags"] = args.rustflags
    report["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
    report["rustc"] = subprocess.check_output(["rustc", "-Vv"], text=True).strip()
    report["source"] = source
    report["source_unchanged"] = source == source_identity()
    report["host"] = platform.platform()
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    report["command"] = sys.argv if argv is None else [sys.argv[0], *argv]
    report["build_flags"] = ["-Z", "emit-stack-sizes", *args.rustflags.split()] if args.binary is None else None
    report["build_provenance"] = "built by this invocation" if args.binary is None else "caller-supplied artifact; build/source configuration requires separate evidence"
    if not report["source_unchanged"]:
      report["passed"] = False
    reports.append(report)
    print(markdown(report))
    if evidence is not None:
      (evidence / "report.json").write_text(json.dumps(report, indent=2) + "\n")
      (evidence / "report.md").write_text(markdown(report))
      (evidence / "SHA256SUMS").write_text("".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
        for path in sorted(evidence.iterdir()) if path.is_file() and path.name != "SHA256SUMS"
      ))
  if args.json is not None:
    args.json.write_text(json.dumps(reports, indent=2) + "\n")
  if args.write_budgets is not None:
    existing = load_budgets(args.write_budgets) if args.write_budgets.exists() else {}
    document = {
      "rule": "Owned stack bound of each `--whole-calls all` root, 12.5% above the reviewed bound and rounded up to "
              "1 KiB. Regenerate with `just stack-budgets` after a reviewed stack change.",
      "targets": updated_budgets(existing, reports, complete=groups == "all"),
    }
    args.write_budgets.write_text(json.dumps(document, indent=2) + "\n")
    if budgets is None:
      # Every reviewed root had an owned bound, or writing would have stopped.
      return 0 if all(report["source_unchanged"] for report in reports) else 1
  if budgets is not None:
    failures = [(report["target"], report["budget_configuration"], problem)
                for report in reports for problem in report["budget_problems"]]
    for target, configuration, problem in failures:
      print(f"budget failure: {target} {configuration}: {problem}")
    if not failures:
      print(f"Every owned bound is within its budget: {len(reports)} configuration(s).")
    return 0 if not failures and all(report["source_unchanged"] for report in reports) else 1
  return 0 if all(report["passed"] for report in reports) else 1


if __name__ == "__main__":
  sys.exit(main())
