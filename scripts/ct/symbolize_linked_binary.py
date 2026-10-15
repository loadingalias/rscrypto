#!/usr/bin/env -S uv run python
"""Reconstruct named final-binary disassembly from linker and object evidence."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Symbol:
  address: int
  size: int
  name: str


def is_local_label(name: str) -> bool:
  return name.startswith((".", "$", "L", "ltmp"))


def parse_nm_line(line: str) -> tuple[int, str, str] | None:
  match = re.match(r"^([0-9a-fA-F]+)\s+(\S)\s+(.+)$", line.strip())
  if match is None:
    return None
  return int(match.group(1), 16), match.group(2), match.group(3)


def demangle_table(artifact_dir: Path, *, text_only: bool = False) -> dict[str, str]:
  names: dict[str, str] = {}
  for raw_path in sorted(artifact_dir.glob("*.o.raw-symbols.txt")) + sorted(
    artifact_dir.glob("*.obj.raw-symbols.txt")
  ):
    demangled_path = raw_path.with_name(raw_path.name.replace(".raw-symbols.txt", ".symbols.txt"))
    if not demangled_path.is_file():
      raise ValueError(f"demangled object symbol map missing for {raw_path.name}")
    raw_rows = [parse_nm_line(line) for line in raw_path.read_text(errors="replace").splitlines()]
    demangled_rows = [parse_nm_line(line) for line in demangled_path.read_text(errors="replace").splitlines()]
    if len(raw_rows) != len(demangled_rows):
      raise ValueError(f"raw and demangled symbol maps differ in length for {raw_path.name}")
    for raw, demangled in zip(raw_rows, demangled_rows, strict=True):
      if raw is None and demangled is None:
        continue
      if raw is None or demangled is None or raw[:2] != demangled[:2]:
        raise ValueError(f"raw and demangled symbol maps are inconsistent for {raw_path.name}")
      if not text_only or raw[1] in {"T", "t"}:
        names[raw[2]] = demangled[2]
  return names


def section_symbol(section: str, known_names: dict[str, str]) -> str | None:
  if not section.startswith(".text."):
    return None
  name = section.removeprefix(".text.")
  for prefix in ("unlikely.", "hot.", "cold."):
    if name.startswith(prefix):
      name = name.removeprefix(prefix)
      break
  if name.startswith("ct_entry_") or name in known_names or name.startswith(("_R", "_ZN")):
    return name
  return None


def parse_msvc_link_map(
  text: str, known_names: dict[str, str], object_text_names: dict[str, str],
) -> list[Symbol]:
  groups: list[tuple[int, int, int]] = []
  functions: list[tuple[int, int, int, str]] = []
  import_thunks: list[Symbol] = []
  section_bases: dict[int, int] = {}
  group_row = re.compile(r"^\s*([0-9a-fA-F]+):([0-9a-fA-F]+)\s+([0-9a-fA-F]+)H\s+\S+\s+CODE\s*$")
  function_row = re.compile(
    r"^\s*([0-9a-fA-F]+):([0-9a-fA-F]+)\s+(\S+)\s+([0-9a-fA-F]+)(?:\s+(.*))?$"
  )
  rows = text.splitlines()
  has_function_flags = any(
    match is not None and re.match(r"f(?:\s|$)", match.group(5) or "")
    for line in rows if (match := function_row.match(line)) is not None
  )
  for line in rows:
    if match := group_row.match(line):
      section, start, size = (int(value, 16) for value in match.groups())
      groups.append((section, start, start + size))
    elif match := function_row.match(line):
      section, offset = (int(match.group(index), 16) for index in (1, 2))
      if is_local_label(match.group(3)):
        continue
      # MSVC can omit `f` on assembly functions even when Rust functions have
      # it. Require matching object text evidence for that exception; arbitrary
      # CODE labels and object data symbols must not split function bodies.
      # LLD can omit flags throughout the map, leaving CODE groups as authority.
      if (has_function_flags and re.match(r"f(?:\s|$)", match.group(5) or "") is None
          and match.group(3) not in object_text_names):
        continue
      in_code = any(group[0] == section and group[1] <= offset < group[2] for group in groups)
      # LLD places DLL import thunks after the listed CODE groups. Bind their
      # exact entry addresses, as for Mach-O stubs, without inventing a length.
      is_import_thunk = (
        not in_code and any(group[0] == section for group in groups)
        and (match.group(5) or "").rstrip().lower().endswith(".dll")
      )
      if not in_code and not is_import_thunk:
        continue
      address = int(match.group(4), 16)
      base = address - offset
      if base < 0 or section_bases.setdefault(section, base) != base:
        raise ValueError("inconsistent MSVC linker-map section address")
      if is_import_thunk:
        import_thunks.append(Symbol(address, 1, known_names.get(match.group(3), match.group(3))))
      else:
        functions.append((section, offset, address, match.group(3)))

  # Maps give starts, not function lengths. Bound each function by the next
  # distinct start in its CODE group, or by that group's end. Keep ICF aliases.
  offsets = {
    group: sorted({offset for section, offset, _, _ in functions
                   if section == group[0] and group[1] <= offset < group[2]})
    for group in groups
  }
  symbols = import_thunks
  for section, offset, address, name in functions:
    containing = [group for group in groups if group[0] == section and group[1] <= offset < group[2]]
    if len(containing) != 1:
      raise ValueError(f"MSVC function {name} is not in exactly one CODE group")
    group = containing[0]
    starts = offsets[group]
    next_index = bisect.bisect_right(starts, offset)
    end = starts[next_index] if next_index < len(starts) else group[2]
    symbols.append(Symbol(address, end - offset, known_names.get(name, name)))
  return symbols


def parse_link_map(
  path: Path, known_names: dict[str, str], *, object_text_names: dict[str, str] | None = None,
) -> list[Symbol]:
  text = path.read_text(errors="replace")
  if "Publics by Value" in text:
    return parse_msvc_link_map(text, known_names, object_text_names or {})
  symbols: list[Symbol] = []
  table_row = re.compile(r"^\s*([0-9a-fA-F]+)\s+([0-9a-fA-F]+)\s+([0-9a-fA-F]+)\s+\d+\s+(.+)$")
  gnu_section = re.compile(r"^\s*(\.text(?:\.[^ ]+)?)\s+0x([0-9a-fA-F]+)\s+0x([0-9a-fA-F]+)(?:\s+.*)?$")
  gnu_wrapped_section = re.compile(r"^\s*(\.text\.[^ ]+)\s*$")
  gnu_wrapped_address = re.compile(r"^\s*0x([0-9a-fA-F]+)\s+0x([0-9a-fA-F]+)(?:\s+.*)?$")
  gnu_named_symbol = re.compile(r"^\s*0x([0-9a-fA-F]+)\s+(\S+)\s*$")
  text_groups: list[tuple[int, int, list[tuple[int, str]]]] = []
  active_text: tuple[int, int, list[tuple[int, str]]] | None = None

  def text_group(address: int, size: int) -> tuple[int, int, list[tuple[int, str]]]:
    group: tuple[int, int, list[tuple[int, str]]] = (address, address + size, [])
    text_groups.append(group)
    return group

  def named_symbol(address: int, name: str) -> None:
    # Local assembly labels are internal offsets, not function boundaries.
    if active_text is not None and active_text[0] <= address < active_text[1] and not is_local_label(name):
      active_text[2].append((address, known_names.get(name, name)))

  pending_gnu_section: str | None = None
  in_discarded_sections = False
  for line in text.splitlines():
    marker = line.strip()
    if marker == "Discarded input sections":
      in_discarded_sections = True
      pending_gnu_section = None
      continue
    if marker in {"Memory Configuration", "Linker script and memory map"}:
      in_discarded_sections = False
      pending_gnu_section = None
      continue
    if in_discarded_sections:
      continue

    match = table_row.match(line)
    if match is not None:
      pending_gnu_section = None
      address = int(match.group(1), 16)
      size = int(match.group(3), 16)
      section_match = re.search(r":\((\.text(?:\.[^)]+)?)\)$", match.group(4))
      if section_match is not None:
        active_text = text_group(address, size)
      elif ":(" in match.group(4):
        active_text = None
      elif len(match.group(4).split()) == 1:
        named_symbol(address, match.group(4))
      raw_name = section_symbol(section_match.group(1), known_names) if section_match else None
      if raw_name is not None and size > 0:
        symbols.append(Symbol(address, size, known_names.get(raw_name, raw_name)))
      continue
    match = gnu_section.match(line)
    if match is not None:
      pending_gnu_section = None
      raw_name = section_symbol(match.group(1), known_names)
      size = int(match.group(3), 16)
      active_text = text_group(int(match.group(2), 16), size)
      if raw_name is not None and size > 0:
        symbols.append(Symbol(int(match.group(2), 16), size, known_names.get(raw_name, raw_name)))
      continue
    match = gnu_wrapped_section.match(line)
    if match is not None:
      pending_gnu_section = match.group(1)
      continue
    match = gnu_wrapped_address.match(line)
    if match is not None and pending_gnu_section is not None:
      raw_name = section_symbol(pending_gnu_section, known_names)
      size = int(match.group(2), 16)
      active_text = text_group(int(match.group(1), 16), size)
      if raw_name is not None and size > 0:
        symbols.append(Symbol(int(match.group(1), 16), size, known_names.get(raw_name, raw_name)))
    elif match := gnu_named_symbol.match(line):
      named_symbol(int(match.group(1), 16), match.group(2))
    pending_gnu_section = None
  for _, end, named in text_groups:
    starts = sorted({address for address, _ in named})
    for address, name in named:
      next_index = bisect.bisect_right(starts, address)
      bound = starts[next_index] if next_index < len(starts) else end
      symbols.append(Symbol(address, bound - address, name))
  return merged_symbols([], symbols)


def parse_final_nm(path: Path) -> list[Symbol]:
  rows = [row for line in path.read_text(errors="replace").splitlines() if (row := parse_nm_line(line)) is not None]
  text_rows = sorted((address, name) for address, kind, name in rows if kind in {"t", "T", "w", "W"} and not is_local_label(name))
  symbols: list[Symbol] = []
  for index, (address, name) in enumerate(text_rows):
    next_address = next((row[0] for row in text_rows[index + 1 :] if row[0] > address), address)
    symbols.append(Symbol(address, max(0, next_address - address), name))
  return symbols


def parse_indirect_symbols(path: Path) -> list[Symbol]:
  symbols = []
  in_text_stubs = False
  for line in path.read_text(errors="replace").splitlines():
    if line.startswith("Indirect symbols for "):
      in_text_stubs = line.startswith("Indirect symbols for (__TEXT,__stubs)")
      continue
    if not in_text_stubs:
      continue
    match = re.match(r"^0x([0-9a-fA-F]+)\s+\d+\s+(.+)$", line.strip())
    if match is not None:
      symbols.append(Symbol(int(match.group(1), 16), 1, match.group(2)))
  return symbols


def parse_dynamic_relocations(path: Path) -> dict[int, tuple[str, str]]:
  relocations: dict[int, tuple[str, str]] = {}
  for line in path.read_text(errors="replace").splitlines():
    match = re.match(r"^\s*([0-9a-fA-F]+)\s+(R_[A-Z0-9_]+)\s+(\S+)\s*$", line)
    if match is not None:
      address = int(match.group(1), 16)
      if address in relocations:
        raise ValueError(f"duplicate dynamic relocation at 0x{address:x}")
      relocations[address] = (match.group(2), match.group(3))
  return relocations


def parse_power_import_thunks(
  link_map: Path, raw_disassembly: Path, got_contents: Path,
  relocations: dict[int, tuple[str, str]],
) -> list[Symbol]:
  """Resolve only the five-instruction ELFv2 little-endian import thunk."""
  raw = raw_disassembly.read_text(errors="replace")
  got = got_contents.read_text(errors="replace")
  if "file format elf64-powerpcle" not in raw or "file format elf64-powerpcle" not in got:
    raise ValueError("POWER thunk evidence is not little-endian ELF64")
  ranges = [
    (int(match.group(1), 16), int(match.group(3), 16))
    for line in link_map.read_text(errors="replace").splitlines()
    if (match := re.match(
      r"^\s*([0-9a-fA-F]+)\s+([0-9a-fA-F]+)\s+([0-9a-fA-F]+)\s+\d+\s+<internal>:\(\.text\.thunk\)\s*$", line,
    )) is not None
  ]
  if not ranges:
    # Other linkers can name import stubs directly. Do not require the LLD
    # synthetic section, and do not invent an alternative symbol identity.
    # Any unresolved section destination still fails the final closure gate.
    return []
  got_match = re.search(
    r"Contents of section \.got:\n\s*([0-9a-fA-F]+)\s+([0-9a-fA-F]{8})\s+([0-9a-fA-F]{8})(?:\s|$)", got,
  )
  if got_match is None:
    raise ValueError("POWER thunk evidence is missing the .got TOC pointer")
  toc = int.from_bytes(bytes.fromhex(got_match.group(2) + got_match.group(3)), "little")
  if toc != int(got_match.group(1), 16) + 0x8000:
    raise ValueError("POWER .got TOC pointer does not match the supported ELFv2 layout")
  instructions = {
    int(match.group(1), 16): int.from_bytes(bytes.fromhex(match.group(2)), "little")
    for line in raw.splitlines()
    if (match := re.match(r"^\s*([0-9a-fA-F]+):\s+((?:[0-9a-fA-F]{2}\s+){4})\S", line)) is not None
  }

  def signed16(value: int) -> int:
    return value - 0x10000 if value & 0x8000 else value

  symbols = []
  for start, size in ranges:
    if size == 0 or size % 20:
      raise ValueError("unsupported POWER import-thunk range size")
    for address in range(start, start + size, 20):
      words = [instructions.get(address + offset) for offset in range(0, 20, 4)]
      # std r2,24(r1); addis r12,r2,hi; ld r12,lo(r12); mtctr r12; bctr.
      # Checking encoded registers, modes, and every instruction prevents an
      # arbitrary indirect transfer from acquiring an external callee identity.
      if (None in words or words[0] != 0xF8410018 or words[1] & 0xFFFF0000 != 0x3D820000
          or words[2] & 0xFFFF0003 != 0xE98C0000 or words[3:] != [0x7D8903A6, 0x4E800420]):
        raise ValueError(f"unsupported POWER import thunk at 0x{address:x}")
      slot = toc + (signed16(words[1] & 0xFFFF) << 16) + signed16(words[2] & 0xFFFC)
      relocation = relocations.get(slot)
      if (relocation is None or relocation[0] != "R_PPC64_JMP_SLOT"
          or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:@[A-Za-z0-9_.]+)?", relocation[1]) is None):
        raise ValueError(f"POWER import thunk at 0x{address:x} has no named JMP_SLOT at 0x{slot:x}")
      symbols.append(Symbol(address, 20, relocation[1] + "@plt"))
  return symbols


def parse_s390_import_thunks(
  raw_disassembly: Path, relocations: dict[int, tuple[str, str]] | None,
) -> list[Symbol]:
  """Resolve the exact s390x PLT load-and-branch path through a named slot."""
  raw = raw_disassembly.read_text(errors="replace")
  if "file format elf64-s390" not in raw:
    return []
  instructions: dict[int, bytes] = {}
  in_plt = False
  for line in raw.splitlines():
    if section := re.match(r"^Disassembly of section (.+):$", line):
      in_plt = section.group(1) == ".plt"
      continue
    if in_plt and (match := re.match(r"^\s*([0-9a-fA-F]+):\s+((?:[0-9a-fA-F]{2}\s+)+)", line)):
      instructions[int(match.group(1), 16)] = bytes.fromhex(match.group(2))
  if not instructions and not relocations:
    return []
  if relocations is None:
    raise ValueError("s390x PLT requires dynamic-relocation evidence")
  expected_slots = {slot for slot, (kind, _) in relocations.items() if kind == "R_390_JMP_SLOT"}
  verified_slots: set[int] = set()
  symbols = []
  for address, first in sorted(instructions.items()):
    # LARL r1,slot; LG r1,0(r1); BR r1. Decode the signed halfword
    # displacement independently of the printed address and symbol annotation.
    if (len(first) != 6 or first[:2] != b"\xc0\x10"
        or instructions.get(address + 6) != b"\xe3\x10\x10\x00\x00\x04"
        or instructions.get(address + 12) != b"\x07\xf1"):
      continue
    slot = address + 2 * int.from_bytes(first[2:], "big", signed=True)
    relocation = relocations.get(slot)
    if (relocation is None or relocation[0] != "R_390_JMP_SLOT"
        or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:@[A-Za-z0-9_.]+)?", relocation[1]) is None):
      raise ValueError(f"s390x PLT entry at 0x{address:x} has no named JMP_SLOT at 0x{slot:x}")
    symbols.append(Symbol(address, 14, relocation[1] + "@plt"))
    verified_slots.add(slot)
  missing = expected_slots - verified_slots
  if missing:
    raise ValueError("s390x JMP_SLOT entries lack verified PLT code: " + ", ".join(f"0x{slot:x}" for slot in sorted(missing)))
  return symbols


def resolve_x86_got_transfer(
  line: str, relocations: dict[int, tuple[str, str]], symbols: dict[int, Symbol],
) -> str:
  """Bind a fixed RIP-relative transfer to its exact dynamic relocation."""
  match = re.match(
    r"^\s*([0-9a-fA-F]+):\s+((?:[0-9a-fA-F]{2}\s+){6})"
    r"(callq?|jmpq?)\s+\*(-?0x[0-9a-fA-F]+)?\(%rip\)\s*#\s*0x([0-9a-fA-F]+)(?:\s+<.*>)?\s*$", line,
  )
  if match is None:
    return line
  encoded = bytes.fromhex(match.group(2))
  expected = b"\xff\x15" if match.group(3).startswith("call") else b"\xff\x25"
  if encoded[:2] != expected:
    raise ValueError("x86 GOT transfer text does not match its opcode")
  displacement = int.from_bytes(encoded[2:], "little", signed=True)
  slot = int(match.group(1), 16) + len(encoded) + displacement
  if displacement != int(match.group(4) or "0", 16) or slot != int(match.group(5), 16):
    raise ValueError("x86 GOT transfer text does not match its encoded slot")
  relocation = relocations.get(slot)
  if relocation is None:
    return line
  kind, value = relocation
  name = None
  if kind == "R_X86_64_RELATIVE":
    relative = re.fullmatch(r"\*ABS\*\+0x([0-9a-fA-F]+)", value)
    symbol = symbols.get(int(relative.group(1), 16)) if relative is not None else None
    if symbol is None:
      raise ValueError(f"x86 GOT slot 0x{slot:x} does not resolve to a mapped text entry")
    name = symbol.name
  elif kind in {"R_X86_64_GLOB_DAT", "R_X86_64_JUMP_SLOT"}:
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:@[A-Za-z0-9_.]+)?", value) is None:
      raise ValueError(f"x86 GOT slot 0x{slot:x} has no supported external symbol name")
    name = value + "@plt"
  if name is None:
    return line
  # The marker distinguishes relocation-backed resolution from an incidental
  # raw objdump label for the GOT slot itself. The closure gate requires it.
  return line[:line.index("#", match.start(3))] + f"# GOT 0x{slot:x} <{name}>"


def merged_symbols(nm_symbols: list[Symbol], map_symbols: list[Symbol]) -> list[Symbol]:
  rows: dict[tuple[int, str], Symbol] = {}
  for symbol in [*map_symbols, *nm_symbols]:
    key = (symbol.address, symbol.name)
    prior = rows.get(key)
    if prior is None or symbol.size > prior.size:
      rows[key] = symbol
  return sorted(rows.values(), key=lambda row: (row.address, row.name))


def canonical_symbols(symbols: list[Symbol]) -> list[Symbol]:
  by_address: dict[int, list[Symbol]] = {}
  for symbol in symbols:
    if (
      symbol.size > 0
      and symbol.name != "__mh_execute_header"
      and not is_local_label(symbol.name)
    ):
      by_address.setdefault(symbol.address, []).append(symbol)

  def priority(symbol: Symbol) -> tuple[int, str]:
    if symbol.name.startswith("ct_entry_"):
      return 0, symbol.name
    if "rscrypto::" in symbol.name or "rscrypto_ct_" in symbol.name:
      return 1, symbol.name
    return 2, symbol.name

  return [min(rows, key=priority) for _, rows in sorted(by_address.items())]


def symbolized_target(address: int, starts: list[int], symbols: list[Symbol]) -> str | None:
  index = bisect.bisect_right(starts, address) - 1
  if index < 0:
    return None
  symbol = symbols[index]
  if address >= symbol.address + symbol.size:
    return None
  offset = address - symbol.address
  return symbol.name if offset == 0 else f"{symbol.name}+0x{offset:x}"


def recover_function_boundaries(
  raw_lines: list[str], symbols: list[Symbol], binary: Path, objdump: Path,
  recovery_path: Path, objdump_args: list[str],
) -> list[str]:
  """Restart decoding where stripped inline data consumed a mapped entrypoint."""
  instruction_address = re.compile(r"^\s*([0-9a-fA-F]+):")
  addresses = {
    int(match.group(1), 16) for line in raw_lines
    if (match := re.match(r"^\s*([0-9a-fA-F]+):\s+(?:[0-9a-fA-F]{2})+(?:\s|$)", line)) is not None
  }
  recoveries = []
  for index, symbol in enumerate(symbols):
    if symbol.address in addresses:
      continue
    end = symbol.address + symbol.size
    if index + 1 < len(symbols):
      end = min(end, symbols[index + 1].address)
    command = [str(objdump), "--disassemble", "--reloc", "--demangle", *objdump_args,
               f"--start-address=0x{symbol.address:x}", f"--stop-address=0x{end:x}", str(binary)]
    decoded = subprocess.check_output(command, text=True)
    rows = [line for line in decoded.splitlines()
            if (match := instruction_address.match(line)) is not None
            and symbol.address <= int(match.group(1), 16) < end]
    if not rows or int(instruction_address.match(rows[0]).group(1), 16) != symbol.address:
      raise ValueError(f"cannot recover mapped function start: {symbol.name}")
    # Keep the original raw dump intact. Replace all misdecoded instruction and
    # relocation rows in this bounded range, not just its first instruction.
    replaced = [i for i, line in enumerate(raw_lines)
                if (match := instruction_address.match(line)) is not None
                and symbol.address <= int(match.group(1), 16) < end]
    if not replaced:
      raise ValueError(f"no raw instruction range for mapped function: {symbol.name}")
    raw_lines[replaced[0]:replaced[-1] + 1] = rows
    recoveries.append({"symbol": symbol.name, "start": symbol.address, "end": end,
                       "command": command, "disassembly": decoded})
  recovery_path.write_text(json.dumps({
    "binary": str(binary), "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
    "recoveries": recoveries,
  }, indent=2) + "\n")
  return raw_lines


def symbolize(
  raw_disassembly: Path, out_path: Path, symbols: list[Symbol], *,
  binary: Path | None = None, objdump: Path | None = None, objdump_args: list[str] | None = None,
  dynamic_relocations: dict[int, tuple[str, str]] | None = None,
) -> None:
  canonical = canonical_symbols(symbols)
  if not canonical:
    raise ValueError("no final linked text symbols found")
  by_address = {symbol.address: symbol for symbol in canonical}
  starts = [symbol.address for symbol in canonical]
  # Some targets (notably s390x) print numeric transfer destinations without
  # a label. They need the same linker-map resolution as existing labels.
  target_re = re.compile(r"0x([0-9a-fA-F]+)(?:\s+<.*>)?\s*$")
  # RISC-V AUIPC/JALR pairs carry a resolved section-relative annotation,
  # while the printed operand is still an offset from a register.
  raw_lines = raw_disassembly.read_text(errors="replace").splitlines()
  x86_elf = any("file format elf64-x86-64" in line for line in raw_lines)
  if binary is not None and objdump is not None:
    raw_lines = recover_function_boundaries(
      raw_lines, canonical, binary, objdump,
      out_path.with_name(out_path.name.replace(".disasm.txt", ".boundary-recovery.json")), objdump_args or [],
    )
  raw_labels = {
    match.group(2): int(match.group(1), 16)
    for line in raw_lines
    if (match := re.match(r"^\s*([0-9a-fA-F]+) <(\.[^<>]+)>:$", line))
  }
  # ELF nm can bound _init by the next named text symbol across .plt. The
  # decoder's exact import entry is stronger evidence than that inferred range.
  plt_targets = {
    int(match.group(1), 16): match.group(2)
    for line in raw_lines
    if (match := re.match(r"^\s*([0-9a-fA-F]+) <([^<>]+@plt)>:$", line))
  }
  section_target_re = re.compile(r"<(\.[^<>+]+)(?:\+0x([0-9a-fA-F]+))?>$")
  output: list[str] = []
  seen: set[int] = set()
  for line in raw_lines:
    existing_label = re.match(r"^\s*([0-9a-fA-F]+) <(.+)>:$", line)
    if existing_label is not None:
      address = int(existing_label.group(1), 16)
      if address in by_address:
        output.append(f"{address:016x} <{by_address[address].name}>:")
        seen.add(address)
        continue
      # GNU keeps assembly-local labels that LLD may omit. Within a mapped
      # function they mark offsets, not new bodies; retain every instruction.
      # Unknown boundaries and section labels still terminate the prior body.
      if (
        existing_label.group(2).startswith(("L", ".L", "$", "ltmp"))
        and symbolized_target(address, starts, canonical) is not None
      ):
        continue
    instruction = re.match(r"^\s*([0-9a-fA-F]+):", line)
    if instruction is not None:
      address = int(instruction.group(1), 16)
      if address in by_address and address not in seen:
        output.append(f"{address:016x} <{by_address[address].name}>:")
        seen.add(address)

    def replace_target(match: re.Match[str]) -> str:
      address = int(match.group(1), 16)
      name = plt_targets.get(address) if address not in by_address else None
      if name is None:
        name = symbolized_target(address, starts, canonical)
      return match.group(0) if name is None else f"0x{address:x} <{name}>"

    if x86_elf and dynamic_relocations is not None:
      line = resolve_x86_got_transfer(line, dynamic_relocations, by_address)
    line = target_re.sub(replace_target, line)
    if reference := section_target_re.search(line):
      base = raw_labels.get(reference.group(1))
      if base is not None:
        name = symbolized_target(base + int(reference.group(2) or "0", 16), starts, canonical)
        if name is not None:
          line = line[:reference.start()] + f"<{name}>"
    output.append(line)
  missing = sorted(symbol.name for symbol in canonical if symbol.address not in seen)
  if missing:
    raise ValueError(f"final disassembly missing {len(missing)} mapped function start(s), first: {missing[0]}")
  out_path.write_text("\n".join(output) + "\n")


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--artifact-dir", required=True, type=Path)
  parser.add_argument("--raw-disassembly", required=True, type=Path)
  parser.add_argument("--nm-symbols", required=True, type=Path)
  parser.add_argument("--link-map", type=Path)
  parser.add_argument("--indirect-symbols", type=Path)
  parser.add_argument("--dynamic-relocations", type=Path)
  parser.add_argument("--got-contents", type=Path, help="POWER ELFv2 .got contents for exact import-thunk resolution")
  parser.add_argument("--out-disassembly", required=True, type=Path)
  parser.add_argument("--out-symbols", required=True, type=Path)
  parser.add_argument("--binary", type=Path, help="original linked binary for bounded decoder restarts")
  parser.add_argument("--llvm-objdump", type=Path, help="matching LLVM decoder for the original binary")
  parser.add_argument("--objdump-arg", action="append", default=[], help="extra decoder argument")
  args = parser.parse_args()
  if (args.binary is None) != (args.llvm_objdump is None):
    parser.error("--binary and --llvm-objdump must be provided together")
  if args.got_contents is not None and (args.dynamic_relocations is None or args.link_map is None):
    parser.error("--got-contents requires --dynamic-relocations and --link-map")

  known_names = demangle_table(args.artifact_dir)
  nm_symbols = parse_final_nm(args.nm_symbols)
  map_symbols = (parse_link_map(args.link_map, known_names,
                                object_text_names=demangle_table(args.artifact_dir, text_only=True))
                 if args.link_map and args.link_map.is_file() else [])
  indirect_symbols = (
    parse_indirect_symbols(args.indirect_symbols)
    if args.indirect_symbols and args.indirect_symbols.is_file()
    else []
  )
  relocations = parse_dynamic_relocations(args.dynamic_relocations) if args.dynamic_relocations else None
  power_symbols = (parse_power_import_thunks(args.link_map, args.raw_disassembly, args.got_contents, relocations)
                   if args.got_contents else [])
  s390_symbols = parse_s390_import_thunks(args.raw_disassembly, relocations)
  symbols = merged_symbols(nm_symbols, [*map_symbols, *indirect_symbols, *power_symbols, *s390_symbols])
  if not symbols:
    raise ValueError("neither the final binary nor the linker map contains text symbols")
  args.out_symbols.write_text(
    "".join(f"{symbol.address:016x} {symbol.size:016x} T {symbol.name}\n" for symbol in symbols)
  )
  symbolize(args.raw_disassembly, args.out_disassembly, symbols,
            binary=args.binary, objdump=args.llvm_objdump, objdump_args=args.objdump_arg,
            dynamic_relocations=relocations)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
