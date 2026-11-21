#!/usr/bin/env -S uv run python
"""Describe linked-binary data addresses independently of their placement.

The assembly scanner binds each reviewed function to an instruction identity.
A PC-relative, TOC-relative, or `larl` reference encodes the distance to its
target, so relinking moves it even when the referenced object is unchanged.
This module resolves such a target address against the linked binary itself:

- code resolves to the containing function and offset;
- a writable object resolves to its symbol and offset;
- a read-only object resolves to its symbol (when it has a stable name), the
  offset, and a digest of the complete object contents;
- pointer slots inside read-only contents resolve recursively, so a moved
  pointer target with unchanged contents keeps the same digest.

A read-only digest covers one allocation unit. ELF units are linker-map input
sections; Mach-O units are ld64 map atoms. Without a map, a PE or Mach-O unit
runs from the nearest known object start to the next one. Such a unit and an
ld64 atom can end in alignment padding, whose length depends on where the linker
placed the next object, so a reference with a known read extent drops the zero
bytes after that extent (`without_padding`). Merged string and literal sections bind only the
referenced literal. A read-only LLVM global-merge unit (`_MergedGlobals`) holds
many objects. The assembly that the same compilation emitted gives each one as
an alias with a size (`assembly_merge_units`), so a reference binds the object
it designates instead of every other constant the compiler merged with it.

A `core::panic::Location` (a pointer to a `.rs` path, the path's length, a line,
and a column) describes as `loc`, also inside other contents. A location is
read only by the panic it is passed to, so its file, checkout path, and line
record placement in the source tree, not behavior of the code under review.

Any address the binary cannot explain stays unresolved, and the caller keeps
the exact, placement-dependent instruction bytes.
"""

from __future__ import annotations

import bisect
import hashlib
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

POINTER_SIZE = 8
# Pointer chains in read-only data are followed this far before the digest
# falls back to the raw, placement-dependent target address.
MAX_DESCRIPTOR_DEPTH = 3
# Extent element marker for an LLVM global-merge unit, which holds many objects.
MERGED = -1
# `core::panic::Location<'static>` on a 64-bit target: `&str` (pointer, length), line, column.
LOCATION_SIZE = 3 * POINTER_SIZE
# Longest source path accepted as a `Location` file name.
MAX_LOCATION_PATH = 4096
# Extents checked before an address that no later-starting extent contains.
EXTENT_SEARCH = 16

ELF_RELATIVE = {62: {8, 37}, 183: {1027, 1032}, 243: {3, 58}, 21: {22, 248}, 22: {12, 61}}
ELF_SYMBOLIC = {62: {1, 6, 7}, 183: {257, 1025, 1026}, 243: {2, 5}, 21: {20, 21, 38}, 22: {10, 11, 22}}

# Names whose numbering or address encoding changes with unrelated code.
ANONYMOUS_NAME = re.compile(
  r"^(?:\.L|L_|L\d|l_|ltmp|anon\.|\.Lanon|alloc_|\.L_MergedGlobals|_MergedGlobals|GCC_except_table|__unnamed|"
  r"\.rodata|\.data|\.rdata|\.bss|\.tbss|\.tdata|\$)"
)


def is_anonymous(name: str) -> bool:
  return ANONYMOUS_NAME.match(name) is not None or name.startswith((".", "$"))


@dataclass(frozen=True)
class Section:
  name: str
  address: int
  size: int
  offset: int | None
  executable: bool
  writable: bool
  slot_table: bool = False
  # 0 for NUL-terminated string literals, N for N-byte literal pools.
  element: int | None = None

  @property
  def end(self) -> int:
    return self.address + self.size


@dataclass(frozen=True)
class Slot:
  """A pointer-sized location that the loader fills."""

  address: int | None = None
  symbol: str | None = None
  addend: int = 0


@dataclass
class LinkedImage:
  data: bytes
  sections: list[Section]
  little_endian: bool = True
  # Known object starts with an optional stable name.
  starts: dict[int, str | None] = field(default_factory=dict)
  # Allocation units from a linker map: (start, end, element size). The
  # element size is 0 for merged NUL-terminated strings, N for merged N-byte
  # constants, MERGED for an LLVM global-merge unit, and None for one allocation.
  extents: list[tuple[int, int, int | None]] = field(default_factory=list)
  slots: dict[int, Slot] = field(default_factory=dict)
  # Function code ranges from the linked disassembly: (start, end, name).
  functions: list[tuple[int, int, str]] = field(default_factory=list)
  toc_base: int | None = None
  # Without map units, bound an object by the surrounding known starts.
  bounded_by_starts: bool = False
  # The map units can end in the alignment padding before the next unit, as ld64 atoms do.
  padded_units: bool = False
  # The objects of the read-only global-merge units, as (start, end), from the compiler's assembly.
  members: list[tuple[int, int]] = field(default_factory=list)
  # The argument registers that each function can read before it writes them (`condition_slice.argument_reads`),
  # which `asm_heuristics.prepare_linked_image` sets.
  argument_reads: dict[str, frozenset[str]] = field(default_factory=dict)
  # The LLVM IR parameter lists of each function, by demangled name (`ir_parameters`).
  parameters: dict[str, frozenset[tuple[str, ...]]] = field(default_factory=dict)

  def __post_init__(self) -> None:
    self.sections.sort(key=lambda section: section.address)
    self._section_starts = [section.address for section in self.sections]
    self.extents.sort(key=lambda extent: (extent[0], extent[1]))
    self._extent_starts = [start for start, _, _ in self.extents]
    self._object_starts = sorted(self.starts)
    self._slot_addresses = sorted(self.slots)
    self.set_functions(self.functions)
    self._locations: dict[int, bool] = {}
    self._merged = [(start, end) for start, end, element in self.extents if element == MERGED]
    self._merged_starts = [start for start, _ in self._merged]
    self.set_members(self.members)

  def set_functions(self, functions: list[tuple[int, int, str]]) -> None:
    self.functions = sorted(functions)
    self._function_starts = [start for start, _, _ in self.functions]

  def set_members(self, members: list[tuple[int, int]]) -> None:
    self.members = sorted(members)
    self._member_starts = [start for start, _ in self.members]
    self._digests: dict[tuple[int, int, int], str] = {}

  def section_at(self, address: int) -> Section | None:
    index = bisect.bisect_right(self._section_starts, address) - 1
    if index >= 0:
      section = self.sections[index]
      if section.address <= address < section.end:
        return section
    return None

  def function_at(self, address: int) -> tuple[int, int, str] | None:
    index = bisect.bisect_right(self._function_starts, address) - 1
    if index >= 0 and self.functions[index][0] <= address < self.functions[index][1]:
      return self.functions[index]
    return None

  def read(self, address: int, size: int) -> bytes | None:
    section = self.section_at(address)
    if section is None or address + size > section.end:
      return None
    if section.offset is None:
      return bytes(size)
    start = section.offset + address - section.address
    return self.data[start : start + size] if start + size <= len(self.data) else None

  def merged_unit_count(self) -> int:
    return len(self._merged)

  def merged_unit(self, address: int) -> tuple[int, int] | None:
    """Return the LLVM global-merge unit that contains `address`."""
    index = bisect.bisect_right(self._merged_starts, address) - 1
    if index >= 0 and self._merged[index][0] <= address < self._merged[index][1]:
      return self._merged[index]
    return None

  def without_padding(self, start: int, end: int, address: int, minimum_end: int) -> int:
    """End an inferred or padded bound before trailing zero bytes, which may be alignment padding.

    Only a reference with a known extent, `minimum_end`, trims, and never below that extent: a reference whose
    extent is unknown can read every byte. The bound never cuts a pointer slot or a `Location`, whose high bytes
    are often zero.
    """
    if not minimum_end:
      return end
    contents = self.read(start, end - start)
    if contents is None:
      return max(end, address + 1, minimum_end)
    trimmed = start + len(contents.rstrip(b"\0"))
    first = bisect.bisect_left(self._slot_addresses, start)
    for slot_address in self._slot_addresses[first : bisect.bisect_left(self._slot_addresses, end)]:
      size = LOCATION_SIZE if self.location_at(slot_address) else POINTER_SIZE
      if slot_address + POINTER_SIZE <= end:
        trimmed = max(trimmed, min(slot_address + size, end))
    return max(trimmed, address + 1, minimum_end)

  def region(self, address: int, minimum_end: int = 0) -> tuple[int, int, str | None] | None:
    """Return the allocation unit, merged member, or merged literal that contains `address`.

    `minimum_end` keeps an inferred bound over every byte that the referencing access reads.
    """
    section = self.section_at(address)
    if section is None:
      return None
    unit: tuple[int, int, int | None] | None = None
    index = bisect.bisect_right(self._extent_starts, address) - 1
    # A small extent can start inside a larger one, so search back for the extent that contains `address`.
    for candidate in range(index, max(index - EXTENT_SEARCH, -1), -1):
      start, end, element = self.extents[candidate]
      if start <= address < end and section.address <= start and end <= section.end:
        unit = (start, end, element)
        break
    if unit is None and self.bounded_by_starts:
      index = bisect.bisect_right(self._object_starts, address) - 1
      start = max(self._object_starts[index] if index >= 0 else section.address, section.address)
      following = self._object_starts[index + 1] if index + 1 < len(self._object_starts) else section.end
      end = self.without_padding(start, min(following, section.end), address, minimum_end)
      return (start, min(end, section.end), self.starts.get(start))
    if unit is not None and unit[2] is None and self.padded_units:
      start, end, _ = unit
      return (start, min(self.without_padding(start, end, address, minimum_end), end), self.starts.get(start))
    if unit is None:
      return None
    start, end, element = unit
    if element == MERGED:
      index = bisect.bisect_right(self._member_starts, address) - 1
      if index >= 0:
        member_start, member_end = self.members[index]
        if address < member_end and minimum_end <= member_end:
          return (member_start, member_end, None)
      # An address in alignment padding, or an access past the end of its object, can reach any object.
      return (start, end, None)
    element = section.element if element is None else element
    if element is not None:
      return self.element_region(address, start, end, element)
    return (start, end, self.starts.get(start))

  def element_region(self, address: int, start: int, end: int, element: int) -> tuple[int, int, None] | None:
    """Bound one merged literal: a NUL-terminated string or an N-byte constant."""
    if element == 0:
      contents = self.read(address, end - address)
      if contents is None:
        return None
      terminator = contents.find(b"\0")
      return (address, address + (terminator + 1 if terminator >= 0 else len(contents)), None)
    first = start + (address - start) // element * element
    return (first, min(first + element, end), None)

  def location_at(self, address: int) -> bool:
    """Whether `address` starts a `core::panic::Location` in read-only data.

    A location is a pointer to its file name, the name's length, a line, and a
    column. The pointer must be one the loader relocates (when the image has
    relocations), and it must address exactly that many bytes of read-only
    UTF-8 text ending in `.rs`.
    """
    if address in self._locations:
      return self._locations[address]
    found = False
    section = self.section_at(address)
    raw = self.read(address, LOCATION_SIZE) if section is not None and not section.writable else None
    if raw is not None:
      pointer, length, line, column = struct.unpack(("<" if self.little_endian else ">") + "QQII", raw)
      slot = self.slots.get(address)
      if self.slots:
        pointer = slot.address if slot is not None and slot.address is not None else None
      target = self.section_at(pointer) if pointer is not None else None
      if (target is not None and not target.executable and not target.writable and line >= 1 and column >= 1
          and 0 < length <= MAX_LOCATION_PATH):
        name = self.read(pointer, length)
        if name is not None and name.endswith(b".rs") and b"\0" not in name:
          try:
            name.decode("utf-8")
            found = True
          except UnicodeDecodeError:
            pass
    self._locations[address] = found
    return found

  def describe(self, address: int, depth: int = 0, minimum_end: int = 0) -> str | None:
    """Return a placement-independent descriptor, or None when unexplained."""
    if function := self.function_at(address):
      return f"code:{function[2]}+0x{address - function[0]:x}"
    section = self.section_at(address)
    if section is None or section.executable:
      return None
    if section.slot_table:
      slot = self.slots.get(address)
      return f"slot:{self.describe_slot(slot, depth)}" if slot is not None else None
    if not section.writable and self.location_at(address):
      return "loc"
    region = self.region(address, minimum_end)
    if region is None:
      return None
    start, end, name = region
    offset = address - start
    if section.writable:
      return f"data:{name}+0x{offset:x}" if name else None
    digest = self.digest(start, end, depth)
    return f"const:{name or '-'}+0x{offset:x}/{end - start}:{digest}"

  def describe_unit(self, address: int) -> str | None:
    """Describe the whole global-merge unit that starts at `address`, for a base that escapes."""
    unit = self.merged_unit(address)
    if unit is None or unit[0] != address:
      return None
    return f"merged:{unit[1] - unit[0]}:{self.digest(unit[0], unit[1], 0)}"

  def describe_slot(self, slot: Slot, depth: int) -> str:
    if slot.symbol is not None:
      return f"import:{slot.symbol}+0x{slot.addend:x}"
    target = None
    if slot.address is not None and depth < MAX_DESCRIPTOR_DEPTH:
      target = self.describe(slot.address, depth + 1)
    return target if target is not None else f"abs:0x{slot.address or 0:x}"

  def digest(self, start: int, end: int, depth: int) -> str:
    key = (start, end, depth)
    if key in self._digests:
      return self._digests[key]
    contents = self.read(start, end - start)
    if contents is None:
      value = "unreadable"
    else:
      parts = []
      position = start
      index = bisect.bisect_left(self._slot_addresses, start)
      while index < len(self._slot_addresses) and self._slot_addresses[index] < end:
        slot_address = self._slot_addresses[index]
        index += 1
        if slot_address < position or slot_address + POINTER_SIZE > end:
          # A slot straddling the extent edge keeps its raw bytes.
          continue
        parts.append(contents[position - start : slot_address - start].hex())
        if slot_address + LOCATION_SIZE <= end and self.location_at(slot_address):
          parts.append("<loc>")
          position = slot_address + LOCATION_SIZE
          continue
        parts.append(f"<{self.describe_slot(self.slots[slot_address], depth)}>")
        position = slot_address + POINTER_SIZE
      parts.append(contents[position - start :].hex())
      value = hashlib.sha256("|".join(parts).encode()).hexdigest()[:24]
    self._digests[key] = value
    return value


def linked_binary_path(artifact_dir: Path) -> Path | None:
  disassembly = sorted(artifact_dir.glob("*.binary.disasm.txt"))
  if len(disassembly) != 1:
    return None
  path = artifact_dir / disassembly[0].name.removesuffix(".binary.disasm.txt")
  return path if path.is_file() else None


def nm_names(path: Path) -> dict[int, list[str]]:
  names: dict[int, list[str]] = {}
  if not path.is_file():
    return names
  for line in path.read_text(errors="replace").splitlines():
    match = re.match(r"^([0-9a-fA-F]+)\s+(\S)\s+(.+)$", line.strip())
    if match is not None and match.group(2) not in {"t", "T", "w", "W", "U", "a", "A", "N", "-"}:
      names.setdefault(int(match.group(1), 16), []).append(match.group(3))
  return names


def stable_name(names: list[str]) -> str | None:
  stable = sorted({name for name in names if not is_anonymous(name)})
  return "|".join(stable) if stable else None


def merged_element(input_section: str) -> int | None:
  """Return the literal size of an ELF merge section, 0 for strings, or MERGED for a global-merge unit."""
  if "_MergedGlobals" in input_section:
    return MERGED
  if re.fullmatch(r"\.rodata\.str\d+\.\d+", input_section):
    return 0
  if match := re.fullmatch(r"\.rodata\.cst(\d+)", input_section):
    return int(match.group(1))
  return None


@dataclass(frozen=True)
class AssemblyUnit:
  """One global-merge unit as the compiler's assembly defines it."""

  name: str
  contents: bytes
  # (offset, width) of each value that a relocation gives, such as `.quad .Lanon.1`.
  relocated: tuple[tuple[int, int], ...]
  # (offset, size) of each object that the unit holds.
  members: tuple[tuple[int, int], ...]


ASSEMBLY_LABEL = re.compile(r"^(\S+):$")
ASSEMBLY_ALIAS = re.compile(r"^(\S+)\s*=\s*(\S+?)(?:\+(\d+))?$")
ASSEMBLY_SIZE = re.compile(r"^\.size\s+(\S+),\s*(\d+)$")
ASSEMBLY_VALUE_WIDTH = {".byte": 1, ".short": 2, ".2byte": 2, ".long": 4, ".4byte": 4, ".quad": 8, ".8byte": 8}
ASSEMBLY_ESCAPES = {"b": 8, "f": 12, "n": 10, "r": 13, "t": 9, '"': 34, "\\": 92}


def assembly_string(text: str) -> bytes:
  """The bytes of a quoted `.ascii` operand, with the escapes that LLVM prints."""
  if len(text) < 2 or text[0] != '"' or text[-1] != '"':
    raise ValueError(f"unexpected string operand {text!r}")
  data, position, body = bytearray(), 0, text[1:-1]
  while position < len(body):
    character = body[position]
    if character != "\\":
      data.extend(character.encode())
      position += 1
    elif body[position + 1 : position + 2] in ASSEMBLY_ESCAPES:
      data.append(ASSEMBLY_ESCAPES[body[position + 1]])
      position += 2
    elif match := re.match(r"[0-7]{1,3}", body[position + 1 :]):
      data.append(int(match.group(0), 8) & 0xFF)
      position += 1 + len(match.group(0))
    else:
      raise ValueError(f"unexpected escape in string operand {text!r}")
  return bytes(data)


def assembly_merge_units(text: str, little_endian: bool) -> list[AssemblyUnit]:
  """Read each global-merge unit, its contents, and its objects from compiler assembly.

  LLVM emits a unit as one label whose data directives hold every object in order, with zero padding
  between them, and it gives each object an alias with a size: `.Lanon.1 = .L_MergedGlobals.2+411` and
  `.size .Lanon.1, 7`. A directive that this reader does not know stops it, so it never guesses a layout.
  """
  bodies: dict[str, list[str]] = {}
  sizes: dict[str, int] = {}
  aliases: list[tuple[str, str, int]] = []
  current = None
  for raw in text.splitlines():
    line = raw.strip()
    if current is None and "_MergedGlobals" not in line and not line.startswith(".size"):
      continue
    if match := ASSEMBLY_SIZE.match(line):
      sizes[match.group(1)] = int(match.group(2))
      if match.group(1) == current:
        current = None
    elif match := ASSEMBLY_LABEL.match(line):
      current = match.group(1) if "_MergedGlobals" in match.group(1) else None
      if current is not None:
        bodies[current] = []
    elif (match := ASSEMBLY_ALIAS.match(line)) and "_MergedGlobals" in match.group(2):
      aliases.append((match.group(1), match.group(2), int(match.group(3) or 0)))
    elif current is not None and line:
      bodies[current].append(line)
  units = []
  order = "little" if little_endian else "big"
  for name, body in bodies.items():
    contents, relocated = bytearray(), []
    for line in body:
      directive, operand = (line.split(None, 1) + [""])[:2]
      if directive in {".ascii", ".asciz"}:
        contents.extend(assembly_string(operand) + (b"\0" if directive == ".asciz" else b""))
        continue
      operand = re.sub(r"\s+#.*$", "", operand)
      if directive in {".zero", ".space"}:
        count, _, value = operand.partition(",")
        contents.extend(bytes([int(value or "0", 0) & 0xFF]) * int(count, 0))
      elif directive in ASSEMBLY_VALUE_WIDTH:
        width = ASSEMBLY_VALUE_WIDTH[directive]
        try:
          contents.extend((int(operand, 0) % (1 << (8 * width))).to_bytes(width, order))
        except ValueError:
          relocated.append((len(contents), width))
          contents.extend(bytes(width))
      else:
        raise ValueError(f"unexpected directive in global-merge unit {name}: {line}")
    if sizes.get(name) != len(contents):
      raise ValueError(f"global-merge unit {name} holds {len(contents)} bytes, but its size is {sizes.get(name)}")
    members = sorted((offset, sizes.get(alias, -1)) for alias, unit, offset in aliases if unit == name)
    covered = bytearray(len(contents))
    for offset, size in members:
      if size < 0 or offset + size > len(contents) or any(covered[offset : offset + size]):
        raise ValueError(f"global-merge unit {name} has an object at +{offset} with an unknown or overlapping size")
      covered[offset : offset + size] = b"\1" * size
    padding = [position for position, used in enumerate(covered) if not used]
    if any(contents[position] for position in padding) or any(
        not covered[offset] for offset, width in relocated for offset in range(offset, offset + width)):
      raise ValueError(f"global-merge unit {name} has contents that no object holds")
    units.append(AssemblyUnit(name, bytes(contents), tuple(relocated), tuple(members)))
  return units


def merge_unit_members(image: LinkedImage, units: list[AssemblyUnit]) -> list[tuple[int, int]]:
  """The objects of each read-only global-merge unit in `image`, matched to the assembly by contents.

  Each linked unit must equal exactly one member table: its bytes match an assembly unit except where a
  relocation fills them. Any other result, including an assembly unit that matches two linked units, is an error.
  """
  matched: dict[str, list[int]] = {}
  members: list[tuple[int, int]] = []
  for start, end in image._merged:
    linked = image.read(start, end - start)
    found = []
    for unit in units:
      if linked is None or len(unit.contents) != len(linked):
        continue
      expected, actual = bytearray(unit.contents), bytearray(linked)
      for offset, width in unit.relocated:
        actual[offset : offset + width] = bytes(width)
      if expected == actual:
        found.append(unit)
    if not found:
      raise ValueError(f"no assembly member table matches the global-merge unit at 0x{start:x}")
    if len({unit.members for unit in found}) != 1:
      raise ValueError(f"assembly units {', '.join(unit.name for unit in found)} have the bytes of the global-merge "
                       f"unit at 0x{start:x} but different member tables")
    for unit in found:
      matched.setdefault(unit.name, []).append(start)
    members.extend((start + offset, start + offset + size) for offset, size in found[0].members)
  for name, starts in matched.items():
    if len(starts) > 1:
      raise ValueError(f"assembly unit {name} matches the global-merge units at "
                       f"{', '.join(f'0x{start:x}' for start in starts)}")
  return members


def evidence_crate(binary: Path) -> str:
  """The crate name of the linked binary, which names its objects and its assembly."""
  return binary.name.split(".")[0].replace("-", "_")


def evidence_assembly(artifact_dir: Path, binary: Path) -> Path:
  """The assembly that the compilation of the linked binary's own crate emitted."""
  crate = evidence_crate(binary)
  paths = sorted(path for path in artifact_dir.glob("*.s") if re.split(r"[.-]", path.name)[0] == crate)
  if len(paths) != 1:
    raise ValueError(f"expected one assembly file of crate {crate} in {artifact_dir}, found {len(paths)}")
  return paths[0]


def link_map_extents(path: Path) -> list[tuple[int, int, str, str]]:
  """Return (start, size, input section, input file) rows from a GNU ld or lld ELF map."""
  if not path.is_file():
    return []
  text = path.read_text(errors="replace")
  if "Publics by Value" in text:
    return []
  rows: list[tuple[int, int, str, str]] = []
  lld_row = re.compile(r"^\s*([0-9a-fA-F]+)\s+[0-9a-fA-F]+\s+([0-9a-fA-F]+)\s+\d+\s+(\S.*):\((\S+)\)\s*$")
  gnu_row = re.compile(r"^ (\.\S+)\s+0x([0-9a-fA-F]+)\s+0x([0-9a-fA-F]+)\s+(\S.*)$")
  gnu_name = re.compile(r"^ (\.\S+)\s*$")
  gnu_address = re.compile(r"^\s+0x([0-9a-fA-F]+)\s+0x([0-9a-fA-F]+)\s+(\S.*)$")
  pending: str | None = None
  discarded = False
  for line in text.splitlines():
    marker = line.strip()
    if marker == "Discarded input sections":
      discarded = True
      continue
    if marker in {"Memory Configuration", "Linker script and memory map"}:
      discarded = False
      continue
    if discarded:
      continue
    if match := lld_row.match(line):
      rows.append((int(match.group(1), 16), int(match.group(2), 16), match.group(4), match.group(3).strip()))
      pending = None
    elif match := gnu_row.match(line):
      rows.append((int(match.group(2), 16), int(match.group(3), 16), match.group(1), match.group(4).strip()))
      pending = None
    elif match := gnu_name.match(line):
      pending = match.group(1)
    elif (match := gnu_address.match(line)) and pending is not None:
      rows.append((int(match.group(1), 16), int(match.group(2), 16), pending, match.group(3).strip()))
      pending = None
    else:
      pending = None
  return [row for row in rows if row[0] > 0 and row[1] > 0]


def msvc_link_map_starts(path: Path, names: dict[str, str]) -> dict[int, str | None]:
  starts: dict[int, str | None] = {}
  if not path.is_file():
    return starts
  row = re.compile(r"^\s*[0-9a-fA-F]+:[0-9a-fA-F]+\s+(\S+)\s+([0-9a-fA-F]+)\s")
  for line in path.read_text(errors="replace").splitlines():
    if match := row.match(line):
      raw = match.group(1)
      name = names.get(raw, raw)
      address = int(match.group(2), 16)
      if address:
        starts[address] = starts.get(address) or (None if is_anonymous(name) else name)
  return starts


def load_elf(data: bytes, artifact_dir: Path, binary: Path) -> LinkedImage:
  little = data[5] == 1
  order = "<" if little else ">"
  machine = struct.unpack_from(order + "H", data, 0x12)[0]
  phoff, shoff = struct.unpack_from(order + "QQ", data, 0x20)
  phentsize, phnum, shentsize, shnum, shstrndx = struct.unpack_from(order + "HHHHH", data, 0x36)
  relro: list[tuple[int, int]] = []
  for index in range(phnum):
    p_type, _flags, _offset, vaddr, _paddr, _filesz, memsz = struct.unpack_from(
      order + "IIQQQQQ", data, phoff + index * phentsize)
    if p_type == 0x6474E552:
      relro.append((vaddr, vaddr + memsz))
  headers = [struct.unpack_from(order + "IIQQQQIIQQ", data, shoff + index * shentsize) for index in range(shnum)]
  names_offset = headers[shstrndx][4]

  def cstring(offset: int) -> str:
    end = data.index(b"\0", offset)
    return data[offset:end].decode(errors="replace")

  sections: list[Section] = []
  by_index: dict[int, Section] = {}
  for index, (name, kind, flags, address, offset, size, *_rest) in enumerate(headers):
    if not flags & 0x2 or size == 0:
      continue
    section_name = cstring(names_offset + name)
    in_relro = any(start <= address and address + size <= end for start, end in relro)
    section = Section(
      name=section_name,
      address=address,
      size=size,
      offset=None if kind == 8 else offset,
      executable=bool(flags & 0x4),
      writable=bool(flags & 0x1) and not in_relro,
      slot_table=section_name in {".got", ".got.plt", ".toc", ".plt"} and not flags & 0x4,
    )
    sections.append(section)
    by_index[index] = section

  def symbols(table_index: int) -> list[tuple[str, int, int, int, int]]:
    _name, _kind, _flags, _address, offset, size, link, _info, _align, entsize = headers[table_index]
    strings = headers[link][4]
    rows = []
    for position in range(offset, offset + size, entsize or 24):
      st_name, info, _other, shndx, value, symbol_size = struct.unpack_from(order + "IBBHQQ", data, position)
      rows.append((cstring(strings + st_name), info & 0xF, shndx, value, symbol_size))
    return rows

  demangled = nm_names(artifact_dir / f"{binary.name}.binary.nm-symbols.txt")
  starts: dict[int, str | None] = {}
  extents: list[tuple[int, int, int | None]] = []
  toc_base = None
  for index, header in enumerate(headers):
    if header[1] != 2:
      continue
    for raw, kind, shndx, value, size in symbols(index):
      if shndx == 0 or shndx >= 0xFF00 or kind in {3, 4, 6} or shndx not in by_index:
        continue
      if raw == ".TOC.":
        toc_base = value
      if by_index[shndx].executable:
        continue
      name = stable_name(demangled.get(value, [])) if not is_anonymous(raw) else None
      starts[value] = starts.get(value) or name
  crate = evidence_crate(binary)
  for start, size, input_section, input_file in link_map_extents(artifact_dir / f"{binary.name}.link-map.txt"):
    # Thread-local templates use offsets that overlap ordinary data, and relocation input sections
    # take the names of the sections they relocate; neither is an object that code addresses.
    if input_section.startswith((".tbss", ".tdata", ".rela", ".rel.")):
      continue
    section = next((row for row in sections if row.address <= start < row.end), None)
    if section is None or section.executable:
      continue
    starts.setdefault(start, None)
    element = merged_element(input_section)
    # Writable data binds its symbol, not its contents, so only read-only merge units split into members. Only the
    # evidence crate's assembly holds member tables: a unit of another object, such as the libunwind that zig links
    # for POWER, is one allocation.
    owned = re.split(r"[.-]", Path(input_file).name)[0] == crate
    extents.append((start, start + size, None if element == MERGED and (section.writable or not owned) else element))

  slots: dict[int, Slot] = {}
  for header in headers:
    if header[1] != 4:
      continue
    _name, _kind, _flags, _address, offset, size, link, _info, _align, entsize = header
    dynamic = symbols(link) if link and headers[link][1] in {2, 11} else []
    for position in range(offset, offset + size, entsize or 24):
      r_offset, r_info, addend = struct.unpack_from(order + "QQq", data, position)
      kind, symbol_index = r_info & 0xFFFFFFFF, r_info >> 32
      if kind in ELF_RELATIVE.get(machine, set()):
        slots[r_offset] = Slot(address=addend)
      elif kind in ELF_SYMBOLIC.get(machine, set()) and symbol_index < len(dynamic):
        raw, _sym_kind, shndx, value, _size = dynamic[symbol_index]
        slots[r_offset] = Slot(address=value + addend) if shndx else Slot(symbol=raw.split("@")[0], addend=addend)
  image = LinkedImage(data, sections, little, starts=starts, extents=extents, slots=slots, toc_base=toc_base)
  if image.merged_unit_count():
    text = evidence_assembly(artifact_dir, binary).read_text(errors="replace")
    image.set_members(merge_unit_members(image, assembly_merge_units(text, little)))
  return image


def load_pe(data: bytes, artifact_dir: Path, binary: Path) -> LinkedImage:
  from symbolize_linked_binary import demangle_table

  pe = struct.unpack_from("<I", data, 0x3C)[0]
  number, = struct.unpack_from("<H", data, pe + 6)
  optional_size, = struct.unpack_from("<H", data, pe + 20)
  optional = pe + 24
  image_base, = struct.unpack_from("<Q", data, optional + 24)
  reloc_rva, reloc_size = struct.unpack_from("<II", data, optional + 112 + 5 * 8)
  sections = []
  for index in range(number):
    header = optional + optional_size + index * 40
    name = data[header : header + 8].rstrip(b"\0").decode(errors="replace")
    virtual_size, rva, raw_size, raw_offset = struct.unpack_from("<IIII", data, header + 8)
    characteristics, = struct.unpack_from("<I", data, header + 36)
    size = max(virtual_size, raw_size) if virtual_size == 0 else virtual_size
    sections.append(Section(name, image_base + rva, size, raw_offset if raw_size else None,
                            bool(characteristics & 0x20000000), bool(characteristics & 0x80000000)))
  image = LinkedImage(data, sections, True)
  starts = msvc_link_map_starts(artifact_dir / f"{binary.stem}.link-map.txt", demangle_table(artifact_dir))
  slots: dict[int, Slot] = {}
  for address, name in list(starts.items()):
    if name and name.startswith("__imp_"):
      slots[address] = Slot(symbol=name.removeprefix("__imp_"))
  position = reloc_rva
  while reloc_size and position < reloc_rva + reloc_size:
    block = image.read(image_base + position, 8)
    if block is None:
      break
    page, block_size = struct.unpack("<II", block)
    if block_size < 8:
      break
    entries = image.read(image_base + position + 8, block_size - 8) or b""
    for (entry,) in struct.iter_unpack("<H", entries):
      if entry >> 12 == 10:
        slot = image_base + page + (entry & 0xFFF)
        value = image.read(slot, 8)
        if value is not None and slot not in slots:
          slots[slot] = Slot(address=struct.unpack("<Q", value)[0])
    position += block_size
  return LinkedImage(data, sections, True, starts=starts, slots=slots, bounded_by_starts=True)


def uleb(data: bytes, position: int) -> tuple[int, int]:
  value = shift = 0
  while True:
    byte = data[position]
    position += 1
    value |= (byte & 0x7F) << shift
    shift += 7
    if byte < 0x80:
      return value, position


def sleb(data: bytes, position: int) -> tuple[int, int]:
  value = shift = 0
  while True:
    byte = data[position]
    position += 1
    value |= (byte & 0x7F) << shift
    shift += 7
    if byte < 0x80:
      return (value - (1 << shift) if byte & 0x40 else value), position


def macho_rebases(data: bytes, start: int, size: int, segments: list[int]) -> list[int]:
  addresses = []
  position, end = start, start + size
  segment = offset = 0
  while position < end:
    byte = data[position]
    position += 1
    opcode, immediate = byte & 0xF0, byte & 0x0F
    if opcode == 0x00:
      break
    if opcode == 0x20:
      segment = immediate
      offset, position = uleb(data, position)
    elif opcode == 0x30:
      value, position = uleb(data, position)
      offset += value
    elif opcode == 0x40:
      offset += immediate * POINTER_SIZE
    elif opcode in {0x50, 0x60}:
      count = immediate
      if opcode == 0x60:
        count, position = uleb(data, position)
      for _ in range(count):
        addresses.append(segments[segment] + offset)
        offset += POINTER_SIZE
    elif opcode == 0x70:
      addresses.append(segments[segment] + offset)
      value, position = uleb(data, position)
      offset += value + POINTER_SIZE
    elif opcode == 0x80:
      count, position = uleb(data, position)
      skip, position = uleb(data, position)
      for _ in range(count):
        addresses.append(segments[segment] + offset)
        offset += skip + POINTER_SIZE
    elif opcode != 0x10:
      raise ValueError(f"unsupported Mach-O rebase opcode 0x{opcode:x}")
  return addresses


def macho_binds(data: bytes, start: int, size: int, segments: list[int]) -> dict[int, Slot]:
  slots: dict[int, Slot] = {}
  position, end = start, start + size
  segment = offset = addend = 0
  symbol = ""
  while position < end:
    byte = data[position]
    position += 1
    opcode, immediate = byte & 0xF0, byte & 0x0F
    if opcode == 0x00:
      continue  # Lazy binding streams separate entries with DONE.
    if opcode in {0x10, 0x30, 0x50}:
      continue
    if opcode == 0x20:
      _, position = uleb(data, position)
    elif opcode == 0x40:
      terminator = data.index(b"\0", position)
      symbol = data[position:terminator].decode(errors="replace")
      position = terminator + 1
    elif opcode == 0x60:
      addend, position = sleb(data, position)
    elif opcode == 0x70:
      segment = immediate
      offset, position = uleb(data, position)
    elif opcode == 0x80:
      value, position = uleb(data, position)
      offset = (offset + value) & 0xFFFFFFFFFFFFFFFF
    elif opcode in {0x90, 0xA0, 0xB0, 0xC0}:
      count, skip = 1, 0
      if opcode == 0xC0:
        count, position = uleb(data, position)
        skip, position = uleb(data, position)
      for _ in range(count):
        slots[segments[segment] + offset] = Slot(symbol=symbol.removeprefix("_"), addend=addend)
        offset += skip + POINTER_SIZE
      if opcode == 0xA0:
        value, position = uleb(data, position)
        offset = (offset + value) & 0xFFFFFFFFFFFFFFFF
      elif opcode == 0xB0:
        offset += immediate * POINTER_SIZE
    else:
      raise ValueError(f"unsupported Mach-O bind opcode 0x{opcode:x}")
  return slots


def load_macho(data: bytes, artifact_dir: Path, binary: Path) -> LinkedImage:
  ncmds, = struct.unpack_from("<I", data, 16)
  position = 32
  sections: list[Section] = []
  segments: list[int] = []
  symtab = dyld = None
  for _ in range(ncmds):
    cmd, cmdsize = struct.unpack_from("<II", data, position)
    if cmd == 0x19:
      segname = data[position + 8 : position + 24].rstrip(b"\0").decode()
      vmaddr, _vmsize, _fileoff, _filesize, _maxprot, initprot, nsects, flags = struct.unpack_from(
        "<QQQQiiII", data, position + 24)
      segments.append(vmaddr)
      read_only_after_fixups = segname == "__DATA_CONST" or bool(flags & 0x10)
      for index in range(nsects):
        header = position + 72 + index * 80
        sectname = data[header : header + 16].rstrip(b"\0").decode()
        address, size, offset = struct.unpack_from("<QQI", data, header + 32)
        section_flags, = struct.unpack_from("<I", data, header + 64)
        section_type = section_flags & 0xFF
        sections.append(Section(
          name=f"{segname},{sectname}",
          address=address,
          size=size,
          offset=None if section_type in {0x1, 0xC, 0x12} else offset,
          executable=bool(section_flags & 0x80000400),
          writable=bool(initprot & 0x2) and not read_only_after_fixups,
          slot_table=section_type in {0x6, 0x7},
          element={0x2: 0, 0x3: 4, 0x4: 8, 0xE: 16}.get(section_type),
        ))
    elif cmd == 0x2:
      symtab = struct.unpack_from("<IIII", data, position + 8)
    elif cmd in {0x22, 0x80000022}:
      dyld = struct.unpack_from("<IIIIIIIIII", data, position + 8)
    position += cmdsize
  demangled = nm_names(artifact_dir / f"{binary.name}.binary.nm-symbols.txt")
  starts: dict[int, str | None] = {}
  if symtab is not None:
    symoff, nsyms, stroff, _strsize = symtab
    for index in range(nsyms):
      strx, n_type, n_sect, _desc, value = struct.unpack_from("<IBBHQ", data, symoff + index * 16)
      if n_type & 0xE0 or n_type & 0x0E != 0x0E or n_sect == 0 or n_sect > len(sections):
        continue
      if sections[n_sect - 1].executable:
        continue
      raw = data[stroff + strx : data.index(b"\0", stroff + strx)].decode(errors="replace")
      name = None if is_anonymous(raw.removeprefix("_")) else stable_name(demangled.get(value, []))
      starts[value] = starts.get(value) or name
  slots: dict[int, Slot] = {}
  image = LinkedImage(data, sections, True)
  if dyld is not None:
    rebase_off, rebase_size, bind_off, bind_size, weak_off, weak_size, lazy_off, lazy_size, *_ = dyld
    for address in macho_rebases(data, rebase_off, rebase_size, segments):
      value = image.read(address, 8)
      if value is not None:
        slots[address] = Slot(address=struct.unpack("<Q", value)[0])
    for offset, size in ((bind_off, bind_size), (weak_off, weak_size), (lazy_off, lazy_size)):
      if size:
        slots.update(macho_binds(data, offset, size, segments))
  extents = []
  for start, size in ld64_map_atoms(artifact_dir / f"{binary.name}.ld64-map.txt"):
    section = next((row for row in sections if row.address <= start < row.end), None)
    if section is not None and not section.executable and size > 0:
      starts.setdefault(start, None)
      extents.append((start, start + size, None))
  return LinkedImage(data, sections, True, starts=starts, extents=extents, slots=slots, bounded_by_starts=not extents,
                     padded_units=bool(extents))


def ld64_map_atoms(path: Path) -> list[tuple[int, int]]:
  """Return (address, size) for each atom in the `# Symbols:` table of an ld64 map."""
  if not path.is_file():
    return []
  atoms = []
  in_symbols = False
  row = re.compile(r"^0x([0-9a-fA-F]+)\s+0x([0-9a-fA-F]+)\s+\[\s*\d+\]\s")
  for line in path.read_text(errors="replace").splitlines():
    if line.startswith("# Symbols:"):
      in_symbols = True
    elif line.startswith("# Dead Stripped Symbols:"):
      in_symbols = False
    elif in_symbols and (match := row.match(line)):
      atoms.append((int(match.group(1), 16), int(match.group(2), 16)))
  return atoms


def load_linked_image(artifact_dir: Path) -> LinkedImage | None:
  binary = linked_binary_path(artifact_dir)
  if binary is None:
    return None
  data = binary.read_bytes()
  if data[:4] == b"\x7fELF" and data[4] == 2:
    image = load_elf(data, artifact_dir, binary)
  elif data[:2] == b"MZ":
    image = load_pe(data, artifact_dir, binary)
  elif data[:4] == b"\xcf\xfa\xed\xfe":
    image = load_macho(data, artifact_dir, binary)
  else:
    return None
  image.parameters = ir_parameters(artifact_dir)
  return image


# A function that a `.ll` file defines or declares: `define internal fastcc void @_RNv…5panic(`.
IR_FUNCTION = re.compile(r'(?:define|declare)\b[^@]*@("[^"]*"|[-\w.$]+)\(')


def ir_parameters(artifact_dir: Path) -> dict[str, frozenset[tuple[str, ...]]]:
  """The LLVM IR parameter lists, with their attributes, of each function in the compilation's `.ll` files.

  The key is the demangled name that the linked disassembly prints. Mach-O symbols add a leading underscore.
  A function that one module defines and another declares has one list for each form.
  """
  from symbolize_linked_binary import demangle_table

  names = demangle_table(artifact_dir)
  found: dict[str, set[tuple[str, ...]]] = {}
  for path in sorted(artifact_dir.glob("*.ll")):
    for line in path.read_text(errors="replace").splitlines():
      if not line.startswith(("define", "declare")) or (match := IR_FUNCTION.match(line)) is None:
        continue
      name = match.group(1).strip('"')
      demangled = names.get(name) or names.get(f"_{name}") or name
      found.setdefault(demangled, set()).add(split_ir_parameters(line, match.end()))
  return {name: frozenset(signatures) for name, signatures in found.items()}


def split_ir_parameters(line: str, start: int) -> tuple[str, ...]:
  """The parameters of the IR parameter list that starts at `line[start]`, split at commas outside brackets."""
  parameters, depth, piece = [], 0, start
  for position in range(start, len(line)):
    character = line[position]
    if character in "([{<":
      depth += 1
    elif character in ")]}>":
      if depth == 0:
        parameters.append(line[piece:position])
        break
      depth -= 1
    elif character == "," and depth == 0:
      parameters.append(line[piece:position])
      piece = position + 1
  return tuple(parameter.strip() for parameter in parameters if parameter.strip())
