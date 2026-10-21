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

A read-only digest covers a whole allocation unit, because code can address
one object from the start of another object that the compiler merged with it.
ELF units are linker-map input sections; Mach-O units are ld64 map atoms.
Without a map, a PE or Mach-O unit runs from the nearest known object start to
the next one. x86-64 code generation does not merge globals; a Mach-O artifact
without an ld64 map relies on its symbol table alone.
Merged string and literal sections bind only the referenced literal. Any
address the binary cannot explain stays unresolved, and the caller keeps the
exact, placement-dependent instruction bytes.
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
  # constants, and None for one allocation.
  extents: list[tuple[int, int, int | None]] = field(default_factory=list)
  slots: dict[int, Slot] = field(default_factory=dict)
  # Function code ranges from the linked disassembly: (start, end, name).
  functions: list[tuple[int, int, str]] = field(default_factory=list)
  toc_base: int | None = None
  # Without map units, bound an object by the surrounding known starts.
  bounded_by_starts: bool = False

  def __post_init__(self) -> None:
    self.sections.sort(key=lambda section: section.address)
    self._section_starts = [section.address for section in self.sections]
    self.extents.sort(key=lambda extent: (extent[0], extent[1]))
    self._extent_starts = [start for start, _, _ in self.extents]
    self._object_starts = sorted(self.starts)
    self._slot_addresses = sorted(self.slots)
    self.set_functions(self.functions)
    self._digests: dict[tuple[int, int, int], str] = {}

  def set_functions(self, functions: list[tuple[int, int, str]]) -> None:
    self.functions = sorted(functions)
    self._function_starts = [start for start, _, _ in self.functions]

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

  def region(self, address: int) -> tuple[int, int, str | None] | None:
    """Return the allocation unit, or the merged literal, that contains `address`."""
    section = self.section_at(address)
    if section is None:
      return None
    unit: tuple[int, int, int | None] | None = None
    index = bisect.bisect_right(self._extent_starts, address) - 1
    if index >= 0:
      start, end, element = self.extents[index]
      if start <= address < end and section.address <= start and end <= section.end:
        unit = (start, end, element)
    if unit is None and self.bounded_by_starts:
      index = bisect.bisect_right(self._object_starts, address) - 1
      start = max(self._object_starts[index] if index >= 0 else section.address, section.address)
      following = self._object_starts[index + 1] if index + 1 < len(self._object_starts) else section.end
      unit = (start, min(following, section.end), None)
    if unit is None:
      return None
    start, end, element = unit
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

  def describe(self, address: int, depth: int = 0) -> str | None:
    """Return a placement-independent descriptor, or None when unexplained."""
    if function := self.function_at(address):
      return f"code:{function[2]}+0x{address - function[0]:x}"
    section = self.section_at(address)
    if section is None or section.executable:
      return None
    if section.slot_table:
      slot = self.slots.get(address)
      return f"slot:{self.describe_slot(slot, depth)}" if slot is not None else None
    region = self.region(address)
    if region is None:
      return None
    start, end, name = region
    offset = address - start
    if section.writable:
      return f"data:{name}+0x{offset:x}" if name else None
    digest = self.digest(start, end, depth)
    return f"const:{name or '-'}+0x{offset:x}/{end - start}:{digest}"

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
  """Return the literal size of an ELF merge section, 0 for strings."""
  if re.fullmatch(r"\.rodata\.str\d+\.\d+", input_section):
    return 0
  if match := re.fullmatch(r"\.rodata\.cst(\d+)", input_section):
    return int(match.group(1))
  return None


def link_map_extents(path: Path) -> list[tuple[int, int, str]]:
  """Return (start, size, input section) rows from a GNU ld or lld ELF map."""
  if not path.is_file():
    return []
  text = path.read_text(errors="replace")
  if "Publics by Value" in text:
    return []
  rows: list[tuple[int, int, str]] = []
  lld_row = re.compile(r"^\s*([0-9a-fA-F]+)\s+[0-9a-fA-F]+\s+([0-9a-fA-F]+)\s+\d+\s+\S.*:\((\S+)\)\s*$")
  gnu_row = re.compile(r"^ (\.\S+)\s+0x([0-9a-fA-F]+)\s+0x([0-9a-fA-F]+)\s+\S")
  gnu_name = re.compile(r"^ (\.\S+)\s*$")
  gnu_address = re.compile(r"^\s+0x([0-9a-fA-F]+)\s+0x([0-9a-fA-F]+)\s+\S")
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
      rows.append((int(match.group(1), 16), int(match.group(2), 16), match.group(3)))
      pending = None
    elif match := gnu_row.match(line):
      rows.append((int(match.group(2), 16), int(match.group(3), 16), match.group(1)))
      pending = None
    elif match := gnu_name.match(line):
      pending = match.group(1)
    elif (match := gnu_address.match(line)) and pending is not None:
      rows.append((int(match.group(1), 16), int(match.group(2), 16), pending))
      pending = None
    else:
      pending = None
  return [(start, size, name) for start, size, name in rows if start > 0 and size > 0]


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
  for start, size, input_section in link_map_extents(artifact_dir / f"{binary.name}.link-map.txt"):
    section = next((row for row in sections if row.address <= start < row.end), None)
    if section is None or section.executable:
      continue
    starts.setdefault(start, None)
    extents.append((start, start + size, merged_element(input_section)))

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
  return LinkedImage(data, sections, little, starts=starts, extents=extents, slots=slots, toc_base=toc_base)


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
  return LinkedImage(data, sections, True, starts=starts, extents=extents, slots=slots, bounded_by_starts=not extents)


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
    return load_elf(data, artifact_dir, binary)
  if data[:2] == b"MZ":
    return load_pe(data, artifact_dir, binary)
  if data[:4] == b"\xcf\xfa\xed\xfe":
    return load_macho(data, artifact_dir, binary)
  return None
