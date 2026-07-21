#!/usr/bin/env python3
"""Reconstruct named final-binary disassembly from linker and object evidence."""

from __future__ import annotations

import argparse
import bisect
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Symbol:
  address: int
  size: int
  name: str


def parse_nm_line(line: str) -> tuple[int, str, str] | None:
  match = re.match(r"^([0-9a-fA-F]+)\s+(\S)\s+(.+)$", line.strip())
  if match is None:
    return None
  return int(match.group(1), 16), match.group(2), match.group(3)


def demangle_table(artifact_dir: Path) -> dict[str, str]:
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


def parse_link_map(path: Path, known_names: dict[str, str]) -> list[Symbol]:
  symbols: list[Symbol] = []
  table_row = re.compile(r"^\s*([0-9a-fA-F]+)\s+([0-9a-fA-F]+)\s+([0-9a-fA-F]+)\s+\d+\s+(.+)$")
  gnu_section = re.compile(r"^\s*(\.text(?:\.[^ ]+)?)\s+0x([0-9a-fA-F]+)\s+0x([0-9a-fA-F]+)(?:\s+.*)?$")
  gnu_wrapped_section = re.compile(r"^\s*(\.text\.[^ ]+)\s*$")
  gnu_wrapped_address = re.compile(r"^\s*0x([0-9a-fA-F]+)\s+0x([0-9a-fA-F]+)(?:\s+.*)?$")
  pending_gnu_section: str | None = None
  in_discarded_sections = False
  for line in path.read_text(errors="replace").splitlines():
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
      raw_name = section_symbol(section_match.group(1), known_names) if section_match else None
      if raw_name is not None and size > 0:
        symbols.append(Symbol(address, size, known_names.get(raw_name, raw_name)))
      continue
    match = gnu_section.match(line)
    if match is not None:
      pending_gnu_section = None
      raw_name = section_symbol(match.group(1), known_names)
      size = int(match.group(3), 16)
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
      if raw_name is not None and size > 0:
        symbols.append(Symbol(int(match.group(1), 16), size, known_names.get(raw_name, raw_name)))
    pending_gnu_section = None
  return symbols


def parse_final_nm(path: Path) -> list[Symbol]:
  rows = [row for line in path.read_text(errors="replace").splitlines() if (row := parse_nm_line(line)) is not None]
  text_rows = sorted((address, name) for address, kind, name in rows if kind in {"t", "T", "w", "W"})
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
      and not symbol.name.startswith((".", "Ltmp", "LBB"))
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


def symbolize(raw_disassembly: Path, out_path: Path, symbols: list[Symbol]) -> None:
  canonical = canonical_symbols(symbols)
  if not canonical:
    raise ValueError("no final linked text symbols found")
  by_address = {symbol.address: symbol for symbol in canonical}
  starts = [symbol.address for symbol in canonical]
  target_re = re.compile(r"0x([0-9a-fA-F]+)\s+<.*>")
  output: list[str] = []
  seen: set[int] = set()
  for line in raw_disassembly.read_text(errors="replace").splitlines():
    existing_label = re.match(r"^\s*([0-9a-fA-F]+) <.+>:$", line)
    if existing_label is not None:
      address = int(existing_label.group(1), 16)
      if address in by_address:
        output.append(f"{address:016x} <{by_address[address].name}>:")
        seen.add(address)
        continue
    instruction = re.match(r"^\s*([0-9a-fA-F]+):", line)
    if instruction is not None:
      address = int(instruction.group(1), 16)
      if address in by_address and address not in seen:
        output.append(f"{address:016x} <{by_address[address].name}>:")
        seen.add(address)

    def replace_target(match: re.Match[str]) -> str:
      address = int(match.group(1), 16)
      name = symbolized_target(address, starts, canonical)
      return match.group(0) if name is None else f"0x{address:x} <{name}>"

    output.append(target_re.sub(replace_target, line))
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
  parser.add_argument("--out-disassembly", required=True, type=Path)
  parser.add_argument("--out-symbols", required=True, type=Path)
  args = parser.parse_args()

  known_names = demangle_table(args.artifact_dir)
  nm_symbols = parse_final_nm(args.nm_symbols)
  map_symbols = parse_link_map(args.link_map, known_names) if args.link_map and args.link_map.is_file() else []
  indirect_symbols = (
    parse_indirect_symbols(args.indirect_symbols)
    if args.indirect_symbols and args.indirect_symbols.is_file()
    else []
  )
  symbols = merged_symbols(nm_symbols, [*map_symbols, *indirect_symbols])
  if not symbols:
    raise ValueError("neither the final binary nor the linker map contains text symbols")
  args.out_symbols.write_text(
    "".join(f"{symbol.address:016x} {symbol.size:016x} T {symbol.name}\n" for symbol in symbols)
  )
  symbolize(args.raw_disassembly, args.out_disassembly, symbols)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
