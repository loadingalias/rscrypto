#!/usr/bin/env -S uv run python
"""Fail when production source holds an item that exists only for tests (AGENTS.md RULE 1).

A test file is a module that a `cfg` requiring `test` declares, or a descendant of one. Outside test
files and inline test modules, an item that a `cfg` requiring `test` gates is a violation, and so is
`cfg_attr(test, ...)`. A `#[test]` function and `extern crate` are allowed.
"""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
# RULE1-PURITY (docs/tasks/delivery.md) removes each entry. An entry that matches nothing fails.
ALLOWED = {
  ('src/traits/ct.rs', 'tests::DIT_ENTRIES.with(|entries| entries.set(entries.get().strict_add(1)));'),
}
MOD = re.compile(r'^\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+(\w+)\s*([;{])')
PATH = re.compile(r'#\[path\s*=\s*"([^"]+)"\]')
INCLUDE = re.compile(r'^\s*include!\("([^"]+\.rs)"\);')


def requires_test(expr):
  """Whether a `cfg` predicate holds only when `test` holds."""
  expr = expr.strip()
  if expr == 'test':
    return True
  match = re.fullmatch(r'(all|any|not)\((.*)\)', expr, re.S)
  if not match:
    return False
  parts, depth, current = [], 0, ''
  for char in match.group(2):
    depth += (char == '(') - (char == ')')
    if char == ',' and depth == 0:
      parts.append(current)
      current = ''
    else:
      current += char
  parts = [part for part in parts + [current] if part.strip()]
  if match.group(1) == 'all':
    return any(requires_test(part) for part in parts)
  if match.group(1) == 'any':
    return bool(parts) and all(requires_test(part) for part in parts)
  return False


def items(lines):
  """Yield (attributes, item line index) for each attributed item."""
  index = 0
  while index < len(lines):
    if not lines[index].lstrip().startswith('#['):
      index += 1
      continue
    attributes = []
    while index < len(lines) and re.match(r'\s*(#\[|///|//)', lines[index]):
      if lines[index].lstrip().startswith('#['):
        text, depth = '', 0
        while True:
          text += lines[index].strip()
          depth += lines[index].count('[') - lines[index].count(']')
          index += 1
          if depth <= 0:
            break
        attributes.append(text)
      else:
        index += 1
    yield attributes, index
    index += 1


def gated(attribute):
  match = re.fullmatch(r'#\[cfg\((.*)\)\]', attribute, re.S)
  return bool(match) and requires_test(match.group(1))


def child(path, name, attributes):
  explicit = next((m.group(1) for a in attributes if (m := PATH.search(a))), None)
  if explicit:
    return path.parent / explicit
  base = path.parent if path.name in ('lib.rs', 'mod.rs') else path.parent / path.stem
  return next((c for c in (base / f'{name}.rs', base / name / 'mod.rs') if c.exists()), None)


def violations(root):
  found, queue = [], [(root / 'src/lib.rs', False)]
  while queue:
    path, test_file = queue.pop()
    lines = path.read_text(encoding='utf-8').split('\n')
    attributed = dict((index, attributes) for attributes, index in items(lines))
    for index, line in enumerate(lines):
      module = MOD.match(line)
      if module and module.group(2) == ';':
        attributes = attributed.get(index, [])
        target = child(path, module.group(1), attributes)
        if target:
          queue.append((target, test_file or any(gated(a) for a in attributes)))
    ranges = []
    for index, attributes in attributed.items():
      if index >= len(lines) or (ranges and index <= ranges[-1][1]):
        continue
      line, test = lines[index], any(gated(a) for a in attributes)
      module = MOD.match(line)
      if module and module.group(2) == ';':
        continue
      if module and test:
        depth, end = 0, index
        while True:
          depth += lines[end].count('{') - lines[end].count('}')
          if depth == 0:
            break
          end += 1
        ranges.append((index, end))
        continue
      if test_file:
        continue
      if '#[test]' in attributes or line.lstrip().startswith('extern crate'):
        continue
      if test or any(re.match(r'#\[cfg_attr\(\s*test\s*,', a) for a in attributes):
        found.append((path.relative_to(root).as_posix(), index + 1, line.strip()))
    for index, line in enumerate(lines):
      if included := INCLUDE.match(line):
        test = test_file or any(start <= index <= end for start, end in ranges)
        queue.append((path.parent / included.group(1), test))
  return found


def main():
  found = violations(ROOT)
  new = [v for v in found if (v[0], v[2]) not in ALLOWED]
  stale = ALLOWED - {(v[0], v[2]) for v in found}
  for path, line, text in new:
    print(f'{path}:{line}: test-only item in production code: {text}', file=sys.stderr)
  for path, text in sorted(stale):
    print(f'{path}: stale RULE 1 allowance: {text}', file=sys.stderr)
  return 1 if new or stale else 0


if __name__ == '__main__':
  sys.exit(main())
