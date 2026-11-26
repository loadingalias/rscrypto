#!/usr/bin/env -S uv run python
"""Test the RULE 1 check: test-gated production items fail, and test modules and files pass."""

import tempfile
from pathlib import Path

import rule1

FILES = {
  'src/lib.rs': '''
mod production;
#[cfg(test)]
mod test_file;
#[cfg(all(
  test,
  feature = "std"
))]
mod gated_file;
''',
  'src/production.rs': '''
#[cfg(test)]
fn hook() {}

#[cfg(all(
  feature = "std",
  test
))]
const MULTILINE: u8 = 0;

#[cfg_attr(test, derive(Debug))]
struct Probe;

#[cfg(any(test, feature = "diag"))]
fn widened() {}

#[cfg(not(test))]
fn release() {}

fn body() {
  #[cfg(test)]
  observe();
}

#[cfg(all(test, not(miri)))]
#[test]
fn test_function() {}

#[cfg(test)]
extern crate std;

#[cfg(test)]
mod tests {
  #[cfg(test)]
  fn helper() {}
  include!("production/included.rs");
}

include!("production/spliced.rs");
''',
  'src/production/included.rs': '#[cfg(test)]\nfn included_in_tests() {}\n',
  'src/production/spliced.rs': '#[cfg(test)]\nfn spliced() {}\n',
  'src/test_file.rs': 'mod nested;\n#[cfg(test)]\nfn in_test_file() {}\n',
  'src/test_file/nested.rs': '#[cfg(test)]\nfn in_nested_test_file() {}\n',
  'src/gated_file.rs': '#[cfg(test)]\nfn in_gated_file() {}\n',
}


def main():
  with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    for name, text in FILES.items():
      (root / name).parent.mkdir(parents=True, exist_ok=True)
      (root / name).write_text(text.lstrip('\n'))
    found = {(path, text) for path, _, text in rule1.violations(root)}
  assert found == {
    ('src/production.rs', 'fn hook() {}'),
    ('src/production.rs', 'const MULTILINE: u8 = 0;'),
    ('src/production.rs', 'struct Probe;'),
    ('src/production.rs', 'observe();'),
    ('src/production/spliced.rs', 'fn spliced() {}'),
  }, found


if __name__ == '__main__':
  main()
