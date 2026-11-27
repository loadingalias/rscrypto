#!/usr/bin/env -S uv run python
"""Verify BINSEC flag provenance and the startup model."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import binsec

TARGET = 'x86_64-unknown-linux-gnu'


class BinsecBuildTests(unittest.TestCase):
  def test_binsec_build_records_the_flags_it_uses(self):
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      binary = root / 'target/ct-binsec-build' / TARGET / 'binsec' / binsec.HARNESS_BIN
      binary.parent.mkdir(parents=True)
      binary.touch()
      for status in (0, 7):
        with self.subTest(status=status), patch.dict(os.environ, {'CARGO_ENCODED_RUSTFLAGS': '-C\x1fdebuginfo=1'}, clear=True), \
             patch.object(binsec, 'ROOT', root), \
             patch.object(binsec, 'configure_cross_linker'), \
             patch.object(binsec.subprocess, 'run', return_value=subprocess.CompletedProcess([], status)) as build:
          if status:
            with self.assertRaises(SystemExit) as failure:
              binsec.build_harness(TARGET, 'release', [])
            self.assertEqual(failure.exception.code, status)
          else:
            actual, flags = binsec.build_harness(TARGET, 'release', [])
            self.assertEqual(actual, binary)
            self.assertEqual(flags, ['-C', 'debuginfo=1', '-C', 'target-cpu=x86-64', '-C',
                                    'relocation-model=static', '-C', 'link-arg=-no-pie'])
            self.assertEqual(build.call_args.kwargs['env']['CARGO_ENCODED_RUSTFLAGS'].split('\x1f'), flags)

  def test_binsec_script_sets_the_lse_flag_after_the_section_load_only_if_the_elf_defines_it(self):
    # Before the load, the load writes the file's 0 over the flag. Without the symbol, BINSEC
    # cannot resolve the name and stops.
    load = 'load sections .text, .bss from file'
    flag = '@[<__aarch64_have_lse_atomics>, 1] := 1'
    for symbols, present in (({'leaf', 'done', '__aarch64_have_lse_atomics'}, True), ({'leaf', 'done'}, False)):
      with self.subTest(symbols=symbols):
        lines = binsec.binsec_script(
          {}, start_symbol='leaf', done_symbol='done', load_sections=['.text', '.bss'], symbols=symbols
        ).splitlines()
        self.assertEqual(flag in lines, present)
        if present:
          self.assertGreater(lines.index(flag), lines.index(load))


if __name__ == '__main__':
  unittest.main()
