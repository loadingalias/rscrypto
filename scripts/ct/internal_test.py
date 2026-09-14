#!/usr/bin/env python3
"""Verify internal compiler opt-in and BINSEC flag provenance."""

import contextlib
import io
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import binsec
import internal

TARGET = 'x86_64-unknown-linux-gnu'


class InternalBuildTests(unittest.TestCase):
  def test_windows_shell_wrappers_run_through_bash(self):
    command = ['scripts/lib/python.sh', 'scripts/test/evidence_suite.py', '--case', 'argument with spaces']
    self.assertEqual(internal.host_command(command, 'nt'), ['bash', *command])

  def test_native_commands_and_posix_shell_wrappers_are_unchanged(self):
    native = ['cargo', 'test', '--locked']
    shell_wrapper = ['scripts/lib/toolchain.sh', '--exec', *native]
    self.assertIs(internal.host_command(native, 'nt'), native)
    self.assertIs(internal.host_command(shell_wrapper, 'posix'), shell_wrapper)

  def test_shell_export_preserves_encoded_arguments(self):
    original = '-C\x1flink-arg=path with spaces\x1f--cfg\x1fevidence="gcm"'
    output = io.StringIO()
    with patch.dict(os.environ, {'CARGO_ENCODED_RUSTFLAGS': original}, clear=True), \
         patch('sys.argv', ['internal.py', '--target', TARGET, '--print-encoded-rustflags']), \
         contextlib.redirect_stdout(output), patch.object(internal.subprocess, 'run') as execute:
      self.assertEqual(internal.main(), 0)
    self.assertEqual(output.getvalue(), original + '\x1f--cfg\x1frscrypto_internal')
    execute.assert_not_called()

  def test_flag_precedence_and_argument_boundaries(self):
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      (root / '.cargo').mkdir()
      (root / '.cargo/config.toml').write_text(
        '[target.x86_64-unknown-linux-gnu]\nrustflags = ["-C", "target-cpu=x86-64"]\n')
      cases = [
        ({}, ['-C', 'target-cpu=x86-64']),
        ({'CARGO_ENCODED_RUSTFLAGS': '', 'RUSTFLAGS': '-C target-cpu=native'}, []),
        ({'RUSTFLAGS': ''}, []),
        ({'CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS': '-C target-feature=+aes'},
         ['-C', 'target-cpu=x86-64', '-C', 'target-feature=+aes']),
        ({'CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS': '--cfg evidence="hmac"'},
         ['-C', 'target-cpu=x86-64', '--cfg', 'evidence="hmac"']),
        ({'RUSTFLAGS': '--cfg evidence="hmac"'}, ['--cfg', 'evidence="hmac"']),
        ({'CARGO_ENCODED_RUSTFLAGS': '-C\x1flink-arg=path with spaces'}, ['-C', 'link-arg=path with spaces']),
        ({'CARGO_ENCODED_RUSTFLAGS': '-C\x1f\x1fdebuginfo=1'}, ['-C', '', 'debuginfo=1']),
        ({'RUSTFLAGS': '-C target-cpu=native', 'CARGO_ENCODED_RUSTFLAGS': '-C\x1ftarget-cpu=generic'},
         ['-C', 'target-cpu=generic']),
      ]
      for environment, expected in cases:
        with self.subTest(environment=environment), patch.dict(os.environ, environment, clear=True), \
             patch.object(internal, 'ROOT', root):
          result, flags = internal.build_environment(TARGET, ['-C', 'relocation-model=static'])
          self.assertEqual(flags, [*expected, '-C', 'relocation-model=static', '--cfg', 'rscrypto_internal'])
          self.assertEqual(result['CARGO_ENCODED_RUSTFLAGS'].split('\x1f'), flags)
          self.assertEqual(dict(os.environ), environment)
      (root / '.cargo/config.toml').unlink()
      with patch.dict(os.environ, {'CARGO_BUILD_RUSTFLAGS': '--cfg evidence="hmac"'}, clear=True), \
           patch.object(internal, 'ROOT', root):
        _, flags = internal.build_environment(TARGET)
        self.assertEqual(flags, ['--cfg', 'evidence="hmac"', '--cfg', 'rscrypto_internal'])

  def test_binsec_build_records_the_flags_it_uses(self):
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      binary = root / 'target/ct-binsec-build' / TARGET / 'release' / binsec.HARNESS_BIN
      binary.parent.mkdir(parents=True)
      binary.touch()
      for status in (0, 7):
        with self.subTest(status=status), patch.dict(os.environ, {'CARGO_ENCODED_RUSTFLAGS': '-C\x1fdebuginfo=1'}, clear=True), \
             patch.object(binsec, 'ROOT', root), patch.object(internal, 'ROOT', root), \
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
                                    'relocation-model=static', '-C', 'link-arg=-no-pie',
                                    '--cfg', 'rscrypto_internal'])
            self.assertEqual(build.call_args.kwargs['env']['CARGO_ENCODED_RUSTFLAGS'].split('\x1f'), flags)


if __name__ == '__main__':
  unittest.main()
