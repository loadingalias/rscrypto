#!/usr/bin/env python3
"""Verify compile-only preparation and complete native measurement after transfer."""

import contextlib
import copy
import io
import json
import os
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

import runner
import settings
import transfer
import run_test


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.fixture = run_test.RunnerTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        (self.root / 'benches').unlink()
        shutil.copytree(run_test.ROOT / 'benches', self.root / 'benches')
        (self.root / '.gitignore').write_text('/target/\n/benchmark_results/\n/bin/\n/*.jsonl\n')
        self.archive = self.root / 'target/bench.tar.gz'
        self.archive.parent.mkdir()
        self.fixture.tool('s390x-linux-gnu-gcc', "print('gcc fixture')")
        self.target = 's390x-unknown-linux-gnu'
        self.selectors = ['sha256', 'sha512', 'crc32-ieee']
        self.previous = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, self.previous)

    def invoke(self, operation, *extra):
        args = runner.parse(['bench', *self.selectors, '--target', self.target,
                             '--' + operation + '-archive', str(self.archive), *extra])
        # These are host-executable external-tool fixtures, not IBM ISA evidence.
        # Real ELF identity/rejection is covered by test-transfer; CI must execute
        # the actual cross-built binaries on each native architecture.
        machine = 'x86_64' if operation == 'prepare' else 's390x'
        with patch.dict(os.environ, self.fixture.env, clear=True), patch.object(runner, 'ROOT', self.root), \
             patch.object(settings, 'CONFIG', self.root / '.config/criterion.json'), \
             patch('platform.system', return_value='Linux'), patch('platform.machine', return_value=machine), \
             patch.object(transfer, 'verify_elf'), contextlib.redirect_stdout(io.StringIO()):
            runner.bench(args, runner.load_catalog())

    def test_preparation_never_discovers_and_consumer_never_builds(self):
        with patch.object(runner, 'discover', side_effect=AssertionError('foreign discovery')):
            self.invoke('prepare')
        builds = (self.root / 'builds.jsonl').read_text().splitlines()
        self.assertEqual(len(builds), 2)  # sha256 and sha512 share the sha2 build.
        self.assertTrue(all('--target' in json.loads(row) for row in builds))
        self.assertFalse(self.fixture.calls(listing=True))
        self.assertFalse(self.fixture.calls())
        self.assertFalse(self.fixture.runs())
        # Remove the original executable tree and reject any attempted rebuild.
        for binary in (self.root / 'bin').glob('*--*'):
            binary.unlink()
        with patch.object(runner, 'build', side_effect=AssertionError('native compilation')):
            self.invoke('run')
        self.assertEqual(len(self.fixture.calls(listing=True)), 2)
        self.assertEqual(len(self.fixture.calls()), 2)
        plan = self.fixture.plan()
        self.assertEqual(set(plan[0]['cases']), {'sha256/rscrypto/64', 'sha256/other/64', 'sha512/rscrypto/64'})
        self.assertEqual(plan[1]['cases'], ['crc32/rscrypto/64'])
        self.assertTrue(all(entry['compatibility']['host']['machine'] == 's390x' for entry in plan))
        run = self.fixture.runs()[0]
        original = json.loads((run / 'input/bundle.json').read_text())
        self.assertEqual(original['metadata']['build']['host']['machine'], 'x86_64')
        self.assertEqual((run / 'status.txt').read_text(), 'state=complete\nexit_code=0\n')
        self.assertEqual(len(runner.verify(run)), 2)
        self.assertEqual((self.root / 'builds.jsonl').read_text().splitlines(), builds)

    def test_changed_request_sampling_source_or_binary_never_measures(self):
        self.invoke('prepare')
        for extra in (['--sample-size', '24'], ['--filter', '^sha256/rscrypto/64$']):
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, 'selection or sampling'):
                self.invoke('run', *extra)
        source = self.root / 'Cargo.toml'
        source.write_text(source.read_text() + '\n# changed source\n')
        with self.assertRaisesRegex(ValueError, 'source does not match'):
            self.invoke('run')
        source.write_text(source.read_text().removesuffix('\n# changed source\n'))
        incoming = self.root / 'target/edit'
        transfer.bundle.unpack(self.archive, incoming)
        binary = next((incoming / 'bin').rglob('sha2'))
        binary.write_bytes(b'changed binary')
        self.archive.unlink()
        transfer.bundle.pack(incoming, self.archive)
        with self.assertRaisesRegex(ValueError, 'files changed'):
            self.invoke('run')
        self.assertFalse(self.fixture.calls(listing=True))
        self.assertFalse(self.fixture.calls())

    def test_missing_configuration_is_rejected_even_in_a_sealed_bundle(self):
        self.invoke('prepare')
        incoming = self.root / 'target/edit'
        transfer.bundle.unpack(self.archive, incoming)
        manifest = json.loads((incoming / 'bundle.json').read_text())
        metadata = copy.deepcopy(manifest['metadata'])
        metadata['artifacts'].pop()
        transfer.bundle.seal(self.root, incoming, transfer.KIND, self.target, manifest['source'], metadata)
        self.archive.unlink()
        transfer.bundle.pack(incoming, self.archive)
        with self.assertRaisesRegex(ValueError, 'configurations are incomplete'):
            self.invoke('run')
        self.assertFalse(self.fixture.calls(listing=True))
        self.assertFalse(self.fixture.calls())


if __name__ == '__main__':
    unittest.main()
