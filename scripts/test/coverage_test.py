#!/usr/bin/env python3
"""Regressions for execution attribution and failure-safe coverage publication."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('coverage', Path(__file__).with_name('test-coverage.py'))
coverage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coverage)


class Coverage(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='coverage test ')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.work = self.root / 'work'
        self.work.mkdir()
        self.output = self.root / 'output'
        self.output.mkdir()
        self.binary = self.root / 'binary'
        self.binary.write_bytes(b'original executable')
        self.objects = {str(self.binary): coverage.file_identity(self.binary)}
        self.listing = {'rust-suites': {'example::tests': {
            'status': 'listed', 'binary-path': str(self.binary), 'testcases': {
                'runs': {'ignored': False, 'filter-match': {'status': 'matches'}},
                'ignored': {'ignored': True, 'filter-match': {'status': 'mismatch', 'reason': 'ignored'}},
                'filtered': {'ignored': False, 'filter-match': {'status': 'mismatch', 'reason': 'expression'}},
            },
        }}}
        self.result = self.work / 'test-one'
        self.result.mkdir()
        self.record = {'binary_id': 'example::tests', 'test': 'runs',
                       'binary': str(self.binary.resolve())}
        self.write_record()
        self.profile = self.result / '1-2.profraw'
        self.profile.write_bytes(b'profile supplied by external LLVM runtime')
        self.llvm_calls = []

    def write_record(self):
        (self.result / 'execution.json').write_text(json.dumps(self.record))

    def test_discovery_and_skipped_tests_do_not_count_as_executions(self):
        discovery = self.work / 'discovery'
        discovery.mkdir()
        (discovery / '3-4.profraw').write_bytes(b'listing only')
        tests, profiles = coverage.execution_inventory(self.listing, self.work, self.root, {})
        self.assertEqual([(test['test'], test['executed']) for test in tests],
                         [('runs', True), ('ignored', False), ('filtered', False)])
        self.assertEqual(profiles, [self.profile])
        self.profile.unlink()
        with self.assertRaisesRegex(RuntimeError, 'no usable coverage profiles'):
            coverage.execution_inventory(self.listing, self.work, self.root, {})

    def test_missing_duplicate_and_wrong_binary_executions_fail(self):
        for field, value in [('test', 'unknown'), ('binary', 'different')]:
            with self.subTest(field=field):
                original = self.record[field]
                self.record[field] = value
                self.write_record()
                with self.assertRaises(RuntimeError):
                    coverage.execution_inventory(self.listing, self.work, self.root, {})
                self.record[field] = original
        self.write_record()
        duplicate = self.work / 'test-two'
        duplicate.mkdir()
        (duplicate / 'execution.json').write_text(json.dumps(self.record))
        with self.assertRaisesRegex(RuntimeError, 'duplicate'):
            coverage.execution_inventory(self.listing, self.work, self.root, {})
        (duplicate / 'execution.json').unlink()
        (self.result / 'execution.json').unlink()
        with self.assertRaisesRegex(RuntimeError, 'missing test executions'):
            coverage.execution_inventory(self.listing, self.work, self.root, {})
        listing = copy.deepcopy(self.listing)
        listing['rust-suites']['example::tests']['testcases'].pop('runs')
        with self.assertRaisesRegex(RuntimeError, 'no selected tests'):
            coverage.execution_inventory(listing, self.work, self.root, {})

    def test_executable_replacement_is_rejected(self):
        coverage.verify_objects(self.objects)
        self.binary.write_bytes(b'replacement executable')
        with self.assertRaisesRegex(RuntimeError, 'executable changed'):
            coverage.verify_objects(self.objects)

    def corpus(self):
        directory = self.root / 'fuzz/corpus/example'
        directory.mkdir(parents=True)
        (directory / 'seed').write_bytes(b'abc')
        (self.root / 'fuzz/committed-seeds.txt').write_text('fuzz/corpus/example/seed\n')
        return self.root / 'fuzz/Cargo.toml', directory

    def test_corpus_modes_and_receipts_bind_actual_inputs(self):
        manifest, directory = self.corpus()
        seed = directory / 'seed'
        discovery = directory / 'discovery'
        discovery.write_bytes(b'')
        committed = coverage.corpus_inventory(self.root, manifest, 'committed')
        self.assertEqual(committed, {str(seed): {
            'bytes': 3, 'sha256': 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'}})
        local = coverage.corpus_inventory(self.root, manifest, 'local')
        self.assertEqual(set(local), {str(seed), str(discovery)})
        self.assertEqual(local[str(discovery)]['sha256'],
                         'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855')
        receipt = self.result / 'corpus-inputs'
        with self.assertRaisesRegex(RuntimeError, 'differ from the selected inventory'):
            coverage.execution_inventory(self.listing, self.work, self.root, committed)
        receipt.write_bytes(str(discovery).encode() + b'\0')
        with self.assertRaisesRegex(RuntimeError, 'not present before execution'):
            coverage.execution_inventory(self.listing, self.work, self.root, committed)
        receipt.write_bytes(str(seed).encode() + b'\0')
        tests, _ = coverage.execution_inventory(self.listing, self.work, self.root, committed)
        self.assertEqual(tests[0]['corpus_inputs'], ['fuzz/corpus/example/seed'])

    def test_corpus_changed_during_execution_fails(self):
        manifest, directory = self.corpus()
        seed = directory / 'seed'
        binaries = {'rust-binaries': {'example::tests': {'binary-path': str(self.binary)}}}
        suite = self.work / 'suite'

        def execute(*args, **kwargs):
            result = suite / 'test-one'
            result.mkdir()
            (result / 'execution.json').write_text(json.dumps(self.record))
            (result / '1.profraw').write_bytes(b'profile')
            (result / 'corpus-inputs').write_bytes(str(seed).encode() + b'\0')
            seed.write_bytes(b'changed')

        with patch.object(coverage, 'instrument', return_value={}), \
             patch.object(coverage, 'capture', side_effect=[json.dumps(binaries), json.dumps(self.listing)]), \
             patch.object(coverage, 'run', side_effect=execute), \
             self.assertRaisesRegex(RuntimeError, 'corpus changed'):
            coverage.collect_suite(self.root, suite, {'CARGO_BUILD_TARGET': 'fixture'}, manifest, [], [])

    def test_failed_nextest_never_reconciles_or_exports_partial_execution(self):
        binaries = {'rust-binaries': {'example::tests': {'binary-path': str(self.binary)}}}
        directory = self.work / 'suite'
        with patch.object(coverage, 'instrument', return_value={}), \
             patch.object(coverage, 'capture', side_effect=[json.dumps(binaries), json.dumps(self.listing)]), \
             patch.object(coverage, 'run', side_effect=subprocess.CalledProcessError(100, 'nextest')), \
             patch.object(coverage, 'execution_inventory') as inventory, \
             self.assertRaises(subprocess.CalledProcessError):
            coverage.collect_suite(self.root, directory, {'CARGO_BUILD_TARGET': 'fixture'},
                                   self.root / 'Cargo.toml', [], [])
        inventory.assert_not_called()

    def llvm(self, args, root, env, *, output=None, capture_errors=False):
        # Stub only the external LLVM processes, leaving report validation and publication real.
        self.llvm_calls.append(args)
        if args[1] == 'merge':
            Path(args[args.index('-o') + 1]).write_bytes(b'merged')
        elif args[1] == 'export':
            output.write('SF:source.rs\nDA:1,1\nend_of_record\n')
        elif args[1] == 'report':
            output.write(self.summary)
        elif args[1] == 'show':
            html = Path(next(arg.removeprefix('-output-dir=') for arg in args if arg.startswith('-output-dir=')))
            html.mkdir()
            (html / 'index.html').write_text('coverage')
        return subprocess.CompletedProcess(args, 0, stderr='')

    def report(self, suites=None):
        with patch.object(coverage, 'run', side_effect=self.llvm), \
             patch.object(coverage, 'verify_mappings', return_value=''), \
             patch.object(coverage.bundle, 'source_identity', return_value={'revision': 'same'}):
            coverage.report(self.root, self.work, self.output, self.objects, [self.profile],
                            self.root, {}, {'source': {'revision': 'same'}, 'suites': suites or []})

    def test_invalid_summary_never_publishes(self):
        self.summary = 'missing total\n'
        with self.assertRaisesRegex(RuntimeError, 'TOTAL'):
            self.report()
        self.assertEqual(list(self.output.iterdir()), [])

    def test_each_llvm_failure_leaves_no_published_report(self):
        self.summary = 'TOTAL 1 0 100%\n'
        llvm = self.llvm
        for operation in ('merge', 'export', 'report', 'show'):
            def fail(args, *positional, **keywords):
                if args[1] == operation:
                    raise subprocess.CalledProcessError(1, args)
                return llvm(args, *positional, **keywords)

            with self.subTest(operation=operation), patch.object(self, 'llvm', side_effect=fail), \
                 self.assertRaises(subprocess.CalledProcessError):
                self.report()
            self.assertEqual(list(self.output.iterdir()), [])
            shutil.rmtree(self.work / 'suites')

    def test_publication_failure_removes_partial_artifacts(self):
        self.summary = 'TOTAL 1 0 100%\n'
        rename = Path.rename

        def fail(path, target):
            if path.name == 'provenance.json':
                raise OSError('injected publication failure')
            return rename(path, target)

        with patch.object(Path, 'rename', fail), self.assertRaisesRegex(OSError, 'publication failure'):
            self.report()
        self.assertEqual(list(self.output.iterdir()), [])

    def test_success_binds_published_artifacts(self):
        self.summary = 'TOTAL 1 0 100%\n'
        self.report()
        evidence = json.loads((self.output / 'provenance.json').read_text())
        self.assertEqual(evidence['summary_total'], self.summary.strip())
        self.assertEqual(evidence['object_count'], 1)
        for name, identity in evidence['artifacts'].items():
            self.assertEqual(identity, coverage.file_identity(self.output / name))

    def test_suite_contributions_are_exported_separately_and_bound(self):
        self.summary = 'TOTAL 1 0 100%\n'
        suite = {'name': 'first', 'binaries': self.objects,
                 'profile_paths': [str(self.profile.relative_to(self.work))]}
        self.report([suite])
        evidence = json.loads((self.output / 'provenance.json').read_text())
        self.assertEqual(evidence['suites'][0]['coverage'], 'suites/0.lcov')
        self.assertNotIn('profile_paths', evidence['suites'][0])
        self.assertEqual((self.output / 'suites/0.lcov').read_text(), 'SF:source.rs\nDA:1,1\nend_of_record\n')
        self.assertEqual(evidence['artifacts']['suites/0.lcov'],
                         coverage.file_identity(self.output / 'suites/0.lcov'))
        profiles = [next(arg for arg in args if arg.startswith('-instr-profile='))
                    for args in self.llvm_calls if args[1] == 'export']
        self.assertEqual(profiles, [f'-instr-profile={self.work / "suite-report-0/merged.profdata"}',
                                    f'-instr-profile={self.work / "merged.profdata"}'])

    def test_preflight_failure_invalidates_previous_report(self):
        output = self.root / 'coverage'
        output.mkdir()
        (output / 'provenance.json').write_text('old success')
        (output / 'total.lcov').write_text('old coverage')
        (self.root / 'rust-toolchain.toml').write_text('[toolchain]\nchannel="fixture"\n')
        with patch.object(coverage, 'ROOT', self.root), patch.object(sys, 'argv', ['test-coverage.py']), \
             patch.object(coverage, 'capture', side_effect=RuntimeError('preflight failed')), \
             self.assertRaisesRegex(RuntimeError, 'preflight failed'):
            coverage.main()
        self.assertEqual([path.name for path in output.iterdir()], ['.lock'])

    def test_launcher_separates_listing_and_propagates_failure(self):
        child = self.root / 'child.py'
        child.write_text('import os, pathlib, sys\n'
                         'pathlib.Path(os.environ["LLVM_PROFILE_FILE"].replace("%p", "1").replace("%m", "2"))'
                         '.write_bytes(b"profile")\nsys.exit(int(sys.argv[1]))\n')
        env = {key: value for key, value in os.environ.items() if not key.startswith('NEXTEST_')}
        command = [sys.executable, str(Path(__file__).with_name('coverage_run.py')),
                   str(self.work), sys.executable, str(child)]
        subprocess.run([*command, '0'], env=env, check=True)
        self.assertTrue((self.work / 'discovery/1-2.profraw').is_file())
        env.update(NEXTEST_TEST_NAME='child', NEXTEST_BINARY_ID='example')
        result = subprocess.run([*command, '7'], env=env)
        self.assertEqual(result.returncode, 7)
        records = [json.loads(path.read_text()) for path in self.work.glob('test-*/execution.json')]
        self.assertIn({'binary_id': 'example', 'test': 'child', 'binary': str(Path(sys.executable).resolve())}, records)


if __name__ == '__main__':
    unittest.main()
